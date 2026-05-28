# Phase 1: Digital Twin Inner Loop — Data Generation & System Identification

## Goal

Build a pipeline that: (1) simulates an E/I neural population in Cleo with optogenetic actuation, (2) generates training data by driving structured noise, (3) fits two forward dynamics models (linear N4SID and control-affine Neural ODE).

## Environment Setup

> [!IMPORTANT]
> The `cleo` conda env has no PyTorch. We'll create `dtmodeling` cloned from `lfads-torch-cuda12` (which already has PyTorch + CUDA) and add Cleo + torchdiffeq.

```bash
conda create -n dtmodeling --clone lfads-torch-cuda12
conda activate dtmodeling
pip install -e ~/cleo  # install cleo in editable mode
pip install torchdiffeq h5py
```

---

## File Structure

All code lives in `~/cleo/modeling/` (not part of the published `cleo` package):

```
modeling/
├── __init__.py          (exists)
├── data.py              (exists — Neo → arrays/H5 conversion)
├── plant.py             [NEW] Cleo E/I network + devices setup
├── datagen.py           [NEW] OU noise driving + data collection
├── models/
│   ├── __init__.py      (exists)
│   ├── n4sid.py         [NEW] Subspace state-space identification
│   └── canode.py        [NEW] Control-affine Neural ODE
└── scripts/
    ├── generate_data.py [NEW] Run Cleo sim → save u(t), x(t)
    ├── fit_n4sid.py     [NEW] Fit linear model
    └── fit_canode.py    [NEW] Train Neural ODE
```

---

## Proposed Changes

### Environment

#### [NEW] conda env `dtmodeling`
Clone from `lfads-torch-cuda12`, add `cleo` (editable), `torchdiffeq`, `h5py`.

---

### Plant Setup

#### [NEW] modeling/plant.py

Creates the biological simulation:

**Network (1,000 LIF neurons, 800E/200I):**
```python
ng = b2.NeuronGroup(1000, """
    dv/dt = (-(v - E_L) + Rm * (I_syn + I_opto + I_bg)) / tau_m : volt
    I_syn : amp
    I_opto : amp
""", threshold="v > -50*mV", reset="v = E_L",
   namespace={"tau_m": 20*ms, "Rm": 500*Mohm, "E_L": -70*mV, "I_bg": 60*pA})
```

- First 800 neurons = excitatory, last 200 = inhibitory
- Recurrent synapses: E→all (w=0.1mV), I→all (w=-0.4mV), p_connect=0.1
- 3D coordinates via `assign_coords_rand_rect_prism` in a 500×500×500 μm volume
- Density ~8,000/mm³ (realistic for organoid/culture)

**Actuation (2 OpticFibers in different quadrants):**
```python
light = cleo.light.Light(
    light_model=cleo.light.fiber473nm(),
    coords=[[-125, -125, 0], [125, 125, 0]] * um,  # opposing quadrants of 500μm volume
    name="fibers",
)
```
- 2 fibers at opposing corners → different spatial influence patterns
- Each fiber independently controllable (update with 2-element array)

**Opsin (ProportionalCurrentOpsin):**
```python
opsin = cleo.opto.ProportionalCurrentOpsin(
    I_per_Irr=...,  # current/irradiance ratio, calibrated to ~100-500 pA at 1-10 mW/mm²
    name="opto",
)
sim.inject(opsin, ng, Iopto_var_name="I_opto")
```
- Linear: I = I_per_Irr × Irr × rho_rel
- Control-affine by construction

**Measurement (50-channel MultiUnitActivity probe):**
```python
mua = cleo.ephys.MultiUnitActivity(name="mua")
probe = cleo.ephys.Probe(
    coords=grid_coords_50ch * um,  # 50 channels spanning 500μm volume
    signals=[mua],
    save_history=True,
)
```
- 50 channels, each detecting ~20-40 neurons (detection radius ~50-100 μm)
- `get_state()` returns spike counts per channel per sample period
- Smoothing to continuous rates happens in the IOProcessor via `exp_firing_rate_estimate()`

> [!NOTE]
> Cleo doesn't output continuous rates directly. We use `MultiUnitActivity` (spike counts) + exponential smoothing in the IOProcessor to produce the continuous state vector x(t). This is realistic — it mirrors what real-time systems do.

**Exposed function:**
```python
def build_plant(n_exc=800, n_inh=200, n_channels=50, seed=42) -> tuple[cleo.CLSimulator, dict]:
    """Returns (simulator, device_refs_dict)"""
```

---

### Data Generation

#### [NEW] modeling/datagen.py

**Ornstein-Uhlenbeck process:**
```python
def ou_process(n_steps, n_channels, dt, tau=50e-3, sigma=5.0, mu=10.0):
    """Generate OU noise: dx = (mu - x)/tau * dt + sigma * sqrt(2*dt/tau) * dW"""
```
- `tau` = 50 ms (correlation time — slow enough to excite dynamics)
- `sigma` controls amplitude
- `mu` = baseline drive level
- Generates `(n_steps, n_channels)` array of input values

**IOProcessor for data collection:**
```python
class DataCollectionIOP(cleo.LatencyIOProcessor):
    """Drives fibers with pre-computed OU inputs, records smoothed rates."""
    
    def __init__(self, u_trajectory, sample_period, tau_smooth=20*ms):
        super().__init__(sample_period=sample_period)
        self.u = u_trajectory       # (n_steps, n_inputs)
        self.x_history = []         # recorded rates
        self.u_history = []         # applied inputs
        self.rates = None           # running rate estimate
        self.tau_smooth = tau_smooth
    
    def process(self, state_dict, t_samp):
        # 1. Get spike counts from MUA
        counts = state_dict["probe"]["mua"]
        
        # 2. Exponential smoothing → continuous rates
        self.rates = exp_firing_rate_estimate(
            counts, self.sample_period, self.rates, self.tau_smooth
        )
        self.x_history.append(self.rates.copy())
        
        # 3. Look up pre-computed OU input
        u_now = self.u[self.i_step]
        self.u_history.append(u_now.copy())
        
        # 4. Drive fibers
        return {"fibers": u_now * mwatt/mm**2}, t_samp
```

**Output format** (saved to HDF5):
```
data.h5
├── x          (n_channels, n_steps)  — smoothed firing rates
├── u          (n_inputs, n_steps)    — optical input trajectory
├── t          (n_steps,)             — time points in seconds
└── attrs: {dt, tau_smooth, n_inputs, n_channels, ...}
```

---

### N4SID Baseline Model

#### [NEW] modeling/models/n4sid.py

Fits ẋ = Ax + Bu using Subspace State-Space Identification.

```python
class N4SIDModel:
    """Linear state-space model via N4SID algorithm."""
    
    def __init__(self, n_states: int):
        self.n_states = n_states
        self.A = None  # state transition
        self.B = None  # input matrix
        self.C = None  # observation matrix
        self.D = None  # feedthrough
    
    def fit(self, u, x, dt):
        """Fit via N4SID: construct Hankel matrices, SVD, extract system."""
    
    def predict(self, x0, u, dt):
        """Forward simulate: x_{k+1} = A x_k + B u_k"""
```

Implementation: Hankel matrices → oblique projection → SVD → extract A,B,C,D. Pure numpy.

---

### Control-Affine Neural ODE

#### [NEW] modeling/models/canode.py

Models ẋ = f_θ(x) + g_φ(x)·u(t) as a Neural ODE.

```python
class DriftNet(nn.Module):       # f_θ(x): autonomous dynamics MLP
class ControlNet(nn.Module):     # g_φ(x): control susceptibility, outputs (n_x, n_u) matrix
class ControlAffineODE(nn.Module):
    # ẋ = f(x) + g(x) · u(t)
    # Integrated with torchdiffeq.odeint
```

Training: chop into 200ms windows, integrate ODE from x(t₀), MSE loss vs recorded x(t).

---

## Resolved Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Neurons | 1,000 (800E/200I) | ~8k/mm³ density, tractable for Brian2 |
| Volume | 500×500×500 μm | Fits 50 MUA channels at ~25μm pitch |
| MUA channels | 50 | Each sees ~20-40 neurons, good state coverage |
| Sim duration | 5 minutes | 300k timesteps, ~10-20 min wall time |
| I_per_Irr | Calibrate to ~100-500 pA at 1-10 mW/mm² | Matches 4-state opsin current range |
| Data format | Contiguous 3D arrays (n_trials, n_channels, trial_bins) | Fast GPU batching |

---

## Verification Plan

### Automated Tests
1. **Plant builds and runs**: `python -c "from modeling.plant import build_plant; sim, _ = build_plant(); sim.run(100*ms)"`
2. **Data generation completes**: `python modeling/scripts/generate_data.py` → produces valid HDF5
3. **N4SID fits and predicts**: prediction MSE < spontaneous variance
4. **NODE trains**: loss decreases over first 100 epochs
5. **NODE beats N4SID**: lower prediction MSE on held-out test window

### Manual Verification
- Plot x(t) predictions vs ground truth for both models
- Visualize learned g(x) to confirm spatial selectivity of the two fibers
- Compare N4SID eigenvalues with NODE linearization around fixed point
