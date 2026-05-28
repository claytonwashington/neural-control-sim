# Phase 1: Digital Twin — Inner Loop

Parent: [task.md](file:///home/cbwash2/cleo/task.md)
Branch: `feature/digital-twin-phase1`
Agent: Antigravity

## Parameters

| Param | Value |
|-------|-------|
| Neurons | 1,000 (800E / 200I) |
| Volume | 500×500×500 μm |
| MUA channels | 50 |
| Fibers | 2 (opposing quadrants) |
| Opsin | ProportionalCurrentOpsin |
| Sim duration | 10 trials x 30s (multi-trial mode) |
| Sample period | 1 ms |
| Conda env | `dtmodeling` (Python 3.11, PyTorch 2.5.1+cu121, torchdiffeq 0.2.5) |

## Checklist

### Environment
- `[x]` Create `dtmodeling` conda env (Python 3.11 + pip torch cu121 + torchdiffeq + cleo)
- `[x]` Verify: `python -c 'import cleo, torch, torchdiffeq'` — all pass

### Plant (modeling/plant.py)
- `[x]` E/I LIF network (800E/200I) with recurrent synapses
- `[x]` 3D spatial layout (500×500×500 μm via `assign_coords_rand_rect_prism`)
- `[x]` 2 OpticFiber light sources in opposing quadrants
- `[x]` ProportionalCurrentOpsin (I_per_Irr calibrated)
- `[x]` 50-channel MultiUnitActivity probe spanning volume
- `[x]` `build_plant()` function returns (sim, device_refs)
- `[x]` Smoke test: sim.run(100*ms) completes

### Data Generation (modeling/datagen.py)
- `[x]` OU process generator (configurable tau, sigma, mu)
- `[x]` DataCollectionIOP with exp_firing_rate_estimate smoothing
- `[x]` Fixed: MUA get_state() returns (i, t, y) tuple — unpack y for counts
- `[x]` Fixed: Initialize rates to zeros * Hz on first call
- `[x]` `generate_dataset()` single-trial function: verified x=(50,2000), u=(2,2000)
- `[x]` `generate_multi_trial()` multi-trial function (independent seeds + initial conditions)
- `[ ]` Full multi-trial run completes on gpu2

### Data Generation Script (modeling/scripts/generate_data.py)
- `[x]` CLI script with --mode single/multi
- `[x]` Default: 10 trials x 30s
- `[ ]` Full run completes on gpu2 and produces valid HDF5

### N4SID Model (modeling/models/n4sid.py)
- `[x]` N4SID class: fit(u, x, dt) and predict(x0, u, dt)
- `[x]` Hankel matrix construction + SVD
- `[ ]` Test: fits synthetic linear system correctly
- `[ ]` Test: fits Cleo data, prediction MSE < spontaneous variance

### Control-Affine NODE (modeling/models/canode.py)
- `[x]` DriftNet MLP: f_θ(x)
- `[x]` ControlNet MLP: g_φ(x) → (n_x, n_u) matrix
- `[x]` ControlAffineODE: ẋ = f(x) + g(x)·u with torchdiffeq
- `[x]` Training loop with windowed MSE loss
- `[x]` Input interpolation for ODE solver
- `[ ]` Test: loss decreases over 100 epochs
- `[ ]` Test: prediction MSE < N4SID on held-out data

### Training Scripts
- `[x]` modeling/scripts/fit_n4sid.py
- `[x]` modeling/scripts/fit_canode.py
- `[ ]` Both save trained models + prediction plots

### Validation
- `[ ]` Side-by-side plot: N4SID vs NODE vs ground truth x(t)
- `[ ]` Compare MSE on held-out test window
