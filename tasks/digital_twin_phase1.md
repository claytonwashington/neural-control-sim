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
- `[x]` Full multi-trial run completes on gpu2

### Data Generation Script (modeling/scripts/generate_data.py)
- `[x]` CLI script with --mode single/multi
- `[x]` Default: 10 trials x 30s
- `[x]` Full run completes on gpu2 and produces valid HDF5

### N4SID Model (modeling/models/n4sid.py)
- `[x]` N4SID class: fit(u, x, dt) and predict(x0, u, dt)
- `[x]` Hankel matrix construction + SVD
- `[x]` Test: fits synthetic linear system correctly
- `[x]` Test: fits Cleo data, prediction MSE < spontaneous variance

### Control-Affine NODE (modeling/models/canode.py)
- `[x]` DriftNet MLP: f_θ(x)
- `[x]` ControlNet MLP: g_φ(x) → (n_x, n_u) matrix
- `[x]` ControlAffineODE: ẋ = f(x) + g(x)·u with torchdiffeq
- `[x]` Training loop with windowed MSE loss
- `[x]` Input interpolation for ODE solver
- `[x]` Test: loss decreases over 100 epochs
- `[x]` Test: prediction MSE < N4SID on held-out data

### Training Scripts
- `[x]` modeling/scripts/fit_n4sid.py
- `[x]` modeling/scripts/fit_canode.py
- `[x]` Both save trained models + prediction plots

### Validation
- [x] Side-by-side plot: N4SID vs NODE vs ground truth x(t)
- [x] Compare MSE on held-out test window

### Parallel Hyperparameter Sweep
- [x] modeling/scripts/sweep_canode.py (optimized to use all 8 GPUs in parallel)
- [x] Run parallel sweep for different hidden sizes, layers, learning rates, and solvers
- [x] Select the best model configuration based on evaluation metrics

## Summary of Results

### 1. Baseline Model Comparison (200ms windowed R²)
* **N4SID Subspace Model:** Avg $R^2 = 0.2059$ (MSE = `23600.07` raw scale)
* **CA-NODE Baseline:** Avg $R^2 = 0.8401$ (MSE = `0.2238` normalized scale)

### 2. Hyperparameter Sweep Results (200ms windowed R²)
All 8 configurations were run in parallel on `gpu1` using separate GPU cards.

| Run ID | Method | Hidden | Layers | LR | Compile | Training Time | Avg 200ms $R^2$ | Avg 200ms MSE | Open-Loop $R^2$ |
|---|---|---|---|---|---|---|---|---|---|
| **run_04** | **dopri5** | **128** | **2** | **5e-4** | **True** | **5.1 mins** | **0.8988** | **0.1417** | **0.2257** |
| run_07 | rk4 | 128 | 2 | 1e-3 | True | 26.8 mins | 0.8769 | 0.1723 | 0.2394 |
| run_05 | dopri5 | 256 | 2 | 5e-4 | True | 6.5 mins | 0.8714 | 0.1796 | 0.2306 |
| run_01 | dopri5 | 128 | 2 | 1e-3 | True | 6.4 mins | 0.8701 | 0.1817 | 0.2302 |
| run_06 | dopri5 | 256 | 3 | 5e-4 | True | 6.9 mins | 0.8691 | 0.1828 | 0.2263 |
| run_00 | dopri5 | 128 | 2 | 1e-3 | False | 5.1 mins | 0.8689 | 0.1832 | 0.2319 |
| run_02 | dopri5 | 256 | 2 | 1e-3 | True | 6.9 mins | 0.8278 | 0.2400 | 0.2214 |
| run_03 | dopri5 | 256 | 3 | 1e-3 | True | 8.2 mins | 0.8258 | 0.2426 | 0.2482 |

### Key Findings
1. **Best Model:** Configuration `run_04` (`hidden=128`, `n_layers=2`, `lr=5e-4`, `method="dopri5"`) achieved the highest windowed performance of **`0.8988` $R^2$**.
2. **Learning Rate Sensitivity:** Slower learning rate (`5e-4` vs `1e-3`) systematically improved performance across all architectures, indicating that `1e-3` was overshooting.
3. **Model Capacity:** The smaller `hidden=128` network outperformed the larger `hidden=256` versions, likely due to overfitting on the small 8-trial training dataset.
4. **Solver Efficiency:** `dopri5` was more than 4x faster than `rk4` with almost identical accuracy, showing it is the superior choice for future training/evaluations.
