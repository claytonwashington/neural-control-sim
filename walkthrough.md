# Cleo Digital Twin Modeling Walkthrough

This document provides a walkthrough of the Digital Twin modeling pipeline designed to perform system identification on the Cleo closed-loop spiking neural network model.

---

## 1. Pipeline Overview

The system identification pipeline consists of three core components:
1. **Data Generation**: Running SNN simulation trials in Cleo under stochastic Ornstein-Uhlenbeck (OU) optogenetic stimulation, recording multi-unit activity (MUA).
2. **Linear Subspace Baseline (N4SID)**: Fitting a state-space model to set a baseline prediction accuracy.
3. **Control-Affine Neural ODE (CA-NODE)**: Training a continuous-time neural ODE to model non-linear neural dynamics under optogenetic input.

---

## 2. Code Structure

- [modeling/models/canode.py](file:///snel/home/cbwash2/cleo-worktrees/spectral-loss/modeling/models/canode.py): Definition of the CA-NODE model architecture, including DriftNet, ControlNet, and optional state/input skip connections.
- [modeling/scripts/fit_canode.py](file:///snel/home/cbwash2/cleo-worktrees/spectral-loss/modeling/scripts/fit_canode.py): Main script to train the CA-NODE model on generated HDF5 trials.
- [modeling/scripts/sweep_canode.py](file:///snel/home/cbwash2/cleo-worktrees/spectral-loss/modeling/scripts/sweep_canode.py): Multi-GPU coordinator to run parallel sweeps across hidden dimensions, solvers, skip connections, and loss parameters.
- [results/dashboard.html](file:///snel/home/cbwash2/cleo-worktrees/spectral-loss/results/dashboard.html): A living HTML dashboard displaying metrics, sweeps, predictions, and chronological findings.

---

## 3. Key Modeling Hypotheses & Sweeps

### Phase 2: Data Scaling
- **Goal**: Scaling the dataset to 50 trials (48 train, 2 test) to unlock larger models.
- **Result**: `h128 l2 lr=1e-4` achieved the best prediction performance on a 200ms window ($R^2 \approx 0.9016$). Bigger models (h256/h512) tended to overfit even with scaled data.

### Phase 2.5 & 2.6: Skip Connections
- **Goal**: Allowing high-frequency input transients to bypass the continuous ODE integrator.
- **Result**: Raw MLP skips overfit stimulus patterns ($R^2 \approx 0.8895$). Constraining the skip path to linear-only and adding strong L2 regularization (`skip_weight_decay = 1.0`) recovered the performance ceiling ($R^2 \approx 0.9144$), but did not outperform the pure CA-NODE baseline ($R^2 \approx 0.9219$).

### Phase 2.7: Multi-Scale / Multi-Rate Integration
- **Goal**: Splitting fast input-driven variables from slow baseline variables.
- **Result**: Decoupling using sub-stepping factor $M=10$ with `zero_fast` state split achieved $R^2 \approx 0.9130$, showing that sub-stepping the optogenetic integration loop improves SNN dynamics resolution.

### Idea 3: Frequency-Aware (Spectral) Loss
- **Goal**: Penalizing discrepancies in the FFT-magnitude domain to force the optimizer to align high-frequency spectral components:
  $$\mathcal{L} = \text{MSE}(x_{\text{pred}}, x_{\text{true}}) + \alpha \cdot \text{MSE}(|\text{FFT}(x_{\text{pred}})|, |\text{FFT}(x_{\text{true}})|)$$
- **Result**: Sweeping $\alpha \in [0, 2.0]$ successfully reduced validation FFT loss from `20.57` (at $\alpha=0.01$) to `20.32` (at $\alpha=2.0$). It aligns high-frequency predictions at the cost of a minor time-domain $R^2$ trade-off (maintaining $R^2 \approx 0.920$ vs baseline $0.922$).

---

## 4. Run Guide

### Training a Single Model
To train CA-NODE with a specific spectral alpha:
```bash
conda run -n dtmodeling python -m modeling.scripts.fit_canode \
  --data data/training_trials_50.h5 \
  --hidden 128 \
  --n-layers 2 \
  --lr 1e-4 \
  --spectral-alpha 0.1 \
  --compile
```

### Running the Sweeps
To run a parallel GPU sweep over spectral alpha configurations:
```bash
conda run -n dtmodeling python -m modeling.scripts.sweep_canode \
  --data data/training_trials_50.h5 \
  --grid spectral \
  --gpu-ids 1,2,3,4,5,6 \
  --max-workers 6 \
  --n-epochs 200
```
