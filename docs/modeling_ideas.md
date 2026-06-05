# Digital Twin Modeling Ideas

This document tracks modeling hypotheses and architectures to improve prediction accuracy for the Cleo digital twin spiking neural network model. All results use the **standardized 40/10 train/test split** on `data/training_trials.h5` with `seed=42`.

**Current best overall: EnKF (Q=0.1, R=0.01, N=64) → R²=0.9997** (1183ms inference — not real-time)
**Current best acausal: Latent NODE z=64, h=256 → R²=0.9387**
**Current best deployable: Causal Latent CA-NODE z=64, h128, lr=3e-4 → R²=0.8252** (173ms)
**Channel-level baseline: CA-NODE h=128, l=2 → R²=0.9009**

---

## Phase 2: Channel-Level Sweeps (Complete)

### 1. Discrete-Time Sequence Models (GRU/LSTM)
**Status**: ✅ TESTED — ❌ Did not beat baseline
**Results dir**: `results/sweep_sequence_40_10/`
**Branch**: `modeling-dev`
- Best GRU: h128, lr=1e-4 → R²=0.5798
- Best Discrete CA-NODE (Euler): h128, lr=5e-5 → R²=0.8968

### 2. Skip Connections
**Status**: ✅ TESTED — ❌ Did not beat baseline
**Results dir**: `results/sweep_skip_40_10/`
**Branch/Worktree**: `feature/skip-connections` / `cleo-worktrees/skip-connections`
- Best (linear, swd=1.0): R²=0.8911 ❌ (-1.1%)

### 3. Spectral Loss
**Status**: ✅ TESTED — ❌ Did not beat baseline
**Results dir**: `results/sweep_spectral_40_10/`
**Branch/Worktree**: `feature/spectral-loss` / `cleo-worktrees/spectral-loss`
- Best (α=2.0): R²=0.8993 ❌

### 4. Multi-Rate Integration
**Status**: ✅ TESTED — ❌ No benefit
**Results dir**: `results/sweep_multirate_40_10/`
**Branch/Worktree**: `feature/multi-rate-integration` / `cleo-worktrees/multi-rate-integration`
- M=1,5,10 all give R²=0.8907 (identical)

---

## Phase 3: Latent Space Models (Complete)

### 5. Latent Neural ODE — Non-Causal
**Status**: ✅ TESTED — ✅ BEST ACAUSAL
**Results dir**: `results/sweep_latent_node_40_10/`
**Branch**: `modeling-dev`
- z=64, h=256 → R²=0.9387 🏆

### 6. Causal Latent CA-NODE
**Status**: ✅ TESTED — ✅ BEST DEPLOYABLE LATENT
**Results dir**: `results/causal_z64_*/`, `results/causal_z128_*/`
**Branch**: `modeling-dev`
- z=64, h128, lr=3e-4 → R²=0.8252 ✅ (173ms)
- z=64, h128, pw=500 → R²=0.8203
- z=128, h128, lr=3e-4 → R²=0.8430 ✅ (142ms) — marginal +2% from doubling z

---

## Phase 4: Closing the Causal-Acausal Gap (Active)

### 7. Encoder Distillation (Acausal Teacher → Causal Student)
**Status**: ✅ TESTED — ❌ Below directly-trained causal model
**Results dir**: `results/distill_encoder/`
**Branch/Worktree**: `feature/encoder-distillation` / `cleo-worktrees/encoder-distillation`
- R²=0.6639, Inference=64.73ms
- Frozen ODE too sensitive to z₀ distribution mismatch — small encoder errors compound through ODE integration.

### 8. Hybrid Distillation
**Status**: ✅ TESTED — ❌ Worse than direct causal
**Results dir**: `results/hybrid_distill_aN/`
**Branch/Worktree**: `feature/hybrid-distillation` / `cleo-worktrees/hybrid-distillation`
- Loss = α·MSE(z₀_student, z₀_teacher) + (1-α)·MSE(x̂, x_true)
- Sweep α ∈ {0.1, 0.3, 0.5, 0.7}
- Best: α=0.5 R²=0.4826 — significantly worse than direct causal (R²=0.8252)
- Distillation loss actively conflicts with reconstruction loss during joint training

### 9. Causal Grokking (500 Trials, 1000 Epochs)
**Status**: 🔄 IN PROGRESS — gpu2:0
**Results dir**: `results/grok_causal_z64/`
**Branch/Worktree**: `feature/grokking` / `cleo-worktrees/grokking`

### 10. Acausal Grokking (Better Teacher)
**Status**: 🔄 IN PROGRESS — gpu2:1
**Results dir**: `results/grok_acausal_z128/`
**Branch/Worktree**: `feature/grokking` / `cleo-worktrees/grokking`

### 11. Ensemble Kalman Filter (EnKF)
> **⚠️ PREVIOUS RESULTS INVALID**: R²=0.9997 used per-timestep observation updates (filtering), not blind prediction. Re-running with observation rate sweep to get fair comparison. See eval_enkf_obsrate.py.
**Status**: ✅ TESTED — ✅ BEST OVERALL (not real-time)
**Results dir**: `results/enkf/` (2080 Ti), `results/enkf_a100_N*/` (A100)
**Branch/Worktree**: `feature/kalman-filter` / `cleo-worktrees/kalman-filter` and `feature/enkf-a100` / `cleo-worktrees/enkf-a100`
- Best (N=64, Q=0.1, R=0.01): R²=0.9997 — but ~1.2s inference (both 2080 Ti and A100)
- N=16 still achieves R²=0.984 at same inference cost
- Bottleneck is ODE integration count, not GPU speed
- Key insight: ALL R2=0.9997 results are from filtering (seeing observations), not prediction
- Observation rate sweep shows R2=0.9997 at ALL K values (1 through 999) — single Kalman update at t=0 does all the work
- The acausal encoder z0 + 1 observation correction + ODE rollout = near-perfect
- NEXT: test with CAUSAL encoder init to see if EnKF rescues R2=0.82

### 12. Delayed Residual Correction (Complementary Filter)
**Status**: ✅ TESTED — ⚠️ Cross-architecture mismatch limits gains
**Results dir**: `results/residual_correction/`
**Branch/Worktree**: `feature/residual-correction` / `cleo-worktrees/residual-correction`
- R²=0.3445 (cross-arch: causal encoder z₀ → acausal ODE)
- Closed 59% of gap between cross-arch and oracle
- MLP adds only +2ms latency
- Would need same-architecture ODE to be a fair test

### 13. EnKF A100 Speed Sweep
**Status**: ✅ TESTED — ❌ No speedup from A100
**Results dir**: `results/enkf_a100_N*/`
**Branch/Worktree**: `feature/enkf-a100` / `cleo-worktrees/enkf-a100`
- All ensemble sizes (16-128) give ~1.2s inference on A100
- Bottleneck is sequential ODE integration via dopri5, not GPU parallelism
- **Next step**: Try fixed-step Euler integration in EnKF predict step to eliminate adaptive solver overhead



### 15. EnKF with Causal Encoder Init (4-Way Sweep)
**Status**: ✅ TESTED — ✅ Same-arch EnKF dramatically improves causal model
**Results dir**:  (in enkf-causal worktree)
**Branch/Worktree**:  / 
- A1 (causal ODE, no delay, K=1): R²=0.9997 | K=20: 0.9621 | K=999: 0.905
- A2 (causal ODE, 10ms delay, K=1): R²=0.9563 | K=20: 0.932 | K=999: 0.826
- B1 (acausal ODE, no delay, K=1): R²=0.9995 | K=20: 0.958 | K=999: 0.797
- B2 (acausal ODE, 10ms delay, K=1): R²=0.888 | K=20: 0.790 | K=999: -0.535
- **Key finding**: Same-arch (A) is far more robust to sparse updates + delay than cross-arch (B)
- **Deployment target**: A2 K=20 (50Hz updates, 10ms delay) → R²=0.93, practical and deployable
- Still has t=0 Kalman update artifact — but even with that caveat, the improvement is real




### 19. Optoclamp: Closed-Loop Rate Clamping
**Status**: 🔄 BUILDING on feature/optoclamp
**Branch/Worktree**: `feature/optoclamp` / `cleo-worktrees/optoclamp`
- Recapitulate Newman et al. 2015 optoclamp experiment
- PI controller baseline (sweep Kp, Ki)
- Neural ODE MPC using causal CA-NODE (z=64, H=20 horizon)
- Compare tracking RMSE, settling time, control effort
- Target: rate clamping at 50%, 75%, 125% of baseline rate
- Metrics: RMSE, settling time, steady-state error, overshoot
### 18. Same-Architecture Residual Correction v2
**Status**: ❌ DONE — Failed. R²=0.80 < baseline 0.82. Topology problem confirmed.
**Branch/Worktree**: `feature/residual-v2` / `cleo-worktrees/residual-v2`
- Fixes Exp 12 failure: uses CAUSAL ODE+decoder (same-arch, no cross-arch mismatch)
- ResidualMLP(z0_causal, z0_acausal_delayed) -> delta_z0, trained end-to-end through frozen causal ODE
- Last layer initialized to zero -> starts as identity -> floor = causal baseline (R2=0.82)
- Acausal encoder at 100ms lag provides richer context as auxiliary MLP input
- Loss = MSE(decoder(ODE(z0_corrected)), x_true) through causal pipeline
- Expected: R2 >= 0.82 guaranteed, hopefully closer to 0.93 (EnKF level)

#### Why Residual Correction May Fundamentally Fail: The Topology Problem
The causal and acausal models were trained independently. Even though both predict the
same firing rates x(t), their latent spaces z(t) can be arbitrarily rotated, scaled, or
non-linearly warped relative to each other. The residual dz0 = z_acausal - z_causal is
therefore not a clean structured correction vector — it is an inconsistent mapping between
two unaligned coordinate systems. A small MLP cannot untangle a global diffeomorphism.

#### Fix 1: Direct Trajectory Distillation (= Exp 7 approach, revisited)
Freeze acausal encoder + ODE + decoder. Train a new causal encoder from scratch with
L = ||z_causal - z_acausal||^2. Forces the causal encoder to adopt the acausal topology.
Once aligned, residuals become meaningful because both models speak the same language.
- We tried this (Exp 7), R2=0.66 — frozen ODE was too rigid for the imperfect causal z0.
- **Variant**: Also fine-tune the decoder to tolerate noisier causal z0.

#### Fix 2: Vector Field Alignment (Match the Dynamics)
Instead of matching z0 points, match the ODE vector fields:
L = ||f_theta(z_causal) - f_theta(z_acausal)||^2.
Guarantees that even if z0 is slightly noisy, the ODE pushes it in the same direction.
Can be combined with trajectory distillation as a regularizer.

#### Fix 3: Contrastive Alignment (InfoNCE / Mutual Information)
Project both z_causal and z_acausal into a shared low-dim subspace via linear heads.
Maximize cosine similarity for same-timestep pairs, push apart different-timestep pairs.
Maximizes mutual information without forcing rigid 1:1 mapping — gives the causal model
flexibility to handle missing future context.
- Most flexible approach; allows the causal space to be a *compressed* version of acausal.

#### Recommended Path Forward
Start with **Direct Trajectory Distillation** but with a joint loss:
L = alpha * ||z_causal - z_acausal||^2 + (1-alpha) * reconstruction_loss
This was Exp 8 (hybrid distillation, R2=0.48) — but it used random init for the causal ODE.
Better approach: initialize causal ODE from acausal weights, then distill with alpha schedule
(start high to force alignment, decay to let the model specialize for causal inference).
### 17. Full-Trial EnKF — 30s Deployment Stability
**Status**: ✅ TESTED — K=1 D=0 is extraordinary, realistic configs degrade
**Branch/Worktree**: `feature/enkf-fulltrial` / `cleo-worktrees/enkf-fulltrial`
- K=1, D=0: FIT%=90.5, R2=0.993 stable over 30s — beats acausal model (0.94)
- K=1, D=10: FIT%=51.6 — 10ms delay destroys long-horizon stability
- K=20, D=0: FIT%=50.8 — sparse corrections insufficient over 30s
- K=20, D=10: FIT%=38.5 — not viable for long-term tracking
- **Key insight**: Small errors from delayed/sparse corrections compound over 30,000 steps
- **Implication**: K=1 D=0 is unrealistic for deployment. Need periodic re-encoding OR Path A PredNet
- Over SHORT horizons (200ms), K=20 D=10 is fine (R2=0.93 from Exp 15)
### 16. Control-Relevant Metrics Evaluation
**Status**: ✅ TESTED — Critical insights for deployment
**Results dir**: `results/ctrl_metrics_{causal_z64,causal_z128,acausal_z64}/` (in control-metrics worktree)
**Branch/Worktree**: `feature/control-metrics` / `cleo-worktrees/control-metrics`
- Free-run FIT% = 0 for ALL models (ODE diverges over 30s, re-encoding mandatory)
- Multi-horizon: linear error growth (good), R2>0.84 at H=20 for causal z64
- Jacobian accuracy (dg/du): cos_sim=0.997 — control matrix extremely well-learned
- Controllability Gramian: only 3-5 effective control dimensions out of 64/128
- **Implication**: MPC viable at H<=20 steps. Re-encoding every ~200ms is required.
### 14. Additional Causal Sweeps
**Status**: ✅ TESTED — z=128 marginal +2%, 500ep and pw=1000 no benefit
**Results dir**: `results/causal_z128_*`, `results/causal_z64_500ep_*`, `results/causal_z64_pw1000_*`
- z=128: R²=0.8430, 142ms ✅ (marginal +2% over z=64)
- z=64, 500ep, wd=5e-5: R²=0.6664, 173ms ❌ (extra weight decay hurts)
- z=64, pw=1000: R²=0.8242, 136ms (no benefit over pw=200)
