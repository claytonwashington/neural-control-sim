# Digital Twin Modeling Ideas

This document tracks modeling hypotheses and architectures to improve prediction accuracy ($R^2$ on 200ms prediction windows) for the Cleo digital twin spiking neural network model. All results below use the **standardized 40/10 train/test split** on `data/training_trials.h5` with `seed=42`.

**Current best overall: EnKF (Q=0.1, R=0.01, N=64) → R²=0.9997** (1127ms inference on 2080 Ti)
**Current best acausal: Latent NODE z=64, h=256, lr=5e-4 → R²=0.9387**
**Current best causal/deployable: Causal Latent CA-NODE z=64, h128, lr=3e-4 → R²=0.8252** (173ms inference)
**Channel-level baseline: CA-NODE h=128, l=2, lr=1e-4 → R²=0.9009**

---

## Phase 2: Channel-Level Sweeps (Complete)

### 1. Discrete-Time Sequence Models (GRU/LSTM)
**Status**: ✅ TESTED — ❌ Did not beat baseline
**Results dir**: `results/sweep_sequence_40_10/`
**Branch**: `modeling-dev`
- Best GRU: h128, lr=1e-4 → **R²=0.5798** ❌
- Best Discrete CA-NODE (Euler): h128, lr=5e-5 → **R²=0.8968**
- Control-affine inductive bias is essential (+29.5% R² over GRU).

### 2. Skip Connections
**Status**: ✅ TESTED — ❌ Did not beat baseline
**Results dir**: `results/sweep_skip_40_10/`
**Branch**: `feature/skip-connections` → `cleo-worktrees/skip-connections`
- No-skip baseline: **R²=0.9009** ✅
- Best skip (linear, swd=1.0): **R²=0.8911** ❌ (-1.1%)
- All skip paths hurt due to overfitting.

### 3. Spectral Loss
**Status**: ✅ TESTED — ❌ Did not beat baseline
**Results dir**: `results/sweep_spectral_40_10/`
**Branch**: `feature/spectral-loss` → `cleo-worktrees/spectral-loss`
- Best spectral (α=2.0): **R²=0.8993** ❌
- Zero improvement. Time-domain MSE already captures frequency content.

### 4. Multi-Rate Integration
**Status**: ✅ TESTED — ❌ No benefit
**Results dir**: `results/sweep_multirate_40_10/`
**Branch**: `feature/multi-rate-integration` → `cleo-worktrees/multi-rate-integration`
- M=1,5,10 all give **R²=0.8907** (identical)
- 1ms timestep already resolves the dynamics; sub-ms sub-stepping adds nothing.
- Weights differ across M but converge to same solution.

---

## Phase 3: Latent Space Models (Complete)

### 5. Latent Neural ODE (LFADS-Style) — Non-Causal
**Status**: ✅ TESTED — ✅ BEST ACAUSAL MODEL
**Results dir**: `results/sweep_latent_node_40_10/`
**Branch**: `modeling-dev`
- z=64, h=256, lr=5e-4 → **R²=0.9387** 🏆 (acausal, offline only)
- z=64, h=128, lr=1e-3 → **R²=0.9340**
- z=32 → R²≈0.44–0.63 (too compressed)
- **Key insight**: z=64 is the critical latent dimension.

### 6. Causal Latent CA-NODE
**Status**: ✅ TESTED — ✅ BEST DEPLOYABLE LATENT MODEL
**Results dir**: `results/sweep_latent_canode_40_10/` (z=32), `results/causal_z64_*` (z=64)
**Branch**: `modeling-dev`
- z=32, pw=200, h128 → **R²=0.5531** ❌ (initial, wrong z_dim)
- z=64, h128, lr=3e-4, pw=200 → **R²=0.8252** ✅ (173ms inference)
- z=64, h128, lr=5e-4, pw=500 → **R²=0.8203** (longer window, marginal)
- z=64, h256, lr=5e-4, pw=200 → **R²=0.7091** (overfitting)
- **Key insight**: z=32→z=64 caused +47% R² jump; must match acausal z_dim.

---

## Phase 4: Closing the Causal-Acausal Gap (Active)

### 7. Encoder Distillation (Acausal Teacher → Causal Student)
**Status**: 🔄 IN PROGRESS — training on gpu1:4
**Results dir**: `results/distill_encoder/`
**Branch**: `feature/encoder-distillation` → `cleo-worktrees/encoder-distillation`
**Hypothesis**: Freeze acausal ODE+decoder, train causal GRU encoder to match z₀_teacher targets.

### 8. Hybrid Distillation
**Status**: 🔄 IN PROGRESS — launching α sweep on gpu1:0,1,2,7
**Results dir**: `results/hybrid_distill_aN/`
**Branch**: `feature/hybrid-distillation` → `cleo-worktrees/hybrid-distillation`
**Hypothesis**: Combined loss α·MSE(z₀_student, z₀_teacher) + (1-α)·MSE(x̂, x_true).

### 9. Causal Grokking (500 Trials, 1000 Epochs)
**Status**: 🔄 IN PROGRESS — training on gpu2:0
**Results dir**: `results/grok_causal_z64/`
**Branch**: `feature/grokking` → `cleo-worktrees/grokking`
**Hypothesis**: 10× more data + 5× more epochs + higher weight decay → causal encoder grokking.

### 10. Acausal Grokking (Better Teacher)
**Status**: 🔄 IN PROGRESS — training on gpu2:1
**Results dir**: `results/grok_acausal_z128/`
**Branch**: `feature/grokking` → `cleo-worktrees/grokking`
**Hypothesis**: z=128, h=512 on 500 trials → push acausal R² from 0.94 → 0.97+.

### 11. Ensemble Kalman Filter (EnKF) — RTX 2080 Ti
**Status**: ✅ TESTED — ✅ BEST OVERALL MODEL (but slow)
**Results dir**: `results/enkf/`
**Branch**: `feature/kalman-filter` → `cleo-worktrees/kalman-filter`
- Q=0.1, R=0.01, N=64 → **R²=0.9997** (1127ms inference ❌)
- Q=0.01, R=0.01, N=64 → **R²=0.9937** (1141ms inference ❌)
- Q=0.1, R=0.1, N=64 → **R²=0.9936** (1126ms inference ❌)
- **Key insight**: EnKF with learned ODE dynamics essentially solves the problem. Bottleneck is inference speed (64 parallel ODE solves per step).

### 12. Delayed Residual Correction (Complementary Filter)
**Status**: 🔄 IN PROGRESS — training on gpu1:6
**Results dir**: `results/residual_correction/`
**Branch**: `feature/residual-correction` → `cleo-worktrees/residual-correction`
**Hypothesis**: MLP predicts acausal−causal z₀ residual at 100ms lag. Smith Predictor analogy.

### 13. EnKF A100 Sweep (Speed Optimization)
**Status**: 🔄 IN PROGRESS — sweeping ensemble sizes on gpu2:5
**Results dir**: `results/enkf_a100_N*/`
**Branch**: `feature/enkf-a100` → `cleo-worktrees/enkf-a100`
**Hypothesis**: A100 GPUs are ~3-4× faster than 2080 Ti. Sweeping ensemble sizes N∈{16,32,64,128} to find the speed/accuracy Pareto frontier. Goal: get inference under 200ms while maintaining R²>0.95.

### 14. Additional Causal Sweeps
**Status**: 🔄 IN PROGRESS — on gpu2:2,3,4
**Results dir**: `results/causal_z128_*`, `results/causal_z64_500ep_*`, `results/causal_z64_pw1000_*`
- z=128, h128, pw200 → testing if z>64 helps further
- z=64, 500ep, wd=5e-5 → grokking-lite on 50 trials
- z=64, pw=1000 → does 1 second of context dramatically help?
