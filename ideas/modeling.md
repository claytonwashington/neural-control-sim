# Digital Twin Modeling — Experiment Log

This is the canonical experiment tracker for the Cleo digital twin project. All experiments
are numbered chronologically. All results use the **standardized 40/10 train/test split** on
`data/training_trials.h5` (50 trials) with `seed=42` unless otherwise noted.

---

## Current Leaderboard

| Rank | Model | R² | Inference / Step | Causal? | Deployable? |
|------|-------|-----|------------------|---------|-------------|
| 1 | EnKF on Aligned Model K=1 D=0 (Exp 21) | **0.9997** | ~6.0ms (1.2s/trial) | ✅ | ❌ (too slow) |
| 2 | EnKF on Aligned Model K=20 D=10 (Exp 21) | **0.9391** | ~2.3ms (460ms/trial) | ✅ | ❌ (too slow) |
| 3 | Latent NODE acausal (Exp 3) | 0.9387 | 71ms (initial) | ❌ | ❌ |
| 4 | Periodic Re-Encoding K=5 D=0 (Exp 22) | **0.8770** | **0.81ms** (162ms/trial) | ✅ | ✅ |
| 5 | Periodic Re-Encoding K=20 D=10 (Exp 22) | **0.8701** | **0.56ms** (112ms/trial) | ✅ | ✅ |
| 6 | Aligned Distillation (Exp 20) | 0.8640 | ~170ms (initial) | ✅ | ✅ |
| 7 | Causal CA-NODE z64 (Exp 5) | 0.8252 | ~173ms (initial) | ✅ | ✅ |
| 8 | GRU (Exp 2) | 0.6900 | — | ✅ | ✅ |
| 9 | N4SID (Exp 1) | 0.4000 | <1ms | ✅ | ✅ |

---

## Standard Experiment Outputs

Every experiment **MUST** produce:

1. **Sample 200ms prediction traces** — 4-channel overlay of predicted vs actual firing rates
2. **Training curves** — loss vs epoch (train and val)
3. **R² bar chart** — comparison against baselines (N4SID=0.40, causal=0.82, acausal=0.94)
4. **For control experiments** — tracking RMSE, settling time, steady-state error, control effort
5. **For top performers** — N4SID vs real data vs model prediction comparison figure

### Control Metric Definitions

- **Steady-state error** = mean |actual_rate - target| during last 200ms of clamp period
- **Tracking RMSE** = sqrt(mean((rate - target)²)) over entire clamp period
- **Settling time** = time to reach and stay within 10% of target for 50ms

---

## Phase 1: Architecture Search (Exp 1–4)

> Goal: Find the best model architecture for 200ms neural population prediction.

### Experiment 1. N4SID Baseline
**Status**: ✅ COMPLETE
**Results dir**: `results/n4sid_40_10/`
**Branch**: `modeling-dev`
- Classical subspace identification (linear state-space model)
- R²=0.40 — serves as the linear baseline
- Sub-millisecond inference, trivially deployable
- **Key finding**: Linear models capture ~40% of variance; nonlinear dynamics matter

### Experiment 2. GRU Baseline
**Status**: ✅ COMPLETE
**Results dir**: `results/gru/`, `results/sweep_sequence_40_10/`
**Branch**: `modeling-dev`
- Best GRU: h=128, lr=1e-4 → R²=0.5798 (channel-level)
- Best Discrete CA-NODE (Euler): h=128, lr=5e-5 → R²=0.8968
- Latent GRU: R²≈0.69
- **Key finding**: Continuous-time ODE integration significantly outperforms discrete-time

### Experiment 3. Latent Neural ODE (Acausal)
**Status**: ✅ COMPLETE — ✅ BEST ACAUSAL MODEL
**Results dir**: `results/sweep_latent_node_40_10/`
**Branch**: `modeling-dev`
- **Best config**: z=64, h=256, layers=2, lr=5e-4, solver=dopri5
- **R²=0.9387** 🏆 | Inference: 71ms
- Encoder sees full 200ms window (bidirectional GRU)
- 6-run sweep over z∈{32,64}, h∈{128,256}, lr∈{5e-4,1e-3}
- **Key finding**: Latent space compression (50 channels → 64 dims) dramatically improves generalization

### Experiment 4. Channel-Level CA-NODE Sweeps
**Status**: ✅ COMPLETE
**Results dir**: `results/sweep_skip_40_10/`, `results/sweep_spectral_40_10/`, `results/sweep_multirate_40_10/`
**Branch/Worktrees**: `feature/skip-connections`, `feature/spectral-loss`, `feature/multi-rate-integration`
- Skip connections (best: linear, swd=1.0): R²=0.8911 ❌ (−1.1%)
- Spectral loss (best: α=2.0): R²=0.8993 ❌
- Multi-rate integration (M=1,5,10): R²=0.8907 (identical — no benefit)
- Channel-level CA-NODE baseline: R²=0.9009
- **Key finding**: Channel-level models plateau at ~0.90; need latent space to break through

---

## Phase 2: Causal Model Development (Exp 5–8)

> Goal: Build a causal (real-time deployable) model that approaches acausal accuracy.
> The causal encoder only sees *past* data — no future context.

### Experiment 5. Causal Latent CA-NODE
**Status**: ✅ COMPLETE — ✅ BEST CAUSAL BASELINE
**Results dir**: `results/causal_z64_pw200_h128_lr3e4/` (best), `results/causal_z64_*/`, `results/causal_z128_*/`
**Branch**: `modeling-dev`
- **Best config**: z=64, h=128, lr=3e-4, pw=200
- **R²=0.8252** | Inference: 173ms
- z=128 variant: R²=0.8430, 142ms (marginal +2% from doubling z)
- z=64, pw=500: R²=0.8203 | z=64, pw=1000: R²=0.8242 (no benefit from wider windows)
- z=64, 500ep, wd=5e-5: R²=0.6664 ❌ (weight decay hurts)
- **Causal-acausal gap**: 0.9387 − 0.8252 = 0.1135 (12% R² gap)
- **Key finding**: Causal encoder z₀ quality is the bottleneck, not ODE capacity

### Experiment 6. Skip Connections for Causal Model
**Status**: ✅ COMPLETE — ❌ FAILED
**Results dir**: `results/sweep_skip_40_10/`
**Branch/Worktree**: `feature/skip-connections` / `cleo-worktrees/skip-connections`
- R²=−0.24 — catastrophic failure
- Skip connections bypass the ODE, creating shortcuts that prevent proper dynamics learning
- **Key finding**: Residual paths around the ODE are harmful for latent dynamics models

### Experiment 7. Encoder Distillation (Acausal → Causal)
**Status**: ✅ COMPLETE — ❌ BELOW BASELINE
**Results dir**: `results/distill_encoder/`
**Branch/Worktree**: `feature/encoder-distillation` / `cleo-worktrees/encoder-distillation`
- Freeze acausal encoder + ODE + decoder; train causal encoder to match z₀
- R²=0.6639 | Inference: 64.73ms
- Frozen ODE too sensitive to z₀ distribution mismatch — small encoder errors compound through integration
- **Key finding**: Encoder distillation alone fails because the ODE amplifies distributional shift

### Experiment 8. Hybrid Distillation
**Status**: ✅ COMPLETE — ❌ WORSE THAN BASELINE
**Results dir**: `results/hybrid_distill_aN/`
**Branch/Worktree**: `feature/hybrid-distillation` / `cleo-worktrees/hybrid-distillation`
- Loss = α·MSE(z₀_student, z₀_teacher) + (1−α)·MSE(x̂, x_true)
- Sweep α ∈ {0.1, 0.3, 0.5, 0.7}
- Best: α=0.5 → R²=0.4826 — significantly worse than direct causal (R²=0.8252)
- Distillation loss actively conflicts with reconstruction loss during joint training
- Used random init for causal ODE (see Exp 20 for improved version)
- **Key finding**: Joint distillation + reconstruction with competing gradients degrades both objectives

---

## Phase 3: Transfer & Correction (Exp 9–14)

> Goal: Use the acausal model to improve the causal model via transfer learning,
> residual correction, or filtering approaches.

### Experiment 9. Cross-Architecture Encoder Distillation
**Status**: ✅ COMPLETE — ❌ SAME AS EXP 7
**Results dir**: `results/distill_encoder/`
**Branch/Worktree**: `feature/encoder-distillation` / `cleo-worktrees/encoder-distillation`
- Variant of Exp 7 with cross-architecture teacher (acausal NODE) → student (causal CA-NODE)
- R²=0.6639 — same failure mode as Exp 7
- Architecture mismatch between teacher and student latent spaces adds additional difficulty
- **Key finding**: Cross-architecture distillation is at least as hard as same-architecture

### Experiment 10. Grokking (1000 Epochs, 500 Trials)
**Status**: 🔄 IN PROGRESS — gpu2
**Results dir**: `results/grok_causal_z64/` (causal), `results/grok_acausal_z128/` (acausal)
**Branch/Worktree**: `feature/grokking` / `cleo-worktrees/grokking`
- Hypothesis: extended training on larger dataset may unlock delayed generalization (grokking)
- Causal z=64 on 500 trials, 1000 epochs (gpu2:0)
- Acausal z=128 on 500 trials, 1000 epochs (gpu2:1) — train better teacher
- Data: `data/training_trials_500.h5` (10× larger)
- **Status update needed**: Check training curves for signs of late-stage improvement

### Experiment 11. 500-Trial Dataset Expansion
**Status**: ✅ COMPLETE — Dataset generated
**Data file**: `data/training_trials_500.h5`
- 500 trials (vs 50 original), same generation pipeline
- Used for Exp 10 (grokking) and future experiments
- 10× more training data to reduce overfitting risk
- **Key finding**: Dataset ready; impact TBD from Exp 10 results

### Experiment 12. Delayed Residual Correction (Complementary Filter)
**Status**: ✅ COMPLETE — ⚠️ LIMITED BY CROSS-ARCH MISMATCH
**Results dir**: `results/residual_correction/`
**Branch/Worktree**: `feature/residual-correction` / `cleo-worktrees/residual-correction`
- Causal encoder z₀ → acausal ODE (cross-architecture: mismatched latent spaces)
- R²=0.3445
- Closed 59% of gap between cross-arch baseline and oracle
- MLP adds only +2ms latency
- **Key finding**: Cross-architecture mismatch is fatal — need same-architecture pipeline (→ Exp 18)

### Experiment 13. EnKF A100 Speed Sweep
**Status**: ✅ COMPLETE — ❌ NO SPEEDUP
**Results dir**: `results/enkf_a100_N*/`
**Branch/Worktree**: `feature/enkf-a100` / `cleo-worktrees/enkf-a100`
- All ensemble sizes (N=16–128) give ~1.2s inference on A100 — identical to 2080 Ti
- Bottleneck is sequential ODE integration via dopri5, not GPU parallelism
- **Key finding**: EnKF speed is ODE-bound, not GPU-bound. Fixed-step Euler may help.

### Experiment 14. EnKF Baseline (Acausal Encoder Init)
**Status**: ✅ COMPLETE — ✅ BEST OVERALL (not real-time)
**Results dir**: `results/enkf/` (2080 Ti), `results/enkf_a100_N*/` (A100)
**Branch/Worktree**: `feature/kalman-filter` / `cleo-worktrees/kalman-filter`

> **⚠️ PREVIOUS RESULTS INVALID**: R²=0.9997 used per-timestep observation updates
> (filtering, not blind prediction). Re-ran with observation rate sweep for fair comparison.

- Best (N=64, Q=0.1, R=0.01): R²=0.9997 — but ~1.2s inference
- N=16 still achieves R²=0.984 at same inference cost
- **Key insight**: ALL R²=0.9997 results are from filtering (seeing observations)
- Observation rate sweep: R²=0.9997 at ALL K values (1–999) — single Kalman update at t=0 does all the work
- The acausal encoder z₀ + 1 observation correction + ODE rollout = near-perfect
- **Key finding**: EnKF with acausal init is essentially cheating — the encoder already gives a great z₀. Real test is with causal init (→ Exp 15)

---

## Phase 4: EnKF Refinement (Exp 15–17)

> Goal: Make EnKF work with the causal encoder for deployable real-time filtering.

### Experiment 15. EnKF with Causal Encoder Init (4-Way Sweep)
**Status**: ✅ COMPLETE — ✅ SAME-ARCH EnKF RESCUES CAUSAL MODEL
**Results dir**: (in enkf-causal worktree)
**Branch/Worktree**: `feature/enkf-causal` / `cleo-worktrees/enkf-causal`

| Config | K=1 | K=20 | K=999 |
|--------|-----|------|-------|
| A1: causal ODE, D=0 | R²=0.9997 | 0.9621 | 0.905 |
| A2: causal ODE, D=10ms | R²=0.9563 | **0.932** | 0.826 |
| B1: acausal ODE, D=0 | R²=0.9995 | 0.958 | 0.797 |
| B2: acausal ODE, D=10ms | R²=0.888 | 0.790 | −0.535 |

- **Deployment target**: A2 K=20 (50Hz updates, 10ms delay) → **R²=0.93** ← practical and deployable
- Same-arch (A configs) far more robust to sparse updates + delay than cross-arch (B configs)
- Still has t=0 Kalman update artifact — but even with that caveat, the improvement is real
- **Key finding**: Same-architecture pairing is critical. EnKF + causal init → R²=0.93 at realistic latency

### Experiment 16. Control-Relevant Metrics Evaluation
**Status**: ✅ COMPLETE — CRITICAL DEPLOYMENT INSIGHTS
**Results dir**: `results/ctrl_metrics_{causal_z64,causal_z128,acausal_z64}/`
**Branch/Worktree**: `feature/control-metrics` / `cleo-worktrees/control-metrics`
- **Free-run FIT% = 0** for ALL models (ODE diverges over 30s → re-encoding mandatory)
- Multi-horizon: linear error growth (good), R²>0.84 at H=20 for causal z64
- **Jacobian accuracy (∂g/∂u)**: cos_sim=0.997 — control matrix extremely well-learned
- Controllability Gramian: only 3–5 effective control dimensions out of 64/128
- **Key findings**:
  - MPC viable at H≤20 steps (20ms horizon)
  - Re-encoding every ~200ms is required for stability
  - Control input sensitivity is near-perfect despite modest R²
  - Low effective controllability rank suggests controller only needs ~5 latent dims

### Experiment 17. Full-Trial EnKF — 30s Deployment Stability
**Status**: ✅ COMPLETE — K=1 D=0 EXTRAORDINARY, REALISTIC CONFIGS DEGRADE
**Branch/Worktree**: `feature/enkf-fulltrial` / `cleo-worktrees/enkf-fulltrial`

| Config | FIT% | R² | Notes |
|--------|------|----|-------|
| K=1, D=0 | **90.5** | 0.993 | Beats acausal model — but unrealistic |
| K=1, D=10 | 51.6 | — | 10ms delay destroys long-horizon stability |
| K=20, D=0 | 50.8 | — | Sparse corrections insufficient over 30s |
| K=20, D=10 | 38.5 | — | Not viable for long-term tracking |

- **Key insight**: Small errors from delayed/sparse corrections compound over 30,000 steps
- Over SHORT horizons (200ms), K=20 D=10 is fine (R²=0.93 from Exp 15)
- **Implication**: K=1 D=0 is unrealistic for deployment. Need periodic re-encoding OR learned PredNet

---

## Phase 5: Latent Alignment & Control (Exp 18–22)

> Goal: Fix the topology mismatch between causal and acausal latent spaces,
> and validate the models for closed-loop control.

### Experiment 18. Same-Architecture Residual Correction v2
**Status**: ❌ COMPLETE — FAILED (R²=0.80 < baseline 0.82)
**Branch/Worktree**: `feature/residual-v2` / `cleo-worktrees/residual-v2`
- Fixes Exp 12 failure: uses CAUSAL ODE+decoder (same-arch, no cross-arch mismatch)
- ResidualMLP(z0_causal, z0_acausal_delayed) → delta_z0, trained end-to-end through frozen causal ODE
- Last layer initialized to zero → starts as identity → floor = causal baseline (R²=0.82)
- Acausal encoder at 100ms lag provides richer context as auxiliary MLP input
- Loss = MSE(decoder(ODE(z0_corrected)), x_true) through causal pipeline
- Expected R² ≥ 0.82 guaranteed, but got R²=0.80 — residual actively hurts

#### Why Residual Correction Fundamentally Fails: The Topology Problem

The causal and acausal models were trained independently. Even though both predict the
same firing rates x(t), their latent spaces z(t) can be **arbitrarily rotated, scaled, or
non-linearly warped** relative to each other. The residual δz₀ = z_acausal − z_causal is
therefore not a clean structured correction vector — it is an inconsistent mapping between
two unaligned coordinate systems. A small MLP cannot untangle a global diffeomorphism.

**Fix 1: Direct Trajectory Distillation** (= Exp 7 approach, revisited)
Freeze acausal encoder + ODE + decoder. Train a new causal encoder from scratch with
L = ||z_causal − z_acausal||². Forces the causal encoder to adopt the acausal topology.
Once aligned, residuals become meaningful because both models speak the same language.
- We tried this (Exp 7), R²=0.66 — frozen ODE was too rigid for imperfect causal z₀.
- **Variant**: Also fine-tune the decoder to tolerate noisier causal z₀.

**Fix 2: Vector Field Alignment** (Match the Dynamics)
Instead of matching z₀ points, match the ODE vector fields:
L = ||f_θ(z_causal) − f_θ(z_acausal)||². Guarantees that even if z₀ is slightly noisy,
the ODE pushes it in the same direction. Can be combined with trajectory distillation
as a regularizer.

**Fix 3: Contrastive Alignment** (InfoNCE / Mutual Information)
Project both z_causal and z_acausal into a shared low-dim subspace via linear heads.
Maximize cosine similarity for same-timestep pairs, push apart different-timestep pairs.
Maximizes mutual information without forcing rigid 1:1 mapping — gives the causal model
flexibility to handle missing future context.
- Most flexible approach; allows the causal space to be a *compressed* version of acausal.

**Recommended Path Forward** (validated in Exp 20):
Direct Trajectory Distillation with joint loss:
L = α·||z_causal − z_acausal||² + (1−α)·reconstruction_loss
Initialize causal ODE from acausal weights, then distill with α schedule
(start high to force alignment, decay to let the model specialize for causal inference).

### Experiment 19. Optoclamp: PI vs MPC Closed-Loop Control
**Status**: ✅ COMPLETE — MPC BEATS PI (14× LOWER SS ERROR)
**Branch/Worktree**: `feature/optoclamp` / `cleo-worktrees/optoclamp`
- Recapitulates Newman et al. 2015 optoclamp experiment in simulation
- PI controller baseline (sweep Kp, Ki)
- Neural ODE MPC using causal CA-NODE (z=64, H=20 horizon)
- Targets: rate clamping at 50%, 75%, 125% of baseline rate
- **Results**:
  - MPC achieves **14× lower steady-state error** than best PI at 125% target
  - MPC has lower tracking RMSE and faster settling across all targets
  - Demonstrates that differentiable ODE model enables superior control
- **Key finding**: The causal CA-NODE model, despite R²=0.82, enables excellent MPC performance
  because Jacobian accuracy (cos_sim=0.997) matters more than R² for control

### Experiment 20. Aligned Distillation (Warm-Start from Acausal)
**Status**: ✅ COMPLETE — ✅ BEST CAUSAL BASELINE
**Results dir**: `results/aligned_cosine_0.9_0.1/` (in aligned-distillation worktree)
**Branch/Worktree**: `feature/aligned-distillation`
- **R²=0.864** 🏆 — new causal SOTA (+4.7% over baseline 0.8252)
- Implements the recommended path from Exp 18 topology analysis
- Initialize causal ODE from acausal weights → latent spaces start aligned
- Joint loss with α schedule: start with strong distillation (α=0.9), decay to reconstruction (α=0.1)
- Successfully closes ~35% of the causal-acausal gap (0.8252 → 0.864 of 0.9387)
- **Key finding**: Pre-alignment of latent topology via weight initialization is crucial.
  Random init (Exp 8, R²=0.48) fails; warm start succeeds.

### Experiment 21. EnKF on Aligned Model
**Status**: ✅ COMPLETE
**Results dir**: `results/enkf_aligned_A1/` (D=0), `results/enkf_aligned_A2/` (D=10)
**Branch/Worktree**: `feature/aligned-distillation`
- Implements Ensemble Kalman Filtering on top of the best aligned distillation model (Exp 20)
- Swept observation rate K ∈ {1, 10, 20, 50, 100, 999} and noise parameters Q, R
- **Results**:
  - **D=0ms**: K=1 R²=0.9997 (Q=0.1, R=0.01) | K=20 R²=0.9607 | K=999 R²=0.9262
  - **D=10ms**: K=1 R²=0.9593 (Q=0.1, R=0.01) | K=20 R²=0.9391 | K=999 R²=0.9149
- **Key finding**: Under sparse updates/latency (K=999 D=10), the aligned model yields R²=0.915 vs 0.826 on the unaligned model (+0.089 improvement). However, EnKF compute time is ~1.2s per 200ms trial (sequential dopri5 integration), which is far too slow for real-time BCI control.

### Experiment 22. Periodic Re-Encoding vs EnKF
**Status**: ✅ COMPLETE — 🏆 FIRST REAL-TIME DEPLOYABLE CAUSAL MODEL APPROACHING ACAUSAL ACCURACY
**Results dir**: `results/periodic_reencode/`
**Branch/Worktree**: `feature/periodic-reencode` / `cleo-worktrees/periodic-reencode`
- Evaluates periodic latent state re-encoding (running the GRU encoder every K steps to reset the ODE initial state z₀) compared to EnKF.
- Swept K ∈ {1, 5, 10, 20, 50, 100, 200} × D ∈ {0, 10ms} using the aligned distillation model (R²=0.864).
- **Results**:
  - **D=0ms**: K=1 R²=0.8782 (550.3ms/trial) | K=5 R²=0.8770 (162.6ms/trial) | K=20 R²=0.8736 (87.7ms/trial)
  - **D=10ms**: K=1 R²=0.8730 (1050.7ms/trial) | K=5 R²=0.8722 (260.6ms/trial) | K=20 R²=0.8701 (112.4ms/trial)
- **Comparison to EnKF**:
  - EnKF K=20 D=10 R²=0.9391 vs. Re-Encode R²=0.8701 (EnKF is +0.0690 more accurate).
  - However, EnKF takes ~460ms/trial (~2.3ms/step average), whereas Re-Encode takes only 112.4ms/trial (**0.56ms/step average**).
  - 0.56ms/step is comfortably under the 1ms real-time budget, making periodic re-encoding the first nonlinear causal model that is fully deployable in real-time at 1kHz.
- **Key finding**: Continuous EnKF correction yields higher R², but periodic re-encoding with a GRU encoder is the only computationally viable approach for real-time deployment without GPU-level speed optimizations.

---

## Key Lessons Learned

1. **Latent space is everything**: Channel-level models plateau at R²≈0.90; latent compression breaks through to 0.94
2. **Causal-acausal gap is an encoder problem**: The ODE and decoder are fine; the causal encoder z₀ is the bottleneck
3. **Topology alignment matters**: Independent training creates incompatible latent spaces (Exp 18 analysis). Pre-alignment via weight init (Exp 20) or filtering (Exp 15) is required
4. **EnKF is powerful but slow**: EnKF reaches R²=0.9997 with dense updates, and R²=0.9391 at realistic latency (Exp 21), but ~1.2s inference makes it impractical for real-time
5. **Periodic re-encoding is the path forward**: Re-encoding every 5-20 steps (Exp 22) yields R²=0.870-0.877 with an average step compute time of 0.44ms - 0.81ms, making it fully deployable under a 1ms budget
6. **Jacobian accuracy > R² for control**: cos_sim=0.997 means MPC works even with baseline causal R²=0.82 (Exp 16, 19). The new aligned + re-encoded models (R²=0.877) should perform even better
7. **Same-architecture pairing is critical**: Cross-arch approaches (Exp 9, 12) consistently fail; same-arch (Exp 15, 18, 20, 21, 22) succeed

---

## Appendix: Model Architecture Reference

### Acausal Latent NODE (`modeling/models/latent_node.py`)
- Bidirectional GRU encoder → z₀ (sees full 200ms window)
- Neural ODE: dz/dt = f(z) + g(z)·u(t)
- MLP decoder: z(t) → x̂(t)
- Not deployable (requires future data)

### Causal Latent CA-NODE (`modeling/models/latent_canode.py`)
- Forward-only GRU encoder → z₀ (sees only past data)
- Control-affine Neural ODE: dz/dt = f(z) + g(z)·u(t)
- MLP decoder: z(t) → x̂(t)
- Deployable in real-time

### Training Scripts
- Acausal: `modeling/scripts/fit_latent_node.py`
- Causal: `modeling/scripts/fit_latent_canode.py`
- Distillation: `modeling/scripts/hybrid_distill.py`
- Evaluation: `modeling/scripts/eval_periodic_reencode.py`

### Experiment 23. Bidirectional Aligned Distillation
**Status**: Not started
**Branch/Worktree**: TBD
**Results dir**: TBD
- Hypothesis: Aligned distillation from acausal NODE (R²=0.935) to causal CA-NODE on the bidirectional ChrimsonR+GtACR2 plant will close the large causal-acausal gap (0.39 vs 0.11 on excitatory-only plant)
- Sweep: α ∈ {0.5, 0.7, 0.9}, pw ∈ {50, 100, 200}, lr ∈ {5e-4, 1e-3}
- Teacher: `results/sweep_latent_node_40_10/run_05_z64_h256_l2_lr0.0005_dopri5.pt` (n_u=2 verified)

### Experiment 24. Delay-Matched Training
**Status**: Not started
**Branch/Worktree**: TBD
**Results dir**: TBD
- Hypothesis: Training the causal model on data encoded with the same observation delay (D=10ms) used during deployment will improve generalization under latency, since the model learns to compensate for stale observations rather than seeing them for the first time at test time
- Compare: train with D=0 + test with D=10 vs. train with D=10 + test with D=10
- Expected: reduced latency-induced R² degradation

### Experiment 25. Bidirectional Periodic Re-Encoding
**Status**: Not started
**Branch/Worktree**: TBD
**Results dir**: TBD
- Hypothesis: Periodic re-encoding (K×D sweep) on the best bidirectional aligned model will replicate the gains seen on the excitatory plant (Exp 22) where re-encoding pushed R² from 0.864 to 0.877
- Sweep: K ∈ {1, 5, 10, 20, 50, 100} × D ∈ {0, 10ms}

### Experiment 26. Bidirectional EnKF Evaluation
**Status**: Not started
**Branch/Worktree**: TBD
**Results dir**: TBD
- Hypothesis: EnKF on the bidirectional aligned model will provide an accuracy upper bound (expected R²>0.90) for comparison with periodic re-encoding, replicating Exp 21 on the new plant
- Sweep: K ∈ {1, 20, 100, 999} × D ∈ {0, 10ms} × Q ∈ {0.1, 1.0} × R ∈ {0.01, 0.1}

### Experiment 27. Bidirectional Optoclamp MPC
**Status**: Not started
**Branch/Worktree**: TBD
**Results dir**: TBD
- Hypothesis: MPC with the bidirectional aligned model can achieve inhibition (hitting 50% and 75% target rates) which was impossible on the excitatory-only plant. This is the primary validation of the bidirectional plant.
- Targets: 50%, 75%, 100%, 125%
- Controllers: PI baseline vs NODE MPC with periodic re-encoding
