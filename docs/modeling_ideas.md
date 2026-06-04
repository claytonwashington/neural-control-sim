# Digital Twin Modeling Ideas

This document tracks modeling hypotheses and architectures to improve prediction accuracy ($R^2$ on 200ms prediction windows) for the Cleo digital twin spiking neural network model. All results below use the **standardized 40/10 train/test split** on `data/training_trials.h5` with `seed=42`.

**Current best: Latent NODE z=64, h=256, lr=5e-4 → R²=0.9387**
**Channel-level baseline: CA-NODE h=128, l=2, lr=1e-4 → R²=0.9009**

---

## 1. Discrete-Time Sequence Models (GRU/LSTM) ✅ TESTED — ❌ Did not beat baseline
**Hypothesis**: The continuous-time integration in the Neural ODE ($\int (f(x) + g(x)u)dt$) acts as a low-pass filter, smoothing out high-frequency neural firing transients. Replacing the ODE solver with a discrete-time RNN (e.g., GRU or LSTM) operating at the 1ms sampling rate can capture sharp transitions and fast stim-onsets without integration smoothing.
*   **Results (40/10 split)**:
    *   Best GRU: h128 l1 lr=1e-4 → **R²=0.5798** ❌
    *   Best Discrete CA-NODE (Euler): h128 l2 lr=1e-4 → **R²=0.8744** (good, but trails continuous ODE)
*   **Analysis**: The control-affine inductive bias is essential — Discrete CA-NODE dramatically outperforms GRU (+29.5% R²) despite identical parameter counts. Continuous-time integration adds +2.6% over single Euler steps.

## 2. Regularized & Simpler Skip Connections ✅ TESTED — ❌ Did not beat baseline
**Hypothesis**: High-frequency dynamics can bypass the ODE integrator via skip paths.
*   **Results (40/10 split)**:
    *   No-skip baseline: **R²=0.9009** ✅ Best channel-level model
    *   Best skip variant (linear, swd=1.0): **R²=0.8911** ❌ (-1.1%)
    *   All skip paths consistently degrade test R² due to overfitting

## 3. Frequency-Aware Loss (Spectral Loss) ✅ TESTED — ❌ Did not beat baseline
**Hypothesis**: Standard MSE loss in the time domain is dominated by large-scale low-frequency signals. Adding an FFT-magnitude penalty forces the optimizer to align predictions with the high-frequency spectral components.
*   **Results (40/10 split)**:
    *   Baseline (α=0.0): **R²=0.9009**
    *   Best spectral (α=2.0): **R²=0.8993** ❌
    *   All spectral α values (0.01 to 2.0) cluster at R²≈0.899, providing zero improvement.
*   **Analysis**: Time-domain MSE already captures frequency content implicitly. Idea conclusively ruled out.

## 4. Latent Neural ODE (LFADS-Style) ✅ TESTED — ✅ NEW BEST MODEL
**Hypothesis**: Fitting a high-dimensional system (50 channels) directly inside the Neural ODE integrator forces the model to fit high-frequency channel-level noise. Projecting the 50 channels to a lower-dimensional latent space, running the Neural ODE in latent space, and decoding back will denoise the firing rates and capture the underlying shared dynamical manifold.
*   **Flow**:
    $$x_0 \xrightarrow{\text{Encoder}} z_0 \xrightarrow{\text{ODE}(u(t))} z(t) \xrightarrow{\text{Decoder}} \hat{x}(t)$$
*   **Results (40/10 split)**:
    *   **Non-Causal Latent NODE** (encoder sees full trial):
        *   z=64, h=256, lr=5e-4, dopri5 → **R²=0.9387** 🏆 **ALL-TIME BEST**
        *   z=64, h=128, lr=1e-3, dopri5 → **R²=0.9340** 🥈
        *   z=32 → R²≈0.44–0.63 (too compressed)
        *   z=16 → R²=0.37 (far too compressed)
    *   **Causal Latent CA-NODE** (encoder sees only past window):
        *   Best: pw=100ms, h=128, lr=5e-4 → **R²=0.5252** ❌
        *   Causality constraint devastates accuracy (0.53 vs 0.94)
*   **Key Insight**: z=64 is the critical latent dimension — it preserves the dynamical manifold while filtering channel-level noise. The 50-channel observations lie on a ~64-dimensional manifold.

## 5. Multi-Scale / Multi-Rate Integration 🔄 RE-RUNNING (40/10 split)
**Hypothesis**: The system contains slow baseline drift and fast optogenetic responses. Splitting the state into slow and fast variables integrated at different step sizes will prevent the solver from smoothing out fast stimulus-driven components.
*   **Status**: 6 configs currently training on gpu2 (A40s) using standardized 40/10 split. Previous results were on a non-standardized split.

## 6. Extended Training Sweep with LR Warmup 🔄 IN PROGRESS
**Hypothesis**: Slower learning rates with longer training (500 epochs) + warmup may improve convergence.
*   **Status**: 12 configs running on gpu2. Awaiting completion for 40/10 standardized comparison.

---

## Phase 4: Closing the Causal-Acausal Gap

### 7. Encoder Distillation (Acausal Teacher → Causal Student)
**Status**: 📋 Not started  
**Hypothesis**: Freeze the acausal model's ODE + decoder (proven R²=0.94). Train only a causal GRU encoder to match the acausal z₀ targets. Separates dynamics learning (solved) from causal state inference (unsolved). The causal encoder gets denoised latent targets instead of learning end-to-end.

### 8. Hybrid Distillation
**Status**: 📋 Not started  
**Hypothesis**: Same as encoder distillation but allow ODE+decoder to fine-tune with combined loss: α·MSE(z₀_student, z₀_teacher) + (1-α)·MSE(x̂, x_true). Lets the system adapt dynamics to work better with the noisier causal z₀ estimates. Sweep α ∈ {0.1, 0.3, 0.5, 0.7, 0.9}.

### 9. Causal Grokking (500 Trials, 1000 Epochs)
**Status**: 📋 Not started (data gen in progress)  
**Hypothesis**: With 10× more training data + 5× more epochs + higher weight decay (1e-4), the causal encoder may "grok" and discover an efficient algorithm for z₀ inference from past context. Watch for the characteristic test-loss plateau → sudden drop.

### 10. Acausal Grokking (Better Teacher)
**Status**: 📋 Not started (data gen in progress)  
**Hypothesis**: Train a larger acausal Latent NODE (z=128, h=512) on 500 trials for 1000 epochs. Could push acausal R² from 0.94 → 0.97+, producing a better teacher for distillation.

### 11. Kalman Filter State Estimation (EnKF/UKF)
**Status**: 📋 Not started  
**Hypothesis**: Instead of training a neural encoder, use a principled Ensemble Kalman Filter with the learned ODE as the dynamics model and the decoder as the observation model. No additional training needed — just plug in the learned ODE and run the filter. Maintains state covariance for uncertainty estimation.

### 12. Delayed Residual Correction (Complementary Filter)
**Status**: 📋 Not started  
**Hypothesis**: Run both causal and acausal models during deployment. The acausal model operates at a 100ms lag (centered ±100ms window). A small residual MLP learns to predict Δz₀ = z₀_acausal - z₀_causal and adds this correction to the real-time causal estimate. Analogous to a Smith Predictor in control theory. The residual is structured (not noise) and learnable. Can be fine-tuned online during deployment.
