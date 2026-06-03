# Digital Twin Modeling Ideas

This document tracks modeling hypotheses and architectures to improve prediction accuracy ($R^2$ on 200ms prediction windows) for the Cleo digital twin spiking neural network model. Currently, the best baseline control-affine Neural ODE (CA-NODE) model plateaus around $R^2 \approx 0.90$.

---

## 1. Discrete-Time Sequence Models (GRU/LSTM) ✅ TESTED — ❌ Did not beat baseline
**Hypothesis**: The continuous-time integration in the Neural ODE ($\int (f(x) + g(x)u)dt$) acts as a low-pass filter, smoothing out high-frequency neural firing transients. Replacing the ODE solver with a discrete-time RNN (e.g., GRU or LSTM) operating at the 1ms sampling rate can capture sharp transitions and fast stim-onsets without integration smoothing.
*   **Architecture**:
    $$h_t = \text{GRU}(h_{t-1}, [x_{t-1}, u_t])$$
    $$\hat{x}_t = \text{MLP}(h_t)$$
*   **Results**: Tested on `gpu1` (`h128`, 500 epochs, batch size 512, 10 test trials).
    *   Train loss: **`0.0552`** (extremely low)
    *   Val loss: **`0.0587`** (extremely low)
    *   Avg Test $R^2$ (200ms window): **`0.5152`** ❌ (severe overfitting)
    *   Open-loop $R^2$ (2s horizon): **Negative** across all trials (severe drift)
*   **Analysis**: While the GRU fits the training data windows perfectly, it lacks the strong regularizing inductive bias of the continuous-time Neural ODE. Because it has no continuity constraint or state-dependent derivative structure, it uses its recurrent capacity to memorize trial-specific noise and stimulus shapes. 
*   **Proposed Alternative (Euler-Discretized Control-Affine Model - ED-CA-NODE)**:
    Instead of a black-box GRU, we can discretize the control-affine ODE using a single Euler step per 1ms interval:
    $$\hat{x}_t = \hat{x}_{t-1} + f_\theta(\hat{x}_{t-1}) + g_\phi(\hat{x}_{t-1}) u_t$$
    This model preserves the strong control-affine physical bias (which prevents overfitting) but avoids the adaptive multi-step numerical smoothing of `dopri5`. It trains extremely fast and can capture step-by-step high-frequency transitions directly.


## 2. Regularized & Simpler Skip Connections ✅ TESTED — ❌ Did not beat baseline (R²=0.9144 vs 0.9219 baseline)
**Hypothesis**: High-frequency dynamics can bypass the ODE integrator via skip paths. In Phase 2.5, a 2-layer MLP input-skip connection ($u(t) \to x(t)$) overfit to trial-specific stimulus patterns. Restricting the skip paths to be **purely linear** ($u(t) \to x(t)$ and $x_{\text{ode}}(t) \to x(t)$) or utilizing L2 regularization (weight decay) on the skip parameters will prevent overfitting.
*   **Formulation**:
    $$\hat{x}(t) = x_{\text{ode}}(t) + W_x x_{\text{ode}}(t) + W_u u(t)$$
    with zero initialization for $W_x$ and $W_u$.

## 3. Frequency-Aware Loss (Spectral Loss) 🔄 IN PROGRESS
**Hypothesis**: Standard MSE loss in the time domain is dominated by large-scale low-frequency signals. Adding an FFT-magnitude penalty forces the optimizer to align predictions with the high-frequency spectral components of the true multi-unit activity.
*   **Loss Formulation**:
    $$\mathcal{L} = \text{MSE}(x_{\text{pred}}, x_{\text{true}}) + \alpha \cdot \text{MSE}(|\text{FFT}(x_{\text{pred}})|, |\text{FFT}(x_{\text{true}})|)$$

## 4. Latent Neural ODE (LFADS-Style) ✅ TESTED — ❌ Did not beat baseline
**Hypothesis**: Fitting a high-dimensional system (50 channels) directly inside the Neural ODE integrator forces the model to fit high-frequency channel-level noise. Projecting the 50 channels to a lower-dimensional latent space ($z\_dim \in \{16, 32, 64\}$), running the Neural ODE in latent space, and decoding back will denoise the firing rates and capture the underlying shared dynamical manifold.
*   **Flow**:
    $$x_0 \xrightarrow{\text{Encoder}} z_0 \xrightarrow{\text{ODE}(u(t))} z(t) \xrightarrow{\text{Decoder}} \hat{x}(t)$$
*   **Results**:
    *   **Latent Neural ODE (Standard)**: Swept latent space size $z\_dim \in \{16, 32, 64\}$, hidden MLP size $h \in \{128, 256\}$, and solver methods (`dopri5`, `rk4`).
        *   Best configuration: $z\_dim=32, h=256$, AdamW ($lr=1\text{e-}3$, `dopri5`), which achieved an average 200ms prediction $R^2 \approx 0.5515$ (val loss $\approx 0.0617$, train loss $\approx 0.0763$).
    *   **Latent CA-NODE** (combining control-affine dynamics and latent space representation): Swept past window sizes, learning rates, and MLP structures.
        *   Best configuration: $z\_dim=32, h=256, past\_window=200, lr=5\text{e-}4$, which achieved an average 200ms prediction $R^2 \approx 0.4405$ (val loss $\approx 0.1009$, train loss $\approx 0.1013$).
*   **Analysis**: While the latent space representation does succeed in denoising observations, both architectures significantly underperform compared to the baseline CA-NODE which operates directly on the 50 channels ($R^2 \approx 0.90$). This indicates that (a) compressing the 50 channels into a lower-dimensional latent space loses critical predictive information, or (b) the mapping between latent dynamics and observations is non-linear and cannot be modeled by a simple linear decoder, or (c) the encoder training is bottlenecked by the causal setup.

## 5. Multi-Scale / Multi-Rate Integration 🔄 IN PROGRESS
**Hypothesis**: The system contains slow baseline drift and fast optogenetic responses. Splitting the state $x$ into slow variables (integrated via standard ODE) and fast variables (integrated at a much finer step size $\delta t = \Delta t / M$) will prevent the solver from smoothing out fast stimulus-driven components.
*   **Formulation**:
    $$\hat{x}(t) = x_{\text{slow}}(t) + x_{\text{fast}}(t)$$
    where:
    *   Slow variables: $\frac{d x_{\text{slow}}}{dt} = f_{\text{slow}}(x_{\text{slow}}) + g_{\text{slow}}(x_{\text{slow}}) u(t)$ (solved at macro step size $\Delta t$).
    *   Fast variables: $\frac{d x_{\text{fast}}}{dt} = f_{\text{fast}}(x_{\text{slow}}, x_{\text{fast}}) + g_{\text{fast}}(x_{\text{slow}}, x_{\text{fast}}) u(t)$ (solved at micro step size $\delta t = \Delta t / M$ via Euler sub-stepping).
*   **Initialization Options**:
    1.  **Zero Fast Initialization**: 
        $$x_{\text{slow}}(0) = x_0$$
        $$x_{\text{fast}}(0) = 0$$
        This assumes that the fast optogenetic transient component starts at zero deviation from the baseline.
    2.  **Learnable Projection Initialization**:
        $$x_{\text{slow}}(0) = W_{\text{slow}} x_0 + b_{\text{slow}}$$
        $$x_{\text{fast}}(0) = W_{\text{fast}} x_0 + b_{\text{fast}}$$
        where $W_{\text{slow}}, W_{\text{fast}} \in \mathbb{R}^{n_x \times n_x}$ and $b_{\text{slow}}, b_{\text{fast}} \in \mathbb{R}^{n_x}$ are learnable parameters. They are initialized close to identity and zero respectively, allowing the network to learn how to optimaly split the initial observed state between slow and fast dynamics.

## 6. Extended Training Sweep with LR Warmup ✅ TESTED — R²=0.9219 (noskip baseline, unfair 48/2 split)
**Hypothesis**: Slower learning rates (like $\eta = 1\text{e-}4$, the best from Phase 2) prevent gradient explosion during integration but require more epochs to converge. Training for 500–1000 epochs (instead of 200) with a cosine decay and initial linear warmup will ensure the model converges to the true global optimum.
