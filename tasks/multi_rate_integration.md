# Multi-Scale / Multi-Rate Integration for CA-NODE

Parent: [task.md](file:///snel/home/cbwash2/cleo/task.md)
Branch: `feature/multi-rate-integration`
Agent: Antigravity

## Goal

Standard Neural ODEs smooth out high-frequency neural transients because the adaptive solver adjusts step sizes based on the global error, which is dominated by slow, large-scale dynamics. Decoupling the system into slow baseline dynamics and fast optogenetic responses, and integrating them at different step sizes (multi-rate integration), will capture the fast stimulus-driven components without smoothing or instability.

## Formulation

We decompose the predicted state $\hat{x}(t) \in \mathbb{R}^{n_x}$ (the 50 channel MUA firing rates) into a slow component $x_{\text{slow}}(t)$ and a fast component $x_{\text{fast}}(t)$:

$$\hat{x}(t) = x_{\text{slow}}(t) + x_{\text{fast}}(t)$$

### 1. Slow Dynamics (Macro-rate)
The slow component $x_{\text{slow}}(t)$ is governed by a control-affine ODE integrated using the standard macro step size $\Delta t$ (or evaluated at $\Delta t$ using adaptive solvers like `dopri5`):

$$\frac{d x_{\text{slow}}}{dt} = f_{\text{slow}}(x_{\text{slow}}) + g_{\text{slow}}(x_{\text{slow}}) u(t)$$

where:
- $f_{\text{slow}}$ is a Drift MLP mapping $\mathbb{R}^{n_x} \to \mathbb{R}^{n_x}$
- $g_{\text{slow}}$ is a Control MLP mapping $\mathbb{R}^{n_x} \to \mathbb{R}^{n_x \times n_u}$
- Initial condition: $x_{\text{slow}}(0) = x_0$ (the initial true state)

### 2. Fast Dynamics (Micro-rate via Sub-stepping)
The fast component $x_{\text{fast}}(t)$ represents direct, fast optogenetic responses. To capture sharp stimulus-driven transients, it is integrated at a much finer step size $\delta t = \Delta t / M$ (where $M$ is the sub-stepping factor, e.g., $M = 10$).

The fast dynamics are coupled to both the slow state and the fast state:

$$\frac{d x_{\text{fast}}}{dt} = f_{\text{fast}}(x_{\text{slow}}, x_{\text{fast}}) + g_{\text{fast}}(x_{\text{slow}}, x_{\text{fast}}) u(t)$$

where:
- $f_{\text{fast}}$ is a Drift MLP mapping $\mathbb{R}^{2n_x} \to \mathbb{R}^{n_x}$
- $g_{\text{fast}}$ is a Control MLP mapping $\mathbb{R}^{2n_x} \to \mathbb{R}^{n_x \times n_u}$
- Initial condition: $x_{\text{fast}}(0) = 0$ (modeled as stimulus-driven deviation starting from zero)

### Sub-stepping Algorithm
For each macro time interval $[t_k, t_{k+1}]$ (where $t_{k+1} - t_k = \Delta t$):
1. Linearly interpolate the slow state $x_{\text{slow}}(t)$ and the control input $u(t)$ for sub-step $j \in \{0, 1, \dots, M-1\}$:
   $$x_{\text{slow}}^{(j)} = \left(1 - \frac{j}{M}\right) x_{\text{slow}}(t_k) + \frac{j}{M} x_{\text{slow}}(t_{k+1})$$
   $$u^{(j)} = \left(1 - \frac{j}{M}\right) u(t_k) + \frac{j}{M} u(t_{k+1})$$
2. Update the fast state using Euler integration with step size $\delta t$:
   $$x_{\text{fast}}^{(j+1)} = x_{\text{fast}}^{(j)} + \delta t \cdot \left[ f_{\text{fast}}\left(x_{\text{slow}}^{(j)}, x_{\text{fast}}^{(j)}\right) + g_{\text{fast}}\left(x_{\text{slow}}^{(j)}, x_{\text{fast}}^{(j)}\right) u^{(j)} \right]$$
3. Assign the fast state at the next macro step: $x_{\text{fast}}(t_{k+1}) = x_{\text{fast}}^{(M)}$

## Checklist

### 1. Implementation
- [ ] Create `MultiRateControlAffineODE` in `modeling/models/canode.py` inheriting from `nn.Module`.
- [ ] Implement the slow dynamics networks and the fast dynamics networks.
- [ ] Implement the sub-stepped integration loop inside the `predict()` method.
- [ ] Support custom sub-stepping factor $M$ as an initialization argument.
- [ ] Update `fit_canode.py` to support `--model-type multi-rate` and `--sub-steps M` arguments.
- [ ] Update `sweep_canode.py` to support sweeping over model types and sub-steps.

### 2. Verification & Testing
- [ ] Test the forward pass and gradient backpropagation of `MultiRateControlAffineODE` on dummy inputs.
- [ ] Run a test training for 5 epochs to verify training stability and ensure no memory leaks/slowdowns.

### 3. Hyperparameter Sweep
- [ ] Run a grid sweep over sub-stepping factor $M \in \{1, 5, 10, 20\}$ to find the optimal trade-off between accuracy and training speed.
- [ ] Run a comparative sweep of `MultiRateControlAffineODE` vs baseline `ControlAffineODE` with matching parameters on the 50-trial dataset.

### 4. Analysis & Dashboard
- [ ] Evaluate the best multi-rate model on the held-out test trials.
- [ ] Generate prediction plot and compare $R^2$ performance on 200ms windows against baseline.
- [ ] Add the multi-rate results and training curves to the results dashboard at `results/dashboard.html`.
