# Architecture Specifications: Closing the Causal-Acausal Gap

Two deployment architectures for real-time neural state estimation. Both aim to close
the causal-acausal gap without scaling up the causal encoder.

## High-Level Comparison

| Feature | Path A: Continuous PredNet | Path B: Delayed Residual Distillation |
|---------|---------------------------|--------------------------------------|
| Core Concept | Hierarchical Neural ODEs with continuous prediction errors | Fast real-time predictor + slow parallel teacher |
| Compute Profile | Self-contained, forward-only at deployment | Two models in parallel (1 fast, 1 slow/delayed) |
| "Truth" Signal | Self-generated via bottom-up prediction errors | Provided continuously by delayed acausal model |
| Adaptability | Fixed at deployment (weights frozen) | Online learning (adapts to biological drift) |

---

## Path A: Continuous PredNet (Self-Contained Hierarchy)

**Goal**: Build a 2D grid of continuous dynamics (Time × Abstraction). Instead of a single
giant causal encoder, use a stacked hierarchy of lightweight Neural ODEs. Lower layers
track fast, noisy sensory dynamics; upper layers track slow, abstract behavioral intent.
Layers communicate only via continuous prediction errors.

### Mathematical Formulation

**Layer 0 (Sensory Level)**: Matches the raw neural data x(t).

```
dz⁰/dt = f_θ⁰(z⁰(t)) + g_φ⁰(x(t) - D(z⁰(t)))
```

Where D is the emission decoder.

**Layer 1 (Abstraction Level)**: Predicts the state of Layer 0.

```
dz¹/dt = f_θ¹(z¹(t)) + g_φ¹(z⁰(t) - proj(z¹(t)))
```

Where `proj` maps the higher-dimensional abstraction down to the Layer 0 space.

### Key Properties

- The `g_φ` terms are **continuous prediction error forcing** — analogous to the Kalman
  innovation term `K·(x_obs - x_pred)` but learned and nonlinear.
- Layer 0 is driven by **sensory prediction errors** (data vs model)
- Layer 1 is driven by **representational prediction errors** (fast state vs slow prediction)
- At deployment, no teacher is needed — bottom-up residuals self-stabilize

### Implementation Requirements

**Module Structure**: `HierarchicalNODE` class containing `ModuleList` of `ODELayer` sub-modules.
Each layer has its own drift `f_θ` and residual-forcing `g_φ` networks.

**Inference Loop**: Joint ODE integration with concatenated state `[z⁰, z¹]` passed to `odeint`.

**Training**:
- Use pre-trained acausal model to generate `z_true` targets for TOP layer only
- Loss = `||z¹_pred - z_true||²` (top-layer MSE)
- Intermediate layers self-organize via end-to-end backprop
- At deployment, acausal teacher is discarded

---

## Path B: Continuous Delayed Distillation (Online Learning)

**Goal**: Run a lightweight causal model for real-time control, while a heavy acausal model
runs in parallel with Δ ms lag. The slow model's outputs are used as delayed ground-truth
to take online gradient steps on a residual correction network.

### Mathematical Formulation

**Time t (Real-Time Inference)**: Fast causal encoder + residual MLP.

```
z_fast(t) = Encoder_causal(x[t-W:t])
ẑ(t)      = z_fast(t) + MLP_ψ(z_fast(t))
```

**Time t + Δ (Delayed Truth)**: Acausal model finishes processing.

```
z_true(t) = Encoder_acausal(x[t-W:t+Δ])
```

**Online Update**: Calculate true residual from the past, update MLP_ψ.

```
Δz(t)    = z_true(t) - z_fast(t)
L_online = || Δz(t) - MLP_ψ(z_fast(t)) ||²
```

### Key Properties

- Causal encoder and acausal encoder are **FROZEN** — only MLP_ψ trains online
- MLP_ψ operates on z_fast from the **SAME architecture** (no cross-arch mismatch)
- Online learning handles biological drift naturally
- Gradient isolation: only MLP_ψ's computational graph is retained

### Implementation Requirements

**Module Structure**:
- Pre-trained CausalEncoder (frozen)
- Pre-trained AcausalEncoder (frozen)
- Lightweight ResidualMLP (2 layers, 128 hidden, trainable online)

**Deployment Loop (Two Threads)**:
- Thread 1: Process data up to t, run CausalEncoder + ResidualMLP, output prediction
- Thread 2: Process data up to t+Δ, run AcausalEncoder, compute L_online,
  backward() and step() on ResidualMLP optimizer only

**Memory Management**: Gradients isolated to ResidualMLP to prevent memory leaks.
