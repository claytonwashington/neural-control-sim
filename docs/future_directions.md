# Future Directions

Research directions beyond the current Phase 4 modeling sweep. These are ideas that either require significant new infrastructure, address higher-level questions, or propose novel architectures.

---

## 1. Control Layer Integration

**Priority**: HIGH — blocks the ultimate validation of all modeling work

> [!NOTE]
> All active control ideas, controller implementations, closed-loop experiments (including the optoclamp results), and future control plans have been moved to [CONTROL_IDEAS.md](CONTROL_IDEAS.md). Please refer to that document for the current state and roadmap of control-layer work.

---

## 2. Continuous PredNet — Hierarchical Neural ODE (Path A)

**Priority**: HIGH — novel architecture, principled approach to causal state estimation
**Full spec**: [architecture_specs.md](docs/architecture_specs.md)

### Concept
A stacked hierarchy of lightweight Neural ODEs communicating via **continuous prediction errors**.
Lower layers track fast sensory dynamics; upper layers track slow abstract intent.
The prediction error forcing term `g_φ(x(t) - D(z(t)))` is a **learned, nonlinear Kalman innovation**.

```
Layer 0 (sensory):     dz⁰/dt = f_θ⁰(z⁰) + g_φ⁰(x(t) - D(z⁰))
Layer 1 (abstraction): dz¹/dt = f_θ¹(z¹) + g_φ¹(z⁰ - proj(z¹))
```

### Why This Is Promising
- **Self-contained at deployment** — no teacher or acausal model needed
- The g_φ terms are learned versions of the EnKF's Kalman update
- Joint ODE integration means the hierarchy is differentiable end-to-end
- Natural multiscale decomposition (fast sensory + slow behavioral)
- Training uses acausal z_true for TOP layer only; intermediate layers self-organize

### Concerns
- Joint integration of [z⁰, z¹] creates a potentially stiff ODE system
- Training stability through coupled ODEs may be challenging
- Unclear if the hierarchy helps with the core z₀ estimation problem

---

## 2b. Continuous Delayed Distillation (Path B)

**Priority**: HIGH — fixes the exact failure mode of our residual correction experiment
**Full spec**: [architecture_specs.md](docs/architecture_specs.md)

### Concept
Run causal + acausal models in parallel during deployment. The acausal model operates
at a Δ ms lag. As it completes, its outputs become delayed ground-truth for online
gradient updates to a small residual MLP.

```
Real-time:  ẑ(t) = z_fast(t) + MLP_ψ(z_fast(t))
Delayed:    z_true(t) = Encoder_acausal(x[t-W:t+Δ])
Online:     L = || (z_true - z_fast) - MLP_ψ(z_fast) ||²
```

### Why This Fixes Our Previous Failure
- Our Exp 12 (residual correction) used **cross-architecture** evaluation
  (causal encoder z₀ → acausal ODE), causing R²=-0.51 baseline
- Path B keeps MLP_ψ operating on **same-architecture** z_fast
- Online learning adapts to biological drift (a real deployment concern)
- Only MLP_ψ trains (67K params) — no memory leak risk

### Key Advantage Over Path A
- **Adaptability**: Path A is frozen at deployment; Path B adapts in real-time
- Handles electrode drift, pharmacological changes, plasticity
- Much simpler to implement and debug

---

## 3. Optimal Experimental Design


**Priority**: MEDIUM — improves data efficiency

### Current Approach
OU noise stimulation — reasonable but not optimal.

### Proposed Approach
1. **Phase 1 (done)**: OU noise for initial data collection
2. **Phase 2 (next)**: Include step responses, chirps, PRBS, and no-stim periods
3. **Phase 3**: D-optimal input design using Fisher Information from current model

### Key Insight for Control-Affine Models
- f(x) is learned from spontaneous dynamics (no-stim periods)
- g(x) is learned from stimulation periods (active inputs)
- Our current data mixes both — explicit separation would help
- The input design primarily helps identify g(x), which is what MPC needs most

---

## 4. EnKF Speed Optimization

**Priority**: HIGH — EnKF achieves R2=0.999 but is 6x too slow

### The Bottleneck
EnKF runs 64 parallel ODE solves per step using dopri5 (adaptive solver). The solver overhead dominates — A100 gives zero speedup over 2080 Ti.

### Approaches to Try
1. **Fixed-step Euler in predict step**: Replace dopri5 with single Euler step for the 1ms EnKF predict. Accuracy may degrade but filter corrections should compensate.
2. **Reduced-order EnKF**: Use z_dim=16 instead of 64. Requires retraining a smaller latent model first.
3. **Amortized EnKF**: Train a neural network to approximate the EnKF update rule, eliminating the ODE solve entirely. Related to neural process / amortized inference literature.
4. **Continuous-time Kalman filter**: Instead of discrete EnKF steps, derive a continuous Kalman-Bucy filter for the learned ODE. Avoids repeated ODE solves.

---

## 5. Advanced Metrics and Evaluation

**Priority**: HIGH — needed before control layer

### Metrics to Implement
1. Free-run simulation FIT% (run model open-loop from initial state)
2. Multi-horizon error growth curves (1, 5, 10, 20, 50, 100, 200 steps)
3. Jacobian accuracy: cosine similarity of dg/du vs finite-difference ground truth
4. Empirical controllability Gramian from learned model
5. Sobolev/derivative-informed training loss

### Training Improvements
1. Multi-step shooting loss (not just single windows)
2. Jacobian matching regularization
3. Stability regularization (penalize positive Jacobian eigenvalues)

---

## 6. Drift Robustness and Online Adaptation

**Priority**: LOW (for now) — matters at deployment

### The Problem
Real neural systems drift over time (electrode movement, plasticity, pharmacology). Our models assume stationarity.

### Approaches
- Online fine-tuning of encoder/decoder (keep ODE frozen)
- Domain adaptation via batch normalization statistics
- Continual learning with elastic weight consolidation
- The residual correction MLP (Idea 12) was designed to be fine-tuned online — revisit this with same-architecture ODE

---

## Roadmap: Principled Order of Operations

### Phase 6: Control Layer (In Progress)
We have successfully implemented and executed the first closed-loop control experiments (PI vs. Neural ODE MPC optoclamp), and established control-relevant model metrics. 

Details of these experiments, key results, and next steps for the control layer are tracked in [CONTROL_IDEAS.md](CONTROL_IDEAS.md).

### Phase 7: Spiking Data on Current Plant
**Why second**: Real experiments produce spikes, not firing rates. We need a spike-to-rate
front-end (or direct spike modeling) before any of our models are deployable.

**Deliverables**:
1. Spike sorting / binning pipeline integrated with Cleo
2. Evaluate existing models on spike-derived firing rates
3. Quantify the rate estimation to model accuracy to control performance chain
4. Potentially train models directly on spike counts (Poisson observation model)

**Key questions answered**:
- How much does spike-to-rate estimation degrade model performance?
- Is there a spike observation model that preserves R2?
- Can we close the loop with noisy spiking observations?

**Prerequisites**: Phase 6 (need the control layer to measure impact)

### Phase 8: New Plants / Generalization
**Why last**: Different plants (larger networks, different cell types, different stimulation
modalities) introduce many new variables. We should nail the methodology on the current
plant first before scaling.

**Deliverables**:
1. Second plant with different network topology
2. Transfer learning: can an encoder/ODE pre-trained on Plant A fine-tune to Plant B?
3. Multi-plant meta-learning (if transfer works)
4. Scaling laws: how does model quality scale with network size, num electrodes, num opto sites?

**Key questions answered**:
- Does our approach generalize beyond the current 50-neuron simulation?
- What transfers and what needs re-training?
- Where do we hit computational limits?

**Prerequisites**: Phase 7 (want spiking pipeline validated first)
