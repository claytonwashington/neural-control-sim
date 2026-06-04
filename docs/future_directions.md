# Future Directions

Research directions beyond the current Phase 4 modeling sweep. These are ideas that either require significant new infrastructure, address higher-level questions, or propose novel architectures.

---

## 1. Control Layer Integration

**Priority**: HIGH — blocks the ultimate validation of all modeling work

### Motivation
We are optimizing R2 in a vacuum. LDS models with R2 ~ 0.3-0.6 already enable successful closed-loop optogenetic control (Bolus et al. 2021). Our causal model at R2=0.82 is far above this, but we have no idea whether neural ODE + MPC actually outperforms simple LDS + LQR.

### What We Need
1. **Better metrics**: Free-run simulation FIT%, multi-horizon error growth curves, Jacobian accuracy (dg/du)
2. **MPC controller**: Implement NeuralODEController(LatencyIOProcessor) in Cleo
3. **Baselines**: PI controller, LDS + LQR via ldsCtrlEst
4. **Key experiment**: Model accuracy vs control performance curve — sweep models of varying quality, run MPC with each, plot tracking RMSE vs model R2/FIT%

### Control Objectives (in order of difficulty)
- Rate clamping (hold firing rate at target) — start here
- Rate tracking (follow time-varying trajectory)
- Pattern generation (reproduce spatiotemporal patterns)
- Perturbation rejection (maintain target under disturbances)

### Relevant Literature
- Bolus, Willats, Rozell, Stanley (2021). State-space optimal feedback control of optogenetically driven neural activity. J. Neural Engineering.
- Newman et al. (eLife). Optoclamp for closed-loop optogenetic control.
- Johnsen et al. Cleo: Closed-Loop simulation testbed (our framework).
- ldsCtrlEst library (CLOCTools, Georgia Tech)

See also: [control_layer_analysis.md] in the conversation artifacts.

---

## 2. Residual Prediction Hierarchical Neural ODEs

**Priority**: MEDIUM — novel architecture idea

### Concept
Instead of a single monolithic ODE, use a hierarchy of ODEs that predict residuals at multiple timescales:

```
Level 0 (slow dynamics):   dz0/dt = f0(z0) + g0(z0)u        (timescale ~100ms)
Level 1 (fast residual):   dz1/dt = f1(z0, z1) + g1(z0, z1)u (timescale ~10ms)
Level 2 (ultra-fast):      dz2/dt = f2(z0, z1, z2) + g2(...)u (timescale ~1ms)

Prediction: x_hat = decoder(z0 + z1 + z2)
```

### Why This Might Work
- Neural circuits operate at multiple timescales (slow NMDA, fast AMPA, ultra-fast GABAa)
- Residual connections prevent gradient issues in deep hierarchies
- Each level can use a different ODE solver tolerance (slow = coarse, fast = fine)
- The hierarchy naturally separates control-relevant slow dynamics from noise-like fast dynamics
- Could enable faster inference: skip Level 2 during control (only need slow dynamics for MPC)

### Relationship to Existing Work
- Multi-rate integration (Idea 4) tried sub-stepping but within a single ODE — this is fundamentally different
- Neural ODE with skip connections (Idea 2) added residual paths but not hierarchical dynamics
- This is closer to the clock-hierarchical RNN literature (Chung et al. 2016, Hierarchical Multiscale RNNs)
- Also relates to "slow feature analysis" and the idea that control-relevant features change slowly

### Implementation Sketch
- 3-level hierarchy with z_dim = [32, 16, 16] (total = 64, matching current best)
- Level 0 uses Euler integration with dt=10ms (fast)
- Level 1 uses Euler with dt=1ms
- Level 2 uses dopri5 with dt=1ms (only when accuracy matters)
- Training: each level has its own loss term targeting residuals at its timescale
- For MPC: only integrate Level 0 + 1 (skip Level 2) — much faster inference

### Open Questions
- How to define the "target residual" for each level during training?
- Should levels share parameters or be completely independent?
- Does the hierarchy help with causal state estimation (the main bottleneck)?
- What is the right decomposition of timescales for our specific neural circuit?

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
