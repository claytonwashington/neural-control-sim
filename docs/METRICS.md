# Evaluation Metrics for Neural State Estimation & Control

This document explains every metric we use (or plan to use) for evaluating our digital twin models.
The metrics are ordered from "standard ML" to "control-relevant."

---

## 1. Windowed R² (Current Primary Metric)

**What it measures**: Variance explained by the model over a 200ms prediction window.

**How it works**:
1. Encode z₀ from a context window (past 200ms for causal, full window for acausal)
2. Integrate the ODE forward 200 steps (200ms at dt=1ms)
3. Decode z(t) → x̂(t) at each step
4. Compare x̂ vs x_true across all 50 channels and 200 timesteps

```
R² = 1 - Σ(x_true - x̂)² / Σ(x_true - mean(x_true))²
```

**Range**: (-∞, 1]. R²=1 is perfect. R²=0 means the model is no better than predicting the mean. R²<0 means worse than the mean.

**Strengths**: Easy to interpret, standard, allows comparison across models.

**Weaknesses for control**:
- Averages over all 200 steps equally — but control only needs the first ~20 steps (MPC horizon)
- Averages over all 50 channels equally — but some channels may be more controllable than others
- Doesn't tell you about long-horizon stability or Jacobian accuracy
- A model can have high R² but terrible control performance if it gets the *dynamics* right but the *input response* wrong

**Current leaderboard**: Acausal R²=0.94, Causal R²=0.84, Channel-level R²=0.90

---

## 2. MSE (Mean Squared Error)

**What it measures**: Average squared prediction error in normalized firing rate units.

```
MSE = (1/NT) Σᵢ Σₜ (x̂ᵢ(t) - xᵢ(t))²
```

**Relationship to R²**: R² = 1 - MSE/Var(x). They contain the same information for a given dataset, but MSE is more directly interpretable as "prediction error magnitude."

**Why we track both**: R² normalizes by signal variance (so R²=0.90 means the same thing regardless of the signal's scale). MSE doesn't normalize, so it's useful for comparing error magnitudes across different experimental conditions.

---

## 3. Free-Run Simulation FIT%

**What it measures**: How well the model tracks the true trajectory when run open-loop for the *entire trial duration* (30 seconds), not just 200ms windows.

**How it works**:
1. Encode z₀ from the very first window of the trial
2. Integrate the ODE forward for the **entire remaining trial** (up to 30,000 steps)
3. No re-encoding, no observation feedback — pure open-loop prediction
4. Compute FIT%

```
FIT% = max(0, 100 × (1 - ||x_true - x̂||₂ / ||x_true - mean(x_true)||₂))
```

Note: FIT% uses the L2 *norm* (not squared), unlike R² which uses sum of squares. This makes FIT% more sensitive to large errors.

**Range**: [0, 100]. FIT%=100 is perfect. FIT%=0 means the model is no better than the mean.

**Why it matters for control**:
- In deployment, you can't re-encode every 200ms — the model must maintain its state estimate continuously
- Free-run FIT% reveals whether the ODE is **stable** over long horizons
- A model with R²=0.90 on 200ms windows might have FIT%=0 over 30s if the ODE drifts
- Standard metric in system identification (MATLAB's `compare` function uses this)

**Expected behavior**: FIT% will be much lower than R² because errors compound over 30,000 steps. Even the acausal model (R²=0.94) probably has FIT% < 50% because small ODE drift accumulates.

---

## 4. Multi-Horizon Error Growth Curves

**What it measures**: How fast prediction error grows as you predict further into the future.

**How it works**:
For each test window:
1. Encode z₀
2. Predict forward H steps for H ∈ {1, 2, 5, 10, 20, 50, 100, 200}
3. Compute R² or MSE using only the prediction at step H (not the full trajectory)

Produces a curve: R²(H) or MSE(H) vs prediction horizon H.

**Why it matters for control**:
- **MPC only needs short-horizon accuracy.** If the model is accurate for 20ms but degrades at 200ms, it's still perfectly usable for MPC with a 20-step horizon.
- The *shape* of the error growth curve tells you about the ODE's stability:
  - Linear growth → bounded drift (good)
  - Exponential growth → unstable dynamics (bad)
  - Plateau → the model converges to a fixed point regardless of input (also bad)
- Comparing error growth between causal and acausal models reveals *where* the gap appears — is it immediate (encoder quality) or delayed (ODE quality)?

**Expected insights**:
- At H=1, causal and acausal should be similar (one ODE step from z₀)
- At H=200, the gap reflects accumulated z₀ error through ODE dynamics
- If causal catches up at H=20, MPC will work fine despite lower R²

---

## 5. Jacobian Accuracy: ∂g/∂u (Control Input Sensitivity)

**What it measures**: How accurately the model predicts the *effect of stimulation* on neural activity.

**Background**: Our models are control-affine:
```
dx/dt = f(x) + g(x) · u(t)
         ↑         ↑
     drift    control matrix
```

The control matrix g(x) maps stimulation inputs u(t) to their effect on the state. For MPC to work, g(x) must be accurate — it tells the controller "if I increase stimulation by Δu, the state will change by g(x)·Δu."

**How to evaluate**:
1. From training data, find triplets (x_t, u_t, x_{t+1})
2. **Learned**: g_learned = model's control network evaluated at x_t
3. **Empirical**: Use finite differences from data pairs with different u values at similar x:
   ```
   g_empirical ≈ (x_{t+1}(u+δ) - x_{t+1}(u-δ)) / (2δ)
   ```
4. Compare via cosine similarity: cos(g_learned, g_empirical)

**Why it matters for control**:
- A model can perfectly predict spontaneous dynamics (high R²) but completely misrepresent stimulation effects (wrong g(x)), making it useless for control
- This is the metric most directly tied to MPC performance
- If g(x) accuracy is high, even a model with modest R² could enable good control

**Caveat**: Estimating g_empirical from data requires sufficient input variability. Our OU noise stimulation provides this, but the estimate will be noisy.

---

## 6. Empirical Controllability Gramian

**What it measures**: Which *directions* in latent space can be influenced by the stimulation input.

**Background**: In linear control theory, the controllability Gramian is:
```
W_c = Σₜ (Aᵗ B)(Aᵗ B)ᵀ
```
where A = ∂f/∂x (state transition Jacobian) and B = g(x) (control matrix).

For our nonlinear models, we compute an *empirical* version by evaluating A and B along actual trajectories:
```
A(t) = ∂f/∂x |_{x=x(t)}    (Jacobian of drift at current state)
B(t) = g(x(t))               (control matrix at current state)
W_c ≈ (1/T) Σₜ B(t) B(t)ᵀ + A(t)B(t)(A(t)B(t))ᵀ + ...
```

**What the eigenvalues tell us**:
- **Large eigenvalues** → latent directions that are highly controllable (small inputs produce large state changes)
- **Small/zero eigenvalues** → uncontrollable directions (stimulation can't influence these)
- **Condition number** (ratio of largest to smallest eigenvalue) → how "balanced" controllability is

**Why it matters**:
- If z_dim=64 but only 5 eigenvalues are significant, the controller can only influence 5 effective dimensions — the rest are "along for the ride"
- Knowing which dimensions are controllable helps design better control objectives
- A well-conditioned Gramian means the controller can efficiently reach any desired state
- Comparing Gramians between causal and acausal models reveals whether the causal model preserves controllability structure

---

## Metric Summary Table

| Metric | What it tells you | Current | Needed for control? |
|--------|------------------|---------|:---:|
| **R² (200ms)** | Overall prediction quality | ✅ Implemented | Indirect |
| **MSE** | Error magnitude | ✅ Implemented | Indirect |
| **FIT% (free-run)** | Long-horizon ODE stability | 🔜 To implement | Yes |
| **Error growth curves** | Short-horizon accuracy profile | 🔜 To implement | **Critical** |
| **Jacobian accuracy** | Stimulation response fidelity | 🔜 To implement | **Critical** |
| **Controllability Gramian** | Which states can be controlled | 🔜 To implement | Yes |

> [!IMPORTANT]
> **For MPC, the error growth curve and Jacobian accuracy are the two most important metrics.**
> A model with R²=0.80 but accurate g(x) and good 20-step predictions will outperform
> a model with R²=0.95 but poor input sensitivity. We've been optimizing the wrong thing.
