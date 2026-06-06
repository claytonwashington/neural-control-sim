# Control Ideas & Experiments

> This document tracks all control-layer work: controller implementations,
> closed-loop experiments, and control-relevant model evaluation.
> For prediction/state-estimation experiments, see [modeling_ideas.md](modeling_ideas.md).

## Standard Control Experiment Outputs
Every control experiment MUST produce:
1. **Tracking traces** — target rate vs actual rate over time for each target level
2. **Metrics table** — RMSE, settling time, steady-state error, control effort per config
3. **Comparison bar chart** — side-by-side with baseline controllers (PI, LDS+LQR)
4. **Control input traces** — stimulation amplitudes over time (shows controller behavior)

### Metric Definitions
- **Steady-state error** = mean |actual_rate − target| during last 200ms of clamp period
- **Tracking RMSE** = sqrt(mean((rate − target)²)) over entire clamp period
- **Settling time** = time to reach and stay within 10% of target for ≥50ms
- **Control effort** = mean |u(t)| over the clamp period
- **Overshoot** = max((rate − target) / target) × 100% after initial rise

---

## Controller Inventory

### PI Controller (Baseline)
**File**: `modeling/controllers/pi_controller.py`
- Proportional-Integral feedback on population mean rate
- Tuned via grid sweep: Kp ∈ {0.1, 0.5, 1.0, 2.0}, Ki = Kp/100
- Best gains: Kp=2.0, Ki=0.02
- Limitation: monotonic u→rate mapping, cannot suppress below baseline with excitatory-only opsins

### Neural ODE MPC
**File**: `modeling/controllers/node_mpc.py`
- Model Predictive Control using learned CA-NODE dynamics
- Horizon H=20 steps (20ms), MPC iterations=30, Adam optimizer
- Uses Euler rollout with warm-start from previous solution
- Reads spike counts from Cleo's `LatencyIOProcessor`, converts to rates
- Normalization using checkpoint stats (x_mean, x_std, u_mean, u_std)

### LDS + LQR (Planned)
- N4SID system identification → linear state-space model
- LQR optimal feedback gain K = dlqr(A, B, Q, R)
- Baseline for linear model-based control
- Depends on `ldsCtrlEst` library (CLOCTools, Georgia Tech)

---

## Experiments

### Exp 19. Optoclamp: PI vs MPC Closed-Loop Control
**Status**: ✅ COMPLETE — MPC beats PI (14× lower SS error)
**Branch/Worktree**: `feature/optoclamp` / `cleo-worktrees/optoclamp`
**Script**: `modeling/scripts/run_optoclamp.py`

**Setup**:
- 1000 LIF neurons (800E/200I), 2 optic fibers (ChR2), 50-channel MUA
- Baseline spontaneous rate: ~129 Hz
- Targets: 50%, 75%, 125% of baseline
- Simulation: 1.4s per trial (200ms baseline + 1.2s clamp)
- MPC compute delay: 5ms, PI compute delay: 0ms

**Results**:

| Controller | Target | RMSE | Settling (ms) | SS Error (Hz) |
|---|---|---|---|---|
| PI | 50% (↓) | 163.8 | ∞ | 210.6 |
| PI | 75% (↓) | 194.4 | ∞ | 246.9 |
| PI | 125% (↑) | 101.3 | 64 | 127.9 |
| **NODE MPC** | 50% (↓) | 170.8 | ∞ | 210.6 |
| **NODE MPC** | 75% (↓) | 160.8 | ∞ | 211.1 |
| **NODE MPC** | 125% (↑) | **56.4** | 735 | **9.3** |

**Key Findings**:
- Neither controller can push rates DOWN (excitatory-only opsins — need NpHR for inhibitory)
- MPC achieves 14× lower steady-state error at 125% target (9.3 vs 127.9 Hz)
- MPC settles slower (735ms vs 64ms) but reaches a far more accurate steady state
- PI reacts fast but has large persistent offset; MPC plans ahead for precision

**Latency Gaps**:
- Current setup does NOT model observation latency (spike counts assumed instant)
- Full round-trip latency in deployment:
  ```
  Neural activity → Recording (1ms) → Spike sorting (2-5ms) →
  Rate estimation (1ms) → State estimation (1ms) → MPC solve (5-10ms) →
  DAC output (0.1ms) → Light delivery (0.1ms) ≈ 10-20ms total
  ```
- Next step: re-run with combined observation delay (10ms) + compute delay (5ms)

---

### Control-Relevant Model Metrics (from Exp 16)

| Metric | Causal z64 | Causal z128 | Acausal z64 |
|---|---|---|---|
| Free-run FIT% (30s) | 0 | 0 | 0 |
| R² at H=20 | 0.84 | 0.82 | 0.95 |
| Jacobian cos_sim (∂g/∂u) | **0.997** | 0.995 | 0.998 |
| Effective control dims | 3-5 / 64 | 3-5 / 128 | 3-5 / 64 |

**Implications for control**:
- Free-run FIT%=0 → ODE diverges without re-encoding. MPC must re-encode every ~200ms.
- Jacobian accuracy 99.7% → learned control matrix is nearly perfect. MPC gradients are trustworthy.
- Only 3-5 effective control dimensions → controller only needs ~5 latent dims. Low-rank MPC possible.
- MPC viable at H≤20 steps (20ms horizon).

---

## Next Steps (Priority Order)

### 1. Bidirectional Optoclamp
**Priority**: HIGH
- Add NpHR (inhibitory opsin) to plant alongside ChR2
- Enables downward rate clamping (50%, 75% targets)
- Re-run PI + MPC comparison with 2-channel control (excite + inhibit)
- Expected: both controllers should now handle all targets; MPC advantage may grow

### 2. Optoclamp with Full Latency
**Priority**: HIGH
- Add 10ms observation delay to the `LatencyIOProcessor`
- Combined with 5ms MPC compute delay → 15ms round-trip
- Test if MPC degrades gracefully or needs state prediction through the delay
- If degraded: integrate EnKF state estimator into the control loop

### 3. MPC with Aligned Model
**Priority**: MEDIUM
- Replace the original causal model (R²=0.825) with the aligned distill model (R²=0.864)
- Better model → better MPC predictions → tighter control
- Also test with the best encoder strategy from Exp 22 (periodic re-encoding)

### 4. LDS + LQR Baseline
**Priority**: MEDIUM
- Fit N4SID linear model to the same plant data
- Compute LQR gains, run closed-loop
- Provides "how much does nonlinearity help?" comparison
- Depends on resolving `ldsCtrlEst` import (`ModuleNotFoundError`)

### 5. Control Performance vs Model Accuracy Curve
**Priority**: MEDIUM
- Run optoclamp with models of increasing quality: N4SID → causal → aligned → acausal
- Plot RMSE / SS error vs model R²
- Answers: "what R² threshold enables useful control?"
- Critical for knowing when prediction model is "good enough"

### 6. Multi-Target Tracking
**Priority**: LOW
- Time-varying target rate (step sequence, ramp, sinusoidal)
- Tests controller's ability to track dynamic setpoints, not just clamp
- More realistic for BCI applications (neural prosthetics need dynamic targets)

---

## Relevant Literature
- Bolus, Willats, Rozell, Stanley (2021). State-space optimal feedback control of optogenetically driven neural activity. J. Neural Engineering.
- Newman et al. (eLife). Optoclamp for closed-loop optogenetic control.
- Johnsen et al. Cleo: Closed-Loop simulation testbed.
- ldsCtrlEst library (CLOCTools, Georgia Tech)
