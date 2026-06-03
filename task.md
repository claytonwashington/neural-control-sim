# Cleo Project Tasks

To ensure coordinate development across different agents (IDE, 2.0 CLI, Jules, Gemini CLI), follow these rules:
1. **Branch Check**: Make sure your active branch is registered in [branches.md](file:///home/cbwash2/cleo/branches.md). Do not work directly on `main` unless it's a minor change.
2. **Subtask Linking**: If a task has granular steps, create a markdown file in the `tasks/` directory, link it here, and link it back to this file.

---

## Active Task List

- `[ ]` example_parent_task ([details](file:///home/cbwash2/cleo/tasks/README.md))
- `[x]` Phase 1: Digital Twin Inner Loop ([details](file:///home/cbwash2/cleo/tasks/digital_twin_phase1.md))
  - Branch: `feature/digital-twin-phase1`
  - Agent: Antigravity
  - Cleo E/I plant + OU data gen + N4SID + control-affine NODE
- `[x]` Phase 2: Data Scaling & Architecture Sweep ([details](file:///home/cbwash2/cleo/tasks/digital_twin_phase2.md))
  - Branch: `feature/digital-twin-phase1`
  - Agent: Antigravity
  - 50-trial dataset + 27-config sweep → h128 l2 lr=1e-4 best (R²=0.9016). Bigger models don't help.
- `[x]` Phase 2.5: Skip Connections (R²=0.8895 — worse, overfits)
  - state skip (linear) + input skip (MLP) bypassing ODE integrator
  - Train loss improved but generalization degraded → skip_input MLP overfits
- `[x]` Phase 2.6: Simpler Skip Connections (Linear-only, completed R²=0.9144)
  - Branch: `feature/skip-connections`
  - Worktree: `/snel/home/cbwash2/cleo-worktrees/skip-connections`
  - Linear-only state/input skip connections. Overfitting was avoided by applying separate weight decay (best: `swd=1.0` got R²=0.9144), but still did not beat the pure CA-NODE baseline (R²=0.9219).
- `[ ]` Frequency-Aware Loss: Spectral Loss to break R² ceiling ([details](file:///snel/home/cbwash2/cleo/tasks/spectral_loss.md))
  - Branch: `feature/spectral-loss`
  - Worktree: `/snel/home/cbwash2/cleo-worktrees/spectral-loss`
  - Agent: Antigravity (Instance 2)
  - Penalize errors in FFT magnitude domain to capture high-frequency transients.
- `[ ]` Multi-Scale / Multi-Rate Integration: Decoupled slow/fast step sizes ([details](file:///snel/home/cbwash2/cleo/tasks/multi_rate_integration.md))
  - Branch: `feature/multi-rate-integration`
  - Worktree: `/snel/home/cbwash2/cleo-worktrees/multi-rate-integration`
  - Agent: Antigravity (Instance 3)
  - Decouple slow baseline dynamics and fast optogenetic responses with sub-stepped fast integration.
- `[ ]` Phase 2.8: Extended Training Sweep (Cosine Decay + Linear Warmup) ([details](file:///snel/home/cbwash2/cleo-worktrees/extended-training/tasks/extended_training.md))
  - Branch: `feature/extended-training`
  - Worktree: `/snel/home/cbwash2/cleo-worktrees/extended-training`
  - Agent: Antigravity (Instance 4)
  - Train for 500–1000 epochs with cosine decay and initial linear warmup to confirm if slower learning rates converge to a higher global optimum.
- `[x]` Causal Latent CA-NODE: Causal encoding and forward forecasting sweep (Completed R²=0.4405)
  - Branch: `feature/digital-twin-phase1` (Main repo / local sync)
  - Agent: Antigravity (Primary)
  - Inferred initial latent state causally via GRU encoder on past window, integrated 200ms forward. Best $R^2 \approx 0.4405$ (200ms past window, h256, lr=0.0005).

### Backlog — Ideas to Break the R²≈0.90 Ceiling

- `[ ]` Latent NODE: encoder (50ch → ~10 latent) → ODE in latent space → decoder (LFADS-style)
- `[ ]` Discrete-time model: replace ODE integrator with a GRU/LSTM to avoid integration smoothing entirely
