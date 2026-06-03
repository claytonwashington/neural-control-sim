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
- `[ ]` Frequency-Aware Loss: Spectral Loss to break R² ceiling ([details](file:///snel/home/cbwash2/cleo/tasks/spectral_loss.md))
  - Branch: `feature/spectral-loss`
  - Agent: Antigravity
  - Penalize errors in FFT magnitude domain to capture high-frequency transients.
- `[ ]` Multi-Scale / Multi-Rate Integration: Decoupled slow/fast step sizes ([details](file:///snel/home/cbwash2/cleo/tasks/multi_rate_integration.md))
  - Branch: `feature/multi-rate-integration`
  - Agent: Antigravity
  - Decouple slow baseline dynamics and fast optogenetic responses with sub-stepped fast integration.

### Backlog — Ideas to Break the R²≈0.90 Ceiling

- `[ ]` Simpler skip connections: linear-only (no MLP), or state-skip only, with stronger regularization
- `[ ]` Latent NODE: encoder (50ch → ~10 latent) → ODE in latent space → decoder (LFADS-style)
- `[ ]` Longer training: 500–1000 epochs with warmup (200 may be insufficient)
- `[ ]` Discrete-time model: replace ODE integrator with a GRU/LSTM to avoid integration smoothing entirely
