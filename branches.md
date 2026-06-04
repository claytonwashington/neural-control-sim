# Branch ↔ Worktree ↔ Experiment Mapping

## Active Phase 4 Experiments

| Branch | Worktree Path | Experiment | Status | GPU |
|--------|---------------|------------|--------|-----|
|  |  | Exp 2: Encoder Distillation | 🔄 Running | gpu1:4 |
|  |  | Exp 3: Hybrid Distillation (α sweep) | 🔄 Running | gpu1:0-2,7 |
|  |  | Exp 4+5: Causal+Acausal Grokking | 🔄 Running | gpu2:0-1 |
|  |  | Exp 6: EnKF State Estimation | 🔄 Running | gpu1:5 |
|  |  | Exp 7: Delayed Residual Correction | 🔄 Running | gpu1:6 |

## Additional Causal Sweep (on grokking branch)
| Config | tmux | GPU |
|--------|------|-----|
| z=128, h128, pw200, lr=3e-4 | causal_z128 | gpu2:2 |
| z=64, h128, pw200, 500ep, wd=5e-5 | causal_z64_500ep | gpu2:3 |
| z=64, h128, pw1000, lr=3e-4 | causal_z64_pw1000 | gpu2:4 |

## Completed Phase 3 Experiments

| Branch | Status | Result |
|--------|--------|--------|
|  | ✅ Completed | Merged to modeling-dev |
|  | ✅ Completed | R²=0.8907 (no benefit) |
|  | ✅ Completed | R²=0.8911 (hurt) |
|  | ✅ Completed | R²=0.899 (no effect) |
