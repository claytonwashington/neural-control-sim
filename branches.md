# Branch / Worktree / Experiment Mapping

## Active Phase 4 Experiments

| Branch | Worktree | Experiment | Status | GPU |
|--------|----------|------------|--------|-----|
| feature/encoder-distillation | cleo-worktrees/encoder-distillation | Exp 2: Encoder Distillation | Running | gpu1:4 |
| feature/hybrid-distillation | cleo-worktrees/hybrid-distillation | Exp 3: Hybrid Distillation | Starting | gpu1:0,1,2,7 |
| feature/grokking | cleo-worktrees/grokking | Exp 4+5: Grokking (causal+acausal) | Running | gpu2:0-1 |
| feature/kalman-filter | cleo-worktrees/kalman-filter | Exp 6: EnKF (2080 Ti) | Done | gpu1:5 |
| feature/residual-correction | cleo-worktrees/residual-correction | Exp 7: Residual Correction | Done | gpu1:6 |
| feature/enkf-a100 | cleo-worktrees/enkf-a100 | Exp 13: EnKF A100 Speed Sweep | Running | gpu2:5 |

## Additional Causal Sweeps (in progress)

| Config | tmux session | GPU | Status |
|--------|-------------|-----|--------|
| z=128, h128, pw200, lr=3e-4 | causal_z128 | gpu2:2 | Running |
| z=64, h128, pw200, 500ep, wd=5e-5 | causal_z64_500ep | gpu2:3 | Running |
| z=64, h128, pw1000, lr=3e-4 | causal_z64_pw1000 | gpu2:4 | Running |

## Completed Phase 2-3 Experiments

| Branch | Worktree | Result |
|--------|----------|--------|
| feature/extended-training | cleo-worktrees/extended-training | Merged to modeling-dev |
| feature/multi-rate-integration | cleo-worktrees/multi-rate-integration | R2=0.8907 (no benefit) |
| feature/skip-connections | cleo-worktrees/skip-connections | R2=0.8911 (hurt) |
| feature/spectral-loss | cleo-worktrees/spectral-loss | R2=0.899 (no effect) |
