# Branch / Worktree / Experiment Mapping

## Active Phase 4 Experiments

| Branch | Worktree | Experiment | Status | GPU |
|--------|----------|------------|--------|-----|
| feature/hybrid-distillation | cleo-worktrees/hybrid-distillation | Exp 8: Hybrid Distillation | Pending launch | gpu1 |
| feature/grokking | cleo-worktrees/grokking | Exp 9: Causal Grokking | Running | gpu2:0 |
| feature/grokking | cleo-worktrees/grokking | Exp 10: Acausal Grokking | Running | gpu2:1 |
| feature/enkf-causal | cleo-worktrees/enkf-causal | Exp 15: EnKF Causal Init | Done | gpu1:0-3 |
| feature/enkf-fulltrial | cleo-worktrees/enkf-fulltrial | Exp 17: EnKF Full Trial | Launching | gpu1:0-3 |
| feature/control-metrics | cleo-worktrees/control-metrics | Exp 16: Control Metrics | Done | gpu1:4 |
| feature/control-metrics | cleo-worktrees/control-metrics | Control Metrics Suite | Launching | gpu1:4 |

## Completed Phase 4 Experiments

| Branch | Worktree | Experiment | Result |
|--------|----------|------------|--------|
| feature/encoder-distillation | cleo-worktrees/encoder-distillation | Exp 7: Encoder Distill | R2=0.6639 (worse than direct) |
| feature/kalman-filter | cleo-worktrees/kalman-filter | Exp 11: EnKF (2080 Ti) | R2=0.9997 (1.1s, not RT) |
| feature/residual-correction | cleo-worktrees/residual-correction | Exp 12: Residual Corr | R2=0.34 (cross-arch limit) |
| feature/enkf-a100 | cleo-worktrees/enkf-a100 | Exp 13: EnKF A100 | R2=1.0 but no speed gain |
| feature/hybrid-distillation | cleo-worktrees/hybrid-distillation | Exp 8: Hybrid Distill | R2=0.48 (worse than direct) |

## Additional Runs (no new scripts needed)

| Config | tmux | GPU | Status | Result |
|--------|------|-----|--------|--------|
| z=128, h128 | - | gpu2:2 | Done | R2=0.8430, 142ms |
| z=64, 500ep, wd=5e-5 | causal_z64_500ep | gpu2:3 | Running | - |
| z=64, pw=1000 | causal_z64_pw1000 | gpu2:4 | Running | - |

## Completed Phase 2-3 Experiments

| Branch | Worktree | Result |
|--------|----------|--------|
| feature/extended-training | cleo-worktrees/extended-training | Merged to modeling-dev |
| feature/multi-rate-integration | cleo-worktrees/multi-rate-integration | R2=0.8907 (no benefit) |
| feature/skip-connections | cleo-worktrees/skip-connections | R2=0.8911 (hurt) |
| feature/spectral-loss | cleo-worktrees/spectral-loss | R2=0.899 (no effect) |
