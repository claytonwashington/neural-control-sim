<!-- PREFLIGHT:AUTO-MANAGED — Active Experiments table is updated by preflight.py -->
# Branch / Worktree / Experiment Mapping

## Active Experiments

| Branch | Worktree | Experiment | Results Dir | Status |
|--------|----------|------------|-------------|--------|
| feature/bidirectional-plant | cleo-worktrees/bidirectional-plant | Bidirectional plant setup (ChrimsonR+GtACR2) | — | Active — dataset generated |
| feature/optoclamp | cleo-worktrees/optoclamp | Exp 19: Optoclamp MPC (excitatory plant) | — | Done — results collected |
| feature/mpc-latency | cleo-worktrees/mpc-latency | Exp 19b: MPC under 15ms latency | — | Done — results collected |
| feature/grokking | cleo-worktrees/grokking | Exp 10: Grokking (causal z64) | — | Done — R²=0.8535 (causal), acausal killed |
| feature/bidir-aligned-distill | cleo-worktrees/bidir-aligned-distill | [Exp 23](ideas/modeling.md) | `results/bidir_aligned_distill/` | Done — R²=0.8613 |
| feature/bidir-optoclamp | cleo-worktrees/bidir-optoclamp | [Exp 27](ideas/modeling.md) | `results/bidir_optoclamp/` | Preflight complete |
| feature/bidir-v2-plant | cleo-worktrees/bidir-v2-plant | [Exp 42](ideas/modeling.md) | `results/lfads_nocon_ext/` | Failed —  |
| feature/bidir-v2-plant | cleo-worktrees/bidir-v2-plant | [Exp 43](ideas/modeling.md) | `results/lfads_nocon_noext/` | Preflight complete |
| feature/bidir-v2-plant | cleo-worktrees/bidir-v2-plant | [Exp 44](ideas/modeling.md) | `results/spiking_canode_poisson_v2_p0/` | Preflight complete |
| feature/mua-decoding-metric | cleo-worktrees/mua-decoding-metric | [Exp 45](ideas/modeling.md) | `results/mua_decoding_metric/` | Done — R²=0.0000 |

## Completed Experiments

| Branch | Worktree | Experiment | Result |
|--------|----------|------------|-------------|--------|
| modeling-dev | cleo (main) | Exp 1: N4SID Baseline | — | R²=0.40 |
| modeling-dev | cleo (main) | Exp 2: GRU Baseline | — | R²=0.69 |
| modeling-dev | cleo (main) | Exp 3: Latent NODE (Acausal) | — | R²=0.9387 🏆 |
| modeling-dev | cleo (main) | Exp 5: Causal Latent CA-NODE | — | R²=0.8252 |
| modeling-dev | cleo (main) | Exp 11: 500-Trial Dataset | — | Generated |
| feature/skip-connections | cleo-worktrees/skip-connections | Exp 4/6: Skip Connections | — | R²=0.8911 ❌ |
| feature/spectral-loss | cleo-worktrees/spectral-loss | Exp 4: Spectral Loss | — | R²=0.8993 ❌ |
| feature/multi-rate-integration | cleo-worktrees/multi-rate-integration | Exp 4: Multi-Rate Integration | — | R²=0.8907 ❌ |
| feature/encoder-distillation | cleo-worktrees/encoder-distillation | Exp 7/9: Encoder Distillation | — | R²=0.6639 ❌ |
| feature/hybrid-distillation | cleo-worktrees/hybrid-distillation | Exp 8: Hybrid Distillation | — | R²=0.4826 ❌ |
| feature/extended-training | cleo-worktrees/extended-training | Extended Training Sweep | — | Merged to modeling-dev |
| feature/residual-correction | cleo-worktrees/residual-correction | Exp 12: Residual Correction | — | R²=0.3445 ❌ |
| feature/enkf-a100 | cleo-worktrees/enkf-a100 | Exp 13: EnKF A100 Speed | — | No speedup ❌ |
| feature/kalman-filter | cleo-worktrees/kalman-filter | Exp 14: EnKF Baseline | — | R²=0.9997 (not RT) |
| feature/enkf-causal | cleo-worktrees/enkf-causal | Exp 15: EnKF Causal Init | — | Done |
| feature/control-metrics | cleo-worktrees/control-metrics | Exp 16: Control Metrics | — | Done |
| feature/enkf-fulltrial | cleo-worktrees/enkf-fulltrial | Exp 17: Full-Trial EnKF | — | Done |
| feature/residual-v2 | cleo-worktrees/residual-v2 | Exp 18: Residual v2 | — | Done |
| feature/aligned-distill | cleo-worktrees/aligned-distill | Exp 20: Aligned Distillation | — | R²=0.8640 ✅ |
| feature/periodic-reencode | cleo-worktrees/periodic-reencode | Exp 21/22: EnKF + Periodic Re-Encoding | — | R²=0.877 ✅ |

## Tooling / Infrastructure Branches

(Not preflight experiments — kept out of the auto-managed table above.)

| Branch | Worktree | Purpose | Status |
|--------|----------|---------|--------|
| feature/dashboard-validation | cleo-worktrees/dashboard-validation | Plants page + per-experiment validation plots (true-vs-inferred firing rates across channels & top PCs) + clickable leaderboard experiment pages | Active |
