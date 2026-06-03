# Phase 2: Data Scaling & Architecture Sweep

Parent: [task.md](file:///home/cbwash2/cleo/task.md)
Branch: `feature/digital-twin-phase1`
Agent: Antigravity

## Goal

Phase 1 showed that hidden=128 outperformed hidden=256 due to overfitting on 8 training trials.
This phase tests the hypothesis that **scaling data unlocks bigger architectures** by:
1. Generating a 50-trial dataset (5× the original)
2. Running a full architecture sweep across hidden sizes, depths, and learning rates
3. Comparing scaling behavior: does more data + bigger model = better R²?

## Parameters

| Param | Value |
|-------|-------|
| Trials | 50 (40 train / 10 test) |
| Duration per trial | 30s |
| Total simulation time | 1500s (25 min) |
| Sample period | 1 ms |
| MUA channels | 50 |
| Fibers | 2 |
| Smoothing τ | 20 ms |
| OU params | τ=50ms, σ=2 mW/mm², μ=5 mW/mm² |
| Solver | dopri5 (default) |
| Evaluation | 200ms windowed R² |

## Sweep Grid

| Hyperparameter | Values |
|----------------|--------|
| Hidden dim | 128, 256, 512 |
| Layers | 2, 3, 4 |
| Learning rate | 5e-4, 2e-4, 1e-4 |
| Method | dopri5 |
| Compile | True |
| Epochs | 200 |
| Batch size | 512 |
| **Total configs** | **27** |
| **GPUs** | 8× RTX 2080 Ti (gpu1) |
| **Estimated time** | ~4 rounds × 10 min = ~40 min |

## Checklist

### Data Generation
- `[ ]` Generate 50-trial dataset on gpu1 (48 CPUs, ~20 min est.)
- `[ ]` Save as `data/training_trials_50.h5`
- `[ ]` Verify shapes: x=(50, 50, 30000), u=(50, 2, 30000)

### Sweep Script Updates
- `[ ]` Update `sweep_canode.py` to accept `--configs-file` or expand grid inline
- `[ ]` Add `--n-test-trials` arg to control train/test split
- `[ ]` Verify GPU distribution works for >8 configs (queue system)

### Parallel Sweep Execution
- `[ ]` Launch 27-config sweep on gpu1 (8 GPUs, queued)
- `[ ]` Monitor and collect results in `results/sweep_v2/`
- `[ ]` Parse all logs for 200ms windowed R² metrics

### Analysis
- `[ ]` Compare best R² across hidden sizes: does 256/512 beat 128 with 5× data?
- `[ ]` Plot scaling curves: R² vs hidden size, R² vs n_layers
- `[ ]` Identify the Pareto frontier (accuracy vs training time)
- `[ ]` Update `results/dashboard.html` with Phase 2 results

### Documentation
- `[ ]` Update walkthrough.md
- `[ ]` Commit to feature branch
