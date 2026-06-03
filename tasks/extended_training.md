# Phase 2.8: Extended Training Sweep (Cosine Decay + Linear Warmup)

This task implements learning rate warmup and cosine annealing decay for CA-NODE training, and executes an extended sweep (500–1000 epochs) to investigate if slower learning rates converge to a higher global optimum without gradient instability.

## Implementation Plan

1. **Modify Training Script (`modeling/scripts/fit_canode.py`)**:
   - Add command line arguments:
     - `--lr-warmup`: Number of epochs for linear warmup (default: 0).
     - `--n-epochs`: Support 500-1000 epochs (already exists, but check).
     - `--lr-decay`: Enable cosine annealing decay (boolean flag).
   - Implement the learning rate scheduler:
     - Use `torch.optim.lr_scheduler.LambdaLR` with a custom lambda function combining linear warmup and cosine decay.
     - Ensure the scheduler is updated per epoch.
2. **Verify Implementation**:
   - Run a short test run to confirm the learning rate scheduler scales the learning rate correctly across training epochs.
3. **Execute Sweep (`modeling/scripts/sweep_canode.py`)**:
   - Add a grid configuration for extended training (e.g. `--grid extended`).
   - Run sweep in `tmux` on a GPU machine.
4. **Dashboard and Log Update**:
   - Update `dashboard.html` with results of the sweep.
   - Update `task.md` and `branches.md`.

## Progress Checklist

- [ ] Add CLI arguments for LR warmup and cosine decay to `fit_canode.py`
- [ ] Implement linear warmup + cosine decay scheduler in `train_canode` or `fit_canode.py`
- [ ] Verify learning rate scheduler behavior with a short run
- [ ] Add extended grid sweep configuration to `sweep_canode.py`
- [ ] Launch extended training sweep in tmux
- [ ] Update results dashboard and coordination files
