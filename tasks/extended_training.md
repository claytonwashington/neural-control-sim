## Sweep Details

The extended training sweep is defined under the `"extended"` grid option in `sweep_canode.py`. It comprises **12 configurations** to evaluate if larger models and slower learning rates can converge to a better global optimum when trained longer with learning rate warmup and decay.

* **Total Epochs**: 500
* **Hidden Sizes (`hidden`)**: `128`, `256`
* **Learning Rates (`lr`)**: `1e-4`, `5e-5` (exploring slower learning rates to prevent integration gradient explosion)
* **Warmup Epochs (`lr_warmup`)**: `0` (no warmup), `20` epochs, `50` epochs (linear scaling of learning rate from 0 up to the base rate)
* **Learning Rate Decay**: Cosine Annealing decay enabled for all models post-warmup.
* **Skip Connections**: Disabled (fully autonomous Neural ODE baseline).

## Seeding & Reproducibility (40/10 Split)
* All model training processes are seeded with `DEFAULT_SEED = 42` to guarantee identical neural network initializations.
* The train/test split is fixed to **40 training trials and 10 testing trials** (using the 50-trial dataset).
* The train/validation window split is deterministic (seeded via a manual Torch generator during `random_split`).

## Dual Checkpointing Logic
Validation loss fluctuates due to adaptive integration steps in Neural ODEs. To avoid saving sub-optimal final parameters:
1. The training loop keeps track of the epoch with the lowest validation loss.
2. The parameters for that best epoch are automatically restored to the model at the end of training and saved to the standard path (e.g., `results/sweep_extended/run_xx.pt`).
3. The final (most recent) epoch's model parameters are saved with a `_last.pt` suffix (e.g., `results/sweep_extended/run_xx_last.pt`) for evaluation.

## Progress Checklist

- [x] Add CLI arguments for LR warmup and cosine decay to `fit_canode.py`
- [x] Implement linear warmup + cosine decay scheduler in `train_canode` or `fit_canode.py`
- [x] Verify learning rate scheduler behavior with a short run
- [x] Add extended grid sweep configuration to `sweep_canode.py`
- [x] Launch extended training sweep in tmux
- [x] Update results dashboard and coordination files
