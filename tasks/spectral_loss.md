# Spectral Loss: Frequency-Aware Loss Implementation

Parent: [task.md](file:///mnt/cbwash2/cleo/task.md)
Branch: `feature/spectral-loss`
Agent: Antigravity

## Goal

Standard Mean Squared Error (MSE) loss in the time domain is dominated by large-scale low-frequency signals. Adding an FFT-magnitude penalty forces the optimizer to align predictions with the high-frequency spectral components of the true multi-unit activity, potentially breaking the $R^2 \approx 0.90$ ceiling on 200ms prediction windows.

## Loss Formulation

$$\mathcal{L} = \text{MSE}(x_{\text{pred}}, x_{\text{true}}) + \alpha \cdot \text{MSE}(|\text{FFT}(x_{\text{pred}})|, |\text{FFT}(x_{\text{true}})|)$$

where:
- $\text{FFT}$ is the 1D discrete Fourier transform (typically `torch.fft.rfft`) computed along the time dimension of each window.
- $|\cdot|$ is the magnitude (absolute value) of the complex coefficients.
- $\alpha$ is the spectral loss coefficient.

## Checklist

### Implementation
- `[ ]` Implement FFT-magnitude MSE loss in `train_canode` or a separate utility.
- `[ ]` Add `--spectral-alpha` argument to `fit_canode.py` (default: `0.0` to preserve baseline behavior).
- `[ ]` Ensure the training/validation curves and log output report both time-domain MSE and spectral loss components.
- `[ ]` Update `sweep_canode.py` to support sweeping `--spectral-alpha`.

### Verification & Testing
- `[ ]` Test implementation locally with `alpha = 0.0` to verify identical behavior to baseline.
- `[ ]` Run a trial training with `alpha = 0.1` and `alpha = 1.0` inside a `tmux` session to ensure stability and convergence.

### Hyperparameter Sweep
- `[ ]` Run a sweep over different values of $\alpha$ (e.g., `0.01`, `0.05`, `0.1`, `0.2`, `0.5`, `1.0`, `2.0`) to find the optimal spectral weight.
- `[ ]` Compare training speed and stability across different values.

### Analysis & Dashboard
- `[ ]` Compare the best 200ms windowed $R^2$ to the baseline (without spectral loss, $R^2 \approx 0.9016$).
- `[ ]` Plot the frequency spectrum (Power Spectral Density / FFT magnitude) of true vs. predicted signals to confirm high-frequency alignment.
- `[ ]` Update the results dashboard at `results/dashboard.html` with results from the sweep.
- `[ ]` Update `walkthrough.md` and commit changes.
