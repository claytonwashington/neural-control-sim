import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.signal import welch

# Add repository path to sys.path
sys.path.append("/mnt/cbwash2/cleo")

from modeling.data import load_trials_h5
from modeling.models.canode import ControlAffineODE

def main():
    data_path = "data/training_trials_50.h5"
    output_dir = "results/sweep_spectral"
    
    # Load data
    print("Loading data...")
    data = load_trials_h5(data_path)
    x_all, u_all = data["x"], data["u"]
    dt = float(data["dt"])
    n_trials, n_channels, n_steps = x_all.shape
    
    # Use last trial for evaluation (Trial 49)
    test_idx = 49
    x_test = x_all[test_idx]  # (n_ch, T)
    u_test = u_all[test_idx]  # (n_u, T)
    
    # We will load the stats from the baseline model to normalize
    baseline_model_path = os.path.join(output_dir, "run_00_h128_l2_lr0.0001_dopri5_sa0.0_noskip_compiled.pt")
    if not os.path.exists(baseline_model_path):
        print(f"Error: Baseline model not found at {baseline_model_path}")
        return
        
    checkpoint = torch.load(baseline_model_path, map_location="cpu")
    x_mean = checkpoint["x_mean"]
    x_std = checkpoint["x_std"]
    u_mean = checkpoint["u_mean"]
    u_std = checkpoint["u_std"]
    
    x_test_n = (x_test - x_mean) / x_std
    u_test_n = (u_test - u_mean) / u_std
    
    horizon = 200
    n_ch, T_val = x_test_n.shape
    n_windows = (T_val - horizon) // horizon
    t_win = torch.arange(horizon, dtype=torch.float32) * dt
    
    alphas = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0]
    colors = ["r", "g", "b", "m", "c", "y"]
    labels = ["sa=0.0 (Baseline)", "sa=0.01", "sa=0.1", "sa=0.5", "sa=1.0", "sa=2.0"]
    
    predictions = {}
    
    # Reconstruct true concatenated windows
    true_concat = []
    for w in range(n_windows):
        t0 = w * horizon
        t1 = t0 + horizon
        true_concat.append(x_test_n[:, t0:t1])
    true_concat = np.concatenate(true_concat, axis=1)  # (n_ch, n_windows * horizon)
    
    for alpha in alphas:
        # Find model file
        # Map them explicitly:
        idx_map = {0.0: 0, 0.01: 1, 0.1: 3, 0.5: 5, 1.0: 6, 2.0: 7}
        idx = idx_map[alpha]
        run_name = f"run_{idx:02d}_h128_l2_lr0.0001_dopri5_sa{alpha}_noskip_compiled"
        model_path = os.path.join(output_dir, f"{run_name}.pt")
        
        if not os.path.exists(model_path):
            print(f"Skipping sa={alpha}, model not found at {model_path}")
            continue
            
        print(f"Evaluating model for sa={alpha} ...")
        ckpt = torch.load(model_path, map_location="cpu")
        model = ControlAffineODE(
            n_x=n_channels,
            n_u=u_all.shape[1],
            hidden=ckpt["hidden"],
            n_layers=ckpt["n_layers"],
            use_skip=ckpt.get("use_skip", False),
        )
        
        # Load weights
        state_dict = ckpt["model_state"]
        cleaned_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned_state_dict)
        model.eval()
        
        pred_concat = []
        for w in range(n_windows):
            t0 = w * horizon
            t1 = t0 + horizon
            x0 = torch.tensor(x_test_n[:, t0], dtype=torch.float32)
            u_win = torch.tensor(u_test_n[:, t0:t1].T, dtype=torch.float32)
            
            with torch.no_grad():
                x_pred_win = model.predict(
                    x0.unsqueeze(0), t_win, u_win.unsqueeze(0), method="dopri5"
                )  # (1, T, n_x)
                pred_concat.append(x_pred_win[0].numpy().T)
                
        predictions[alpha] = np.concatenate(pred_concat, axis=1)  # (n_ch, T_eval_concat)
        
    print("Computing Power Spectral Density (PSD)...")
    # Sample rate is 1000 Hz
    fs = 1.0 / dt
    
    # Compute PSD for true signal
    f, psd_true = welch(true_concat, fs=fs, nperseg=256, axis=1)
    psd_true_mean = psd_true.mean(axis=0)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot PSD
    ax = axes[0]
    ax.semilogy(f, psd_true_mean, "k", lw=2, label="Ground Truth")
    
    for alpha, color, label in zip(alphas, colors, labels):
        if alpha not in predictions:
            continue
        _, psd_pred = welch(predictions[alpha], fs=fs, nperseg=256, axis=1)
        psd_pred_mean = psd_pred.mean(axis=0)
        ax.semilogy(f, psd_pred_mean, color, lw=1.2, alpha=0.8, label=label)
        
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density")
    ax.set_title("PSD Comparison: True vs Predicted Signals (Averaged across channels)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    
    # Plot PSD Ratio / Error to see differences in high frequency
    ax = axes[1]
    for alpha, color, label in zip(alphas, colors, labels):
        if alpha not in predictions:
            continue
        _, psd_pred = welch(predictions[alpha], fs=fs, nperseg=256, axis=1)
        psd_pred_mean = psd_pred.mean(axis=0)
        
        # Log ratio of PSD (Predicted / True)
        ratio = psd_pred_mean / psd_true_mean
        ax.plot(f, ratio, color, lw=1.2, label=f"Ratio for {label}")
        
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.5, label="Perfect match")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD Ratio (Predicted / True)")
    ax.set_title("PSD Match Quality (Value near 1 is better)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    plot_path = "results/sweep_spectral/psd_comparison.png"
    fig.savefig(plot_path, dpi=150)
    print(f"PSD comparison plot saved to {plot_path}")

if __name__ == "__main__":
    main()
