#!/usr/bin/env python
"""Plot side-by-side prediction comparison: N4SID vs CA-NODE vs Ground Truth."""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    data_path = "data/training_trials.h5"
    n4sid_path = "results/n4sid/n4sid_model.npz"
    canode_path = "results/canode/model.pt"
    output_dir = "results"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    from modeling.data import load_trials_h5
    print("Loading data...")
    data = load_trials_h5(data_path)
    x_all, u_all = data["x"], data["u"]
    dt = float(data["dt"])
    n_trials, n_channels, n_steps = x_all.shape
    
    # Split matching training scripts (last 2 trials are test)
    n_test = 2
    n_train = n_trials - n_test
    x_test_trials = x_all[n_train:]
    u_test_trials = u_all[n_train:]
    
    # 2. Load N4SID
    from modeling.models.n4sid import N4SIDModel
    print("Loading N4SID...")
    n4_data = np.load(n4sid_path)
    n4_model = N4SIDModel(n_states=n4_data["A"].shape[0])
    n4_model.A = n4_data["A"]
    n4_model.B = n4_data["B"]
    n4_model.C = n4_data["C"]
    n4_model.D = n4_data["D"]
    n4_model.n_outputs = n4_model.C.shape[0]
    n4_model.n_inputs = n4_model.B.shape[1]
    
    # 3. Load CA-NODE
    from modeling.models.canode import ControlAffineODE
    print("Loading CA-NODE...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(canode_path, map_location=device)
    state_dict = checkpoint["model_state"]
    
    # Clean state dict keys if compiled
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        cleaned_key = k.replace("_orig_mod.", "")
        cleaned_state_dict[cleaned_key] = v
        
    canode_model = ControlAffineODE(
        n_x=checkpoint["n_x"],
        n_u=checkpoint["n_u"],
        hidden=checkpoint["hidden"],
        n_layers=checkpoint["n_layers"],
        use_skip=checkpoint.get("use_skip", True),
        skip_type=checkpoint.get("skip_type", "mlp")
    )
    canode_model.load_state_dict(cleaned_state_dict)
    canode_model = canode_model.to(device)
    canode_model.eval()
    
    x_mean = checkpoint["x_mean"]
    x_std = checkpoint["x_std"]
    u_mean = checkpoint["u_mean"]
    u_std = checkpoint["u_std"]
    
    # 4. Generate predictions on Trial 8 (first test trial)
    ti = 0
    x_test = x_test_trials[ti] # shape (n_channels, T)
    u_test = u_test_trials[ti] # shape (n_inputs, T)
    
    # Normalize for CA-NODE
    x_test_n = (x_test - x_mean) / x_std
    u_test_n = (u_test - u_mean) / u_std
    
    horizon = 200 # 200ms
    n_plot_windows = 5 # plot 1 second total
    T_plot = horizon * n_plot_windows
    
    # Prepare N4SID predictions (windowed)
    n4_pred = np.zeros((n_channels, T_plot))
    for w in range(n_plot_windows):
        t0 = w * horizon
        t1 = t0 + horizon
        x0 = x_test[:, t0]
        u_win = u_test[:, t0:t1]
        n4_pred[:, t0:t1] = n4_model.predict(x0, u_win)
        
    # Prepare CA-NODE predictions (windowed)
    canode_pred_n = np.zeros((n_channels, T_plot))
    t_win = torch.arange(horizon, dtype=torch.float32).to(device) * dt
    for w in range(n_plot_windows):
        t0 = w * horizon
        t1 = t0 + horizon
        x0_n = torch.tensor(x_test_n[:, t0], dtype=torch.float32).to(device)
        u_win_n = torch.tensor(u_test_n[:, t0:t1].T, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            canode_model.set_input(t_win, u_win_n.unsqueeze(0))
            x_pred_win = canode_model.integrate(x0_n, t_win)
            canode_pred_n[:, t0:t1] = x_pred_win.cpu().numpy().T
            
    # Un-normalize CA-NODE predictions to raw space for comparison
    canode_pred = canode_pred_n * x_std + x_mean
    
    # Compute R2 and MSE over the plotted window (1.0 second)
    n4_mse = np.mean((n4_pred - x_test[:, :T_plot]) ** 2)
    canode_mse = np.mean((canode_pred - x_test[:, :T_plot]) ** 2)
    var = np.var(x_test[:, :T_plot])
    n4_r2 = 1 - n4_mse / var
    canode_r2 = 1 - canode_mse / var
    
    print(f"Comparison (1.0s windowed):")
    print(f"  N4SID:   MSE={n4_mse:.2f}, R2={n4_r2:.4f}")
    print(f"  CA-NODE: MSE={canode_mse:.2f}, R2={canode_r2:.4f}")
    
    # 5. Plot
    n_plot_ch = 5
    t_plot_sec = np.arange(T_plot) * dt
    fig, axes = plt.subplots(n_plot_ch, 1, figsize=(14, 2.5 * n_plot_ch), sharex=True)
    
    # Select channels with high variance for visualization
    vars = np.var(x_test[:, :T_plot], axis=1)
    top_channels = np.argsort(vars)[::-1][:n_plot_ch]
    
    for idx, ch in enumerate(top_channels):
        ax = axes[idx] if n_plot_ch > 1 else axes
        # Ground Truth
        ax.plot(t_plot_sec, x_test[ch, :T_plot], "k", alpha=0.8, lw=1.2, label="Ground truth")
        # N4SID
        ax.plot(t_plot_sec, n4_pred[ch, :], "b--", alpha=0.8, lw=1.0, label="N4SID")
        # CA-NODE
        ax.plot(t_plot_sec, canode_pred[ch, :], "r", alpha=0.8, lw=1.0, label="CA-NODE")
        
        # Draw window boundaries
        for w in range(1, n_plot_windows):
            ax.axvline(w * horizon * dt, color="gray", alpha=0.3, ls="--", lw=0.8)
            
        ax.set_ylabel(f"Ch {ch} (Hz)")
        if idx == 0:
            ax.legend(loc="upper right", fontsize=10)
            
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Digital Twin Model Comparison on test trial (200ms horizon)\n"
                 f"N4SID R\u00b2 = {n4_r2:.3f} | CA-NODE R\u00b2 = {canode_r2:.3f}", fontsize=14)
    fig.tight_layout()
    
    out_img = os.path.join(output_dir, "comparison_prediction.png")
    fig.savefig(out_img, dpi=150)
    print(f"Saved side-by-side comparison plot to {out_img}")


if __name__ == "__main__":
    main()
