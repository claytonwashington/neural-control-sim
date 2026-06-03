#!/usr/bin/env python
"""Train Latent Neural ODE on generated Cleo data.

Uses trial-level train/test split: last 2 trials held out for testing.
"""

import argparse
import os

# Limit CPU threads to avoid contention
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
torch.set_num_threads(4)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modeling.data import load_trials_h5
from modeling.models.canode import TrajectoryWindowDataset
from modeling.models.latent_node import LatentNeuralODE


def train_latent_node(
    model: LatentNeuralODE,
    x: np.ndarray,
    u: np.ndarray,
    dt: float,
    n_epochs: int = 500,
    lr: float = 1e-3,
    batch_size: int = 64,
    window_size: int = 200,
    stride: int = 100,
    val_fraction: float = 0.1,
    device: str = "cuda",
    verbose: bool = True,
    method: str = "dopri5",
    kl_weight: float = 1.0,
) -> dict:
    model = model.to(device)

    dataset = TrajectoryWindowDataset(x, u, dt, window_size, stride)
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = {"train_loss": [], "val_loss": [], "train_recon": [], "train_kl": []}

    if verbose:
        print(f"  Dataset: {len(dataset)} windows, {n_train} train, {n_val} val")
        print(f"  Batch size: {batch_size}, batches/epoch: {len(train_loader)}")

    for epoch in range(n_epochs):
        # --- Training ---
        model.train()
        train_losses, recon_losses, kl_losses = [], [], []
        
        # KL annealing (optional, using linear warmup over first 50 epochs)
        current_kl_weight = kl_weight * min(1.0, epoch / 50.0)

        for x_batch, u_batch, t_batch in train_loader:
            x_batch = x_batch.to(device)
            u_batch = u_batch.to(device)
            t_vec = t_batch[0].to(device)
            
            T = x_batch.shape[1]
            n_x = x_batch.shape[2]

            x_pred, mu, logvar = model(x_batch, u_batch, t_vec, method=method, deterministic=False)

            recon_loss = F.mse_loss(x_pred, x_batch)
            
            # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
            # Average over batch, sum over latent dim
            kl_div = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            # Scale KL down to match MSE magnitude (which is averaged over T and n_x)
            kl_scaled = kl_div / (T * n_x)

            loss = recon_loss + current_kl_weight * kl_scaled
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            recon_losses.append(recon_loss.item())
            kl_losses.append(kl_scaled.item())

        avg_train = np.mean(train_losses)
        history["train_loss"].append(avg_train)
        history["train_recon"].append(np.mean(recon_losses))
        history["train_kl"].append(np.mean(kl_losses))

        # --- Validation ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, u_batch, t_batch in val_loader:
                x_batch = x_batch.to(device)
                u_batch = u_batch.to(device)
                t_vec = t_batch[0].to(device)

                # For validation, we typically use deterministic prediction
                x_pred = model.predict(x_batch, u_batch, t_vec, method=method)
                val_losses.append(F.mse_loss(x_pred, x_batch).item())

        avg_val = np.mean(val_losses) if val_losses else float("nan")
        history["val_loss"].append(avg_val)
        scheduler.step()

        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            print(f"  Epoch {epoch:4d}/{n_epochs} | "
                  f"train: {avg_train:.6f} (R: {np.mean(recon_losses):.6f}, KL: {np.mean(kl_losses):.6f}) | "
                  f"val: {avg_val:.6f} | lr: {optimizer.param_groups[0]['lr']:.2e}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train Latent Neural ODE")
    parser.add_argument("--data", type=str, default="data/training_trials.h5")
    parser.add_argument("--n-epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--z-dim", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--n-test-trials", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--method", type=str, default="dopri5")
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default="results/latent_node")
    parser.add_argument("--save-model", type=str, default="results/latent_node/model.pt")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    data = load_trials_h5(args.data)
    x_all, u_all = data["x"], data["u"]
    dt = float(data["dt"])
    n_trials, n_channels, n_steps = x_all.shape
    n_inputs = u_all.shape[1]
    
    # Train/test split
    n_test = args.n_test_trials
    n_train = n_trials - n_test
    x_train_trials = x_all[:n_train]
    u_train_trials = u_all[:n_train]
    x_test_trials = x_all[n_train:]
    u_test_trials = u_all[n_train:]

    x_train = np.concatenate([x_train_trials[i] for i in range(n_train)], axis=1)
    u_train = np.concatenate([u_train_trials[i] for i in range(n_train)], axis=1)

    # Normalize
    x_mean = x_train.mean(axis=1, keepdims=True)
    x_std = x_train.std(axis=1, keepdims=True) + 1e-8
    u_mean = u_train.mean(axis=1, keepdims=True)
    u_std = u_train.std(axis=1, keepdims=True) + 1e-8

    x_train_n = (x_train - x_mean) / x_std
    u_train_n = (u_train - u_mean) / u_std

    # Build model
    print(f"\nBuilding Latent NODE: n_x={n_channels}, n_u={n_inputs}, z_dim={args.z_dim}, "
          f"hidden={args.hidden}, layers={args.n_layers}")
    model = LatentNeuralODE(
        n_x=n_channels,
        n_u=n_inputs,
        z_dim=args.z_dim,
        hidden_dim=args.hidden,
        n_layers=args.n_layers,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Train
    history = train_latent_node(
        model, x_train_n, u_train_n, dt,
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        window_size=args.window_size,
        stride=args.stride,
        device=device,
        method=args.method,
        kl_weight=args.kl_weight,
    )

    # Save
    torch.save({
        "model_state": model.state_dict(),
        "n_x": n_channels,
        "n_u": n_inputs,
        "z_dim": args.z_dim,
        "hidden": args.hidden,
        "n_layers": args.n_layers,
        "x_mean": x_mean,
        "x_std": x_std,
        "u_mean": u_mean,
        "u_std": u_std,
        "history": history,
    }, args.save_model)

    # Evaluate windowed
    print("\nEvaluating on test trials (200-step windows)...")
    model.eval()
    model = model.to(device)
    
    horizon = 200
    windowed_r2s = []
    windowed_mses = []
    
    for ti in range(n_test):
        x_test = x_test_trials[ti]
        u_test = u_test_trials[ti]
        x_test_n = (x_test - x_mean) / x_std
        u_test_n = (u_test - u_mean) / u_std

        n_ch_val, T_val = x_test_n.shape
        n_windows = (T_val - horizon) // horizon
        window_mses = []
        t_win = torch.arange(horizon, dtype=torch.float32).to(device) * dt

        for w in range(n_windows):
            t0 = w * horizon
            t1 = t0 + horizon
            x_win = torch.tensor(x_test_n[:, t0:t1].T, dtype=torch.float32).to(device)
            u_win = torch.tensor(u_test_n[:, t0:t1].T, dtype=torch.float32).to(device)
            x_true_win = x_test_n[:, t0:t1]

            with torch.no_grad():
                x_pred_win = model.predict(
                    x_win.unsqueeze(0), u_win.unsqueeze(0), t_win, method=args.method
                )
                x_pred_win_np = x_pred_win[0].cpu().numpy().T
            window_mses.append(np.mean((x_pred_win_np - x_true_win) ** 2))

        trial_mse = np.mean(window_mses)
        trial_var = np.var(x_test_n)
        trial_r2 = 1 - trial_mse / trial_var
        windowed_mses.append(trial_mse)
        windowed_r2s.append(trial_r2)
        print(f"  Trial {n_train + ti}: MSE={trial_mse:.4f}, R2={trial_r2:.4f}")
    
    print(f"  Avg {horizon}-step: MSE={np.mean(windowed_mses):.4f}, R2={np.mean(windowed_r2s):.4f}")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.semilogy(history["train_recon"], label="Train Recon")
    ax.semilogy(history["val_loss"], label="Val Recon")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    fig.savefig(os.path.join(args.output_dir, "training_curve.png"), dpi=150)
    print("Done.")

if __name__ == "__main__":
    main()
