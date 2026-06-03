#!/usr/bin/env python
"""Train Latent Control-Affine Neural ODE on generated Cleo data.

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
from modeling.models.latent_canode import LatentControlAffineODE


def train_latent_canode(
    model: LatentControlAffineODE,
    x: np.ndarray,
    u: np.ndarray,
    dt: float,
    n_epochs: int = 500,
    lr: float = 1e-3,
    batch_size: int = 64,
    past_window: int = 100,
    future_window: int = 200,
    stride: int = 100,
    val_fraction: float = 0.1,
    device: str = "cuda",
    verbose: bool = True,
    method: str = "dopri5",
    kl_weight: float = 1.0,
) -> dict:
    model = model.to(device)

    window_size = past_window + future_window
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
            
            # Causal split
            x_past = x_batch[:, :past_window]
            u_past = u_batch[:, :past_window]
            x_future = x_batch[:, past_window:]
            u_future = u_batch[:, past_window:]
            t_future = t_batch[0, :future_window].to(device)  # Reset time origin to 0 for future integration
            
            n_x = x_batch.shape[2]

            x_pred_future, mu, logvar = model(x_past, u_past, u_future, t_future, method=method, deterministic=False)

            recon_loss = F.mse_loss(x_pred_future, x_future)
            
            # KL divergence
            kl_div = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            kl_scaled = kl_div / (future_window * n_x)

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
                # Causal split
                x_past = x_batch[:, :past_window]
                u_past = u_batch[:, :past_window]
                x_future = x_batch[:, past_window:]
                u_future = u_batch[:, past_window:]
                t_future = t_batch[0, :future_window].to(device)

                x_pred_future = model.predict(x_past, u_past, u_future, t_future, method=method)
                val_losses.append(F.mse_loss(x_pred_future, x_future).item())

        avg_val = np.mean(val_losses) if val_losses else float("nan")
        history["val_loss"].append(avg_val)
        scheduler.step()

        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            print(f"  Epoch {epoch:4d}/{n_epochs} | "
                  f"train: {avg_train:.6f} (R: {np.mean(recon_losses):.6f}, KL: {np.mean(kl_losses):.6f}) | "
                  f"val: {avg_val:.6f} | lr: {optimizer.param_groups[0]['lr']:.2e}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train Latent CA-NODE")
    parser.add_argument("--data", type=str, default="data/training_trials.h5")
    parser.add_argument("--n-epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--z-dim", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--past-window", type=int, default=100)
    parser.add_argument("--future-window", type=int, default=200)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--n-test-trials", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--method", type=str, default="dopri5")
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default="results/latent_canode")
    parser.add_argument("--save-model", type=str, default="results/latent_canode/model.pt")

    from modeling.wandb_utils import add_wandb_args
    add_wandb_args(parser)
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
    print(f"\nBuilding Latent CA-NODE: n_x={n_channels}, n_u={n_inputs}, z_dim={args.z_dim}, "
          f"hidden={args.hidden}, layers={args.n_layers}")
    model = LatentControlAffineODE(
        n_x=n_channels,
        n_u=n_inputs,
        z_dim=args.z_dim,
        hidden_dim=args.hidden,
        n_layers=args.n_layers,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Initialize wandb
    from modeling.wandb_utils import wandb_init, wandb_log, wandb_log_image, wandb_summary, wandb_finish
    wandb_init(args, model=model, script_name="fit_latent_canode", extra_config={
        "n_channels": n_channels,
        "n_inputs": n_inputs,
        "n_train_trials": n_train,
        "n_test_trials": n_test,
        "dt": dt,
    })

    # Train
    import time as _time
    t_train_start = _time.time()
    history = train_latent_canode(
        model, x_train_n, u_train_n, dt,
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        past_window=args.past_window,
        future_window=args.future_window,
        stride=args.stride,
        device=device,
        method=args.method,
        kl_weight=args.kl_weight,
    )
    train_time_s = _time.time() - t_train_start

    # Log epoch-level metrics to wandb
    for epoch_i in range(len(history["train_loss"])):
        wandb_log({
            "train/loss": history["train_loss"][epoch_i],
            "val/loss": history["val_loss"][epoch_i],
            "train/recon_loss": history["train_recon"][epoch_i],
            "train/kl_loss": history["train_kl"][epoch_i],
            "epoch": epoch_i,
        }, step=epoch_i)

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
    import time
    print(f"\nEvaluating on test trials ({args.future_window}-step future horizons)...")
    model.eval()
    model = model.to(device)
    
    horizon = args.future_window
    windowed_r2s = []
    windowed_mses = []
    inference_times = []
    
    for ti in range(n_test):
        x_test = x_test_trials[ti]
        u_test = u_test_trials[ti]
        x_test_n = (x_test - x_mean) / x_std
        u_test_n = (u_test - u_mean) / u_std

        n_ch_val, T_val = x_test_n.shape
        n_windows = (T_val - args.past_window - args.future_window) // args.stride + 1
        window_mses = []
        t_future = torch.arange(args.future_window, dtype=torch.float32).to(device) * dt

        for w in range(n_windows):
            t0 = w * args.stride
            t_split = t0 + args.past_window
            t1 = t_split + args.future_window
            
            x_past_win = torch.tensor(x_test_n[:, t0:t_split].T, dtype=torch.float32).unsqueeze(0).to(device)
            u_past_win = torch.tensor(u_test_n[:, t0:t_split].T, dtype=torch.float32).unsqueeze(0).to(device)
            
            x_future_win = torch.tensor(x_test_n[:, t_split:t1].T, dtype=torch.float32).unsqueeze(0).to(device)
            u_future_win = torch.tensor(u_test_n[:, t_split:t1].T, dtype=torch.float32).unsqueeze(0).to(device)
            
            x_true_future = x_test_n[:, t_split:t1]

            t_start = time.perf_counter()
            with torch.no_grad():
                x_pred_future = model.predict(
                    x_past_win, u_past_win, u_future_win, t_future, method=args.method
                )
                x_pred_future_np = x_pred_future[0].cpu().numpy().T
            t_end = time.perf_counter()
            inference_times.append((t_end - t_start) * 1000.0)
            
            window_mses.append(np.mean((x_pred_future_np - x_true_future) ** 2))

        trial_mse = np.mean(window_mses)
        trial_var = np.var(x_test_n)
        trial_r2 = 1 - trial_mse / trial_var
        windowed_mses.append(trial_mse)
        windowed_r2s.append(trial_r2)
        print(f"  Trial {n_train + ti}: MSE={trial_mse:.4f}, R2={trial_r2:.4f}")
    
    print(f"  Avg {horizon}-step: MSE={np.mean(windowed_mses):.4f}, R2={np.mean(windowed_r2s):.4f}")
    print(f"  Avg Inference Time (batch=1, horizon={horizon}): {np.mean(inference_times):.2f} ms")

    # Log test metrics to wandb
    wandb_summary({
        "test/windowed_r2": float(np.mean(windowed_r2s)),
        "test/windowed_mse": float(np.mean(windowed_mses)),
        "test/avg_inference_ms": float(np.mean(inference_times)),
        "train/best_train_loss": float(min(history["train_loss"])),
        "train/best_val_loss": float(min(history["val_loss"])),
        "train/time_seconds": train_time_s,
        "model/n_parameters": n_params,
    })

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.semilogy(history["train_recon"], label="Train Recon")
    ax.semilogy(history["val_loss"], label="Val Recon")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    fig.savefig(os.path.join(args.output_dir, "training_curve.png"), dpi=150)

    # Log plot to wandb
    wandb_log_image("plots/training_curve",
                    os.path.join(args.output_dir, "training_curve.png"),
                    caption="Latent CA-NODE Training Curve")

    wandb_finish()
    print("Done.")

if __name__ == "__main__":
    main()
