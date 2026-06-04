#!/usr/bin/env python
"""Hybrid Distillation: Train causal Latent CA-NODE with combined loss.

Experiment 8: Both encoder AND ODE+decoder are trainable, with a combined loss:
  Loss = alpha * MSE(z0_student, z0_teacher) + (1 - alpha) * MSE(x_hat, x_true)

The teacher z0 comes from a frozen acausal (bidirectional) encoder that sees
the FULL window. The student is a causal encoder that only sees the past.
Unlike Experiment 7 (encoder-only distillation), here the ODE and decoder
can also adapt during training.
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
torch.set_num_threads(4)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modeling.data import load_trials_h5
from modeling.models.canode import TrajectoryWindowDataset
from modeling.models.latent_canode import LatentControlAffineODE
from modeling.models.latent_node import LatentNeuralODE
from modeling.config import DEFAULT_SEED, DEFAULT_TEST_TRIALS, set_seed


def load_teacher(checkpoint_path: str, device: str) -> LatentNeuralODE:
    """Load the frozen acausal teacher model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    teacher = LatentNeuralODE(
        n_x=ckpt["n_x"],
        n_u=ckpt["n_u"],
        z_dim=ckpt["z_dim"],
        hidden_dim=ckpt["hidden"],
        n_layers=ckpt["n_layers"],
    )
    teacher.load_state_dict(ckpt["model_state"])
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher, ckpt


def train_hybrid_distill(
    student: LatentControlAffineODE,
    teacher: LatentNeuralODE,
    x: np.ndarray,
    u: np.ndarray,
    dt: float,
    alpha: float = 0.5,
    n_epochs: int = 300,
    lr: float = 3e-4,
    batch_size: int = 64,
    past_window: int = 200,
    future_window: int = 200,
    stride: int = 100,
    val_fraction: float = 0.1,
    device: str = "cuda",
    verbose: bool = True,
    method: str = "dopri5",
    kl_weight: float = 1.0,
    seed: int = DEFAULT_SEED,
    weight_decay: float = 1e-5,
) -> dict:
    """Train causal student with hybrid distillation + reconstruction loss."""
    student = student.to(device)

    window_size = past_window + future_window
    dataset = TrajectoryWindowDataset(x, u, dt, window_size, stride)
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            pin_memory=True)

    # All student params are trainable
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = {
        "train_loss": [], "val_loss": [],
        "train_recon": [], "train_distill": [], "train_kl": [],
    }

    if verbose:
        print(f"  Dataset: {len(dataset)} windows, {n_train} train, {n_val} val")
        print(f"  Batch size: {batch_size}, batches/epoch: {len(train_loader)}")
        print(f"  Alpha (distill weight): {alpha}")

    for epoch in range(n_epochs):
        # --- Training ---
        student.train()
        train_losses, recon_losses, distill_losses, kl_losses = [], [], [], []

        # KL annealing (linear warmup over first 50 epochs)
        current_kl_weight = kl_weight * min(1.0, epoch / 50.0)

        for x_batch, u_batch, t_batch in train_loader:
            x_batch = x_batch.to(device)
            u_batch = u_batch.to(device)

            # Causal split: past_window for encoding, future_window for prediction
            x_past = x_batch[:, :past_window]      # (B, past_window, n_x)
            u_past = u_batch[:, :past_window]       # (B, past_window, n_u)
            x_future = x_batch[:, past_window:]     # (B, future_window, n_x)
            u_future = u_batch[:, past_window:]     # (B, future_window, n_u)
            t_future = t_batch[0, :future_window].to(device)

            n_x = x_batch.shape[2]

            # --- Teacher: acausal encoder sees FULL window ---
            with torch.no_grad():
                # Teacher encoder is bidirectional - feed it the entire window
                z0_teacher_mu, _ = teacher.encoder(x_batch, u_batch)
                z0_teacher = z0_teacher_mu.detach()  # (B, z_dim) - use mean, no sampling

            # --- Student: causal encoder sees only past ---
            mu_student, logvar_student = student.encoder(x_past, u_past)
            z0_student = student.sample_z0(mu_student, logvar_student)  # (B, z_dim)

            # --- Student: integrate ODE forward and decode ---
            student.set_input(t_future, u_future)
            from torchdiffeq import odeint
            z_seq = odeint(student.ode_func, z0_student, t_future,
                           method=method, rtol=1e-4, atol=1e-5)  # (T, B, z_dim)
            z_seq = z_seq.permute(1, 0, 2)  # (B, T, z_dim)
            x_pred_future = student.decoder(z_seq)  # (B, T, n_x)

            # --- Losses ---
            # Reconstruction loss on future predictions
            recon_loss = F.mse_loss(x_pred_future, x_future)

            # Distillation loss: align student z0 with teacher z0
            distill_loss = F.mse_loss(mu_student, z0_teacher)

            # KL divergence (regularization)
            kl_div = -0.5 * torch.mean(torch.sum(
                1 + logvar_student - mu_student.pow(2) - logvar_student.exp(), dim=1
            ))
            kl_scaled = kl_div / (future_window * n_x)

            # Combined loss
            loss = (alpha * distill_loss
                    + (1 - alpha) * recon_loss
                    + current_kl_weight * kl_scaled)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            recon_losses.append(recon_loss.item())
            distill_losses.append(distill_loss.item())
            kl_losses.append(kl_scaled.item())

        avg_train = np.mean(train_losses)
        history["train_loss"].append(avg_train)
        history["train_recon"].append(np.mean(recon_losses))
        history["train_distill"].append(np.mean(distill_losses))
        history["train_kl"].append(np.mean(kl_losses))

        # --- Validation ---
        student.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, u_batch, t_batch in val_loader:
                x_batch = x_batch.to(device)
                u_batch = u_batch.to(device)

                x_past = x_batch[:, :past_window]
                u_past = u_batch[:, :past_window]
                x_future = x_batch[:, past_window:]
                u_future = u_batch[:, past_window:]
                t_future = t_batch[0, :future_window].to(device)

                x_pred_future = student.predict(x_past, u_past, u_future, t_future, method=method)
                val_losses.append(F.mse_loss(x_pred_future, x_future).item())

        avg_val = np.mean(val_losses) if val_losses else float("nan")
        history["val_loss"].append(avg_val)
        scheduler.step()

        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            _lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:4d}/{n_epochs} | "
                  f"train: {avg_train:.6f} (R: {np.mean(recon_losses):.6f}, "
                  f"D: {np.mean(distill_losses):.6f}, KL: {np.mean(kl_losses):.6f}) | "
                  f"val: {avg_val:.6f} | lr: {_lr:.2e}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Hybrid Distillation: Causal CA-NODE")
    parser.add_argument("--data", type=str, default="data/training_trials.h5")
    parser.add_argument("--teacher-ckpt", type=str,
                        default="results/sweep_latent_node_40_10/run_05_z64_h256_l2_lr0.0005_dopri5.pt",
                        help="Path to acausal teacher checkpoint")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for distillation loss (1-alpha for reconstruction)")
    parser.add_argument("--n-epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--z-dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--past-window", type=int, default=200)
    parser.add_argument("--future-window", type=int, default=200)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--n-test-trials", type=int, default=DEFAULT_TEST_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--method", type=str, default="dopri5")
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--output-dir", type=str, default="results/hybrid_distill")
    parser.add_argument("--save-model", type=str, default=None)

    from modeling.wandb_utils import add_wandb_args
    add_wandb_args(parser)
    args = parser.parse_args()

    if args.save_model is None:
        args.save_model = os.path.join(args.output_dir, "model.pt")

    set_seed(args.seed)
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

    # Load teacher
    print(f"\nLoading teacher from: {args.teacher_ckpt}")
    teacher, teacher_ckpt = load_teacher(args.teacher_ckpt, device)
    print(f"  Teacher: z_dim={teacher_ckpt['z_dim']}, hidden={teacher_ckpt['hidden']}, "
          f"layers={teacher_ckpt['n_layers']}")

    # Verify teacher uses same normalization
    # Re-normalize teacher with OUR stats (teacher was trained on same data split)
    print(f"  Teacher n_x={teacher_ckpt['n_x']}, student n_x={n_channels}")

    # Build student
    print(f"\nBuilding Student CA-NODE: n_x={n_channels}, n_u={n_inputs}, z_dim={args.z_dim}, "
          f"hidden={args.hidden}, layers={args.n_layers}")
    student = LatentControlAffineODE(
        n_x=n_channels,
        n_u=n_inputs,
        z_dim=args.z_dim,
        hidden_dim=args.hidden,
        n_layers=args.n_layers,
    )
    n_params = sum(p.numel() for p in student.parameters())
    print(f"  Student Parameters: {n_params:,}")

    # Initialize wandb
    from modeling.wandb_utils import wandb_init, wandb_log, wandb_log_image, wandb_summary, wandb_finish
    wandb_init(args, model=student, script_name="hybrid_distill", extra_config={
        "n_channels": n_channels,
        "n_inputs": n_inputs,
        "n_train_trials": n_train,
        "n_test_trials": n_test,
        "dt": dt,
        "alpha": args.alpha,
        "teacher_ckpt": args.teacher_ckpt,
        "experiment": "hybrid_distillation",
    })

    # Train
    import time as _time
    t_train_start = _time.time()
    history = train_hybrid_distill(
        student, teacher, x_train_n, u_train_n, dt,
        alpha=args.alpha,
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        past_window=args.past_window,
        future_window=args.future_window,
        stride=args.stride,
        device=device,
        method=args.method,
        kl_weight=args.kl_weight,
        seed=args.seed,
        weight_decay=args.weight_decay,
    )
    train_time_s = _time.time() - t_train_start

    # Log epoch-level metrics to wandb
    for epoch_i in range(len(history["train_loss"])):
        wandb_log({
            "train/loss": history["train_loss"][epoch_i],
            "val/loss": history["val_loss"][epoch_i],
            "train/recon_loss": history["train_recon"][epoch_i],
            "train/distill_loss": history["train_distill"][epoch_i],
            "train/kl_loss": history["train_kl"][epoch_i],
            "epoch": epoch_i,
        }, step=epoch_i)

    # Save
    torch.save({
        "model_state": student.state_dict(),
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
        "alpha": args.alpha,
        "teacher_ckpt": args.teacher_ckpt,
    }, args.save_model)
    print(f"\nModel saved to {args.save_model}")

    # Evaluate windowed R2 on test trials
    import time
    print(f"\nEvaluating on test trials ({args.future_window}-step future horizons)...")
    student.eval()
    student = student.to(device)

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
            u_future_win = torch.tensor(u_test_n[:, t_split:t1].T, dtype=torch.float32).unsqueeze(0).to(device)

            x_true_future = x_test_n[:, t_split:t1]

            t_start = time.perf_counter()
            with torch.no_grad():
                x_pred_future = student.predict(
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

    avg_r2 = np.mean(windowed_r2s)
    avg_mse = np.mean(windowed_mses)
    avg_inference = np.mean(inference_times)
    print(f"\n  Avg {horizon}-step: MSE={avg_mse:.4f}, R2={avg_r2:.4f}")
    print(f"  Avg Inference Time (batch=1, horizon={horizon}): {avg_inference:.2f} ms")

    # Log test metrics to wandb
    wandb_summary({
        "test/windowed_r2": float(avg_r2),
        "test/windowed_mse": float(avg_mse),
        "test/avg_inference_ms": float(avg_inference),
        "train/best_train_loss": float(min(history["train_loss"])),
        "train/best_val_loss": float(min(history["val_loss"])),
        "train/time_seconds": train_time_s,
        "model/n_parameters": n_params,
    })

    # Save summary text
    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Hybrid Distillation (Experiment 8)\n")
        f.write(f"==================================\n")
        f.write(f"Alpha: {args.alpha}\n")
        f.write(f"Teacher: {args.teacher_ckpt}\n")
        f.write(f"Student: z_dim={args.z_dim}, hidden={args.hidden}, layers={args.n_layers}\n")
        f.write(f"Training: epochs={args.n_epochs}, lr={args.lr}, past_window={args.past_window}\n")
        f.write(f"Train time: {train_time_s:.1f}s\n")
        f.write(f"\nResults ({horizon}-step windowed):\n")
        f.write(f"  R2:  {avg_r2:.4f}\n")
        f.write(f"  MSE: {avg_mse:.4f}\n")
        f.write(f"  Inference: {avg_inference:.2f} ms\n")
        for ti, (r2, mse) in enumerate(zip(windowed_r2s, windowed_mses)):
            f.write(f"  Trial {n_train + ti}: R2={r2:.4f}, MSE={mse:.4f}\n")

    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].semilogy(history["train_recon"], label="Train Recon")
    axes[0].semilogy(history["val_loss"], label="Val Recon")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Reconstruction Loss")
    axes[0].legend()

    axes[1].semilogy(history["train_distill"], label="Train Distill")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")
    axes[1].set_title(f"Distillation Loss (alpha={args.alpha})")
    axes[1].legend()

    axes[2].semilogy(history["train_loss"], label="Train Total")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("Total Loss")
    axes[2].legend()

    fig.suptitle(f"Hybrid Distillation alpha={args.alpha}")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "training_curve.png"), dpi=150)

    wandb_log_image("plots/training_curve",
                    os.path.join(args.output_dir, "training_curve.png"),
                    caption=f"Hybrid Distillation Training Curve (alpha={args.alpha})")

    wandb_finish()
    print("Done.")


if __name__ == "__main__":
    main()
