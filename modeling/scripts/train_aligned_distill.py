#!/usr/bin/env python3
"""Aligned Distillation: Train causal encoder with alpha-scheduled joint loss.

Anchors the causal model to the acausal latent space via:
  1. Decoder initialization from acausal model (exact weight transfer)
  2. Alpha-scheduled loss: L = alpha * ||z_causal - z_acausal||^2 + (1-alpha) * MSE(x_hat, x_true)
  3. Alpha starts high (force alignment) and decays (let model specialize)

The causal model uses the same z_dim=64 as the acausal model but builds its own
control-affine ODE (f, g separate from acausal's combined ode_func).

Usage:
    python -m modeling.scripts.train_aligned_distill \
        --alpha-schedule cosine --alpha-init 0.9 --alpha-final 0.1 \
        --hidden 256 --freeze-decoder-epochs 50 \
        --output-dir results/aligned_distill_cosine_0.9 \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modeling.models.latent_canode import LatentControlAffineODE
from modeling.models.latent_node import LatentNeuralODE


def get_alpha(epoch, n_epochs, schedule, alpha_init, alpha_final):
    """Compute alpha for the current epoch."""
    if schedule == "constant":
        return alpha_init
    elif schedule == "cosine":
        # Cosine decay from alpha_init to alpha_final
        progress = epoch / max(n_epochs - 1, 1)
        return alpha_final + 0.5 * (alpha_init - alpha_final) * (1 + math.cos(math.pi * progress))
    elif schedule == "step":
        # Step decay: alpha_init for first half, alpha_final for second half
        if epoch < n_epochs // 2:
            return alpha_init
        else:
            return alpha_final
    elif schedule == "linear":
        progress = epoch / max(n_epochs - 1, 1)
        return alpha_init + (alpha_final - alpha_init) * progress
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


def load_data(data_path, n_test_trials=10, seed=42, past_window=200, future_window=200):
    """Load and prepare data."""
    with h5py.File(data_path, "r") as f:
        x_all = f["x"][:]  # (n_trials, n_channels, n_steps)
        u_all = f["u"][:]  # (n_trials, n_inputs, n_steps)

    n_trials = x_all.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_trials)
    test_idx = perm[:n_test_trials]
    train_idx = perm[n_test_trials:]

    x_train = x_all[train_idx]
    u_train = u_all[train_idx]
    x_test = x_all[test_idx]
    u_test = u_all[test_idx]

    # Compute normalization stats from training data
    x_mean = x_train.mean(axis=(0, 2), keepdims=True)  # (1, n_ch, 1)
    x_std = x_train.std(axis=(0, 2), keepdims=True) + 1e-8
    u_mean = u_train.mean(axis=(0, 2), keepdims=True)
    u_std = u_train.std(axis=(0, 2), keepdims=True) + 1e-8

    # Normalize
    x_train_n = (x_train - x_mean) / x_std
    u_train_n = (u_train - u_mean) / u_std
    x_test_n = (x_test - x_mean) / x_std
    u_test_n = (u_test - u_mean) / u_std

    # Extract windows
    W = past_window + future_window
    windows_x, windows_u = [], []
    for trial in range(x_train_n.shape[0]):
        n_steps = x_train_n.shape[2]
        for t in range(0, n_steps - W, past_window):
            windows_x.append(x_train_n[trial, :, t : t + W].T)  # (W, n_ch)
            windows_u.append(u_train_n[trial, :, t : t + W].T)  # (W, n_u)

    windows_x = np.array(windows_x, dtype=np.float32)
    windows_u = np.array(windows_u, dtype=np.float32)

    stats = {
        "x_mean": x_mean.squeeze(-1).T,  # (n_ch, 1)
        "x_std": x_std.squeeze(-1).T,
        "u_mean": u_mean.squeeze(-1).T,
        "u_std": u_std.squeeze(-1).T,
    }

    return windows_x, windows_u, x_test_n, u_test_n, stats


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print(f"Aligned Distillation: schedule={args.alpha_schedule}, "
          f"alpha={args.alpha_init}->{args.alpha_final}, h={args.hidden}")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────
    print("\nLoading data...")
    data_path = args.data
    windows_x, windows_u, x_test_n, u_test_n, stats = load_data(
        data_path, n_test_trials=args.n_test_trials, seed=args.seed,
        past_window=args.past_window, future_window=args.future_window,
    )
    n_x = windows_x.shape[2]
    n_u = windows_u.shape[2]
    print(f"  Windows: {windows_x.shape[0]}, n_x={n_x}, n_u={n_u}")

    # ── Load acausal teacher ───────────────────────────────────────────
    print("\nLoading acausal teacher...")
    acausal_ckpt = torch.load(args.acausal_checkpoint, map_location=device, weights_only=False)
    acausal_model = LatentNeuralODE(
        n_x=acausal_ckpt["n_x"],
        n_u=acausal_ckpt["n_u"],
        z_dim=acausal_ckpt["z_dim"],
        hidden_dim=acausal_ckpt["hidden"],
        n_layers=acausal_ckpt["n_layers"],
    ).to(device)
    acausal_model.load_state_dict(acausal_ckpt["model_state"])
    acausal_model.eval()
    for p in acausal_model.parameters():
        p.requires_grad_(False)
    z_dim = acausal_ckpt["z_dim"]
    print(f"  Acausal: z={z_dim}, h={acausal_ckpt['hidden']}, n_layers={acausal_ckpt['n_layers']}")

    # ── Build causal student ───────────────────────────────────────────
    print("\nBuilding causal student...")
    causal_model = LatentControlAffineODE(
        n_x=n_x,
        n_u=n_u,
        z_dim=z_dim,
        hidden_dim=args.hidden,
        n_layers=args.n_layers,
    ).to(device)

    # Initialize decoder from acausal model (exact weight transfer)
    acausal_decoder_w = acausal_ckpt["model_state"]["decoder.weight"]
    acausal_decoder_b = acausal_ckpt["model_state"]["decoder.bias"]
    causal_model.decoder.weight.data.copy_(acausal_decoder_w)
    causal_model.decoder.bias.data.copy_(acausal_decoder_b)
    print(f"  Decoder initialized from acausal model")
    print(f"  Student: z={z_dim}, h={args.hidden}, n_layers={args.n_layers}")

    # ── Normalize using acausal model's stats for teacher ──────────────
    # The acausal model has its own normalization; we use the same data norm
    # for both since we normalized the data consistently above.
    # The teacher's z0 targets come from the acausal encoder applied to the
    # FULL window (past+future), while the student only sees the past.

    # ── Pre-extract acausal z0 targets ─────────────────────────────────
    print("\nPre-extracting acausal z0 targets...")
    all_x = torch.tensor(windows_x, dtype=torch.float32, device=device)
    all_u = torch.tensor(windows_u, dtype=torch.float32, device=device)

    # Acausal encoder sees FULL window (past + future)
    pw = args.past_window
    fw = args.future_window
    z0_targets = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(all_x), batch_size):
            x_batch = all_x[i : i + batch_size]
            u_batch = all_u[i : i + batch_size]
            # Acausal encoder: pass full window
            mu, logvar = acausal_model.encoder(x_batch, u_batch)
            z0_targets.append(mu.cpu())
    z0_targets = torch.cat(z0_targets, dim=0)
    print(f"  z0 targets shape: {z0_targets.shape}")

    # ── Prepare dataset ────────────────────────────────────────────────
    # Split windows into train/val
    n_total = len(windows_x)
    n_val = max(1, n_total // 10)
    n_train = n_total - n_val

    x_past_all = all_x[:, :pw, :]  # (N, pw, n_x)
    u_past_all = all_u[:, :pw, :]
    x_future_all = all_x[:, pw:, :]  # (N, fw, n_x)
    u_future_all = all_u[:, pw:, :]

    train_dataset = TensorDataset(
        x_past_all[:n_train].cpu(), u_past_all[:n_train].cpu(),
        x_future_all[:n_train].cpu(), u_future_all[:n_train].cpu(),
        z0_targets[:n_train],
    )
    val_dataset = TensorDataset(
        x_past_all[n_train:].cpu(), u_past_all[n_train:].cpu(),
        x_future_all[n_train:].cpu(), u_future_all[n_train:].cpu(),
        z0_targets[n_train:],
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"  Train: {n_train}, Val: {n_val}")

    # ── Optimizer ──────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(causal_model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    # ── Training loop ──────────────────────────────────────────────────
    print("\nTraining...")
    history = {
        "train_loss": [], "val_loss": [],
        "train_distill": [], "train_recon": [],
        "alpha": [], "lr": [],
    }

    dt_tensor = torch.linspace(0, fw * 1e-3, fw, device=device)
    best_val = float("inf")

    for epoch in range(args.n_epochs):
        alpha = get_alpha(epoch, args.n_epochs, args.alpha_schedule,
                          args.alpha_init, args.alpha_final)
        history["alpha"].append(alpha)

        # Optionally freeze decoder for initial epochs
        if epoch < args.freeze_decoder_epochs:
            causal_model.decoder.weight.requires_grad_(False)
            causal_model.decoder.bias.requires_grad_(False)
        else:
            causal_model.decoder.weight.requires_grad_(True)
            causal_model.decoder.bias.requires_grad_(True)

        # Train
        causal_model.train()
        epoch_loss, epoch_distill, epoch_recon, n_batches = 0, 0, 0, 0

        for x_past, u_past, x_future, u_future, z0_target in train_loader:
            x_past = x_past.to(device)
            u_past = u_past.to(device)
            x_future = x_future.to(device)
            u_future = u_future.to(device)
            z0_target = z0_target.to(device)

            optimizer.zero_grad()

            # Causal encoder: only sees past
            x_pred, mu, logvar = causal_model.forward(
                x_past, u_past, u_future, dt_tensor, method="euler", deterministic=True
            )

            # Distillation loss: match acausal z0
            distill_loss = nn.functional.mse_loss(mu, z0_target)

            # Reconstruction loss: predict future
            recon_loss = nn.functional.mse_loss(x_pred, x_future)

            # Combined loss
            loss = alpha * distill_loss + (1 - alpha) * recon_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(causal_model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_distill += distill_loss.item()
            epoch_recon += recon_loss.item()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / n_batches
        avg_distill = epoch_distill / n_batches
        avg_recon = epoch_recon / n_batches
        history["train_loss"].append(avg_loss)
        history["train_distill"].append(avg_distill)
        history["train_recon"].append(avg_recon)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # Validate
        causal_model.eval()
        val_loss_sum, val_n = 0, 0
        with torch.no_grad():
            for x_past, u_past, x_future, u_future, z0_target in val_loader:
                x_past = x_past.to(device)
                u_past = u_past.to(device)
                x_future = x_future.to(device)
                u_future = u_future.to(device)
                z0_target = z0_target.to(device)

                x_pred, mu, logvar = causal_model.forward(
                    x_past, u_past, u_future, dt_tensor, method="euler", deterministic=True
                )
                distill_loss = nn.functional.mse_loss(mu, z0_target)
                recon_loss = nn.functional.mse_loss(x_pred, x_future)
                loss = alpha * distill_loss + (1 - alpha) * recon_loss
                val_loss_sum += loss.item()
                val_n += 1

        val_loss = val_loss_sum / val_n if val_n > 0 else float("inf")
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": causal_model.state_dict(),
                "n_x": n_x, "n_u": n_u, "z_dim": z_dim,
                "hidden": args.hidden, "n_layers": args.n_layers,
                "x_mean": stats["x_mean"], "x_std": stats["x_std"],
                "u_mean": stats["u_mean"], "u_std": stats["u_std"],
                "history": history,
                "alpha_schedule": args.alpha_schedule,
                "alpha_init": args.alpha_init,
                "alpha_final": args.alpha_final,
            }, os.path.join(args.output_dir, "best_model.pt"))

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:>4d}/{args.n_epochs} | "
                  f"alpha={alpha:.3f} | "
                  f"distill={avg_distill:.5f} | recon={avg_recon:.5f} | "
                  f"val={val_loss:.5f} | lr={optimizer.param_groups[0]['lr']:.2e}")

    # ── Save final model ───────────────────────────────────────────────
    torch.save({
        "model_state": causal_model.state_dict(),
        "n_x": n_x, "n_u": n_u, "z_dim": z_dim,
        "hidden": args.hidden, "n_layers": args.n_layers,
        "x_mean": stats["x_mean"], "x_std": stats["x_std"],
        "u_mean": stats["u_mean"], "u_std": stats["u_std"],
        "history": history,
        "alpha_schedule": args.alpha_schedule,
        "alpha_init": args.alpha_init,
        "alpha_final": args.alpha_final,
    }, os.path.join(args.output_dir, "model.pt"))

    # ── Evaluate best model ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Evaluating best model (dopri5, 200-step windows)")
    print("=" * 60)

    best_ckpt = torch.load(os.path.join(args.output_dir, "best_model.pt"),
                            map_location=device, weights_only=False)
    causal_model.load_state_dict(best_ckpt["model_state"])
    causal_model.eval()

    from torchdiffeq import odeint

    pw = args.past_window
    fw = args.future_window
    dt_eval = torch.linspace(0, fw * 1e-3, fw, device=device)

    all_preds, all_trues = [], []
    with torch.no_grad():
        for trial in range(x_test_n.shape[0]):
            x_trial = torch.tensor(x_test_n[trial].T, dtype=torch.float32, device=device)
            u_trial = torch.tensor(u_test_n[trial].T, dtype=torch.float32, device=device)

            n_steps = x_trial.shape[0]
            for t in range(0, n_steps - pw - fw, pw):
                x_past = x_trial[t : t + pw].unsqueeze(0)
                u_past = u_trial[t : t + pw].unsqueeze(0)
                x_future = x_trial[t + pw : t + pw + fw]
                u_future = u_trial[t + pw : t + pw + fw].unsqueeze(0)

                x_pred = causal_model.predict(x_past, u_past, u_future, dt_eval, method="dopri5")
                all_preds.append(x_pred.squeeze(0).cpu().numpy())
                all_trues.append(x_future.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)

    ss_res = np.sum((all_trues - all_preds) ** 2)
    ss_tot = np.sum((all_trues - all_trues.mean(axis=0)) ** 2)
    r2 = 1 - ss_res / ss_tot
    mse = np.mean((all_trues - all_preds) ** 2)

    print(f"\n  R² = {r2:.4f}")
    print(f"  MSE = {mse:.6f}")

    # Also compute z0 alignment quality
    z0_cos_sims = []
    with torch.no_grad():
        for i in range(0, len(all_x), batch_size):
            x_batch = all_x[i : i + batch_size]
            u_batch = all_u[i : i + batch_size]
            mu_causal, _ = causal_model.encoder(x_batch[:, :pw, :].to(device), u_batch[:, :pw, :].to(device))
            z0_t = z0_targets[i : i + batch_size].to(device)
            cos_sim = nn.functional.cosine_similarity(mu_causal, z0_t, dim=1)
            z0_cos_sims.append(cos_sim.cpu().numpy())
    z0_cos_sim = np.concatenate(z0_cos_sims).mean()
    print(f"  z0 cosine similarity (causal vs acausal): {z0_cos_sim:.4f}")

    # Save results
    results = {
        "r2": float(r2),
        "mse": float(mse),
        "z0_cos_sim": float(z0_cos_sim),
        "alpha_schedule": args.alpha_schedule,
        "alpha_init": args.alpha_init,
        "alpha_final": args.alpha_final,
        "hidden": args.hidden,
        "freeze_decoder_epochs": args.freeze_decoder_epochs,
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(history["train_loss"], label="Train")
    axes[0, 0].plot(history["val_loss"], label="Val")
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(history["train_distill"], label="Distill", color="blue")
    axes[0, 1].plot(history["train_recon"], label="Recon", color="red")
    axes[0, 1].set_title("Component Losses")
    axes[0, 1].legend()

    axes[1, 0].plot(history["alpha"])
    axes[1, 0].set_title("Alpha Schedule")
    axes[1, 0].set_ylabel("alpha")

    axes[1, 1].bar(["Aligned\nDistill", "Causal\nBaseline", "Acausal"], [r2, 0.8252, 0.9387])
    axes[1, 1].set_title(f"R² Comparison (ours: {r2:.4f})")
    axes[1, 1].set_ylabel("R²")

    plt.suptitle(f"Aligned Distill: {args.alpha_schedule} alpha={args.alpha_init}->{args.alpha_final}, h={args.hidden}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training_curves.png"), dpi=150)
    plt.close()

    print(f"\nResults saved to {args.output_dir}/")
    print(f"R² = {r2:.4f} (baseline causal: 0.8252, acausal: 0.9387)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/snel/home/cbwash2/cleo/data/training_trials.h5")
    parser.add_argument("--acausal-checkpoint", type=str,
                        default="/snel/home/cbwash2/cleo/results/sweep_latent_node_40_10/run_05_z64_h256_l2_lr0.0005_dopri5.pt")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    # Architecture
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--past-window", type=int, default=200)
    parser.add_argument("--future-window", type=int, default=200)

    # Alpha schedule
    parser.add_argument("--alpha-schedule", type=str, default="cosine",
                        choices=["constant", "cosine", "step", "linear"])
    parser.add_argument("--alpha-init", type=float, default=0.9)
    parser.add_argument("--alpha-final", type=float, default=0.1)

    # Training
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--freeze-decoder-epochs", type=int, default=50,
                        help="Freeze decoder for this many epochs to anchor latent space")
    parser.add_argument("--n-test-trials", type=int, default=10)

    args = parser.parse_args()
    main(args)
