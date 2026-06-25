#!/usr/bin/env python3
"""Train Latent CA-NODE on spike-sorted neural data with Poisson NLL.

Uses raw spike counts (no normalization) and Poisson negative log-likelihood
loss, following LFADS conventions. Handles variable neuron counts via masking.

Data lives in placement groups within an HDF5 file:
    placement_<p>/x_sorted:           (n_trials, max_n_sorted, n_bins) int16
    placement_<p>/n_sorted_per_trial:  (n_trials,) int64
    u:                                 (n_trials, 2, n_bins*10) -- 1ms resolution

Usage:
    python -m modeling.scripts.train_spiking_canode_vae \\
        --data data/spiking_plant3.h5 --placement 0 \\
        --z-dim 64 --hidden 256 --lr 5e-4 --epochs 200 \\
        --output-dir results/spiking_canode_poisson_p0/z64_h256_lr5e-4 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modeling.models.latent_canode import LatentControlAffineODE
from modeling.scripts.preflight_check import add_preflight_args, validate_preflight


# ---------------------------------------------------------------------------
# Data loading — raw spike counts, no normalization on x
# ---------------------------------------------------------------------------

BIN_WIDTH_S = 0.01  # 10 ms bins


def load_spiking_data_poisson(
    data_path: str,
    placement: int = 0,
    n_test_trials: int = 10,
    seed: int = 42,
    past_steps: int = 20,
    future_steps: int = 20,
):
    """Load spike-sorted data for Poisson NLL training.

    Unlike the MSE version, spike counts are NOT converted to Hz or normalized.
    Only u (control input) is z-scored.

    Returns
    -------
    windows_x : ndarray (N_windows, W, max_n_sorted) — raw spike counts
    windows_u : ndarray (N_windows, W, 2) — z-scored control input
    masks     : ndarray (N_windows, max_n_sorted) — 1 for real neurons
    x_test, u_test_n, masks_test : test trial arrays
    stats : dict with u normalization params
    max_n_sorted : int
    """
    with h5py.File(data_path, "r") as f:
        grp = f["placement_%d" % placement]
        x_sorted = grp["x_sorted"][:]            # (n_trials, max_n, n_bins) int16
        n_sorted = grp["n_sorted_per_trial"][:]   # (n_trials,)
        u_1ms = f["u"][:]                         # (n_trials, 2, n_bins*10)

    n_trials, max_n, n_bins = x_sorted.shape
    n_u = u_1ms.shape[1]

    # Raw spike counts as float32 (NOT Hz, NOT normalized)
    x_counts = x_sorted.astype(np.float32)

    # Subsample u from 1ms to 10ms (average within each bin)
    u_10ms = u_1ms.reshape(n_trials, n_u, n_bins, 10).mean(axis=-1)

    # Per-trial neuron masks
    neuron_masks = np.zeros((n_trials, max_n), dtype=np.float32)
    for i in range(n_trials):
        neuron_masks[i, :n_sorted[i]] = 1.0

    # Train/test split
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_trials)
    test_idx = perm[:n_test_trials]
    train_idx = perm[n_test_trials:]

    x_train = x_counts[train_idx]
    u_train = u_10ms[train_idx]
    m_train = neuron_masks[train_idx]
    x_test = x_counts[test_idx]
    u_test = u_10ms[test_idx]
    m_test = neuron_masks[test_idx]

    # Normalize u only (not x)
    u_mean = u_train.mean(axis=(0, 2))  # (2,)
    u_std = u_train.std(axis=(0, 2)) + 1e-8
    u_train_n = (u_train - u_mean[None, :, None]) / u_std[None, :, None]
    u_test_n = (u_test - u_mean[None, :, None]) / u_std[None, :, None]

    # Zero out padded neurons in count data (they're already 0 but be safe)
    x_train *= m_train[:, :, None]
    x_test *= m_test[:, :, None]

    # Extract sliding windows
    W = past_steps + future_steps
    win_x, win_u, win_m = [], [], []
    for trial in range(x_train.shape[0]):
        for t in range(0, n_bins - W, past_steps):
            win_x.append(x_train[trial, :, t:t+W].T)   # (W, max_n)
            win_u.append(u_train_n[trial, :, t:t+W].T)  # (W, 2)
            win_m.append(m_train[trial])                  # (max_n,)

    windows_x = np.array(win_x, dtype=np.float32)
    windows_u = np.array(win_u, dtype=np.float32)
    masks = np.array(win_m, dtype=np.float32)

    stats = {
        "u_mean": u_mean,
        "u_std": u_std,
    }

    return windows_x, windows_u, masks, x_test, u_test_n, m_test, stats, max_n


# ---------------------------------------------------------------------------
# Poisson NLL loss (masked)
# ---------------------------------------------------------------------------

def masked_poisson_nll(log_rates, counts, mask):
    """Poisson NLL loss over non-padded neurons only.

    Following lfads-torch convention: model outputs log-rates,
    F.poisson_nll_loss(log_input=True) computes:
        loss = exp(log_rate) - count * log_rate

    Parameters
    ----------
    log_rates : (B, T, n_x) — decoder output (log firing rates per bin)
    counts    : (B, T, n_x) — raw spike counts (0, 1, 2, ...)
    mask      : (B, n_x)    — 1 for real neurons, 0 for padded
    """
    mask_expanded = mask.unsqueeze(1)  # (B, 1, n_x)
    nll = F.poisson_nll_loss(
        log_rates, counts,
        log_input=True, full=False, reduction="none"
    )  # (B, T, n_x)
    masked_nll = nll * mask_expanded
    n_valid = mask_expanded.sum() * log_rates.shape[1]
    return masked_nll.sum() / n_valid.clamp(min=1)


def masked_mse_on_rates(log_rates, counts, mask, bin_width_s=BIN_WIDTH_S):
    """MSE between predicted rates (Hz) and true rates (Hz) for R2 calc."""
    rates_pred = torch.exp(log_rates) / bin_width_s  # spikes/bin -> Hz
    rates_true = counts / bin_width_s
    mask_expanded = mask.unsqueeze(1)
    diff_sq = (rates_pred - rates_true) ** 2 * mask_expanded
    n_valid = mask_expanded.sum() * log_rates.shape[1]
    return diff_sq.sum() / n_valid.clamp(min=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pw = args.past_window
    fw = args.future_window
    dt_s = BIN_WIDTH_S

    print("=" * 60)
    print("Spiking CA-NODE Training (Poisson NLL)")
    print("  placement=%d, z_dim=%d, hidden=%d, lr=%.1e" %
          (args.placement, args.z_dim, args.hidden, args.lr))
    print("  past=%d steps (%dms), future=%d steps (%dms)" %
          (pw, pw * 10, fw, fw * 10))
    print("  Loss: Poisson NLL (raw spike counts, no normalization)")
    print("=" * 60)

    # -- Load data ---------------------------------------------------------
    print("\nLoading spiking data (Poisson mode)...")
    (windows_x, windows_u, masks,
     x_test, u_test_n, m_test, stats, max_n_sorted) = load_spiking_data_poisson(
        args.data, placement=args.placement,
        n_test_trials=10, seed=args.seed,
        past_steps=pw, future_steps=fw,
    )
    n_x = max_n_sorted
    n_u = windows_u.shape[2]
    print("  Windows: %d, n_x=%d (max sorted), n_u=%d" %
          (windows_x.shape[0], n_x, n_u))
    print("  x_counts range: [%.1f, %.1f] (should be 0-small int)" %
          (windows_x.min(), windows_x.max()))

    # -- Build model -------------------------------------------------------
    print("\nBuilding LatentControlAffineODE...")
    model = LatentControlAffineODE(
        n_x=n_x, n_u=n_u,
        z_dim=args.z_dim,
        hidden_dim=args.hidden,
        n_layers=args.n_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("  Parameters: %d" % n_params)

    # -- Initialize decoder bias to log(mean_spike_count + eps) per neuron --
    # This is critical for Poisson NLL: random bias produces arbitrary log-rates,
    # causing exp(log_rate) >> true_rate and preventing convergence (Exp 38 lesson).
    with torch.no_grad():
        # Compute mean spike count per neuron across all training windows (future portion)
        all_x_tmp = torch.tensor(windows_x, dtype=torch.float32)
        all_m_tmp = torch.tensor(masks, dtype=torch.float32)
        future_counts = all_x_tmp[:, pw:, :]  # (N, fw, n_x)
        mask_expanded = all_m_tmp.unsqueeze(1).expand_as(future_counts)
        # Mean count per neuron (masked)
        sum_counts = (future_counts * mask_expanded).sum(dim=(0, 1))
        n_valid = mask_expanded.sum(dim=(0, 1)).clamp(min=1)
        mean_counts = sum_counts / n_valid  # (n_x,)
        log_mean = torch.log(mean_counts + 1e-5)
        model.decoder.bias.copy_(log_mean.to(device))
        print("  Decoder bias initialized to log(mean_count + 1e-5)")
        print("    mean_count range: [%.3f, %.3f]" % (mean_counts.min(), mean_counts.max()))
        print("    log_mean range: [%.3f, %.3f]" % (log_mean.min(), log_mean.max()))

    # -- Prepare datasets --------------------------------------------------
    all_x = torch.tensor(windows_x, dtype=torch.float32)
    all_u = torch.tensor(windows_u, dtype=torch.float32)
    all_m = torch.tensor(masks, dtype=torch.float32)

    x_past_all = all_x[:, :pw, :]
    u_past_all = all_u[:, :pw, :]
    x_future_all = all_x[:, pw:, :]
    u_future_all = all_u[:, pw:, :]

    n_total = len(windows_x)
    n_val = max(1, n_total // 10)
    n_train = n_total - n_val

    train_ds = TensorDataset(
        x_past_all[:n_train], u_past_all[:n_train],
        x_future_all[:n_train], u_future_all[:n_train],
        all_m[:n_train],
    )
    val_ds = TensorDataset(
        x_past_all[n_train:], u_past_all[n_train:],
        x_future_all[n_train:], u_future_all[n_train:],
        all_m[n_train:],
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print("  Train windows: %d, Val windows: %d" % (n_train, n_val))

    t_future = torch.linspace(0, (fw - 1) * dt_s, fw, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # -- Training loop -----------------------------------------------------
    print("\nTraining...")
    history = {"train_loss": [], "val_loss": [], "lr": [], "val_loss_euler": [], "train_kl": [], "val_kl": []}
    best_val = float("inf")
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        epoch_kl = 0.0

        for x_past, u_past, x_future, u_future, mask in train_loader:
            x_past = x_past.to(device)
            u_past = u_past.to(device)
            x_future = x_future.to(device)
            u_future = u_future.to(device)
            mask = mask.to(device)


            kl_weight = min(1.0, epoch / 80.0) if epoch >= 20 else 0.0
            optimizer.zero_grad()

            # Forward: encoder sees past counts, ODE integrates, decoder outputs LOG-RATES
            log_rates_pred, mu, logvar = model.forward(
                x_past, u_past, u_future, t_future,
                method="dopri5", deterministic=False,
            )

            # Poisson NLL loss on raw spike counts

            # KL(q(z0|x) || N(0, I))
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
            kl_loss = kl_div.mean()
            
            nll_loss = masked_poisson_nll(log_rates_pred, x_future, mask)
            loss = nll_loss + kl_weight * kl_loss


            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_kl += kl_loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = epoch_loss / max(n_batches, 1)
        avg_kl = epoch_kl / max(n_batches, 1)
        history["train_loss"].append(avg_train)
        history["train_kl"].append(avg_kl)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # -- Validation ----------------------------------------------------
        model.eval()
        val_loss_sum, val_n = 0.0, 0
        val_kl_sum = 0.0
        val_loss_euler_sum = 0.0
        with torch.no_grad():
            for x_past, u_past, x_future, u_future, mask in val_loader:
                x_past = x_past.to(device)
                u_past = u_past.to(device)
                x_future = x_future.to(device)
                u_future = u_future.to(device)
                mask = mask.to(device)

                log_rates, mu, logvar = model.forward(
                    x_past, u_past, u_future, t_future,
                    method="dopri5", deterministic=False,
                )

                val_nll = masked_poisson_nll(log_rates, x_future, mask).item()
                val_kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
                val_kl_loss = val_kl_div.mean().item()
                val_loss_sum += val_nll + val_kl_loss
                val_kl_sum += val_kl_loss


                log_rates_e, _, _ = model.forward(
                    x_past, u_past, u_future, t_future,
                    method="euler", deterministic=False,
                )
                val_loss_euler_sum += masked_poisson_nll(log_rates_e, x_future, mask).item() + val_kl_loss
                val_n += 1

        val_loss = val_loss_sum / max(val_n, 1)
        val_kl = val_kl_sum / max(val_n, 1)
        history["val_kl"].append(val_kl)
        val_loss_euler = val_loss_euler_sum / max(val_n, 1)
        history["val_loss"].append(val_loss)
        history["val_loss_euler"].append(val_loss_euler)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "n_x": n_x, "n_u": n_u,
                "z_dim": args.z_dim, "hidden": args.hidden,
                "n_layers": args.n_layers,
                "placement": args.placement,
                "u_mean": stats["u_mean"], "u_std": stats["u_std"],
                "bin_width_s": BIN_WIDTH_S,
                "past_steps": pw, "future_steps": fw,
                "loss_type": "poisson_vae",
                "history": history,
                "epoch": epoch,
            }, os.path.join(args.output_dir, "best_model.pt"))

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            elapsed = time.time() - t0
            print("  Epoch %4d/%d | train=%.5f | val=%.5f | val_euler=%.5f | "
                  "lr=%.2e" %
                  (epoch, args.epochs, avg_train, val_loss, val_loss_euler,
                   optimizer.param_groups[0]["lr"]))

    # -- Evaluate best model on test trials --------------------------------
    print("\n" + "=" * 60)
    print("Evaluating best model on test trials")
    print("=" * 60)

    best_ckpt = torch.load(os.path.join(args.output_dir, "best_model.pt"),
                           map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    model.eval()

    results = {}
    for method in ["dopri5", "euler"]:
        all_preds, all_trues, all_masks_flat = [], [], []
        with torch.no_grad():
            for trial in range(x_test.shape[0]):
                x_trial = torch.tensor(x_test[trial].T, dtype=torch.float32,
                                       device=device)
                u_trial = torch.tensor(u_test_n[trial].T, dtype=torch.float32,
                                       device=device)
                mask_trial = torch.tensor(m_test[trial], dtype=torch.float32,
                                          device=device)

                n_bins = x_trial.shape[0]
                for t in range(0, n_bins - pw - fw, pw):
                    xp = x_trial[t:t+pw].unsqueeze(0)
                    up = u_trial[t:t+pw].unsqueeze(0)
                    xf = x_trial[t+pw:t+pw+fw]
                    uf = u_trial[t+pw:t+pw+fw].unsqueeze(0)

                    log_rates, _, _ = model.forward(xp, up, uf, t_future, method=method, deterministic=False)
                    log_rates = log_rates.squeeze(0)

                    # Convert to rates (Hz) for R2 calculation
                    rates_pred = torch.exp(log_rates).cpu().numpy() / BIN_WIDTH_S
                    rates_true = xf.cpu().numpy() / BIN_WIDTH_S

                    all_preds.append(rates_pred)
                    all_trues.append(rates_true)
                    all_masks_flat.append(mask_trial.cpu().numpy())

        preds = np.concatenate(all_preds, axis=0)
        trues = np.concatenate(all_trues, axis=0)
        masks_flat = np.array(all_masks_flat)
        mask_full = np.repeat(masks_flat, fw, axis=0)

        diff_sq = (trues - preds) ** 2 * mask_full
        mse = diff_sq.sum() / mask_full.sum()
        resid = diff_sq.sum()
        true_masked = trues * mask_full
        mean_per_neuron = (true_masked.sum(axis=0) / mask_full.sum(axis=0).clip(1))
        ss_tot = (((trues - mean_per_neuron[None, :]) ** 2) * mask_full).sum()
        r2 = 1 - resid / ss_tot

        results[method] = {"r2": float(r2), "mse": float(mse)}
        print("  %s: R2=%.4f, MSE=%.6f (on Hz rates)" % (method, r2, mse))

    # -- Save results ------------------------------------------------------
    results["config"] = {
        "z_dim": args.z_dim, "hidden": args.hidden,
        "n_layers": args.n_layers, "lr": args.lr,
        "epochs": args.epochs, "placement": args.placement,
        "n_x": n_x, "n_u": n_u,
        "past_steps": pw, "future_steps": fw,
        "n_params": n_params,
        "best_epoch": int(best_ckpt["epoch"]),
        "loss_type": "poisson_vae",
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # -- Plot training curves ----------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val (dopri5)")
    axes[0].plot(history["val_loss_euler"], label="Val (euler)", ls="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Poisson NLL")
    axes[0].set_title("Training Curves")
    axes[0].legend()

    axes[1].plot(history["lr"])
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("LR")
    axes[1].set_title("Learning Rate")

    axes[2].bar(["dopri5", "euler"],
               [results["dopri5"]["r2"], results["euler"]["r2"]])
    axes[2].set_ylabel("R2 (Hz rates)")
    axes[2].set_title("Test R2")
    axes[2].set_ylim(0, 1)
    for i, method in enumerate(["dopri5", "euler"]):
        axes[2].text(i, results[method]["r2"] + 0.02,
                     "%.4f" % results[method]["r2"], ha="center")

    plt.suptitle("Spiking CA-NODE (Poisson): z=%d h=%d lr=%.0e" %
                 (args.z_dim, args.hidden, args.lr))
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training_curves.png"), dpi=150)
    plt.close()

    print("\nResults saved to %s/" % args.output_dir)
    print("Final: R2(dopri5)=%.4f, R2(euler)=%.4f" %
          (results["dopri5"]["r2"], results["euler"]["r2"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Latent CA-NODE on spiking data (Poisson NLL)")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--placement", type=int, default=0)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--z-dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--past-window", type=int, default=20)
    parser.add_argument("--future-window", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    add_preflight_args(parser)
    args = parser.parse_args()
    validate_preflight(args)
    main(args)
