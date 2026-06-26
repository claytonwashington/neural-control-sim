#!/usr/bin/env python3
"""Train Acausal Latent CA-NODE on spike-sorted data with Poisson NLL + KL.

LFADS-style autoencoder: a bidirectional GRU encodes the ENTIRE trial to
produce q(z0), then the CA-ODE integrates forward over the FULL trial and
reconstructs spike counts via a Poisson observation model.

This is Exp 62 — the offline system-identification model.  Once it works,
its latent dynamics can be distilled into a causal (real-time) encoder via
Aligned Distillation (Exp 29-style).

Usage:
    python -m modeling.scripts.train_spiking_canode_vae_acausal \\
        --data data/spiking_plant3.h5 --placement 0 \\
        --z-dim 64 --hidden 256 --lr 5e-4 --epochs 500 \\
        --output-dir results/spiking_canode_vae_acausal_p0 \\
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
from torchdiffeq import odeint

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modeling.models.latent_canode import LatentControlAffineODE
from modeling.scripts.preflight_check import add_preflight_args, validate_preflight


BIN_WIDTH_S = 0.01  # 10 ms bins


# ---------------------------------------------------------------------------
# Data loading — full trials, no sliding windows
# ---------------------------------------------------------------------------

def load_spiking_data_acausal(
    data_path: str,
    placement: int = 0,
    n_test_trials: int = 10,
    seed: int = 42,
):
    """Load spike-sorted data as full trials for acausal autoencoding.

    Returns
    -------
    x_train : (n_train, T, n_x) float32 — raw spike counts
    u_train : (n_train, T, n_u) float32 — z-scored control input
    m_train : (n_train, n_x) float32 — neuron mask
    x_test, u_test, m_test : same for test set
    stats : dict with u normalization params
    n_x : int
    """
    with h5py.File(data_path, "r") as f:
        grp = f["placement_%d" % placement]
        x_sorted = grp["x_sorted"][:]            # (n_trials, max_n, n_bins) int16
        n_sorted = grp["n_sorted_per_trial"][:]   # (n_trials,)
        u_1ms = f["u"][:]                         # (n_trials, 2, n_bins*10)

    n_trials, max_n, n_bins = x_sorted.shape
    n_u = u_1ms.shape[1]

    # Raw spike counts as float32
    x_counts = x_sorted.astype(np.float32)  # (n_trials, max_n, n_bins)

    # Subsample u from 1ms to 10ms (average within each bin)
    u_10ms = u_1ms.reshape(n_trials, n_u, n_bins, 10).mean(axis=-1)  # (n_trials, n_u, n_bins)

    # Per-trial neuron masks
    neuron_masks = np.zeros((n_trials, max_n), dtype=np.float32)
    for i in range(n_trials):
        neuron_masks[i, :n_sorted[i]] = 1.0

    # Train/test split
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_trials)
    test_idx = perm[:n_test_trials]
    train_idx = perm[n_test_trials:]

    x_train = x_counts[train_idx]  # (n_train, max_n, n_bins)
    u_train = u_10ms[train_idx]    # (n_train, n_u, n_bins)
    m_train = neuron_masks[train_idx]
    x_test = x_counts[test_idx]
    u_test = u_10ms[test_idx]
    m_test = neuron_masks[test_idx]

    # Normalize u only
    u_mean = u_train.mean(axis=(0, 2))  # (n_u,)
    u_std = u_train.std(axis=(0, 2)) + 1e-8
    u_train_n = (u_train - u_mean[None, :, None]) / u_std[None, :, None]
    u_test_n = (u_test - u_mean[None, :, None]) / u_std[None, :, None]

    # Zero out padded neurons
    x_train *= m_train[:, :, None]
    x_test *= m_test[:, :, None]

    # Transpose to (n_trials, T, C) for batch-first convention
    x_train = x_train.transpose(0, 2, 1)   # (n_train, n_bins, max_n)
    u_train_n = u_train_n.transpose(0, 2, 1)  # (n_train, n_bins, n_u)
    x_test = x_test.transpose(0, 2, 1)
    u_test_n = u_test_n.transpose(0, 2, 1)

    stats = {"u_mean": u_mean, "u_std": u_std}
    return x_train, u_train_n, m_train, x_test, u_test_n, m_test, stats, max_n, n_bins


# ---------------------------------------------------------------------------
# Poisson NLL loss (masked)
# ---------------------------------------------------------------------------

def masked_poisson_nll(log_rates, counts, mask):
    """Poisson NLL loss over non-padded neurons."""
    mask_expanded = mask.unsqueeze(1)  # (B, 1, n_x)
    nll = F.poisson_nll_loss(
        log_rates, counts,
        log_input=True, full=False, reduction="none"
    )  # (B, T, n_x)
    masked_nll = nll * mask_expanded
    n_valid = mask_expanded.sum() * log_rates.shape[1]
    return masked_nll.sum() / n_valid.clamp(min=1)


# ---------------------------------------------------------------------------
# Acausal forward pass
# ---------------------------------------------------------------------------

def acausal_forward(model, x_full, u_full, t_full, deterministic=False, method="dopri5"):
    """Acausal forward: encode full trial, integrate full trial, decode full trial.

    Parameters
    ----------
    model : LatentControlAffineODE
    x_full : (B, T, n_x) — full trial spike counts
    u_full : (B, T, n_u) — full trial control input
    t_full : (T,) — time grid
    deterministic : bool — if True, use mean z0 instead of sampling
    method : str — ODE solver method

    Returns
    -------
    log_rates : (B, T, n_x)
    mu : (B, z_dim)
    logvar : (B, z_dim)
    """
    # 1. Encode the ENTIRE trial bidirectionally
    mu, logvar = model.encoder(x_full, u_full)

    # 2. Sample z0
    if deterministic:
        z0 = mu
    else:
        z0 = model.sample_z0(mu, logvar)

    # 3. Integrate CA-ODE over the FULL trial
    model.set_input(t_full, u_full)
    z_seq = odeint(model.ode_func, z0, t_full, method=method, rtol=1e-4, atol=1e-5)
    z_seq = z_seq.permute(1, 0, 2)  # (B, T, z_dim)

    # 4. Decode
    log_rates = model.decoder(z_seq)  # (B, T, n_x)

    return log_rates, mu, logvar


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dt_s = BIN_WIDTH_S

    print("=" * 60)
    print("Acausal Spiking CA-NODE VAE Training (LFADS-style)")
    print("  placement=%d, z_dim=%d, hidden=%d, lr=%.1e" %
          (args.placement, args.z_dim, args.hidden, args.lr))
    print("  Loss: Poisson NLL + KL (annealed)")
    print("  Mode: ACAUSAL — encoder sees full trial")
    print("=" * 60)

    # -- Load data ---------------------------------------------------------
    print("\nLoading spiking data (acausal mode — full trials)...")
    (x_train, u_train, m_train,
     x_test, u_test, m_test, stats, n_x, n_bins) = load_spiking_data_acausal(
        args.data, placement=args.placement,
        n_test_trials=10, seed=args.seed,
    )
    n_u = u_train.shape[2]
    print("  Train trials: %d, Test trials: %d" % (x_train.shape[0], x_test.shape[0]))
    print("  n_x=%d, n_u=%d, T=%d bins (%.1fs)" % (n_x, n_u, n_bins, n_bins * dt_s))
    print("  x_counts range: [%.1f, %.1f]" % (x_train.min(), x_train.max()))

    # -- Build model -------------------------------------------------------
    print("\nBuilding LatentControlAffineODE (acausal)...")
    model = LatentControlAffineODE(
        n_x=n_x, n_u=n_u,
        z_dim=args.z_dim,
        hidden_dim=args.hidden,
        n_layers=args.n_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("  Parameters: %d" % n_params)

    # -- Initialize decoder bias to log(mean_spike_count + eps) per neuron --
    with torch.no_grad():
        x_tmp = torch.tensor(x_train, dtype=torch.float32)  # (n_train, T, n_x)
        m_tmp = torch.tensor(m_train, dtype=torch.float32)  # (n_train, n_x)
        m_exp = m_tmp.unsqueeze(1).expand_as(x_tmp)  # (n_train, T, n_x)
        sum_counts = (x_tmp * m_exp).sum(dim=(0, 1))
        n_valid = m_exp.sum(dim=(0, 1)).clamp(min=1)
        mean_counts = sum_counts / n_valid
        log_mean = torch.log(mean_counts + 1e-5)
        model.decoder.bias.copy_(log_mean.to(device))
        print("  Decoder bias initialized to log(mean_count + 1e-5)")
        print("    mean_count range: [%.3f, %.3f]" % (mean_counts.min(), mean_counts.max()))

    # -- Prepare datasets --------------------------------------------------
    # Full-trial tensors
    tx_train = torch.tensor(x_train, dtype=torch.float32)  # (n_train, T, n_x)
    tu_train = torch.tensor(u_train, dtype=torch.float32)  # (n_train, T, n_u)
    tm_train = torch.tensor(m_train, dtype=torch.float32)  # (n_train, n_x)

    n_total = tx_train.shape[0]
    n_val = max(1, n_total // 5)  # 20% validation for small dataset
    n_train_split = n_total - n_val

    train_ds = TensorDataset(
        tx_train[:n_train_split], tu_train[:n_train_split], tm_train[:n_train_split]
    )
    val_ds = TensorDataset(
        tx_train[n_train_split:], tu_train[n_train_split:], tm_train[n_train_split:]
    )
    # Small batch size — each trial is 3000 timesteps, very memory-heavy
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print("  Train trials: %d, Val trials: %d" % (n_train_split, n_val))

    t_full = torch.linspace(0, (n_bins - 1) * dt_s, n_bins, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # -- Training loop -----------------------------------------------------
    print("\nTraining...")
    history = {"train_loss": [], "train_nll": [], "train_kl": [],
               "val_loss": [], "val_nll": [], "val_kl": [], "lr": []}
    best_val = float("inf")
    t0 = time.time()

    # KL annealing params
    kl_warmup = args.kl_warmup
    kl_anneal = args.kl_anneal

    for epoch in range(args.epochs):
        model.train()
        ep_loss, ep_nll, ep_kl, n_batches = 0.0, 0.0, 0.0, 0

        # KL weight schedule
        if epoch < kl_warmup:
            kl_weight = 0.0
        elif epoch < kl_warmup + kl_anneal:
            kl_weight = (epoch - kl_warmup) / kl_anneal
        else:
            kl_weight = 1.0

        for x_full, u_full, mask in train_loader:
            x_full = x_full.to(device)
            u_full = u_full.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()

            log_rates, mu, logvar = acausal_forward(
                model, x_full, u_full, t_full,
                deterministic=False, method="euler",
            )

            # Poisson NLL
            nll_loss = masked_poisson_nll(log_rates, x_full, mask)

            # KL divergence
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
            kl_loss = kl_div.mean()

            loss = nll_loss + kl_weight * kl_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss += loss.item()
            ep_nll += nll_loss.item()
            ep_kl += kl_loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = ep_loss / max(n_batches, 1)
        avg_nll = ep_nll / max(n_batches, 1)
        avg_kl = ep_kl / max(n_batches, 1)
        history["train_loss"].append(avg_loss)
        history["train_nll"].append(avg_nll)
        history["train_kl"].append(avg_kl)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # -- Validation ----------------------------------------------------
        model.eval()
        v_loss, v_nll, v_kl, v_n = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for x_full, u_full, mask in val_loader:
                x_full = x_full.to(device)
                u_full = u_full.to(device)
                mask = mask.to(device)

                log_rates, mu, logvar = acausal_forward(
                    model, x_full, u_full, t_full,
                    deterministic=False, method="euler",
                )
                nll = masked_poisson_nll(log_rates, x_full, mask).item()
                kl = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)).mean().item()
                v_loss += nll + kl_weight * kl
                v_nll += nll
                v_kl += kl
                v_n += 1

        val_loss = v_loss / max(v_n, 1)
        val_nll = v_nll / max(v_n, 1)
        val_kl = v_kl / max(v_n, 1)
        history["val_loss"].append(val_loss)
        history["val_nll"].append(val_nll)
        history["val_kl"].append(val_kl)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "n_x": n_x, "n_u": n_u, "n_bins": n_bins,
                "z_dim": args.z_dim, "hidden": args.hidden,
                "n_layers": args.n_layers,
                "placement": args.placement,
                "u_mean": stats["u_mean"], "u_std": stats["u_std"],
                "bin_width_s": BIN_WIDTH_S,
                "mode": "acausal",
                "loss_type": "poisson_vae_acausal",
                "history": history,
                "epoch": epoch,
            }, os.path.join(args.output_dir, "best_model.pt"))

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            elapsed = time.time() - t0
            print("  Epoch %4d/%d | loss=%.4f (nll=%.4f kl=%.4f) | "
                  "val=%.4f (nll=%.4f kl=%.4f) | kl_w=%.3f | lr=%.2e | %.0fs" %
                  (epoch, args.epochs, avg_loss, avg_nll, avg_kl,
                   val_loss, val_nll, val_kl, kl_weight,
                   optimizer.param_groups[0]["lr"], elapsed))

    # -- Evaluate best model on test trials --------------------------------
    print("\n" + "=" * 60)
    print("Evaluating best model on held-out test trials")
    print("=" * 60)

    best_ckpt = torch.load(os.path.join(args.output_dir, "best_model.pt"),
                           map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    model.eval()

    tx_test = torch.tensor(x_test, dtype=torch.float32, device=device)
    tu_test = torch.tensor(u_test, dtype=torch.float32, device=device)
    tm_test = torch.tensor(m_test, dtype=torch.float32, device=device)

    results = {}
    for method in ["euler"]:
        with torch.no_grad():
            # Process test trials one at a time (memory)
            all_preds, all_trues = [], []
            for i in range(tx_test.shape[0]):
                x_i = tx_test[i:i+1]  # (1, T, n_x)
                u_i = tu_test[i:i+1]
                m_i = tm_test[i:i+1]

                log_rates, _, _ = acausal_forward(
                    model, x_i, u_i, t_full,
                    deterministic=False, method=method,
                )
                rates_pred = torch.exp(log_rates).squeeze(0).cpu().numpy() / BIN_WIDTH_S
                rates_true = x_i.squeeze(0).cpu().numpy() / BIN_WIDTH_S
                mask_i = m_i.squeeze(0).cpu().numpy()  # (n_x,)

                # Mask out padded neurons
                rates_pred = rates_pred * mask_i[None, :]
                rates_true = rates_true * mask_i[None, :]

                all_preds.append(rates_pred)
                all_trues.append(rates_true)

            preds = np.concatenate(all_preds, axis=0)  # (n_test * T, n_x)
            trues = np.concatenate(all_trues, axis=0)

            # R2 over all valid neurons
            mask_full = np.tile(m_test, (n_bins, 1))  # broadcast mask
            # Actually need to repeat per trial
            mask_rep = np.concatenate([np.tile(m_test[i], (n_bins, 1)) for i in range(m_test.shape[0])], axis=0)
            diff_sq = (trues - preds) ** 2 * mask_rep
            resid = diff_sq.sum()
            mean_per_n = (trues * mask_rep).sum(axis=0) / mask_rep.sum(axis=0).clip(1)
            ss_tot = (((trues - mean_per_n[None, :]) ** 2) * mask_rep).sum()
            r2 = 1 - resid / ss_tot
            mse = diff_sq.sum() / mask_rep.sum()

        results[method] = {"r2": float(r2), "mse": float(mse)}
        print("  %s: R2=%.4f, MSE=%.4f (on Hz rates)" % (method, r2, mse))

    results["config"] = {
        "z_dim": args.z_dim, "hidden": args.hidden,
        "n_layers": args.n_layers, "lr": args.lr,
        "epochs": args.epochs, "placement": args.placement,
        "n_x": n_x, "n_u": n_u, "n_bins": n_bins,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "loss_type": "poisson_vae_acausal",
        "mode": "acausal",
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # -- Plot training curves ----------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("ELBO")
    axes[0].set_title("Total Loss")
    axes[0].legend()

    axes[1].plot(history["train_nll"], label="Train")
    axes[1].plot(history["val_nll"], label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("NLL")
    axes[1].set_title("Poisson NLL")
    axes[1].legend()

    axes[2].plot(history["train_kl"], label="Train")
    axes[2].plot(history["val_kl"], label="Val")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("KL")
    axes[2].set_title("KL Divergence")
    axes[2].legend()

    axes[3].plot(history["lr"])
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("LR")
    axes[3].set_title("Learning Rate")

    plt.suptitle("Acausal Spiking CA-NODE VAE: z=%d h=%d lr=%.0e | R2=%.4f" %
                 (args.z_dim, args.hidden, args.lr, results["euler"]["r2"]))
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training_curves.png"), dpi=150)
    plt.close()

    print("\nResults saved to %s/" % args.output_dir)
    print("Final: R2(euler)=%.4f" % results["euler"]["r2"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Acausal Latent CA-NODE VAE on spiking data")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--placement", type=int, default=0)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--z-dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size (small — each trial is 3000 bins)")
    parser.add_argument("--kl-warmup", type=int, default=20,
                        help="Epochs before KL annealing starts")
    parser.add_argument("--kl-anneal", type=int, default=80,
                        help="Epochs over which KL weight ramps 0->1")
    add_preflight_args(parser)
    args = parser.parse_args()
    validate_preflight(args)
    main(args)
