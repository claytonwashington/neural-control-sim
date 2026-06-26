#!/usr/bin/env python3
"""Train Acausal Latent CA-NODE on spike-sorted data with Poisson NLL + KL.

LFADS-style autoencoder on TRIALIZED segments: each 30s trial is chopped
into overlapping 1-second (100-bin) windows. A bidirectional GRU encodes
each segment to infer q(z0), then the CA-ODE integrates forward and
reconstructs that same segment via a Poisson observation model.

This is Exp 63 — the offline system-identification model. Once it works,
its latent dynamics can be distilled into a causal encoder via Aligned
Distillation (Exp 29-style).

Usage:
    python -m modeling.scripts.train_spiking_canode_vae_acausal \\
        --data data/spiking_plant3.h5 --placement 0 \\
        --z-dim 64 --hidden 256 --lr 5e-4 --epochs 500 \\
        --seg-len 100 --seg-overlap 50 \\
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
# Data loading — trialized overlapping segments
# ---------------------------------------------------------------------------

def load_spiking_data_trialized(
    data_path: str,
    placement: int = 0,
    seg_len: int = 100,
    seg_overlap: int = 50,
    n_test_trials: int = 10,
    seed: int = 42,
):
    """Load spike-sorted data and chop into overlapping segments.

    Follows the standard LFADS / Pandarinath et al. (2018) convention:
    continuous data is segmented into 1s windows with overlap.

    Parameters
    ----------
    seg_len : int
        Segment length in bins (default 100 = 1000ms)
    seg_overlap : int
        Overlap between segments in bins (default 50 = 500ms)

    Returns
    -------
    train_x : (n_train_segs, seg_len, n_x) float32 — raw spike counts
    train_u : (n_train_segs, seg_len, n_u) float32 — z-scored control
    train_m : (n_train_segs, n_x) float32 — neuron masks
    val_x, val_u, val_m : same for validation
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

    # Raw spike counts, transpose to (trials, time, neurons)
    x_counts = x_sorted.astype(np.float32).transpose(0, 2, 1)  # (50, 3000, 126)

    # Downsample u from 1ms to 10ms, transpose
    u_10ms = u_1ms.reshape(n_trials, n_u, n_bins, 10).mean(axis=-1)
    u_10ms = u_10ms.transpose(0, 2, 1).astype(np.float32)  # (50, 3000, 2)

    # Per-trial neuron masks
    neuron_masks = np.zeros((n_trials, max_n), dtype=np.float32)
    for i in range(n_trials):
        neuron_masks[i, :n_sorted[i]] = 1.0

    # Train/test split (same seed as other spiking experiments)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_trials)
    test_idx = perm[:n_test_trials]
    train_idx = perm[n_test_trials:]

    # Normalize u on training set only
    u_train_all = u_10ms[train_idx]
    u_mean = u_train_all.mean(axis=(0, 1))  # (n_u,)
    u_std = u_train_all.std(axis=(0, 1)) + 1e-8
    u_10ms_n = (u_10ms - u_mean[None, None, :]) / u_std[None, None, :]

    # Zero out padded neurons
    x_counts *= neuron_masks[:, None, :]  # broadcast (trials, 1, neurons)

    # Chop into overlapping segments
    stride = seg_len - seg_overlap

    def chop(indices):
        segs_x, segs_u, segs_m = [], [], []
        for i in indices:
            for t in range(0, n_bins - seg_len + 1, stride):
                segs_x.append(x_counts[i, t:t+seg_len])   # (seg_len, n_x)
                segs_u.append(u_10ms_n[i, t:t+seg_len])    # (seg_len, n_u)
                segs_m.append(neuron_masks[i])               # (n_x,)
        return (np.array(segs_x, dtype=np.float32),
                np.array(segs_u, dtype=np.float32),
                np.array(segs_m, dtype=np.float32))

    train_x, train_u, train_m = chop(train_idx)

    # Validation: use every 5th segment from training trials (LFADS convention)
    # so train/val come from same distribution
    n_total = train_x.shape[0]
    val_mask = np.zeros(n_total, dtype=bool)
    val_mask[::5] = True
    val_x, val_u, val_m = train_x[val_mask], train_u[val_mask], train_m[val_mask]
    train_x, train_u, train_m = train_x[~val_mask], train_u[~val_mask], train_m[~val_mask]

    # Also chop test trials for final eval
    test_x, test_u, test_m = chop(test_idx)

    stats = {"u_mean": u_mean, "u_std": u_std}
    return (train_x, train_u, train_m,
            val_x, val_u, val_m,
            test_x, test_u, test_m,
            stats, max_n, seg_len)


# ---------------------------------------------------------------------------
# Poisson NLL loss (masked)
# ---------------------------------------------------------------------------

def masked_poisson_nll(log_rates, counts, mask):
    mask_expanded = mask.unsqueeze(1)  # (B, 1, n_x)
    nll = F.poisson_nll_loss(
        log_rates, counts,
        log_input=True, full=False, reduction="none"
    )  # (B, T, n_x)
    masked_nll = nll * mask_expanded
    n_valid = mask_expanded.sum() * log_rates.shape[1]
    return masked_nll.sum() / n_valid.clamp(min=1)


# ---------------------------------------------------------------------------
# Acausal forward: encode segment, integrate segment, decode segment
# ---------------------------------------------------------------------------

def acausal_forward(model, x_seg, u_seg, t_seg, deterministic=False, method="euler"):
    """Encode a segment bidirectionally, integrate CA-ODE, reconstruct.

    Parameters
    ----------
    x_seg : (B, seg_len, n_x)
    u_seg : (B, seg_len, n_u)
    t_seg : (seg_len,)
    """
    # 1. Bidirectional encoder over the ENTIRE segment
    mu, logvar = model.encoder(x_seg, u_seg)

    # 2. Sample z0
    if deterministic:
        z0 = mu
    else:
        z0 = model.sample_z0(mu, logvar)

    # 3. Integrate CA-ODE over the segment
    model.set_input(t_seg, u_seg)
    z_seq = odeint(model.ode_func, z0, t_seg, method=method, rtol=1e-4, atol=1e-5)
    z_seq = z_seq.permute(1, 0, 2)  # (B, seg_len, z_dim)

    # 4. Decode
    log_rates = model.decoder(z_seq)  # (B, seg_len, n_x)
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
    seg_len = args.seg_len

    print("=" * 60)
    print("Acausal Spiking CA-NODE VAE (Trialized Segments)")
    print("  placement=%d, z_dim=%d, hidden=%d, lr=%.1e" %
          (args.placement, args.z_dim, args.hidden, args.lr))
    print("  seg_len=%d bins (%dms), overlap=%d bins (%dms)" %
          (seg_len, seg_len * 10, args.seg_overlap, args.seg_overlap * 10))
    print("  Loss: Poisson NLL + KL (annealed)")
    print("  Mode: ACAUSAL — bidirectional encoder over each segment")
    print("=" * 60)

    # -- Load data ---------------------------------------------------------
    print("\nLoading spiking data (trialized segments)...")
    (train_x, train_u, train_m,
     val_x, val_u, val_m,
     test_x, test_u, test_m,
     stats, n_x, _seg_len) = load_spiking_data_trialized(
        args.data, placement=args.placement,
        seg_len=seg_len, seg_overlap=args.seg_overlap,
        n_test_trials=10, seed=args.seed,
    )
    n_u = train_u.shape[2]
    print("  Train segs: %d, Val segs: %d, Test segs: %d" %
          (train_x.shape[0], val_x.shape[0], test_x.shape[0]))
    print("  n_x=%d, n_u=%d, seg_len=%d bins" % (n_x, n_u, seg_len))

    # -- Build model -------------------------------------------------------
    model = LatentControlAffineODE(
        n_x=n_x, n_u=n_u,
        z_dim=args.z_dim,
        hidden_dim=args.hidden,
        n_layers=args.n_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("  Parameters: %d" % n_params)

    # -- Initialize decoder bias -------------------------------------------
    with torch.no_grad():
        x_tmp = torch.tensor(train_x, dtype=torch.float32)
        m_tmp = torch.tensor(train_m, dtype=torch.float32)
        m_exp = m_tmp.unsqueeze(1).expand_as(x_tmp)
        sum_counts = (x_tmp * m_exp).sum(dim=(0, 1))
        n_valid = m_exp.sum(dim=(0, 1)).clamp(min=1)
        mean_counts = sum_counts / n_valid
        log_mean = torch.log(mean_counts + 1e-5)
        model.decoder.bias.copy_(log_mean.to(device))
        print("  Decoder bias init: mean_count [%.3f, %.3f]" %
              (mean_counts.min(), mean_counts.max()))

    # -- Prepare datasets --------------------------------------------------
    train_ds = TensorDataset(
        torch.tensor(train_x), torch.tensor(train_u), torch.tensor(train_m)
    )
    val_ds = TensorDataset(
        torch.tensor(val_x), torch.tensor(val_u), torch.tensor(val_m)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    t_seg = torch.linspace(0, (seg_len - 1) * dt_s, seg_len, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # -- Training loop -----------------------------------------------------
    print("\nTraining...")
    history = {"train_loss": [], "train_nll": [], "train_kl": [],
               "val_loss": [], "val_nll": [], "val_kl": [], "lr": []}
    best_val = float("inf")
    best_epoch = 0
    t0 = time.time()

    kl_warmup = args.kl_warmup
    kl_anneal = args.kl_anneal

    for epoch in range(args.epochs):
        model.train()
        ep_loss, ep_nll, ep_kl, n_batches = 0.0, 0.0, 0.0, 0

        if epoch < kl_warmup:
            kl_weight = 0.0
        elif epoch < kl_warmup + kl_anneal:
            kl_weight = (epoch - kl_warmup) / kl_anneal
        else:
            kl_weight = 1.0

        for x_seg, u_seg, mask in train_loader:
            x_seg = x_seg.to(device)
            u_seg = u_seg.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()

            log_rates, mu, logvar = acausal_forward(
                model, x_seg, u_seg, t_seg,
                deterministic=False, method="euler",
            )

            nll_loss = masked_poisson_nll(log_rates, x_seg, mask)
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
            for x_seg, u_seg, mask in val_loader:
                x_seg = x_seg.to(device)
                u_seg = u_seg.to(device)
                mask = mask.to(device)

                log_rates, mu, logvar = acausal_forward(
                    model, x_seg, u_seg, t_seg,
                    deterministic=False, method="euler",
                )
                nll = masked_poisson_nll(log_rates, x_seg, mask).item()
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
                "n_x": n_x, "n_u": n_u, "seg_len": seg_len,
                "z_dim": args.z_dim, "hidden": args.hidden,
                "n_layers": args.n_layers,
                "placement": args.placement,
                "u_mean": stats["u_mean"], "u_std": stats["u_std"],
                "bin_width_s": BIN_WIDTH_S,
                "mode": "acausal_trialized",
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

    # -- Evaluate on test segments -----------------------------------------
    print("\n" + "=" * 60)
    print("Evaluating best model on held-out test segments")
    print("=" * 60)

    best_ckpt = torch.load(os.path.join(args.output_dir, "best_model.pt"),
                           map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    model.eval()

    test_ds = TensorDataset(
        torch.tensor(test_x), torch.tensor(test_u), torch.tensor(test_m)
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    all_preds, all_trues, all_masks = [], [], []
    with torch.no_grad():
        for x_seg, u_seg, mask in test_loader:
            x_seg = x_seg.to(device)
            u_seg = u_seg.to(device)
            mask = mask.to(device)

            log_rates, _, _ = acausal_forward(
                model, x_seg, u_seg, t_seg,
                deterministic=False, method="euler",
            )
            rates_pred = torch.exp(log_rates).cpu().numpy() / BIN_WIDTH_S
            rates_true = x_seg.cpu().numpy() / BIN_WIDTH_S
            m_np = mask.cpu().numpy()

            all_preds.append(rates_pred.reshape(-1, n_x))
            all_trues.append(rates_true.reshape(-1, n_x))
            all_masks.append(np.repeat(m_np, seg_len, axis=0))

    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)
    masks_r = np.concatenate(all_masks, axis=0)

    diff_sq = (trues - preds) ** 2 * masks_r
    resid = diff_sq.sum()
    mean_per_n = (trues * masks_r).sum(axis=0) / masks_r.sum(axis=0).clip(1)
    ss_tot = (((trues - mean_per_n[None, :]) ** 2) * masks_r).sum()
    r2 = 1 - resid / ss_tot
    mse = diff_sq.sum() / masks_r.sum()

    print("  euler: R2=%.4f, MSE=%.4f (on Hz rates)" % (r2, mse))

    results = {
        "euler": {"r2": float(r2), "mse": float(mse)},
        "config": {
            "z_dim": args.z_dim, "hidden": args.hidden,
            "n_layers": args.n_layers, "lr": args.lr,
            "epochs": args.epochs, "placement": args.placement,
            "n_x": n_x, "n_u": n_u, "seg_len": seg_len,
            "seg_overlap": args.seg_overlap,
            "n_params": n_params,
            "best_epoch": best_epoch,
            "loss_type": "poisson_vae_acausal",
            "mode": "acausal_trialized",
        },
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # -- Plot training curves ----------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("ELBO")
    axes[0].set_title("Total Loss"); axes[0].legend()

    axes[1].plot(history["train_nll"], label="Train")
    axes[1].plot(history["val_nll"], label="Val")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("NLL")
    axes[1].set_title("Poisson NLL"); axes[1].legend()

    axes[2].plot(history["train_kl"], label="Train")
    axes[2].plot(history["val_kl"], label="Val")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("KL")
    axes[2].set_title("KL Divergence"); axes[2].legend()

    axes[3].plot(history["lr"])
    axes[3].set_xlabel("Epoch"); axes[3].set_ylabel("LR")
    axes[3].set_title("Learning Rate")

    plt.suptitle("Acausal CA-NODE VAE (trialized %d-bin segs) | R2=%.4f" %
                 (seg_len, r2))
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training_curves.png"), dpi=150)
    plt.close()

    print("\nResults saved to %s/" % args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Acausal CA-NODE VAE on trialized spiking segments")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--placement", type=int, default=0)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--z-dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--seg-len", type=int, default=100,
                        help="Segment length in bins (100 = 1000ms)")
    parser.add_argument("--seg-overlap", type=int, default=50,
                        help="Segment overlap in bins (50 = 500ms)")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--kl-warmup", type=int, default=20)
    parser.add_argument("--kl-anneal", type=int, default=80)
    add_preflight_args(parser)
    args = parser.parse_args()
    validate_preflight(args)
    main(args)
