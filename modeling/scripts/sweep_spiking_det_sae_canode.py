#!/usr/bin/env python3
"""Deterministic Sequential Autoencoder with CA-NODE dynamics (CtD-style).

Architecture matching NODELatentSAE from the Computation-through-Dynamics
benchmark (Versteeg et al.):
  - Bidirectional GRU encoder over trialized segments (1s, 100 bins)
  - Deterministic IC (NO KL / VAE — just linear projection from GRU hidden)
  - Dropout on IC
  - Forward integration via Euler step: z_{t+1} = z_t + 0.1 * f(z_t, u_t)
  - Poisson NLL reconstruction loss
  - Separate learning rates for encoder, decoder (ODE), and readout

This is Exp 64 — a hyperparameter sweep over z_dim, hidden, lr.

Usage:
    python -m modeling.scripts.sweep_spiking_det_sae_canode \\
        --data data/spiking_plant3.h5 --placement 0 \\
        --output-dir results/spiking_det_sae_canode_sweep_p0 \\
        --gpus 1,2,3,4,5,6,7 --epochs 500
"""

from __future__ import annotations

import argparse
import itertools
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

from modeling.scripts.preflight_check import add_preflight_args, validate_preflight


BIN_WIDTH_S = 0.01  # 10 ms bins


# ---------------------------------------------------------------------------
# Data loading — trialized overlapping segments (shared with Exp 63)
# ---------------------------------------------------------------------------

def load_spiking_data_trialized(
    data_path: str,
    placement: int = 0,
    seg_len: int = 100,
    seg_overlap: int = 50,
    n_test_trials: int = 10,
    seed: int = 42,
):
    """Load spike-sorted data and chop into overlapping 1s segments."""
    with h5py.File(data_path, "r") as f:
        grp = f["placement_%d" % placement]
        x_sorted = grp["x_sorted"][:]            # (n_trials, max_n, n_bins)
        n_sorted = grp["n_sorted_per_trial"][:]   # (n_trials,)
        u_1ms = f["u"][:]                         # (n_trials, 2, n_bins*10)

    n_trials, max_n, n_bins = x_sorted.shape
    n_u = u_1ms.shape[1]

    x_counts = x_sorted.astype(np.float32).transpose(0, 2, 1)  # (T, time, neurons)
    u_10ms = u_1ms.reshape(n_trials, n_u, n_bins, 10).mean(axis=-1)
    u_10ms = u_10ms.transpose(0, 2, 1).astype(np.float32)

    neuron_masks = np.zeros((n_trials, max_n), dtype=np.float32)
    for i in range(n_trials):
        neuron_masks[i, :n_sorted[i]] = 1.0

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_trials)
    test_idx = perm[:n_test_trials]
    train_idx = perm[n_test_trials:]

    u_train_all = u_10ms[train_idx]
    u_mean = u_train_all.mean(axis=(0, 1))
    u_std = u_train_all.std(axis=(0, 1)) + 1e-8
    u_10ms_n = (u_10ms - u_mean[None, None, :]) / u_std[None, None, :]
    x_counts *= neuron_masks[:, None, :]

    stride = seg_len - seg_overlap

    def chop(indices):
        segs_x, segs_u, segs_m = [], [], []
        for i in indices:
            for t in range(0, n_bins - seg_len + 1, stride):
                segs_x.append(x_counts[i, t:t+seg_len])
                segs_u.append(u_10ms_n[i, t:t+seg_len])
                segs_m.append(neuron_masks[i])
        return (np.array(segs_x, dtype=np.float32),
                np.array(segs_u, dtype=np.float32),
                np.array(segs_m, dtype=np.float32))

    train_x, train_u, train_m = chop(train_idx)
    n_total = train_x.shape[0]
    val_mask = np.zeros(n_total, dtype=bool)
    val_mask[::5] = True
    val_x, val_u, val_m = train_x[val_mask], train_u[val_mask], train_m[val_mask]
    train_x, train_u, train_m = train_x[~val_mask], train_u[~val_mask], train_m[~val_mask]
    test_x, test_u, test_m = chop(test_idx)

    stats = {"u_mean": u_mean, "u_std": u_std}
    return (train_x, train_u, train_m,
            val_x, val_u, val_m,
            test_x, test_u, test_m,
            stats, max_n)


# ---------------------------------------------------------------------------
# Deterministic Sequential Autoencoder with CA-NODE dynamics
# ---------------------------------------------------------------------------

class MLPVectorField(nn.Module):
    """MLP vector field: f(z, u) -> dz."""
    def __init__(self, z_dim, u_dim, hidden_dim, n_layers):
        super().__init__()
        layers = [nn.Linear(z_dim + u_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, z_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z, u):
        return self.net(torch.cat([z, u], dim=-1))


class DetSAECaNODE(nn.Module):
    """Deterministic Sequential Autoencoder with Control-Affine NODE.

    Matches CtD benchmark architecture:
    - Bidirectional GRU encoder -> deterministic IC
    - Euler integration: z_{t+1} = z_t + dt * f(z_t, u_t)
    - Linear readout -> Poisson NLL
    """
    def __init__(self, n_x, n_u, z_dim=32, hidden_dim=128,
                 n_layers=3, encoder_hidden=128, dropout=0.05):
        super().__init__()
        self.n_x = n_x
        self.n_u = n_u
        self.z_dim = z_dim

        # Bidirectional GRU encoder (CtD-style)
        self.encoder = nn.GRU(
            input_size=n_x,
            hidden_size=encoder_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.ic_linear = nn.Linear(2 * encoder_hidden, z_dim)

        # ODE vector field
        self.vf = MLPVectorField(z_dim, n_u, hidden_dim, n_layers)

        # Linear readout (log-rates)
        self.readout = nn.Linear(z_dim, n_x)

    def forward(self, x_seg, u_seg, dt=0.1):
        """Encode segment, integrate forward, decode.

        Parameters
        ----------
        x_seg : (B, T, n_x) — spike counts
        u_seg : (B, T, n_u) — control inputs
        dt : float — Euler step size (matching CtD's 0.1 scaling)

        Returns
        -------
        log_rates : (B, T, n_x)
        latents : (B, T, z_dim)
        """
        B, T, _ = x_seg.shape

        # Encode entire segment bidirectionally
        _, h_n = self.encoder(x_seg)  # h_n: (2, B, encoder_hidden)
        h_n = torch.cat([h_n[0], h_n[1]], dim=-1)  # (B, 2*encoder_hidden)
        h_n = self.dropout(h_n)
        z0 = self.ic_linear(h_n)  # (B, z_dim)

        # Euler integration step-by-step
        z = z0
        latents = []
        for t in range(T):
            latents.append(z)
            if t < T - 1:
                dz = self.vf(z, u_seg[:, t, :])
                z = z + dt * dz
        latents = torch.stack(latents, dim=1)  # (B, T, z_dim)

        # Decode
        log_rates = self.readout(latents)  # (B, T, n_x)
        return log_rates, latents


# ---------------------------------------------------------------------------
# Masked Poisson NLL
# ---------------------------------------------------------------------------

def masked_poisson_nll(log_rates, counts, mask):
    mask_expanded = mask.unsqueeze(1)  # (B, 1, n_x)
    nll = F.poisson_nll_loss(
        log_rates, counts, log_input=True, full=False, reduction="none"
    )
    masked_nll = nll * mask_expanded
    n_valid = mask_expanded.sum() * log_rates.shape[1]
    return masked_nll.sum() / n_valid.clamp(min=1)


# ---------------------------------------------------------------------------
# Train one config
# ---------------------------------------------------------------------------

def train_one_config(
    config: dict,
    train_x, train_u, train_m,
    val_x, val_u, val_m,
    test_x, test_u, test_m,
    n_x: int, n_u: int,
    output_dir: str,
    device: torch.device,
    epochs: int = 500,
    batch_size: int = 64,
    seed: int = 42,
):
    """Train a single DetSAE-CaNODE config and return results."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    z_dim = config["z_dim"]
    hidden = config["hidden"]
    lr = config["lr"]
    n_layers = config.get("n_layers", 3)
    dropout = config.get("dropout", 0.05)
    dt = config.get("dt", 0.1)

    run_id = "z%d_h%d_lr%.0e_nl%d" % (z_dim, hidden, lr, n_layers)
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    model = DetSAECaNODE(
        n_x=n_x, n_u=n_u, z_dim=z_dim,
        hidden_dim=hidden, n_layers=n_layers,
        encoder_hidden=hidden, dropout=dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # Initialize readout bias to log(mean_count)
    with torch.no_grad():
        x_tmp = torch.tensor(train_x)
        m_tmp = torch.tensor(train_m)
        m_exp = m_tmp.unsqueeze(1).expand_as(x_tmp)
        sum_counts = (x_tmp * m_exp).sum(dim=(0, 1))
        n_valid = m_exp.sum(dim=(0, 1)).clamp(min=1)
        mean_counts = sum_counts / n_valid
        model.readout.bias.copy_(torch.log(mean_counts + 1e-5).to(device))

    # Separate LRs (CtD-style)
    optimizer = torch.optim.Adam([
        {"params": list(model.encoder.parameters()) + list(model.ic_linear.parameters()),
         "lr": lr, "weight_decay": 1e-5},
        {"params": model.vf.parameters(),
         "lr": lr, "weight_decay": 1e-5},
        {"params": model.readout.parameters(),
         "lr": lr, "weight_decay": 0},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_ds = TensorDataset(
        torch.tensor(train_x), torch.tensor(train_u), torch.tensor(train_m)
    )
    val_ds = TensorDataset(
        torch.tensor(val_x), torch.tensor(val_u), torch.tensor(val_m)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    history = {"train_nll": [], "val_nll": [], "lr": []}
    best_val = float("inf")
    best_epoch = 0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        ep_nll, n_batches = 0.0, 0

        for x_seg, u_seg, mask in train_loader:
            x_seg = x_seg.to(device)
            u_seg = u_seg.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            log_rates, _ = model(x_seg, u_seg, dt=dt)
            loss = masked_poisson_nll(log_rates, x_seg, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_nll += loss.item()
            n_batches += 1

        scheduler.step()
        avg_nll = ep_nll / max(n_batches, 1)
        history["train_nll"].append(avg_nll)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # Validation
        model.eval()
        v_nll, v_n = 0.0, 0
        with torch.no_grad():
            for x_seg, u_seg, mask in val_loader:
                x_seg = x_seg.to(device)
                u_seg = u_seg.to(device)
                mask = mask.to(device)
                log_rates, _ = model(x_seg, u_seg, dt=dt)
                v_nll += masked_poisson_nll(log_rates, x_seg, mask).item()
                v_n += 1
        val_nll = v_nll / max(v_n, 1)
        history["val_nll"].append(val_nll)

        if val_nll < best_val:
            best_val = val_nll
            best_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "config": config,
                "n_x": n_x, "n_u": n_u,
                "epoch": epoch, "val_nll": val_nll,
                "n_params": n_params,
            }, os.path.join(run_dir, "best_model.pt"))

        if epoch % 50 == 0 or epoch == epochs - 1:
            elapsed = time.time() - t0
            print("  [%s] Epoch %4d/%d | train=%.4f | val=%.4f | best=%.4f@%d | %.0fs" %
                  (run_id, epoch, epochs, avg_nll, val_nll, best_val, best_epoch, elapsed))

    # -- Evaluate on test segments --
    ckpt = torch.load(os.path.join(run_dir, "best_model.pt"),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_ds = TensorDataset(
        torch.tensor(test_x), torch.tensor(test_u), torch.tensor(test_m)
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    all_preds, all_trues, all_masks = [], [], []
    with torch.no_grad():
        for x_seg, u_seg, mask in test_loader:
            x_seg = x_seg.to(device)
            u_seg = u_seg.to(device)
            mask = mask.to(device)
            log_rates, _ = model(x_seg, u_seg, dt=dt)
            rates_pred = torch.exp(log_rates).cpu().numpy() / BIN_WIDTH_S
            rates_true = x_seg.cpu().numpy() / BIN_WIDTH_S
            m_np = mask.cpu().numpy()
            all_preds.append(rates_pred.reshape(-1, n_x))
            all_trues.append(rates_true.reshape(-1, n_x))
            all_masks.append(np.repeat(m_np, x_seg.shape[1], axis=0))

    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)
    masks_r = np.concatenate(all_masks, axis=0)

    diff_sq = (trues - preds) ** 2 * masks_r
    resid = diff_sq.sum()
    mean_per_n = (trues * masks_r).sum(axis=0) / masks_r.sum(axis=0).clip(1)
    ss_tot = (((trues - mean_per_n[None, :]) ** 2) * masks_r).sum()
    r2 = 1 - resid / ss_tot
    mse = diff_sq.sum() / masks_r.sum()

    elapsed = time.time() - t0
    results = {
        "r2": float(r2), "mse": float(mse),
        "best_val_nll": float(best_val), "best_epoch": best_epoch,
        "n_params": n_params, "training_time_s": elapsed,
        "config": config, "run_id": run_id,
    }
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Training curve plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(history["train_nll"], label="Train")
    ax.plot(history["val_nll"], label="Val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Poisson NLL")
    ax.set_title("%s | R²=%.4f" % (run_id, r2)); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "training_curve.png"), dpi=100)
    plt.close()

    return results


# ---------------------------------------------------------------------------
# GPU worker for parallel sweep
# ---------------------------------------------------------------------------

def gpu_worker(gpu_id, configs, shared_data, output_dir, epochs, batch_size, seed):
    """Train all configs assigned to this GPU."""
    device = torch.device("cuda:%d" % gpu_id)
    (train_x, train_u, train_m,
     val_x, val_u, val_m,
     test_x, test_u, test_m,
     n_x, n_u) = shared_data

    results = []
    for config in configs:
        print("\n[GPU %d] Starting %s" % (gpu_id, config))
        r = train_one_config(
            config=config,
            train_x=train_x, train_u=train_u, train_m=train_m,
            val_x=val_x, val_u=val_u, val_m=val_m,
            test_x=test_x, test_u=test_u, test_m=test_m,
            n_x=n_x, n_u=n_u,
            output_dir=output_dir, device=device,
            epochs=epochs, batch_size=batch_size, seed=seed,
        )
        results.append(r)
        print("[GPU %d] %s -> R²=%.4f (val_nll=%.4f)" %
              (gpu_id, r["run_id"], r["r2"], r["best_val_nll"]))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    n_gpus = len(gpu_ids)

    print("=" * 60)
    print("Deterministic SAE CA-NODE Sweep (CtD-style, no VAE)")
    print("  GPUs: %s (%d workers)" % (args.gpus, n_gpus))
    print("  Epochs: %d, Batch size: %d" % (args.epochs, args.batch_size))
    print("=" * 60)

    # Load data once
    print("\nLoading spiking data (trialized segments)...")
    (train_x, train_u, train_m,
     val_x, val_u, val_m,
     test_x, test_u, test_m,
     stats, n_x) = load_spiking_data_trialized(
        args.data, placement=args.placement,
        seg_len=args.seg_len, seg_overlap=args.seg_overlap,
    )
    n_u = train_u.shape[2]
    print("  Train: %d, Val: %d, Test: %d segs" %
          (train_x.shape[0], val_x.shape[0], test_x.shape[0]))
    print("  n_x=%d, n_u=%d, seg_len=%d" % (n_x, n_u, args.seg_len))

    # Build sweep grid
    configs = []
    for z_dim, hidden, lr in itertools.product(
        [10, 32, 64],
        [128, 256],
        [1e-3, 5e-4, 1e-4],
    ):
        configs.append({
            "z_dim": z_dim, "hidden": hidden, "lr": lr,
            "n_layers": 3, "dropout": 0.05, "dt": 0.1,
        })

    print("\n  Sweep: %d configs across %d GPUs" % (len(configs), n_gpus))

    # Distribute configs round-robin across GPUs
    gpu_configs = {g: [] for g in gpu_ids}
    for i, config in enumerate(configs):
        gpu_configs[gpu_ids[i % n_gpus]].append(config)

    shared_data = (train_x, train_u, train_m,
                   val_x, val_u, val_m,
                   test_x, test_u, test_m,
                   n_x, n_u)

    # Run sequentially per GPU using multiprocessing
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    all_results = []

    def worker_wrapper(gpu_id, cfgs):
        return gpu_worker(gpu_id, cfgs, shared_data,
                          args.output_dir, args.epochs, args.batch_size, args.seed)

    # Launch all GPU workers in parallel
    with mp.Pool(processes=n_gpus) as pool:
        async_results = []
        for gpu_id in gpu_ids:
            if gpu_configs[gpu_id]:  # only if there are configs for this GPU
                ar = pool.apply_async(worker_wrapper, (gpu_id, gpu_configs[gpu_id]))
                async_results.append(ar)
        for ar in async_results:
            all_results.extend(ar.get())

    # Sort by R² and save summary
    all_results.sort(key=lambda r: r["r2"], reverse=True)

    print("\n" + "=" * 60)
    print("SWEEP RESULTS (sorted by R²)")
    print("=" * 60)
    for r in all_results:
        print("  %-25s  R²=%.4f  val_nll=%.4f  params=%d  time=%.0fs" %
              (r["run_id"], r["r2"], r["best_val_nll"], r["n_params"], r["training_time_s"]))

    summary = {
        "best": all_results[0],
        "all_results": all_results,
        "sweep_config": {
            "z_dims": [10, 32, 64],
            "hiddens": [128, 256],
            "lrs": [1e-3, 5e-4, 1e-4],
            "n_configs": len(configs),
            "epochs": args.epochs,
            "seg_len": args.seg_len,
            "seg_overlap": args.seg_overlap,
            "architecture": "DetSAE_CaNODE_CtD_style",
        },
    }
    with open(os.path.join(args.output_dir, "sweep_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Summary plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    r2s = [r["r2"] for r in all_results]
    labels = [r["run_id"] for r in all_results]
    axes[0].barh(range(len(r2s)), r2s)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].set_xlabel("R²")
    axes[0].set_title("Sweep Results (sorted)")
    axes[0].axvline(x=0, color="red", linestyle="--", alpha=0.5)

    nlls = [r["best_val_nll"] for r in all_results]
    axes[1].barh(range(len(nlls)), nlls)
    axes[1].set_yticks(range(len(labels)))
    axes[1].set_yticklabels(labels, fontsize=7)
    axes[1].set_xlabel("Best Val NLL")
    axes[1].set_title("Validation NLL")

    plt.suptitle("Det SAE CA-NODE Sweep | Best R²=%.4f (%s)" %
                 (all_results[0]["r2"], all_results[0]["run_id"]))
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "sweep_summary.png"), dpi=150)
    plt.close()

    print("\nBest: %s -> R²=%.4f" % (all_results[0]["run_id"], all_results[0]["r2"]))
    print("Results saved to %s/" % args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deterministic SAE CA-NODE sweep (CtD-style)")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--placement", type=int, default=0)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--gpus", type=str, default="1,2,3,4,5,6,7",
                        help="Comma-separated GPU indices")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seg-len", type=int, default=100)
    parser.add_argument("--seg-overlap", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    add_preflight_args(parser)
    args = parser.parse_args()
    validate_preflight(args)
    main(args)
