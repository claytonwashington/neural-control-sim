#!/usr/bin/env python3
"""Deterministic SAE CA-NODE sweep v2 — FIXED.

Fixes from v1 (Exp 64):
1. Added MSE loss on z-scored firing rates (like Exp 37 which got R²=0.4)
2. Added Coordinated Dropout (from AutoLFADS)
3. Fixed R² to use Gaussian-smoothed spike trains (σ=50ms)
4. Spike counts capped at 5 — Poisson assumption violated!
   -> MSE on z-scored rates is the correct loss for this data
5. Separate encoder/decoder learning rate groups

Sweep:
  loss: [mse_zscore, poisson]
  z_dim: [32, 64]
  hidden: [128, 256]
  lr: [5e-4, 1e-3]
  cd_rate: [0.0, 0.3]
  = 2 * 2 * 2 * 2 * 2 = 32 configs

Usage:
    python -u -m modeling.scripts.sweep_spiking_det_sae_v2 \
        --data data/spiking_plant3.h5 --placement 0 \
        --output-dir results/spiking_det_sae_v2_sweep_p0 \
        --gpus 0,1,2,3,4,5,6,7 --epochs 300
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
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modeling.scripts.preflight_check import add_preflight_args, validate_preflight


BIN_WIDTH_S = 0.01  # 10 ms bins
SMOOTH_SIGMA = 5    # 5 bins = 50ms Gaussian smoothing for R² evaluation


# ---------------------------------------------------------------------------
# Coordinated Dropout (from AutoLFADS / lfads-torch)
# ---------------------------------------------------------------------------

class CoordinatedDropout:
    """Coordinated dropout for sequential autoencoders.
    
    Masks input data to the encoder while blocking gradients from the
    corresponding output bins. This forces the model to reconstruct
    dropped bins from latent dynamics rather than memorizing input→output.
    
    Adapted from lfads-torch/lfads_torch/modules/augmentations.py.
    
    Parameters
    ----------
    cd_rate : float
        Fraction of timesteps to drop from the encoder input (0.0 = disabled).
    cd_pass_rate : float
        Fraction of dropped timesteps where gradients are still passed
        through (prevents total gradient starvation at high cd_rate).
    ic_enc_seq_len : int
        Number of initial timesteps to protect from dropout (IC segment).
        For full-segment encoding set to 0.
    """
    def __init__(self, cd_rate: float = 0.0, cd_pass_rate: float = 0.5,
                 ic_enc_seq_len: int = 0):
        self.cd_rate = cd_rate
        self.cd_pass_rate = cd_pass_rate
        self.ic_enc_seq_len = ic_enc_seq_len
        self._grad_masks = []

    def mask_input(self, x: torch.Tensor) -> torch.Tensor:
        """Apply coordinated dropout to encoder input.
        
        Parameters
        ----------
        x : (B, T, n_x) encoder input (spike counts)
        
        Returns
        -------
        masked_x : (B, T, n_x) with dropped timesteps zeroed & scaled
        """
        if self.cd_rate <= 0 or not x.requires_grad:
            self._grad_masks.append(torch.ones(x.shape[:2], device=x.device))
            return x

        B, T, C = x.shape
        device = x.device

        # Protect IC segment
        unmaskable = x[:, :self.ic_enc_seq_len, :]
        maskable = x[:, self.ic_enc_seq_len:, :]
        T_mask = maskable.shape[1]

        # Sample input mask (1 = keep, 0 = drop)
        cd_mask = torch.bernoulli(
            torch.full((B, T_mask), 1 - self.cd_rate, device=device)
        ).unsqueeze(-1)  # (B, T_mask, 1)

        # Sample pass mask (for gradient blocking)
        pass_mask = torch.bernoulli(
            torch.full((B, T_mask), self.cd_pass_rate, device=device)
        ).unsqueeze(-1)

        # Gradient mask: block grads for dropped bins (unless pass_mask says otherwise)
        grad_mask = torch.logical_or(
            cd_mask.bool(), pass_mask.bool()
        ).float().squeeze(-1)  # (B, T_mask)

        # Prepend ones for IC segment
        full_grad_mask = torch.cat([
            torch.ones(B, self.ic_enc_seq_len, device=device),
            grad_mask
        ], dim=1)  # (B, T)
        self._grad_masks.append(full_grad_mask)

        # Mask and rescale encoder input
        masked = maskable * cd_mask / max(1 - self.cd_rate, 1e-8)
        masked_x = torch.cat([unmaskable, masked], dim=1)
        return masked_x

    def mask_loss(self, per_bin_loss: torch.Tensor) -> torch.Tensor:
        """Block gradients from dropped bins in the reconstruction loss.
        
        Parameters
        ----------
        per_bin_loss : (B, T, n_x) or (B, T) per-bin loss
        
        Returns
        -------
        masked_loss : same shape, with grads blocked for dropped bins
        """
        grad_mask = self._grad_masks.pop(0)
        if per_bin_loss.dim() == 3:
            grad_mask = grad_mask.unsqueeze(-1)  # (B, T, 1)

        grad_loss = per_bin_loss * grad_mask
        nograd_loss = (per_bin_loss * (1 - grad_mask)).detach()
        return grad_loss + nograd_loss

    def reset(self):
        self._grad_masks.clear()


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
    """Load spike-sorted data, compute z-score stats, chop into segments."""
    with h5py.File(data_path, "r") as f:
        grp = f["placement_%d" % placement]
        x_sorted = grp["x_sorted"][:]            # (n_trials, max_n, n_bins) int16
        n_sorted = grp["n_sorted_per_trial"][:]   # (n_trials,)
        u_1ms = f["u"][:]                         # (n_trials, 2, n_bins*10)

    n_trials, max_n, n_bins = x_sorted.shape
    n_u = u_1ms.shape[1]

    # Convert counts to firing rates (Hz)
    x_rates = x_sorted.astype(np.float32) / BIN_WIDTH_S  # (trials, neurons, bins)
    x_rates = x_rates.transpose(0, 2, 1)  # (trials, time, neurons)

    # Downsample u from 1kHz to 100Hz (10ms bins)
    u_10ms = u_1ms.reshape(n_trials, n_u, n_bins, 10).mean(axis=-1)
    u_10ms = u_10ms.transpose(0, 2, 1).astype(np.float32)  # (trials, time, n_u)

    # Raw counts for Poisson loss
    x_counts = x_sorted.astype(np.float32).transpose(0, 2, 1)  # (trials, time, neurons)

    # Neuron masks
    neuron_masks = np.zeros((n_trials, max_n), dtype=np.float32)
    for i in range(n_trials):
        neuron_masks[i, :n_sorted[i]] = 1.0

    # Train/test split
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_trials)
    test_idx = perm[:n_test_trials]
    train_idx = perm[n_test_trials:]

    # Per-neuron z-score stats from training data
    x_train_rates = x_rates[train_idx]  # (n_train, time, neurons)
    m_train = neuron_masks[train_idx]
    x_masked = x_train_rates * m_train[:, None, :]  # broadcast mask
    count_per_neuron = (m_train.sum(axis=0) * n_bins).clip(1)  # (max_n,)
    x_mean = (x_masked.sum(axis=(0, 1)) / count_per_neuron)  # (max_n,)
    x_sq_mean = ((x_masked ** 2).sum(axis=(0, 1)) / count_per_neuron)
    x_std = np.sqrt(np.maximum(x_sq_mean - x_mean ** 2, 0)) + 1e-8

    # Z-score the rates
    x_zscore = (x_rates - x_mean[None, None, :]) / x_std[None, None, :]
    x_zscore *= neuron_masks[:, None, :]  # zero out padded neurons

    # Normalize u
    u_train = u_10ms[train_idx]
    u_mean = u_train.mean(axis=(0, 1))
    u_std = u_train.std(axis=(0, 1)) + 1e-8
    u_10ms_n = (u_10ms - u_mean[None, None, :]) / u_std[None, None, :]

    # Zero out padded in counts too
    x_counts *= neuron_masks[:, None, :]

    stride = seg_len - seg_overlap

    def chop(indices):
        segs_z, segs_c, segs_u, segs_m = [], [], [], []
        for i in indices:
            for t in range(0, n_bins - seg_len + 1, stride):
                segs_z.append(x_zscore[i, t:t+seg_len])    # z-scored rates
                segs_c.append(x_counts[i, t:t+seg_len])    # raw counts
                segs_u.append(u_10ms_n[i, t:t+seg_len])
                segs_m.append(neuron_masks[i])
        return (np.array(segs_z, dtype=np.float32),
                np.array(segs_c, dtype=np.float32),
                np.array(segs_u, dtype=np.float32),
                np.array(segs_m, dtype=np.float32))

    train_z, train_c, train_u, train_m = chop(train_idx)
    n_total = train_z.shape[0]
    val_mask = np.zeros(n_total, dtype=bool)
    val_mask[::5] = True
    val_z = train_z[val_mask]; val_c = train_c[val_mask]
    val_u = train_u[val_mask]; val_m = train_m[val_mask]
    train_z = train_z[~val_mask]; train_c = train_c[~val_mask]
    train_u = train_u[~val_mask]; train_m = train_m[~val_mask]
    test_z, test_c, test_u, test_m = chop(test_idx)

    stats = {
        "x_mean": x_mean, "x_std": x_std,
        "u_mean": u_mean, "u_std": u_std,
    }
    return (train_z, train_c, train_u, train_m,
            val_z, val_c, val_u, val_m,
            test_z, test_c, test_u, test_m,
            stats, max_n)


# ---------------------------------------------------------------------------
# Deterministic Sequential Autoencoder with Euler ODE dynamics
# ---------------------------------------------------------------------------

class DetSAECaNODE(nn.Module):
    """Deterministic SAE with MLP vector field and optional MSE/Poisson readout."""

    def __init__(self, n_x, n_u, z_dim=32, hidden_dim=128,
                 n_layers=3, encoder_hidden=128, dropout=0.05,
                 loss_type="mse_zscore"):
        super().__init__()
        self.n_x = n_x
        self.n_u = n_u
        self.z_dim = z_dim
        self.loss_type = loss_type

        # Bidirectional GRU encoder
        self.encoder = nn.GRU(
            input_size=n_x,
            hidden_size=encoder_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.ic_linear = nn.Linear(2 * encoder_hidden, z_dim)

        # MLP vector field: f(z, u) -> dz
        layers = [nn.Linear(z_dim + n_u, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, z_dim))
        self.vf = nn.Sequential(*layers)

        # Readout
        self.readout = nn.Linear(z_dim, n_x)

    def forward(self, x_enc, u_seg, dt=0.1):
        """Encode, integrate, decode.

        Parameters
        ----------
        x_enc : (B, T, n_x) — encoder input (z-scored rates or raw counts)
        u_seg : (B, T, n_u) — control inputs
        dt : float — Euler step size

        Returns
        -------
        output : (B, T, n_x) — predicted z-scored rates or log_rates
        latents : (B, T, z_dim)
        """
        B, T, _ = x_enc.shape

        # Encode
        _, h_n = self.encoder(x_enc)  # h_n: (2, B, enc_h)
        h_n = torch.cat([h_n[0], h_n[1]], dim=-1)
        h_n = self.dropout(h_n)
        z0 = self.ic_linear(h_n)  # (B, z_dim)

        # Euler integration
        z = z0
        latents = []
        for t in range(T):
            latents.append(z)
            if t < T - 1:
                zu = torch.cat([z, u_seg[:, t, :]], dim=-1)
                dz = self.vf(zu)
                z = z + dt * dz
        latents = torch.stack(latents, dim=1)  # (B, T, z_dim)

        # Decode
        output = self.readout(latents)  # (B, T, n_x)
        return output, latents


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def masked_mse(pred, target, mask):
    """MSE loss with neuron mask."""
    mask_exp = mask.unsqueeze(1)  # (B, 1, n_x)
    sq_err = (pred - target) ** 2 * mask_exp
    n_valid = mask_exp.sum() * pred.shape[1]
    return sq_err.sum() / n_valid.clamp(min=1)


def masked_poisson_nll(log_rates, counts, mask):
    """Poisson NLL with neuron mask."""
    mask_exp = mask.unsqueeze(1)
    nll = F.poisson_nll_loss(
        log_rates, counts, log_input=True, full=False, reduction="none"
    )
    masked_nll = nll * mask_exp
    n_valid = mask_exp.sum() * log_rates.shape[1]
    return masked_nll.sum() / n_valid.clamp(min=1)


# ---------------------------------------------------------------------------
# Train one config
# ---------------------------------------------------------------------------

def train_one_config(
    config: dict,
    train_z, train_c, train_u, train_m,
    val_z, val_c, val_u, val_m,
    test_z, test_c, test_u, test_m,
    stats: dict,
    n_x: int, n_u: int,
    output_dir: str,
    device: torch.device,
    epochs: int = 300,
    batch_size: int = 64,
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    z_dim = config["z_dim"]
    hidden = config["hidden"]
    lr = config["lr"]
    loss_type = config["loss_type"]
    cd_rate = config.get("cd_rate", 0.0)
    n_layers = config.get("n_layers", 3)
    dropout = config.get("dropout", 0.05)
    dt = config.get("dt", 0.1)

    run_id = "%s_z%d_h%d_lr%.0e_cd%.1f" % (loss_type[:3], z_dim, hidden, lr, cd_rate)
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    model = DetSAECaNODE(
        n_x=n_x, n_u=n_u, z_dim=z_dim,
        hidden_dim=hidden, n_layers=n_layers,
        encoder_hidden=hidden, dropout=dropout,
        loss_type=loss_type,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # Initialize readout bias
    with torch.no_grad():
        if loss_type == "poisson":
            x_mean_counts = torch.tensor(stats["x_mean"] * BIN_WIDTH_S).float()
            model.readout.bias.copy_(torch.log(x_mean_counts.clamp(min=1e-5)).to(device))
        # For MSE z-scored, bias=0 is correct (predicting z-scored values)

    # Coordinated dropout
    cd = CoordinatedDropout(cd_rate=cd_rate, cd_pass_rate=0.5, ic_enc_seq_len=0)

    # Optimizer with separate LR groups
    optimizer = torch.optim.Adam([
        {"params": list(model.encoder.parameters()) + list(model.ic_linear.parameters()),
         "lr": lr, "weight_decay": 1e-5},
        {"params": model.vf.parameters(), "lr": lr, "weight_decay": 1e-5},
        {"params": model.readout.parameters(), "lr": lr, "weight_decay": 0},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Select encoder input and loss target based on loss_type
    if loss_type == "mse_zscore":
        train_enc, val_enc, test_enc = train_z, val_z, test_z
        train_tgt, val_tgt, test_tgt = train_z, val_z, test_z
    else:  # poisson
        train_enc, val_enc, test_enc = train_c, val_c, test_c
        train_tgt, val_tgt, test_tgt = train_c, val_c, test_c

    train_ds = TensorDataset(
        torch.tensor(train_enc), torch.tensor(train_tgt),
        torch.tensor(train_u), torch.tensor(train_m)
    )
    val_ds = TensorDataset(
        torch.tensor(val_enc), torch.tensor(val_tgt),
        torch.tensor(val_u), torch.tensor(val_m)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_epoch = 0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        ep_loss, n_batches = 0.0, 0

        for enc_in, tgt, u_seg, mask in train_loader:
            enc_in = enc_in.to(device)
            tgt = tgt.to(device)
            u_seg = u_seg.to(device)
            mask = mask.to(device)

            # Apply coordinated dropout to encoder input
            enc_masked = cd.mask_input(enc_in)

            optimizer.zero_grad()
            output, _ = model(enc_masked, u_seg, dt=dt)

            if loss_type == "mse_zscore":
                per_bin_loss = (output - tgt) ** 2 * mask.unsqueeze(1)
                if cd_rate > 0:
                    per_bin_loss = cd.mask_loss(per_bin_loss)
                loss = per_bin_loss.sum() / (mask.unsqueeze(1).sum() * output.shape[1]).clamp(1)
            else:  # poisson
                per_bin_loss = F.poisson_nll_loss(
                    output, tgt, log_input=True, full=False, reduction="none"
                ) * mask.unsqueeze(1)
                if cd_rate > 0:
                    per_bin_loss = cd.mask_loss(per_bin_loss)
                loss = per_bin_loss.sum() / (mask.unsqueeze(1).sum() * output.shape[1]).clamp(1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss += loss.item()
            n_batches += 1

        scheduler.step()
        cd.reset()
        avg_loss = ep_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        # Validation (no CD)
        model.eval()
        v_loss, v_n = 0.0, 0
        with torch.no_grad():
            for enc_in, tgt, u_seg, mask in val_loader:
                enc_in = enc_in.to(device)
                tgt = tgt.to(device)
                u_seg = u_seg.to(device)
                mask = mask.to(device)
                output, _ = model(enc_in, u_seg, dt=dt)
                if loss_type == "mse_zscore":
                    l = masked_mse(output, tgt, mask).item()
                else:
                    l = masked_poisson_nll(output, tgt, mask).item()
                v_loss += l
                v_n += 1
        val_loss = v_loss / max(v_n, 1)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "config": config, "stats": {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in stats.items()},
                "n_x": n_x, "n_u": n_u,
                "epoch": epoch, "val_loss": val_loss,
                "n_params": n_params,
            }, os.path.join(run_dir, "best_model.pt"))

        if epoch % 50 == 0 or epoch == epochs - 1:
            elapsed = time.time() - t0
            print("  [%s] Epoch %4d/%d | train=%.4f | val=%.4f | best=%.4f@%d | %.0fs" %
                  (run_id, epoch, epochs, avg_loss, val_loss, best_val, best_epoch, elapsed))

    # -- Evaluate on test segments with SMOOTHED R² --
    ckpt = torch.load(os.path.join(run_dir, "best_model.pt"),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_ds = TensorDataset(
        torch.tensor(test_enc), torch.tensor(test_tgt),
        torch.tensor(test_u), torch.tensor(test_m)
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    x_mean_np = stats["x_mean"]
    x_std_np = stats["x_std"]

    all_pred_rates, all_true_rates, all_masks = [], [], []
    with torch.no_grad():
        for enc_in, tgt, u_seg, mask in test_loader:
            enc_in = enc_in.to(device)
            u_seg = u_seg.to(device)
            output, _ = model(enc_in, u_seg, dt=dt)

            if loss_type == "mse_zscore":
                # Un-z-score predictions to get Hz
                pred_rates = output.cpu().numpy() * x_std_np[None, None, :] + x_mean_np[None, None, :]
                true_rates = tgt.numpy() * x_std_np[None, None, :] + x_mean_np[None, None, :]
            else:
                pred_rates = torch.exp(output).cpu().numpy() / BIN_WIDTH_S
                true_rates = tgt.numpy() / BIN_WIDTH_S

            all_pred_rates.append(pred_rates)
            all_true_rates.append(true_rates)
            all_masks.append(mask.numpy())

    preds = np.concatenate(all_pred_rates, axis=0)  # (n_segs, T, n_x)
    trues = np.concatenate(all_true_rates, axis=0)
    masks_r = np.concatenate(all_masks, axis=0)     # (n_segs, n_x)

    # Smooth both predictions and ground truth for R² (50ms Gaussian)
    preds_smooth = gaussian_filter1d(preds, sigma=SMOOTH_SIGMA, axis=1)
    trues_smooth = gaussian_filter1d(trues, sigma=SMOOTH_SIGMA, axis=1)

    # Flatten: (n_segs * T, n_x)
    P = preds_smooth.reshape(-1, n_x)
    T_arr = trues_smooth.reshape(-1, n_x)
    M = np.repeat(masks_r, preds.shape[1], axis=0)  # (n_segs * T, n_x)

    # Masked R²
    diff_sq = (T_arr - P) ** 2 * M
    resid = diff_sq.sum()
    mean_per_n = (T_arr * M).sum(axis=0) / M.sum(axis=0).clip(1)
    ss_tot = (((T_arr - mean_per_n[None, :]) ** 2) * M).sum()
    r2_smooth = 1 - resid / ss_tot

    # Also compute R² on raw (unsmoothed) for comparison
    P_raw = preds.reshape(-1, n_x)
    T_raw = trues.reshape(-1, n_x)
    diff_raw = (T_raw - P_raw) ** 2 * M
    resid_raw = diff_raw.sum()
    mean_raw = (T_raw * M).sum(axis=0) / M.sum(axis=0).clip(1)
    ss_tot_raw = (((T_raw - mean_raw[None, :]) ** 2) * M).sum()
    r2_raw = 1 - resid_raw / ss_tot_raw

    mse = diff_raw.sum() / M.sum()

    elapsed = time.time() - t0
    results = {
        "r2_smooth": float(r2_smooth),
        "r2_raw": float(r2_raw),
        "mse": float(mse),
        "best_val_loss": float(best_val),
        "best_epoch": best_epoch,
        "n_params": n_params,
        "training_time_s": elapsed,
        "config": config,
        "run_id": run_id,
    }
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot training curve
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(history["train_loss"], label="Train")
    ax.plot(history["val_loss"], label="Val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("%s | R²(smooth)=%.4f  R²(raw)=%.4f" % (run_id, r2_smooth, r2_raw))
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "training_curve.png"), dpi=100)
    plt.close()

    return results


# ---------------------------------------------------------------------------
# GPU worker
# ---------------------------------------------------------------------------

def gpu_worker(gpu_id, configs, shared_data, output_dir, epochs, batch_size, seed):
    device = torch.device("cuda:%d" % gpu_id)
    (train_z, train_c, train_u, train_m,
     val_z, val_c, val_u, val_m,
     test_z, test_c, test_u, test_m,
     stats, n_x, n_u) = shared_data

    results = []
    for config in configs:
        print("\n[GPU %d] Starting %s" % (gpu_id, config))
        try:
            r = train_one_config(
                config=config,
                train_z=train_z, train_c=train_c, train_u=train_u, train_m=train_m,
                val_z=val_z, val_c=val_c, val_u=val_u, val_m=val_m,
                test_z=test_z, test_c=test_c, test_u=test_u, test_m=test_m,
                stats=stats, n_x=n_x, n_u=n_u,
                output_dir=output_dir, device=device,
                epochs=epochs, batch_size=batch_size, seed=seed,
            )
            results.append(r)
            print("[GPU %d] %s -> R²(sm)=%.4f R²(raw)=%.4f (val=%.4f)" %
                  (gpu_id, r["run_id"], r["r2_smooth"], r["r2_raw"], r["best_val_loss"]))
        except Exception as e:
            print("[GPU %d] FAILED %s: %s" % (gpu_id, config, e))
            import traceback
            traceback.print_exc()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    n_gpus = len(gpu_ids)

    print("=" * 60)
    print("Det SAE CA-NODE Sweep v2 (MSE+Poisson, +CoordDropout)")
    print("  GPUs: %s (%d workers)" % (args.gpus, n_gpus))
    print("  Epochs: %d, Batch size: %d" % (args.epochs, args.batch_size))
    print("=" * 60)

    print("\nLoading spiking data (trialized segments)...")
    (train_z, train_c, train_u, train_m,
     val_z, val_c, val_u, val_m,
     test_z, test_c, test_u, test_m,
     stats, n_x) = load_spiking_data_trialized(
        args.data, placement=args.placement,
        seg_len=args.seg_len, seg_overlap=args.seg_overlap,
    )
    n_u = train_u.shape[2]
    print("  Train: %d, Val: %d, Test: %d segs" %
          (train_z.shape[0], val_z.shape[0], test_z.shape[0]))
    print("  n_x=%d, n_u=%d, seg_len=%d" % (n_x, n_u, args.seg_len))

    # Build sweep grid
    configs = []
    for loss_type, z_dim, hidden, lr, cd_rate in itertools.product(
        ["mse_zscore", "poisson"],
        [32, 64],
        [128, 256],
        [5e-4, 1e-3],
        [0.0, 0.3],
    ):
        configs.append({
            "z_dim": z_dim, "hidden": hidden, "lr": lr,
            "loss_type": loss_type, "cd_rate": cd_rate,
            "n_layers": 3, "dropout": 0.05, "dt": 0.1,
        })

    print("\n  Sweep: %d configs across %d GPUs" % (len(configs), n_gpus))

    # Distribute round-robin
    gpu_configs = {g: [] for g in gpu_ids}
    for i, c in enumerate(configs):
        gpu_configs[gpu_ids[i % n_gpus]].append(c)

    shared_data = (train_z, train_c, train_u, train_m,
                   val_z, val_c, val_u, val_m,
                   test_z, test_c, test_u, test_m,
                   stats, n_x, n_u)

    import concurrent.futures
    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_gpus) as executor:
        futures = []
        for gpu_id in gpu_ids:
            if gpu_configs[gpu_id]:
                f = executor.submit(
                    gpu_worker, gpu_id, gpu_configs[gpu_id], shared_data,
                    args.output_dir, args.epochs, args.batch_size, args.seed,
                )
                futures.append(f)
        for f in concurrent.futures.as_completed(futures):
            all_results.extend(f.result())

    all_results.sort(key=lambda r: r["r2_smooth"], reverse=True)

    print("\n" + "=" * 70)
    print("SWEEP RESULTS (sorted by R²_smooth)")
    print("=" * 70)
    for r in all_results:
        print("  %-30s  R²_sm=%.4f  R²_raw=%.4f  val=%.4f  time=%.0fs" %
              (r["run_id"], r["r2_smooth"], r["r2_raw"],
               r["best_val_loss"], r["training_time_s"]))

    summary = {
        "best": all_results[0] if all_results else None,
        "all_results": all_results,
        "sweep_config": {
            "loss_types": ["mse_zscore", "poisson"],
            "z_dims": [32, 64], "hiddens": [128, 256],
            "lrs": [5e-4, 1e-3], "cd_rates": [0.0, 0.3],
            "n_configs": len(configs), "epochs": args.epochs,
            "architecture": "DetSAE_CaNODE_v2",
            "fixes": ["z-scored MSE", "coordinated_dropout", "smoothed_R2"],
        },
    }
    with open(os.path.join(args.output_dir, "sweep_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if all_results:
        print("\nBest: %s -> R²(smooth)=%.4f" %
              (all_results[0]["run_id"], all_results[0]["r2_smooth"]))
    print("Results saved to %s/" % args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Det SAE CA-NODE sweep v2 (MSE+Poisson, +CD)")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--placement", type=int, default=0)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seg-len", type=int, default=100)
    parser.add_argument("--seg-overlap", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    add_preflight_args(parser)
    args = parser.parse_args()
    validate_preflight(args)
    main(args)
