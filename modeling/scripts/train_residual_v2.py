#!/usr/bin/env python
"""Experiment 18: Same-Architecture Residual Correction v2.

Fixes the cross-architecture issue in Exp 12 by using the CAUSAL model's own
ODE + decoder pipeline. A small ResidualMLP learns to correct z0_causal using
delayed acausal information, then runs through the CAUSAL ODE + decoder.

Architecture:
  Frozen: causal encoder, acausal encoder, causal ODE, causal decoder
  Trainable: ResidualMLP(z0_causal, z0_acausal_delayed) -> delta_z0
  Forward: z0_corrected = z0_causal + delta_z0 -> causal ODE -> causal decoder -> x_hat
  Loss: MSE(x_hat, x_true)

The MLP's last layer is zero-initialized so it starts as identity (no correction).
Gradients flow through the frozen ODE (weights frozen via requires_grad=False,
but computation graph remains intact for backprop to the MLP).
"""

from __future__ import annotations

import argparse
import os
import time

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint

torch.set_num_threads(4)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "/snel/home/cbwash2/cleo-worktrees/residual-v2")

from modeling.data import load_trials_h5
from modeling.models.latent_node import LatentNeuralODE
from modeling.models.latent_canode import LatentControlAffineODE
from modeling.config import DEFAULT_SEED, DEFAULT_TEST_TRIALS, set_seed


# ============================================================================
# Residual MLP
# ============================================================================

class ResidualMLP(nn.Module):
    """Learns a correction delta_z0 from concatenated (z0_causal, z0_acausal_delayed).

    Last layer initialized to zero so the initial output is zero (identity).
    """

    def __init__(self, z_dim: int, hidden_dim: int = 128):
        super().__init__()
        input_dim = z_dim * 2  # concat of z0_causal and z0_acausal_delayed
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
        )
        # Zero-init last layer for identity start
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z0_causal: torch.Tensor, z0_acausal_delayed: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z0_causal, z0_acausal_delayed], dim=-1)
        return self.net(x)


# ============================================================================
# Dataset: extracts paired windows for end-to-end training
# ============================================================================

class ResidualV2Dataset(Dataset):
    """Pre-extracts all needed windows for the residual correction training.

    For each valid time point t:
      - causal_x_past: x_causal_norm[:, t-200:t].T  -> (200, n_x)
      - causal_u_past: u_causal_norm[:, t-200:t].T  -> (200, n_u)
      - acausal_x_win: x_acausal_norm[:, t-lag-100:t-lag+100].T -> (200, n_x)
      - acausal_u_win: u_acausal_norm[:, t-lag-100:t-lag+100].T -> (200, n_u)
      - future_u: u_causal_norm[:, t:t+200].T -> (200, n_u)
      - future_x: x_causal_norm[:, t:t+200].T -> (200, n_x)  (target)
    """

    def __init__(
        self,
        x_trials: np.ndarray,   # (n_trials, n_channels, n_steps)
        u_trials: np.ndarray,   # (n_trials, n_inputs, n_steps)
        causal_x_mean: np.ndarray,
        causal_x_std: np.ndarray,
        causal_u_mean: np.ndarray,
        causal_u_std: np.ndarray,
        acausal_x_mean: np.ndarray,
        acausal_x_std: np.ndarray,
        acausal_u_mean: np.ndarray,
        acausal_u_std: np.ndarray,
        dt: float,
        causal_past_window: int = 200,
        acausal_half_window: int = 100,
        future_window: int = 200,
        lag: int = 100,
        stride: int = 50,
    ):
        super().__init__()
        self.dt = dt
        self.future_window = future_window

        causal_x_past_list = []
        causal_u_past_list = []
        acausal_x_win_list = []
        acausal_u_win_list = []
        future_u_list = []
        future_x_list = []

        n_trials = x_trials.shape[0]
        for trial_idx in range(n_trials):
            x_trial = x_trials[trial_idx]  # (n_channels, n_steps)
            u_trial = u_trials[trial_idx]  # (n_inputs, n_steps)
            T = x_trial.shape[1]

            # Normalize with respective stats
            x_causal_n = (x_trial - causal_x_mean) / causal_x_std
            u_causal_n = (u_trial - causal_u_mean) / causal_u_std
            x_acausal_n = (x_trial - acausal_x_mean) / acausal_x_std
            u_acausal_n = (u_trial - acausal_u_mean) / acausal_u_std

            # Valid range: need causal_past_window before t, lag+acausal_hw before t,
            # and future_window after t
            t_start = max(causal_past_window, lag + acausal_half_window)
            t_end = T - future_window

            if t_start >= t_end:
                continue

            for t in range(t_start, t_end, stride):
                # Causal encoder input: x[t-200:t]
                cx_past = x_causal_n[:, t - causal_past_window:t].T  # (200, n_x)
                cu_past = u_causal_n[:, t - causal_past_window:t].T  # (200, n_u)

                # Acausal encoder input: x[t-lag-100:t-lag+100]
                t_delayed = t - lag
                ax_win = x_acausal_n[:, t_delayed - acausal_half_window:t_delayed + acausal_half_window].T  # (200, n_x)
                au_win = u_acausal_n[:, t_delayed - acausal_half_window:t_delayed + acausal_half_window].T  # (200, n_u)

                # Future (for ODE integration and loss) -- use causal normalization
                fu = u_causal_n[:, t:t + future_window].T   # (200, n_u)
                fx = x_causal_n[:, t:t + future_window].T   # (200, n_x) -- target

                causal_x_past_list.append(cx_past)
                causal_u_past_list.append(cu_past)
                acausal_x_win_list.append(ax_win)
                acausal_u_win_list.append(au_win)
                future_u_list.append(fu)
                future_x_list.append(fx)

        self.causal_x_past = torch.tensor(np.array(causal_x_past_list), dtype=torch.float32)
        self.causal_u_past = torch.tensor(np.array(causal_u_past_list), dtype=torch.float32)
        self.acausal_x_win = torch.tensor(np.array(acausal_x_win_list), dtype=torch.float32)
        self.acausal_u_win = torch.tensor(np.array(acausal_u_win_list), dtype=torch.float32)
        self.future_u = torch.tensor(np.array(future_u_list), dtype=torch.float32)
        self.future_x = torch.tensor(np.array(future_x_list), dtype=torch.float32)
        self.t_future = torch.arange(future_window, dtype=torch.float32) * dt

        print(f"  ResidualV2Dataset: {len(self)} windows from {n_trials} trials")

    def __len__(self):
        return self.causal_x_past.shape[0]

    def __getitem__(self, idx):
        return (
            self.causal_x_past[idx],
            self.causal_u_past[idx],
            self.acausal_x_win[idx],
            self.acausal_u_win[idx],
            self.future_u[idx],
            self.future_x[idx],
        )


# ============================================================================
# Training loop
# ============================================================================

def train_residual_v2(
    residual_mlp: ResidualMLP,
    causal_model: LatentControlAffineODE,
    acausal_model: LatentNeuralODE,
    dataset: ResidualV2Dataset,
    n_epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 64,
    val_fraction: float = 0.1,
    device: str = "cuda",
    method: str = "dopri5",
    seed: int = DEFAULT_SEED,
) -> dict:
    """Train the ResidualMLP end-to-end through the frozen causal ODE+decoder."""
    residual_mlp = residual_mlp.to(device)
    causal_model = causal_model.to(device)
    acausal_model = acausal_model.to(device)

    # Freeze both pretrained models entirely
    for p in causal_model.parameters():
        p.requires_grad_(False)
    for p in acausal_model.parameters():
        p.requires_grad_(False)
    causal_model.eval()
    acausal_model.eval()

    # Train/val split
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            pin_memory=True)

    optimizer = torch.optim.Adam(residual_mlp.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = {"train_loss": [], "val_loss": []}
    t_future = dataset.t_future.to(device)

    print(f"  Dataset: {len(dataset)} windows, {n_train} train, {n_val} val")
    print(f"  Batch size: {batch_size}, batches/epoch: {len(train_loader)}")

    for epoch in range(n_epochs):
        # --- Training ---
        residual_mlp.train()
        train_losses = []

        for batch in train_loader:
            cx_past, cu_past, ax_win, au_win, fu, fx_target = [b.to(device) for b in batch]

            # 1. Encode with both encoders (no grad for encoders)
            with torch.no_grad():
                mu_causal, _ = causal_model.encoder(cx_past, cu_past)
                z0_causal = mu_causal  # (B, z_dim)

                mu_acausal, _ = acausal_model.encoder(ax_win, au_win)
                z0_acausal_delayed = mu_acausal  # (B, z_dim)

            # Detach to be sure (already no_grad above, but explicit)
            z0_causal = z0_causal.detach()
            z0_acausal_delayed = z0_acausal_delayed.detach()

            # 2. Compute correction (this IS differentiable)
            delta_z0 = residual_mlp(z0_causal, z0_acausal_delayed)  # (B, z_dim)
            z0_corrected = z0_causal + delta_z0  # (B, z_dim)

            # 3. Run through CAUSAL ODE (frozen weights, but gradients flow through z0)
            causal_model.set_input(t_future, fu)
            z_seq = odeint(causal_model.ode_func, z0_corrected, t_future,
                          method=method, rtol=1e-4, atol=1e-5)  # (T, B, z_dim)
            z_seq = z_seq.permute(1, 0, 2)  # (B, T, z_dim)

            # 4. Decode with CAUSAL decoder
            x_pred = causal_model.decoder(z_seq)  # (B, T, n_x)

            # 5. Loss
            loss = F.mse_loss(x_pred, fx_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(residual_mlp.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        history["train_loss"].append(avg_train)

        # --- Validation ---
        residual_mlp.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                cx_past, cu_past, ax_win, au_win, fu, fx_target = [b.to(device) for b in batch]

                mu_causal, _ = causal_model.encoder(cx_past, cu_past)
                z0_causal = mu_causal

                mu_acausal, _ = acausal_model.encoder(ax_win, au_win)
                z0_acausal_delayed = mu_acausal

                delta_z0 = residual_mlp(z0_causal, z0_acausal_delayed)
                z0_corrected = z0_causal + delta_z0

                causal_model.set_input(t_future, fu)
                z_seq = odeint(causal_model.ode_func, z0_corrected, t_future,
                              method=method, rtol=1e-4, atol=1e-5)
                z_seq = z_seq.permute(1, 0, 2)
                x_pred = causal_model.decoder(z_seq)

                val_losses.append(F.mse_loss(x_pred, fx_target).item())

        avg_val = np.mean(val_losses) if val_losses else float("nan")
        history["val_loss"].append(avg_val)
        scheduler.step()

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch:4d}/{n_epochs} | "
                  f"train: {avg_train:.6f} | val: {avg_val:.6f} | "
                  f"lr: {optimizer.param_groups[0]['lr']:.2e}")

    return history


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate(
    residual_mlp: ResidualMLP,
    causal_model: LatentControlAffineODE,
    acausal_model: LatentNeuralODE,
    x_test_trials: np.ndarray,
    u_test_trials: np.ndarray,
    causal_x_mean: np.ndarray,
    causal_x_std: np.ndarray,
    causal_u_mean: np.ndarray,
    causal_u_std: np.ndarray,
    acausal_x_mean: np.ndarray,
    acausal_x_std: np.ndarray,
    acausal_u_mean: np.ndarray,
    acausal_u_std: np.ndarray,
    dt: float,
    causal_past_window: int = 200,
    acausal_half_window: int = 100,
    future_window: int = 200,
    lag: int = 100,
    stride: int = 100,
    device: str = "cuda",
    method: str = "dopri5",
    n_train: int = 40,
):
    """Evaluate both causal baseline and corrected model on test trials."""
    residual_mlp.eval()
    causal_model.eval()
    acausal_model.eval()

    n_test = x_test_trials.shape[0]
    t_future = torch.arange(future_window, dtype=torch.float32, device=device) * dt

    per_trial_baseline_r2 = []
    per_trial_corrected_r2 = []
    per_trial_baseline_mse = []
    per_trial_corrected_mse = []
    baseline_times = []
    corrected_times = []

    for ti in range(n_test):
        x_trial = x_test_trials[ti]  # (n_channels, n_steps)
        u_trial = u_test_trials[ti]  # (n_inputs, n_steps)
        T = x_trial.shape[1]

        # Normalize
        x_causal_n = (x_trial - causal_x_mean) / causal_x_std
        u_causal_n = (u_trial - causal_u_mean) / causal_u_std
        x_acausal_n = (x_trial - acausal_x_mean) / acausal_x_std
        u_acausal_n = (u_trial - acausal_u_mean) / acausal_u_std

        t_start = max(causal_past_window, lag + acausal_half_window)
        t_end = T - future_window

        if t_start >= t_end:
            continue

        baseline_window_mses = []
        corrected_window_mses = []
        trial_var = np.var(x_causal_n)

        for t in range(t_start, t_end, stride):
            # Prepare inputs
            cx_past = torch.tensor(
                x_causal_n[:, t - causal_past_window:t].T, dtype=torch.float32
            ).unsqueeze(0).to(device)
            cu_past = torch.tensor(
                u_causal_n[:, t - causal_past_window:t].T, dtype=torch.float32
            ).unsqueeze(0).to(device)

            t_delayed = t - lag
            ax_win = torch.tensor(
                x_acausal_n[:, t_delayed - acausal_half_window:t_delayed + acausal_half_window].T,
                dtype=torch.float32
            ).unsqueeze(0).to(device)
            au_win = torch.tensor(
                u_acausal_n[:, t_delayed - acausal_half_window:t_delayed + acausal_half_window].T,
                dtype=torch.float32
            ).unsqueeze(0).to(device)

            fu = torch.tensor(
                u_causal_n[:, t:t + future_window].T, dtype=torch.float32
            ).unsqueeze(0).to(device)

            x_true = x_causal_n[:, t:t + future_window]  # (n_channels, future_window)

            # --- Baseline: causal only ---
            t0 = time.perf_counter()
            mu_causal, _ = causal_model.encoder(cx_past, cu_past)
            z0_causal = mu_causal

            causal_model.set_input(t_future, fu)
            z_seq = odeint(causal_model.ode_func, z0_causal, t_future,
                          method=method, rtol=1e-4, atol=1e-5)
            z_seq_b = z_seq.permute(1, 0, 2)
            x_pred_baseline = causal_model.decoder(z_seq_b)
            x_pred_baseline_np = x_pred_baseline[0].cpu().numpy().T  # (n_channels, T)
            t1 = time.perf_counter()
            baseline_times.append((t1 - t0) * 1000.0)

            baseline_mse = np.mean((x_pred_baseline_np - x_true) ** 2)
            baseline_window_mses.append(baseline_mse)

            # --- Corrected ---
            t0 = time.perf_counter()
            mu_acausal, _ = acausal_model.encoder(ax_win, au_win)
            z0_acausal_delayed = mu_acausal

            delta_z0 = residual_mlp(z0_causal, z0_acausal_delayed)
            z0_corrected = z0_causal + delta_z0

            causal_model.set_input(t_future, fu)
            z_seq_c = odeint(causal_model.ode_func, z0_corrected, t_future,
                            method=method, rtol=1e-4, atol=1e-5)
            z_seq_c = z_seq_c.permute(1, 0, 2)
            x_pred_corrected = causal_model.decoder(z_seq_c)
            x_pred_corrected_np = x_pred_corrected[0].cpu().numpy().T
            t1 = time.perf_counter()
            corrected_times.append((t1 - t0) * 1000.0)

            corrected_mse = np.mean((x_pred_corrected_np - x_true) ** 2)
            corrected_window_mses.append(corrected_mse)

        trial_baseline_mse = np.mean(baseline_window_mses)
        trial_corrected_mse = np.mean(corrected_window_mses)
        trial_baseline_r2 = 1 - trial_baseline_mse / trial_var
        trial_corrected_r2 = 1 - trial_corrected_mse / trial_var

        per_trial_baseline_r2.append(trial_baseline_r2)
        per_trial_corrected_r2.append(trial_corrected_r2)
        per_trial_baseline_mse.append(trial_baseline_mse)
        per_trial_corrected_mse.append(trial_corrected_mse)

        print(f"  Trial {n_train + ti}: Baseline R2={trial_baseline_r2:.4f}, "
              f"Corrected R2={trial_corrected_r2:.4f}, "
              f"Delta={trial_corrected_r2 - trial_baseline_r2:+.4f}")

    return {
        "baseline_r2": float(np.mean(per_trial_baseline_r2)),
        "baseline_r2_std": float(np.std(per_trial_baseline_r2)),
        "baseline_mse": float(np.mean(per_trial_baseline_mse)),
        "corrected_r2": float(np.mean(per_trial_corrected_r2)),
        "corrected_r2_std": float(np.std(per_trial_corrected_r2)),
        "corrected_mse": float(np.mean(per_trial_corrected_mse)),
        "avg_inference_ms_baseline": float(np.mean(baseline_times)),
        "avg_inference_ms_corrected": float(np.mean(corrected_times)),
        "per_trial_baseline_r2": per_trial_baseline_r2,
        "per_trial_corrected_r2": per_trial_corrected_r2,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Experiment 18: Residual Correction v2 (Same Architecture)")
    parser.add_argument("--data", type=str,
                        default="/snel/home/cbwash2/cleo/data/training_trials.h5")
    parser.add_argument("--causal-ckpt", type=str,
                        default="/snel/home/cbwash2/cleo/results/causal_z64_pw200_h128_lr3e4/model.pt")
    parser.add_argument("--acausal-ckpt", type=str,
                        default="/snel/home/cbwash2/cleo/results/sweep_latent_node_40_10/run_05_z64_h256_l2_lr0.0005_dopri5.pt")
    parser.add_argument("--lag", type=int, default=100,
                        help="Acausal delay in timesteps (default: 100 = 100ms)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128,
                        help="Hidden dim for residual MLP")
    parser.add_argument("--causal-past-window", type=int, default=200)
    parser.add_argument("--future-window", type=int, default=200)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--n-test-trials", type=int, default=DEFAULT_TEST_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--method", type=str, default="dopri5")
    parser.add_argument("--output-dir", type=str, default="results/residual_v2")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # ---- Load data ----
    print("\nLoading data...")
    data = load_trials_h5(args.data)
    x_all, u_all = data["x"], data["u"]
    dt = float(data["dt"])
    n_trials, n_channels, n_steps = x_all.shape
    n_inputs = u_all.shape[1]
    print(f"  {n_trials} trials, {n_channels} channels, {n_steps} steps, dt={dt}")

    n_test = args.n_test_trials
    n_train = n_trials - n_test
    x_train_trials = x_all[:n_train]
    u_train_trials = u_all[:n_train]
    x_test_trials = x_all[n_train:]
    u_test_trials = u_all[n_train:]

    # ---- Load causal model ----
    print("\nLoading causal model...")
    causal_ckpt = torch.load(args.causal_ckpt, map_location="cpu", weights_only=False)
    causal_model = LatentControlAffineODE(
        n_x=causal_ckpt["n_x"],
        n_u=causal_ckpt["n_u"],
        z_dim=causal_ckpt["z_dim"],
        hidden_dim=causal_ckpt["hidden"],
        n_layers=causal_ckpt["n_layers"],
    )
    causal_model.load_state_dict(causal_ckpt["model_state"])
    print(f"  Causal: z_dim={causal_ckpt['z_dim']}, hidden={causal_ckpt['hidden']}, "
          f"layers={causal_ckpt['n_layers']}")

    causal_x_mean = causal_ckpt["x_mean"]  # (n_channels, 1)
    causal_x_std = causal_ckpt["x_std"]
    causal_u_mean = causal_ckpt["u_mean"]
    causal_u_std = causal_ckpt["u_std"]

    # ---- Load acausal model ----
    print("Loading acausal model...")
    acausal_ckpt = torch.load(args.acausal_ckpt, map_location="cpu", weights_only=False)
    acausal_model = LatentNeuralODE(
        n_x=acausal_ckpt["n_x"],
        n_u=acausal_ckpt["n_u"],
        z_dim=acausal_ckpt["z_dim"],
        hidden_dim=acausal_ckpt["hidden"],
        n_layers=acausal_ckpt["n_layers"],
    )
    acausal_model.load_state_dict(acausal_ckpt["model_state"])
    print(f"  Acausal: z_dim={acausal_ckpt['z_dim']}, hidden={acausal_ckpt['hidden']}, "
          f"layers={acausal_ckpt['n_layers']}")

    acausal_x_mean = acausal_ckpt["x_mean"]
    acausal_x_std = acausal_ckpt["x_std"]
    acausal_u_mean = acausal_ckpt["u_mean"]
    acausal_u_std = acausal_ckpt["u_std"]

    z_dim = causal_ckpt["z_dim"]
    assert z_dim == acausal_ckpt["z_dim"], \
        f"z_dim mismatch: causal={z_dim}, acausal={acausal_ckpt['z_dim']}"

    # ---- Build ResidualMLP ----
    print(f"\nBuilding ResidualMLP: input={z_dim*2}, hidden={args.hidden}, output={z_dim}")
    residual_mlp = ResidualMLP(z_dim=z_dim, hidden_dim=args.hidden)
    n_params = sum(p.numel() for p in residual_mlp.parameters())
    print(f"  Parameters: {n_params:,}")

    # ---- Build dataset ----
    print("\nBuilding training dataset...")
    acausal_hw = 100  # half-window for 200-step acausal window
    dataset = ResidualV2Dataset(
        x_trials=x_train_trials,
        u_trials=u_train_trials,
        causal_x_mean=causal_x_mean,
        causal_x_std=causal_x_std,
        causal_u_mean=causal_u_mean,
        causal_u_std=causal_u_std,
        acausal_x_mean=acausal_x_mean,
        acausal_x_std=acausal_x_std,
        acausal_u_mean=acausal_u_mean,
        acausal_u_std=acausal_u_std,
        dt=dt,
        causal_past_window=args.causal_past_window,
        acausal_half_window=acausal_hw,
        future_window=args.future_window,
        lag=args.lag,
        stride=args.stride,
    )

    # ---- Train ----
    print("\n" + "=" * 60)
    print("Training Residual MLP (end-to-end through frozen causal ODE)")
    print("=" * 60)
    t_train_start = time.time()
    history = train_residual_v2(
        residual_mlp=residual_mlp,
        causal_model=causal_model,
        acausal_model=acausal_model,
        dataset=dataset,
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
        method=args.method,
        seed=args.seed,
    )
    train_time_s = time.time() - t_train_start
    print(f"\nTraining completed in {train_time_s:.1f}s")

    # ---- Evaluate ----
    print("\n" + "=" * 60)
    print("Evaluating on test trials...")
    print("=" * 60)

    results = evaluate(
        residual_mlp=residual_mlp,
        causal_model=causal_model,
        acausal_model=acausal_model,
        x_test_trials=x_test_trials,
        u_test_trials=u_test_trials,
        causal_x_mean=causal_x_mean,
        causal_x_std=causal_x_std,
        causal_u_mean=causal_u_mean,
        causal_u_std=causal_u_std,
        acausal_x_mean=acausal_x_mean,
        acausal_x_std=acausal_x_std,
        acausal_u_mean=acausal_u_mean,
        acausal_u_std=acausal_u_std,
        dt=dt,
        causal_past_window=args.causal_past_window,
        acausal_half_window=acausal_hw,
        future_window=args.future_window,
        lag=args.lag,
        stride=100,  # eval stride
        device=device,
        method=args.method,
        n_train=n_train,
    )

    # ---- Print summary ----
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    improvement = results["corrected_r2"] - results["baseline_r2"]
    print(f"  Causal baseline: R2 = {results['baseline_r2']:.4f} +/- {results['baseline_r2_std']:.4f}  "
          f"MSE = {results['baseline_mse']:.6f}  "
          f"Inference = {results['avg_inference_ms_baseline']:.2f} ms")
    print(f"  Corrected:       R2 = {results['corrected_r2']:.4f} +/- {results['corrected_r2_std']:.4f}  "
          f"MSE = {results['corrected_mse']:.6f}  "
          f"Inference = {results['avg_inference_ms_corrected']:.2f} ms")
    print(f"\n  Improvement: {improvement:+.4f} R2")
    print(f"  Training time: {train_time_s:.1f}s")

    # ---- Save ----
    save_path = os.path.join(args.output_dir, "residual_v2.pt")
    torch.save({
        "residual_mlp_state": residual_mlp.state_dict(),
        "z_dim": z_dim,
        "hidden": args.hidden,
        "lag": args.lag,
        "history": history,
        "results": results,
        "args": vars(args),
        "train_time_s": train_time_s,
    }, save_path)
    print(f"  Saved checkpoint to {save_path}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Training curve
    axes[0].semilogy(history["train_loss"], label="Train", alpha=0.8)
    axes[0].semilogy(history["val_loss"], label="Val", alpha=0.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Residual MLP Training (End-to-End)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Per-trial comparison
    trials = range(len(results["per_trial_baseline_r2"]))
    axes[1].plot(list(trials), results["per_trial_baseline_r2"], "o-",
                 label="Causal baseline", alpha=0.7, color="#e74c3c")
    axes[1].plot(list(trials), results["per_trial_corrected_r2"], "s-",
                 label="Corrected", alpha=0.7, color="#2ecc71")
    axes[1].set_xlabel("Test Trial")
    axes[1].set_ylabel("R2")
    axes[1].set_title("Per-Trial R2 Comparison")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Summary bar chart
    methods = ["Causal\nbaseline", "Corrected"]
    r2s = [results["baseline_r2"], results["corrected_r2"]]
    stds = [results["baseline_r2_std"], results["corrected_r2_std"]]
    colors = ["#e74c3c", "#2ecc71"]
    bars = axes[2].bar(methods, r2s, yerr=stds, capsize=5, color=colors, alpha=0.8)
    axes[2].set_ylabel("R2")
    axes[2].set_title(f"Avg R2 (Delta = {improvement:+.4f})")
    for bar, r2 in zip(bars, r2s):
        axes[2].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                     f"{r2:.4f}", ha="center", va="bottom", fontsize=11)
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "residual_v2_results.png")
    fig.savefig(plot_path, dpi=150)
    print(f"  Plot saved to {plot_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
