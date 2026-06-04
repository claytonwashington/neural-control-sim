#!/usr/bin/env python
"""Experiment 7: Delayed Residual Correction.

Train a small residual MLP that predicts the difference between acausal and
causal encoder z0 outputs, then use this to correct the causal model at
deployment.

Strategy:
  - At each time t, the causal encoder sees x(t-pw:t) and produces z0_causal(t)
  - The acausal encoder sees a centered window x(t-lag-hw:t-lag+hw) and produces
    z0_acausal(t-lag) -- this is the acausal z0 available 'lag' steps later
  - We train a small MLP to predict dz0 = z0_acausal(t) - z0_causal(t)
    from inputs [z0_causal(t), z0_acausal(t-lag), u(t-lag:t)]
  - At test time: z0_corrected = z0_causal(t) + MLP(z0_causal(t), z0_acausal(t-lag), u_lag)
  - Then we plug z0_corrected into the acausal ODE + decoder for prediction
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
from torch.utils.data import TensorDataset, DataLoader
from torchdiffeq import odeint

torch.set_num_threads(4)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modeling.data import load_trials_h5
from modeling.models.latent_node import LatentNeuralODE
from modeling.models.latent_canode import LatentControlAffineODE
from modeling.config import DEFAULT_SEED, DEFAULT_TEST_TRIALS, set_seed


# ============================================================================
# Residual MLP
# ============================================================================

class ResidualCorrectionMLP(nn.Module):
    """Small MLP to predict dz0 = z0_acausal(t) - z0_causal(t).

    Input: [z0_causal(t), z0_acausal(t-lag), u_lag_flat]
    Output: dz0(t)  (same dimension as z_dim)
    """

    def __init__(self, z_dim, n_u, lag_steps, hidden_dim=128, dropout=0.1):
        super().__init__()
        input_dim = z_dim + z_dim + n_u * lag_steps
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, z_dim),
        )
        # Initialize last layer near zero for stable start
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z0_causal, z0_acausal_delayed, u_lag_flat):
        x = torch.cat([z0_causal, z0_acausal_delayed, u_lag_flat], dim=-1)
        return self.net(x)


# ============================================================================
# Extract z0 pairs from data
# ============================================================================

@torch.no_grad()
def extract_z0_pairs(
    acausal_model,
    causal_model,
    x_trials,
    u_trials,
    x_mean,
    x_std,
    u_mean,
    u_std,
    dt,
    causal_past_window=200,
    acausal_window=200,
    lag_steps=100,
    stride=50,
    device="cuda",
):
    """Extract paired z0 encodings from both models across all trials."""
    acausal_model.eval()
    causal_model.eval()

    acausal_hw = acausal_window // 2

    all_z0_causal = []
    all_z0_acausal_target = []
    all_z0_acausal_delayed = []
    all_u_lag = []

    n_trials = x_trials.shape[0]

    for trial_idx in range(n_trials):
        x_trial = x_trials[trial_idx]
        u_trial = u_trials[trial_idx]

        x_n = (x_trial - x_mean) / x_std
        u_n = (u_trial - u_mean) / u_std

        T = x_n.shape[1]

        t_start = max(causal_past_window, lag_steps + acausal_hw, acausal_hw)
        t_end = T - acausal_hw

        if t_start >= t_end:
            continue

        for t in range(t_start, t_end, stride):
            # 1. Causal encoder: past window ending at t
            x_past = x_n[:, t - causal_past_window:t].T
            u_past = u_n[:, t - causal_past_window:t].T
            x_past_t = torch.tensor(x_past, dtype=torch.float32).unsqueeze(0).to(device)
            u_past_t = torch.tensor(u_past, dtype=torch.float32).unsqueeze(0).to(device)

            mu_causal, _ = causal_model.encoder(x_past_t, u_past_t)
            z0_causal = mu_causal

            # 2. Acausal encoder at time t (target)
            x_acausal_t = x_n[:, t - acausal_hw:t + acausal_hw].T
            u_acausal_t = u_n[:, t - acausal_hw:t + acausal_hw].T
            x_at = torch.tensor(x_acausal_t, dtype=torch.float32).unsqueeze(0).to(device)
            u_at = torch.tensor(u_acausal_t, dtype=torch.float32).unsqueeze(0).to(device)

            mu_acausal_t, _ = acausal_model.encoder(x_at, u_at)
            z0_acausal_target = mu_acausal_t

            # 3. Acausal encoder at time t-lag (available delayed)
            t_delayed = t - lag_steps
            x_acausal_d = x_n[:, t_delayed - acausal_hw:t_delayed + acausal_hw].T
            u_acausal_d = u_n[:, t_delayed - acausal_hw:t_delayed + acausal_hw].T
            x_ad = torch.tensor(x_acausal_d, dtype=torch.float32).unsqueeze(0).to(device)
            u_ad = torch.tensor(u_acausal_d, dtype=torch.float32).unsqueeze(0).to(device)

            mu_acausal_d, _ = acausal_model.encoder(x_ad, u_ad)
            z0_acausal_delayed = mu_acausal_d

            # 4. u in the lag window: u(t-lag:t)
            u_lag_window = u_n[:, t - lag_steps:t].T
            u_lag_flat = torch.tensor(u_lag_window.flatten(), dtype=torch.float32).unsqueeze(0)

            all_z0_causal.append(z0_causal.cpu())
            all_z0_acausal_target.append(z0_acausal_target.cpu())
            all_z0_acausal_delayed.append(z0_acausal_delayed.cpu())
            all_u_lag.append(u_lag_flat)

        if (trial_idx + 1) % 10 == 0:
            print(f"    Processed {trial_idx + 1}/{n_trials} trials...")

    z0_causal = torch.cat(all_z0_causal, dim=0)
    z0_acausal_target = torch.cat(all_z0_acausal_target, dim=0)
    z0_acausal_delayed = torch.cat(all_z0_acausal_delayed, dim=0)
    u_lag_flat = torch.cat(all_u_lag, dim=0)

    delta_z0 = z0_acausal_target - z0_causal

    print(f"  Extracted {z0_causal.shape[0]} z0 pairs from {n_trials} trials")
    print(f"  z0_causal stats: mean={z0_causal.mean():.4f}, std={z0_causal.std():.4f}")
    print(f"  z0_acausal_target stats: mean={z0_acausal_target.mean():.4f}, std={z0_acausal_target.std():.4f}")
    print(f"  delta_z0 stats: mean={delta_z0.mean():.4f}, std={delta_z0.std():.4f}")
    print(f"  u_lag_flat shape: {u_lag_flat.shape}")

    return z0_causal, z0_acausal_target, z0_acausal_delayed, u_lag_flat, delta_z0


# ============================================================================
# Train the residual MLP
# ============================================================================

def train_residual_mlp(
    mlp,
    z0_causal,
    z0_acausal_delayed,
    u_lag_flat,
    delta_z0,
    n_epochs=200,
    lr=1e-3,
    batch_size=256,
    val_fraction=0.1,
    device="cuda",
    seed=DEFAULT_SEED,
):
    """Train the residual correction MLP."""
    mlp = mlp.to(device)

    dataset = TensorDataset(z0_causal, z0_acausal_delayed, u_lag_flat, delta_z0)
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None

    print(f"  Training residual MLP: {n_train} train, {n_val} val samples")

    for epoch in range(n_epochs):
        mlp.train()
        train_losses = []
        for z0_c, z0_ad, u_l, dz in train_loader:
            z0_c = z0_c.to(device)
            z0_ad = z0_ad.to(device)
            u_l = u_l.to(device)
            dz = dz.to(device)

            dz_pred = mlp(z0_c, z0_ad, u_l)
            loss = F.mse_loss(dz_pred, dz)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        history["train_loss"].append(avg_train)

        mlp.eval()
        val_losses = []
        with torch.no_grad():
            for z0_c, z0_ad, u_l, dz in val_loader:
                z0_c = z0_c.to(device)
                z0_ad = z0_ad.to(device)
                u_l = u_l.to(device)
                dz = dz.to(device)

                dz_pred = mlp(z0_c, z0_ad, u_l)
                val_losses.append(F.mse_loss(dz_pred, dz).item())

        avg_val = np.mean(val_losses) if val_losses else float("nan")
        history["val_loss"].append(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}

        scheduler.step()

        if epoch % 20 == 0 or epoch == n_epochs - 1:
            print(f"    Epoch {epoch:4d}/{n_epochs} | train: {avg_train:.6f} | val: {avg_val:.6f} | "
                  f"lr: {optimizer.param_groups[0]['lr']:.2e}")

    if best_state is not None:
        mlp.load_state_dict(best_state)
        print(f"  Loaded best model with val loss: {best_val_loss:.6f}")

    return history


# ============================================================================
# Evaluate
# ============================================================================

@torch.no_grad()
def evaluate_corrected(
    acausal_model,
    causal_model,
    residual_mlp,
    x_test_trials,
    u_test_trials,
    x_mean,
    x_std,
    u_mean,
    u_std,
    dt,
    causal_past_window=200,
    acausal_window=200,
    future_window=200,
    lag_steps=100,
    stride=100,
    device="cuda",
    method="dopri5",
    n_train=40,
):
    """Evaluate causal-only vs corrected model on test trials."""
    acausal_model.eval()
    causal_model.eval()
    residual_mlp.eval()

    acausal_hw = acausal_window // 2
    n_test = x_test_trials.shape[0]

    causal_only_r2s = []
    causal_only_mses = []
    corrected_r2s = []
    corrected_mses = []
    acausal_r2s = []
    acausal_mses = []
    inference_times_causal = []
    inference_times_corrected = []

    for ti in range(n_test):
        x_trial = x_test_trials[ti]
        u_trial = u_test_trials[ti]

        x_n = (x_trial - x_mean) / x_std
        u_n = (u_trial - u_mean) / u_std

        T = x_n.shape[1]

        t_start = max(causal_past_window, lag_steps + acausal_hw, acausal_hw)
        t_end = T - future_window - acausal_hw

        if t_start >= t_end:
            print(f"  Trial {n_train + ti}: SKIPPED (too short)")
            continue

        trial_causal_mses = []
        trial_corrected_mses = []
        trial_acausal_mses = []

        t_future = torch.arange(future_window, dtype=torch.float32).to(device) * dt

        for t in range(t_start, t_end, stride):
            x_true_future = x_n[:, t:t + future_window]

            # --- Causal encoder ---
            x_past = torch.tensor(
                x_n[:, t - causal_past_window:t].T, dtype=torch.float32
            ).unsqueeze(0).to(device)
            u_past = torch.tensor(
                u_n[:, t - causal_past_window:t].T, dtype=torch.float32
            ).unsqueeze(0).to(device)

            t_start_inf = time.perf_counter()
            mu_causal, _ = causal_model.encoder(x_past, u_past)
            z0_causal = mu_causal

            # --- Causal-only prediction: use causal z0 with acausal ODE+decoder ---
            u_future_tensor = torch.tensor(
                u_n[:, t:t + future_window].T, dtype=torch.float32
            ).unsqueeze(0).to(device)

            acausal_model.ode_func.set_input(t_future, u_future_tensor)
            z_seq_causal = odeint(
                acausal_model.ode_func, z0_causal, t_future,
                method=method, rtol=1e-4, atol=1e-5
            )
            z_seq_causal = z_seq_causal.permute(1, 0, 2)
            x_pred_causal = acausal_model.decoder(z_seq_causal)
            x_pred_causal_np = x_pred_causal[0].cpu().numpy().T
            t_end_inf = time.perf_counter()
            inference_times_causal.append((t_end_inf - t_start_inf) * 1000.0)

            # --- Corrected prediction ---
            t_start_inf2 = time.perf_counter()

            t_delayed = t - lag_steps
            x_acausal_d = torch.tensor(
                x_n[:, t_delayed - acausal_hw:t_delayed + acausal_hw].T, dtype=torch.float32
            ).unsqueeze(0).to(device)
            u_acausal_d = torch.tensor(
                u_n[:, t_delayed - acausal_hw:t_delayed + acausal_hw].T, dtype=torch.float32
            ).unsqueeze(0).to(device)

            mu_acausal_d, _ = acausal_model.encoder(x_acausal_d, u_acausal_d)
            z0_acausal_delayed = mu_acausal_d

            u_lag_window = torch.tensor(
                u_n[:, t - lag_steps:t].T.flatten(), dtype=torch.float32
            ).unsqueeze(0).to(device)

            delta_z0_pred = residual_mlp(z0_causal, z0_acausal_delayed, u_lag_window)
            z0_corrected = z0_causal + delta_z0_pred

            acausal_model.ode_func.set_input(t_future, u_future_tensor)
            z_seq_corrected = odeint(
                acausal_model.ode_func, z0_corrected, t_future,
                method=method, rtol=1e-4, atol=1e-5
            )
            z_seq_corrected = z_seq_corrected.permute(1, 0, 2)
            x_pred_corrected = acausal_model.decoder(z_seq_corrected)
            x_pred_corrected_np = x_pred_corrected[0].cpu().numpy().T
            t_end_inf2 = time.perf_counter()
            inference_times_corrected.append((t_end_inf2 - t_start_inf2) * 1000.0)

            # --- Acausal baseline (oracle) ---
            x_acausal_t = torch.tensor(
                x_n[:, t - acausal_hw:t + acausal_hw].T, dtype=torch.float32
            ).unsqueeze(0).to(device)
            u_acausal_t = torch.tensor(
                u_n[:, t - acausal_hw:t + acausal_hw].T, dtype=torch.float32
            ).unsqueeze(0).to(device)

            mu_acausal_t, _ = acausal_model.encoder(x_acausal_t, u_acausal_t)
            z0_acausal = mu_acausal_t

            acausal_model.ode_func.set_input(t_future, u_future_tensor)
            z_seq_acausal = odeint(
                acausal_model.ode_func, z0_acausal, t_future,
                method=method, rtol=1e-4, atol=1e-5
            )
            z_seq_acausal = z_seq_acausal.permute(1, 0, 2)
            x_pred_acausal = acausal_model.decoder(z_seq_acausal)
            x_pred_acausal_np = x_pred_acausal[0].cpu().numpy().T

            trial_causal_mses.append(np.mean((x_pred_causal_np - x_true_future) ** 2))
            trial_corrected_mses.append(np.mean((x_pred_corrected_np - x_true_future) ** 2))
            trial_acausal_mses.append(np.mean((x_pred_acausal_np - x_true_future) ** 2))

        if not trial_causal_mses:
            continue

        trial_var = np.var(x_n)

        causal_mse = np.mean(trial_causal_mses)
        corrected_mse = np.mean(trial_corrected_mses)
        acausal_mse = np.mean(trial_acausal_mses)

        causal_r2 = 1 - causal_mse / trial_var
        corrected_r2 = 1 - corrected_mse / trial_var
        acausal_r2 = 1 - acausal_mse / trial_var

        causal_only_r2s.append(causal_r2)
        causal_only_mses.append(causal_mse)
        corrected_r2s.append(corrected_r2)
        corrected_mses.append(corrected_mse)
        acausal_r2s.append(acausal_r2)
        acausal_mses.append(acausal_mse)

        print(f"  Trial {n_train + ti}: Causal R2={causal_r2:.4f} | "
              f"Corrected R2={corrected_r2:.4f} | Acausal R2={acausal_r2:.4f}")

    results = {
        "causal_only_r2": float(np.mean(causal_only_r2s)),
        "causal_only_mse": float(np.mean(causal_only_mses)),
        "corrected_r2": float(np.mean(corrected_r2s)),
        "corrected_mse": float(np.mean(corrected_mses)),
        "acausal_r2": float(np.mean(acausal_r2s)),
        "acausal_mse": float(np.mean(acausal_mses)),
        "causal_only_r2_std": float(np.std(causal_only_r2s)),
        "corrected_r2_std": float(np.std(corrected_r2s)),
        "acausal_r2_std": float(np.std(acausal_r2s)),
        "avg_inference_ms_causal": float(np.mean(inference_times_causal)),
        "avg_inference_ms_corrected": float(np.mean(inference_times_corrected)),
        "per_trial_causal_r2": causal_only_r2s,
        "per_trial_corrected_r2": corrected_r2s,
        "per_trial_acausal_r2": acausal_r2s,
    }

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Exp 7: Delayed Residual Correction")
    parser.add_argument("--data", type=str, default="data/training_trials.h5")
    parser.add_argument("--acausal-ckpt", type=str,
                        default="results/sweep_latent_node_40_10/run_05_z64_h256_l2_lr0.0005_dopri5.pt")
    parser.add_argument("--causal-ckpt", type=str,
                        default="results/causal_z64_pw200_h128_lr3e4/model.pt")
    parser.add_argument("--lag-steps", type=int, default=100,
                        help="Lag in timesteps (100 = 100ms at dt=0.001)")
    parser.add_argument("--causal-past-window", type=int, default=200,
                        help="Past window for causal encoder")
    parser.add_argument("--acausal-window", type=int, default=200,
                        help="Window size for acausal encoder")
    parser.add_argument("--future-window", type=int, default=200,
                        help="Future prediction horizon for evaluation")
    parser.add_argument("--extract-stride", type=int, default=50,
                        help="Stride for z0 pair extraction")
    parser.add_argument("--eval-stride", type=int, default=100,
                        help="Stride for evaluation windows")
    parser.add_argument("--mlp-hidden", type=int, default=128)
    parser.add_argument("--mlp-dropout", type=float, default=0.1)
    parser.add_argument("--mlp-epochs", type=int, default=200)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-batch-size", type=int, default=256)
    parser.add_argument("--n-test-trials", type=int, default=DEFAULT_TEST_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--method", type=str, default="dopri5")
    parser.add_argument("--output-dir", type=str, default="results/residual_correction")
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

    # ---- Load acausal model ----
    print("\nLoading acausal model...")
    acausal_ckpt = torch.load(args.acausal_ckpt, map_location="cpu", weights_only=False)
    acausal_model = LatentNeuralODE(
        n_x=acausal_ckpt["n_x"],
        n_u=acausal_ckpt["n_u"],
        z_dim=acausal_ckpt["z_dim"],
        hidden_dim=acausal_ckpt["hidden"],
        n_layers=acausal_ckpt["n_layers"],
    )
    acausal_model.load_state_dict(acausal_ckpt["model_state"])
    acausal_model.eval()
    acausal_model = acausal_model.to(device)
    print(f"  Acausal: z_dim={acausal_ckpt['z_dim']}, hidden={acausal_ckpt['hidden']}, "
          f"layers={acausal_ckpt['n_layers']}")

    # ---- Load causal model ----
    print("Loading causal model...")
    causal_ckpt = torch.load(args.causal_ckpt, map_location="cpu", weights_only=False)
    causal_model = LatentControlAffineODE(
        n_x=causal_ckpt["n_x"],
        n_u=causal_ckpt["n_u"],
        z_dim=causal_ckpt["z_dim"],
        hidden_dim=causal_ckpt["hidden"],
        n_layers=causal_ckpt["n_layers"],
    )
    causal_model.load_state_dict(causal_ckpt["model_state"])
    causal_model.eval()
    causal_model = causal_model.to(device)
    print(f"  Causal: z_dim={causal_ckpt['z_dim']}, hidden={causal_ckpt['hidden']}, "
          f"layers={causal_ckpt['n_layers']}")

    x_mean = acausal_ckpt["x_mean"]
    x_std = acausal_ckpt["x_std"]
    u_mean = acausal_ckpt["u_mean"]
    u_std = acausal_ckpt["u_std"]

    # ---- Extract z0 pairs from training data ----
    print("\n" + "=" * 60)
    print("Step 1: Extracting z0 pairs from training trials...")
    print("=" * 60)
    t_extract_start = time.time()
    z0_causal, z0_acausal_target, z0_acausal_delayed, u_lag_flat, delta_z0 = extract_z0_pairs(
        acausal_model=acausal_model,
        causal_model=causal_model,
        x_trials=x_train_trials,
        u_trials=u_train_trials,
        x_mean=x_mean,
        x_std=x_std,
        u_mean=u_mean,
        u_std=u_std,
        dt=dt,
        causal_past_window=args.causal_past_window,
        acausal_window=args.acausal_window,
        lag_steps=args.lag_steps,
        stride=args.extract_stride,
        device=device,
    )
    t_extract = time.time() - t_extract_start
    print(f"  Extraction took {t_extract:.1f}s")

    # ---- Train residual MLP ----
    print("\n" + "=" * 60)
    print("Step 2: Training residual correction MLP...")
    print("=" * 60)

    z_dim = acausal_ckpt["z_dim"]
    n_u = acausal_ckpt["n_u"]

    residual_mlp = ResidualCorrectionMLP(
        z_dim=z_dim,
        n_u=n_u,
        lag_steps=args.lag_steps,
        hidden_dim=args.mlp_hidden,
        dropout=args.mlp_dropout,
    )
    n_params_mlp = sum(p.numel() for p in residual_mlp.parameters())
    print(f"  Residual MLP parameters: {n_params_mlp:,}")

    t_train_start = time.time()
    mlp_history = train_residual_mlp(
        mlp=residual_mlp,
        z0_causal=z0_causal,
        z0_acausal_delayed=z0_acausal_delayed,
        u_lag_flat=u_lag_flat,
        delta_z0=delta_z0,
        n_epochs=args.mlp_epochs,
        lr=args.mlp_lr,
        batch_size=args.mlp_batch_size,
        device=device,
        seed=args.seed,
    )
    t_train = time.time() - t_train_start
    print(f"  Training took {t_train:.1f}s")

    residual_mlp = residual_mlp.to(device)

    # ---- Evaluate ----
    print("\n" + "=" * 60)
    print("Step 3: Evaluating on test trials...")
    print("=" * 60)

    results = evaluate_corrected(
        acausal_model=acausal_model,
        causal_model=causal_model,
        residual_mlp=residual_mlp,
        x_test_trials=x_test_trials,
        u_test_trials=u_test_trials,
        x_mean=x_mean,
        x_std=x_std,
        u_mean=u_mean,
        u_std=u_std,
        dt=dt,
        causal_past_window=args.causal_past_window,
        acausal_window=args.acausal_window,
        future_window=args.future_window,
        lag_steps=args.lag_steps,
        stride=args.eval_stride,
        device=device,
        method=args.method,
        n_train=n_train,
    )

    # ---- Print summary ----
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    gap = abs(results["acausal_r2"] - results["causal_only_r2"])
    improvement = results["corrected_r2"] - results["causal_only_r2"]
    pct = (improvement / gap * 100) if gap > 0 else 0.0
    print(f"  Causal-only:    R2 = {results['causal_only_r2']:.4f} +/- {results['causal_only_r2_std']:.4f}  "
          f"MSE = {results['causal_only_mse']:.4f}  "
          f"Inference = {results['avg_inference_ms_causal']:.2f} ms")
    print(f"  Corrected:      R2 = {results['corrected_r2']:.4f} +/- {results['corrected_r2_std']:.4f}  "
          f"MSE = {results['corrected_mse']:.4f}  "
          f"Inference = {results['avg_inference_ms_corrected']:.2f} ms")
    print(f"  Acausal oracle: R2 = {results['acausal_r2']:.4f} +/- {results['acausal_r2_std']:.4f}  "
          f"MSE = {results['acausal_mse']:.4f}")
    print(f"\n  Improvement: {improvement:.4f} R2 ({pct:.1f}% of gap to acausal)")

    # ---- Save ----
    save_path = os.path.join(args.output_dir, "residual_correction.pt")
    torch.save({
        "residual_mlp_state": residual_mlp.state_dict(),
        "z_dim": z_dim,
        "n_u": n_u,
        "lag_steps": args.lag_steps,
        "mlp_hidden": args.mlp_hidden,
        "mlp_dropout": args.mlp_dropout,
        "mlp_history": mlp_history,
        "results": results,
        "args": vars(args),
    }, save_path)
    print(f"\n  Saved to {save_path}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].semilogy(mlp_history["train_loss"], label="Train")
    axes[0].semilogy(mlp_history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Residual MLP Training")
    axes[0].legend()

    trials = range(len(results["per_trial_causal_r2"]))
    axes[1].plot(list(trials), results["per_trial_causal_r2"], "o-", label="Causal-only", alpha=0.7)
    axes[1].plot(list(trials), results["per_trial_corrected_r2"], "s-", label="Corrected", alpha=0.7)
    axes[1].plot(list(trials), results["per_trial_acausal_r2"], "^-", label="Acausal oracle", alpha=0.7)
    axes[1].set_xlabel("Test Trial")
    axes[1].set_ylabel("R2")
    axes[1].set_title("Per-Trial R2 Comparison")
    axes[1].legend()

    methods = ["Causal\nonly", "Corrected", "Acausal\noracle"]
    r2s = [results["causal_only_r2"], results["corrected_r2"], results["acausal_r2"]]
    stds = [results["causal_only_r2_std"], results["corrected_r2_std"], results["acausal_r2_std"]]
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    bars = axes[2].bar(methods, r2s, yerr=stds, capsize=5, color=colors, alpha=0.8)
    axes[2].set_ylabel("R2")
    axes[2].set_title("Avg R2 Comparison")
    for bar, r2 in zip(bars, r2s):
        axes[2].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                     f"{r2:.4f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "residual_correction_results.png"), dpi=150)
    print(f"  Plot saved to {os.path.join(args.output_dir, 'residual_correction_results.png')}")

    print("\nDone.")


if __name__ == "__main__":
    main()
