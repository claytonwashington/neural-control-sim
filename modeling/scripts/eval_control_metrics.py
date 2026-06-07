#!/usr/bin/env python
"""Control-Relevant Metrics Evaluation Suite.

Evaluates trained Latent Neural ODE models on control-relevant metrics:
1. Free-Run Simulation FIT%
2. Multi-Horizon Error Growth Curves
3. Jacobian Accuracy (∂g/∂u) — causal models only
4. Empirical Controllability Gramian — causal models only

Usage:
    python -m modeling.scripts.eval_control_metrics \
        --checkpoint /path/to/model.pt \
        --model-type causal \
        --data /path/to/training_trials.h5 \
        --output-dir results/ctrl_metrics \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(4)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modeling.data import load_trials_h5
from modeling.config import DEFAULT_SEED, DEFAULT_TEST_TRIALS, set_seed


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, model_type: str, device: str):
    """Load a trained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    n_x = ckpt["n_x"]
    n_u = ckpt["n_u"]
    z_dim = ckpt["z_dim"]
    hidden = ckpt["hidden"]
    n_layers = ckpt["n_layers"]
    x_mean = ckpt["x_mean"]
    x_std = ckpt["x_std"]
    u_mean = ckpt["u_mean"]
    u_std = ckpt["u_std"]

    if model_type == "causal":
        from modeling.models.latent_canode import LatentControlAffineODE
        model = LatentControlAffineODE(
            n_x=n_x, n_u=n_u, z_dim=z_dim,
            hidden_dim=hidden, n_layers=n_layers,
        )
    elif model_type == "acausal":
        from modeling.models.latent_node import LatentNeuralODE
        model = LatentNeuralODE(
            n_x=n_x, n_u=n_u, z_dim=z_dim,
            hidden_dim=hidden, n_layers=n_layers,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model = model.to(device)

    norm = {
        "x_mean": x_mean, "x_std": x_std,
        "u_mean": u_mean, "u_std": u_std,
    }
    meta = {
        "n_x": n_x, "n_u": n_u, "z_dim": z_dim,
        "hidden": hidden, "n_layers": n_layers,
    }
    return model, norm, meta


def normalize(arr, mean, std):
    return (arr - mean) / std


# ---------------------------------------------------------------------------
# Metric 1: Free-Run Simulation FIT%
# ---------------------------------------------------------------------------

def compute_free_run_fit(model, model_type, test_trials_x, test_trials_u,
                         norm, dt, past_window, device, method="dopri5"):
    """Encode z0 from first window, run ODE for entire remaining trial."""
    x_mean, x_std = norm["x_mean"], norm["x_std"]
    u_mean, u_std = norm["u_mean"], norm["u_std"]

    n_test = len(test_trials_x)
    fit_scores = []
    per_trial_info = []

    print("\n" + "=" * 60)
    print("METRIC 1: Free-Run Simulation FIT%")
    print("=" * 60)

    for ti in range(n_test):
        x_trial = test_trials_x[ti]   # (n_ch, T_trial)
        u_trial = test_trials_u[ti]   # (n_u, T_trial)
        T_trial = x_trial.shape[1]

        x_n = normalize(x_trial, x_mean, x_std)
        u_n = normalize(u_trial, u_mean, u_std)

        # Encode from the first past_window steps
        x_past = torch.tensor(x_n[:, :past_window].T, dtype=torch.float32).unsqueeze(0).to(device)
        u_past = torch.tensor(u_n[:, :past_window].T, dtype=torch.float32).unsqueeze(0).to(device)

        # Future = everything from past_window onward
        T_future = T_trial - past_window
        x_true_future = x_n[:, past_window:]  # (n_ch, T_future)
        u_future = torch.tensor(u_n[:, past_window:].T, dtype=torch.float32).unsqueeze(0).to(device)
        t_future = torch.arange(T_future, dtype=torch.float32, device=device) * dt

        # Break into chunks to avoid OOM (max 2000 steps per chunk)
        chunk_size = 2000
        n_chunks = (T_future + chunk_size - 1) // chunk_size
        x_pred_chunks = []

        with torch.no_grad():
            if model_type == "causal":
                # Encode z0 from past
                mu, logvar = model.encoder(x_past, u_past)
                z0 = mu  # deterministic

                z_current = z0
                for ci in range(n_chunks):
                    start = ci * chunk_size
                    end = min(start + chunk_size, T_future)
                    t_chunk = t_future[start:end] - t_future[start]  # reset time origin
                    u_chunk = u_future[:, start:end, :]

                    model.set_input(t_chunk, u_chunk)
                    from torchdiffeq import odeint
                    z_seq = odeint(model.ode_func, z_current, t_chunk,
                                   method=method, rtol=1e-4, atol=1e-5)
                    z_seq = z_seq.permute(1, 0, 2)  # (B, T_chunk, z_dim)
                    x_pred_chunk = model.decoder(z_seq)
                    x_pred_chunks.append(x_pred_chunk[0].cpu().numpy().T)

                    # Carry forward the last z state
                    z_current = z_seq[:, -1, :]

            else:
                # Acausal: encoder uses full window but we only have past
                mu, logvar = model.encoder(x_past, u_past)
                z0 = mu

                z_current = z0
                for ci in range(n_chunks):
                    start = ci * chunk_size
                    end = min(start + chunk_size, T_future)
                    t_chunk = t_future[start:end] - t_future[start]
                    u_chunk = u_future[:, start:end, :]

                    model.ode_func.set_input(t_chunk, u_chunk)
                    from torchdiffeq import odeint
                    z_seq = odeint(model.ode_func, z_current, t_chunk,
                                   method=method, rtol=1e-4, atol=1e-5)
                    z_seq = z_seq.permute(1, 0, 2)
                    x_pred_chunk = model.decoder(z_seq)
                    x_pred_chunks.append(x_pred_chunk[0].cpu().numpy().T)
                    z_current = z_seq[:, -1, :]

        x_pred_future = np.concatenate(x_pred_chunks, axis=1)  # (n_ch, T_future)

        # FIT% = max(0, 100 * (1 - ||x_true - x_pred||_2 / ||x_true - mean(x_true)||_2))
        residual_norm = np.linalg.norm(x_true_future - x_pred_future)
        baseline_norm = np.linalg.norm(x_true_future - x_true_future.mean(axis=1, keepdims=True))

        if baseline_norm > 0:
            fit_pct = max(0.0, 100.0 * (1.0 - residual_norm / baseline_norm))
        else:
            fit_pct = 0.0

        fit_scores.append(fit_pct)
        per_trial_info.append({
            "trial": ti,
            "fit_pct": fit_pct,
            "T_future": T_future,
            "residual_norm": float(residual_norm),
            "baseline_norm": float(baseline_norm),
        })
        print(f"  Trial {ti}: FIT% = {fit_pct:.2f}  (T_future={T_future}, "              f"residual={residual_norm:.2f}, baseline={baseline_norm:.2f})")

    avg_fit = np.mean(fit_scores)
    print(f"\n  Average FIT%: {avg_fit:.2f} (std={np.std(fit_scores):.2f})")

    return {
        "avg_fit_pct": float(avg_fit),
        "std_fit_pct": float(np.std(fit_scores)),
        "per_trial": per_trial_info,
    }


# ---------------------------------------------------------------------------
# Metric 2: Multi-Horizon Error Growth Curves
# ---------------------------------------------------------------------------

def compute_horizon_curves(model, model_type, test_trials_x, test_trials_u,
                           norm, dt, past_window, device, method="dopri5",
                           stride=100, max_windows_per_trial=50):
    """Compute R²(H) and MSE(H) for multiple prediction horizons."""
    x_mean, x_std = norm["x_mean"], norm["x_std"]
    u_mean, u_std = norm["u_mean"], norm["u_std"]

    horizons = [1, 2, 5, 10, 20, 50, 100, 200]
    max_H = max(horizons)

    print("\n" + "=" * 60)
    print("METRIC 2: Multi-Horizon Error Growth Curves")
    print("=" * 60)

    # Collect predictions and ground truth at each horizon
    # For each horizon H: accumulate (pred_at_H, true_at_H) pairs
    horizon_preds = {H: [] for H in horizons}
    horizon_trues = {H: [] for H in horizons}
    n_windows_total = 0

    for ti in range(len(test_trials_x)):
        x_trial = test_trials_x[ti]
        u_trial = test_trials_u[ti]
        T_trial = x_trial.shape[1]

        x_n = normalize(x_trial, x_mean, x_std)
        u_n = normalize(u_trial, u_mean, u_std)

        # Minimum window: past_window + max_H
        if T_trial < past_window + max_H:
            print(f"  Trial {ti}: skipping (T={T_trial} < {past_window + max_H})")
            continue

        n_windows = min(
            max_windows_per_trial,
            (T_trial - past_window - max_H) // stride + 1
        )

        for w in range(n_windows):
            t0 = w * stride
            t_split = t0 + past_window
            t_end = t_split + max_H

            if t_end > T_trial:
                break

            x_past = torch.tensor(x_n[:, t0:t_split].T, dtype=torch.float32).unsqueeze(0).to(device)
            u_past = torch.tensor(u_n[:, t0:t_split].T, dtype=torch.float32).unsqueeze(0).to(device)
            x_true_fut = x_n[:, t_split:t_end]  # (n_ch, max_H)
            u_fut = torch.tensor(u_n[:, t_split:t_end].T, dtype=torch.float32).unsqueeze(0).to(device)
            t_fut = torch.arange(max_H, dtype=torch.float32, device=device) * dt

            with torch.no_grad():
                if model_type == "causal":
                    x_pred = model.predict(x_past, u_past, u_fut, t_fut, method=method)
                else:
                    # For acausal, we use the past window as the encoding context
                    mu, logvar = model.encoder(x_past, u_past)
                    z0 = mu
                    model.ode_func.set_input(t_fut, u_fut)
                    from torchdiffeq import odeint
                    z_seq = odeint(model.ode_func, z0, t_fut, method=method,
                                   rtol=1e-4, atol=1e-5)
                    z_seq = z_seq.permute(1, 0, 2)
                    x_pred = model.decoder(z_seq)

                x_pred_np = x_pred[0].cpu().numpy().T  # (n_ch, max_H)

            for H in horizons:
                # H is 1-indexed: step H means index H-1
                idx = H - 1
                horizon_preds[H].append(x_pred_np[:, idx])
                horizon_trues[H].append(x_true_fut[:, idx])

            n_windows_total += 1

    print(f"  Total windows evaluated: {n_windows_total}")

    # Compute R² and MSE at each horizon
    results_r2 = {}
    results_mse = {}
    for H in horizons:
        preds = np.array(horizon_preds[H])  # (N_windows, n_ch)
        trues = np.array(horizon_trues[H])
        mse = np.mean((preds - trues) ** 2)
        var = np.var(trues)
        r2 = 1.0 - mse / var if var > 0 else 0.0
        results_r2[H] = float(r2)
        results_mse[H] = float(mse)
        print(f"  H={H:>3d}:  R² = {r2:.4f},  MSE = {mse:.6f}")

    return {
        "horizons": horizons,
        "r2_by_horizon": results_r2,
        "mse_by_horizon": results_mse,
        "n_windows": n_windows_total,
    }


# ---------------------------------------------------------------------------
# Metric 3: Jacobian Accuracy (∂g/∂u)
# ---------------------------------------------------------------------------

def compute_jacobian_accuracy(model, model_type, test_trials_x, test_trials_u,
                              norm, dt, past_window, device, method="dopri5",
                              stride=200, eps=1e-3, max_samples=200):
    """Compute cosine similarity between learned g(z) and finite-difference Jacobian.

    Only applies to causal (control-affine) models.
    """
    if model_type != "causal":
        print("\n" + "=" * 60)
        print("METRIC 3: Jacobian Accuracy — SKIPPED (acausal model)")
        print("=" * 60)
        return {"skipped": True, "reason": "Not a control-affine model"}

    x_mean, x_std = norm["x_mean"], norm["x_std"]
    u_mean, u_std = norm["u_mean"], norm["u_std"]

    print("\n" + "=" * 60)
    print("METRIC 3: Jacobian Accuracy (∂g/∂u)")
    print("=" * 60)

    cosine_sims = []
    n_samples = 0

    for ti in range(len(test_trials_x)):
        if n_samples >= max_samples:
            break

        x_trial = test_trials_x[ti]
        u_trial = test_trials_u[ti]
        T_trial = x_trial.shape[1]

        x_n = normalize(x_trial, x_mean, x_std)
        u_n = normalize(u_trial, u_mean, u_std)

        # The causal model needs past_window + at least 2 future steps
        future_window = 2  # just need 1-step forward integration
        if T_trial < past_window + future_window:
            continue

        n_windows = (T_trial - past_window - future_window) // stride + 1

        for w in range(n_windows):
            if n_samples >= max_samples:
                break

            t0 = w * stride
            t_split = t0 + past_window
            t_end = t_split + future_window

            if t_end > T_trial:
                break

            x_past = torch.tensor(x_n[:, t0:t_split].T, dtype=torch.float32).unsqueeze(0).to(device)
            u_past = torch.tensor(u_n[:, t0:t_split].T, dtype=torch.float32).unsqueeze(0).to(device)
            u_fut_np = u_n[:, t_split:t_end]
            t_fut = torch.arange(future_window, dtype=torch.float32, device=device) * dt

            # 1. Encode z0
            with torch.no_grad():
                mu, _ = model.encoder(x_past, u_past)
                z0 = mu  # (1, z_dim)

            # 2. Get learned g(z0): (1, z_dim, n_u)
            with torch.no_grad():
                g_learned = model.g(z0)  # (1, z_dim, n_u)
                g_learned_np = g_learned[0].cpu().numpy()  # (z_dim, n_u)

            # 3. Finite-difference: perturb each u dimension
            n_u = u_fut_np.shape[0]
            g_fd = np.zeros_like(g_learned_np)  # (z_dim, n_u)

            for ui in range(n_u):
                u_plus = u_fut_np.copy()
                u_minus = u_fut_np.copy()
                u_plus[ui, :] += eps
                u_minus[ui, :] -= eps

                u_plus_t = torch.tensor(u_plus.T, dtype=torch.float32).unsqueeze(0).to(device)
                u_minus_t = torch.tensor(u_minus.T, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    # Forward 1 ODE step with u_plus
                    model.set_input(t_fut, u_plus_t)
                    from torchdiffeq import odeint
                    z_plus = odeint(model.ode_func, z0, t_fut,
                                    method=method, rtol=1e-5, atol=1e-6)
                    z_plus_final = z_plus[-1]  # (1, z_dim) at final time

                    # Forward 1 ODE step with u_minus
                    model.set_input(t_fut, u_minus_t)
                    z_minus = odeint(model.ode_func, z0, t_fut,
                                     method=method, rtol=1e-5, atol=1e-6)
                    z_minus_final = z_minus[-1]  # (1, z_dim)

                    dz_du = (z_plus_final - z_minus_final) / (2 * eps)
                    g_fd[:, ui] = dz_du[0].cpu().numpy()

            # 4. Compute cosine similarity between g_learned and g_fd
            # Flatten both matrices and compare
            g_l_flat = g_learned_np.flatten()
            g_fd_flat = g_fd.flatten()

            norm_l = np.linalg.norm(g_l_flat)
            norm_fd = np.linalg.norm(g_fd_flat)

            if norm_l > 1e-10 and norm_fd > 1e-10:
                cos_sim = float(np.dot(g_l_flat, g_fd_flat) / (norm_l * norm_fd))
                cosine_sims.append(cos_sim)

            n_samples += 1

    if cosine_sims:
        avg_cos = float(np.mean(cosine_sims))
        std_cos = float(np.std(cosine_sims))
        med_cos = float(np.median(cosine_sims))
    else:
        avg_cos = std_cos = med_cos = 0.0

    print(f"  Samples evaluated: {n_samples}")
    print(f"  Avg Cosine Similarity: {avg_cos:.4f} (std={std_cos:.4f})")
    print(f"  Median Cosine Similarity: {med_cos:.4f}")

    return {
        "avg_cosine_similarity": avg_cos,
        "std_cosine_similarity": std_cos,
        "median_cosine_similarity": med_cos,
        "n_samples": n_samples,
    }


# ---------------------------------------------------------------------------
# Metric 4: Empirical Controllability Gramian
# ---------------------------------------------------------------------------

def compute_controllability_gramian(model, model_type, test_trials_x, test_trials_u,
                                    norm, dt, past_window, device, method="dopri5",
                                    stride=200, max_samples=200):
    """Compute the empirical controllability Gramian from B(t) = g(z(t)).

    Only applies to causal (control-affine) models.
    W_c = (1/T) Σ_t B(t) B(t)^T
    """
    if model_type != "causal":
        print("\n" + "=" * 60)
        print("METRIC 4: Controllability Gramian — SKIPPED (acausal model)")
        print("=" * 60)
        return {"skipped": True, "reason": "Not a control-affine model"}

    x_mean, x_std = norm["x_mean"], norm["x_std"]
    u_mean, u_std = norm["u_mean"], norm["u_std"]

    print("\n" + "=" * 60)
    print("METRIC 4: Empirical Controllability Gramian")
    print("=" * 60)

    z_dim = None
    W_c_accum = None
    A_accumulator = None
    n_samples = 0

    for ti in range(len(test_trials_x)):
        if n_samples >= max_samples:
            break

        x_trial = test_trials_x[ti]
        u_trial = test_trials_u[ti]
        T_trial = x_trial.shape[1]

        x_n = normalize(x_trial, x_mean, x_std)
        u_n = normalize(u_trial, u_mean, u_std)

        future_window = 2
        if T_trial < past_window + future_window:
            continue

        n_windows = (T_trial - past_window - future_window) // stride + 1

        for w in range(n_windows):
            if n_samples >= max_samples:
                break

            t0 = w * stride
            t_split = t0 + past_window

            if t_split + future_window > T_trial:
                break

            x_past = torch.tensor(x_n[:, t0:t_split].T, dtype=torch.float32).unsqueeze(0).to(device)
            u_past = torch.tensor(u_n[:, t0:t_split].T, dtype=torch.float32).unsqueeze(0).to(device)

            # 1. Encode z0
            with torch.no_grad():
                mu, _ = model.encoder(x_past, u_past)
                z0 = mu[0]  # (z_dim,)

            if z_dim is None:
                z_dim = z0.shape[0]
                W_c_accum = np.zeros((z_dim, z_dim))
                A_accumulator = np.zeros((z_dim, z_dim))

            # 2. B(t) = g(z) — control matrix
            with torch.no_grad():
                B_t = model.g(z0.unsqueeze(0))  # (1, z_dim, n_u)
                B_np = B_t[0].cpu().numpy()  # (z_dim, n_u)

            # 3. A(t) = Jacobian of f(z) w.r.t z
            z0_for_jac = z0.detach().requires_grad_(False)
            try:
                A_t = torch.autograd.functional.jacobian(
                    model.f, z0_for_jac.unsqueeze(0), create_graph=False
                )
                # A_t shape: (1, z_dim, 1, z_dim) -> (z_dim, z_dim)
                A_np = A_t.squeeze().cpu().numpy()
                if A_np.shape == (z_dim, z_dim):
                    A_accumulator += A_np
            except Exception as e:
                # If Jacobian fails, just accumulate B contribution
                pass

            # 4. Accumulate W_c = (1/T) Σ B(t) B(t)^T
            W_c_accum += B_np @ B_np.T

            n_samples += 1

    if n_samples == 0 or W_c_accum is None:
        print("  No samples collected!")
        return {"error": "No samples"}

    W_c = W_c_accum / n_samples
    A_avg = A_accumulator / n_samples

    # Eigendecomposition
    eigvals = np.linalg.eigvalsh(W_c)
    eigvals = np.sort(eigvals)[::-1]  # descending

    max_eig = eigvals[0]
    min_eig = eigvals[-1]
    condition_number = max_eig / min_eig if min_eig > 1e-15 else float("inf")
    effective_rank = int(np.sum(eigvals > 0.01 * max_eig))

    print(f"  Samples evaluated: {n_samples}")
    print(f"  z_dim: {z_dim}")
    print(f"  Top 10 eigenvalues: {eigvals[:10]}")
    print(f"  Condition number: {condition_number:.2e}")
    print(f"  Effective rank (>1% of max): {effective_rank} / {z_dim}")
    print(f"  Max eigenvalue: {max_eig:.6f}")
    print(f"  Min eigenvalue: {min_eig:.6e}")

    # Also compute A's spectral properties
    A_eigvals = np.linalg.eigvals(A_avg)
    max_real_A = float(np.max(np.real(A_eigvals)))
    print(f"\n  Drift Jacobian (A) spectral radius: {np.max(np.abs(A_eigvals)):.4f}")
    print(f"  Max real part of A eigenvalues: {max_real_A:.4f}")
    print(f"  (negative = stable, positive = unstable)")

    return {
        "n_samples": n_samples,
        "z_dim": z_dim,
        "eigenvalues": eigvals.tolist(),
        "condition_number": float(condition_number),
        "effective_rank": effective_rank,
        "max_eigenvalue": float(max_eig),
        "min_eigenvalue": float(min_eig),
        "A_spectral_radius": float(np.max(np.abs(A_eigvals))),
        "A_max_real_eigenvalue": max_real_A,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_horizon_curves(results, output_dir):
    """Plot R²(H) and MSE(H) curves."""
    horizons = results["horizons"]
    r2_vals = [results["r2_by_horizon"][str(H)] if isinstance(list(results["r2_by_horizon"].keys())[0], str) else results["r2_by_horizon"][H] for H in horizons]
    mse_vals = [results["mse_by_horizon"][str(H)] if isinstance(list(results["mse_by_horizon"].keys())[0], str) else results["mse_by_horizon"][H] for H in horizons]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(horizons, r2_vals, "o-", color="tab:blue", linewidth=2, markersize=8)
    ax1.set_xlabel("Prediction Horizon H (steps)", fontsize=12)
    ax1.set_ylabel("R²", fontsize=12)
    ax1.set_title("R² vs Prediction Horizon", fontsize=14)
    ax1.set_xscale("log")
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.grid(True, alpha=0.3)

    ax2.plot(horizons, mse_vals, "o-", color="tab:red", linewidth=2, markersize=8)
    ax2.set_xlabel("Prediction Horizon H (steps)", fontsize=12)
    ax2.set_ylabel("MSE", fontsize=12)
    ax2.set_title("MSE vs Prediction Horizon", fontsize=14)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "horizon_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved horizon curves to {path}")


def plot_gramian_eigenvalues(results, output_dir):
    """Plot eigenvalue spectrum of the controllability Gramian."""
    if results.get("skipped"):
        return

    eigvals = np.array(results["eigenvalues"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.semilogy(range(1, len(eigvals) + 1), np.maximum(eigvals, 1e-20), "o-", markersize=3)
    ax1.set_xlabel("Index", fontsize=12)
    ax1.set_ylabel("Eigenvalue", fontsize=12)
    ax1.set_title("Controllability Gramian Eigenvalues", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Cumulative energy
    total = np.sum(eigvals[eigvals > 0])
    cumsum = np.cumsum(eigvals[eigvals > 0]) / total if total > 0 else np.zeros(1)
    ax2.plot(range(1, len(cumsum) + 1), cumsum, "o-", markersize=3)
    ax2.set_xlabel("Number of Eigenvalues", fontsize=12)
    ax2.set_ylabel("Cumulative Energy", fontsize=12)
    ax2.set_title("Gramian Cumulative Energy", fontsize=14)
    ax2.axhline(y=0.99, color="red", linestyle="--", alpha=0.5, label="99%")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "gramian_eigenvalues.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved Gramian eigenvalue plot to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate control-relevant metrics for Latent ODE models"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--model-type", type=str, required=True,
                        choices=["causal", "acausal"],
                        help="Model architecture type")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to HDF5 trial data")
    parser.add_argument("--output-dir", type=str, default="results/ctrl_metrics",
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--method", type=str, default="dopri5",
                        help="ODE solver method")
    parser.add_argument("--past-window", type=int, default=200,
                        help="Past context window size for encoding z0")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-test-trials", type=int, default=DEFAULT_TEST_TRIALS)
    parser.add_argument("--stride", type=int, default=100,
                        help="Stride between windows for horizon/Jacobian metrics")
    parser.add_argument("--max-jacobian-samples", type=int, default=200,
                        help="Max samples for Jacobian accuracy")
    parser.add_argument("--max-gramian-samples", type=int, default=200,
                        help="Max samples for Gramian computation")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # Load data
    print("\nLoading data...")
    data = load_trials_h5(args.data)
    x_all, u_all = data["x"], data["u"]
    dt = float(data["dt"])
    n_trials, n_channels, n_steps = x_all.shape
    n_inputs = u_all.shape[1]
    print(f"  Trials: {n_trials}, Channels: {n_channels}, Steps/trial: {n_steps}, Inputs: {n_inputs}, dt: {dt}")

    # Train/test split (same as training)
    n_test = args.n_test_trials
    n_train = n_trials - n_test
    test_trials_x = x_all[n_train:]  # (n_test, n_ch, T)
    test_trials_u = u_all[n_train:]

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model, norm, meta = load_model(args.checkpoint, args.model_type, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model type: {args.model_type}")
    print(f"  z_dim={meta['z_dim']}, hidden={meta['hidden']}, n_layers={meta['n_layers']}")
    print(f"  Parameters: {n_params:,}")

    # Adjust past_window for acausal model
    past_window = args.past_window

    t_start = time.time()

    # --- Metric 1: Free-Run Simulation FIT% ---
    fit_results = compute_free_run_fit(
        model, args.model_type, test_trials_x, test_trials_u,
        norm, dt, past_window, device, method=args.method,
    )

    # --- Metric 2: Multi-Horizon Error Growth Curves ---
    horizon_results = compute_horizon_curves(
        model, args.model_type, test_trials_x, test_trials_u,
        norm, dt, past_window, device, method=args.method,
        stride=args.stride,
    )

    # --- Metric 3: Jacobian Accuracy ---
    jacobian_results = compute_jacobian_accuracy(
        model, args.model_type, test_trials_x, test_trials_u,
        norm, dt, past_window, device, method=args.method,
        stride=args.stride, max_samples=args.max_jacobian_samples,
    )

    # --- Metric 4: Controllability Gramian ---
    gramian_results = compute_controllability_gramian(
        model, args.model_type, test_trials_x, test_trials_u,
        norm, dt, past_window, device, method=args.method,
        stride=args.stride, max_samples=args.max_gramian_samples,
    )

    total_time = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Total evaluation time: {total_time:.1f}s")
    print(f"{'=' * 60}")

    # --- Save results ---
    all_results = {
        "checkpoint": args.checkpoint,
        "model_type": args.model_type,
        "meta": meta,
        "n_params": n_params,
        "past_window": past_window,
        "method": args.method,
        "total_time_s": total_time,
        "free_run_fit": fit_results,
        "horizon_curves": horizon_results,
        "jacobian_accuracy": jacobian_results,
        "controllability_gramian": gramian_results,
    }

    results_path = os.path.join(args.output_dir, "control_metrics.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # --- Plots ---
    plot_horizon_curves(horizon_results, args.output_dir)
    if not gramian_results.get("skipped"):
        plot_gramian_eigenvalues(gramian_results, args.output_dir)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Model: {args.model_type} (z={meta['z_dim']}, h={meta['hidden']})")
    print(f"  Free-Run FIT%:  {fit_results['avg_fit_pct']:.2f} ± {fit_results['std_fit_pct']:.2f}")
    print(f"  Horizon R² curve:")
    for H in horizon_results["horizons"]:
        r2 = horizon_results["r2_by_horizon"][H]
        mse = horizon_results["mse_by_horizon"][H]
        print(f"    H={H:>3d}: R²={r2:.4f}, MSE={mse:.6f}")
    if not jacobian_results.get("skipped"):
        print(f"  Jacobian Cosine Similarity: {jacobian_results['avg_cosine_similarity']:.4f} ± {jacobian_results['std_cosine_similarity']:.4f}")
    if not gramian_results.get("skipped"):
        print(f"  Gramian Condition Number: {gramian_results['condition_number']:.2e}")
        print(f"  Gramian Effective Rank: {gramian_results['effective_rank']} / {gramian_results['z_dim']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
