#!/usr/bin/env python
"""Experiment 17: Full-Trial EnKF Deployment Stability Test.

Runs the Ensemble Kalman Filter (EnKF) over entire 30-second trials (not just
200-ms windows) to evaluate long-term state-estimation stability.

Uses the causal model's encoder (for z₀ from the first 200 steps), ODE
(f+g control-affine dynamics), and decoder -- all from the same checkpoint.

Metrics:
  - Full-trial FIT% = max(0, 100 * (1 - ||x_true - x_hat||₂ / ||x_true - mean(x_true)||₂))
  - Time-varying R²: sliding 200-step windows at specific timepoints
  - Per-step MSE trajectory

Usage:
    python -m modeling.scripts.eval_enkf_fulltrial \
        --obs-every-k 1 --obs-delay 0 \
        --output-dir results/fulltrial_k1_d0 --device cuda
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
import torch.nn as nn
torch.set_num_threads(4)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modeling.data import load_trials_h5
from modeling.config import DEFAULT_SEED, DEFAULT_TEST_TRIALS, set_seed
from modeling.models.latent_canode import LatentControlAffineODE

# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #
ABS_DATA_PATH = "/snel/home/cbwash2/cleo/data/training_trials.h5"
ABS_CAUSAL_CKPT = "/snel/home/cbwash2/cleo/results/causal_z64_pw200_h128_lr3e4/model.pt"
ENCODER_WINDOW = 200  # first 200 steps used for encoder init


# --------------------------------------------------------------------------- #
#  Model loader
# --------------------------------------------------------------------------- #

def load_causal_model(checkpoint_path: str, device: str = "cpu"):
    """Load the causal LatentControlAffineODE and return (model, ckpt_dict)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = LatentControlAffineODE(
        n_x=ckpt["n_x"],
        n_u=ckpt["n_u"],
        z_dim=ckpt["z_dim"],
        hidden_dim=ckpt["hidden"],
        n_layers=ckpt["n_layers"],
    )
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    return model, ckpt


# --------------------------------------------------------------------------- #
#  CausalODEWrapper: f(z) + g(z)@u as a single [z,u] -> dz call
# --------------------------------------------------------------------------- #

class CausalODEWrapper(nn.Module):
    """Wraps DriftNet f(z) + ControlNet g(z)@u into forward([z,u]) -> dz/dt."""

    def __init__(self, drift_net, control_net, z_dim: int, n_u: int):
        super().__init__()
        self.drift_net = drift_net
        self.control_net = control_net
        self.z_dim = z_dim
        self.n_u = n_u

    def forward(self, zu: torch.Tensor) -> torch.Tensor:
        z = zu[:, :self.z_dim]
        u = zu[:, self.z_dim:]
        drift = self.drift_net(z)
        g_z = self.control_net(z)
        control = torch.bmm(g_z, u.unsqueeze(-1)).squeeze(-1)
        return drift + control


# --------------------------------------------------------------------------- #
#  Full-Trial EnKF
# --------------------------------------------------------------------------- #

class FullTrialEnKF:
    """EnKF running step-by-step for an entire trial.

    - Predict: z_i += dt * ODE(z_i, u_t) + noise*Q
    - Update (every K steps, with delay D): Kalman correction using x(t-D)
    - Skip update when t < ENCODER_WINDOW or t - D < 0
    - Accumulate predictions on CPU
    """

    def __init__(self, ode_func_net, decoder, z_dim, n_x, n_u,
                 n_ensemble=64, Q=0.1, R=0.01,
                 obs_every_k=1, obs_delay=0, device="cpu"):
        self.ode_net = ode_func_net
        self.decoder = decoder
        self.z_dim = z_dim
        self.n_x = n_x
        self.n_u = n_u
        self.N = n_ensemble
        self.Q = Q
        self.R = R
        self.obs_every_k = obs_every_k
        self.obs_delay = obs_delay
        self.device = device

    @torch.no_grad()
    def run_trial(self, z0, u_full, x_obs_full, dt):
        """Run EnKF over a full trial.

        Parameters
        ----------
        z0         : (z_dim,) initial latent from causal encoder
        u_full     : (T, n_u) control inputs for the ENTIRE trial (normalized)
        x_obs_full : (T, n_x) ground-truth observations (normalized)
        dt         : float

        Returns
        -------
        x_preds_cpu : (T, n_x) decoded predictions on CPU
        mse_per_step : (T,) MSE at each step on CPU
        """
        T = u_full.shape[0]
        N = self.N
        device = self.device
        K_obs = self.obs_every_k
        D_obs = self.obs_delay

        # Initialize ensemble: (N, z_dim)
        z_ens = z0.unsqueeze(0).expand(N, -1).clone()
        z_ens = z_ens + torch.randn_like(z_ens) * self.Q

        # Preallocate output on CPU
        x_preds_cpu = torch.zeros(T, self.n_x)
        mse_per_step = torch.zeros(T)

        # Observation noise covariance (only diagonal needed, but we solve Pxx)
        R2_eye = (self.R ** 2) * torch.eye(self.n_x, device=device)

        # Move data to GPU
        u_gpu = u_full.to(device)
        x_obs_gpu = x_obs_full.to(device)

        # Progress tracking
        report_interval = T // 10

        for t in range(T):
            # ---- PREDICT ----
            if t > 0:
                z_flat = z_ens  # (N, z_dim)
                u_t = u_gpu[t - 1].unsqueeze(0).expand(N, -1)  # (N, n_u)
                zu = torch.cat([z_flat, u_t], dim=-1)  # (N, z_dim+n_u)
                dzdt = self.ode_net(zu)  # (N, z_dim)
                z_ens = z_flat + dt * dzdt
                z_ens = z_ens + torch.randn_like(z_ens) * self.Q

            # ---- UPDATE ----
            obs_t_idx = t - D_obs
            do_update = (
                t >= ENCODER_WINDOW
                and t % K_obs == 0
                and obs_t_idx >= 0
                and obs_t_idx < T
            )

            if do_update:
                # Predicted observations from ensemble
                x_pred_ens = self.decoder(z_ens)  # (N, n_x)

                # Ensemble means
                z_mean = z_ens.mean(dim=0)             # (z_dim,)
                x_pred_mean = x_pred_ens.mean(dim=0)   # (n_x,)

                # Anomalies
                dz = z_ens - z_mean.unsqueeze(0)              # (N, z_dim)
                dx = x_pred_ens - x_pred_mean.unsqueeze(0)    # (N, n_x)

                # Cross-covariance Pzx: (z_dim, n_x)
                Pzx = (dz.T @ dx) / (N - 1)

                # Observation covariance Pxx: (n_x, n_x)
                Pxx = (dx.T @ dx) / (N - 1) + R2_eye

                # Kalman gain: K = Pzx @ inv(Pxx)  -> (z_dim, n_x)
                K_gain = torch.linalg.solve(Pxx.T, Pzx.T).T

                # Stochastic EnKF update
                x_obs_t = x_obs_gpu[obs_t_idx]  # (n_x,)
                obs_perturbed = x_obs_t.unsqueeze(0) +                     torch.randn(N, self.n_x, device=device) * self.R
                innovations = obs_perturbed - x_pred_ens  # (N, n_x)

                # Update ensemble: (N, z_dim)
                z_ens = z_ens + innovations @ K_gain.T

            # Record prediction from posterior mean
            z_mean = z_ens.mean(dim=0)
            x_hat = self.decoder(z_mean.unsqueeze(0)).squeeze(0)  # (n_x,)

            # Store on CPU
            x_hat_cpu = x_hat.cpu()
            x_preds_cpu[t] = x_hat_cpu
            mse_per_step[t] = ((x_hat_cpu - x_obs_full[t]) ** 2).mean()

            if report_interval > 0 and t > 0 and t % report_interval == 0:
                pct = 100 * t / T
                recent_mse = mse_per_step[max(0, t-200):t].mean().item()
                print(f"    step {t}/{T} ({pct:.0f}%) — recent MSE={recent_mse:.6f}")

        return x_preds_cpu, mse_per_step


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #

def compute_fit_pct(x_true, x_pred):
    """Full-trial FIT%.

    FIT% = max(0, 100 * (1 - ||x_true - x_pred||_F / ||x_true - mean(x_true)||_F))
    x_true, x_pred: (T, n_x) numpy arrays
    """
    residual_norm = np.linalg.norm(x_true - x_pred)
    baseline_norm = np.linalg.norm(x_true - x_true.mean(axis=0, keepdims=True))
    if baseline_norm > 0:
        return max(0.0, 100.0 * (1.0 - residual_norm / baseline_norm))
    return 0.0


def compute_windowed_r2(x_true, x_pred, window=200, positions=None):
    """Compute R² in sliding windows at specified positions (timestep indices).

    Returns dict: {position: r2_value}
    """
    T = x_true.shape[0]
    if positions is None:
        positions = [0, 1000, 5000, 10000, 20000, 29000]

    results = {}
    for pos in positions:
        if pos + window > T:
            continue
        xt = x_true[pos:pos+window]
        xp = x_pred[pos:pos+window]
        ss_res = np.sum((xt - xp) ** 2)
        ss_tot = np.sum((xt - xt.mean(axis=0, keepdims=True)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        results[pos] = float(r2)
    return results


def compute_r2_trajectory(x_true, x_pred, window=200, stride=200):
    """Compute R² in non-overlapping windows across the trial.

    Returns list of (center_step, r2) tuples.
    """
    T = x_true.shape[0]
    trajectory = []
    for start in range(0, T - window + 1, stride):
        xt = x_true[start:start+window]
        xp = x_pred[start:start+window]
        ss_res = np.sum((xt - xp) ** 2)
        ss_tot = np.sum((xt - xt.mean(axis=0, keepdims=True)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        trajectory.append((start + window // 2, float(r2)))
    return trajectory


# --------------------------------------------------------------------------- #
#  Main evaluation
# --------------------------------------------------------------------------- #

def evaluate_fulltrial(
    model, ckpt,
    data_path, obs_every_k, obs_delay,
    n_ensemble=64, Q=0.1, R=0.01,
    device="cpu", seed=42,
):
    """Run EnKF over all 10 test trials and collect metrics."""
    data = load_trials_h5(data_path)
    x_all, u_all = data["x"], data["u"]
    dt = float(data["dt"])
    n_trials, n_channels, n_steps = x_all.shape

    n_test = DEFAULT_TEST_TRIALS
    n_train = n_trials - n_test

    # Normalization from the causal checkpoint
    x_mean = ckpt["x_mean"]
    x_std  = ckpt["x_std"]
    u_mean = ckpt["u_mean"]
    u_std  = ckpt["u_std"]

    z_dim = ckpt["z_dim"]
    n_x = ckpt["n_x"]
    n_u = ckpt["n_u"]

    # Build ODE wrapper and EnKF
    ode_net = CausalODEWrapper(
        drift_net=model.f, control_net=model.g,
        z_dim=z_dim, n_u=n_u,
    )
    enkf = FullTrialEnKF(
        ode_func_net=ode_net, decoder=model.decoder,
        z_dim=z_dim, n_x=n_x, n_u=n_u,
        n_ensemble=n_ensemble, Q=Q, R=R,
        obs_every_k=obs_every_k, obs_delay=obs_delay,
        device=device,
    )

    # R² timepoints (in steps): 0-200ms, 1s, 5s, 10s, 20s, 29s
    r2_positions = [0, 1000, 5000, 10000, 20000, 29000]

    all_fit_pcts = []
    all_r2_at_positions = {p: [] for p in r2_positions}
    all_r2_trajectories = []
    all_mse_trajectories = []
    all_trial_times = []

    for ti in range(n_test):
        trial_idx = n_train + ti
        print(f"\n  Trial {trial_idx} ({ti+1}/{n_test}):")

        x_raw = x_all[trial_idx]  # (n_channels, n_steps)
        u_raw = u_all[trial_idx]  # (n_u, n_steps)
        T_trial = x_raw.shape[1]

        # Normalize
        x_norm = (x_raw - x_mean) / (x_std + 1e-8)
        u_norm = (u_raw - u_mean) / (u_std + 1e-8)

        # Prepare tensors: (T, n_x) and (T, n_u)
        x_full = torch.tensor(x_norm.T, dtype=torch.float32)  # (T, n_x)
        u_full = torch.tensor(u_norm.T, dtype=torch.float32)  # (T, n_u)

        # Encode z0 from first ENCODER_WINDOW steps
        x_past = x_full[:ENCODER_WINDOW].unsqueeze(0).to(device)  # (1, 200, n_x)
        u_past = u_full[:ENCODER_WINDOW].unsqueeze(0).to(device)  # (1, 200, n_u)
        with torch.no_grad():
            mu, logvar = model.encoder(x_past, u_past)
            z0 = mu.squeeze(0)  # (z_dim,) deterministic

        # Run EnKF for entire trial
        t_start = time.perf_counter()
        torch.manual_seed(seed + ti)
        x_preds, mse_per_step = enkf.run_trial(z0, u_full, x_full, dt)
        t_elapsed = time.perf_counter() - t_start
        all_trial_times.append(t_elapsed)

        # Convert to numpy for metrics
        x_true_np = x_full.numpy()      # (T, n_x)
        x_pred_np = x_preds.numpy()      # (T, n_x)
        mse_np = mse_per_step.numpy()    # (T,)

        # FIT%
        fit = compute_fit_pct(x_true_np, x_pred_np)
        all_fit_pcts.append(fit)
        print(f"    FIT% = {fit:.2f}%  ({t_elapsed:.1f}s)")

        # Windowed R² at key timepoints
        r2_windows = compute_windowed_r2(x_true_np, x_pred_np, window=200, positions=r2_positions)
        for pos, r2_val in r2_windows.items():
            all_r2_at_positions[pos].append(r2_val)
        r2_str = ", ".join(f"t={p}:{r2_windows.get(p, float('nan')):.4f}" for p in r2_positions if p in r2_windows)
        print(f"    R² windows: {r2_str}")

        # R² trajectory
        r2_traj = compute_r2_trajectory(x_true_np, x_pred_np, window=200, stride=200)
        all_r2_trajectories.append(r2_traj)

        # Downsample MSE trajectory for storage (every 100 steps)
        mse_downsampled = mse_np[::100].tolist()
        all_mse_trajectories.append(mse_downsampled)

        # Full-trial R² and MSE
        ss_res = np.sum((x_true_np - x_pred_np) ** 2)
        ss_tot = np.sum((x_true_np - x_true_np.mean(axis=0, keepdims=True)) ** 2)
        full_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        full_mse = float(mse_np.mean())
        print(f"    Full-trial R²={full_r2:.4f}, MSE={full_mse:.6f}")

    # Aggregate results
    avg_fit = float(np.mean(all_fit_pcts))
    std_fit = float(np.std(all_fit_pcts))
    avg_time = float(np.mean(all_trial_times))

    avg_r2_at_pos = {}
    for pos in r2_positions:
        vals = all_r2_at_positions[pos]
        if vals:
            avg_r2_at_pos[str(pos)] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "values": vals,
            }

    # Average R² trajectory across trials
    # All trials have the same length, so trajectories align
    if all_r2_trajectories:
        min_len = min(len(t) for t in all_r2_trajectories)
        avg_r2_traj = []
        for i in range(min_len):
            step = all_r2_trajectories[0][i][0]
            r2_vals = [all_r2_trajectories[ti][i][1] for ti in range(n_test)]
            avg_r2_traj.append({
                "step": step,
                "r2_mean": float(np.mean(r2_vals)),
                "r2_std": float(np.std(r2_vals)),
            })
    else:
        avg_r2_traj = []

    return {
        "obs_every_k": obs_every_k,
        "obs_delay": obs_delay,
        "n_ensemble": n_ensemble,
        "Q": Q, "R": R,
        "avg_fit_pct": avg_fit,
        "std_fit_pct": std_fit,
        "per_trial_fit_pct": all_fit_pcts,
        "avg_inference_time_s": avg_time,
        "r2_at_positions": avg_r2_at_pos,
        "avg_r2_trajectory": avg_r2_traj,
        "mse_trajectories": all_mse_trajectories,
        "per_trial_times_s": all_trial_times,
    }


# --------------------------------------------------------------------------- #
#  Plotting
# --------------------------------------------------------------------------- #

def plot_r2_trajectory(results, output_dir):
    """Plot average R² over time within trials."""
    traj = results["avg_r2_trajectory"]
    if not traj:
        return

    steps = [t["step"] for t in traj]
    r2_means = [t["r2_mean"] for t in traj]
    r2_stds = [t["r2_std"] for t in traj]
    times_s = [s / 1000.0 for s in steps]  # convert to seconds

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(times_s, r2_means, "b-", linewidth=1.5, label="Mean R²")
    ax.fill_between(times_s,
                     [m - s for m, s in zip(r2_means, r2_stds)],
                     [m + s for m, s in zip(r2_means, r2_stds)],
                     alpha=0.2, color="blue")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=0.8252, color="red", linestyle="--", alpha=0.5, label="Causal baseline (R²=0.8252)")
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Windowed R² (200-step windows)", fontsize=12)
    K = results["obs_every_k"]
    D = results["obs_delay"]
    ax.set_title(f"EnKF R² Stability — K={K}, delay={D}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, "r2_trajectory.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved R² trajectory plot to {path}")


def plot_mse_trajectory(results, output_dir):
    """Plot average MSE over time."""
    mse_trajs = results["mse_trajectories"]
    if not mse_trajs:
        return

    min_len = min(len(t) for t in mse_trajs)
    mse_array = np.array([t[:min_len] for t in mse_trajs])
    mse_mean = mse_array.mean(axis=0)
    mse_std = mse_array.std(axis=0)
    times_s = np.arange(min_len) * 0.1  # 100-step downsampling at dt=1ms

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.semilogy(times_s, mse_mean, "r-", linewidth=1.5, label="Mean MSE")
    ax.fill_between(times_s,
                     np.maximum(mse_mean - mse_std, 1e-8),
                     mse_mean + mse_std,
                     alpha=0.2, color="red")
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("MSE (log scale)", fontsize=12)
    K = results["obs_every_k"]
    D = results["obs_delay"]
    ax.set_title(f"EnKF MSE Over Time — K={K}, delay={D}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, "mse_trajectory.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved MSE trajectory plot to {path}")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 17: Full-Trial EnKF Deployment Stability Test"
    )
    parser.add_argument("--data", type=str, default=ABS_DATA_PATH)
    parser.add_argument("--checkpoint", type=str, default=ABS_CAUSAL_CKPT)
    parser.add_argument("--obs-every-k", type=int, required=True,
                        help="Observation interval K: update every K steps")
    parser.add_argument("--obs-delay", type=int, required=True,
                        help="Observation delay D: use x(t-D) at update step t")
    parser.add_argument("--n-ensemble", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="results/fulltrial")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # Load model
    print(f"\nLoading causal model from {args.checkpoint}...")
    model, ckpt = load_causal_model(args.checkpoint, device=device)
    print(f"  n_x={ckpt['n_x']}, n_u={ckpt['n_u']}, z_dim={ckpt['z_dim']}, "
          f"hidden={ckpt['hidden']}, n_layers={ckpt['n_layers']}")

    # Fixed Q, R from Exp 15 best
    Q = 0.1
    R_val = 0.01

    print(f"\n{'='*80}")
    print(f"Experiment 17: Full-Trial EnKF")
    print(f"  K={args.obs_every_k}, delay={args.obs_delay}")
    print(f"  Q={Q}, R={R_val}, ensemble={args.n_ensemble}")
    print(f"{'='*80}")

    t_total_start = time.time()

    results = evaluate_fulltrial(
        model=model, ckpt=ckpt,
        data_path=args.data,
        obs_every_k=args.obs_every_k,
        obs_delay=args.obs_delay,
        n_ensemble=args.n_ensemble,
        Q=Q, R=R_val,
        device=device,
        seed=args.seed,
    )

    t_total = time.time() - t_total_start

    # Print summary
    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"  Config: K={args.obs_every_k}, delay={args.obs_delay}, Q={Q}, R={R_val}")
    print(f"  Ensemble size: {args.n_ensemble}")
    print(f"  FIT%: {results['avg_fit_pct']:.2f} ± {results['std_fit_pct']:.2f}")
    print(f"  Per-trial FIT%: {[f'{f:.2f}' for f in results['per_trial_fit_pct']]}")
    print(f"  Avg inference time per trial: {results['avg_inference_time_s']:.1f}s")
    print(f"  Total time: {t_total:.1f}s")

    print(f"\n  R² at key timepoints (200-step windows):")
    timepoint_labels = {
        "0": "0-200ms",
        "1000": "1s",
        "5000": "5s",
        "10000": "10s",
        "20000": "20s",
        "29000": "29s",
    }
    for pos_str, info in results["r2_at_positions"].items():
        label = timepoint_labels.get(pos_str, f"t={pos_str}")
        print(f"    {label}: R²={info['mean']:.4f} ± {info['std']:.4f}")

    # R² stability assessment
    traj = results["avg_r2_trajectory"]
    if len(traj) >= 10:
        early_r2 = np.mean([t["r2_mean"] for t in traj[:5]])
        late_r2 = np.mean([t["r2_mean"] for t in traj[-5:]])
        r2_drop = early_r2 - late_r2
        print(f"\n  R² stability: early={early_r2:.4f}, late={late_r2:.4f}, drop={r2_drop:.4f}")
        if abs(r2_drop) < 0.05:
            print(f"  => STABLE: R² does not degrade significantly over 30s")
        elif r2_drop > 0.05:
            print(f"  => DEGRADING: R² drops by {r2_drop:.4f} from early to late")
        else:
            print(f"  => IMPROVING: R² improves by {-r2_drop:.4f} from early to late")

    # Comparison
    print(f"\n  Comparison:")
    print(f"    Baseline (free-run, no EnKF): FIT% ≈ 0")
    print(f"    EnKF K={args.obs_every_k}, D={args.obs_delay}: FIT% = {results['avg_fit_pct']:.2f}")

    # Save results
    results["total_time_s"] = t_total
    results["args"] = vars(args)
    results_path = os.path.join(args.output_dir, "fulltrial_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, "item") else str(x))
    print(f"\nResults saved to {results_path}")

    # Plots
    plot_r2_trajectory(results, args.output_dir)
    plot_mse_trajectory(results, args.output_dir)

    print(f"\nDone! All outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
