"""Ensemble Kalman Filter (EnKF) with explicit Euler integration.

Uses the acausal LatentNeuralODE learned dynamics with a single fixed-step
Euler integrator in the predict step, instead of an adaptive ODE solver.

The EnKF operates per-timestep:
  PREDICT: z_i(t+1) = z_i(t) + dt * f(z_i(t), u(t)) + noise(Q)
  UPDATE:  z_i(t+1) += K * (x_obs - x_pred_i)  via ensemble Kalman gain K

This is the "fast" variant: the predict step uses a single Euler step
(one MLP forward pass), eliminating all adaptive solver overhead.

Evaluation uses 200-step windowed R2 for comparison with other models.
Optimized: batches ALL windows of a trial through the EnKF simultaneously.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from modeling.config import DEFAULT_SEED, DEFAULT_TEST_TRIALS, set_seed
from modeling.data import load_trials_h5
from modeling.models.latent_node import LatentNeuralODE


# --------------------------------------------------------------------------- #
#  Load acausal model
# --------------------------------------------------------------------------- #

def load_acausal_model(checkpoint_path: str, device: str = "cpu"):
    """Load the acausal LatentNeuralODE and return its components."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = LatentNeuralODE(
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
#  Batched Ensemble Kalman Filter (Euler predict step)
# --------------------------------------------------------------------------- #

class BatchedEnKF_Euler:
    """EnKF with fixed-step Euler integration in the predict step.

    The predict step uses a single Euler step per timestep:
        z(t+1) = z(t) + dt * f_theta(z(t), u(t))
    where f_theta is the learned ODE dynamics MLP.

    This eliminates all adaptive solver overhead (no dopri5, no step-size
    control, no error estimation). The Kalman update at each timestep
    compensates for the cruder dynamics prediction.

    Parameters
    ----------
    ode_func_net : nn.Sequential
        MLP mapping [z, u] -> dz/dt  (the learned dynamics)
    decoder : nn.Linear
        Maps z -> x_hat
    z_dim, n_x : int
    n_ensemble : int
        Number of ensemble members (particles) per window
    Q, R : float
        Process noise std, observation noise std
    device : str
    """

    def __init__(self, ode_func_net, decoder, z_dim, n_x,
                 n_ensemble=64, Q=0.01, R=0.1, device="cpu"):
        self.ode_net = ode_func_net
        self.decoder = decoder
        self.z_dim = z_dim
        self.n_x = n_x
        self.N = n_ensemble
        self.Q = Q
        self.R = R
        self.device = device

    @torch.no_grad()
    def run_batched(self, z0_batch, u_batch, x_obs_batch, dt):
        """Run EnKF on multiple windows simultaneously.

        Parameters
        ----------
        z0_batch    : (W, z_dim)        initial states for W windows
        u_batch     : (W, T, n_u)       control inputs
        x_obs_batch : (W, T, n_x)       observations
        dt          : float             timestep size

        Returns
        -------
        x_preds     : (W, T, n_x)  decoded predictions from posterior means
        """
        W, T, n_u = u_batch.shape
        N = self.N
        device = self.device

        # Initialize ensemble: (W, N, z_dim)
        z_ens = z0_batch.unsqueeze(1).expand(W, N, -1).clone()
        z_ens = z_ens + torch.randn_like(z_ens) * self.Q

        x_preds = torch.zeros(W, T, self.n_x, device=device)

        # Precompute observation noise covariance
        R2_eye = (self.R ** 2) * torch.eye(self.n_x, device=device)

        for t in range(T):
            # ---- PREDICT (Euler step) ----
            if t > 0:
                # Flatten for MLP: (W*N, z_dim + n_u)
                z_flat = z_ens.reshape(W * N, self.z_dim)
                # u at t-1 for predict step, broadcast to all particles
                u_t = u_batch[:, t - 1].unsqueeze(1).expand(W, N, n_u).reshape(W * N, n_u)
                zu = torch.cat([z_flat, u_t], dim=-1)

                # Single Euler step: z(t) = z(t-1) + dt * f(z(t-1), u(t-1))
                dzdt = self.ode_net(zu)  # (W*N, z_dim)
                z_flat = z_flat + dt * dzdt
                z_ens = z_flat.reshape(W, N, self.z_dim)

                # Add process noise
                z_ens = z_ens + torch.randn_like(z_ens) * self.Q

            # ---- UPDATE (Kalman correction) ----
            # Predicted observations: (W, N, n_x)
            z_flat = z_ens.reshape(W * N, self.z_dim)
            x_pred_flat = self.decoder(z_flat)  # (W*N, n_x)
            x_pred_ens = x_pred_flat.reshape(W, N, self.n_x)

            # Ensemble means
            z_mean = z_ens.mean(dim=1)             # (W, z_dim)
            x_pred_mean = x_pred_ens.mean(dim=1)   # (W, n_x)

            # Anomalies
            dz = z_ens - z_mean.unsqueeze(1)             # (W, N, z_dim)
            dx = x_pred_ens - x_pred_mean.unsqueeze(1)   # (W, N, n_x)

            # Cross-cov Pzx: (W, z_dim, n_x)
            Pzx = torch.bmm(dz.transpose(1, 2), dx) / (N - 1)

            # Obs-cov Pxx: (W, n_x, n_x)
            Pxx = torch.bmm(dx.transpose(1, 2), dx) / (N - 1)
            Pxx = Pxx + R2_eye.unsqueeze(0)

            # Kalman gain: K = Pzx @ inv(Pxx), (W, z_dim, n_x)
            K = torch.linalg.solve(Pxx.transpose(1, 2), Pzx.transpose(1, 2)).transpose(1, 2)

            # Stochastic EnKF update with perturbed observations
            x_obs_t = x_obs_batch[:, t]  # (W, n_x)
            obs_perturbed = x_obs_t.unsqueeze(1) + torch.randn(W, N, self.n_x, device=device) * self.R
            innovations = obs_perturbed - x_pred_ens  # (W, N, n_x)

            # Update: z_ens += innovations @ K^T  -> (W, N, z_dim)
            z_ens = z_ens + torch.bmm(innovations, K.transpose(1, 2))

            # Record prediction from posterior mean
            z_mean = z_ens.mean(dim=1)  # (W, z_dim)
            x_preds[:, t] = self.decoder(z_mean)

        return x_preds


# --------------------------------------------------------------------------- #
#  Evaluation
# --------------------------------------------------------------------------- #

def evaluate_enkf(
    model, ckpt, data_path, Q, R,
    n_ensemble=64, horizon=200, stride=200,
    device="cpu", seed=DEFAULT_SEED, verbose=True,
    window_batch_size=64,
):
    """Evaluate EnKF (Euler) on test trials using windowed R2."""
    data = load_trials_h5(data_path)
    x_all, u_all = data["x"], data["u"]
    dt = float(data["dt"])
    n_trials, n_channels, n_steps = x_all.shape

    n_test = DEFAULT_TEST_TRIALS
    n_train = n_trials - n_test
    x_test_trials = x_all[n_train:]
    u_test_trials = u_all[n_train:]

    x_mean = ckpt["x_mean"]
    x_std = ckpt["x_std"]
    u_mean = ckpt["u_mean"]
    u_std = ckpt["u_std"]

    z_dim = ckpt["z_dim"]
    n_x = ckpt["n_x"]

    ode_net = model.ode_func.net
    decoder = model.decoder

    enkf = BatchedEnKF_Euler(
        ode_func_net=ode_net, decoder=decoder,
        z_dim=z_dim, n_x=n_x,
        n_ensemble=n_ensemble, Q=Q, R=R, device=device,
    )

    windowed_r2s = []
    windowed_mses = []
    inference_times = []

    for ti in range(n_test):
        x_test = x_test_trials[ti]
        u_test = u_test_trials[ti]
        x_test_n = (x_test - x_mean) / x_std
        u_test_n = (u_test - u_mean) / u_std

        n_ch, T_val = x_test_n.shape
        n_windows = (T_val - horizon) // stride
        if n_windows <= 0:
            n_windows = 1

        # Prepare all windows for this trial
        x_windows = []
        u_windows = []
        for w in range(n_windows):
            t0 = w * stride
            t1 = t0 + horizon
            if t1 > T_val:
                break
            x_windows.append(x_test_n[:, t0:t1].T)  # (T, n_x)
            u_windows.append(u_test_n[:, t0:t1].T)  # (T, n_u)

        x_windows = torch.tensor(np.array(x_windows), dtype=torch.float32, device=device)  # (W, T, n_x)
        u_windows = torch.tensor(np.array(u_windows), dtype=torch.float32, device=device)  # (W, T, n_u)
        W_actual = x_windows.shape[0]

        # Get z0 for all windows via encoder (process in batches to avoid OOM)
        z0_all = []
        enc_batch = min(window_batch_size, W_actual)
        for b_start in range(0, W_actual, enc_batch):
            b_end = min(b_start + enc_batch, W_actual)
            with torch.no_grad():
                mu, _ = model.encoder(x_windows[b_start:b_end], u_windows[b_start:b_end])
                z0_all.append(mu)
        z0_all = torch.cat(z0_all, dim=0)  # (W, z_dim)

        # Run EnKF in batches of windows
        trial_mse_sum = 0.0
        trial_windows = 0

        # Synchronize CUDA for accurate timing
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        for b_start in range(0, W_actual, window_batch_size):
            b_end = min(b_start + window_batch_size, W_actual)
            x_preds = enkf.run_batched(
                z0_all[b_start:b_end],
                u_windows[b_start:b_end],
                x_windows[b_start:b_end],
                dt,
            )
            # MSE per window
            mse_per_window = ((x_preds - x_windows[b_start:b_end]) ** 2).mean(dim=(1, 2))
            trial_mse_sum += mse_per_window.sum().item()
            trial_windows += b_end - b_start

        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.perf_counter()
        inference_times.append((t_end - t_start) * 1000.0)

        trial_mse = trial_mse_sum / trial_windows
        trial_var = float(np.var(x_test_n))
        trial_r2 = 1 - trial_mse / trial_var
        windowed_mses.append(trial_mse)
        windowed_r2s.append(trial_r2)
        if verbose:
            print(f"  Trial {n_train + ti}: MSE={trial_mse:.6f}, R2={trial_r2:.4f}")

    avg_r2 = float(np.mean(windowed_r2s))
    avg_mse = float(np.mean(windowed_mses))
    avg_time = float(np.mean(inference_times))

    return {
        "Q": Q, "R": R, "n_ensemble": n_ensemble,
        "avg_r2": avg_r2, "avg_mse": avg_mse,
        "avg_inference_ms": avg_time,
        "per_trial_r2": [float(r) for r in windowed_r2s],
        "per_trial_mse": [float(m) for m in windowed_mses],
    }


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="EnKF evaluation with Euler integration (fast variant)"
    )
    parser.add_argument("--data", type=str, default="data/training_trials.h5")
    parser.add_argument("--checkpoint", type=str,
                        default="results/sweep_latent_node_40_10/run_05_z64_h256_l2_lr0.0005_dopri5.pt")
    parser.add_argument("--output-dir", type=str, default="results/enkf_euler")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n-ensemble", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--stride", type=int, default=200)
    parser.add_argument("--window-batch-size", type=int, default=32,
                        help="Number of windows to process simultaneously in the EnKF")
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
    print(f"\nLoading acausal model from {args.checkpoint}...")
    model, ckpt = load_acausal_model(args.checkpoint, device=device)
    print(f"  n_x={ckpt['n_x']}, n_u={ckpt['n_u']}, z_dim={ckpt['z_dim']}, "
          f"hidden={ckpt['hidden']}, n_layers={ckpt['n_layers']}")

    # ------------------------------------------------------------------ #
    # GPU warmup
    # ------------------------------------------------------------------ #
    if device != "cpu":
        print("\nGPU warmup...")
        dummy_z = torch.randn(64, ckpt["z_dim"], device=device)
        dummy_u = torch.randn(64, ckpt["n_u"], device=device)
        dummy_zu = torch.cat([dummy_z, dummy_u], dim=-1)
        for _ in range(10):
            _ = model.ode_func.net(dummy_zu)
        torch.cuda.synchronize()
        print("  Warmup complete.")

    # ------------------------------------------------------------------ #
    # Sweep Q and R
    # ------------------------------------------------------------------ #
    Q_values = [0.001, 0.01, 0.1]
    R_values = [0.01, 0.1, 1.0]

    all_results = []
    best_r2 = -float("inf")
    best_config = None

    print(f"\n{'='*70}")
    print(f"EnKF (Euler) Sweep: {len(Q_values)} Q x {len(R_values)} R = {len(Q_values)*len(R_values)} configs")
    print(f"  Integration method: Fixed-step Euler (single MLP eval per step)")
    print(f"  Ensemble size: {args.n_ensemble}")
    print(f"  Horizon: {args.horizon}, Stride: {args.stride}")
    print(f"  Window batch size: {args.window_batch_size}")
    print(f"{'='*70}")

    for Q in Q_values:
        for R in R_values:
            print(f"\n--- Q={Q}, R={R} ---")
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)

            result = evaluate_enkf(
                model=model, ckpt=ckpt, data_path=args.data,
                Q=Q, R=R,
                n_ensemble=args.n_ensemble,
                horizon=args.horizon, stride=args.stride,
                device=device, seed=args.seed,
                window_batch_size=args.window_batch_size,
            )
            all_results.append(result)
            print(f"  => R2={result['avg_r2']:.4f}, MSE={result['avg_mse']:.6f}, "
                  f"Time={result['avg_inference_ms']:.1f}ms")

            if result["avg_r2"] > best_r2:
                best_r2 = result["avg_r2"]
                best_config = result

    # ------------------------------------------------------------------ #
    # Summary table
    # ------------------------------------------------------------------ #
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY (EnKF Euler)")
    print(f"{'='*70}")
    print(f"{'Q':>8s} {'R':>8s} {'R2':>10s} {'MSE':>12s} {'Time(ms)':>10s}")
    print("-" * 52)
    for r in all_results:
        marker = " *" if r["avg_r2"] == best_r2 else ""
        print(f"{r['Q']:8.3f} {r['R']:8.3f} {r['avg_r2']:10.4f} {r['avg_mse']:12.6f} "
              f"{r['avg_inference_ms']:10.1f}{marker}")

    print(f"\nBest config: Q={best_config['Q']}, R={best_config['R']}")
    print(f"  R2={best_config['avg_r2']:.4f}, MSE={best_config['avg_mse']:.6f}")
    print(f"  Avg inference time per trial: {best_config['avg_inference_ms']:.1f}ms")

    # ------------------------------------------------------------------ #
    # Ensemble convergence with best Q/R
    # ------------------------------------------------------------------ #
    print(f"\n{'='*70}")
    print("ENSEMBLE SIZE CONVERGENCE (best Q/R)")
    print(f"{'='*70}")

    for N_ens in [32, 64, 128]:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        r = evaluate_enkf(
            model=model, ckpt=ckpt, data_path=args.data,
            Q=best_config["Q"], R=best_config["R"],
            n_ensemble=N_ens,
            horizon=args.horizon, stride=args.stride,
            device=device, seed=args.seed, verbose=False,
            window_batch_size=args.window_batch_size,
        )
        print(f"  N={N_ens:4d}: R2={r['avg_r2']:.4f}, MSE={r['avg_mse']:.6f}, "
              f"Time={r['avg_inference_ms']:.1f}ms")

    # Save
    results_path = os.path.join(args.output_dir, "enkf_euler_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "all_results": all_results,
            "best_config": best_config,
            "args": vars(args),
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ------------------------------------------------------------------ #
    # Comparison context
    # ------------------------------------------------------------------ #
    print(f"\n{'='*70}")
    print("COMPARISON CONTEXT")
    print(f"{'='*70}")
    print(f"  Best acausal (LatentNODE, full-seq R2=0.9387): bidirectional encoder (acausal)")
    print(f"  Best causal  (LatentCANODE, R2=0.8252):        causal encoder + CA-ODE")
    print(f"  EnKF Euler (this run):                         R2={best_config['avg_r2']:.4f}")
    print(f"\nDone! All results in {args.output_dir}/")


if __name__ == "__main__":
    main()
