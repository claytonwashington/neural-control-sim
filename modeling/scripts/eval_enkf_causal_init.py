"""EnKF with Causal Encoder Initialization — 4-way Sweep (Experiment 15).

Tests whether EnKF observation corrections can rescue the causal encoder
(R²=0.82) under realistic deployment conditions.

Setup:
  - ALWAYS uses the CAUSAL encoder for z₀ initialization (forward-only GRU)
  - ODE + decoder can come from either causal or acausal model (--ode-source)
  - Sweeps obs-every-k K ∈ {1, 10, 20, 50, 100, 999} and Q/R combinations
  - Supports observation delay: at update step t, use x(t - delay) instead of x(t)

Variants:
  A1: causal ODE, no delay       A2: causal ODE, 10ms delay
  B1: acausal ODE, no delay      B2: acausal ODE, 10ms delay
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
from modeling.models.latent_canode import LatentControlAffineODE


# --------------------------------------------------------------------------- #
#  Absolute paths to data and checkpoints (on the cluster)
# --------------------------------------------------------------------------- #
ABS_DATA_PATH = "/snel/home/cbwash2/cleo/data/training_trials.h5"
ABS_CAUSAL_CKPT = "/snel/home/cbwash2/cleo/results/causal_z64_pw200_h128_lr3e4/model.pt"
ABS_ACAUSAL_CKPT = "/snel/home/cbwash2/cleo/results/sweep_latent_node_40_10/run_05_z64_h256_l2_lr0.0005_dopri5.pt"


# --------------------------------------------------------------------------- #
#  Model loaders
# --------------------------------------------------------------------------- #

def load_causal_model(checkpoint_path: str, device: str = "cpu"):
    """Load the causal LatentControlAffineODE."""
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


def load_acausal_model(checkpoint_path: str, device: str = "cpu"):
    """Load the acausal LatentNeuralODE."""
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
#  Wrapper: make causal ODE (drift + control) look like acausal ode_func_net
# --------------------------------------------------------------------------- #

class CausalODEWrapper(nn.Module):
    """Wraps DriftNet f(z) + ControlNet g(z)@u into a single (z,u)->dz call.

    The acausal ODE's .net takes [z, u] concatenated -> dz/dt.
    The causal ODE has separate f(z) and g(z) networks.
    This wrapper presents the same interface: forward([z, u]) -> dz/dt.
    """

    def __init__(self, drift_net, control_net, z_dim: int, n_u: int):
        super().__init__()
        self.drift_net = drift_net
        self.control_net = control_net
        self.z_dim = z_dim
        self.n_u = n_u

    def forward(self, zu: torch.Tensor) -> torch.Tensor:
        """zu: (B, z_dim + n_u) -> dz: (B, z_dim)"""
        z = zu[:, :self.z_dim]
        u = zu[:, self.z_dim:]
        drift = self.drift_net(z)              # (B, z_dim)
        g_z = self.control_net(z)              # (B, z_dim, n_u)
        control = torch.bmm(g_z, u.unsqueeze(-1)).squeeze(-1)  # (B, z_dim)
        return drift + control


# --------------------------------------------------------------------------- #
#  Batched EnKF with sparse observations and delay
# --------------------------------------------------------------------------- #

class BatchedEnKFCausalInit:
    """EnKF with causal-encoder z₀, sparse observation updates, and delay.

    Parameters
    ----------
    ode_func_net : nn.Module
        Maps [z, u] -> dz/dt  (either acausal .net or CausalODEWrapper)
    decoder : nn.Linear
        Maps z -> x_hat
    z_dim, n_x : int
    n_ensemble : int
        Number of ensemble particles per window
    Q, R : float
        Process noise std, observation noise std
    obs_every_k : int
        Only perform Kalman update every K timesteps (1 = every step)
    obs_delay : int
        At update step t, use observation x(t - delay). Skip if t < delay.
    device : str
    """

    def __init__(self, ode_func_net, decoder, z_dim, n_x,
                 n_ensemble=64, Q=0.01, R=0.1,
                 obs_every_k=1, obs_delay=0, device="cpu"):
        self.ode_net = ode_func_net
        self.decoder = decoder
        self.z_dim = z_dim
        self.n_x = n_x
        self.N = n_ensemble
        self.Q = Q
        self.R = R
        self.obs_every_k = obs_every_k
        self.obs_delay = obs_delay
        self.device = device

    @torch.no_grad()
    def run_batched(self, z0_batch, u_batch, x_obs_batch, dt):
        """Run EnKF on multiple windows simultaneously.

        Parameters
        ----------
        z0_batch    : (W, z_dim)   initial states from causal encoder
        u_batch     : (W, T, n_u)  control inputs
        x_obs_batch : (W, T, n_x)  ground-truth observations
        dt          : float

        Returns
        -------
        x_preds     : (W, T, n_x)  decoded predictions from posterior means
        """
        W, T, n_u = u_batch.shape
        N = self.N
        device = self.device
        K_obs = self.obs_every_k
        D_obs = self.obs_delay

        # Initialize ensemble: (W, N, z_dim)
        z_ens = z0_batch.unsqueeze(1).expand(W, N, -1).clone()
        z_ens = z_ens + torch.randn_like(z_ens) * self.Q

        x_preds = torch.zeros(W, T, self.n_x, device=device)

        # Precompute observation noise covariance
        R2_eye = (self.R ** 2) * torch.eye(self.n_x, device=device)

        for t in range(T):
            # ---- PREDICT ----
            if t > 0:
                z_flat = z_ens.reshape(W * N, self.z_dim)
                u_t = u_batch[:, t - 1].unsqueeze(1).expand(W, N, n_u).reshape(W * N, n_u)
                zu = torch.cat([z_flat, u_t], dim=-1)
                dzdt = self.ode_net(zu)  # (W*N, z_dim)
                z_flat = z_flat + dt * dzdt
                z_ens = z_flat.reshape(W, N, self.z_dim)
                z_ens = z_ens + torch.randn_like(z_ens) * self.Q

            # ---- UPDATE (only if conditions met) ----
            do_update = (t % K_obs == 0)
            obs_t_idx = t - D_obs  # which observation to use (delayed)

            if do_update and obs_t_idx >= 0 and obs_t_idx < T:
                # Predicted observations from ensemble
                z_flat = z_ens.reshape(W * N, self.z_dim)
                x_pred_flat = self.decoder(z_flat)  # (W*N, n_x)
                x_pred_ens = x_pred_flat.reshape(W, N, self.n_x)

                # Ensemble means
                z_mean = z_ens.mean(dim=1)              # (W, z_dim)
                x_pred_mean = x_pred_ens.mean(dim=1)    # (W, n_x)

                # Anomalies
                dz = z_ens - z_mean.unsqueeze(1)              # (W, N, z_dim)
                dx = x_pred_ens - x_pred_mean.unsqueeze(1)    # (W, N, n_x)

                # Cross-covariance Pzx: (W, z_dim, n_x)
                Pzx = torch.bmm(dz.transpose(1, 2), dx) / (N - 1)

                # Observation covariance Pxx: (W, n_x, n_x)
                Pxx = torch.bmm(dx.transpose(1, 2), dx) / (N - 1)
                Pxx = Pxx + R2_eye.unsqueeze(0)

                # Kalman gain: K = Pzx @ inv(Pxx)
                K_gain = torch.linalg.solve(
                    Pxx.transpose(1, 2), Pzx.transpose(1, 2)
                ).transpose(1, 2)

                # Stochastic EnKF update with perturbed (delayed) observations
                x_obs_t = x_obs_batch[:, obs_t_idx]  # (W, n_x) — delayed!
                obs_perturbed = x_obs_t.unsqueeze(1) + \
                    torch.randn(W, N, self.n_x, device=device) * self.R
                innovations = obs_perturbed - x_pred_ens  # (W, N, n_x)

                # Update ensemble
                z_ens = z_ens + torch.bmm(innovations, K_gain.transpose(1, 2))

            # Record prediction from posterior mean
            z_mean = z_ens.mean(dim=1)  # (W, z_dim)
            x_preds[:, t] = self.decoder(z_mean)

        return x_preds


# --------------------------------------------------------------------------- #
#  Evaluation loop
# --------------------------------------------------------------------------- #

def evaluate_enkf(
    causal_model, causal_ckpt,
    ode_net, decoder, ode_ckpt,
    data_path, Q, R, obs_every_k,
    obs_delay=0, n_ensemble=64, horizon=200, stride=200,
    device="cpu", seed=DEFAULT_SEED, verbose=True,
    window_batch_size=64,
):
    """Evaluate EnKF on test trials using windowed R².

    Uses the CAUSAL encoder for z₀ (always), but ODE+decoder from ode_source.
    """
    data = load_trials_h5(data_path)
    x_all, u_all = data["x"], data["u"]
    dt = float(data["dt"])
    n_trials, n_channels, n_steps = x_all.shape

    n_test = DEFAULT_TEST_TRIALS
    n_train = n_trials - n_test

    x_test_trials = x_all[n_train:]
    u_test_trials = u_all[n_train:]

    # Normalization stats from the causal model (used by causal encoder)
    c_x_mean = causal_ckpt["x_mean"]
    c_x_std = causal_ckpt["x_std"]
    c_u_mean = causal_ckpt["u_mean"]
    c_u_std = causal_ckpt["u_std"]

    # Normalization stats from the ODE model (used by ODE + decoder)
    o_x_mean = ode_ckpt["x_mean"]
    o_x_std = ode_ckpt["x_std"]
    o_u_mean = ode_ckpt["u_mean"]
    o_u_std = ode_ckpt["u_std"]

    z_dim = ode_ckpt["z_dim"]
    n_x = ode_ckpt["n_x"]

    enkf = BatchedEnKFCausalInit(
        ode_func_net=ode_net, decoder=decoder,
        z_dim=z_dim, n_x=n_x,
        n_ensemble=n_ensemble, Q=Q, R=R,
        obs_every_k=obs_every_k, obs_delay=obs_delay,
        device=device,
    )

    windowed_mses = []
    windowed_r2s = []
    inference_times = []

    for ti in range(n_test):
        x_test_n = x_test_trials[ti]  # (n_channels, n_steps)
        u_test_n = u_test_trials[ti]  # (n_u, n_steps)
        T_val = x_test_n.shape[1]

        # -- Normalize for causal encoder --
        # x_mean shape is (50, 1), broadcasts with (50, 30000) directly
        x_norm_c = (x_test_n - c_x_mean) / (c_x_std + 1e-8)
        u_norm_c = (u_test_n - c_u_mean) / (c_u_std + 1e-8)

        # -- Normalize for ODE/decoder --
        x_norm_o = (x_test_n - o_x_mean) / (o_x_std + 1e-8)
        u_norm_o = (u_test_n - o_u_mean) / (o_u_std + 1e-8)

        # Create windows for causal encoder (needs past context = full window)
        x_windows_c = []
        u_windows_c = []
        # Create windows for ODE/decoder
        x_windows_o = []
        u_windows_o = []

        for w in range(0, 10000):
            t0 = w * stride
            t1 = t0 + horizon
            if t1 > T_val:
                break
            x_windows_c.append(x_norm_c[:, t0:t1].T)  # (T, n_x)
            u_windows_c.append(u_norm_c[:, t0:t1].T)  # (T, n_u)
            x_windows_o.append(x_norm_o[:, t0:t1].T)
            u_windows_o.append(u_norm_o[:, t0:t1].T)

        x_windows_c = torch.tensor(np.array(x_windows_c), dtype=torch.float32, device=device)
        u_windows_c = torch.tensor(np.array(u_windows_c), dtype=torch.float32, device=device)
        x_windows_o = torch.tensor(np.array(x_windows_o), dtype=torch.float32, device=device)
        u_windows_o = torch.tensor(np.array(u_windows_o), dtype=torch.float32, device=device)
        W_actual = x_windows_c.shape[0]

        # Get z0 from CAUSAL encoder (always uses causal normalization)
        z0_all = []
        enc_batch = min(window_batch_size, W_actual)
        for b_start in range(0, W_actual, enc_batch):
            b_end = min(b_start + enc_batch, W_actual)
            with torch.no_grad():
                mu, _ = causal_model.encoder(
                    x_windows_c[b_start:b_end],
                    u_windows_c[b_start:b_end],
                )
                z0_all.append(mu)
        z0_all = torch.cat(z0_all, dim=0)  # (W, z_dim)

        # Run EnKF in batches of windows (ODE + decoder use their own normalization)
        trial_mse_sum = 0.0
        trial_windows = 0
        t_start = time.perf_counter()

        for b_start in range(0, W_actual, window_batch_size):
            b_end = min(b_start + window_batch_size, W_actual)
            x_preds = enkf.run_batched(
                z0_all[b_start:b_end],
                u_windows_o[b_start:b_end],
                x_windows_o[b_start:b_end],
                dt,
            )
            # MSE per window (in normalized space)
            mse_per_window = ((x_preds - x_windows_o[b_start:b_end]) ** 2).mean(dim=(1, 2))
            trial_mse_sum += mse_per_window.sum().item()
            trial_windows += b_end - b_start

        t_end = time.perf_counter()
        inference_times.append((t_end - t_start) * 1000.0)

        trial_mse = trial_mse_sum / trial_windows
        trial_var = float(np.var(x_norm_o))
        trial_r2 = 1 - trial_mse / trial_var
        windowed_mses.append(trial_mse)
        windowed_r2s.append(trial_r2)
        if verbose:
            print(f"  Trial {n_train + ti}: MSE={trial_mse:.6f}, R²={trial_r2:.4f}")

    avg_r2 = float(np.mean(windowed_r2s))
    avg_mse = float(np.mean(windowed_mses))
    avg_time = float(np.mean(inference_times))

    return {
        "Q": Q, "R": R, "obs_every_k": obs_every_k, "obs_delay": obs_delay,
        "n_ensemble": n_ensemble,
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
        description="EnKF with causal encoder init — 4-way sweep"
    )
    parser.add_argument("--data", type=str, default=ABS_DATA_PATH)
    parser.add_argument("--causal-checkpoint", type=str, default=ABS_CAUSAL_CKPT)
    parser.add_argument("--acausal-checkpoint", type=str, default=ABS_ACAUSAL_CKPT)
    parser.add_argument("--ode-source", type=str, required=True,
                        choices=["causal", "acausal"],
                        help="Which model's ODE+decoder to use for EnKF dynamics")
    parser.add_argument("--obs-delay", type=int, default=0,
                        help="Observation delay D: use x(t-D) at update step t")
    parser.add_argument("--output-dir", type=str, default="results/enkf_causal_init")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n-ensemble", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--stride", type=int, default=200)
    parser.add_argument("--window-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # ------------------------------------------------------------------ #
    # Load models
    # ------------------------------------------------------------------ #
    print(f"\nLoading causal model (encoder source) from {args.causal_checkpoint}...")
    causal_model, causal_ckpt = load_causal_model(args.causal_checkpoint, device=device)
    print(f"  n_x={causal_ckpt['n_x']}, n_u={causal_ckpt['n_u']}, z_dim={causal_ckpt['z_dim']}, "
          f"hidden={causal_ckpt['hidden']}, n_layers={causal_ckpt['n_layers']}")

    print(f"\nLoading acausal model from {args.acausal_checkpoint}...")
    acausal_model, acausal_ckpt = load_acausal_model(args.acausal_checkpoint, device=device)
    print(f"  n_x={acausal_ckpt['n_x']}, n_u={acausal_ckpt['n_u']}, z_dim={acausal_ckpt['z_dim']}, "
          f"hidden={acausal_ckpt['hidden']}, n_layers={acausal_ckpt['n_layers']}")

    # Select ODE + decoder based on --ode-source
    if args.ode_source == "causal":
        print("\n=> Using CAUSAL ODE (drift+control) + decoder")
        ode_net = CausalODEWrapper(
            drift_net=causal_model.f,
            control_net=causal_model.g,
            z_dim=causal_ckpt["z_dim"],
            n_u=causal_ckpt["n_u"],
        )
        decoder = causal_model.decoder
        ode_ckpt = causal_ckpt
    else:
        print("\n=> Using ACAUSAL ODE (.net MLP) + decoder")
        ode_net = acausal_model.ode_func.net
        decoder = acausal_model.decoder
        ode_ckpt = acausal_ckpt

    # ------------------------------------------------------------------ #
    # Sweep parameters
    # ------------------------------------------------------------------ #
    K_values = [1, 10, 20, 50, 100, 999]
    Q_values = [0.01, 0.1]
    R_values = [0.01, 0.1]

    all_results = []
    best_r2 = -float("inf")
    best_config = None

    n_configs = len(K_values) * len(Q_values) * len(R_values)
    print(f"\n{'='*80}")
    print(f"EnKF Causal-Init Sweep: ode_source={args.ode_source}, obs_delay={args.obs_delay}")
    print(f"  {len(K_values)} K × {len(Q_values)} Q × {len(R_values)} R = {n_configs} configs")
    print(f"  Ensemble size: {args.n_ensemble}")
    print(f"  Horizon: {args.horizon}, Stride: {args.stride}")
    print(f"  Window batch size: {args.window_batch_size}")
    print(f"{'='*80}")

    config_idx = 0
    for K in K_values:
        for Q in Q_values:
            for R in R_values:
                config_idx += 1
                print(f"\n[{config_idx}/{n_configs}] K={K}, Q={Q}, R={R}, delay={args.obs_delay}")
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)

                result = evaluate_enkf(
                    causal_model=causal_model, causal_ckpt=causal_ckpt,
                    ode_net=ode_net, decoder=decoder, ode_ckpt=ode_ckpt,
                    data_path=args.data,
                    Q=Q, R=R, obs_every_k=K,
                    obs_delay=args.obs_delay,
                    n_ensemble=args.n_ensemble,
                    horizon=args.horizon, stride=args.stride,
                    device=device, seed=args.seed,
                    window_batch_size=args.window_batch_size,
                )
                all_results.append(result)
                print(f"  => R²={result['avg_r2']:.4f}, MSE={result['avg_mse']:.6f}, "
                      f"Time={result['avg_inference_ms']:.1f}ms")

                if result["avg_r2"] > best_r2:
                    best_r2 = result["avg_r2"]
                    best_config = result

    # ------------------------------------------------------------------ #
    # Summary table
    # ------------------------------------------------------------------ #
    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY — ode_source={args.ode_source}, obs_delay={args.obs_delay}")
    print(f"{'='*80}")
    print(f"{'K':>6s} {'Q':>8s} {'R':>8s} {'R²':>10s} {'MSE':>12s} {'Time(ms)':>10s}")
    print("-" * 58)
    for r in all_results:
        marker = " *" if r["avg_r2"] == best_r2 else ""
        print(f"{r['obs_every_k']:6d} {r['Q']:8.3f} {r['R']:8.3f} {r['avg_r2']:10.4f} "
              f"{r['avg_mse']:12.6f} {r['avg_inference_ms']:10.1f}{marker}")

    print(f"\nBest config: K={best_config['obs_every_k']}, Q={best_config['Q']}, R={best_config['R']}")
    print(f"  R²={best_config['avg_r2']:.4f}, MSE={best_config['avg_mse']:.6f}")
    print(f"  Avg inference time per trial: {best_config['avg_inference_ms']:.1f}ms")

    # ------------------------------------------------------------------ #
    # Comparison context
    # ------------------------------------------------------------------ #
    print(f"\n{'='*80}")
    print("COMPARISON CONTEXT")
    print(f"{'='*80}")
    print(f"  Causal baseline (LatentCANODE, R²=0.8252): causal encoder + CA-ODE")
    print(f"  Acausal baseline (LatentNODE, R²=0.9387): bidirectional encoder")
    print(f"  This run (ode_source={args.ode_source}, delay={args.obs_delay}):")
    print(f"    Best R² = {best_config['avg_r2']:.4f}")
    if best_config['avg_r2'] > 0.8252:
        print(f"    => IMPROVEMENT over causal baseline by {best_config['avg_r2'] - 0.8252:.4f}")
    else:
        print(f"    => Below causal baseline by {0.8252 - best_config['avg_r2']:.4f}")

    # ------------------------------------------------------------------ #
    # Save results
    # ------------------------------------------------------------------ #
    results_path = os.path.join(args.output_dir, "enkf_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "ode_source": args.ode_source,
            "obs_delay": args.obs_delay,
            "all_results": all_results,
            "best_config": best_config,
            "args": vars(args),
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print(f"\nDone! All results in {args.output_dir}/")


if __name__ == "__main__":
    main()
