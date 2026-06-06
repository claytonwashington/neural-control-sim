#!/usr/bin/env python3
"""Experiment 22: Periodic Re-Encoding vs EnKF.

Instead of running an ensemble Kalman filter, simply re-encode z₀ from
the GRU encoder every K steps. Between encodes, propagate z via ODE.

This tests whether periodic fresh state estimates from the aligned encoder
can match or exceed EnKF corrections — without ensemble overhead.

Sweep:
  - Re-encode period K ∈ {1, 5, 10, 20, 50, 100, 200}
  - Observation delay D ∈ {0, 10}
  - K=200 = encode once (baseline)
  - K=1 = re-encode every step (encoder-only, no ODE)

For delay D:
  At step s, encoder uses x[t-D-pw : t-D] (delayed observations).
  Then ODE predicts forward D steps to reach current time.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import h5py
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modeling.models.latent_canode import LatentControlAffineODE


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = LatentControlAffineODE(
        n_x=ckpt["n_x"], n_u=ckpt["n_u"], z_dim=ckpt["z_dim"],
        hidden_dim=ckpt["hidden"], n_layers=ckpt["n_layers"],
    ).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt


def ode_step_euler(model, z, u, dt=0.001):
    """Single Euler step: z' = z + dt * (f(z) + g(z) @ u)."""
    dz = model.f(z) + (model.g(z) @ u.unsqueeze(-1)).squeeze(-1)
    return z + dt * dz


def evaluate_periodic_reencode(model, x_test_n, u_test_n, ckpt,
                                 K, delay, pw, fw, device):
    """Evaluate periodic re-encoding strategy.
    
    Args:
        K: re-encode every K steps. K >= fw means encode once.
        delay: observation delay in ms (steps).
        pw: past window (encoder context).
        fw: future window (prediction horizon).
    """
    x_mean = torch.tensor(np.asarray(ckpt["x_mean"]).flatten(), dtype=torch.float32, device=device)
    x_std = torch.tensor(np.asarray(ckpt["x_std"]).flatten(), dtype=torch.float32, device=device)
    u_mean = torch.tensor(np.asarray(ckpt["u_mean"]).flatten(), dtype=torch.float32, device=device)
    u_std = torch.tensor(np.asarray(ckpt["u_std"]).flatten(), dtype=torch.float32, device=device)

    all_preds, all_trues = [], []
    total_time = 0
    n_windows = 0

    with torch.no_grad():
        for trial in range(x_test_n.shape[0]):
            x_trial = torch.tensor(x_test_n[trial].T, dtype=torch.float32, device=device)
            u_trial = torch.tensor(u_test_n[trial].T, dtype=torch.float32, device=device)
            n_steps = x_trial.shape[0]

            # Need enough room for: delay + pw + fw
            start_offset = delay + pw
            for t0 in range(start_offset, n_steps - fw, pw):
                t_start = time.perf_counter()

                preds_window = []
                z = None

                for s in range(fw):
                    t_now = t0 + s  # current prediction step

                    # Should we re-encode at this step?
                    if s % K == 0 or z is None:
                        # Encoder uses observations up to (t_now - delay)
                        enc_end = t_now - delay
                        enc_start = enc_end - pw

                        if enc_start < 0:
                            # Not enough history, use what we have
                            enc_start = 0
                            enc_end = pw

                        x_context = x_trial[enc_start:enc_end].unsqueeze(0)  # (1, pw, n_x)
                        u_context = u_trial[enc_start:enc_end].unsqueeze(0)  # (1, pw, n_u)

                        mu, _ = model.encoder(x_context, u_context)
                        z = mu  # (1, z_dim)

                        # If delay > 0, propagate z forward through the delay
                        for d in range(delay):
                            t_d = enc_end + d
                            if t_d < n_steps:
                                u_d = u_trial[t_d].unsqueeze(0)
                            else:
                                u_d = torch.zeros(1, u_trial.shape[1], device=device)
                            z = ode_step_euler(model, z, u_d)

                    else:
                        # ODE predict step
                        u_now = u_trial[t_now].unsqueeze(0) if t_now < n_steps else torch.zeros(1, u_trial.shape[1], device=device)
                        z = ode_step_euler(model, z, u_now)

                    # Decode prediction
                    x_pred = model.decoder(z)  # (1, n_x)
                    preds_window.append(x_pred.squeeze(0).cpu().numpy())

                torch.cuda.synchronize() if device.type == "cuda" else None
                total_time += time.perf_counter() - t_start

                preds_window = np.array(preds_window)  # (fw, n_x)
                trues_window = x_trial[t0:t0 + fw].cpu().numpy()  # (fw, n_x)

                all_preds.append(preds_window)
                all_trues.append(trues_window)
                n_windows += 1

    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)

    ss_res = np.sum((all_trues - all_preds) ** 2)
    ss_tot = np.sum((all_trues - all_trues.mean(axis=0)) ** 2)
    r2 = 1 - ss_res / ss_tot
    mse = np.mean((all_trues - all_preds) ** 2)
    avg_time = total_time / n_windows * 1000  # ms per window

    return r2, mse, avg_time, n_windows




def update_leaderboard(results_list):
    import sys
    import json
    import subprocess
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    leaderboard_path = os.path.join(repo_root, 'results/leaderboard.json')
    if not os.path.exists(leaderboard_path):
        print(f'Leaderboard file not found: {leaderboard_path}')
        return
        
    with open(leaderboard_path, 'r', encoding='utf-8') as f:
        leaderboard = json.load(f)
        
    targets = [
        {'K': 5, 'D': 0, 'name': 'Causal Latent CA-NODE (K=5, D=0)', 'type': 'Causal Latent ODE (Re-encode)', 'verdict': '✅ Zero-delay Re-encode', 'verdict_color': 'var(--cyan)'},
        {'K': 20, 'D': 10, 'name': 'Causal Latent CA-NODE (K=20, D=10)', 'type': 'Causal Latent ODE (Re-encode)', 'verdict': '✅ 10ms delay, 50Hz Re-encode', 'verdict_color': 'var(--green)', 'is_best_row': True, 'r2_style': 'color:var(--green);font-weight:700'}
    ]
    
    updated_any = False
    for target in targets:
        res = None
        for r in results_list:
            if r['K'] == target['K'] and r['delay'] == target['D']:
                res = r
                break
        if res is None:
            continue
            
        found_idx = -1
        for idx, entry in enumerate(leaderboard):
            if entry.get('model') == target['name']:
                found_idx = idx
                break
                
        entry = {
            'model': target['name'],
            'type': target['type'],
            'r2': round(res['r2'], 4),
            'mse': round(res['mse'], 4),
            'verdict': target['verdict'],
            'verdict_color': target['verdict_color'],
            'is_best_row': target.get('is_best_row', False),
            'r2_style': target.get('r2_style', '')
        }
        
        if found_idx != -1:
            leaderboard[found_idx] = entry
            print(f'Updated leaderboard entry for {target["name"]}: R2={entry["r2"]}, MSE={entry["mse"]}')
        else:
            leaderboard.append(entry)
            print(f'Appended leaderboard entry for {target["name"]}: R2={entry["r2"]}, MSE={entry["mse"]}')
        updated_any = True
        
    if updated_any:
        with open(leaderboard_path, 'w', encoding='utf-8') as f:
            json.dump(leaderboard, f, indent=4)
        print('Successfully saved updated leaderboard.json')
        
        script_path = os.path.join(repo_root, 'modeling/scripts/update_dashboard_leaderboard.py')
        if os.path.exists(script_path):
            print(f'Running {script_path}...')
            res_proc = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
            print(res_proc.stdout)
            if res_proc.stderr:
                print('Error output from update script:', res_proc.stderr)
        else:
            print(f'Update script not found: {script_path}')


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 70)
    print("Experiment 22: Periodic Re-Encoding vs EnKF")
    print("=" * 70)

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model, ckpt = load_model(args.checkpoint, device)
    print(f"  z_dim={ckpt['z_dim']}, hidden={ckpt['hidden']}, n_layers={ckpt['n_layers']}")

    # Load data
    print(f"\nLoading data from {args.data}...")
    with h5py.File(args.data, "r") as f:
        x_all = f["x"][:]
        u_all = f["u"][:]

    n_trials = x_all.shape[0]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_trials)
    test_idx = perm[:args.n_test_trials]

    # Normalize
    train_idx = perm[args.n_test_trials:]
    x_train = x_all[train_idx]
    u_train = u_all[train_idx]
    x_mean = x_train.mean(axis=(0, 2), keepdims=True)
    x_std = x_train.std(axis=(0, 2), keepdims=True) + 1e-8
    u_mean = u_train.mean(axis=(0, 2), keepdims=True)
    u_std = u_train.std(axis=(0, 2), keepdims=True) + 1e-8

    x_test_n = (x_all[test_idx] - x_mean) / x_std
    u_test_n = (u_all[test_idx] - u_mean) / u_std

    print(f"  Test trials: {len(test_idx)}")

    # Sweep
    K_values = args.k_values
    D_values = args.delay_values
    pw = args.past_window
    fw = args.future_window

    results = []

    for D in D_values:
        print(f"\n{'='*50}")
        print(f"  Delay D = {D}ms")
        print(f"{'='*50}")

        for K in K_values:
            print(f"\n  K={K} (re-encode every {K}ms)...", end=" ", flush=True)

            r2, mse, avg_time, n_windows = evaluate_periodic_reencode(
                model, x_test_n, u_test_n, ckpt,
                K=K, delay=D, pw=pw, fw=fw, device=device,
            )

            print(f"R²={r2:.4f}, MSE={mse:.6f}, Time={avg_time:.2f}ms/window ({n_windows} windows)")

            results.append({
                "K": K, "delay": D, "r2": float(r2), "mse": float(mse),
                "time_ms": float(avg_time), "n_windows": n_windows,
            })

    # Save results
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Periodic Re-Encoding")
    print("=" * 70)
    print(f"{'K':>6s}  {'Delay':>5s}  {'R²':>8s}  {'MSE':>10s}  {'Time(ms)':>10s}")
    print("-" * 50)
    for r in results:
        print(f"{r['K']:>6d}  {r['delay']:>5d}  {r['r2']:>8.4f}  {r['mse']:>10.6f}  {r['time_ms']:>10.2f}")

    # Compare with EnKF results if available
    print("\n" + "=" * 70)
    print("COMPARISON: Re-Encode vs EnKF (from Exp 21)")
    print("=" * 70)
    enkf_results = {
        (1, 0): 0.9997, (10, 0): 0.9756, (20, 0): 0.9607,
        (50, 0): 0.9383, (100, 0): 0.9263, (999, 0): 0.9262,
        (1, 10): 0.9593, (10, 10): 0.9472, (20, 10): 0.9391,
        (50, 10): 0.9275, (100, 10): 0.9210, (999, 10): 0.9149,
    }
    print(f"{'K':>6s}  {'D':>3s}  {'ReEncode':>10s}  {'EnKF':>10s}  {'Δ':>10s}")
    print("-" * 50)
    for r in results:
        k, d = r["K"], r["delay"]
        enkf_key = (min(k, 999), d)
        if enkf_key in enkf_results:
            enkf_r2 = enkf_results[enkf_key]
            delta = r["r2"] - enkf_r2
            print(f"{k:>6d}  {d:>3d}  {r['r2']:>10.4f}  {enkf_r2:>10.4f}  {delta:>+10.4f}")
        else:
            print(f"{k:>6d}  {d:>3d}  {r['r2']:>10.4f}  {'--':>10s}  {'--':>10s}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, D in zip(axes, D_values):
        d_results = [r for r in results if r["delay"] == D]
        ks = [r["K"] for r in d_results]
        r2s = [r["r2"] for r in d_results]
        ax.plot(ks, r2s, "o-", color="#22d3ee", linewidth=2, markersize=8, label="Re-encode")

        # EnKF reference
        enkf_ks, enkf_r2s = [], []
        for k in [1, 10, 20, 50, 100, 999]:
            if (k, D) in enkf_results:
                enkf_ks.append(k)
                enkf_r2s.append(enkf_results[(k, D)])
        ax.plot(enkf_ks, enkf_r2s, "s--", color="#f472b6", linewidth=2, markersize=8, label="EnKF (Exp 21)")

        ax.set_xlabel("K (re-encode period, ms)")
        ax.set_ylabel("R²")
        ax.set_title(f"Delay D={D}ms")
        ax.legend()
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.85, 1.01)

    plt.suptitle("Periodic Re-Encoding vs EnKF — Aligned Model", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "reencode_vs_enkf.png"), dpi=150)
    plt.close()
    print(f"\nPlot saved to {args.output_dir}/reencode_vs_enkf.png")
    update_leaderboard(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="/snel/home/cbwash2/cleo-worktrees/aligned-distill/results/aligned_cosine_0.9_0.1/best_model.pt")
    parser.add_argument("--data", type=str,
                        default="/snel/home/cbwash2/cleo/data/training_trials.h5")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-test-trials", type=int, default=10)
    parser.add_argument("--past-window", type=int, default=200)
    parser.add_argument("--future-window", type=int, default=200)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 5, 10, 20, 50, 100, 200])
    parser.add_argument("--delay-values", type=int, nargs="+", default=[0, 10])

    args = parser.parse_args()
    main(args)
