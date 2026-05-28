#!/usr/bin/env python
"""Fit N4SID linear model to generated Cleo data.

Uses trial-level train/test split. Evaluates on short prediction
horizons (200ms windows) matching MPC usage.

Usage:
    python -m modeling.scripts.fit_n4sid [--data data/training_trials.h5]
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def windowed_prediction(model, x_test, u_test, window_size=200):
    """Evaluate model on short prediction windows.

    For each window: start from true x(t0), predict forward window_size
    steps, compute MSE. This matches MPC usage where the controller
    resets to the measured state each cycle.

    Parameters
    ----------
    model : N4SIDModel
    x_test : ndarray, shape (n_ch, T)
    u_test : ndarray, shape (n_u, T)
    window_size : int
        Prediction horizon in steps

    Returns
    -------
    dict with 'mse', 'r2', 'per_window_mse', 'x_pred_windows'
    """
    n_ch, T = x_test.shape
    n_windows = (T - window_size) // window_size
    window_mses = []
    pred_windows = []
    true_windows = []

    for w in range(n_windows):
        t0 = w * window_size
        t1 = t0 + window_size
        x0 = x_test[:, t0]
        u_win = u_test[:, t0:t1]
        x_true_win = x_test[:, t0:t1]

        x_pred_win = model.predict(x0, u_win)
        pred_windows.append(x_pred_win)
        true_windows.append(x_true_win)
        window_mses.append(np.mean((x_pred_win - x_true_win) ** 2))

    # Also compute 1-step prediction for comparison
    # A operates in latent space (n_states), x_test is in observation space (n_ch)
    # Need: y -> latent via C^+, step in latent, map back via C
    C_pinv = np.linalg.pinv(model.C)
    one_step_errors = []
    z = C_pinv @ x_test[:, 0]  # initial latent state
    for t in range(T - 1):
        y_pred = model.C @ z + model.D @ u_test[:, t]
        one_step_errors.append(np.mean((y_pred - x_test[:, t]) ** 2))
        # Reset latent to true observation each step (1-step prediction)
        z = model.A @ (C_pinv @ x_test[:, t]) + model.B @ u_test[:, t]

    overall_mse = np.mean(window_mses)
    overall_var = np.var(x_test)
    r2 = 1 - overall_mse / overall_var

    return {
        "mse": overall_mse,
        "r2": r2,
        "per_window_mse": np.array(window_mses),
        "x_pred_windows": pred_windows,
        "x_true_windows": true_windows,
        "one_step_mse": np.mean(one_step_errors),
        "one_step_r2": 1 - np.mean(one_step_errors) / overall_var,
    }


def main():
    parser = argparse.ArgumentParser(description="Fit N4SID model")
    parser.add_argument("--data", type=str, default="data/training_trials.h5")
    parser.add_argument("--n-states", type=int, default=20,
                        help="Latent state dimension")
    parser.add_argument("--n-block-rows", type=int, default=30,
                        help="Block Hankel matrix rows")
    parser.add_argument("--n-test-trials", type=int, default=2,
                        help="Number of trials held out for testing")
    parser.add_argument("--eval-horizon", type=int, default=200,
                        help="Prediction horizon for evaluation (steps = ms at 1kHz)")
    parser.add_argument("--output-dir", type=str, default="results/n4sid")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from modeling.data import load_trials_h5
    from modeling.models.n4sid import N4SIDModel

    # Load data
    print("Loading data...")
    data = load_trials_h5(args.data)
    x_all, u_all = data["x"], data["u"]
    dt = float(data["dt"])
    n_trials, n_channels, n_steps = x_all.shape
    n_inputs = u_all.shape[1]
    print(f"  x: {x_all.shape}, u: {u_all.shape}, dt: {dt}s")

    # Trial-level split
    n_test = args.n_test_trials
    n_train = n_trials - n_test
    x_train_trials = x_all[:n_train]
    u_train_trials = u_all[:n_train]
    x_test_trials = x_all[n_train:]
    u_test_trials = u_all[n_train:]
    print(f"  Train: {n_train} trials, Test: {n_test} trials")

    # Concatenate train trials for fitting
    x_train = np.concatenate([x_train_trials[i] for i in range(n_train)], axis=1)
    u_train = np.concatenate([u_train_trials[i] for i in range(n_train)], axis=1)
    print(f"  Train concat: x={x_train.shape}, u={u_train.shape}")

    # Fit
    print(f"\nFitting N4SID: n_states={args.n_states}, n_block_rows={args.n_block_rows}")
    model = N4SIDModel(n_states=args.n_states, n_block_rows=args.n_block_rows)
    model.fit(x_train, u_train)

    eigs = model.eigenvalues()
    print(f"  System stable: {model.is_stable()}")
    print(f"  Max |eigenvalue|: {np.max(np.abs(eigs)):.4f}")

    # Windowed evaluation on each test trial
    horizon = args.eval_horizon
    print(f"\nEvaluating with {horizon}-step ({horizon * dt * 1000:.0f}ms) prediction windows...")
    all_results = []
    for ti in range(n_test):
        res = windowed_prediction(
            model, x_test_trials[ti], u_test_trials[ti], window_size=horizon
        )
        all_results.append(res)
        print(f"  Trial {n_train + ti}:")
        print(f"    {horizon}-step window: MSE={res['mse']:.2f}, R2={res['r2']:.4f} ({len(res['per_window_mse'])} windows)")
        print(f"    1-step:          MSE={res['one_step_mse']:.2f}, R2={res['one_step_r2']:.4f}")

    avg_mse = np.mean([r["mse"] for r in all_results])
    avg_r2 = np.mean([r["r2"] for r in all_results])
    avg_1step_mse = np.mean([r["one_step_mse"] for r in all_results])
    avg_1step_r2 = np.mean([r["one_step_r2"] for r in all_results])
    print(f"\n  Avg {horizon}-step: MSE={avg_mse:.2f}, R2={avg_r2:.4f}")
    print(f"  Avg 1-step:    MSE={avg_1step_mse:.2f}, R2={avg_1step_r2:.4f}")

    # Save model
    np.savez(
        os.path.join(args.output_dir, "n4sid_model.npz"),
        A=model.A, B=model.B, C=model.C, D=model.D,
        eigenvalues=eigs,
        avg_mse=avg_mse, avg_r2=avg_r2,
        avg_1step_mse=avg_1step_mse, avg_1step_r2=avg_1step_r2,
        eval_horizon=horizon,
    )

    # --- Plot 1: Windowed predictions (first test trial, first 5 windows) ---
    res = all_results[0]
    n_plot_ch = min(5, n_channels)
    n_plot_windows = min(5, len(res["x_pred_windows"]))

    fig, axes = plt.subplots(n_plot_ch, 1, figsize=(14, 2.5 * n_plot_ch), sharex=True)
    for i in range(n_plot_ch):
        ax = axes[i] if n_plot_ch > 1 else axes
        for w in range(n_plot_windows):
            t_offset = w * horizon * dt
            t_win = np.arange(horizon) * dt + t_offset
            ax.plot(t_win, res["x_true_windows"][w][i], "k", alpha=0.6, lw=0.8)
            ax.plot(t_win, res["x_pred_windows"][w][i], "r", alpha=0.6, lw=0.8)
            if w < n_plot_windows - 1:
                ax.axvline(t_win[-1], color="gray", alpha=0.3, ls="--", lw=0.5)
        ax.set_ylabel(f"Ch {i}")
        if i == 0:
            ax.plot([], [], "k", label="Ground truth")
            ax.plot([], [], "r", label="N4SID")
            ax.legend(fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"N4SID {horizon}-step Prediction (R\u00b2 = {res['r2']:.3f})")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "n4sid_prediction.png"), dpi=150)

    # --- Plot 2: Per-window MSE distribution ---
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 4))
    ax2a.hist(res["per_window_mse"], bins=30, edgecolor="black", alpha=0.7)
    ax2a.set_xlabel(f"{horizon}-step window MSE")
    ax2a.set_ylabel("Count")
    ax2a.set_title("Per-window MSE distribution")
    ax2a.axvline(np.median(res["per_window_mse"]), color="red", ls="--",
                 label=f"Median: {np.median(res['per_window_mse']):.1f}")
    ax2a.legend()

    # Eigenvalue plot
    theta = np.linspace(0, 2 * np.pi, 100)
    ax2b.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3)
    ax2b.scatter(eigs.real, eigs.imag, c="blue", s=40)
    ax2b.set_xlabel("Re")
    ax2b.set_ylabel("Im")
    ax2b.set_title("N4SID Eigenvalues")
    ax2b.set_aspect("equal")
    ax2b.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.output_dir, "n4sid_diagnostics.png"), dpi=150)

    print(f"\nPlots saved to {args.output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
