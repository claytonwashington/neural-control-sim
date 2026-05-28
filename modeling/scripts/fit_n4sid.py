#!/usr/bin/env python
"""Fit N4SID linear model to generated Cleo data.

Usage:
    python -m modeling.scripts.fit_n4sid [--data data/training_continuous.h5]
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Fit N4SID model")
    parser.add_argument("--data", type=str, default="data/training_continuous.h5")
    parser.add_argument("--n-states", type=int, default=20,
                        help="Latent state dimension")
    parser.add_argument("--n-block-rows", type=int, default=30,
                        help="Block Hankel matrix rows")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--output-dir", type=str, default="results/n4sid")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from modeling.data import load_continuous_h5
    from modeling.models.n4sid import N4SIDModel

    # Load data
    print("Loading data...")
    data = load_continuous_h5(args.data)
    x, u, dt = data["x"], data["u"], data["dt"]
    n_channels, n_total = x.shape
    print(f"  x: {x.shape}, u: {u.shape}, dt: {dt}s")

    # Train/test split
    n_test = int(n_total * args.test_fraction)
    n_train = n_total - n_test
    x_train, x_test = x[:, :n_train], x[:, n_train:]
    u_train, u_test = u[:, :n_train], u[:, n_train:]

    # Fit
    print(f"Fitting N4SID: n_states={args.n_states}, n_block_rows={args.n_block_rows}")
    model = N4SIDModel(n_states=args.n_states, n_block_rows=args.n_block_rows)
    model.fit(x_train, u_train)

    eigs = model.eigenvalues()
    print(f"  System stable: {model.is_stable()}")
    print(f"  Max |eigenvalue|: {np.max(np.abs(eigs)):.4f}")

    # Predict on test set
    print("Predicting on test set...")
    x_pred = model.predict(x_test[:, 0], u_test)

    # Compute MSE
    mse = np.mean((x_pred - x_test) ** 2)
    var = np.var(x_test)
    r2 = 1 - mse / var
    print(f"  Test MSE: {mse:.6f}")
    print(f"  Test R^2: {r2:.4f}")

    # Save model
    np.savez(
        os.path.join(args.output_dir, "n4sid_model.npz"),
        A=model.A, B=model.B, C=model.C, D=model.D,
        eigenvalues=eigs, mse=mse, r2=r2,
    )

    # Plot predictions vs ground truth (first 5 channels, first 2000 steps)
    n_plot_ch = min(5, n_channels)
    n_plot_t = min(2000, x_test.shape[1])
    t_plot = np.arange(n_plot_t) * dt

    fig, axes = plt.subplots(n_plot_ch, 1, figsize=(14, 2.5 * n_plot_ch), sharex=True)
    for i in range(n_plot_ch):
        ax = axes[i] if n_plot_ch > 1 else axes
        ax.plot(t_plot, x_test[i, :n_plot_t], "k", alpha=0.7, lw=0.8, label="Ground truth")
        ax.plot(t_plot, x_pred[i, :n_plot_t], "r", alpha=0.7, lw=0.8, label="N4SID")
        ax.set_ylabel(f"Ch {i}")
        if i == 0:
            ax.legend()
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"N4SID Prediction (R\u00b2 = {r2:.3f})")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "n4sid_prediction.png"), dpi=150)
    print(f"  Plot saved to {args.output_dir}/n4sid_prediction.png")

    # Plot eigenvalues
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
    theta = np.linspace(0, 2 * np.pi, 100)
    ax2.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3)
    ax2.scatter(eigs.real, eigs.imag, c="blue", s=40)
    ax2.set_xlabel("Re")
    ax2.set_ylabel("Im")
    ax2.set_title("N4SID Eigenvalues")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.output_dir, "n4sid_eigenvalues.png"), dpi=150)

    print("Done.")


if __name__ == "__main__":
    main()
