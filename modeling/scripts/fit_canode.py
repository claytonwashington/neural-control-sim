#!/usr/bin/env python
"""Train control-affine Neural ODE on generated Cleo data.

Usage:
    python -m modeling.scripts.fit_canode [--data data/training_continuous.h5]
"""

import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Train control-affine Neural ODE")
    parser.add_argument("--data", type=str, default="data/training_continuous.h5")
    parser.add_argument("--n-epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--window-size", type=int, default=200,
                        help="Time steps per training window")
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="results/canode")
    parser.add_argument("--save-model", type=str, default="results/canode/model.pt")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    from modeling.data import load_continuous_h5
    from modeling.models.canode import ControlAffineODE, train_canode

    # Load data
    print("Loading data...")
    data = load_continuous_h5(args.data)
    x, u, dt = data["x"], data["u"], data["dt"]
    n_channels, n_total = x.shape
    n_inputs = u.shape[0]
    print(f"  x: {x.shape}, u: {u.shape}, dt: {dt}s")

    # Train/test split (temporal)
    n_test = int(n_total * args.test_fraction)
    n_train = n_total - n_test
    x_train, x_test = x[:, :n_train], x[:, n_train:]
    u_train, u_test = u[:, :n_train], u[:, n_train:]
    print(f"  Train: {n_train} steps, Test: {n_test} steps")

    # Normalize data (zero mean, unit variance per channel)
    x_mean = x_train.mean(axis=1, keepdims=True)
    x_std = x_train.std(axis=1, keepdims=True) + 1e-8
    u_mean = u_train.mean(axis=1, keepdims=True)
    u_std = u_train.std(axis=1, keepdims=True) + 1e-8

    x_train_n = (x_train - x_mean) / x_std
    x_test_n = (x_test - x_mean) / x_std
    u_train_n = (u_train - u_mean) / u_std
    u_test_n = (u_test - u_mean) / u_std

    # Build model
    print(f"\nBuilding CA-NODE: n_x={n_channels}, n_u={n_inputs}, "
          f"hidden={args.hidden}, layers={args.n_layers}")
    model = ControlAffineODE(
        n_x=n_channels,
        n_u=n_inputs,
        hidden=args.hidden,
        n_layers=args.n_layers,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Train
    print(f"\nTraining for {args.n_epochs} epochs...")
    history = train_canode(
        model, x_train_n, u_train_n, dt,
        n_epochs=args.n_epochs,
        lr=args.lr,
        window_size=args.window_size,
        stride=args.stride,
        device=device,
    )

    # Save model
    torch.save({
        "model_state": model.state_dict(),
        "n_x": n_channels,
        "n_u": n_inputs,
        "hidden": args.hidden,
        "n_layers": args.n_layers,
        "x_mean": x_mean,
        "x_std": x_std,
        "u_mean": u_mean,
        "u_std": u_std,
        "history": history,
    }, args.save_model)
    print(f"Model saved to {args.save_model}")

    # Evaluate on test set (multi-step prediction)
    print("\nEvaluating on test set...")
    model.eval()
    model = model.to(device)

    n_eval = min(2000, x_test_n.shape[1])
    x_eval = torch.tensor(x_test_n[:, :n_eval].T, dtype=torch.float32).to(device)
    u_eval = torch.tensor(u_test_n[:, :n_eval].T, dtype=torch.float32).to(device)
    t_eval = torch.arange(n_eval, dtype=torch.float32).to(device) * dt

    with torch.no_grad():
        model.set_input(t_eval, u_eval)
        x_pred = model.integrate(x_eval[0], t_eval)  # (T, n_x)

    x_pred_np = x_pred.cpu().numpy()  # (T, n_x)
    x_true_np = x_eval.cpu().numpy()  # (T, n_x)

    mse = np.mean((x_pred_np - x_true_np) ** 2)
    var = np.var(x_true_np)
    r2 = 1 - mse / var
    print(f"  Test MSE (normalized): {mse:.6f}")
    print(f"  Test R^2: {r2:.4f}")

    # Plot training curves
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.semilogy(history["train_loss"], label="Train")
    ax.semilogy(history["val_loss"], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("CA-NODE Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "training_curve.png"), dpi=150)

    # Plot predictions vs ground truth
    n_plot_ch = min(5, n_channels)
    t_plot = np.arange(n_eval) * dt

    fig2, axes = plt.subplots(n_plot_ch, 1, figsize=(14, 2.5 * n_plot_ch), sharex=True)
    for i in range(n_plot_ch):
        ax = axes[i] if n_plot_ch > 1 else axes
        ax.plot(t_plot, x_true_np[:, i], "k", alpha=0.7, lw=0.8, label="Ground truth")
        ax.plot(t_plot, x_pred_np[:, i], "r", alpha=0.7, lw=0.8, label="CA-NODE")
        ax.set_ylabel(f"Ch {i}")
        if i == 0:
            ax.legend()
    axes[-1].set_xlabel("Time (s)")
    fig2.suptitle(f"CA-NODE Prediction (R\u00b2 = {r2:.3f})")
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.output_dir, "canode_prediction.png"), dpi=150)
    print(f"  Plots saved to {args.output_dir}/")

    print("Done.")


if __name__ == "__main__":
    main()
