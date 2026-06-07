#!/usr/bin/env python
"""Train discrete-time GRU model on generated Cleo data.

Uses trial-level train/test split matching CA-NODE.

Usage:
    python -m modeling.scripts.fit_gru [--data data/training_trials.h5]
"""

import argparse
import time
import os

# Limit CPU threads
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import numpy as np
import torch
from modeling.scripts.preflight_check import add_preflight_args, validate_preflight
torch.set_num_threads(4)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from modeling.config import DEFAULT_SEED, DEFAULT_TEST_TRIALS, set_seed


def main():
    parser = argparse.ArgumentParser(description="Train discrete-time sequence models")
    parser.add_argument("--model", type=str, default="gru", choices=["gru", "discrete-ca"],
                        help="Model architecture: 'gru' or 'discrete-ca' (Euler-discretized Neural ODE)")
    parser.add_argument("--data", type=str, default="data/training_trials.h5")
    parser.add_argument("--n-epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=1,
                        help="Number of layers (for GRU) or depth (for discrete-ca)")
    parser.add_argument("--window-size", type=int, default=200,
                        help="Time steps per training window")
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--n-test-trials", type=int, default=DEFAULT_TEST_TRIALS,
                        help="Number of trials held out for testing")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--compile", action="store_true",
                        help="Compile the model using torch.compile")
    parser.add_argument("--load-model", type=str, default="",
                        help="Path to load trained model weights instead of training")
    parser.add_argument("--output-dir", type=str, default="auto",
                        help="Output directory (defaults based on model type)")
    parser.add_argument("--save-model", type=str, default="auto",
                        help="Save model path (defaults based on model type)")
    from modeling.wandb_utils import add_wandb_args
    add_wandb_args(parser)
    add_preflight_args(parser)
    args = parser.parse_args()
    set_seed(args.seed)
    validate_preflight(args)

    # Assign dynamic defaults for output paths
    if args.output_dir == "auto":
        args.output_dir = f"results/{args.model}"
    if args.save_model == "auto":
        args.save_model = f"results/{args.model}/model.pt"

    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    from modeling.data import load_trials_h5
    from modeling.models.sequence import GRUModel, DiscreteControlAffineModel, train_sequence_model

    # Load data: x is (n_trials, n_channels, n_steps)
    print("Loading data...")
    data = load_trials_h5(args.data)
    x_all, u_all = data["x"], data["u"]
    dt = float(data["dt"])
    n_trials, n_channels, n_steps = x_all.shape
    n_inputs = u_all.shape[1]
    print(f"  x: {x_all.shape}, u: {u_all.shape}, dt: {dt}s")

    # Trial-level train/test split
    n_test = args.n_test_trials
    n_train = n_trials - n_test
    x_train_trials = x_all[:n_train]
    u_train_trials = u_all[:n_train]
    x_test_trials = x_all[n_train:]
    u_test_trials = u_all[n_train:]
    print(f"  Train: {n_train} trials, Test: {n_test} trials")

    # Concatenate train trials for windowed training
    x_train = np.concatenate([x_train_trials[i] for i in range(n_train)], axis=1)
    u_train = np.concatenate([u_train_trials[i] for i in range(n_train)], axis=1)
    print(f"  Train concat: x={x_train.shape}, u={u_train.shape}")

    # Normalize (per-channel, computed on training data only)
    x_mean = x_train.mean(axis=1, keepdims=True)
    x_std = x_train.std(axis=1, keepdims=True) + 1e-8
    u_mean = u_train.mean(axis=1, keepdims=True)
    u_std = u_train.std(axis=1, keepdims=True) + 1e-8

    x_train_n = (x_train - x_mean) / x_std
    u_train_n = (u_train - u_mean) / u_std

    # Build model
    print(f"\nBuilding {args.model.upper()} Model: n_x={n_channels}, n_u={n_inputs}, "
          f"hidden={args.hidden}, layers={args.n_layers}")
    if args.model == "gru":
        model = GRUModel(
            n_x=n_channels,
            n_u=n_inputs,
            hidden=args.hidden,
            num_layers=args.n_layers,
        )
    elif args.model == "discrete-ca":
        model = DiscreteControlAffineModel(
            n_x=n_channels,
            n_u=n_inputs,
            hidden=args.hidden,
            n_layers=args.n_layers,
        )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    from modeling.wandb_utils import wandb_init, wandb_log, wandb_log_image, wandb_summary, wandb_finish
    wandb_init(args, model=model, script_name="fit_gru")

    if args.compile:
        print("Compiling networks...")
        if args.model == "gru":
            model = torch.compile(model)
        elif args.model == "discrete-ca":
            model.f = torch.compile(model.f)
            model.g = torch.compile(model.g)

    if args.load_model:
        print(f"Loading model weights from {args.load_model}...")
        checkpoint = torch.load(args.load_model, map_location=device)
        state_dict = checkpoint["model_state"]
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            cleaned_key = k.replace("_orig_mod.", "")
            cleaned_state_dict[cleaned_key] = v
        model.load_state_dict(cleaned_state_dict)
        history = checkpoint.get("history", {"train_loss": [], "val_loss": []})
    else:
        # Train
        print(f"\nTraining for {args.n_epochs} epochs with batch size {args.batch_size}...")
        t_train_start = time.time()
        history = train_sequence_model(
            model, x_train_n, u_train_n, dt,
            n_epochs=args.n_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            window_size=args.window_size,
            stride=args.stride,
            device=device,
            seed=args.seed,
        )
        train_time_s = time.time() - t_train_start

        # Save model + normalization stats
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

        # Log per-epoch metrics to wandb
        for epoch_i, (tl, vl) in enumerate(zip(history["train_loss"], history["val_loss"])):
            wandb_log({"train/loss": tl, "val/loss": vl}, step=epoch_i)

    # Evaluate on each test trial
    print("\nEvaluating on test trials (2000-step open-loop)...")
    model.eval()
    model = model.to(device)

    test_results = []
    for ti in range(n_test):
        x_test = x_test_trials[ti]  # (n_ch, T)
        u_test = u_test_trials[ti]  # (n_u, T)

        # Normalize using training stats
        x_test_n = (x_test - x_mean) / x_std
        u_test_n = (u_test - u_mean) / u_std

        n_eval = min(2000, x_test_n.shape[1])
        x_eval = torch.tensor(x_test_n[:, :n_eval].T, dtype=torch.float32).to(device)
        u_eval = torch.tensor(u_test_n[:, :n_eval].T, dtype=torch.float32).to(device)
        t_eval = torch.arange(n_eval, dtype=torch.float32).to(device) * dt

        with torch.no_grad():
            x_pred = model.predict(
                x_eval[0].unsqueeze(0), t_eval, u_eval.unsqueeze(0)
            )  # (1, T, n_x)

        x_pred_np = x_pred[0].cpu().numpy().T  # (n_x, T) -> transpose to match
        x_true_np = x_eval.cpu().numpy().T  # (n_x, T)

        mse = np.mean((x_pred_np - x_true_np) ** 2)
        var = np.var(x_true_np)
        r2 = 1 - mse / var
        test_results.append((x_true_np, x_pred_np, r2, mse))
        print(f"  Trial {n_train + ti}: MSE={mse:.4f}, R2={r2:.4f}")

    # Windowed evaluation (200-step windows) matching N4SID
    horizon = 200
    print(f"\nEvaluating with {horizon}-step ({horizon * dt * 1000:.0f}ms) prediction windows...")
    windowed_r2s = []
    windowed_mses = []
    for ti in range(n_test):
        x_test = x_test_trials[ti]
        u_test = u_test_trials[ti]
        x_test_n = (x_test - x_mean) / x_std
        u_test_n = (u_test - u_mean) / u_std

        n_ch_val, T_val = x_test_n.shape
        n_windows = (T_val - horizon) // horizon
        window_mses = []
        t_win = torch.arange(horizon, dtype=torch.float32).to(device) * dt

        for w in range(n_windows):
            t0 = w * horizon
            t1 = t0 + horizon
            x0 = torch.tensor(x_test_n[:, t0], dtype=torch.float32).to(device)
            u_win = torch.tensor(u_test_n[:, t0:t1].T, dtype=torch.float32).to(device)
            x_true_win = x_test_n[:, t0:t1]

            with torch.no_grad():
                x_pred_win = model.predict(
                    x0.unsqueeze(0), t_win, u_win.unsqueeze(0)
                )  # (1, T, n_x)
                x_pred_win_np = x_pred_win[0].cpu().numpy().T  # (n_x, T)
            window_mses.append(np.mean((x_pred_win_np - x_true_win) ** 2))

        trial_mse = np.mean(window_mses)
        trial_var = np.var(x_test_n)
        trial_r2 = 1 - trial_mse / trial_var
        windowed_mses.append(trial_mse)
        windowed_r2s.append(trial_r2)
        print(f"  Trial {n_train + ti}: {horizon}-step window MSE={trial_mse:.4f}, R_win={trial_r2:.4f}")
    
    print(f"  Avg {horizon}-step: MSE={np.mean(windowed_mses):.4f}, R2={np.mean(windowed_r2s):.4f}")

    # Log summary metrics to wandb
    wandb_summary({
        "test/mean_r2_openloop": float(np.mean([r[2] for r in test_results])),
        "test/mean_mse_openloop": float(np.mean([r[3] for r in test_results])),
        "test/mean_r2_windowed": float(np.mean(windowed_r2s)),
        "test/mean_mse_windowed": float(np.mean(windowed_mses)),
        "train/best_train_loss": float(min(history["train_loss"])),
        "train/best_val_loss": float(min(history["val_loss"])),
        "train/time_seconds": train_time_s if 'train_time_s' in dir() else 0,
        "model/n_params": n_params,
    })

    # Plot training curves
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.semilogy(history["train_loss"], label="Train")
    ax.semilogy(history["val_loss"], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("GRU Model Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "training_curve.png"), dpi=150)

    wandb_log_image("plots/training_curve", os.path.join(args.output_dir, "training_curve.png"), caption="Training curve")

    # Plot predictions for first test trial
    x_true_np, x_pred_np, r2, mse = test_results[0]
    n_plot_ch = min(5, n_channels)
    t_plot = np.arange(x_true_np.shape[0]) * dt

    fig2, axes = plt.subplots(n_plot_ch, 1, figsize=(14, 2.5 * n_plot_ch), sharex=True)
    for i in range(n_plot_ch):
        ax = axes[i] if n_plot_ch > 1 else axes
        ax.plot(t_plot, x_true_np[:, i], "k", alpha=0.7, lw=0.8, label="Ground truth")
        ax.plot(t_plot, x_pred_np[:, i], "r", alpha=0.7, lw=0.8, label="GRU")
        ax.set_ylabel(f"Ch {i}")
        if i == 0:
            ax.legend()
    axes[-1].set_xlabel("Time (s)")
    fig2.suptitle(f"GRU Model Prediction (R\u00b2 = {r2:.3f})")
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.output_dir, "gru_prediction.png"), dpi=150)
    print(f"  Plots saved to {args.output_dir}/")

    wandb_log_image("plots/prediction", os.path.join(args.output_dir, "gru_prediction.png"), caption=f"GRU Prediction (R²={r2:.3f})")

    wandb_finish()
    print("Done.")


if __name__ == "__main__":
    main()
