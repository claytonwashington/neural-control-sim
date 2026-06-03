#!/usr/bin/env python
"""Train control-affine Neural ODE on generated Cleo data.

Uses trial-level train/test split: last 2 trials held out for testing.

Usage:
    python -m modeling.scripts.fit_canode [--data data/training_trials.h5]
"""

import argparse
import os

# Limit CPU threads to avoid contention and thrashing on multi-core systems
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import numpy as np
import torch
torch.set_num_threads(4)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from modeling.config import DEFAULT_SEED, DEFAULT_TEST_TRIALS, set_seed


def main():
    parser = argparse.ArgumentParser(description="Train control-affine Neural ODE")
    parser.add_argument("--data", type=str, default="data/training_trials.h5")
    parser.add_argument("--n-epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--window-size", type=int, default=200,
                        help="Time steps per training window")
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--n-test-trials", type=int, default=DEFAULT_TEST_TRIALS,
                        help="Number of trials held out for testing")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--method", type=str, default="dopri5",
                        help="ODE integration method (e.g. dopri5, rk4)")
    parser.add_argument("--compile", action="store_true",
                        help="Compile the drift and control networks using torch.compile")
    parser.add_argument("--load-model", type=str, default="",
                        help="Path to load trained model weights instead of training")
    parser.add_argument("--output-dir", type=str, default="results/canode")
    parser.add_argument("--save-model", type=str, default="results/canode/model.pt")
    parser.add_argument("--skip", action="store_true", default=True,
                        help="Use skip connections (default: True)")
    parser.add_argument("--no-skip", dest="skip", action="store_false",
                        help="Disable skip connections")
    parser.add_argument("--skip-type", type=str, default="mlp", choices=["mlp", "linear"],
                        help="Type of skip connection (default: mlp)")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="Weight decay for base model parameters (default: 1e-5)")
    parser.add_argument("--skip-weight-decay", type=float, default=None,
                        help="Separate weight decay for skip parameters (default: None)")
    parser.add_argument("--lr-warmup", type=int, default=0,
                        help="Number of epochs for linear learning rate warmup (default: 0)")
    parser.add_argument("--lr-decay", action="store_true", default=True,
                        help="Decay learning rate with cosine annealing (default: True)")
    parser.add_argument("--no-lr-decay", dest="lr_decay", action="store_false",
                        help="Disable cosine annealing learning rate decay")
    parser.add_argument("--spectral-alpha", type=float, default=0.0,
                        help="Coefficient for frequency-aware spectral loss (default: 0.0)")
    parser.add_argument("--model-type", type=str, default="standard", choices=["standard", "multi-rate"],
                        help="Model type to train (default: standard)")
    parser.add_argument("--sub-steps", type=int, default=10,
                        help="Number of sub-steps for multi-rate ODE (default: 10)")
    parser.add_argument("--multirate-init", type=str, default="zero_fast", choices=["zero_fast", "learnable"],
                        help="Initial state split for multi-rate ODE (default: zero_fast)")
    args = parser.parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    from modeling.data import load_trials_h5
    from modeling.models.canode import ControlAffineODE, train_canode

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

    # Build model configuration, taking values from checkpoint if loading
    model_type = args.model_type
    sub_steps = args.sub_steps
    multirate_init = args.multirate_init
    use_skip = args.skip
    skip_type = args.skip_type
    if args.load_model:
        print(f"Loading checkpoint metadata from {args.load_model}...")
        checkpoint_meta = torch.load(args.load_model, map_location=device)
        model_type = checkpoint_meta.get("model_type", model_type)
        sub_steps = checkpoint_meta.get("sub_steps", sub_steps)
        multirate_init = checkpoint_meta.get("multirate_init", multirate_init)
        use_skip = checkpoint_meta.get("use_skip", use_skip)
        skip_type = checkpoint_meta.get("skip_type", skip_type)
        print(f"  Checkpoint metadata: type={model_type}, sub_steps={sub_steps}, init={multirate_init}, skip={use_skip}")

    # Build model
    if model_type == "multi-rate":
        from modeling.models.canode import MultiRateControlAffineODE
        print(f"\nBuilding Multi-Rate CA-NODE: n_x={n_channels}, n_u={n_inputs}, "
              f"hidden={args.hidden}, layers={args.n_layers}, sub_steps={sub_steps}, init={multirate_init}")
        model = MultiRateControlAffineODE(
            n_x=n_channels,
            n_u=n_inputs,
            hidden=args.hidden,
            n_layers=args.n_layers,
            sub_steps=sub_steps,
            init_type=multirate_init,
        )
    else:
        print(f"\nBuilding CA-NODE: n_x={n_channels}, n_u={n_inputs}, "
              f"hidden={args.hidden}, layers={args.n_layers}, skip={use_skip}, skip_type={skip_type}")
        model = ControlAffineODE(
            n_x=n_channels,
            n_u=n_inputs,
            hidden=args.hidden,
            n_layers=args.n_layers,
            use_skip=use_skip,
            skip_type=skip_type,
        )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    if args.compile:
        print("Compiling networks...")
        if model_type == "multi-rate":
            model.f_slow = torch.compile(model.f_slow)
            model.g_slow = torch.compile(model.g_slow)
            model.f_fast = torch.compile(model.f_fast)
            model.g_fast_net = torch.compile(model.g_fast_net)
        else:
            model.f = torch.compile(model.f)
            model.g = torch.compile(model.g)
            if use_skip:
                model.skip_state = torch.compile(model.skip_state)
                model.skip_input = torch.compile(model.skip_input)

    if args.load_model:
        print(f"Loading model weights from {args.load_model}...")
        checkpoint = torch.load(args.load_model, map_location=device)
        # Handle state dict load. (If model was compiled, keys might contain '_orig_mod.')
        state_dict = checkpoint["model_state"]
        # Remove '_orig_mod.' prefix from compiled checkpoints if model is uncompiled
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            cleaned_key = k.replace("_orig_mod.", "")
            cleaned_state_dict[cleaned_key] = v
        model.load_state_dict(cleaned_state_dict)
        history = checkpoint.get("history", {"train_loss": [], "val_loss": []})
    else:
        # Train
        print(f"\nTraining for {args.n_epochs} epochs with batch size {args.batch_size}...")
        history = train_canode(
            model, x_train_n, u_train_n, dt,
            n_epochs=args.n_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            window_size=args.window_size,
            stride=args.stride,
            device=device,
            method=args.method,
            weight_decay=args.weight_decay,
            skip_weight_decay=args.skip_weight_decay,
            spectral_alpha=args.spectral_alpha,
            lr_warmup=args.lr_warmup,
            lr_decay=args.lr_decay,
            seed=args.seed,
        )

        # Save model + normalization stats
        torch.save({
            "model_state": model.state_dict(),
            "n_x": n_channels,
            "n_u": n_inputs,
            "hidden": args.hidden,
            "n_layers": args.n_layers,
            "model_type": model_type,
            "sub_steps": sub_steps,
            "multirate_init": multirate_init,
            "use_skip": use_skip,
            "skip_type": skip_type,
            "lr_warmup": args.lr_warmup,
            "lr_decay": args.lr_decay,
            "x_mean": x_mean,
            "x_std": x_std,
            "u_mean": u_mean,
            "u_std": u_std,
            "history": history,
        }, args.save_model)
        print(f"Model saved to {args.save_model}")

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
                x_eval[0].unsqueeze(0), t_eval, u_eval.unsqueeze(0), method=args.method
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
                    x0.unsqueeze(0), t_win, u_win.unsqueeze(0), method=args.method
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

    # Plot training curves
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.semilogy(history["train_loss"], label="Train (Total)")
    ax.semilogy(history["val_loss"], label="Validation (Total)")
    if "train_loss_mse" in history and len(history["train_loss_mse"]) > 0:
        ax.semilogy(history["train_loss_mse"], ":", alpha=0.7, label="Train (MSE)")
        ax.semilogy(history["val_loss_mse"], ":", alpha=0.7, label="Val (MSE)")
    if "train_loss_fft" in history and len(history["train_loss_fft"]) > 0:
        ax.semilogy(history["train_loss_fft"], "--", alpha=0.7, label="Train (FFT)")
        ax.semilogy(history["val_loss_fft"], "--", alpha=0.7, label="Val (FFT)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("CA-NODE Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "training_curve.png"), dpi=150)

    # Plot predictions for first test trial
    x_true_np, x_pred_np, r2, mse = test_results[0]
    n_plot_ch = min(5, n_channels)
    t_plot = np.arange(x_true_np.shape[0]) * dt

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
