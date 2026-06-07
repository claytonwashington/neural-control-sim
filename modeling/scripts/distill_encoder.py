#!/usr/bin/env python
"""Encoder Distillation: Train a causal GRU encoder to match acausal z₀ targets.

Strategy:
1. Load the best acausal LatentNeuralODE checkpoint (bidirectional encoder).
2. Freeze the ODE dynamics + decoder.
3. Run the acausal encoder on all training windows to extract z₀_teacher targets.
4. Train a new causal (unidirectional) GRU encoder with loss = MSE(z₀_student, z₀_teacher).
5. Evaluate: plug the causal encoder into the full model (frozen ODE + decoder)
   and measure R², MSE, and inference time on test trials.
"""

from __future__ import annotations

import argparse
import os
import time

# Limit CPU threads to avoid contention
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
torch.set_num_threads(4)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchdiffeq import odeint

from modeling.data import load_trials_h5
from modeling.models.latent_node import LatentNeuralODE, BatchLinearInterpolation
from modeling.config import DEFAULT_SEED, DEFAULT_TEST_TRIALS, set_seed


# ============================================================================
# Causal Encoder (Unidirectional GRU)
# ============================================================================

class CausalEncoderGRU(nn.Module):
    """Causal encoder: unidirectional GRU that only reads past observations.

    Unlike the bidirectional EncoderGRU in the acausal model, this only
    processes the sequence left-to-right, making it suitable for real-time
    causal inference.
    """

    def __init__(self, n_x: int, n_u: int, hidden_dim: int = 128, z_dim: int = 32,
                 n_gru_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(
            n_x + n_u, hidden_dim,
            num_layers=n_gru_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if n_gru_layers > 1 else 0.0,
        )
        # Output: just mu (deterministic z0 target from teacher)
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, T, n_x) — past observations
        u : (B, T, n_u) — past inputs

        Returns
        -------
        mu : (B, z_dim)
        logvar : (B, z_dim)
        """
        seq = torch.cat([x, u], dim=-1)  # (B, T, n_x + n_u)
        _, h_n = self.gru(seq)
        # h_n: (n_layers, B, hidden_dim) — take last layer
        h = h_n[-1]  # (B, hidden_dim)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


# ============================================================================
# Distillation Hybrid Model (causal encoder + frozen acausal ODE/decoder)
# ============================================================================

class DistilledLatentNeuralODE(nn.Module):
    """Hybrid model: causal encoder + frozen ODE dynamics + frozen decoder.

    The ODE and decoder weights come from the pretrained acausal model.
    Only the causal encoder is trainable.
    """

    def __init__(self, causal_encoder: CausalEncoderGRU,
                 ode_func, decoder: nn.Linear, z_dim: int):
        super().__init__()
        self.encoder = causal_encoder  # trainable
        self.ode_func = ode_func       # frozen
        self.decoder = decoder         # frozen
        self.z_dim = z_dim

        # Freeze ODE + decoder
        for p in self.ode_func.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False

    def forward(self, x_past: torch.Tensor, u_past: torch.Tensor,
                u_future: torch.Tensor, t_future: torch.Tensor,
                method: str = "dopri5") -> torch.Tensor:
        """Causal forward pass: encode past → ODE integrate future → decode.

        Parameters
        ----------
        x_past : (B, T_past, n_x)
        u_past : (B, T_past, n_u)
        u_future : (B, T_future, n_u)
        t_future : (T_future,)
        method : ODE solver

        Returns
        -------
        x_pred_future : (B, T_future, n_x)
        """
        mu, _ = self.encoder(x_past, u_past)
        z0 = mu  # deterministic

        self.ode_func.set_input(t_future, u_future)
        z_seq = odeint(self.ode_func, z0, t_future,
                       method=method, rtol=1e-4, atol=1e-5)  # (T, B, z_dim)
        z_seq = z_seq.permute(1, 0, 2)  # (B, T, z_dim)

        x_pred = self.decoder(z_seq)  # (B, T, n_x)
        return x_pred


# ============================================================================
# Extract z₀ targets from acausal teacher
# ============================================================================

def extract_z0_targets(teacher: LatentNeuralODE, x_norm: np.ndarray,
                       u_norm: np.ndarray, dt: float,
                       window_size: int, past_window: int,
                       stride: int, device: str,
                       batch_size: int = 256) -> dict:
    """Run acausal encoder on overlapping windows to get z₀ teacher targets.

    For each window of `window_size` steps, we feed the FULL window (past+future)
    to the acausal bidirectional encoder to get z₀. The causal encoder will
    only see the first `past_window` steps.

    Returns dict with:
        x_past: (N, past_window, n_x) — inputs for causal encoder
        u_past: (N, past_window, n_u)
        z0_teacher: (N, z_dim) — targets
        u_future: (N, future_window, n_u) — for eval
        x_future: (N, future_window, n_x) — ground truth for eval
    """
    teacher.eval()
    teacher = teacher.to(device)

    n_ch, T_total = x_norm.shape
    n_u = u_norm.shape[0]
    future_window = window_size - past_window

    # Generate window start indices
    starts = list(range(0, T_total - window_size + 1, stride))
    N = len(starts)
    print(f"  Extracting z0 targets: {N} windows (window={window_size}, "
          f"past={past_window}, future={future_window}, stride={stride})")

    # Pre-build all windows
    x_windows = np.stack([x_norm[:, s:s + window_size].T for s in starts])  # (N, window_size, n_ch)
    u_windows = np.stack([u_norm[:, s:s + window_size].T for s in starts])  # (N, window_size, n_u)

    x_windows_t = torch.tensor(x_windows, dtype=torch.float32)
    u_windows_t = torch.tensor(u_windows, dtype=torch.float32)
    t_vec = torch.arange(window_size, dtype=torch.float32) * dt

    # Extract z0 in batches
    z0_all = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            x_b = x_windows_t[i:i+batch_size].to(device)
            u_b = u_windows_t[i:i+batch_size].to(device)
            mu, _ = teacher.encoder(x_b, u_b)  # (B, z_dim)
            z0_all.append(mu.cpu())

    z0_teacher = torch.cat(z0_all, dim=0)  # (N, z_dim)

    # Split into past/future
    x_past = x_windows_t[:, :past_window]        # (N, past_window, n_ch)
    u_past = u_windows_t[:, :past_window]         # (N, past_window, n_u)
    x_future = x_windows_t[:, past_window:]       # (N, future_window, n_ch)
    u_future = u_windows_t[:, past_window:]       # (N, future_window, n_u)

    print(f"  z0 targets shape: {z0_teacher.shape}, range: "
          f"[{z0_teacher.min():.3f}, {z0_teacher.max():.3f}]")

    return {
        "x_past": x_past,
        "u_past": u_past,
        "z0_teacher": z0_teacher,
        "x_future": x_future,
        "u_future": u_future,
        "t_future": torch.arange(future_window, dtype=torch.float32) * dt,
    }


# ============================================================================
# Training
# ============================================================================

def train_distill_encoder(
    causal_encoder: CausalEncoderGRU,
    train_data: dict,
    n_epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 128,
    val_fraction: float = 0.1,
    device: str = "cuda",
    seed: int = DEFAULT_SEED,
) -> dict:
    """Train causal encoder to match z₀ teacher targets via MSE.

    Parameters
    ----------
    causal_encoder : CausalEncoderGRU
    train_data : dict from extract_z0_targets
    """
    causal_encoder = causal_encoder.to(device)

    x_past = train_data["x_past"]
    u_past = train_data["u_past"]
    z0_teacher = train_data["z0_teacher"]

    N = x_past.shape[0]
    dataset = TensorDataset(x_past, u_past, z0_teacher)

    n_val = max(1, int(N * val_fraction))
    n_train = N - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            pin_memory=True)

    optimizer = torch.optim.AdamW(causal_encoder.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = {"train_loss": [], "val_loss": []}

    print(f"  Encoder distillation: {n_train} train, {n_val} val samples")
    print(f"  Batch size: {batch_size}, batches/epoch: {len(train_loader)}")

    best_val = float("inf")
    best_state = None

    for epoch in range(n_epochs):
        # --- Training ---
        causal_encoder.train()
        train_losses = []

        for x_b, u_b, z0_b in train_loader:
            x_b = x_b.to(device)
            u_b = u_b.to(device)
            z0_b = z0_b.to(device)

            mu, logvar = causal_encoder(x_b, u_b)
            loss = F.mse_loss(mu, z0_b)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(causal_encoder.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        history["train_loss"].append(avg_train)

        # --- Validation ---
        causal_encoder.eval()
        val_losses = []
        with torch.no_grad():
            for x_b, u_b, z0_b in val_loader:
                x_b = x_b.to(device)
                u_b = u_b.to(device)
                z0_b = z0_b.to(device)

                mu, _ = causal_encoder(x_b, u_b)
                val_losses.append(F.mse_loss(mu, z0_b).item())

        avg_val = np.mean(val_losses) if val_losses else float("nan")
        history["val_loss"].append(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.clone() for k, v in causal_encoder.state_dict().items()}

        scheduler.step()

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch:4d}/{n_epochs} | "
                  f"train: {avg_train:.6f} | val: {avg_val:.6f} | "
                  f"best_val: {best_val:.6f} | lr: {optimizer.param_groups[0]['lr']:.2e}")

    # Restore best
    if best_state is not None:
        causal_encoder.load_state_dict(best_state)
        print(f"  Restored best encoder (val_loss={best_val:.6f})")

    return history


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Encoder Distillation")
    parser.add_argument("--data", type=str, default="data/training_trials.h5")
    parser.add_argument("--teacher-ckpt", type=str,
                        default="results/sweep_latent_node_40_10/run_05_z64_h256_l2_lr0.0005_dopri5.pt")
    parser.add_argument("--past-window", type=int, default=200)
    parser.add_argument("--future-window", type=int, default=200)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-gru-layers", type=int, default=2,
                        help="Number of GRU layers in causal encoder")
    parser.add_argument("--encoder-hidden", type=int, default=256,
                        help="Hidden dim for causal encoder GRU")
    parser.add_argument("--n-test-trials", type=int, default=DEFAULT_TEST_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--method", type=str, default="dopri5")
    parser.add_argument("--output-dir", type=str, default="results/distill_encoder")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # ----------------------------------------------------------------
    # 1. Load data
    # ----------------------------------------------------------------
    print("\nLoading data...")
    data = load_trials_h5(args.data)
    x_all, u_all = data["x"], data["u"]
    dt = float(data["dt"])
    n_trials, n_channels, n_steps = x_all.shape
    n_inputs = u_all.shape[1]
    print(f"  {n_trials} trials, {n_channels} channels, {n_steps} steps, dt={dt}")

    # Train/test split
    n_test = args.n_test_trials
    n_train = n_trials - n_test
    x_train_trials = x_all[:n_train]
    u_train_trials = u_all[:n_train]
    x_test_trials = x_all[n_train:]
    u_test_trials = u_all[n_train:]

    # Normalization stats from training data
    x_train_cat = np.concatenate([x_train_trials[i] for i in range(n_train)], axis=1)
    u_train_cat = np.concatenate([u_train_trials[i] for i in range(n_train)], axis=1)
    x_mean = x_train_cat.mean(axis=1, keepdims=True)
    x_std = x_train_cat.std(axis=1, keepdims=True) + 1e-8
    u_mean = u_train_cat.mean(axis=1, keepdims=True)
    u_std = u_train_cat.std(axis=1, keepdims=True) + 1e-8

    # ----------------------------------------------------------------
    # 2. Load teacher (acausal) model
    # ----------------------------------------------------------------
    print(f"\nLoading teacher checkpoint: {args.teacher_ckpt}")
    ckpt = torch.load(args.teacher_ckpt, map_location="cpu")

    teacher_z_dim = ckpt["z_dim"]
    teacher_hidden = ckpt["hidden"]
    teacher_n_layers = ckpt["n_layers"]
    teacher_n_x = ckpt["n_x"]
    teacher_n_u = ckpt["n_u"]

    print(f"  Teacher: n_x={teacher_n_x}, n_u={teacher_n_u}, z_dim={teacher_z_dim}, "
          f"hidden={teacher_hidden}, n_layers={teacher_n_layers}")

    # Use teacher's normalization stats if available (for consistency)
    if "x_mean" in ckpt:
        x_mean = ckpt["x_mean"]
        x_std = ckpt["x_std"]
        u_mean = ckpt["u_mean"]
        u_std = ckpt["u_std"]
        print("  Using normalization stats from teacher checkpoint")

    teacher = LatentNeuralODE(
        n_x=teacher_n_x,
        n_u=teacher_n_u,
        z_dim=teacher_z_dim,
        hidden_dim=teacher_hidden,
        n_layers=teacher_n_layers,
    )
    teacher.load_state_dict(ckpt["model_state"])
    teacher.eval()
    print(f"  Teacher loaded ({sum(p.numel() for p in teacher.parameters()):,} params)")

    # ----------------------------------------------------------------
    # 3. Extract z₀ teacher targets from training data
    # ----------------------------------------------------------------
    print("\nExtracting z₀ teacher targets...")
    # Normalize and concatenate training data
    x_train_n = (x_train_cat - x_mean) / x_std
    u_train_n = (u_train_cat - u_mean) / u_std

    window_size = args.past_window + args.future_window
    train_targets = extract_z0_targets(
        teacher, x_train_n, u_train_n, dt,
        window_size=window_size,
        past_window=args.past_window,
        stride=args.stride,
        device=device,
        batch_size=256,
    )

    # ----------------------------------------------------------------
    # 4. Build and train causal encoder
    # ----------------------------------------------------------------
    print(f"\nBuilding causal encoder: hidden={args.encoder_hidden}, "
          f"z_dim={teacher_z_dim}, n_gru_layers={args.n_gru_layers}")
    causal_encoder = CausalEncoderGRU(
        n_x=teacher_n_x,
        n_u=teacher_n_u,
        hidden_dim=args.encoder_hidden,
        z_dim=teacher_z_dim,
        n_gru_layers=args.n_gru_layers,
    )
    n_enc_params = sum(p.numel() for p in causal_encoder.parameters())
    print(f"  Causal encoder parameters: {n_enc_params:,}")

    print("\nTraining causal encoder...")
    t_train_start = time.time()
    history = train_distill_encoder(
        causal_encoder, train_targets,
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
    )
    train_time = time.time() - t_train_start
    print(f"  Training time: {train_time:.1f}s")

    # ----------------------------------------------------------------
    # 5. Build distilled model and evaluate
    # ----------------------------------------------------------------
    print("\nBuilding distilled model (causal encoder + frozen ODE + decoder)...")
    distilled = DistilledLatentNeuralODE(
        causal_encoder=causal_encoder,
        ode_func=teacher.ode_func,
        decoder=teacher.decoder,
        z_dim=teacher_z_dim,
    )
    distilled = distilled.to(device)
    distilled.eval()

    n_total_params = sum(p.numel() for p in distilled.parameters())
    n_trainable = sum(p.numel() for p in distilled.parameters() if p.requires_grad)
    print(f"  Total params: {n_total_params:,}, trainable: {n_trainable:,}")

    # Evaluate on test trials with causal windowed prediction
    print(f"\nEvaluating on {n_test} test trials "
          f"(past={args.past_window}, future={args.future_window})...")

    eval_stride = args.past_window  # non-overlapping evaluation windows
    windowed_r2s = []
    windowed_mses = []
    inference_times = []
    t_future = torch.arange(args.future_window, dtype=torch.float32).to(device) * dt

    for ti in range(n_test):
        x_test = x_test_trials[ti]
        u_test = u_test_trials[ti]
        x_test_n = (x_test - x_mean) / x_std
        u_test_n = (u_test - u_mean) / u_std

        n_ch, T_val = x_test_n.shape
        n_windows = (T_val - args.past_window - args.future_window) // eval_stride + 1
        window_mses = []

        for w in range(n_windows):
            t0 = w * eval_stride
            t_split = t0 + args.past_window
            t1 = t_split + args.future_window

            x_past_win = torch.tensor(
                x_test_n[:, t0:t_split].T, dtype=torch.float32
            ).unsqueeze(0).to(device)
            u_past_win = torch.tensor(
                u_test_n[:, t0:t_split].T, dtype=torch.float32
            ).unsqueeze(0).to(device)
            u_future_win = torch.tensor(
                u_test_n[:, t_split:t1].T, dtype=torch.float32
            ).unsqueeze(0).to(device)
            x_true_future = x_test_n[:, t_split:t1]

            t_start = time.perf_counter()
            with torch.no_grad():
                x_pred_future = distilled(
                    x_past_win, u_past_win, u_future_win, t_future,
                    method=args.method,
                )
                x_pred_np = x_pred_future[0].cpu().numpy().T  # (n_ch, T_future)
            t_end = time.perf_counter()
            inference_times.append((t_end - t_start) * 1000.0)

            window_mses.append(np.mean((x_pred_np - x_true_future) ** 2))

        trial_mse = np.mean(window_mses)
        trial_var = np.var(x_test_n)
        trial_r2 = 1 - trial_mse / trial_var
        windowed_mses.append(trial_mse)
        windowed_r2s.append(trial_r2)
        print(f"  Trial {n_train + ti}: MSE={trial_mse:.4f}, R²={trial_r2:.4f} "
              f"({len(window_mses)} windows)")

    avg_mse = np.mean(windowed_mses)
    avg_r2 = np.mean(windowed_r2s)
    avg_time = np.mean(inference_times)

    print(f"\n{'='*60}")
    print(f"ENCODER DISTILLATION RESULTS")
    print(f"{'='*60}")
    print(f"  Past window:     {args.past_window}")
    print(f"  Future window:   {args.future_window}")
    print(f"  Test R²:         {avg_r2:.4f}")
    print(f"  Test MSE:        {avg_mse:.4f}")
    print(f"  Inference time:  {avg_time:.2f} ms (batch=1)")
    print(f"  Training time:   {train_time:.1f}s")
    print(f"  Encoder params:  {n_enc_params:,}")
    print(f"{'='*60}")

    # ----------------------------------------------------------------
    # 6. Save
    # ----------------------------------------------------------------
    save_path = os.path.join(args.output_dir, "distilled_model.pt")
    torch.save({
        "causal_encoder_state": causal_encoder.state_dict(),
        "teacher_ckpt_path": args.teacher_ckpt,
        "n_x": teacher_n_x,
        "n_u": teacher_n_u,
        "z_dim": teacher_z_dim,
        "teacher_hidden": teacher_hidden,
        "teacher_n_layers": teacher_n_layers,
        "encoder_hidden": args.encoder_hidden,
        "n_gru_layers": args.n_gru_layers,
        "past_window": args.past_window,
        "future_window": args.future_window,
        "x_mean": x_mean,
        "x_std": x_std,
        "u_mean": u_mean,
        "u_std": u_std,
        "history": history,
        "test_r2": float(avg_r2),
        "test_mse": float(avg_mse),
        "test_inference_ms": float(avg_time),
    }, save_path)
    print(f"\nSaved distilled model to: {save_path}")

    # Plot training curve
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.semilogy(history["train_loss"], label="Train (z₀ MSE)")
    ax.semilogy(history["val_loss"], label="Val (z₀ MSE)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("z₀ MSE Loss")
    ax.set_title("Encoder Distillation Training Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "distill_training_curve.png"), dpi=150)
    print(f"Saved training curve to: {args.output_dir}/distill_training_curve.png")

    # Plot per-trial R² comparison
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 4))
    trial_ids = list(range(n_train, n_train + n_test))
    ax2.bar(trial_ids, windowed_r2s, alpha=0.7)
    ax2.axhline(y=avg_r2, color='r', linestyle='--', label=f"Avg R²={avg_r2:.4f}")
    ax2.set_xlabel("Trial")
    ax2.set_ylabel("R²")
    ax2.set_title("Distilled Model: Per-Trial R²")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.output_dir, "distill_per_trial_r2.png"), dpi=150)

    print("\nDone.")


if __name__ == "__main__":
    main()
