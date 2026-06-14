#!/usr/bin/env python
"""Standardised post-experiment validation plots for digital-twin models.

For a given model checkpoint, run windowed inference on the held-out test trial
and emit, into the run directory:

  * ``val_channel_heatmap.png`` — true / predicted / |error| firing-rate heatmaps
    across all channels for the whole test trial (shared colour scale).
  * ``val_channel_traces.png``  — true-vs-predicted overlays for the top-variance
    channels (extends the existing prediction-trace style).
  * ``val_pc_projection.png``   — true & predicted trajectories projected onto the
    top principal components of the *true* data, plus a variance-explained strip.
  * ``val_metrics.json``        — overall R², per-channel R², per-PC R².

PCA uses a plain NumPy SVD (neither conda env ships scikit-learn).

Inference paths reuse the windowed loops from ``generate_dashboard_assets.py``:
  * ``canode``        -> ControlAffineODE   (reset to truth every 200 ms)
  * ``latent_canode`` -> LatentControlAffineODE (causal: 200 ms past -> 200 ms fut.)
  * ``latent_node``   -> LatentNeuralODE   (acausal encode 200 ms -> integrate)

``gru`` / ``n4sid`` are not yet wired (metadata-only pages); they are skipped.

Run::

    # one experiment by its results sub-dir
    python -m modeling.scripts.validation_plots --results-dir causal_z64_pw200_h128_lr3e4
    # everything in the registry that has a checkpoint
    python -m modeling.scripts.validation_plots --all
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from modeling.config import DEFAULT_SEED, DEFAULT_TEST_TRIALS, set_seed  # noqa: E402
from modeling.data import load_trials_h5  # noqa: E402
from modeling.scripts.dashboard_common import (  # noqa: E402
    COLORS,
    REPO,
    RESULTS,
    SHARED_REPO,
    SHARED_RESULTS,
    SUPPORTED_MODEL_TYPES,
    all_experiments,
    find_checkpoint,
    manifest_to_entry,
    resolve_data,
    slugify,
)

SUPPORTED = SUPPORTED_MODEL_TYPES  # single source of truth in dashboard_common
HORIZON = 200  # ms reset/window length, matches the dashboard convention


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------
def _col(a):
    """Coerce a normalisation stat to broadcast against (n_ch, T)."""
    a = np.asarray(a)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return a


def r2_score(true: np.ndarray, pred: np.ndarray) -> float:
    """R² over all finite entries (flattened)."""
    mask = np.isfinite(pred) & np.isfinite(true)
    if mask.sum() < 2:
        return float("nan")
    t, p = true[mask], pred[mask]
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def per_channel_r2(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """R² per channel; channels are rows of (n_ch, T)."""
    out = np.full(true.shape[0], np.nan)
    for c in range(true.shape[0]):
        out[c] = r2_score(true[c], pred[c])
    return out


# ---------------------------------------------------------------------------
# Model loading + inference
# ---------------------------------------------------------------------------
def _clean_state(sd: dict) -> dict:
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


def _safe_torch_load(ckpt_path, device):
    """Load a checkpoint, preferring the safe ``weights_only=True`` path (L1).

    Our checkpoints embed numpy normalization stats, which ``weights_only=True``
    rejects, so we fall back to a full (code-executing) load — but ONLY for
    checkpoints under the trusted in-repo ``results/`` tree, never an arbitrary
    external path. This removes the "load any .pt" footgun while still loading our
    own training outputs.
    """
    try:
        return torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception:
        rp = os.path.realpath(str(ckpt_path))
        trusted = [os.path.realpath(str(SHARED_RESULTS)) + os.sep,
                   os.path.realpath(str(RESULTS)) + os.sep]
        if not any(rp.startswith(t) for t in trusted):
            raise RuntimeError(
                f"Refusing to full-load an untrusted checkpoint outside results/: {ckpt_path}"
            )
        return torch.load(ckpt_path, map_location=device, weights_only=False)


def load_model(model_type: str, ckpt_path, device: str):
    """Instantiate the right model class and return (model, norms dict)."""
    ckpt = _safe_torch_load(ckpt_path, device)
    state = _clean_state(ckpt["model_state"])
    if model_type == "canode":
        from modeling.models.canode import ControlAffineODE

        # Infer skip-connection config from the checkpoint weights themselves —
        # older checkpoints don't store use_skip/skip_type/skip_hidden.
        use_skip = any(k.startswith("skip_") for k in state)
        # The MLP branch has skip_state.bias + a Sequential skip_input; the
        # linear branch is bias-free single Linears.
        skip_type = "mlp" if ("skip_state.bias" in state or "skip_input.0.weight" in state) else "linear"
        skip_hidden = state["skip_input.0.weight"].shape[0] if "skip_input.0.weight" in state else 64
        model = ControlAffineODE(
            n_x=ckpt["n_x"], n_u=ckpt["n_u"], hidden=ckpt["hidden"],
            n_layers=ckpt["n_layers"], use_skip=ckpt.get("use_skip", use_skip),
            skip_type=ckpt.get("skip_type", skip_type), skip_hidden=skip_hidden,
        )
    elif model_type == "latent_canode":
        from modeling.models.latent_canode import LatentControlAffineODE

        model = LatentControlAffineODE(
            n_x=ckpt["n_x"], n_u=ckpt["n_u"], z_dim=ckpt["z_dim"],
            hidden_dim=ckpt["hidden"], n_layers=ckpt["n_layers"],
        )
    elif model_type == "latent_node":
        from modeling.models.latent_node import LatentNeuralODE

        model = LatentNeuralODE(
            n_x=ckpt["n_x"], n_u=ckpt["n_u"], z_dim=ckpt["z_dim"],
            hidden_dim=ckpt["hidden"], n_layers=ckpt["n_layers"],
        )
    elif model_type == "gru":
        from modeling.models.sequence import GRUModel

        # NB: checkpoint stores `n_layers`, but the constructor arg is `num_layers`.
        model = GRUModel(
            n_x=ckpt["n_x"], n_u=ckpt["n_u"],
            hidden=ckpt["hidden"], num_layers=ckpt["n_layers"],
        )
    else:
        raise ValueError(f"Unsupported model_type for inference: {model_type}")

    model.load_state_dict(state)
    model = model.to(device).eval()
    norms = {
        "x_mean": _col(ckpt["x_mean"]), "x_std": _col(ckpt["x_std"]),
        "u_mean": _col(ckpt["u_mean"]), "u_std": _col(ckpt["u_std"]),
    }
    return model, norms


def infer(model_type, model, x_n, u_n, dt, device, past_window=200):
    """Return predicted normalised states (n_ch, T); NaN where unpredicted."""
    n_ch, T = x_n.shape
    pred = np.full((n_ch, T), np.nan, dtype=np.float32)

    if model_type == "canode":
        t_win = torch.arange(HORIZON, dtype=torch.float32, device=device) * dt
        for t0 in range(0, T - 1, HORIZON):
            t1 = min(t0 + HORIZON, T)
            tw = t_win[: t1 - t0]
            x0 = torch.tensor(x_n[:, t0], dtype=torch.float32, device=device)
            u_win = torch.tensor(u_n[:, t0:t1].T, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                model.set_input(tw, u_win)
                pred[:, t0:t1] = model.integrate(x0, tw).cpu().numpy().T

    elif model_type == "latent_node":
        t_win = torch.arange(HORIZON, dtype=torch.float32, device=device) * dt
        for t0 in range(0, T - 1, HORIZON):
            t1 = min(t0 + HORIZON, T)
            tw = t_win[: t1 - t0]
            x_win = torch.tensor(x_n[:, t0:t1].T, dtype=torch.float32, device=device).unsqueeze(0)
            u_win = torch.tensor(u_n[:, t0:t1].T, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                out = model.predict(x_win, u_win, tw, method="dopri5")
                pred[:, t0:t1] = out[0].cpu().numpy().T

    elif model_type == "gru":
        # Reset to truth at each window start, like canode; GRU.predict takes
        # x0 (B, n_x), t (T,), u (B, T, n_u) and returns (B, T, n_x).
        t_win = torch.arange(HORIZON, dtype=torch.float32, device=device) * dt
        for t0 in range(0, T - 1, HORIZON):
            t1 = min(t0 + HORIZON, T)
            tw = t_win[: t1 - t0]
            x0 = torch.tensor(x_n[:, t0], dtype=torch.float32, device=device).unsqueeze(0)
            u_win = torch.tensor(u_n[:, t0:t1].T, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                out = model.predict(x0, tw, u_win)
                pred[:, t0:t1] = out[0].cpu().numpy().T

    elif model_type == "latent_canode":
        fut = HORIZON
        t_fut = torch.arange(fut, dtype=torch.float32, device=device) * dt
        for t_split in range(past_window, T - 1, fut):
            t0 = t_split - past_window
            t1 = min(t_split + fut, T)
            tf = t_fut[: t1 - t_split]
            x_past = torch.tensor(x_n[:, t0:t_split].T, dtype=torch.float32, device=device).unsqueeze(0)
            u_past = torch.tensor(u_n[:, t0:t_split].T, dtype=torch.float32, device=device).unsqueeze(0)
            u_fut = torch.tensor(u_n[:, t_split:t1].T, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                out = model.predict(x_past, u_past, u_fut, tf, method="dopri5")
                pred[:, t_split:t1] = out[0].cpu().numpy().T
    else:
        raise ValueError(model_type)

    return pred


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _dark(fig):
    fig.patch.set_facecolor(COLORS["bg"])


def _style(ax):
    ax.set_facecolor(COLORS["panel"])
    ax.tick_params(colors="#6b7280", labelsize=8)
    for s in ax.spines.values():
        s.set_color(COLORS["border"])
    ax.xaxis.label.set_color(COLORS["muted"])
    ax.yaxis.label.set_color(COLORS["muted"])


def plot_channel_heatmap(x_true, x_pred, dt, out_path, title):
    n_ch, T = x_true.shape
    step = max(1, T // 1500)  # cap horizontal resolution
    xt, xp = x_true[:, ::step], x_pred[:, ::step]
    err = np.abs(xt - xp)
    extent = [0, T * dt, n_ch, 0]
    vmax = np.nanpercentile(x_true, 99)
    vmin = max(0.0, np.nanpercentile(x_true, 1))

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    _dark(fig)
    panels = [
        ("Ground truth", xt, "magma", vmin, vmax),
        ("Predicted", xp, "magma", vmin, vmax),
        ("|error|", err, "inferno", 0, np.nanpercentile(err, 99) or 1.0),
    ]
    for ax, (lab, data, cmap, lo, hi) in zip(axes, panels):
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad(COLORS["panel"])
        im = ax.imshow(data, aspect="auto", origin="upper", cmap=cm,
                       vmin=lo, vmax=hi, extent=extent, interpolation="nearest")
        _style(ax)
        ax.set_ylabel(f"{lab}\nchannel", fontsize=9)
        cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.025)
        cb.ax.tick_params(colors="#6b7280", labelsize=7)
        cb.outline.set_edgecolor(COLORS["border"])
    axes[-1].set_xlabel("Time (s)", fontsize=10)
    fig.suptitle(title, color=COLORS["text"], fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def plot_channel_traces(x_true, x_pred, dt, out_path, title, n_show=6):
    n_ch, T = x_true.shape
    top = np.argsort(np.nanvar(x_true, axis=1))[::-1][:n_show]
    t = np.arange(T) * dt
    fig, axes = plt.subplots(n_show, 1, figsize=(13, 1.5 * n_show), sharex=True)
    _dark(fig)
    for i, ch in enumerate(top):
        ax = axes[i]
        _style(ax)
        ax.plot(t, x_true[ch], color=COLORS["truth"], lw=1.0, alpha=0.9, label="Truth")
        valid = np.isfinite(x_pred[ch])
        ax.plot(t[valid], x_pred[ch, valid], color=COLORS["pred"], lw=1.2, alpha=0.9, label="Predicted")
        r2 = r2_score(x_true[ch], x_pred[ch])
        ax.set_ylabel(f"Ch {ch}\nR²={r2:.2f}", fontsize=8)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8, facecolor="#1a1f35",
                      edgecolor=COLORS["border"], labelcolor=COLORS["text"])
    axes[-1].set_xlabel("Time (s)", fontsize=10)
    fig.suptitle(title, color=COLORS["text"], fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def pca_project(x_true, x_pred, k=4):
    """PCA on true data (channels=features); project both true & pred.

    Returns scores_true (T,k), scores_pred (T,k), var_explained (k,).
    Pred timepoints that are NaN stay NaN in scores_pred.
    """
    Xt = x_true.T  # (T, n_ch)
    mu = Xt.mean(axis=0, keepdims=True)
    Xc = Xt - mu
    # Right singular vectors are the principal axes.
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:k]  # (k, n_ch)
    var_exp = (S[:k] ** 2) / np.sum(S**2)
    scores_true = Xc @ comps.T  # (T, k)

    Xp = x_pred.T
    valid = np.all(np.isfinite(Xp), axis=1)
    scores_pred = np.full((Xp.shape[0], k), np.nan)
    scores_pred[valid] = (Xp[valid] - mu) @ comps.T
    return scores_true, scores_pred, var_exp


def plot_pc_projection(scores_true, scores_pred, var_exp, dt, out_path, title):
    T, k = scores_true.shape
    t = np.arange(T) * dt
    fig, axes = plt.subplots(k + 1, 1, figsize=(13, 1.6 * (k + 1)),
                             gridspec_kw={"height_ratios": [3] * k + [2]})
    _dark(fig)
    pc_r2 = []
    for i in range(k):
        ax = axes[i]
        _style(ax)
        ax.plot(t, scores_true[:, i], color=COLORS["truth"], lw=1.0, alpha=0.9, label="Truth")
        valid = np.isfinite(scores_pred[:, i])
        ax.plot(t[valid], scores_pred[valid, i], color=COLORS["latent"], lw=1.2, alpha=0.9, label="Predicted")
        r2 = r2_score(scores_true[:, i], scores_pred[:, i])
        pc_r2.append(r2)
        ax.set_ylabel(f"PC{i + 1}\n({var_exp[i] * 100:.1f}%)\nR²={r2:.2f}", fontsize=8)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8, facecolor="#1a1f35",
                      edgecolor=COLORS["border"], labelcolor=COLORS["text"])
    # variance-explained strip
    axb = axes[-1]
    _style(axb)
    axb.bar(np.arange(1, k + 1), var_exp * 100, color=COLORS["pred"], alpha=0.85)
    axb.set_ylabel("Var\nexpl. %", fontsize=8)
    axb.set_xlabel("Principal component", fontsize=10)
    axb.set_xticks(np.arange(1, k + 1))
    fig.suptitle(title, color=COLORS["text"], fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return pc_r2


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def generate_validation_plots(entry: dict, device: str | None = None) -> dict | None:
    """Run inference + emit the three plots and metrics for one experiment entry.

    ``entry`` is a registry-shaped dict (see dashboard_common). Returns a result
    dict (paths + metrics) or None if skipped (no checkpoint / unsupported type).
    """
    name = entry["name"]
    model_type = entry.get("model_type", "")
    ckpt_dir = entry.get("checkpoint_dir")
    if not ckpt_dir:
        print(f"  [skip] {name}: no checkpoint (metadata-only)")
        return None
    if model_type not in SUPPORTED:
        print(f"  [skip] {name}: model_type '{model_type}' inference not implemented")
        return None
    ckpt_path = find_checkpoint(ckpt_dir)
    if ckpt_path is None:
        print(f"  [skip] {name}: checkpoint not found under {ckpt_dir}")
        return None

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(DEFAULT_SEED)

    data_path = resolve_data(entry.get("data_file", "data/training_trials.h5"))
    data = load_trials_h5(str(data_path))
    x_all, u_all, dt = data["x"], data["u"], float(data["dt"])
    n_train = x_all.shape[0] - DEFAULT_TEST_TRIALS
    x_test, u_test = x_all[n_train], u_all[n_train]  # first held-out test trial

    model, norms = load_model(model_type, ckpt_path, device)
    x_n = (x_test - norms["x_mean"]) / norms["x_std"]
    u_n = (u_test - norms["u_mean"]) / norms["u_std"]

    print(f"  {name}: inferring ({model_type}, T={x_test.shape[1]}, dev={device}) ...")
    pred_n = infer(model_type, model, x_n, u_n, dt, device, entry.get("past_window", 200))
    x_pred = pred_n * norms["x_std"] + norms["x_mean"]
    x_true = x_test

    # metrics
    overall = r2_score(x_true, x_pred)
    ch_r2 = per_channel_r2(x_true, x_pred)
    scores_t, scores_p, var_exp = pca_project(x_true, x_pred, k=4)

    out_dir = RESULTS / ckpt_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    r2_tag = f"R²={overall:.3f}"
    # matplotlib lacks emoji glyphs -> strip non-ASCII from figure titles only.
    safe = name.encode("ascii", "ignore").decode("ascii").strip()
    plot_channel_heatmap(x_true, x_pred, dt, out_dir / "val_channel_heatmap.png",
                         f"{safe} — channel firing rates ({r2_tag})")
    plot_channel_traces(x_true, x_pred, dt, out_dir / "val_channel_traces.png",
                        f"{safe} — top-variance channels ({r2_tag})")
    pc_r2 = plot_pc_projection(scores_t, scores_p, var_exp, dt,
                               out_dir / "val_pc_projection.png",
                               f"{safe} — top principal components ({r2_tag})")

    metrics = {
        "experiment": name,
        "slug": slugify(name),
        "model_type": model_type,
        "checkpoint_dir": ckpt_dir,
        "data_file": entry.get("data_file"),
        "test_trial_index": int(n_train),
        "r2_overall": overall,
        "r2_per_channel": [None if np.isnan(v) else round(float(v), 4) for v in ch_r2],
        "var_explained_top4": [round(float(v), 4) for v in var_exp],
        "r2_per_pc_top4": [None if np.isnan(v) else round(float(v), 4) for v in pc_r2],
    }
    with open(out_dir / "val_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"    overall R²={overall:.4f}; wrote val_*.png + val_metrics.json -> {out_dir}")
    return metrics


def _find_entry(results_dir: str) -> dict | None:
    """Match a --results-dir argument to a registry entry by checkpoint_dir or slug."""
    rd = results_dir.rstrip("/")
    for e in all_experiments():
        cd = e.get("checkpoint_dir")
        if cd and (cd.rstrip("/") == rd or cd.split("/")[0] == rd):
            return e
        if slugify(e["name"]) == rd:
            return e
    return None


def _resolve_exp_dir(results_dir: str) -> "Path | None":
    """Resolve a results dir (abs, 'results/foo', or 'foo') to the on-disk dir
    that actually holds a MANIFEST.json (shared checkout preferred)."""
    rd = Path(results_dir)
    candidates = [rd] if rd.is_absolute() else [
        SHARED_REPO / rd, REPO / rd, SHARED_RESULTS / rd.name, RESULTS / rd.name,
    ]
    for c in candidates:
        if (c / "MANIFEST.json").exists():
            return c
    return None


def validate_experiment(results_dir: str, device: str | None = None) -> dict | None:
    """Generate validation plots for one experiment from its MANIFEST.json.

    Used by ``preflight complete``. Resolves the best checkpoint + model_type via
    ``manifest_to_entry`` and runs ``generate_validation_plots``. Returns the
    metrics dict, or None if skipped (no manifest/checkpoint, unsupported model) —
    it never raises, so a completion is never blocked by validation.
    """
    exp_dir = _resolve_exp_dir(results_dir)
    if exp_dir is None:
        print(f"  [skip] validate_experiment: no MANIFEST.json for {results_dir}")
        return None
    entry = manifest_to_entry(exp_dir)
    if entry is None:
        print(f"  [skip] validate_experiment: could not read manifest in {exp_dir}")
        return None
    try:
        return generate_validation_plots(entry, device)
    except Exception as exc:
        print(f"  [warn] validate_experiment failed for {entry.get('name')}: {exc}")
        return None


def main():
    ap = argparse.ArgumentParser(description="Generate validation plots for digital-twin models")
    ap.add_argument("--results-dir", help="results sub-dir (checkpoint dir) or experiment slug")
    ap.add_argument("--experiment-dir",
                    help="experiment results dir with a MANIFEST.json (used by preflight); "
                         "resolves the best run + model_type from the manifest")
    ap.add_argument("--all", action="store_true", help="run every registry entry with a checkpoint")
    ap.add_argument("--model-type", help="override inferred model_type")
    ap.add_argument("--data-file", help="override data file")
    args = ap.parse_args()

    if args.experiment_dir:
        # Best-effort single-experiment validation from its manifest.
        validate_experiment(args.experiment_dir)
        return

    if args.all:
        entries = all_experiments()
    elif args.results_dir:
        e = _find_entry(args.results_dir)
        if e is None:
            # Fall back to a bare checkpoint dir not in the registry.
            e = {"name": args.results_dir, "checkpoint_dir": args.results_dir,
                 "model_type": args.model_type or "", "data_file": args.data_file or "data/training_trials.h5"}
        entries = [e]
    else:
        ap.error("provide --results-dir or --all")

    if args.model_type:
        for e in entries:
            e["model_type"] = args.model_type
    if args.data_file:
        for e in entries:
            e["data_file"] = args.data_file

    print("=== Validation Plots ===")
    done = 0
    for e in entries:
        try:
            if generate_validation_plots(e):
                done += 1
        except Exception as exc:  # keep the batch going if one model fails
            print(f"  [ERROR] {e.get('name', e.get('checkpoint_dir'))}: {exc}")
    print(f"\nDone: generated validation plots for {done}/{len(entries)} experiment(s).")


if __name__ == "__main__":
    main()
