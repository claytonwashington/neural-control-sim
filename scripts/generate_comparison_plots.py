#!/usr/bin/env python
"""Generate comparison visualizations for top-performing Cleo digital twin models.

Creates three figures:
  1. prediction_comparison_top3.png - 200ms prediction traces for 3 models
  2. model_comparison_overlay.png - overlay of all 3 models vs ground truth
  3. r2_comparison_all.png - horizontal bar chart of R2 for all models
"""

import sys
import os

CLEO_ROOT = "/mnt/cbwash2/cleo"
ALIGNED_ROOT = "/mnt/cbwash2/cleo-worktrees/aligned-distill"
sys.path.insert(0, CLEO_ROOT)
sys.path.insert(0, ALIGNED_ROOT)

import numpy as np
import torch
import h5py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Dark theme
BG_DARK = "#1a1a2e"
BG_PANEL = "#16213e"
TEXT_COL = "#e0e0e0"
GRID_COL = "#334155"
ACCENT_BLUE = "#4fc3f7"
ACCENT_ORANGE = "#ff9800"
ACCENT_GREEN = "#66bb6a"
ACCENT_PURPLE = "#ab47bc"
ACCENT_RED = "#ef5350"
ACTUAL_COL = "#b0b0b0"

rcParams.update({
    "figure.facecolor": BG_DARK,
    "axes.facecolor": BG_PANEL,
    "axes.edgecolor": GRID_COL,
    "axes.labelcolor": TEXT_COL,
    "text.color": TEXT_COL,
    "xtick.color": TEXT_COL,
    "ytick.color": TEXT_COL,
    "grid.color": GRID_COL,
    "grid.alpha": 0.3,
    "legend.facecolor": BG_PANEL,
    "legend.edgecolor": GRID_COL,
    "font.family": "sans-serif",
    "font.size": 11,
})

# ── Load test data ──
print("Loading data...")
with h5py.File(os.path.join(CLEO_ROOT, "data/training_trials.h5"), "r") as f:
    x_all = f["x"][:]
    u_all = f["u"][:]
    dt = float(f.attrs["dt"])

n_train = 40
n_test = 10
x_test_trials = x_all[n_train:]
u_test_trials = u_all[n_train:]

trial_idx = 0
t_start = 5000
past_window = 200
future_window = 200
channels = [0, 10, 20, 30]
n_ch_plot = len(channels)

x_trial = x_test_trials[trial_idx]
u_trial = u_test_trials[trial_idx]

t_past_ms = np.arange(past_window) * dt * 1000
t_future_ms = np.arange(future_window) * dt * 1000 + past_window * dt * 1000
t_split = t_start + past_window
t_end = t_split + future_window
x_true_future = x_trial[:, t_split:t_end]
x_past_actual = x_trial[:, t_start:t_split]

# ── 1. N4SID ──
print("Loading N4SID model...")
n4sid_data = np.load(os.path.join(CLEO_ROOT, "results/n4sid/n4sid_model.npz"))
A, B, C, D = n4sid_data["A"], n4sid_data["B"], n4sid_data["C"], n4sid_data["D"]
n4sid_r2 = float(n4sid_data["avg_r2"])
print(f"  N4SID R2 = {n4sid_r2:.4f}")

x0_n4sid = x_trial[:, t_split]
u_future_n4sid = u_trial[:, t_split:t_end]
C_pinv = np.linalg.pinv(C)
z_n4sid = C_pinv @ x0_n4sid
x_pred_n4sid = np.zeros((50, future_window))
for k in range(future_window):
    x_pred_n4sid[:, k] = C @ z_n4sid + D @ u_future_n4sid[:, k]
    z_n4sid = A @ z_n4sid + B @ u_future_n4sid[:, k]

# ── 2. Causal CA-NODE ──
print("Loading Causal CA-NODE model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {device}")

causal_ckpt = torch.load(
    os.path.join(CLEO_ROOT, "results/causal_z64_pw200_h128_lr3e4/model.pt"),
    map_location=device, weights_only=False
)

from modeling.models.latent_canode import LatentControlAffineODE

causal_model = LatentControlAffineODE(
    n_x=int(causal_ckpt["n_x"]), n_u=int(causal_ckpt["n_u"]),
    z_dim=int(causal_ckpt["z_dim"]), hidden_dim=int(causal_ckpt["hidden"]),
    n_layers=int(causal_ckpt["n_layers"]),
)
causal_model.load_state_dict(causal_ckpt["model_state"])
causal_model.to(device)
causal_model.eval()

cx_mean, cx_std = causal_ckpt["x_mean"], causal_ckpt["x_std"]
cu_mean, cu_std = causal_ckpt["u_mean"], causal_ckpt["u_std"]

x_trial_cn = (x_trial - cx_mean) / cx_std
u_trial_cn = (u_trial - cu_mean) / cu_std

x_past_c = torch.tensor(x_trial_cn[:, t_start:t_split].T, dtype=torch.float32).unsqueeze(0).to(device)
u_past_c = torch.tensor(u_trial_cn[:, t_start:t_split].T, dtype=torch.float32).unsqueeze(0).to(device)
u_future_c = torch.tensor(u_trial_cn[:, t_split:t_end].T, dtype=torch.float32).unsqueeze(0).to(device)
t_future_tensor = torch.arange(future_window, dtype=torch.float32).to(device) * dt

with torch.no_grad():
    x_pred_causal_n = causal_model.predict(x_past_c, u_past_c, u_future_c, t_future_tensor, method="dopri5")

x_pred_causal = x_pred_causal_n[0].cpu().numpy().T * cx_std + cx_mean

# ── 3. Aligned Distill ──
print("Loading Aligned Distill model...")
aligned_ckpt = torch.load(
    os.path.join(ALIGNED_ROOT, "results/aligned_cosine_0.9_0.1/best_model.pt"),
    map_location=device, weights_only=False
)

aligned_model = LatentControlAffineODE(
    n_x=int(aligned_ckpt["n_x"]), n_u=int(aligned_ckpt["n_u"]),
    z_dim=int(aligned_ckpt["z_dim"]), hidden_dim=int(aligned_ckpt["hidden"]),
    n_layers=int(aligned_ckpt["n_layers"]),
)
aligned_model.load_state_dict(aligned_ckpt["model_state"])
aligned_model.to(device)
aligned_model.eval()

ax_mean, ax_std = aligned_ckpt["x_mean"], aligned_ckpt["x_std"]
au_mean, au_std = aligned_ckpt["u_mean"], aligned_ckpt["u_std"]

x_trial_an = (x_trial - ax_mean) / ax_std
u_trial_an = (u_trial - au_mean) / au_std

x_past_a = torch.tensor(x_trial_an[:, t_start:t_split].T, dtype=torch.float32).unsqueeze(0).to(device)
u_past_a = torch.tensor(u_trial_an[:, t_start:t_split].T, dtype=torch.float32).unsqueeze(0).to(device)
u_future_a = torch.tensor(u_trial_an[:, t_split:t_end].T, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    x_pred_aligned_n = aligned_model.predict(x_past_a, u_past_a, u_future_a, t_future_tensor, method="dopri5")

x_pred_aligned = x_pred_aligned_n[0].cpu().numpy().T * ax_std + ax_mean

# ═══════════════════════════════════════════════════════════
# FIGURE 1: Prediction traces
# ═══════════════════════════════════════════════════════════
print("\nGenerating Figure 1: Prediction traces...")

fig1, axes1 = plt.subplots(n_ch_plot, 3, figsize=(18, 10), sharex=True)
fig1.suptitle("200ms Prediction Traces \u2014 Test Trial 40, t = 5.2\u20135.4s",
              fontsize=16, fontweight="bold", color=TEXT_COL, y=0.98)

model_names = [f"N4SID (R\u00b2={n4sid_r2:.2f})", "Causal CA-NODE (R\u00b2=0.825)", "Aligned Distill (R\u00b2=0.864)"]
model_preds = [x_pred_n4sid, x_pred_causal, x_pred_aligned]
model_colors = [ACCENT_BLUE, ACCENT_ORANGE, ACCENT_GREEN]

for col, (name, pred, color) in enumerate(zip(model_names, model_preds, model_colors)):
    for row, ch in enumerate(channels):
        ax = axes1[row, col]
        ax.plot(t_future_ms, x_true_future[ch], color=ACTUAL_COL, lw=1.5, alpha=0.8)
        ax.plot(t_future_ms, pred[ch], color=color, lw=1.5, alpha=0.9)
        ax.grid(True, alpha=0.2)
        if row == 0:
            ax.set_title(name, fontsize=12, fontweight="bold", color=color, pad=10)
        if col == 0:
            ax.set_ylabel(f"Ch {ch}\nFiring Rate", fontsize=10)
        if row == n_ch_plot - 1:
            ax.set_xlabel("Time (ms)", fontsize=10)
        ax.fill_between(t_future_ms, x_true_future[ch], pred[ch], alpha=0.1, color=color)

handles = [
    plt.Line2D([0], [0], color=ACTUAL_COL, lw=2, label="Ground Truth"),
    plt.Line2D([0], [0], color=ACCENT_BLUE, lw=2, label="N4SID"),
    plt.Line2D([0], [0], color=ACCENT_ORANGE, lw=2, label="Causal CA-NODE"),
    plt.Line2D([0], [0], color=ACCENT_GREEN, lw=2, label="Aligned Distill"),
]
fig1.legend(handles=handles, loc="lower center", ncol=4, fontsize=11,
            frameon=True, fancybox=True, shadow=True, bbox_to_anchor=(0.5, 0.01))
fig1.tight_layout(rect=[0, 0.05, 1, 0.96])
fig1.savefig(os.path.join(CLEO_ROOT, "results/prediction_comparison_top3.png"),
             dpi=200, bbox_inches="tight", facecolor=BG_DARK)
print("  Saved: results/prediction_comparison_top3.png")

# ═══════════════════════════════════════════════════════════
# FIGURE 2: Model comparison overlay
# ═══════════════════════════════════════════════════════════
print("\nGenerating Figure 2: Model comparison overlay...")

fig2, axes2 = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
fig2.suptitle("Model Comparison Overlay \u2014 Test Trial 40, Channels [0, 10, 20, 30]",
              fontsize=16, fontweight="bold", color=TEXT_COL, y=0.98)

overlay_configs = [
    ("N4SID", x_pred_n4sid, ACCENT_BLUE, f"R\u00b2={n4sid_r2:.2f}"),
    ("Causal CA-NODE", x_pred_causal, ACCENT_ORANGE, "R\u00b2=0.825"),
    ("Aligned Distill", x_pred_aligned, ACCENT_GREEN, "R\u00b2=0.864"),
]

t_full_ms = np.concatenate([t_past_ms, t_future_ms])

for row, (name, pred, color, r2_label) in enumerate(overlay_configs):
    ax = axes2[row]
    for i, ch in enumerate(channels):
        x_actual_full = np.concatenate([x_past_actual[ch], x_true_future[ch]])
        label_actual = "Actual" if i == 0 else None
        ax.plot(t_full_ms, x_actual_full, color=ACTUAL_COL, lw=1.0, alpha=0.6, label=label_actual)

    ax.axvline(x=past_window * dt * 1000, color=ACCENT_RED, ls="--", lw=1.5, alpha=0.7,
               label="Prediction start")

    for i, ch in enumerate(channels):
        ax.plot(t_past_ms, x_past_actual[ch], color=color, lw=0.5, alpha=0.3)
        label_model = name if i == 0 else None
        ax.plot(t_future_ms, pred[ch], color=color, lw=1.8, alpha=0.9, label=label_model)

    ax.set_title(f"{name}  ({r2_label})", fontsize=13, fontweight="bold", color=color, loc="left", pad=8)
    ax.set_ylabel("Firing Rate", fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)

axes2[-1].set_xlabel("Time (ms)", fontsize=11)
fig2.tight_layout(rect=[0, 0, 1, 0.96])
fig2.savefig(os.path.join(CLEO_ROOT, "results/model_comparison_overlay.png"),
             dpi=200, bbox_inches="tight", facecolor=BG_DARK)
print("  Saved: results/model_comparison_overlay.png")

# ═══════════════════════════════════════════════════════════
# FIGURE 3: R2 bar chart
# ═══════════════════════════════════════════════════════════
print("\nGenerating Figure 3: R\u00b2 bar chart...")

models = [
    ("N4SID",           0.40,  "linear"),
    ("GRU",             0.69,  "causal"),
    ("Causal CA-NODE",  0.825, "causal"),
    ("Aligned Distill", 0.864, "enhanced"),
    ("Causal + EnKF",   0.93,  "enhanced"),
    ("Acausal NODE",    0.94,  "acausal"),
]

category_colors = {
    "linear":   ACCENT_BLUE,
    "causal":   ACCENT_ORANGE,
    "enhanced": ACCENT_GREEN,
    "acausal":  ACCENT_PURPLE,
}

category_labels = {
    "linear":   "Linear Baseline",
    "causal":   "Causal Neural ODE",
    "enhanced": "Enhanced / Hybrid",
    "acausal":  "Acausal (Oracle)",
}

fig3, ax3 = plt.subplots(figsize=(12, 6))
fig3.suptitle("Model Comparison \u2014 200ms Prediction R\u00b2",
              fontsize=18, fontweight="bold", color=TEXT_COL, y=0.97)

y_pos = np.arange(len(models))
bar_colors = [category_colors[m[2]] for m in models]
r2_values = [m[1] for m in models]
model_labels_list = [m[0] for m in models]

bars = ax3.barh(y_pos, r2_values, height=0.6, color=bar_colors, edgecolor="white",
                linewidth=0.5, alpha=0.9)

for i, (bar, r2) in enumerate(zip(bars, r2_values)):
    text_x = r2 + 0.015
    ax3.text(text_x, bar.get_y() + bar.get_height() / 2,
             f"{r2:.3f}", va="center", ha="left",
             fontsize=12, fontweight="bold", color=TEXT_COL)

ax3.set_yticks(y_pos)
ax3.set_yticklabels(model_labels_list, fontsize=12, fontweight="bold")
ax3.set_xlabel("R\u00b2 (200ms prediction horizon)", fontsize=13)
ax3.set_xlim(0, 1.08)
ax3.invert_yaxis()
ax3.grid(True, axis="x", alpha=0.2)

for ref_val in [0.5, 0.9]:
    ax3.axvline(x=ref_val, color=GRID_COL, ls=":", lw=1, alpha=0.5)

legend_handles = [
    plt.Rectangle((0, 0), 1, 1, fc=color, ec="white", lw=0.5, alpha=0.9)
    for color in category_colors.values()
]
legend_labels_list = list(category_labels.values())
ax3.legend(legend_handles, legend_labels_list, loc="lower right",
           fontsize=10, frameon=True, fancybox=True, shadow=True,
           title="Category", title_fontsize=11)

ax3.axvspan(0.9, 1.0, alpha=0.05, color=ACCENT_GREEN)
ax3.axvspan(0.8, 0.9, alpha=0.03, color=ACCENT_ORANGE)

fig3.tight_layout(rect=[0, 0, 1, 0.95])
fig3.savefig(os.path.join(CLEO_ROOT, "results/r2_comparison_all.png"),
             dpi=200, bbox_inches="tight", facecolor=BG_DARK)
print("  Saved: results/r2_comparison_all.png")

print("\nAll 3 figures generated successfully!")
print(f"  1. {CLEO_ROOT}/results/prediction_comparison_top3.png")
print(f"  2. {CLEO_ROOT}/results/model_comparison_overlay.png")
print(f"  3. {CLEO_ROOT}/results/r2_comparison_all.png")
