#!/usr/bin/env python
"""Shared helpers and registries for the dashboard tooling.

This module is the single source of truth for:
  * Filesystem roots (where heavy inputs live vs. where artifacts are written).
  * ``slugify`` — experiment name -> stable filename slug (shared by the page
    generator and the leaderboard linker so links always match).
  * ``PLANTS`` — the plant designs to render on the Plants page.
  * ``EXPERIMENT_REGISTRY`` — legacy/leaderboard experiments and the checkpoint
    (if any) used to produce their validation plots. Newer experiments are
    discovered automatically from ``MANIFEST.json`` via ``discover_manifest_experiments``.
  * Small HTML helpers (``png_to_data_uri``, ``make_image_card``) and the dark
    theme colour palette, reused across the dashboard scripts.

Roots design
------------
Heavy inputs (``model.pt`` checkpoints, ``data/*.h5``) are git-ignored and live
only in the shared main checkout. Generated artifacts (validation PNGs, plant
renders, experiment pages, the regenerated dashboard) are written into the
*current* checkout's ``results/`` so they can be committed from a worktree.
When run from the main checkout the two roots coincide.
"""

from __future__ import annotations

import base64
import os
import re
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------------------
# Filesystem roots
# ---------------------------------------------------------------------------
# Shared checkout where git-ignored checkpoints + datasets physically live.
SHARED_REPO = Path(os.environ.get("CLEO_SHARED_REPO", "/snel/home/cbwash2/cleo"))
# This checkout (worktree or main) — where we WRITE artifacts.
REPO = Path(__file__).resolve().parents[2]

SHARED_RESULTS = SHARED_REPO / "results"
SHARED_DATA = SHARED_REPO / "data"
RESULTS = REPO / "results"
EXPERIMENTS_DIR = RESULTS / "experiments"
PLANTS_DIR = RESULTS / "plants"


def find_checkpoint(rel_dir: str | os.PathLike) -> Path | None:
    """Locate ``model.pt`` for a results sub-directory, preferring the shared repo.

    Returns ``None`` if no checkpoint exists in either checkout.
    """
    for root in (SHARED_RESULTS, RESULTS):
        p = root / rel_dir / "model.pt"
        if p.exists():
            return p
    return None


def resolve_data(data_file: str | os.PathLike) -> Path:
    """Resolve a dataset reference (possibly ``data/foo.h5``) to an absolute path.

    Reads from the shared checkout where the ``.h5`` files live.
    """
    p = Path(data_file)
    if p.is_absolute():
        return p
    # Accept both "data/foo.h5" and bare "foo.h5".
    if p.parts and p.parts[0] == "data":
        return SHARED_REPO / p
    return SHARED_DATA / p


# Model families for which validation_plots can run inference (kept here, torch-free,
# so the page generator can explain *why* an experiment has no plots).
SUPPORTED_MODEL_TYPES = {"canode", "latent_canode", "latent_node", "gru", "n4sid"}


def find_n4sid(rel_dir: str | os.PathLike) -> Path | None:
    """Locate an N4SID ``n4sid_model.npz`` (linear state-space, not a torch ckpt)."""
    for root in (SHARED_RESULTS, RESULTS):
        p = root / rel_dir / "n4sid_model.npz"
        if p.exists():
            return p
    return None


def entry_has_checkpoint(entry: dict) -> bool:
    """Whether an experiment has a loadable model (torch model.pt, or N4SID npz)."""
    cd = entry.get("checkpoint_dir")
    if not cd:
        return False
    if entry.get("model_type") == "n4sid":
        return find_n4sid(cd) is not None
    return find_checkpoint(cd) is not None


def no_plots_reason(entry: dict) -> str | None:
    """Human-readable reason an experiment has no validation plots, or None if it
    should have them. Used on the experiment page (L2)."""
    if not entry_has_checkpoint(entry):
        return ("No saved checkpoint — this model was never persisted (the run kept "
                "only logs/curves), so it can't be reloaded for validation.")
    mt = entry.get("model_type", "")
    if mt not in SUPPORTED_MODEL_TYPES:
        return (f"Validation inference for model type “{mt or 'unknown'}” isn't "
                f"implemented yet (only {', '.join(sorted(SUPPORTED_MODEL_TYPES))}).")
    return ("Validation plots haven't been generated yet — run "
            "`build_dashboard.py` / complete the experiment via preflight.")


# ---------------------------------------------------------------------------
# Dark theme palette (matches dashboard.html / existing plot styling)
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#0a0e1a",
    "panel": "#111827",
    "border": "#2a3055",
    "text": "#e5e7eb",
    "muted": "#9ca3af",
    "truth": "#e5e7eb",
    "pred": "#6366f1",
    "canode": "#10b981",
    "latent": "#f59e0b",
    "error": "#ef4444",
}


# ---------------------------------------------------------------------------
# Slug + HTML helpers
# ---------------------------------------------------------------------------
def slugify(name: str) -> str:
    """Convert an experiment display name to a stable, filesystem-safe slug.

    Strips emoji/accents, lowercases, and collapses runs of non-alphanumerics
    to single hyphens. Deterministic so the page generator and the leaderboard
    linker always agree on ``experiments/<slug>.html``.
    """
    norm = unicodedata.normalize("NFKD", name)
    ascii_only = norm.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_only).strip("-").lower()
    return slug or "experiment"


def png_to_data_uri(filepath: str | os.PathLike) -> str:
    """Convert a PNG file to a base64 ``data:`` URI for self-contained HTML."""
    with open(filepath, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def make_image_card(title: str, data_uri: str, caption: str = "") -> str:
    """Create an image card matching the dashboard glassmorphism style."""
    cap = (
        f'<div style="padding: 8px 18px; font-size: 0.8rem; '
        f'color: var(--text-secondary);">{caption}</div>'
        if caption
        else ""
    )
    return f"""      <div class="image-card" style="margin-top: 20px;">
        <div class="img-header">{title}</div>
        <img src="{data_uri}" alt="{title}" onclick="openLightbox(this)" style="cursor:zoom-in; background: #0a0e1a;">
        {cap}
      </div>"""


# ---------------------------------------------------------------------------
# Plant registry — what the Plants page renders
# ---------------------------------------------------------------------------
# Each entry: id, title, builder (module path resolved lazily), metadata.
# Source of truth: PLANTS.md on the feature/bidir-v2-plant branch. Two ACTIVE
# plant designs are rendered; the ChrimsonR+GtACR2 "bidirectional v1" plant is
# RETIRED (broken GtACR2 inhibition) and listed for provenance only.
PLANTS = [
    {
        "id": "unidirectional_v0",
        "title": "Plant 1 — Unidirectional Excitatory (v0)",
        "builder": "build_plant",
        "neurons": "800 excitatory + 200 inhibitory LIF (p=0.1)",
        "opsins": "ChrimsonR (590 nm, excitatory) + GtACR2 (470 nm) — GtACR2 inhibition is "
                  "negligible (E≈E_L), so this plant is effectively excitatory-only",
        "inputs": "u[0] = red fiber (excite) · u[1] = blue fiber (no effective drive)",
        "probe": "50-channel MultiUnitActivity probe (planar MEA, Y=0)",
        "datasets": [
            "data/training_trials.h5",
        ],
        "desc": (
            "The original baseline plant (Exp 1–27). ChrimsonR provides excitation; "
            "GtACR2's reversal potential (E=−69.5 mV) sits almost on top of rest "
            "(E_L=−70 mV), giving only ~0.5 mV of hyperpolarizing drive — so "
            "inhibition is effectively absent and the plant behaves as excitatory-only. "
            "Produced the canonical 50-trial dataset used by N4SID, CA-NODE, the "
            "causal/acausal Latent NODE, and GRU experiments."
        ),
    },
    {
        "id": "bidirectional_v2",
        "title": "Plant 3 — Bidirectional v2 (ChR2-H134R + eNpHR3.0)",
        "builder": "build_plant_v2",
        "neurons": "800 excitatory + 200 inhibitory LIF (p=0.1)",
        "opsins": "ChR2(H134R) (450 nm, excitatory cation channel) + eNpHR3.0 "
                  "(590 nm, inhibitory chloride pump, E=−400 mV)",
        "inputs": "u[0] = blue fiber (excite) · u[1] = yellow fiber (inhibit)",
        "probe": "50-channel MultiUnitActivity probe (planar MEA, Y=0)",
        "datasets": [
            "data/training_trials_bidir_v2.h5",
        ],
        "desc": (
            "The current primary plant (Exp 28–32). eNpHR3.0 is a chloride pump with "
            "E=−400 mV, delivering ~330 mV of hyperpolarizing drive (vs GtACR2's "
            "~0.5 mV) — enabling genuine, monotonic inhibition below baseline "
            "(287 → 6 Hz, ~98% suppression). This is the plant to use for true "
            "bidirectional optoclamp control."
        ),
    },
]

# Retired / provenance-only — not rendered, but available for experiment-page
# plant lookups (e.g. the ChrimsonR+GtACR2 distillation runs ran on this).
RETIRED_PLANTS = [
    {
        "id": "bidirectional_v1_retired",
        "title": "Plant 2 — Bidirectional v1 (ChrimsonR + GtACR2) · RETIRED",
        "builder": None,
        "neurons": "800 excitatory + 200 inhibitory LIF (p=0.1)",
        "opsins": "ChrimsonR (590 nm) + GtACR2 (470 nm) — GtACR2 inhibition broken (E≈E_L)",
        "inputs": "u[0] = red fiber (excite) · u[1] = blue fiber (negligible)",
        "probe": "50-channel MultiUnitActivity probe (planar MEA, Y=0)",
        "datasets": [
            "data/training_trials_bidirectional.h5",
        ],
        "desc": (
            "RETIRED. GtACR2's reversal potential makes inhibition negligible, so this "
            "plant cannot deliver real bidirectional control. Superseded by Plant 3 "
            "(ChR2-H134R + eNpHR3.0). Kept only to document earlier distillation runs."
        ),
    },
]

PLANTS_BY_ID = {p["id"]: p for p in PLANTS + RETIRED_PLANTS}


def _plant_id_for_data(data_file: str) -> str:
    """Map a dataset filename to the plant design that generated it."""
    if "bidir_v2" in data_file:
        return "bidirectional_v2"
    if "bidirectional" in data_file:  # v1 (ChrimsonR+GtACR2) — retired
        return "bidirectional_v1_retired"
    return "unidirectional_v0"


# ---------------------------------------------------------------------------
# Experiment registry — legacy / leaderboard experiments
# ---------------------------------------------------------------------------
# Maps a leaderboard display name to the checkpoint + inference config used to
# produce its validation plots. ``checkpoint_dir`` is relative to results/;
# None means no saved model.pt (metadata-only page). ``model_type`` selects the
# inference path in validation_plots.py. Experiments that ship a MANIFEST.json
# (newer/bidirectional runs) are discovered separately and merged in.
EXPERIMENT_REGISTRY = [
    {
        "name": "Latent NODE (z=64, h256)",
        # Checkpoint recovered via fit_latent_node (the original sweep never
        # persisted model.pt); see results/latent_node_recovered/.
        "checkpoint_dir": "latent_node_recovered",
        "model_type": "latent_node",
        "data_file": "data/training_trials.h5",
        "plant_id": "unidirectional_v0",
        "past_window": 200,
        "note": "Best overall (acausal). Encoder sees the whole trial.",
        "existing_plot": "sweep_latent_node_40_10/run_05_z64_h256_l2_lr0.0005_dopri5/latent_node_prediction.png",
    },
    {
        "name": "CA-NODE (no skip — baseline)",
        "checkpoint_dir": "canode",
        "model_type": "canode",
        "data_file": "data/training_trials.h5",
        "plant_id": "unidirectional_v0",
        "note": "Channel-level control-affine ODE; the deployable real-time baseline.",
    },
    {
        "name": "CA-NODE (skip linear, swd=1.0)",
        "checkpoint_dir": "canode_skip",
        "model_type": "canode",
        "data_file": "data/training_trials.h5",
        "plant_id": "unidirectional_v0",
        "note": "Linear state/input skip connections. Overfit; did not beat baseline.",
    },
    {
        "name": "Causal Latent CA-NODE (z=64, lr=3e-4) \U0001f195",
        "checkpoint_dir": "causal_z64_pw200_h128_lr3e4",
        "model_type": "latent_canode",
        "data_file": "data/training_trials.h5",
        "plant_id": "unidirectional_v0",
        "past_window": 200,
        "note": "Best deployable causal latent model (200 ms past window).",
    },
    {
        "name": "GRU (best)",
        "checkpoint_dir": "gru",
        "model_type": "gru",
        "data_file": "data/training_trials.h5",
        "plant_id": "unidirectional_v0",
        "note": "Black-box RNN baseline. Overfits; weak generalization.",
    },
    {
        "name": "N4SID (40/10)",
        "checkpoint_dir": "n4sid_40_10",  # holds n4sid_model.npz (A/B/C/D matrices)
        "model_type": "n4sid",
        "data_file": "data/training_trials.h5",
        "plant_id": "unidirectional_v0",
        "note": "Linear subspace identification baseline (state-space .npz, not a torch ckpt).",
    },
]


def manifest_to_entry(exp_dir: Path | str) -> dict | None:
    """Build a registry-shaped entry from one results dir's ``MANIFEST.json``.

    Picks the best checkpoint: the ``run_*`` with the highest ``r2`` that has a
    saved ``model.pt`` (falling back to a single-run ``model.pt`` in the dir, then
    None for metadata-only). ``checkpoint_dir`` is relative to ``results/`` so it
    resolves via :func:`find_checkpoint`. Returns None if the dir has no manifest.
    """
    import glob
    import json

    exp_dir = Path(exp_dir)
    manifest_path = exp_dir / "MANIFEST.json"
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path) as f:
            man = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    rel_exp = exp_dir.name

    best_run, best_r2 = None, -1.0
    for rj in glob.glob(str(exp_dir / "run_*" / "results.json")):
        try:
            with open(rj) as f:
                r = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        r2 = r.get("r2", r.get("r2_200step", r.get("val_r2", -1)))
        run_dir = Path(rj).parent
        if r2 > best_r2 and (run_dir / "model.pt").exists():
            best_r2, best_run = r2, run_dir
    if best_run is not None:
        checkpoint_dir = f"{rel_exp}/{best_run.name}"
    elif (exp_dir / "model.pt").exists():
        checkpoint_dir = rel_exp  # single-run experiment
    else:
        checkpoint_dir = None

    data_file = man.get("data_file", "data/training_trials.h5")
    return {
        "name": man.get("experiment_name", rel_exp),
        "checkpoint_dir": checkpoint_dir,
        "results_dir": rel_exp,  # for control experiments: where the control assets live
        "model_type": man.get("model_type", "latent_canode"),
        "data_file": data_file,
        "plant_id": _plant_id_for_data(data_file),
        "past_window": man.get("past_window", 200),
        "hypothesis": man.get("hypothesis", ""),
        "idea_id": man.get("idea_id"),
        "best_r2": man.get("best_r2", best_r2 if best_r2 > 0 else None),
        "note": man.get("notes", ""),
    }


def discover_manifest_experiments(results_root: Path | None = None) -> list[dict]:
    """Discover all experiments that ship a ``MANIFEST.json`` under ``results_root``.

    Returns registry-shaped dicts (see :func:`manifest_to_entry`). Reads from the
    shared results root by default.
    """
    import glob

    root = Path(results_root) if results_root is not None else SHARED_RESULTS
    found: list[dict] = []
    for manifest_path in sorted(glob.glob(str(root / "*" / "MANIFEST.json"))):
        entry = manifest_to_entry(Path(manifest_path).parent)
        if entry is not None:
            found.append(entry)
    return found


def all_experiments() -> list[dict]:
    """Merge the static registry with MANIFEST-discovered experiments (deduped by slug)."""
    merged: dict[str, dict] = {}
    for entry in EXPERIMENT_REGISTRY + discover_manifest_experiments():
        merged.setdefault(slugify(entry["name"]), entry)
    return list(merged.values())
