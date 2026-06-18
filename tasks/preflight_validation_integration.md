# Preflight ↔ Validation/Dashboard Integration

**Branch:** `feature/dashboard-validation` → merged to `modeling-dev`
**Status:** Implemented

## Goal

Make per-experiment **validation plots** (true-vs-inferred firing rates across
channels + top PCs) and the **clickable experiment page** a guaranteed output of
the experiment lifecycle, produced automatically by `preflight complete` — the
same harness step that already merges an experiment's results into the dashboard
leaderboard. Every completed experiment should always end with a results page,
linked from the leaderboard.

## Context

- `modeling/scripts/preflight.py` owns the lifecycle. `preflight complete`
  (`_do_complete`) runs **gates before any state change**, then commits/pushes
  and cleans up the worktree.
  - GATE 1 `_require_eval_results` — needs `results.json`/`sweep_summary.json`.
  - GATE 2 `_require_dashboard_update` — ran *only* `update_dashboard_leaderboard.py`.
- The dashboard tooling (added earlier on this branch): `validation_plots.py`,
  `generate_experiment_pages.py`, `update_dashboard.py --plants-only`,
  `update_dashboard_leaderboard.py`, orchestrated by `build_dashboard.py`. Heavy
  inputs (`model.pt`, `data/*.h5`) are git-ignored and resolved from the shared
  checkout via `dashboard_common` (`find_checkpoint`, `resolve_data`); generated
  artifacts are written to the current checkout's `results/`.
- A `cleosim.pth` puts `/mnt/cbwash2/cleo` on `sys.path`, so `import
  modeling` always resolves; preflight subprocesses therefore run the main
  checkout's code. We still pass `PYTHONPATH=repo_root` so it works from a
  worktree too.

## Implementation

### 1. `MANIFEST.json` records `model_type` (`preflight.py::_do_start`)
- New `start` args `--model-type {canode,latent_canode,latent_node,gru,n4sid,other}`
  (default `latent_canode`) and `--past-window` (default 200), persisted into the
  manifest. Back-compat: older manifests without the field fall back to the default.

### 2. Single-experiment validation (`validation_plots.py`)
- `validate_experiment(results_dir)` + `--experiment-dir <dir>` CLI: reads the
  dir's MANIFEST, resolves the best `run_*/model.pt` and `model_type`/`data_file`
  via `dashboard_common.manifest_to_entry`, then calls the existing
  `generate_validation_plots(entry)`. Unsupported model type / missing checkpoint
  → returns *skipped* (never raises) so the page is metadata-only.

### 3. `dashboard_common.manifest_to_entry(exp_dir)`
- Extracted from `discover_manifest_experiments` (which now calls it): builds the
  registry-shaped entry (name, best `checkpoint_dir`, `model_type`, `data_file`,
  `plant_id`, `past_window`, hypothesis, best_r2) for one results dir.

### 4. GATE 2 becomes the ordered dashboard pipeline (`preflight.py::_require_dashboard_update`)
Run via `[sys.executable, "-m", "modeling.scripts.<mod>"]`, `cwd=repo_root`,
`env PYTHONPATH=repo_root`, in this order (so leaderboard links resolve to
freshly-written pages):
1. `validation_plots --experiment-dir <results_dir>` — **best-effort** (warn, don't block).
2. `generate_experiment_pages` — **hard gate**.
3. `update_dashboard --plants-only` — idempotent (soft).
4. `update_dashboard_leaderboard --results-dir <results_dir>` — **hard gate** (existing behavior, moved last).

Validation is best-effort because GRU/N4SID are intentionally inference-less and
GPU/env hiccups shouldn't block a completion; the **page + leaderboard row are
guaranteed**.

### 5. Commit the artifacts (`preflight.py::_do_complete`)
- The completion commit force-adds (`git add -f`) the generated dashboard outputs
  alongside `branches.md`/idea file: `results/dashboard.html`,
  `results/experiments/`, `results/plants/`, and the experiment's
  `**/val_metrics.json` (only paths that exist) — so GATE 3's push actually
  carries them. (Also fixes the pre-existing gap where dashboard updates were
  never committed.)

### 6. Env note
- `preflight complete` now needs the **`dtmodeling`** env (torch) for the
  validation step; subprocesses use `sys.executable`, so running preflight under
  `dtmodeling` is sufficient. New plant *designs* remain a one-time manual add
  (builder in `plant.py` + entry in `dashboard_common.PLANTS`, then run
  `build_dashboard` once); completion only re-embeds existing plant renders.

## Files
- `modeling/scripts/preflight.py` (start args + manifest, GATE 2 pipeline, commit).
- `modeling/scripts/validation_plots.py` (`--experiment-dir`, `validate_experiment`).
- `modeling/scripts/dashboard_common.py` (`manifest_to_entry`).
- Docs: this file; preflight docstring.

## Verification
- `validation_plots --experiment-dir results/bidir_aligned_distill` → val plots + metrics.
- Run the GATE-2 pipeline function against an existing experiment → page generated,
  leaderboard row links to it, artifacts staged.
- Unsupported model (GRU) → metadata-only page, no hard failure.
- Re-run → idempotent (no duplicate tabs/rows).
