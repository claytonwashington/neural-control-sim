#!/usr/bin/env python
"""One-command refresh of the digital-twin dashboard.

Runs the dashboard pipeline in the canonical order:

  1. generate_plant_assets        — render each plant + plants.json
  2. validation_plots --all       — true-vs-inferred plots for every checkpoint
  3. generate_experiment_pages    — standalone results/experiments/<slug>.html
  4. update_dashboard --plants-only  — inject the Plants tab (idempotent)
  5. update_dashboard_leaderboard — rebuild leaderboard rows + experiment links

All steps are idempotent, so this is safe to re-run after each new experiment.
Run from the repo (or worktree) under the dtmodeling env::

    conda run -n dtmodeling python -m modeling.scripts.build_dashboard
    # reuse existing validation/plant assets (skip the slow GPU step):
    conda run -n dtmodeling python -m modeling.scripts.build_dashboard --skip-validation
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from modeling.scripts.dashboard_common import REPO

STEPS = [
    ("plants", "generate_plant_assets", []),
    ("validation", "validation_plots", ["--all"]),
    ("pages", "generate_experiment_pages", []),
    ("plants_tab", "update_dashboard", ["--plants-only"]),
    ("leaderboard", "update_dashboard_leaderboard", []),
]


def run_step(module: str, extra: list[str]) -> int:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-m", f"modeling.scripts.{module}", *extra]
    print(f"\n{'=' * 70}\n▶ {' '.join(cmd)}\n{'=' * 70}")
    return subprocess.run(cmd, cwd=str(REPO), env=env).returncode


def main():
    ap = argparse.ArgumentParser(description="Rebuild the digital-twin dashboard")
    ap.add_argument("--skip-plants", action="store_true", help="reuse existing plant renders")
    ap.add_argument("--skip-validation", action="store_true",
                    help="reuse existing validation plots (skip GPU inference)")
    args = ap.parse_args()

    skip = set()
    if args.skip_plants:
        skip.add("plants")
    if args.skip_validation:
        skip.add("validation")

    results = {}
    for key, module, extra in STEPS:
        if key in skip:
            print(f"\n[skip] {module}")
            continue
        rc = run_step(module, extra)
        results[module] = rc

    print(f"\n{'=' * 70}\nDashboard build summary")
    for module, rc in results.items():
        print(f"  {'✅' if rc == 0 else '❌'} {module} (exit {rc})")
    failed = [m for m, rc in results.items() if rc != 0]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
