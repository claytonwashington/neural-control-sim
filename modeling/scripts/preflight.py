#!/usr/bin/env python
"""Preflight harness for experiment lifecycle management.

Enforces the full experiment lifecycle:
  1. Link to an idea entry in an ideas/ file
  2. Create/validate a git worktree
  3. Register in branches.md with mutual cross-references
  4. Mark idea as IN PROGRESS
  5. Write MANIFEST.json with preflight token

Also supports completing experiments:
  python -m modeling.scripts.preflight --complete \
    --results-dir results/my_experiment \
    --status completed --best-r2 0.864 \
    --notes "Closes 35% of causal-acausal gap"

On `complete`, the dashboard gate regenerates this experiment's validation plots
(true-vs-inferred firing rates across channels + top PCs), its standalone
experiment page, the plants tab, and the leaderboard row, then commits/pushes
them. Validation inference needs the `dtmodeling` env (torch); run `complete`
under it. The `start` `--model-type` selects the validation inference path.

Usage:
  python -m modeling.scripts.preflight \
    --idea-file ideas/modeling.md \
    --idea-id 23 \
    --branch feature/my-experiment \
    --data data/training_trials.h5 \
    --machine gpu1 --gpu-ids 0,1 \
    --results-dir results/my_experiment
"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone


# ── Helpers ──────────────────────────────────────────────────────────────

def get_repo_root():
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def get_commit_sha():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _assert_main_checkout(repo_root):
    """HARD GATE: `complete` must run from the MAIN checkout on `modeling-dev`.

    Running from a linked worktree would execute that worktree's (possibly stale)
    harness code — the `cleosim.pth` makes this silent — and `_cleanup_worktree`'s
    `git merge <branch>` would run against the wrong HEAD. We detect a linked
    worktree by comparing `--git-dir` to `--git-common-dir` (equal only in the
    main checkout). Set `PREFLIGHT_ALLOW_ANY_CHECKOUT=1` to bypass (tests).
    """
    if os.environ.get("PREFLIGHT_ALLOW_ANY_CHECKOUT"):
        return

    def _git(*a):
        return subprocess.run(["git", *a], cwd=repo_root,
                              capture_output=True, text=True).stdout.strip()

    git_dir = os.path.realpath(os.path.join(repo_root, _git("rev-parse", "--git-dir")))
    common = os.path.realpath(os.path.join(repo_root, _git("rev-parse", "--git-common-dir")))
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")

    problems = []
    if git_dir != common:
        problems.append(f"this is a linked worktree ({repo_root})")
    if branch != "modeling-dev":
        problems.append(f"current branch is '{branch}', not 'modeling-dev'")
    if problems:
        print(
            "ERROR: `preflight complete` must run from the main checkout on modeling-dev.\n"
            "  - " + "\n  - ".join(problems) + "\n"
            "  Run it from the main repo instead:\n"
            "    cd /snel/home/cbwash2/cleo\n"
            "    conda run -n dtmodeling python -m modeling.scripts.preflight complete ...\n"
            "  (set PREFLIGHT_ALLOW_ANY_CHECKOUT=1 to bypass for testing).",
            file=sys.stderr,
        )
        sys.exit(1)


def _md_inline(s):
    """Sanitize free text for a single markdown line/cell.

    Collapses newlines (which would inject extra markdown lines / break a table
    row) and escapes pipes (which break table columns).
    """
    return str(s).replace("\r", " ").replace("\n", " ").replace("|", "\\|").strip()


@contextmanager
def _completion_lock(repo_root):
    """Serialize `preflight complete` across all worktrees of this repo.

    Completion mutates the single shared working tree + git index; without a lock
    two concurrent completions race. We hold an advisory ``flock`` on a file in
    the *shared* git dir (``--git-common-dir``), so worktrees contend on the same
    lock. The lock auto-releases when the process exits, even on crash, so there
    are no stale locks.
    """
    common = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=repo_root, capture_output=True, text=True,
    ).stdout.strip()
    common = common if os.path.isabs(common) else os.path.join(repo_root, common)
    lock_path = os.path.join(common, "preflight-complete.lock")

    f = open(lock_path, "w")
    try:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print("[preflight] Another `preflight complete` is in progress — "
                  "waiting for the lock...")
            fcntl.flock(f, fcntl.LOCK_EX)  # block until the holder releases
        f.write(f"{os.getpid()} {datetime.now(timezone.utc).isoformat()}\n")
        f.flush()
        yield
    finally:
        try:
            fcntl.flock(f, fcntl.LOCK_UN)
        finally:
            f.close()


def generate_token(experiment_name, branch, data_file, timestamp):
    payload = f"{experiment_name}|{branch}|{data_file}|{timestamp}"
    return hashlib.sha256(payload.encode()).hexdigest()


# ── Ideas File Management ────────────────────────────────────────────────

VALID_IDEA_FILES = [
    "ideas/modeling.md",
    "ideas/control.md",
    "ideas/project.md",
    "ideas/future.md",
    "ideas/architecture_specs.md",
]


def validate_idea_file(repo_root, idea_file):
    """Check that the idea file exists and is in the allowed list."""
    if idea_file not in VALID_IDEA_FILES:
        print(f"ERROR: --idea-file must be one of: {VALID_IDEA_FILES}", file=sys.stderr)
        sys.exit(1)
    full_path = os.path.join(repo_root, idea_file)
    if not os.path.exists(full_path):
        print(f"ERROR: Idea file not found: {full_path}", file=sys.stderr)
        sys.exit(1)
    return full_path


def find_idea_entry(ideas_path, idea_id):
    """Find an idea/experiment entry by its number. Returns (line_number, header_line) or None."""
    with open(ideas_path) as f:
        lines = f.readlines()
    # Match patterns like "### Experiment 23." or "## 5. Organoid Control"
    patterns = [
        rf"^###?\s*Experiment\s+{idea_id}\b",
        rf"^###?\s*{idea_id}\.\s",
    ]
    for i, line in enumerate(lines):
        for pat in patterns:
            if re.match(pat, line.strip()):
                return i, line.strip()
    return None


def get_max_idea_id(ideas_path):
    """Find the highest experiment/idea number in the file."""
    with open(ideas_path) as f:
        content = f.read()
    # Match "### Experiment N" or "## N."
    numbers = re.findall(r"(?:###?\s*Experiment\s+|##\s+)(\d+)", content)
    if not numbers:
        return 0
    return max(int(n) for n in numbers)


def mark_idea_in_progress(ideas_path, idea_id, branch, worktree_path, results_dir):
    """Mark an idea entry as IN PROGRESS and add cross-references."""
    with open(ideas_path) as f:
        lines = f.readlines()

    entry = find_idea_entry(ideas_path, idea_id)
    if entry is None:
        print(f"ERROR: Idea #{idea_id} not found in {ideas_path}", file=sys.stderr)
        sys.exit(1)

    line_idx, header = entry

    # Check it's not already completed
    # Look at lines after the header for status
    for j in range(line_idx + 1, min(line_idx + 5, len(lines))):
        if "COMPLETE" in lines[j] and "IN PROGRESS" not in lines[j]:
            print(f"ERROR: Idea #{idea_id} is already marked COMPLETE", file=sys.stderr)
            sys.exit(1)

    # Find where to insert the status (after the header line)
    insert_idx = line_idx + 1
    # Update Status, Branch/Worktree, and Results dir lines
    wt_short = os.path.basename(worktree_path) if worktree_path else "(main)"
    has_status = False
    for j in range(line_idx + 1, min(line_idx + 8, len(lines))):
        stripped = lines[j].strip()
        if stripped.startswith("**Status**"):
            lines[j] = f"**Status**: \U0001f504 IN PROGRESS\n"
            has_status = True
        elif stripped.startswith("**Branch/Worktree**"):
            lines[j] = f"**Branch/Worktree**: `{branch}` / `cleo-worktrees/{wt_short}`\n"
        elif stripped.startswith("**Results dir**"):
            lines[j] = f"**Results dir**: `{results_dir}/`\n"
        elif stripped.startswith("###") or stripped.startswith("## "):
            break

    if not has_status:
        status_block = (
            f"**Status**: \U0001f504 IN PROGRESS\n"
            f"**Branch/Worktree**: `{branch}` / `cleo-worktrees/{wt_short}`\n"
            f"**Results dir**: `{results_dir}/`\n"
        )
        lines.insert(insert_idx, status_block)

    with open(ideas_path, "w") as f:
        f.writelines(lines)

    print(f"[preflight] ✓ Marked idea #{idea_id} as 🔄 IN PROGRESS")


def complete_idea(ideas_path, idea_id, status, best_r2, notes):
    """Mark an idea as completed with results annotation."""
    with open(ideas_path) as f:
        lines = f.readlines()

    entry = find_idea_entry(ideas_path, idea_id)
    if entry is None:
        print(f"ERROR: Idea #{idea_id} not found in {ideas_path}", file=sys.stderr)
        sys.exit(1)

    line_idx, header = entry

    status_emoji = "✅ COMPLETE" if status == "completed" else "❌ DID NOT BEAT BASELINE"
    r2_str = f"R²={best_r2:.4f}" if best_r2 is not None else ""

    for j in range(line_idx + 1, min(line_idx + 10, len(lines))):
        if lines[j].strip().startswith("**Status**"):
            lines[j] = f"**Status**: {status_emoji} — {r2_str}\n"
            break
    else:
        lines.insert(line_idx + 1, f"**Status**: {status_emoji} — {r2_str}\n")

    # Add notes after the status block
    if notes:
        for j in range(line_idx + 1, min(line_idx + 15, len(lines))):
            if lines[j].strip().startswith("**Status**"):
                lines.insert(j + 1, f"**Results notes**: {_md_inline(notes)}\n")
                break

    with open(ideas_path, "w") as f:
        f.writelines(lines)

    print(f"[preflight] ✓ Marked idea #{idea_id} as {status_emoji}")


# ── Branches.md Management ───────────────────────────────────────────────

def update_branches_md(repo_root, branch, worktree_path, experiment_name,
                       idea_id, idea_file, results_dir, status="Preflight complete"):
    """Append a row to branches.md with mutual cross-references."""
    branches_path = os.path.join(repo_root, "branches.md")
    with open(branches_path) as f:
        content = f.read()

    wt_short = os.path.basename(worktree_path) if worktree_path else "(main)"
    new_row = (
        f"| {branch} | cleo-worktrees/{wt_short} | "
        f"[Exp {idea_id}]({idea_file}) | "
        f"`{results_dir}/` | {status} |"
    )

    lines = content.split("\n")
    insert_idx = None
    in_active = False
    for i, line in enumerate(lines):
        if "## Active Experiments" in line:
            in_active = True
        elif in_active and line.startswith("|"):
            insert_idx = i + 1
        elif in_active and not line.startswith("|") and insert_idx is not None:
            break

    if insert_idx is not None:
        lines.insert(insert_idx, new_row)
        with open(branches_path, "w") as f:
            f.write("\n".join(lines))
        print(f"[preflight] ✓ Updated branches.md — Exp {idea_id} → {results_dir}/")
    else:
        print("[preflight] WARNING: Could not find Active Experiments table")


def complete_branches_md(repo_root, results_dir, status, best_r2):
    """Update the status column in branches.md for a completed experiment."""
    branches_path = os.path.join(repo_root, "branches.md")
    with open(branches_path) as f:
        lines = f.readlines()

    r2_str = f"R²={best_r2:.4f}" if best_r2 is not None else ""
    status_str = f"Done — {r2_str}" if status == "completed" else f"Failed — {r2_str}"

    for i, line in enumerate(lines):
        if results_dir in line and line.startswith("|"):
            # Replace the last column (status)
            parts = line.split("|")
            if len(parts) >= 6:
                parts[-2] = f" {status_str} "
                lines[i] = "|".join(parts)

    with open(branches_path, "w") as f:
        f.writelines(lines)

    print(f"[preflight] ✓ Updated branches.md status for {results_dir}")


# ── Worktree Management ─────────────────────────────────────────────────

def create_worktree(repo_root, worktree_path, branch, base_branch="modeling-dev"):
    if os.path.exists(worktree_path):
        print(f"[preflight] Worktree already exists: {worktree_path}")
        return
    result = subprocess.run(
        ["git", "branch", "--list", branch],
        capture_output=True, text=True, cwd=repo_root,
    )
    if branch in result.stdout:
        subprocess.run(
            ["git", "worktree", "add", worktree_path, branch],
            check=True, cwd=repo_root,
        )
    else:
        subprocess.run(
            ["git", "worktree", "add", "-b", branch, worktree_path, base_branch],
            check=True, cwd=repo_root,
        )
    print(f"[preflight] ✓ Created worktree: {worktree_path} on {branch}")


# ── MANIFEST.json ────────────────────────────────────────────────────────

def write_manifest(results_dir, manifest_data):
    """Write MANIFEST.json. Refuses if dir has existing results."""
    if os.path.exists(results_dir):
        existing = [f for f in os.listdir(results_dir) if f != "MANIFEST.json"]
        if existing:
            print(f"[preflight] ERROR: Results dir already has content: {results_dir}")
            print(f"[preflight]   Found {len(existing)} files: {existing[:5]}")
            print(f"[preflight]   Use a unique --results-dir to prevent overwriting.")
            sys.exit(1)
    os.makedirs(results_dir, exist_ok=True)
    manifest_path = os.path.join(results_dir, "MANIFEST.json")
    if os.path.exists(manifest_path):
        print(f"[preflight] ERROR: MANIFEST.json already exists at {manifest_path}")
        sys.exit(1)
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"[preflight] ✓ Wrote MANIFEST.json to {results_dir}")


# ── Git ──────────────────────────────────────────────────────────────────

def git_commit(repo_root, message, files):
    subprocess.run(["git", "add"] + files, check=True, cwd=repo_root)
    subprocess.run(["git", "commit", "-m", message], check=True, cwd=repo_root)


def _completion_artifacts(repo_root, results_dir):
    """The artifacts to version with a completion (Option D).

    We version the hand-authored dashboard *base* (``dashboard.html`` — the
    pipeline mutates it in place via markers, so it can't be regenerated from
    scratch) plus the small, diffable JSON facts (leaderboard, plant metadata,
    experiment index, per-experiment validation metrics).

    We do NOT version the heavy *regenerable* renders: the ~MB-each
    ``experiments/*.html`` pages and ``plants/*.png`` images. Those are rebuilt
    from the facts via ``build_dashboard.py`` (they were the bulk of the
    ~MBs-per-commit history bloat). Returns repo-relative paths that exist.
    """
    import glob

    candidates = [
        "results/dashboard.html",
        "results/leaderboard.json",
        "results/plants/plants.json",
        "results/experiments/index.json",
    ]
    candidates += [
        os.path.relpath(p, repo_root)
        for p in glob.glob(os.path.join(results_dir, "**", "val_metrics.json"), recursive=True)
    ]
    return [
        c for c in candidates
        if not c.startswith("..") and os.path.exists(os.path.join(repo_root, c))
    ]


def _commit_completion(repo_root, message, tracked_files, results_dir):
    """Commit the completion: tracked files + force-added artifacts (Option D).

    Heavy regenerable renders are excluded — see :func:`_completion_artifacts`.
    """
    subprocess.run(["git", "add"] + tracked_files, check=True, cwd=repo_root)

    facts = _completion_artifacts(repo_root, results_dir)
    if facts:
        subprocess.run(["git", "add", "-f"] + facts, check=False, cwd=repo_root)

    # Idempotent: if a prior (partial) completion already committed everything,
    # there's nothing staged — don't crash, just continue to push (M3).
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo_root).returncode == 0:
        print("[preflight] (nothing new to commit — completion is already recorded)")
        return
    subprocess.run(["git", "commit", "-m", message], check=True, cwd=repo_root)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preflight harness — experiment lifecycle management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── START subcommand ──
    start = sub.add_parser("start", help="Register a new experiment")
    start.add_argument("--idea-file", required=True, choices=VALID_IDEA_FILES,
                        help="Which ideas file this experiment comes from")
    start.add_argument("--idea-id", type=int, default=None,
                        help="Idea/experiment number (auto-incremented if omitted)")
    start.add_argument("--experiment-name", required=True, help="Descriptive name")
    start.add_argument("--branch", required=True, help="Git branch name")
    start.add_argument("--hypothesis", required=True, help="What you're testing")
    start.add_argument("--data", required=True, help="Path to data file")
    start.add_argument("--machine", required=True, choices=["gpu1", "gpu2"])
    start.add_argument("--gpu-ids", required=True, help="Comma-separated GPU indices")
    start.add_argument("--results-dir", required=True, help="Results directory (must not exist)")
    start.add_argument("--worktree-path", default=None)
    start.add_argument("--base-branch", default="modeling-dev")
    start.add_argument("--model-type", default="latent_canode",
                       choices=["canode", "latent_canode", "latent_node", "gru", "n4sid", "other"],
                       help="Model family — drives the validation-plot inference path at completion")
    start.add_argument("--past-window", type=int, default=200,
                       help="Causal past-window (ms) for latent_canode validation")
    start.add_argument("--dry-run", action="store_true")

    # ── COMPLETE subcommand ──
    complete = sub.add_parser("complete", help="Mark an experiment as finished")
    complete.add_argument("--results-dir", required=True, help="Results directory with MANIFEST.json")
    complete.add_argument("--status", required=True, choices=["completed", "failed", "baseline"],
                          help="Outcome: completed (beat baseline), failed, baseline (did not beat)")
    complete.add_argument("--best-r2", type=float, default=None, help="Best R² achieved")
    complete.add_argument("--best-mse", type=float, default=None, help="Best MSE achieved")
    complete.add_argument("--notes", type=str, default="", help="Summary of findings")

    # ── GPU-STATUS subcommand ──
    gpu_status = sub.add_parser("gpu-status", help="Show GPU availability")
    gpu_status.add_argument("--machine", default="gpu1", choices=["gpu1", "gpu2"])

    # ── MONITOR subcommand ──
    monitor = sub.add_parser("monitor", help="Block until training completes")
    monitor.add_argument("--results-dir", required=True, help="Results directory to watch")
    monitor.add_argument("--poll-interval", type=int, default=30, help="Seconds between checks")
    monitor.add_argument("--timeout", type=int, default=14400, help="Max seconds to wait (default 4hr)")

    args = parser.parse_args()
    repo_root = get_repo_root()

    if args.command == "start":
        _do_start(args, repo_root)
    elif args.command == "complete":
        _do_complete(args, repo_root)
    elif args.command == "gpu-status":
        _do_gpu_status(args)
    elif args.command == "monitor":
        _do_monitor(args, repo_root)


def _do_start(args, repo_root):
    # Validate data file
    data_path = os.path.join(repo_root, args.data) if not os.path.isabs(args.data) else args.data
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    # Validate idea file
    ideas_path = validate_idea_file(repo_root, args.idea_file)

    # Auto-increment or validate idea ID
    if args.idea_id is None:
        args.idea_id = get_max_idea_id(ideas_path) + 1
        print(f"[preflight] Auto-assigned idea ID: {args.idea_id}")
    else:
        entry = find_idea_entry(ideas_path, args.idea_id)
        if entry is None:
            print(f"ERROR: Idea #{args.idea_id} not found in {args.idea_file}")
            print(f"  Add the idea entry first, then run preflight.", file=sys.stderr)
            sys.exit(1)

    # Generate token
    timestamp = datetime.now(timezone.utc).isoformat()
    token = generate_token(args.experiment_name, args.branch, args.data, timestamp)
    results_dir = os.path.join(repo_root, args.results_dir) if not os.path.isabs(args.results_dir) else args.results_dir

    manifest = {
        "preflight_token": token,
        "idea_file": args.idea_file,
        "idea_id": args.idea_id,
        "experiment_name": args.experiment_name,
        "hypothesis": args.hypothesis,
        "branch": args.branch,
        "worktree_path": args.worktree_path,
        "commit_sha": get_commit_sha(),
        "data_file": args.data,
        "model_type": args.model_type,
        "past_window": args.past_window,
        "machine": args.machine,
        "gpu_ids": args.gpu_ids,
        "results_dir": args.results_dir,
        "created_at": timestamp,
        "status": "preflight_complete",
        "completed_at": None,
        "best_r2": None,
    }

    print(f"\n{'='*60}")
    print(f"PREFLIGHT — Exp {args.idea_id}: {args.experiment_name}")
    print(f"{'='*60}")
    print(f"  Idea file:   {args.idea_file}")
    print(f"  Branch:      {args.branch}")
    print(f"  Worktree:    {args.worktree_path or '(none)'}")
    print(f"  Hypothesis:  {args.hypothesis}")
    print(f"  Data:        {args.data}")
    print(f"  Machine:     {args.machine} (GPUs {args.gpu_ids})")
    print(f"  Results:     {args.results_dir}")
    print(f"  Token:       {token[:16]}...")
    print()

    if args.dry_run:
        print("[dry-run] Would: create worktree, update ideas, update branches.md, write MANIFEST, commit.")
        print(f"\n{token}")
        return

    # Execute
    if args.worktree_path:
        create_worktree(repo_root, args.worktree_path, args.branch, args.base_branch)

    mark_idea_in_progress(ideas_path, args.idea_id, args.branch, args.worktree_path, args.results_dir)
    update_branches_md(repo_root, args.branch, args.worktree_path, args.experiment_name,
                       args.idea_id, args.idea_file, args.results_dir)
    write_manifest(results_dir, manifest)
    git_commit(repo_root,
               f"preflight(Exp {args.idea_id}): {args.experiment_name}",
               ["branches.md", args.idea_file])

    # GATE: Verify cross-references were written correctly
    verify_cross_references(repo_root, args.idea_file, args.idea_id,
                           args.branch, args.results_dir)

    print(f"\n{token}")



def _do_complete(args, repo_root):
    _assert_main_checkout(repo_root)
    results_dir = os.path.join(repo_root, args.results_dir) if not os.path.isabs(args.results_dir) else args.results_dir
    manifest_path = os.path.join(results_dir, "MANIFEST.json")

    if not os.path.exists(manifest_path):
        print(f"ERROR: No MANIFEST.json at {manifest_path}. Was preflight run?", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    idea_file = manifest.get("idea_file")
    idea_id = manifest.get("idea_id")

    if not idea_file or not idea_id:
        print(f"ERROR: MANIFEST.json missing idea_file or idea_id fields.", file=sys.stderr)
        sys.exit(1)

    ideas_path = os.path.join(repo_root, idea_file)
    sep = "=" * 60

    # GATE 1 (cheap, no mutation): fail early if eval results are missing.
    if args.status != "failed":
        _require_eval_results(results_dir)

    # Everything past here mutates the shared working tree. Serialize across
    # worktrees (H1), do all mutations, then commit atomically and push — so a
    # re-run after any failure safely resumes (every step is idempotent: M3).
    with _completion_lock(repo_root):
        print()
        print(sep)
        print(f"POSTFLIGHT — Completing Exp {idea_id}: {manifest.get('experiment_name', '?')}")
        print(sep)
        print(f"  Status:  {args.status}")
        print(f"  R²:      {args.best_r2}")
        print(f"  Notes:   {args.notes}")
        print()

        # 1. Mark the experiment complete (idea entry, branches.md, manifest).
        complete_idea(ideas_path, idea_id, args.status, args.best_r2, args.notes)
        complete_branches_md(repo_root, args.results_dir, args.status, args.best_r2)
        manifest["status"] = args.status
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["best_r2"] = args.best_r2
        if args.best_mse is not None:
            manifest["best_mse"] = args.best_mse
        manifest["notes"] = args.notes
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # 2. Regenerate the dashboard from the now-completed manifest (HARD gates
        #    inside), then commit idea/branches/manifest + dashboard atomically
        #    (M2: regen immediately precedes the commit; minimal dirty window).
        _require_dashboard_update(repo_root, results_dir)
        _commit_completion(
            repo_root,
            f"results(Exp {idea_id}): {args.status} — R²={args.best_r2} — {args.notes[:60]}",
            ["branches.md", idea_file],
            results_dir,
        )
        print(f"[preflight] ✓ Experiment {idea_id} marked as {args.status}")

        # 3. Push (fetch + rebase + retry to survive concurrent pushes: H2).
        _require_git_push(repo_root)

        # 4. Merge + remove the experiment worktree.
        worktree = manifest.get("worktree_path")
        if worktree and os.path.exists(worktree):
            _cleanup_worktree(repo_root, worktree)

    print()
    print(f"[preflight] ✓ Experiment {idea_id} fully completed, pushed, and cleaned up.")


def _require_eval_results(results_dir):
    """HARD GATE: Refuse to complete if no eval results exist."""
    found = []
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f in ("results.json", "sweep_summary.json"):
                found.append(os.path.join(root, f))
    if not found:
        print(
            f"ERROR: No eval results found in {results_dir}\n"
            f"  Expected: results.json or sweep_summary.json\n"
            f"  Run evaluation before completing the experiment.\n"
            f"  If this is a failed experiment, use: --status failed",
            file=sys.stderr,
        )
        sys.exit(1)
    basenames = [os.path.relpath(f, results_dir) for f in found[:5]]
    print(f"[preflight] \u2713 Found {len(found)} result file(s): {basenames}")

    # L3: sanity-check the recorded R\u00b2 values so NaN/inf garbage doesn't silently
    # poison the leaderboard. Warn (don't block) \u2014 some result files legitimately
    # omit r2 (e.g. control experiments).
    import math
    bad, n_r2 = [], 0
    for fp in found:
        try:
            with open(fp) as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        records = data if isinstance(data, list) else [data]
        for rec in records:
            if not isinstance(rec, dict):
                continue
            r2 = rec.get("r2", rec.get("r2_200step", rec.get("val_r2")))
            if r2 is None:
                continue
            n_r2 += 1
            try:
                if not math.isfinite(float(r2)):
                    bad.append((os.path.relpath(fp, results_dir), r2))
            except (TypeError, ValueError):
                bad.append((os.path.relpath(fp, results_dir), r2))
    if bad:
        print(f"[preflight] \u26a0\ufe0f  {len(bad)}/{n_r2} result(s) have non-finite R\u00b2 "
              f"(check before trusting the leaderboard): {bad[:3]}")


def _run_dashboard_step(repo_root, module, extra, timeout=900):
    """Run one dashboard pipeline module via -m, with repo_root on PYTHONPATH."""
    env = dict(os.environ)
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", f"modeling.scripts.{module}", *extra],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=timeout,
    )


def _require_dashboard_update(repo_root, results_dir):
    """HARD GATE: regenerate the full dashboard for this experiment.

    Ordered so leaderboard links resolve to freshly-written pages:
      1. validation_plots  --experiment-dir  (best-effort: warn, don't block)
      2. generate_experiment_pages           (HARD: every experiment gets a page)
      3. update_dashboard --plants-only       (idempotent: soft)
      4. update_dashboard_leaderboard --results-dir  (HARD: leaderboard row)

    Validation is best-effort because GRU/N4SID are intentionally inference-less
    and GPU/env hiccups must not block a completion; the page + leaderboard row
    are guaranteed. Requires the dtmodeling env (torch) for step 1.
    """
    # 1. Per-experiment validation plots (best-effort).
    r = _run_dashboard_step(repo_root, "validation_plots", ["--experiment-dir", results_dir])
    if r.returncode != 0:
        print(f"[preflight] WARNING: validation plots step failed (non-blocking):\n"
              f"  {r.stderr[-300:].strip()}")
    else:
        print("[preflight] \u2713 Validation plots generated")

    # 2. Standalone experiment pages (HARD GATE).
    r = _run_dashboard_step(repo_root, "generate_experiment_pages", [])
    if r.returncode != 0:
        print(f"ERROR: Experiment-page generation failed (exit {r.returncode}):\n"
              f"  stdout: {r.stdout[-300:]}\n  stderr: {r.stderr[-300:]}\n"
              f"Fix and re-run preflight complete.", file=sys.stderr)
        sys.exit(1)
    print("[preflight] \u2713 Experiment pages generated")

    # 3. Plants tab (idempotent, soft).
    r = _run_dashboard_step(repo_root, "update_dashboard", ["--plants-only"])
    if r.returncode != 0:
        print(f"[preflight] WARNING: plants-tab injection failed (non-blocking):\n"
              f"  {r.stderr[-300:].strip()}")

    # 4. Leaderboard row + links (HARD GATE).
    r = _run_dashboard_step(repo_root, "update_dashboard_leaderboard", ["--results-dir", results_dir])
    if r.returncode != 0:
        print(f"ERROR: Dashboard leaderboard update failed (exit {r.returncode}):\n"
              f"  stdout: {r.stdout[-300:]}\n  stderr: {r.stderr[-300:]}\n"
              f"Fix the dashboard script and re-run preflight complete.", file=sys.stderr)
        sys.exit(1)
    print("[preflight] \u2713 Dashboard leaderboard updated")


def _require_git_push(repo_root, attempts=3):
    """HARD GATE: fetch + rebase + push, retrying on concurrent-push rejection.

    A bare push races other agents: if origin advanced since we committed, the
    push is rejected non-fast-forward. We instead fetch, rebase our local
    completion commits onto ``origin/modeling-dev``, and push \u2014 retrying the
    whole cycle a few times. A rebase conflict (e.g. two completions touching the
    same file) is aborted and surfaced clearly rather than corrupting the tree.
    """
    def _git(*a, **kw):
        return subprocess.run(["git", *a], cwd=repo_root,
                              capture_output=True, text=True, **kw)

    for attempt in range(1, attempts + 1):
        _git("fetch", "origin", "modeling-dev", timeout=60)

        ahead = _git("rev-list", "--count", "origin/modeling-dev..HEAD")
        behind = _git("rev-list", "--count", "HEAD..origin/modeling-dev")
        n_ahead = int(ahead.stdout.strip()) if ahead.returncode == 0 else 0
        n_behind = int(behind.stdout.strip()) if behind.returncode == 0 else 0

        if n_ahead == 0 and n_behind == 0:
            print("[preflight] \u2713 Already in sync with remote")
            return

        if n_behind > 0:
            print(f"[preflight] origin advanced by {n_behind} commit(s); rebasing ...")
            rb = _git("rebase", "origin/modeling-dev")
            if rb.returncode != 0:
                _git("rebase", "--abort")
                print(
                    "ERROR: Rebase onto origin/modeling-dev hit a conflict "
                    "(likely a concurrent completion touched the same file).\n"
                    f"  {rb.stdout[-300:]}\n{rb.stderr[-300:]}\n"
                    "  Resolve manually (git pull --rebase), then re-run "
                    "`preflight complete` (it is idempotent).",
                    file=sys.stderr,
                )
                sys.exit(1)
            n_ahead = int(_git("rev-list", "--count", "origin/modeling-dev..HEAD").stdout.strip() or 0)

        print(f"[preflight] Pushing {n_ahead} commit(s) to origin/modeling-dev "
              f"(attempt {attempt}/{attempts}) ...")
        push = _git("push", "origin", "modeling-dev", timeout=60)
        if push.returncode == 0:
            print(f"[preflight] \u2713 Pushed {n_ahead} commit(s) to origin/modeling-dev")
            return

        # Rejected \u2014 someone pushed during our window; refetch + retry.
        print(f"[preflight] push rejected (attempt {attempt}); refetching ...\n"
              f"  {push.stderr[-200:].strip()}")
        time.sleep(1.5 * attempt)

    print(
        "ERROR: Git push failed after retries (remote keeps advancing).\n"
        "  Push manually (git pull --rebase && git push), then re-run "
        "`preflight complete`.",
        file=sys.stderr,
    )
    sys.exit(1)


def _cleanup_worktree(repo_root, worktree_path):
    """Auto-remove completed experiment worktree after verifying merge."""
    # Check what branch the worktree is on
    result = subprocess.run(
        ["git", "-C", worktree_path, "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True,
    )
    branch = result.stdout.strip() if result.returncode == 0 else None

    # Check for uncommitted changes
    result = subprocess.run(
        ["git", "-C", worktree_path, "status", "--porcelain"],
        capture_output=True, text=True,
    )
    has_changes = bool(result.stdout.strip())

    if has_changes:
        print(f"[preflight] WARNING: Worktree has uncommitted changes, skipping cleanup")
        print(f"  Path: {worktree_path}")
        print(f"  Commit or stash changes, then run: git worktree remove {worktree_path}")
        return

    # Check if branch is merged into modeling-dev
    if branch:
        result = subprocess.run(
            ["git", "branch", "--merged", "modeling-dev"],
            capture_output=True, text=True, cwd=repo_root,
        )
        merged_branches = [b.strip().lstrip("* ") for b in result.stdout.strip().split("\n")]
        if branch not in merged_branches:
            # Merge the branch into modeling-dev first
            print(f"[preflight] Merging {branch} into modeling-dev before cleanup...")
            result = subprocess.run(
                ["git", "merge", branch, "--no-edit",
                 "-m", f"merge: {branch} (experiment complete)"],
                capture_output=True, text=True, cwd=repo_root,
            )
            if result.returncode != 0:
                print(f"[preflight] WARNING: Merge failed: {result.stderr.strip()}")
                print(f"  Resolve manually, then: git worktree remove {worktree_path}")
                return
            print(f"[preflight] \u2713 Merged {branch} into modeling-dev")

    # Remove the worktree
    print(f"[preflight] Cleaning up worktree: {worktree_path}")
    result = subprocess.run(
        ["git", "worktree", "remove", "--force", worktree_path],
        capture_output=True, text=True, cwd=repo_root,
    )
    if result.returncode != 0:
        print(f"[preflight] WARNING: Worktree removal failed: {result.stderr.strip()}")
        print(f"  Manual cleanup: git worktree remove {worktree_path}")
    else:
        print(f"[preflight] \u2713 Worktree removed: {worktree_path}")

    # Delete the remote-tracking branch
    if branch and branch != "modeling-dev":
        subprocess.run(
            ["git", "branch", "-d", branch],
            capture_output=True, text=True, cwd=repo_root,
        )
        print(f"[preflight] \u2713 Deleted local branch: {branch}")



def verify_cross_references(repo_root, idea_file, idea_id, branch, results_dir):
    """HARD GATE: Verify branches.md and idea file mutually reference each other.
    
    Called after both files are updated. If either is missing the
    cross-reference, something went wrong and we abort.
    """
    # Check idea file references the branch
    ideas_path = os.path.join(repo_root, idea_file)
    with open(ideas_path) as f:
        idea_content = f.read()
    
    # Find the idea entry
    pattern = rf"### Experiment {idea_id}\b"
    match = re.search(pattern, idea_content)
    if not match:
        print(f"ERROR: Idea #{idea_id} not found in {idea_file} after update", file=sys.stderr)
        sys.exit(1)
    
    # Get the text until next experiment or end
    rest = idea_content[match.start():]
    next_exp = re.search(r"\n### Experiment \d+", rest[10:])
    entry_text = rest[:next_exp.start() + 10] if next_exp else rest
    
    if branch not in entry_text:
        print(f"ERROR: Idea #{idea_id} in {idea_file} does not reference branch '{branch}'", file=sys.stderr)
        print(f"  Entry text:\n{entry_text[:200]}", file=sys.stderr)
        sys.exit(1)
    
    # Check branches.md references the idea file and experiment
    branches_path = os.path.join(repo_root, "branches.md")
    with open(branches_path) as f:
        branches_content = f.read()
    
    # Look for a row containing both the branch name and the idea reference
    found_branch_row = False
    for line in branches_content.split("\n"):
        if branch in line and f"Exp {idea_id}" in line:
            found_branch_row = True
            # Also verify results dir is in the row
            if results_dir and results_dir not in line:
                print(f"ERROR: branches.md row for {branch} missing results dir '{results_dir}'", file=sys.stderr)
                sys.exit(1)
            break
    
    if not found_branch_row:
        print(f"ERROR: branches.md has no row linking '{branch}' to Exp {idea_id}", file=sys.stderr)
        sys.exit(1)
    
    print(f"[preflight] ✓ Cross-references verified: {idea_file} ↔ branches.md")


def _do_gpu_status(args):
    """Show GPU availability with process info."""
    import re as _re

    print(f"\nGPU Status — {args.machine}")
    print("=" * 60)

    # Get GPU info
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("ERROR: nvidia-smi failed", file=sys.stderr)
        sys.exit(1)

    gpus = []
    for line in result.stdout.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        idx, mem_used, mem_total, util = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        gpus.append({"idx": idx, "mem_used": mem_used, "mem_total": mem_total, "util": util})

    # Get process info per GPU
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    # Also get GPU UUID to index mapping
    result2 = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    uuid_to_idx = {}
    if result2.returncode == 0:
        for line in result2.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2:
                uuid_to_idx[parts[1]] = int(parts[0])

    gpu_procs = {g["idx"]: [] for g in gpus}
    if result.returncode == 0 and result.stdout.strip():
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                uuid, pid, mem, proc = parts[0], parts[1], parts[2], parts[3]
                gpu_idx = uuid_to_idx.get(uuid, -1)
                if gpu_idx in gpu_procs:
                    gpu_procs[gpu_idx].append({"pid": pid, "mem": mem, "proc": os.path.basename(proc)})

    # Check MANIFEST files to map PIDs to experiments
    free_gpus = []
    busy_gpus = []
    for g in gpus:
        idx = g["idx"]
        procs = gpu_procs.get(idx, [])
        status = "FREE" if g["mem_used"] < 100 and g["util"] < 5 else "BUSY"
        proc_str = ""
        if procs:
            proc_str = " | ".join(f"pid={p['pid']} {p['proc']} ({p['mem']}MiB)" for p in procs)
        else:
            proc_str = "no compute processes"

        icon = "\u2705" if status == "FREE" else "\u274c"
        print(f"  GPU {idx}: {icon} {status}  ({g['mem_used']}/{g['mem_total']} MiB, {g['util']}% util)")
        if procs:
            for p in procs:
                print(f"         \u2514\u2500 pid {p['pid']}: {p['proc']} ({p['mem']} MiB)")

        if status == "FREE":
            free_gpus.append(idx)
        else:
            busy_gpus.append(idx)

    print(f"\n  Free: {len(free_gpus)} GPUs {free_gpus}")
    print(f"  Busy: {len(busy_gpus)} GPUs {busy_gpus}")
    if free_gpus:
        print(f"  --gpu-ids {','.join(str(g) for g in free_gpus)}")


def _do_monitor(args, repo_root):
    """Block until all training runs in results-dir produce results.json."""
    import time as _time
    import glob as _glob

    results_dir = os.path.join(repo_root, args.results_dir) if not os.path.isabs(args.results_dir) else args.results_dir

    if not os.path.exists(results_dir):
        print(f"ERROR: Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    manifest_path = os.path.join(results_dir, "MANIFEST.json")
    if not os.path.exists(manifest_path):
        print(f"ERROR: No MANIFEST.json in {results_dir}", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"\nMONITOR \u2014 Watching Exp {manifest.get('idea_id', '?')}: {manifest.get('experiment_name', '?')}")
    print(f"  Results dir: {results_dir}")
    print(f"  Poll interval: {args.poll_interval}s")
    print(f"  Timeout: {args.timeout}s ({args.timeout/3600:.1f}h)")
    print()

    start_time = _time.time()
    completed = set()

    while True:
        elapsed = _time.time() - start_time
        if elapsed > args.timeout:
            print(f"\nTIMEOUT after {elapsed/3600:.1f}h")
            sys.exit(1)

        # Find all run_* subdirectories that contain actual training output
        # (filter out empty dirs from failed launches)
        all_run_dirs = sorted(_glob.glob(os.path.join(results_dir, "run_*")))
        run_dirs = [d for d in all_run_dirs if os.path.isdir(d) and len(os.listdir(d)) > 0]
        if not run_dirs:
            # Maybe it's a single-run experiment, check for results.json in root
            if os.path.exists(os.path.join(results_dir, "results.json")):
                print(f"[{elapsed/60:.0f}m] \u2713 Training complete!")
                with open(os.path.join(results_dir, "results.json")) as f:
                    r = json.load(f)
                r2 = r.get("r2", r.get("r2_200step", r.get("val_r2", r.get("best_r2", "?"))))
                print(f"  R\u00b2 = {r2}")
                return
            # No runs yet, keep waiting
            _time.sleep(args.poll_interval)
            continue

        # Check each run dir for results.json
        all_done = True
        for rd in run_dirs:
            run_name = os.path.basename(rd)
            rpath = os.path.join(rd, "results.json")
            if os.path.exists(rpath):
                if run_name not in completed:
                    completed.add(run_name)
                    try:
                        with open(rpath) as f:
                            r = json.load(f)
                        r2 = r.get("r2", r.get("r2_200step", r.get("val_r2", r.get("best_r2", "?"))))
                        mse = r.get("mse", r.get("best_val_loss", r.get("best_mse", "?")))
                        print(f"[{elapsed/60:.0f}m] \u2713 {run_name}: R\u00b2={r2}, MSE={mse}")
                    except Exception:
                        print(f"[{elapsed/60:.0f}m] \u2713 {run_name}: results.json found (parse error)")
            else:
                all_done = False

        if all_done and len(completed) == len(run_dirs):
            print(f"\n{'='*60}")
            print(f"All {len(run_dirs)} runs complete in {elapsed/60:.1f} minutes!")

            # Find best run
            best_r2 = -1
            best_run = None
            for rd in run_dirs:
                rpath = os.path.join(rd, "results.json")
                if os.path.exists(rpath):
                    try:
                        with open(rpath) as f:
                            r = json.load(f)
                        r2 = float(r.get("r2", r.get("r2_200step", r.get("val_r2", r.get("best_r2", -1)))))
                        if r2 > best_r2:
                            best_r2 = r2
                            best_run = os.path.basename(rd)
                    except Exception:
                        pass

            if best_run:
                print(f"Best run: {best_run} (R\u00b2={best_r2:.4f})")
                print(f"\nReady for postflight:")
                print(f"  python -m modeling.scripts.preflight complete \\")
                print(f"    --results-dir {args.results_dir} \\")
                print(f"    --status completed \\")
                print(f"    --best-r2 {best_r2:.4f} \\")
                print(f'    --notes "Best: {best_run}"')
            return

        # Print progress
        n_done = len(completed)
        n_total = len(run_dirs)
        if n_total > 0:
            pct = n_done / n_total * 100
            remaining = n_total - n_done
            bar = "\u2588" * int(pct / 5) + "\u2591" * (20 - int(pct / 5))
            print(f"\r[{elapsed/60:.0f}m] {bar} {n_done}/{n_total} ({pct:.0f}%) — {remaining} running", end="", flush=True)

        _time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
