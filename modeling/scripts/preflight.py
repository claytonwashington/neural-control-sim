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
import hashlib
import json
import os
import re
import subprocess
import sys
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


def generate_token(experiment_name, branch, data_file, timestamp):
    payload = f"{experiment_name}|{branch}|{data_file}|{timestamp}"
    return hashlib.sha256(payload.encode()).hexdigest()


# ── Ideas File Management ────────────────────────────────────────────────

VALID_IDEA_FILES = [
    "ideas/modeling.md",
    "ideas/control.md",
    "ideas/project.md",
    "ideas/future.md",
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
    # Check if there's already a Status line
    has_status = False
    for j in range(line_idx + 1, min(line_idx + 5, len(lines))):
        if lines[j].strip().startswith("**Status**"):
            lines[j] = f"**Status**: 🔄 IN PROGRESS\n"
            has_status = True
            break
        if lines[j].strip().startswith("###") or lines[j].strip().startswith("## "):
            break

    if not has_status:
        wt_short = os.path.basename(worktree_path) if worktree_path else "(main)"
        status_block = (
            f"**Status**: 🔄 IN PROGRESS\n"
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
                lines.insert(j + 1, f"**Results notes**: {notes}\n")
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

    print(f"\n{token}")



def _do_complete(args, repo_root):
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

    # ═══════════════════════════════════════════════════════════
    # ALL GATES FIRE BEFORE ANY STATE CHANGES
    # If any gate fails, nothing is modified.
    # ═══════════════════════════════════════════════════════════

    # GATE 1: Require eval results (unless --status failed)
    if args.status != "failed":
        _require_eval_results(results_dir)

    # GATE 2: Dashboard update must succeed
    _require_dashboard_update(repo_root, results_dir)

    # ═══════════════════════════════════════════════════════════
    # GATES PASSED — now make state changes
    # ═══════════════════════════════════════════════════════════

    print(f"\n{sep}")
    print(f"POSTFLIGHT \u2014 Completing Exp {idea_id}: {manifest.get('experiment_name', '?')}")
    print(f"{sep}")
    print(f"  Status:  {args.status}")
    print(f"  R\u00b2:     {args.best_r2}")
    print(f"  Notes:   {args.notes}")
    print()

    # Update idea entry
    complete_idea(ideas_path, idea_id, args.status, args.best_r2, args.notes)

    # Update branches.md
    complete_branches_md(repo_root, args.results_dir, args.status, args.best_r2)

    # Update MANIFEST.json
    manifest["status"] = args.status
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["best_r2"] = args.best_r2
    if args.best_mse is not None:
        manifest["best_mse"] = args.best_mse
    manifest["notes"] = args.notes
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Commit
    git_commit(repo_root,
               f"results(Exp {idea_id}): {args.status} \u2014 R\u00b2={args.best_r2} \u2014 {args.notes[:60]}",
               ["branches.md", idea_file])
    print(f"[preflight] \u2713 Experiment {idea_id} marked as {args.status}")

    # GATE 3: Git push (auto-pushes, fails if push fails)
    _require_git_push(repo_root)

    # AUTO 4: Worktree cleanup
    worktree = manifest.get("worktree_path")
    if worktree and os.path.exists(worktree):
        _cleanup_worktree(repo_root, worktree)

    print(f"\n[preflight] \u2713 Experiment {idea_id} fully completed, pushed, and cleaned up.")


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


def _require_dashboard_update(repo_root, results_dir):
    """HARD GATE: Run dashboard update; exit non-zero if it fails."""
    dashboard_script = os.path.join(repo_root, "modeling/scripts/update_dashboard_leaderboard.py")
    if not os.path.exists(dashboard_script):
        print("[preflight] SKIP: Dashboard script not found (non-blocking)")
        return
    result = subprocess.run(
        ["python", dashboard_script, "--results-dir", results_dir],
        cwd=repo_root, capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        print(
            f"ERROR: Dashboard update failed (exit {result.returncode}):\n"
            f"  stdout: {result.stdout[:300]}\n"
            f"  stderr: {result.stderr[:300]}\n"
            f"Fix the dashboard script and re-run preflight complete.",
            file=sys.stderr,
        )
        sys.exit(1)
    print("[preflight] \u2713 Dashboard updated")


def _require_git_push(repo_root):
    """HARD GATE: Auto-push to remote; exit non-zero if push fails."""
    result = subprocess.run(
        ["git", "rev-list", "--count", "origin/modeling-dev..HEAD"],
        capture_output=True, text=True, cwd=repo_root,
    )
    ahead = int(result.stdout.strip()) if result.returncode == 0 else 0
    if ahead == 0:
        print("[preflight] \u2713 Already in sync with remote")
        return
    print(f"[preflight] Pushing {ahead} commit(s) to origin/modeling-dev ...")
    result = subprocess.run(
        ["git", "push", "origin", "modeling-dev"],
        capture_output=True, text=True, cwd=repo_root, timeout=60,
    )
    if result.returncode != 0:
        print(
            f"ERROR: Git push failed:\n  {result.stderr[:500]}\n"
            f"Push manually and re-run preflight complete.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"[preflight] \u2713 Pushed {ahead} commit(s) to origin/modeling-dev")


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

        # Find all run_* subdirectories
        run_dirs = sorted(_glob.glob(os.path.join(results_dir, "run_*")))
        if not run_dirs:
            # Maybe it's a single-run experiment, check for results.json in root
            if os.path.exists(os.path.join(results_dir, "results.json")):
                print(f"[{elapsed/60:.0f}m] \u2713 Training complete!")
                with open(os.path.join(results_dir, "results.json")) as f:
                    r = json.load(f)
                r2 = r.get("r2_200step", r.get("val_r2", r.get("best_r2", "?")))
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
                        r2 = r.get("r2_200step", r.get("val_r2", r.get("best_r2", "?")))
                        mse = r.get("best_val_loss", r.get("best_mse", "?"))
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
                        r2 = float(r.get("r2_200step", r.get("val_r2", r.get("best_r2", -1))))
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
