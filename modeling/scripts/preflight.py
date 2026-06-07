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

    args = parser.parse_args()
    repo_root = get_repo_root()

    if args.command == "start":
        _do_start(args, repo_root)
    elif args.command == "complete":
        _do_complete(args, repo_root)


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

    print(f"\n{'='*60}")
    print(f"POSTFLIGHT — Completing Exp {idea_id}: {manifest.get('experiment_name', '?')}")
    print(f"{'='*60}")
    print(f"  Status:  {args.status}")
    print(f"  R²:     {args.best_r2}")
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

    git_commit(repo_root,
               f"results(Exp {idea_id}): {args.status} — R²={args.best_r2} — {args.notes[:60]}",
               ["branches.md", idea_file])

    print(f"[preflight] ✓ Experiment {idea_id} marked as {args.status}")


if __name__ == "__main__":
    main()
