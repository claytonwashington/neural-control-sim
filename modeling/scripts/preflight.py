#!/usr/bin/env python
"""Preflight harness for experiment registration.

Must be called before any training run. Creates worktrees, registers experiments
in branches.md and modeling_ideas.md, and produces a MANIFEST.json with a
preflight token that training scripts require.

Usage:
    python -m modeling.scripts.preflight \
      --experiment-name "Bidirectional Causal Sweep" \
      --branch feature/bidirectional-plant \
      --hypothesis "Causal CA-NODE on 2-input bidirectional plant" \
      --data data/training_trials_bidirectional.h5 \
      --machine gpu1 \
      --gpu-ids 2,3,4,5 \
      --results-dir results/bidir_sweep_causal \
      [--worktree-path /snel/home/cbwash2/cleo-worktrees/bidir-causal] \
      [--base-branch modeling-dev] \
      [--experiment-id 23]  # auto-incremented if not given
      [--dry-run]  # print plan without executing
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone


def get_repo_root():
    """Get the git repo root directory."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def get_commit_sha():
    """Get current HEAD commit SHA."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def auto_increment_experiment_id(repo_root):
    """Parse modeling_ideas.md for the highest experiment number and return max+1."""
    ideas_path = os.path.join(repo_root, "docs", "modeling_ideas.md")
    if not os.path.exists(ideas_path):
        return 1
    with open(ideas_path) as f:
        content = f.read()
    numbers = re.findall(r"### Experiment (\d+)", content)
    if not numbers:
        return 1
    return max(int(n) for n in numbers) + 1


def generate_token(experiment_name, branch, data_file, timestamp):
    """Generate SHA256 preflight token from experiment metadata."""
    payload = f"{experiment_name}|{branch}|{data_file}|{timestamp}"
    return hashlib.sha256(payload.encode()).hexdigest()


def update_branches_md(repo_root, branch, worktree_path, experiment_name, experiment_id, status="Preflight complete"):
    """Append a row to the Active Experiments table in branches.md."""
    branches_path = os.path.join(repo_root, "branches.md")
    with open(branches_path) as f:
        content = f.read()
    
    # Find the Active Experiments table and append after it
    worktree_short = os.path.basename(worktree_path) if worktree_path else "(main repo)"
    new_row = f"| {branch} | cleo-worktrees/{worktree_short} | Exp {experiment_id}: {experiment_name} | {status} |"
    
    # Find the end of the Active Experiments table (blank line after table rows)
    lines = content.split("\n")
    insert_idx = None
    in_active_table = False
    for i, line in enumerate(lines):
        if "## Active Experiments" in line:
            in_active_table = True
        elif in_active_table and line.startswith("|"):
            insert_idx = i + 1  # After this row
        elif in_active_table and not line.startswith("|") and insert_idx is not None:
            break
    
    if insert_idx is not None:
        lines.insert(insert_idx, new_row)
        with open(branches_path, "w") as f:
            f.write("\n".join(lines))
        print(f"[preflight] ✓ Updated branches.md — added Exp {experiment_id}")
    else:
        print("[preflight] WARNING: Could not find Active Experiments table in branches.md")


def update_modeling_ideas(repo_root, experiment_id, experiment_name, hypothesis, results_dir, branch, worktree_path):
    """Append a stub entry to modeling_ideas.md."""
    ideas_path = os.path.join(repo_root, "docs", "modeling_ideas.md")
    worktree_short = os.path.basename(worktree_path) if worktree_path else "cleo"
    
    stub = f"""
### Experiment {experiment_id}. {experiment_name}
**Status**: 🔄 IN PROGRESS
**Results dir**: `{results_dir}/`
**Branch/Worktree**: `{branch}` / `cleo-worktrees/{worktree_short}`
- Hypothesis: {hypothesis}
"""
    
    with open(ideas_path, "a") as f:
        f.write(stub)
    print(f"[preflight] ✓ Updated modeling_ideas.md — added Exp {experiment_id}")


def create_worktree(repo_root, worktree_path, branch, base_branch="modeling-dev"):
    """Create a git worktree if it doesn't exist."""
    if os.path.exists(worktree_path):
        print(f"[preflight] Worktree already exists: {worktree_path}")
        return
    
    # Check if branch exists
    result = subprocess.run(
        ["git", "branch", "--list", branch],
        capture_output=True, text=True, cwd=repo_root,
    )
    if branch in result.stdout:
        # Branch exists, just create worktree
        subprocess.run(
            ["git", "worktree", "add", worktree_path, branch],
            check=True, cwd=repo_root,
        )
    else:
        # Create new branch from base
        subprocess.run(
            ["git", "worktree", "add", "-b", branch, worktree_path, base_branch],
            check=True, cwd=repo_root,
        )
    print(f"[preflight] ✓ Created worktree: {worktree_path} on {branch}")


def write_manifest(results_dir, manifest_data):
    """Write MANIFEST.json to the results directory."""
    os.makedirs(results_dir, exist_ok=True)
    manifest_path = os.path.join(results_dir, "MANIFEST.json")
    
    if os.path.exists(manifest_path):
        print(f"[preflight] ERROR: MANIFEST.json already exists at {manifest_path}")
        print(f"[preflight] Delete it or use a different --results-dir")
        sys.exit(1)
    
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"[preflight] ✓ Wrote MANIFEST.json to {results_dir}")


def git_commit(repo_root, experiment_name, experiment_id):
    """Commit branches.md and modeling_ideas.md."""
    subprocess.run(
        ["git", "add", "branches.md", "docs/modeling_ideas.md"],
        check=True, cwd=repo_root,
    )
    msg = f"preflight({experiment_name}): registered Exp {experiment_id}"
    subprocess.run(
        ["git", "commit", "-m", msg],
        check=True, cwd=repo_root,
    )
    print(f"[preflight] ✓ Committed registration for Exp {experiment_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Preflight harness — register experiments before training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Example:
  python -m modeling.scripts.preflight \\
    --experiment-name "Bidirectional Causal Sweep" \\
    --branch feature/bidirectional-plant \\
    --hypothesis "Causal CA-NODE on 2-input bidirectional plant" \\
    --data data/training_trials_bidirectional.h5 \\
    --machine gpu1 --gpu-ids 2,3,4,5 \\
    --results-dir results/bidir_sweep_causal
"""
    )
    parser.add_argument("--experiment-name", required=True, help="Descriptive experiment name")
    parser.add_argument("--branch", required=True, help="Git branch (e.g. feature/my-experiment)")
    parser.add_argument("--hypothesis", required=True, help="What you're testing and why")
    parser.add_argument("--data", required=True, help="Path to training data file")
    parser.add_argument("--machine", required=True, choices=["gpu1", "gpu2"], help="Target machine")
    parser.add_argument("--gpu-ids", required=True, help="Comma-separated GPU indices")
    parser.add_argument("--results-dir", required=True, help="Output directory for results")
    parser.add_argument("--worktree-path", default=None, help="Path for git worktree (created if needed)")
    parser.add_argument("--base-branch", default="modeling-dev", help="Base branch for new worktrees")
    parser.add_argument("--experiment-id", type=int, default=None, help="Experiment number (auto-incremented if omitted)")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without executing")
    args = parser.parse_args()

    repo_root = get_repo_root()

    # Validate data file
    data_path = os.path.join(repo_root, args.data) if not os.path.isabs(args.data) else args.data
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    # Auto-increment experiment ID
    exp_id = args.experiment_id or auto_increment_experiment_id(repo_root)

    # Generate timestamp and token
    timestamp = datetime.now(timezone.utc).isoformat()
    token = generate_token(args.experiment_name, args.branch, args.data, timestamp)

    # Resolve results dir
    results_dir = os.path.join(repo_root, args.results_dir) if not os.path.isabs(args.results_dir) else args.results_dir

    manifest = {
        "preflight_token": token,
        "experiment_id": exp_id,
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

    # Print plan
    print(f"\n{'='*60}")
    print(f"PREFLIGHT — Experiment {exp_id}: {args.experiment_name}")
    print(f"{'='*60}")
    print(f"  Branch:      {args.branch}")
    print(f"  Worktree:    {args.worktree_path or '(none)'}")
    print(f"  Hypothesis:  {args.hypothesis}")
    print(f"  Data:        {args.data}")
    print(f"  Machine:     {args.machine} (GPUs {args.gpu_ids})")
    print(f"  Results:     {args.results_dir}")
    print(f"  Token:       {token[:16]}...")
    print()

    if args.dry_run:
        print("[dry-run] Would create worktree, update branches.md, modeling_ideas.md, write MANIFEST.json, and commit.")
        print(f"\n{token}")
        return

    # Execute
    if args.worktree_path:
        create_worktree(repo_root, args.worktree_path, args.branch, args.base_branch)

    update_branches_md(repo_root, args.branch, args.worktree_path or repo_root, args.experiment_name, exp_id)
    update_modeling_ideas(repo_root, exp_id, args.experiment_name, args.hypothesis, args.results_dir, args.branch, args.worktree_path)
    write_manifest(results_dir, manifest)
    git_commit(repo_root, args.experiment_name, exp_id)

    # Print token on last line (for capture by scripts)
    print(f"\n{token}")


if __name__ == "__main__":
    main()
