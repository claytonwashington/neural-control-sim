#!/usr/bin/env python
"""Idea-status / laggard detector for the experiment registry.

Keeps ideas from getting lost. A *laggard* is an experiment that has **started**
(an idea marked 🔄 IN PROGRESS, or a `MANIFEST.json` that isn't completed) — or a
**completed experiment that was never recorded** in `ideas/`. Never-started
backlog ideas are NOT laggards (they're the idea pool).

Kinds:
  * IN_PROGRESS     — idea entry marked 🔄 IN PROGRESS in an ideas file.
  * STALE           — a MANIFEST.json stuck at preflight_complete/running past a
                      grace period (started but never finished).
  * UNTRACKED       — a MANIFEST.json whose (idea_file, idea_id) has no entry in
                      ideas/ (it ran but was never registered).
  * DONE_NOT_CLOSED — an idea marked IN PROGRESS whose results already exist, or a
                      completed MANIFEST whose idea entry isn't ✅ COMPLETE.
  * UNMERGED        — a completed/baseline MANIFEST whose feature branch still exists
                      and was never merged into modeling-dev (results committed in a
                      worktree but never landed). Catches what the status checks miss.

Used by `preflight start` (warn + threshold gate), `preflight audit`, and a Stop
hook. CLI::

    python -m modeling.scripts.idea_status            # full report (exit 1 if laggards)
    python -m modeling.scripts.idea_status --quiet     # one-line summary
    python -m modeling.scripts.idea_status --json       # machine-readable
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

IDEA_FILES = [
    "ideas/modeling.md",
    "ideas/control.md",
    "ideas/project.md",
    "ideas/future.md",
    "ideas/architecture_specs.md",
]
CLOSED_MANIFEST_STATUS = {"completed", "failed", "baseline", "abandoned"}
DEFAULT_GRACE_DAYS = 2
RESULT_MARKERS = ("results.json", "sweep_summary.json", "model.pt", "n4sid_model.npz")


# ── helpers ────────────────────────────────────────────────────────────────
def main_repo() -> str:
    """The MAIN checkout (canonical ideas + results live there, git-ignored), no
    matter which worktree we're invoked from. Derived from --git-common-dir so it
    works from any linked worktree."""
    common = subprocess.run(["git", "rev-parse", "--git-common-dir"],
                            capture_output=True, text=True).stdout.strip()
    if common:
        common = os.path.abspath(common)
        if os.path.basename(common) == ".git":
            return os.path.dirname(common)
    r = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                       capture_output=True, text=True).stdout.strip()
    return r or os.getcwd()


def _git(root: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)


def _branch_exists(root: str, branch: str) -> bool:
    return _git(root, "rev-parse", "--verify", "--quiet", branch).returncode == 0


def _is_merged(root: str, branch: str, into: str = "modeling-dev") -> bool:
    """True if `branch`'s tip is already an ancestor of `into` (i.e. landed)."""
    return _git(root, "merge-base", "--is-ancestor", branch, into).returncode == 0


def _idea_statuses(root: str) -> dict[tuple[str, int], dict]:
    """{(idea_file, idea_id): {name, status}} parsed from the ideas files.

    `ideas/modeling.md` numbers as "### Experiment N"; the backlog files use
    "## N." — and reuse numbers, so we key by (file, id)."""
    out: dict[tuple[str, int], dict] = {}
    for f in IDEA_FILES:
        p = os.path.join(root, f)
        if not os.path.exists(p):
            continue
        txt = open(p).read()
        for m in re.finditer(r"^#{2,3}\s*(?:Experiment\s+)?(\d+)[.\s]\s*(.+)$", txt, re.M):
            iid, name = int(m.group(1)), m.group(2).strip()[:70]
            tail = txt[m.end():m.end() + 500]
            sm = re.search(r"\*\*Status\*\*:?\s*([^\n]*)", tail)
            out[(f, iid)] = {"name": name, "status": (sm.group(1).strip() if sm else "")}
    return out


def _has_results(exp_dir: str) -> bool:
    for _root, _dirs, files in os.walk(exp_dir):
        if any(fn in RESULT_MARKERS for fn in files):
            return True
    return False


def _age_days(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0
    except (ValueError, TypeError):
        return None


def _manifests(root: str, include_worktrees: bool) -> list[dict]:
    paths = glob.glob(os.path.join(root, "results", "*", "MANIFEST.json"))
    if include_worktrees:
        # worktrees live in a sibling dir of the main checkout
        wt = os.path.join(os.path.dirname(os.path.abspath(root)), "cleo-worktrees",
                          "*", "results", "*", "MANIFEST.json")
        paths += glob.glob(wt)
    found = []
    for mp in sorted(set(paths)):
        try:
            j = json.load(open(mp))
        except (OSError, json.JSONDecodeError):
            continue
        exp_dir = os.path.dirname(mp)
        loc = "main" if mp.startswith(os.path.join(root, "results")) else \
            mp.split("cleo-worktrees" + os.sep, 1)[1].split(os.sep)[0]
        # Recover idea_id from the name ("Exp 34: …") when the field is unset, so
        # the manifest links to its idea entry instead of double-counting as UNTRACKED.
        iid = j.get("idea_id")
        if iid is None:
            nm = re.search(r"\b(?:Exp|Experiment)\s+(\d+)", j.get("experiment_name", ""))
            if nm:
                iid = int(nm.group(1))
        found.append({
            "idea_file": j.get("idea_file") or ("ideas/modeling.md" if iid is not None else None),
            "idea_id": iid,
            "name": j.get("experiment_name", os.path.basename(exp_dir)),
            "status": j.get("status"),
            "branch": j.get("branch"),
            "created_at": j.get("created_at"),
            "results_dir": j.get("results_dir", os.path.basename(exp_dir)),
            "location": loc,
            "exp_dir": exp_dir,
            "machine": j.get("machine"),
            "gpu_ids": j.get("gpu_ids"),
            "run_host": j.get("run_host"),
            "tmux_session": j.get("tmux_session"),
            "run_pid": j.get("run_pid"),
            "started_at": j.get("started_at"),
        })
    return found


# ── core scan ──────────────────────────────────────────────────────────────
def scan(root: str | None = None, include_worktrees: bool = True,
         grace_days: float = DEFAULT_GRACE_DAYS) -> list[dict]:
    """Return a deduped list of laggard dicts:
    {kind, idea_file, idea_id, name, status, age_days, location, action}."""
    root = root or main_repo()
    ideas = _idea_statuses(root)
    manifests = _manifests(root, include_worktrees)

    laggards: dict[str, dict] = {}  # keyed by stable id to dedupe

    def add(key, kind, *, idea_file=None, idea_id=None, name="", status="",
            age=None, location="main", action=""):
        # keep the most actionable kind if the same experiment matches twice
        order = {"UNMERGED": 4, "UNTRACKED": 3, "DONE_NOT_CLOSED": 2, "STALE": 1, "IN_PROGRESS": 0}
        if key in laggards and order.get(laggards[key]["kind"], 0) >= order.get(kind, 0):
            return
        laggards[key] = dict(kind=kind, idea_file=idea_file, idea_id=idea_id, name=name,
                             status=status, age_days=age, location=location, action=action)

    # 1) ideas marked IN PROGRESS
    for (f, iid), e in ideas.items():
        s = e["status"]
        if "PROGRESS" in s and "COMPLETE" not in s:
            add((f, iid), "IN_PROGRESS", idea_file=f, idea_id=iid, name=e["name"], status=s,
                action="finish it, then: preflight complete --results-dir results/<dir> ...")

    # 2) manifests
    for m in manifests:
        key = (m["idea_file"], m["idea_id"]) if m["idea_id"] is not None else ("?", m["results_dir"])
        age = _age_days(m["created_at"])
        idea = ideas.get((m["idea_file"], m["idea_id"]))
        if idea is None:
            add(key, "UNTRACKED", idea_file=m["idea_file"], idea_id=m["idea_id"], name=m["name"],
                status=m["status"], age=age, location=m["location"],
                action=f"add an entry to {m['idea_file'] or 'ideas/modeling.md'} for this experiment")
        elif m["status"] not in CLOSED_MANIFEST_STATUS:
            # Liveness-aware: a run whose tmux session / PID is dead is stale
            # regardless of age; otherwise fall back to the grace period.
            from modeling.scripts import runtime_info
            alive = runtime_info.is_alive(m)
            if alive is False:
                add(key, "STALE", idea_file=m["idea_file"], idea_id=m["idea_id"], name=m["name"],
                    status=m["status"], age=age, location=m["location"],
                    action=f"process dead (tmux/{m.get('tmux_session')} pid/{m.get('run_pid')}); "
                           "run `preflight complete` (or abandon) to close it")
            elif alive is not True and (age or 0) >= grace_days:
                add(key, "STALE", idea_file=m["idea_file"], idea_id=m["idea_id"], name=m["name"],
                    status=m["status"], age=age, location=m["location"],
                    action="run `preflight complete` (or --status failed) to close it")
        elif "COMPLETE" not in (idea["status"] or ""):
            add(key, "DONE_NOT_CLOSED", idea_file=m["idea_file"], idea_id=m["idea_id"], name=m["name"],
                status=m["status"], age=age, location=m["location"],
                action=f"mark {m['idea_file']} Exp {m['idea_id']} ✅ COMPLETE (already done)")

        # UNMERGED: a closed (completed/baseline) experiment whose feature branch
        # still exists and was never landed on modeling-dev. This is the blind spot
        # the idea/manifest status checks miss — results committed in a worktree but
        # never merged. Only fires while the branch ref still exists (the happy path
        # merges + deletes it, so merged-and-removed branches never trip this).
        br = m.get("branch")
        if (m["status"] in CLOSED_MANIFEST_STATUS and br
                and _branch_exists(root, br) and not _is_merged(root, br)):
            add(key, "UNMERGED", idea_file=m["idea_file"], idea_id=m["idea_id"], name=m["name"],
                status=m["status"], age=age, location=m["location"],
                action=f"land {br} on modeling-dev: run `preflight complete` (merges + removes the worktree)")

    return sorted(laggards.values(), key=lambda d: (-(d["age_days"] or 0), str(d["idea_id"])))


# ── reporting ──────────────────────────────────────────────────────────────
ICON = {"IN_PROGRESS": "🔄", "STALE": "⏳", "UNTRACKED": "❓", "DONE_NOT_CLOSED": "📌",
        "UNMERGED": "🔀"}


def format_report(laggards: list[dict]) -> str:
    if not laggards:
        return "[idea-status] ✓ No open laggards — the idea registry is clean."
    lines = [f"[idea-status] ⚠️  {len(laggards)} open laggard(s) — close these so ideas don't get lost:"]
    for d in laggards:
        age = f"{d['age_days']:.0f}d" if d["age_days"] is not None else "—"
        eid = f"Exp {d['idea_id']}" if d["idea_id"] is not None else "(unregistered)"
        lines.append(f"  {ICON.get(d['kind'], '•')} {d['kind']:15} {eid:8} {d['name'][:46]:46} "
                     f"[{d['location']}, {age}]")
        lines.append(f"       → {d['action']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Detect laggard experiments in the idea registry")
    ap.add_argument("--quiet", action="store_true", help="one-line summary only")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--no-worktrees", action="store_true", help="scan only the main checkout")
    ap.add_argument("--grace-days", type=float, default=DEFAULT_GRACE_DAYS)
    args = ap.parse_args()

    laggards = scan(include_worktrees=not args.no_worktrees, grace_days=args.grace_days)
    if args.json:
        print(json.dumps(laggards, indent=2, default=str))
    elif args.quiet:
        print(f"[idea-status] {len(laggards)} open laggard(s)"
              + (": " + ", ".join(f"Exp {d['idea_id']}" if d['idea_id'] else d['name'][:20]
                                   for d in laggards[:6]) if laggards else ""))
    else:
        print(format_report(laggards))
    sys.exit(1 if laggards else 0)


if __name__ == "__main__":
    main()
