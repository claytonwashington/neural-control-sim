"""Smoke tests for the experiment harness (preflight + dashboard pipeline).

These guard against regressions in the parts that are easy to break silently —
HTML/markdown escaping, manifest parsing, slug stability, and the preflight CLI.
They deliberately avoid torch so they run in the lightweight `cleo` env:

    conda run -n cleo pytest tests/test_harness_smoke.py

(`validation_plots` is intentionally *not* imported — it pulls in torch.)
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Make `import modeling...` resolve to THIS checkout, not whatever the
# conda cleosim.pth points at.
sys.path.insert(0, str(REPO))

from modeling.scripts.dashboard_common import manifest_to_entry, slugify  # noqa: E402
from modeling.scripts.generate_experiment_pages import build_page  # noqa: E402
from modeling.scripts.preflight import (  # noqa: E402
    _completion_artifacts,
    _completion_lock,
    _md_inline,
)

MALICIOUS = '<script>alert(1)</script> & "q" |pipe'


def test_slugify_stable_and_safe():
    assert slugify("Causal Latent CA-NODE (z=64) 🆕") == "causal-latent-ca-node-z-64"
    # idempotent + filesystem-safe
    s = slugify(MALICIOUS)
    assert "/" not in s and "<" not in s and " " not in s
    assert slugify(s) == s


def test_md_inline_sanitizes_pipes_and_newlines():
    out = _md_inline("line1\nline2 | col")
    assert "\n" not in out
    assert "|" not in out.replace("\\|", "")  # only escaped pipes remain


def test_completion_artifacts_versions_base_and_facts_not_heavy_renders(tmp_path):
    """Option D: version dashboard.html (the base template) + JSON facts, but
    NOT the regenerable experiment pages / plant PNGs."""
    r = tmp_path / "results"
    (r / "plants").mkdir(parents=True)
    (r / "experiments").mkdir(parents=True)
    exp = r / "myexp" / "run_00"
    exp.mkdir(parents=True)
    # versioned: base template + JSON facts
    (r / "dashboard.html").write_text("<html></html>")
    (r / "leaderboard.json").write_text("[]")
    (r / "plants" / "plants.json").write_text("[]")
    (r / "experiments" / "index.json").write_text("[]")
    (exp / "val_metrics.json").write_text("{}")
    # regenerable: must be excluded
    (r / "experiments" / "foo.html").write_text("x")
    (r / "plants" / "p_setup.png").write_bytes(b"x")

    arts = _completion_artifacts(str(tmp_path), str(r / "myexp"))

    assert "results/dashboard.html" in arts
    assert "results/leaderboard.json" in arts
    assert "results/plants/plants.json" in arts
    assert "results/experiments/index.json" in arts
    assert any(a.endswith("val_metrics.json") for a in arts)
    # no heavy renders
    assert not any(a.endswith("_setup.png") for a in arts)
    assert not any(a.endswith(".html") and "experiments/" in a for a in arts)


def test_completion_lock_is_mutually_exclusive(tmp_path):
    """H1: while the completion lock is held, a second acquirer must fail (and
    the lock must be reusable after release)."""
    import fcntl

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)

    with _completion_lock(str(tmp_path)):
        common = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=tmp_path, capture_output=True, text=True,
        ).stdout.strip()
        common = common if os.path.isabs(common) else os.path.join(str(tmp_path), common)
        f2 = open(os.path.join(common, "preflight-complete.lock"), "w")
        with pytest.raises(OSError):  # held -> non-blocking acquire raises
            fcntl.flock(f2, fcntl.LOCK_EX | fcntl.LOCK_NB)
        f2.close()

    # released -> acquirable again
    with _completion_lock(str(tmp_path)):
        pass


def test_experiment_page_escapes_injection():
    """A malicious experiment name must not survive as raw HTML (H3)."""
    entry = {
        "name": MALICIOUS,
        "checkpoint_dir": None,           # metadata-only -> no file I/O
        "model_type": "<b>canode</b>",
        "data_file": "data/x.h5",
        "plant_id": "does_not_exist",     # -> no plant card
        "hypothesis": "drives rate <down> & up",
    }
    html = build_page(entry, {})
    assert "<script>alert(1)</script>" not in html      # not raw
    assert "&lt;script&gt;" in html                     # escaped form present
    assert "<b>canode</b>" not in html                  # model_type escaped too
    assert "drives rate <down>" not in html             # hypothesis escaped


def test_manifest_to_entry(tmp_path):
    exp = tmp_path / "my_exp"
    run = exp / "run_00"
    run.mkdir(parents=True)
    (exp / "MANIFEST.json").write_text(json.dumps({
        "experiment_name": "My Exp", "data_file": "data/training_trials.h5",
        "model_type": "canode", "idea_id": 99, "past_window": 150,
    }))
    (run / "results.json").write_text(json.dumps({"r2": 0.87}))
    (run / "model.pt").write_bytes(b"")  # presence is enough for selection

    entry = manifest_to_entry(exp)
    assert entry is not None
    assert entry["name"] == "My Exp"
    assert entry["model_type"] == "canode"
    assert entry["past_window"] == 150
    assert entry["checkpoint_dir"] == "my_exp/run_00"
    # no manifest -> None
    assert manifest_to_entry(tmp_path / "nope") is None


def test_preflight_start_dry_run_accepts_model_type():
    """The CLI parses --model-type and a dry-run start exits cleanly."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env["PREFLIGHT_ALLOW_LAGGARDS"] = "1"  # bypass the laggard gate; we're testing arg parsing
    r = subprocess.run(
        [sys.executable, "-m", "modeling.scripts.preflight", "start",
         "--idea-file", "ideas/modeling.md", "--idea-id", "23",
         "--experiment-name", "smoke", "--branch", "feature/smoke",
         "--hypothesis", "h", "--data", "data/training_trials.h5",
         "--machine", "gpu1", "--gpu-ids", "0",
         "--results-dir", "results/__smoke_does_not_exist__",
         "--model-type", "canode", "--dry-run"],
        cwd=str(REPO), env=env, capture_output=True, text=True,
    )
    if "Data file not found" in r.stderr:
        pytest.skip("dataset not present in this checkout")
    assert r.returncode == 0, r.stderr
    assert "dry-run" in (r.stdout + r.stderr).lower()


def test_idea_status_classifies_laggards(tmp_path):
    """The laggard detector flags WIP + untracked, but not done / never-started."""
    from modeling.scripts.idea_status import scan

    (tmp_path / "ideas").mkdir()
    (tmp_path / "ideas" / "modeling.md").write_text(
        "### Experiment 1. Done thing\n**Status**: ✅ COMPLETE\n\n"
        "### Experiment 2. Wip thing\n**Status**: 🔄 IN PROGRESS\n\n"
        "### Experiment 3. Idle thing\n**Status**: Not started\n")
    ghost = tmp_path / "results" / "ghost"
    ghost.mkdir(parents=True)
    (ghost / "MANIFEST.json").write_text(json.dumps({
        "idea_file": "ideas/modeling.md", "idea_id": 9,
        "experiment_name": "Ghost run", "status": "completed",
    }))

    laggards = scan(root=str(tmp_path), include_worktrees=False)
    kinds = {(d["idea_id"], d["kind"]) for d in laggards}
    assert (2, "IN_PROGRESS") in kinds                 # WIP flagged
    assert (9, "UNTRACKED") in kinds                   # ran but never registered
    assert not any(d["idea_id"] == 1 for d in laggards)  # completed not flagged
    assert not any(d["idea_id"] == 3 for d in laggards)  # never-started backlog not flagged


def test_all_training_scripts_enforce_preflight():
    """Coverage guard for the mandatory-launch gate.

    The launch/token enforcement lives in `validate_preflight`, so it only bites
    scripts that actually call it. A new `fit_*`/`sweep_*` that forgets the call
    would start training without the provenance + token check — a silent hole.
    This statically asserts every training entrypoint calls `validate_preflight`
    (as a bare name or `module.validate_preflight`), so the hole can't be added.
    """
    import ast

    scripts = sorted((REPO / "modeling" / "scripts").glob("fit_*.py")) + \
        sorted((REPO / "modeling" / "scripts").glob("sweep_*.py"))
    assert scripts, "found no fit_*/sweep_* scripts — glob or layout changed?"

    missing = []
    for path in scripts:
        tree = ast.parse(path.read_text(), filename=str(path))
        called = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Name):
                    called.add(f.id)
                elif isinstance(f, ast.Attribute):
                    called.add(f.attr)
        if "validate_preflight" not in called:
            missing.append(path.name)

    assert not missing, (
        "these training scripts never call validate_preflight, so they bypass the "
        f"preflight token + mandatory-launch gate: {missing}. Add "
        "`validate_preflight(args)` after parsing args (see preflight_check.py)."
    )


def test_idea_status_flags_unmerged_branch(tmp_path):
    """A completed experiment whose feature branch never landed on modeling-dev is
    UNMERGED; an experiment whose branch IS merged is not."""
    import subprocess

    from modeling.scripts.idea_status import scan

    def git(*args):
        subprocess.run(["git", *args], cwd=str(tmp_path), check=True,
                       capture_output=True, text=True)

    git("init", "-q")
    git("config", "user.email", "t@t.t")
    git("config", "user.name", "t")
    (tmp_path / "ideas").mkdir()
    (tmp_path / "ideas" / "modeling.md").write_text(
        "### Experiment 5. Unmerged thing\n**Status**: ✅ COMPLETE\n\n"
        "### Experiment 6. Merged thing\n**Status**: ✅ COMPLETE\n")
    (tmp_path / "seed.txt").write_text("base")
    git("add", "-A")
    git("commit", "-qm", "base")
    git("branch", "-M", "modeling-dev")          # works on old git without init -b

    # Exp 6: a feature branch that gets merged back (the happy path).
    git("checkout", "-q", "-b", "feature/merged")
    (tmp_path / "f6.txt").write_text("x")
    git("add", "-A")
    git("commit", "-qm", "f6")
    git("checkout", "-q", "modeling-dev")
    git("merge", "-q", "--no-ff", "-m", "merge f6", "feature/merged")

    # Exp 5: a feature branch with an unmerged commit (the failure path).
    git("checkout", "-q", "-b", "feature/unmerged")
    (tmp_path / "f5.txt").write_text("y")
    git("add", "-A")
    git("commit", "-qm", "f5")
    git("checkout", "-q", "modeling-dev")

    for iid, branch in ((5, "feature/unmerged"), (6, "feature/merged")):
        d = tmp_path / "results" / f"exp{iid}"
        d.mkdir(parents=True)
        (d / "MANIFEST.json").write_text(json.dumps({
            "idea_file": "ideas/modeling.md", "idea_id": iid,
            "experiment_name": f"Exp {iid}", "status": "completed", "branch": branch,
        }))

    laggards = scan(root=str(tmp_path), include_worktrees=False)
    kinds = {(d["idea_id"], d["kind"]) for d in laggards}
    assert (5, "UNMERGED") in kinds                    # committed in branch, never landed
    assert not any(d["idea_id"] == 6 for d in laggards)  # merged branch is clean
