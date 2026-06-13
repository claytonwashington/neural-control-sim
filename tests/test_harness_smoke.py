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
from modeling.scripts.preflight import _md_inline  # noqa: E402

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
