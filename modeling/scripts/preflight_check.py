"""Preflight token validation for training scripts.

All training scripts (fit_*.py, sweep_*.py) must validate a preflight
token before starting. This ensures every experiment is registered in
branches.md and modeling_ideas.md before it runs.

Usage in training scripts:
    from modeling.scripts.preflight_check import add_preflight_args, validate_preflight

    # In argparse setup:
    add_preflight_args(parser)

    # After parsing args:
    validate_preflight(args)
"""

import json
import os
import sys


def add_preflight_args(parser):
    """Add --preflight-token to an argparse parser."""
    parser.add_argument(
        "--preflight-token",
        type=str,
        default=None,
        help="Preflight token from `python -m modeling.scripts.preflight`. Required.",
    )


def validate_preflight(args):
    """Validate that the preflight token matches the MANIFEST.json in output-dir.

    Raises SystemExit if:
    - No --preflight-token is given
    - MANIFEST.json doesn't exist in --output-dir
    - Token doesn't match
    """
    output_dir = getattr(args, "output_dir", None)
    token = getattr(args, "preflight_token", None)

    # Also accept token via env var (used by sweep scripts to pass to children)
    if token is None:
        token = os.environ.get("PREFLIGHT_TOKEN")

    if token is None:
        print(
            "ERROR: --preflight-token is required (or set PREFLIGHT_TOKEN env var).\n"
            "  Run preflight first: python -m modeling.scripts.preflight start --help\n"
            "\nThis ensures your experiment is registered in ideas/ "
            "and branches.md before training starts.",
            file=sys.stderr,
        )
        sys.exit(1)

    if output_dir is None:
        print("ERROR: Cannot validate preflight without --output-dir.", file=sys.stderr)
        sys.exit(1)

    # Search for MANIFEST.json in output_dir or parent dirs (sweep children
    # run in subdirectories of the preflight results dir)
    manifest_path = None
    search_dir = os.path.abspath(output_dir)
    for _ in range(5):  # max 5 levels up
        candidate = os.path.join(search_dir, "MANIFEST.json")
        if os.path.exists(candidate):
            manifest_path = candidate
            break
        parent = os.path.dirname(search_dir)
        if parent == search_dir:
            break
        search_dir = parent

    if manifest_path is None:
        print(
            f"ERROR: No MANIFEST.json found at or above {output_dir}.\n"
            f"Run preflight first: python -m modeling.scripts.preflight start --help",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    expected_token = manifest.get("preflight_token")
    if expected_token != token:
        print(
            f"ERROR: Preflight token mismatch.\n"
            f"  Given:    {token}\n"
            f"  Expected: {expected_token}\n"
            f"Was MANIFEST.json regenerated? Re-run preflight.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Launch provenance ──
    # Every sanctioned run must come through `preflight launch`, which mints a
    # per-experiment nonce into the manifest and injects PREFLIGHT_LAUNCH_NONCE
    # into the session env (children inherit it). A manual `tmux new-session` or
    # bare CLI run won't carry the nonce → refused, so run location is always
    # recorded and `preflight status` can always find it. Escape hatch for
    # debugging/resume: PREFLIGHT_ALLOW_MANUAL=1.
    launch_nonce = manifest.get("launch_nonce")
    env_nonce = os.environ.get("PREFLIGHT_LAUNCH_NONCE")
    if not (launch_nonce and env_nonce == launch_nonce):
        if os.environ.get("PREFLIGHT_ALLOW_MANUAL"):
            print("[preflight] ⚠ Launch check bypassed (PREFLIGHT_ALLOW_MANUAL set); "
                  "run location may not be recorded.", file=sys.stderr)
        else:
            print(
                "ERROR: This run did not come through `preflight launch`.\n"
                "  Start training with:\n"
                "    python -m modeling.scripts.preflight launch \\\n"
                "      --results-dir <dir> --command '<training cmd>'\n"
                "  so host/tmux-session/PID are recorded and `preflight status` can\n"
                "  find it. To start manually anyway (debug/resume): PREFLIGHT_ALLOW_MANUAL=1.",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"[preflight] ✓ Token validated for experiment: {manifest.get('experiment_name', '?')}")
    print(f"[preflight]   Branch: {manifest.get('branch', '?')}")
    print(f"[preflight]   Data: {manifest.get('data_file', '?')}")

    # Self-register runtime location (host/tmux session/pid) so `preflight status`
    # can later say exactly where this run is. Best-effort — never break training.
    try:
        from modeling.scripts.runtime_info import record_run_start
        updated = record_run_start(manifest_path)
        if updated and updated.get("tmux_session"):
            print(f"[preflight]   Registered: host={updated.get('run_host')} "
                  f"tmux={updated.get('tmux_session')} pid={updated.get('run_pid')}")
        manifest = updated or manifest
    except Exception:
        pass
    return manifest


def update_manifest_on_completion(output_dir, best_r2=None, best_mse=None, extra=None):
    """Update MANIFEST.json when training completes."""
    manifest_path = os.path.join(output_dir, "MANIFEST.json")
    if not os.path.exists(manifest_path):
        return

    from datetime import datetime, timezone

    with open(manifest_path) as f:
        manifest = json.load(f)

    manifest["status"] = "completed"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    if best_r2 is not None:
        manifest["best_r2"] = best_r2
    if best_mse is not None:
        manifest["best_mse"] = best_mse
    if extra:
        manifest.update(extra)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[preflight] ✓ MANIFEST.json updated: status=completed, R²={best_r2}")
