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

    if token is None:
        print(
            "ERROR: --preflight-token is required. Run preflight first:\n"
            "  python -m modeling.scripts.preflight --help\n"
            "\nThis ensures your experiment is registered in branches.md "
            "and modeling_ideas.md before training starts.",
            file=sys.stderr,
        )
        sys.exit(1)

    if output_dir is None:
        print("ERROR: Cannot validate preflight without --output-dir.", file=sys.stderr)
        sys.exit(1)

    manifest_path = os.path.join(output_dir, "MANIFEST.json")
    if not os.path.exists(manifest_path):
        print(
            f"ERROR: No MANIFEST.json found at {manifest_path}.\n"
            f"Run preflight first to create it:\n"
            f"  python -m modeling.scripts.preflight \\\n"
            f"    --results-dir {output_dir} ...",
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

    print(f"[preflight] ✓ Token validated for experiment: {manifest.get('experiment_name', '?')}")
    print(f"[preflight]   Branch: {manifest.get('branch', '?')}")
    print(f"[preflight]   Data: {manifest.get('data_file', '?')}")
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
