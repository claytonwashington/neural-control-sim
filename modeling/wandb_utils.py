"""Shared Weights & Biases (wandb) integration utilities.

Provides thin wrappers around wandb's API that gracefully no-op when wandb
is not installed or when the user passes --no-wandb.  Import individual
helpers in your training script *after* parsing CLI args so that wandb
is only imported (and authenticated) when actually needed.
"""

import os
from typing import Any, Dict, Optional

_DEFAULT_PROJECT = "cleo-digital-twin"

# Module-level flag: set to True once wandb.init() succeeds.
_wandb_active = False
_wandb = None  # Lazy-loaded wandb module


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def add_wandb_args(parser):
    """Add standard wandb-related arguments to an argparse.ArgumentParser."""
    group = parser.add_argument_group("Weights & Biases")
    group.add_argument(
        "--no-wandb", action="store_true", default=False,
        help="Disable wandb logging entirely",
    )
    group.add_argument(
        "--wandb-project", type=str, default=_DEFAULT_PROJECT,
        help=f"wandb project name (default: {_DEFAULT_PROJECT})",
    )
    group.add_argument(
        "--wandb-entity", type=str, default=None,
        help="wandb entity (team or user); uses default if unset",
    )
    group.add_argument(
        "--wandb-name", type=str, default=None,
        help="wandb run display name",
    )
    group.add_argument(
        "--wandb-tags", type=str, nargs="*", default=None,
        help="Space-separated tags for the wandb run",
    )
    group.add_argument(
        "--wandb-group", type=str, default=None,
        help="wandb group name (e.g. sweep ID)",
    )
    return parser


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def wandb_init(
    args,
    model=None,
    extra_config: Optional[Dict[str, Any]] = None,
    script_name: Optional[str] = None,
) -> bool:
    """Initialize a wandb run.  Returns True on success, False otherwise.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI args (must include the flags added by ``add_wandb_args``).
    model : torch.nn.Module, optional
        If provided, the number of trainable parameters is logged in config.
    extra_config : dict, optional
        Additional key/value pairs merged into the wandb config.
    script_name : str, optional
        Identifier for the training script (logged as config["script"]).
    """
    global _wandb_active, _wandb

    if getattr(args, "no_wandb", False):
        return False

    try:
        import wandb as _wb
    except ImportError:
        print("[wandb] wandb not installed \u2013 skipping experiment tracking.")
        return False

    _wandb = _wb

    # Build config dict from args
    config = {k: v for k, v in vars(args).items() if not k.startswith("wandb") and k != "no_wandb"}
    if model is not None:
        config["n_params"] = sum(p.numel() for p in model.parameters())
    if script_name is not None:
        config["script"] = script_name
    if extra_config:
        config.update(extra_config)

    try:
        _wb.init(
            project=getattr(args, "wandb_project", _DEFAULT_PROJECT),
            entity=getattr(args, "wandb_entity", None),
            name=getattr(args, "wandb_name", None),
            tags=getattr(args, "wandb_tags", None),
            group=getattr(args, "wandb_group", None),
            config=config,
            reinit=True,
        )
        if model is not None:
            _wb.watch(model, log="gradients", log_freq=50)
        _wandb_active = True
        print(f"[wandb] Run initialized: {_wb.run.url}")
        return True
    except Exception as e:
        print(f"[wandb] Failed to initialize: {e}")
        _wandb_active = False
        return False


def wandb_finish():
    """Finish the active wandb run (no-op if inactive)."""
    global _wandb_active
    if _wandb_active and _wandb is not None:
        try:
            _wandb.finish()
        except Exception:
            pass
        _wandb_active = False


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def wandb_log(metrics: Dict[str, Any], step: Optional[int] = None):
    """Log metrics to wandb (no-op if inactive)."""
    if not _wandb_active or _wandb is None:
        return
    try:
        if step is not None:
            _wandb.log(metrics, step=step)
        else:
            _wandb.log(metrics)
    except Exception:
        pass


def wandb_log_image(key: str, path: str, caption: Optional[str] = None):
    """Log an image file to wandb (no-op if inactive)."""
    if not _wandb_active or _wandb is None:
        return
    if not os.path.exists(path):
        return
    try:
        img = _wandb.Image(path, caption=caption)
        _wandb.log({key: img})
    except Exception:
        pass


def wandb_summary(metrics: Dict[str, Any]):
    """Set wandb run summary metrics (no-op if inactive)."""
    if not _wandb_active or _wandb is None:
        return
    try:
        for k, v in metrics.items():
            _wandb.run.summary[k] = v
    except Exception:
        pass
