"""Runtime/liveness helpers for the experiment registry (stdlib-only).

Shared by `preflight.py` (launch/status/abandon), `preflight_check.py` (auto
self-registration when a training run validates its token), and `idea_status.py`
(liveness-aware STALE detection). Keeping it dependency-free means training
scripts can import it without pulling in torch/brian2.

The contract: every running experiment's `MANIFEST.json` records *where it is* —
`run_host`, `tmux_session`, `run_pid`, `started_at`, `status="running"` — so a
single `preflight status` can answer "what's open and which session/PID to check"
without guessing.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import tempfile
from datetime import datetime, timezone


def current_host() -> str:
    return socket.gethostname()


def current_pid() -> int:
    return os.getpid()


def current_tmux_session() -> str | None:
    """The tmux session name we're running inside, or None if not under tmux."""
    if not os.environ.get("TMUX"):
        return None
    r = subprocess.run(["tmux", "display-message", "-p", "#S"],
                       capture_output=True, text=True)
    return r.stdout.strip() or None


def session_alive(session: str | None, host: str | None = None) -> bool | None:
    """True/False if the tmux session is alive, or None if unknowable (the
    experiment ran on a different host, so we can't inspect its tmux server)."""
    if not session:
        return None
    if host and host != current_host():
        return None
    r = subprocess.run(["tmux", "has-session", "-t", session],
                       capture_output=True, text=True)
    return r.returncode == 0


def pid_alive(pid: int | None, host: str | None = None) -> bool | None:
    """True/False if the PID is alive locally, or None if unknowable (remote
    host or no pid recorded)."""
    if not pid:
        return None
    if host and host != current_host():
        return None
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except (PermissionError, ValueError, TypeError):
        return True  # exists but not ours / unparseable → treat as alive


def is_alive(manifest: dict) -> bool | None:
    """Best-effort liveness for an experiment from its manifest. Prefers the
    tmux session, falls back to the PID. None = can't tell (e.g. remote host)."""
    host = manifest.get("run_host")
    s = session_alive(manifest.get("tmux_session"), host)
    if s is not None:
        return s
    return pid_alive(manifest.get("run_pid"), host)


def atomic_update_manifest(manifest_path: str, updates: dict) -> dict | None:
    """Merge `updates` into the manifest and write atomically (temp + os.replace),
    so concurrent sweep children can't corrupt it. Returns the new manifest or
    None on failure. Never raises — registration must never break training."""
    try:
        with open(manifest_path) as f:
            m = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    m.update(updates)
    d = os.path.dirname(manifest_path) or "."
    try:
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".manifest.tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(m, f, indent=2)
        os.replace(tmp, manifest_path)
    except OSError:
        try:
            os.unlink(tmp)
        except (OSError, NameError, UnboundLocalError):
            pass
        return None
    return m


def record_run_start(manifest_path: str, session: str | None = None) -> dict | None:
    """Stamp the manifest with this process's runtime location when a training
    run begins. Idempotent: skips if already marked running on the same host
    (so sweep children don't clobber the orchestrator's record). Best-effort."""
    try:
        with open(manifest_path) as f:
            m = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    host = current_host()
    if m.get("status") == "running" and m.get("run_host") == host and m.get("tmux_session"):
        return m  # already registered by the orchestrator
    return atomic_update_manifest(manifest_path, {
        "status": "running",
        "run_host": host,
        "run_pid": current_pid(),
        "tmux_session": session or current_tmux_session(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    })
