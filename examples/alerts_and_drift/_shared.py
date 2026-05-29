"""Helper to let alerts/drift demos seed their own per-agent config.

Each demo's alerts (sensitive_actions, budget) live under [agents.<id>] in
tj config. Without that block the AlertEngine silently no-ops. This helper
lets each demo idempotently inject the agent block it needs before bootstrap
runs, so demos work out of the box on a fresh `tj onboard`.
"""
from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]

import tomli_w

from tokenjam.core.config import SEARCH_PATHS, find_config_file


def ensure_demo_agent_config(
    agent_id: str, agent_block: dict[str, Any]
) -> list[Path]:
    """Idempotently merge `agent_block` into [agents.<agent_id>] in every
    tj config file that currently exists on disk.

    The original helper wrote to `find_config_file()` only — which returns
    the first match in the search-path order (project-local before global).
    That broke a real footgun (#68 §6 → §5): a tester who'd run
    `tj onboard --claude-code` had a global config AND a project-local
    config, the daemon was launched with the global one, but the helper
    only updated the project-local file. Demo agents stayed invisible
    to the running daemon and no alerts fired.

    Now: write to every config file in SEARCH_PATHS that exists. Idempotent
    merge — existing keys stay, only missing ones get added. Returns the
    list of paths actually touched.

    If no config file exists anywhere, creates .tj/config.toml in cwd as
    a fallback (same as the old behaviour).
    """
    touched: list[Path] = []
    for candidate in SEARCH_PATHS:
        p = Path(candidate)
        if not p.exists():
            continue
        _merge_into(p, agent_id, agent_block)
        try:
            touched.append(p.resolve())
        except OSError:
            touched.append(p)
    if not touched:
        fallback = Path(".tj/config.toml")
        fallback.parent.mkdir(parents=True, exist_ok=True)
        _merge_into(fallback, agent_id, agent_block, allow_create=True)
        touched.append(fallback.resolve())
    return touched


def warn_if_daemon_running() -> bool:
    """Print a heads-up if a tj serve daemon is listening on the default port.

    Demos write config changes that the AlertEngine needs to load. If the
    daemon is up, it loaded config at startup and won't auto-reload on
    file changes — alerts for the freshly-added demo agent will silently
    fail to fire (issue #68 §6).

    Returns True if a daemon appears to be running; the caller can choose
    whether to abort, prompt the user, or just continue.
    """
    port = int(os.environ.get("TJ_PORT", "7391"))
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.2)
    try:
        running = s.connect_ex(("127.0.0.1", port)) == 0
    except OSError:
        running = False
    finally:
        s.close()
    if running:
        print(
            "\n[demo] A tj serve daemon appears to be running on port "
            f"{port}. The daemon's AlertEngine loaded its config at "
            "startup and does NOT hot-reload — alerts for this demo's "
            "agent will only fire if the daemon is restarted to pick "
            "up the freshly-added config:\n"
            "\n    tj stop && tj serve &\n"
            "\nOr stop the daemon entirely (`tj stop`) so the SDK writes "
            "directly to DuckDB and fires alerts in-process.\n",
            file=sys.stderr,
        )
    return running


def _deep_merge_missing(target: dict[str, Any], source: dict[str, Any]) -> None:
    for k, v in source.items():
        if k not in target:
            target[k] = v
        elif isinstance(v, dict) and isinstance(target[k], dict):
            _deep_merge_missing(target[k], v)


def _merge_into(
    cfg_path: Path,
    agent_id: str,
    agent_block: dict[str, Any],
    *,
    allow_create: bool = False,
) -> None:
    if cfg_path.exists():
        with cfg_path.open("rb") as f:
            data = tomllib.load(f)
    elif allow_create:
        data = {}
    else:
        return
    agents = data.setdefault("agents", {})
    existing = agents.setdefault(agent_id, {})
    _deep_merge_missing(existing, agent_block)
    with cfg_path.open("wb") as f:
        tomli_w.dump(data, f)
