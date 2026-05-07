"""Helper to let alerts/drift demos seed their own per-agent config.

Each demo's alerts (sensitive_actions, budget) live under [agents.<id>] in
ocw config. Without that block the AlertEngine silently no-ops. This helper
lets each demo idempotently inject the agent block it needs before bootstrap
runs, so demos work out of the box on a fresh `ocw onboard`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]

import tomli_w

from tj.core.config import find_config_file


def ensure_demo_agent_config(agent_id: str, agent_block: dict[str, Any]) -> Path:
    """Idempotently merge `agent_block` into [agents.<agent_id>] in the active ocw config.

    Existing keys are left alone — this only fills in missing config so a tester
    can override demo settings if they want. Writes to .ocw/config.toml in the
    cwd if no config exists yet. Returns the path written.
    """
    cfg_path = find_config_file()
    if cfg_path is None:
        cfg_path = Path(".ocw/config.toml")
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {}
    else:
        cfg_path = Path(cfg_path)
        with cfg_path.open("rb") as f:
            data = tomllib.load(f)

    agents = data.setdefault("agents", {})
    existing = agents.setdefault(agent_id, {})
    _deep_merge_missing(existing, agent_block)

    with cfg_path.open("wb") as f:
        tomli_w.dump(data, f)
    return cfg_path


def _deep_merge_missing(target: dict[str, Any], source: dict[str, Any]) -> None:
    for k, v in source.items():
        if k not in target:
            target[k] = v
        elif isinstance(v, dict) and isinstance(target[k], dict):
            _deep_merge_missing(target[k], v)
