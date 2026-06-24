"""Base-URL env wiring for the proxy (#219).

``tj proxy enable`` points an agent's provider base-URLs at the local proxy so
its traffic flows through it; ``disable`` removes that wiring. The wiring lives
in Claude Code's global settings (``~/.claude/settings.json`` ``env`` block) —
the same file ``tj onboard --claude-code`` already manages — so it survives
restarts and is discoverable.

"Absence is safe": we only ever set/remove the two base-URL keys, and we only
remove a key whose value points at *our* proxy (never a user's custom base URL).
``tj doctor`` uses :func:`find_orphaned_wiring` to flag the footgun where the
env still points at the proxy but the proxy is disabled (traffic would hit a
dead port).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Standard provider base-URL env vars honored by the Anthropic / OpenAI SDKs and
# Claude Code. Mapping the proxy onto these routes that provider's traffic
# through it. (Anthropic SDK / Claude Code: ANTHROPIC_BASE_URL; OpenAI: OPENAI_BASE_URL.)
BASE_URL_ENV_VARS = ("ANTHROPIC_BASE_URL", "OPENAI_BASE_URL")


def proxy_base_url(config: Any) -> str:
    """The URL agents should target to route through the proxy."""
    return f"http://{config.proxy.host}:{config.proxy.port}"


def claude_settings_path() -> Path:
    return Path.home() / ".claude" / "settings.json"


def _load_settings(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text() or "{}")
    except (OSError, json.JSONDecodeError):
        return {}


def _write_settings(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def apply_env_wiring(config: Any, *, settings_path: Path | None = None) -> list[str]:
    """Point the provider base-URLs at the proxy in Claude Code's settings.

    Returns the list of env vars wired. Idempotent.
    """
    path = settings_path or claude_settings_path()
    settings = _load_settings(path)
    env: dict = settings.get("env", {}) or {}
    url = proxy_base_url(config)
    for var in BASE_URL_ENV_VARS:
        env[var] = url
    settings["env"] = env
    _write_settings(path, settings)
    return list(BASE_URL_ENV_VARS)


def remove_env_wiring(config: Any, *, settings_path: Path | None = None) -> list[str]:
    """Remove ONLY the base-URL keys that point at our proxy. Idempotent.

    A user's custom base URL (pointing elsewhere) is left untouched.
    """
    path = settings_path or claude_settings_path()
    settings = _load_settings(path)
    env: dict = settings.get("env", {}) or {}
    url = proxy_base_url(config)
    removed: list[str] = []
    for var in BASE_URL_ENV_VARS:
        if env.get(var) == url:
            env.pop(var, None)
            removed.append(var)
    if removed:
        settings["env"] = env
        _write_settings(path, settings)
    return removed


def detect_wiring(config: Any, *, settings_path: Path | None = None) -> dict[str, str]:
    """Base-URL env keys currently pointing at THIS proxy (host:port)."""
    path = settings_path or claude_settings_path()
    env = _load_settings(path).get("env", {}) or {}
    url = proxy_base_url(config)
    return {var: env[var] for var in BASE_URL_ENV_VARS if env.get(var) == url}


def find_orphaned_wiring(config: Any, *, settings_path: Path | None = None) -> list[str]:
    """Env vars that point at the proxy while the proxy is disabled (#219 doctor).

    This is the dangerous state: an agent's traffic is being sent to a port with
    no listener. Returns the offending env var names (empty when consistent).
    """
    if config.proxy.enabled:
        return []
    return list(detect_wiring(config, settings_path=settings_path).keys())
