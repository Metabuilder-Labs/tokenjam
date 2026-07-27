"""Bare ``tj`` landing screen (#240): branded banner + next-best-action.

Shown when ``tj`` is run with no subcommand. A fresh user is pointed at
onboarding; a returning (already-configured) user gets the highest-value
commands surfaced instead of a bare Click help dump, so the post-onboard nudge
isn't a one-time thing they scroll past.

Deliberately does NOT open the database — it reads only config presence and the
DB file's existence on disk, so it stays fast and works even while ``tj serve``
holds the DuckDB write lock.
"""
from __future__ import annotations

import os
from pathlib import Path

from tokenjam.cli.banner import print_welcome_banner
from tokenjam.core.config import StorageConfig, find_config_file
from tokenjam.utils.formatting import console


def _tj_config_env_path() -> Path | None:
    """A valid ``TJ_CONFIG`` path, if the env var is set and points at a real
    file — ``None`` otherwise (including when it's set but broken).

    Shared by ``_is_set_up`` and ``_db_has_data``: both need to honor
    ``TJ_CONFIG`` (like every other config-reading path in this codebase, via
    ``resolve_config_path``) but must NOT raise on a bad value. The bare ``tj``
    landing screen renders before ``main.py``'s group callback validates
    config (it returns early, skipping ``load_config``), so this is the one
    context where ``resolve_config_path``'s fail-loud contract would crash a
    screen that's supposed to always render something actionable.
    """
    tj_config = os.environ.get("TJ_CONFIG")
    if tj_config and Path(tj_config).expanduser().is_file():
        return Path(tj_config).expanduser()
    return None


def _db_has_data() -> bool:
    """Cheaply detect a populated on-disk DB WITHOUT opening it.

    A user who backfilled (or otherwise ingested spans) but never ran full
    ``tj onboard`` has no config file, yet is clearly set up (#506). We must
    not open the DB here — that would fail while ``tj serve`` holds the write
    lock — so we treat the presence of a non-empty DB file as the signal.

    Resolves the DB path from an existing config's ``[storage] path`` when one
    is discoverable (honoring ``TJ_CONFIG`` first, then search-path discovery),
    else from the default (``~/.tj/telemetry.duckdb``).
    """
    db_path = StorageConfig.path
    cfg_file = _tj_config_env_path() or find_config_file()
    if cfg_file is not None:
        # Cheap TOML parse (no DB open) to honor a custom storage path.
        try:
            from tokenjam.core.config import load_config

            db_path = load_config(str(cfg_file)).storage.path
        except Exception:
            pass
    try:
        p = Path(db_path).expanduser()
        return p.is_file() and p.stat().st_size > 0
    except OSError:
        return False


def _is_set_up() -> bool:
    """True if the user is set up: any discoverable config OR a populated DB.

    ``find_config_file`` covers the global (``~/.config/tj/config.toml``),
    project-local (``.tj/config.toml``), and ``tokenjam.toml`` locations; we
    also honor ``TJ_CONFIG`` so an env-pointed config counts. Failing that, a
    non-empty DB (e.g. from ``tj backfill``) means they're set up too (#506).
    """
    tj_config = os.environ.get("TJ_CONFIG")
    if tj_config and Path(tj_config).expanduser().is_file():
        return True
    if find_config_file() is not None:
        return True
    return _db_has_data()


def _current_repo_unregistered() -> bool:
    """True if this repo's Claude Code agent has no project namespace yet AND
    the user already has at least one other Claude Code agent registered
    elsewhere — i.e. a good moment to nudge `--add-project`.

    The second condition matters: a Codex-only or SDK-only user has no
    `claude-code-*` agent at all, and `--add-project` (which only ever writes
    a `claude-code-*` entry) would not help them — nudging them anyway would
    just nag forever with an irrelevant command.

    Resolves the config the same way `tj onboard --add-project` and the rest
    of the CLI do (TJ_CONFIG, then project-local, then global — see
    `load_config`'s `config_path` resolution), so this reads the exact config
    that would actually be written to / already governs this repo, not
    always the default global path regardless of an active override.

    Best-effort: any failure (no git, unreadable config, etc.) reads as "don't
    nudge" rather than raising out of the home screen.
    """
    try:
        from tokenjam.cli.cmd_onboard import _derive_project_name
        from tokenjam.core.config import load_config

        config = load_config()
        if config.config_path is None:
            return False
        has_any_claude_code_agent = any(
            aid.startswith("claude-code-") for aid in config.agents
        )
        if not has_any_claude_code_agent:
            return False
        agent_id = f"claude-code-{_derive_project_name()}"
        agent = config.agents.get(agent_id)
        return agent is None or not agent.project
    except Exception:
        return False


def print_home() -> None:
    """Render the bare-``tj`` home screen."""
    print_welcome_banner()

    if not _is_set_up():
        console.print("[heading]Not set up yet.[/heading] Get started:")
        console.print()
        console.print("  [accent]tj onboard[/accent]   "
                      "[dim]interactive setup: asks how you use AI agents "
                      "and wires the right path[/dim]")
        console.print()
        console.print(
            "[dim]Run it once inside each project so sessions and "
            "proposals group per project in the dashboard.[/dim]"
        )
        console.print()
        console.print(
            "[dim]Docs: https://github.com/Metabuilder-Labs/tokenjam[/dim]"
        )
        return

    console.print("[ok]\u2713 You're set up.[/ok] Next best actions:")
    console.print()
    console.print("  [accent]tj status[/accent]      "
                  "[muted]agent overview: what's running, recent cost[/muted]")
    console.print("  [accent]tj tokenmaxx[/accent]   "
                  "[dim]your shareable efficiency tier[/dim]")
    console.print("  [accent]tj optimize[/accent]    "
                  "[dim]cost-saving candidates from your usage[/dim]")
    console.print("  [accent]tj serve[/accent]       "
                  "[muted]open Lens (web UI) at[/muted] [url]http://127.0.0.1:7391/[/url]")
    console.print()
    if _current_repo_unregistered():
        console.print(
            "[dim]New repo? [/dim][accent]tj onboard --add-project[/accent]"
            "[dim]   registers this one without repeating full setup[/dim]"
        )
        console.print()
    console.print("[muted]Full command list:[/muted]  [accent]tj --help[/accent]")
