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


def _db_has_data() -> bool:
    """Cheaply detect a populated on-disk DB WITHOUT opening it.

    A user who backfilled (or otherwise ingested spans) but never ran full
    ``tj onboard`` has no config file, yet is clearly set up (#506). We must
    not open the DB here — that would fail while ``tj serve`` holds the write
    lock — so we treat the presence of a non-empty DB file as the signal.

    Resolves the DB path from an existing config's ``[storage] path`` when one
    is discoverable, else from the default (``~/.tj/telemetry.duckdb``).
    """
    db_path = StorageConfig.path
    cfg_file = find_config_file()
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


def print_home() -> None:
    """Render the bare-``tj`` home screen."""
    print_welcome_banner()

    if not _is_set_up():
        console.print("[bold]Not set up yet.[/bold] Get started:")
        console.print()
        console.print("  [bold]tj onboard[/bold]   "
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

    console.print("[bold]You're set up.[/bold] Next best actions:")
    console.print()
    console.print("  [bold]tj status[/bold]      "
                  "[dim]agent overview — what's running, recent cost[/dim]")
    console.print("  [bold]tj tokenmaxx[/bold]   "
                  "[dim]your shareable efficiency tier[/dim]")
    console.print("  [bold]tj optimize[/bold]    "
                  "[dim]cost-saving candidates from your usage[/dim]")
    console.print("  [bold]tj serve[/bold]       "
                  "[dim]open Lens (web UI) at http://127.0.0.1:7391/[/dim]")
    console.print()
    console.print("[dim]Full command list:[/dim]  tj --help")
