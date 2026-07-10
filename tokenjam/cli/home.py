"""Bare ``tj`` landing screen (#240): branded banner + next-best-action.

Shown when ``tj`` is run with no subcommand. A fresh user is pointed at
onboarding; a returning (already-configured) user gets the highest-value
commands surfaced instead of a bare Click help dump, so the post-onboard nudge
isn't a one-time thing they scroll past.

Deliberately does NOT open the database — it reads only config presence, so it
stays fast and works even while ``tj serve`` holds the DuckDB write lock.
"""
from __future__ import annotations

from tokenjam.cli.banner import print_welcome_banner
from tokenjam.core.config import find_config_file
from tokenjam.utils.formatting import console


def print_home() -> None:
    """Render the bare-``tj`` home screen."""
    print_welcome_banner()

    if find_config_file() is None:
        console.print("[bold]Not set up yet.[/bold] Get started:")
        console.print()
        console.print("  [bold]tj onboard --claude-code[/bold]   "
                      "[dim]capture Claude Code usage (recommended)[/dim]")
        console.print("  [bold]tj onboard[/bold]                 "
                      "[dim]generic setup for the Python SDK[/dim]")
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
