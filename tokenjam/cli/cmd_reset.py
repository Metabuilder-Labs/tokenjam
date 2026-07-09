from __future__ import annotations

import click

from tokenjam.utils.formatting import console


@click.command("reset")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def cmd_reset(ctx: click.Context, yes: bool) -> None:
    """Wipe TokenJam's config/daemon/wiring — keeps the tokenjam package installed.

    The config-only counterpart to `tj uninstall` (which also removes the
    package itself): use `tj reset` to reconfigure or pause TokenJam without
    reinstalling the CLI afterward. Run `tj onboard` again to set back up.
    """
    if not yes:
        confirmed = click.confirm(
            "This will delete all TokenJam config, telemetry history, daemon, "
            "and shell wiring — the tokenjam package itself stays installed. "
            "Continue?",
            default=False,
        )
        if not confirmed:
            console.print("[dim]Cancelled.[/dim]")
            return

    from tokenjam.cli.cmd_uninstall import _teardown_side_effects
    _teardown_side_effects(ctx)

    console.print()
    console.print("[dim]Run[/dim]  tj onboard  [dim]to set up again.[/dim]")
