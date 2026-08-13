from __future__ import annotations

import click

from tokenjam.cli.tj_status import TjCommand, tj_status
from tokenjam.utils.formatting import console


#: No class-level `status_message`: the confirmation prompt below must not
#: run under a live spinner. `tj_status` is called manually after the
#: prompt resolves, wrapping only the teardown itself.
@click.command("reset", cls=TjCommand)
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def cmd_reset(ctx: click.Context, yes: bool) -> None:
    """Reset tj config/daemon (keeps the package).

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
    with tj_status("Resetting tj…", ctx):
        _teardown_side_effects(ctx)

    console.print()
    console.print("[dim]Run[/dim]  tj onboard  [dim]to set up again.[/dim]")
