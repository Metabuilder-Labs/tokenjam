import click
from tokenjam.core.config import load_config
from tokenjam.core.db import open_db


@click.group(
    invoke_without_command=True,
    epilog="Upgrade with: pipx upgrade tokenjam "
           "(then `tj stop && tj serve &` to reload the daemon). "
           "Verify with `tj --version`.",
)
@click.version_option(package_name="tokenjam")
@click.option("--config", "config_path", default=None, envvar="TJ_CONFIG",
              help="Config file path (default: auto-discover)")
@click.option("--json", "output_json", is_flag=True,
              help="Output machine-readable JSON")
@click.option("--no-color", is_flag=True)
@click.option("--db", "db_path", default=None, help="Database path override")
@click.option("--agent", default=None, help="Filter to specific agent_id")
@click.option("-v", "--verbose", is_flag=True)
@click.pass_context
def cli(ctx: click.Context, config_path: str | None, output_json: bool,
        no_color: bool, db_path: str | None, agent: str | None,
        verbose: bool) -> None:
    """tj - local-first observability for AI agents."""
    ctx.ensure_object(dict)

    # Bare `tj` (no subcommand) → branded home screen (#240): banner +
    # next-best-action. Reads only config presence, never opens the DB.
    if ctx.invoked_subcommand is None:
        if no_color:
            from rich import reconfigure
            reconfigure(no_color=True)
        from tokenjam.cli.home import print_home
        print_home()
        return

    config = load_config(config_path)
    if db_path:
        config.storage.path = db_path

    # Commands that don't need a database connection
    no_db_commands = {"stop", "uninstall", "onboard", "mcp", "demo", "policy"}
    invoked = ctx.invoked_subcommand
    if invoked in no_db_commands:
        ctx.obj["config"] = config
        ctx.obj["db"] = None
        ctx.obj["output_json"] = output_json
        ctx.obj["no_color"] = no_color
        ctx.obj["agent"] = agent
        ctx.obj["verbose"] = verbose
        if no_color:
            from rich import reconfigure
            reconfigure(no_color=True)
        return

    db = None
    try:
        db = open_db(config.storage)
    except Exception as e:
        err_msg = str(e).lower()
        if "lock" in err_msg or "already open" in err_msg or "i/o error" in err_msg:
            from tokenjam.core.api_backend import probe_api
            api_key = config.api.auth.api_key if config.api.auth.enabled else None
            db = probe_api(config.api.host, config.api.port, api_key)
            if db is None:
                raise click.ClickException(
                    "Database is locked (tj serve is running?) and the API "
                    f"is not reachable at http://{config.api.host}:{config.api.port}. "
                    "Start tj serve or stop the process holding the DB lock."
                ) from e
            ctx.obj["api_mode"] = True
        else:
            raise

    ctx.obj["config"] = config
    ctx.obj["db"] = db
    ctx.obj["output_json"] = output_json
    ctx.obj["no_color"] = no_color
    ctx.obj["agent"] = agent
    ctx.obj["verbose"] = verbose
    if no_color:
        from rich import reconfigure
        reconfigure(no_color=True)


# Register all subcommands
from tokenjam.cli.cmd_onboard import cmd_onboard  # noqa: E402
from tokenjam.cli.cmd_status import cmd_status  # noqa: E402
from tokenjam.cli.cmd_traces import cmd_traces, cmd_trace  # noqa: E402
from tokenjam.cli.cmd_cost import cmd_cost  # noqa: E402
from tokenjam.cli.cmd_alerts import cmd_alerts  # noqa: E402
from tokenjam.cli.cmd_tools import cmd_tools  # noqa: E402
from tokenjam.cli.cmd_export import cmd_export  # noqa: E402
from tokenjam.cli.cmd_serve import cmd_serve  # noqa: E402
from tokenjam.cli.cmd_stop import cmd_stop  # noqa: E402
from tokenjam.cli.cmd_uninstall import cmd_uninstall  # noqa: E402
from tokenjam.cli.cmd_doctor import cmd_doctor  # noqa: E402
from tokenjam.cli.cmd_budget import cmd_budget  # noqa: E402
from tokenjam.cli.cmd_optimize import cmd_optimize  # noqa: E402
from tokenjam.cli.cmd_route import cmd_route  # noqa: E402
from tokenjam.cli.cmd_tokenmaxx import cmd_tokenmaxx  # noqa: E402
from tokenjam.cli.cmd_backfill import cmd_backfill  # noqa: E402
from tokenjam.cli.cmd_report import cmd_report  # noqa: E402
from tokenjam.cli.cmd_policy import cmd_policy  # noqa: E402

cli.add_command(cmd_onboard, name="onboard")
cli.add_command(cmd_status, name="status")
cli.add_command(cmd_traces, name="traces")
cli.add_command(cmd_trace, name="trace")
cli.add_command(cmd_cost, name="cost")
cli.add_command(cmd_alerts, name="alerts")
cli.add_command(cmd_tools, name="tools")
cli.add_command(cmd_export, name="export")
cli.add_command(cmd_serve, name="serve")
cli.add_command(cmd_stop, name="stop")
cli.add_command(cmd_uninstall, name="uninstall")
cli.add_command(cmd_doctor, name="doctor")
cli.add_command(cmd_budget, name="budget")
cli.add_command(cmd_optimize, name="optimize")
cli.add_command(cmd_route, name="route")
cli.add_command(cmd_tokenmaxx, name="tokenmaxx")
cli.add_command(cmd_backfill, name="backfill")
cli.add_command(cmd_report, name="report")
cli.add_command(cmd_policy, name="policy")

# cmd_drift is provided by task 05 — register if available
try:
    from tokenjam.cli.cmd_drift import cmd_drift  # noqa: E402
    cli.add_command(cmd_drift, name="drift")
except ImportError:
    pass

from tokenjam.cli.cmd_mcp import cmd_mcp  # noqa: E402
cli.add_command(cmd_mcp, name="mcp")

from tokenjam.cli.cmd_demo import cmd_demo  # noqa: E402
cli.add_command(cmd_demo, name="demo")
