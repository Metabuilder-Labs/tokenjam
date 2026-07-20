import os
from typing import TYPE_CHECKING

import click
from tokenjam.core.config import load_config
from tokenjam.core.db import open_db

if TYPE_CHECKING:
    from tokenjam.core.api_backend import ApiBackend
    from tokenjam.core.db import DuckDBBackend


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
    """tj - the self-improvement loop for AI agents."""
    ctx.ensure_object(dict)

    # Bare `tj` (no subcommand) → branded home screen (#240): banner +
    # next-best-action. Reads only config presence, never opens the DB.
    #
    # Bare `npx tokenjam` is a separate path: the npm wrapper
    # (npm-wrapper/bin/tj.js) sets TJ_NPX_ZERO_INSTALL_REPORT=1 when it invokes
    # `tj` with no args, since a brand-new npx user has no config yet and the
    # home screen would dead-end them ("you're set up", suggesting commands
    # they don't have). That env var routes here to the same zero-install
    # report `cmd_quickstart` renders, invoked directly via `ctx.invoke` — it
    # has no public/typeable subcommand name (#6 retired the `quickstart`
    # command; the report itself lives on, reached only from this branch).
    if ctx.invoked_subcommand is None:
        if no_color:
            from rich import reconfigure
            reconfigure(no_color=True)
        if os.environ.get("TJ_NPX_ZERO_INSTALL_REPORT"):
            from tokenjam.cli.cmd_quickstart import cmd_quickstart
            ctx.invoke(cmd_quickstart, since="30d", root_path=None,
                       full=False, output_json=output_json)
            return
        from tokenjam.cli.home import print_home
        print_home()
        return

    config = load_config(config_path)
    if db_path:
        config.storage.path = db_path

    # Commands that don't need a database connection
    no_db_commands = {
        "stop", "uninstall", "onboard", "mcp", "demo", "policy",
        "proxy", "summarize", "pricing", "otel-resource-attrs", "session-end",
        "statusline",
        # `ping` emits a test span through the SDK (which resolves its own
        # HTTP-vs-direct path); it must not take the CLI's DuckDB lock (#80).
        "ping",
    }
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

    db: "DuckDBBackend | ApiBackend | None" = None
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
from tokenjam.cli.cmd_reset import cmd_reset  # noqa: E402
from tokenjam.cli.cmd_doctor import cmd_doctor  # noqa: E402
from tokenjam.cli.cmd_budget import cmd_budget  # noqa: E402
from tokenjam.cli.cmd_optimize import cmd_optimize  # noqa: E402
from tokenjam.cli.cmd_route import cmd_route  # noqa: E402
from tokenjam.cli.cmd_tokenmaxx import cmd_tokenmaxx  # noqa: E402
from tokenjam.cli.cmd_backfill import cmd_backfill  # noqa: E402
from tokenjam.cli.cmd_report import cmd_report  # noqa: E402
from tokenjam.cli.cmd_policy import cmd_policy  # noqa: E402
from tokenjam.cli.cmd_pricing import cmd_pricing  # noqa: E402
from tokenjam.cli.cmd_proxy import cmd_proxy  # noqa: E402
from tokenjam.cli.cmd_context import cmd_context  # noqa: E402
from tokenjam.cli.cmd_session_story import cmd_session_story  # noqa: E402
from tokenjam.cli.cmd_quota_audit import cmd_quota_audit  # noqa: E402
from tokenjam.cli.cmd_otel import cmd_otel_resource_attrs  # noqa: E402
from tokenjam.cli.cmd_session_end import cmd_session_end  # noqa: E402
from tokenjam.cli.cmd_statusline import cmd_statusline  # noqa: E402
from tokenjam.cli.cmd_loop import cmd_loop  # noqa: E402
from tokenjam.cli.cmd_resume_brief import cmd_resume_brief  # noqa: E402
from tokenjam.cli.cmd_ping import cmd_ping  # noqa: E402
from tokenjam.cli.cmd_relearn import cmd_relearn  # noqa: E402

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
cli.add_command(cmd_reset, name="reset")
cli.add_command(cmd_doctor, name="doctor")
cli.add_command(cmd_budget, name="budget")
cli.add_command(cmd_optimize, name="optimize")
cli.add_command(cmd_route, name="route")
cli.add_command(cmd_tokenmaxx, name="tokenmaxx")
cli.add_command(cmd_backfill, name="backfill")
cli.add_command(cmd_report, name="report")
cli.add_command(cmd_policy, name="policy")
cli.add_command(cmd_pricing, name="pricing")
cli.add_command(cmd_proxy, name="proxy")
cli.add_command(cmd_context, name="context")
cli.add_command(cmd_session_story, name="session-story")
cli.add_command(cmd_quota_audit, name="quota-audit")
cli.add_command(cmd_otel_resource_attrs, name="otel-resource-attrs")
cli.add_command(cmd_session_end, name="session-end")
cli.add_command(cmd_statusline, name="statusline")
cli.add_command(cmd_loop, name="loop")
cli.add_command(cmd_resume_brief, name="resume-brief")
cli.add_command(cmd_ping, name="ping")
cli.add_command(cmd_relearn, name="relearn")

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

from tokenjam.cli.cmd_summarize import cmd_summarize  # noqa: E402
cli.add_command(cmd_summarize, name="summarize")

# The self-improve loop's terminal write path (list/apply/enable/revert) plus
# the on-demand verify recompute. Registered last so its group stays a single
# self-contained block.
from tokenjam.cli.cmd_relearn import cmd_relearn  # noqa: E402
cli.add_command(cmd_relearn, name="relearn")
