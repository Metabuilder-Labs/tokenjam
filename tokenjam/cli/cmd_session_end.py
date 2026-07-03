"""`tj session-end` — report a terminal's Claude Code session(s) as closed.

Called best-effort by the `claude` shell wrapper when `claude` exits or is
interrupted. It POSTs to the running daemon's /api/v1/sessions/close endpoint
so the dashboard can move that terminal's tile to the archive immediately
(Claude Code emits no "session closed" telemetry of its own).

Talks to the daemon over HTTP, never the DB (it is in `no_db_commands`). It
must NEVER break the user's shell, so any failure (daemon down, network error)
exits 0 silently; pass -v to surface what happened.
"""
from __future__ import annotations

import json as json_mod
import urllib.error
import urllib.request

import click

_REQUEST_TIMEOUT_S = 2.0


@click.command("session-end")
@click.option("--instance", default=None,
              help="service.instance.id of the terminal whose sessions to close")
@click.option("--session", default=None,
              help="session_id to close (alternative/addition to --instance)")
@click.pass_context
def cmd_session_end(ctx: click.Context, instance: str | None, session: str | None) -> None:
    """Mark a terminal's active sessions as closed (best-effort, over HTTP)."""
    if not instance and not session:
        raise click.UsageError("Provide --instance and/or --session.")

    config = ctx.obj["config"]
    verbose = ctx.obj.get("verbose", False)

    payload: dict[str, str] = {}
    if instance:
        payload["instance_id"] = instance
    if session:
        payload["session_id"] = session

    url = f"http://{config.api.host}:{config.api.port}/api/v1/sessions/close"
    headers = {"Content-Type": "application/json"}
    secret = config.security.ingest_secret
    if secret:
        headers["Authorization"] = f"Bearer {secret}"

    request = urllib.request.Request(
        url, data=json_mod.dumps(payload).encode("utf-8"),
        headers=headers, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_S) as resp:
            if verbose:
                body = resp.read().decode("utf-8", errors="replace")
                click.echo(f"session-end: {resp.status} {body}")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        # Daemon may be down / unreachable. Best-effort: never break the shell.
        if verbose:
            click.echo(f"session-end: skipped ({exc})", err=True)
