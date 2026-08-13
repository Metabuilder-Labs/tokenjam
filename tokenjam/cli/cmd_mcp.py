"""tj mcp — start the stdio MCP server (for SDK / API users).

The MCP is tj's in-request-path surface: it belongs in an SDK / API integration
where tj already sits in the loop. Claude Code / Codex subscription users should
NOT wire this — an in-loop MCP is a per-turn token burden on them (+36% measured);
`tj onboard --claude-code` gives them the zero-token statusline instead.
"""
from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import click
import duckdb

from tokenjam.cli.tj_status import TjCommand
from tokenjam.core.config import load_config, resolve_config_path


def _port_open(host: str, port: int) -> bool:
    """Return True if something is listening on host:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect((host, port))
            return True
        except (ConnectionRefusedError, OSError):
            return False


def _start_and_wait(host: str, port: int, timeout: float = 10.0) -> bool:
    """Start tj serve in the background and wait up to *timeout* seconds for it
    to accept connections. Returns True if the server is ready in time."""
    tj_bin = shutil.which("tj") or sys.argv[0]
    popen_kwargs: dict = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.DETACHED_PROCESS
    else:
        popen_kwargs["start_new_session"] = True

    try:
        subprocess.Popen([tj_bin, "serve"], **popen_kwargs)
    except (FileNotFoundError, OSError):
        return False

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(0.25)
        if _port_open(host, port):
            return True
    return False


#: Structural opt-out: mcp speaks stdio JSON-RPC, and ANY stray write to
#: stdout (a spinner line included) corrupts the protocol stream — this file
#: has no print statement at all, deliberately. Never wire a status here.
@click.command("mcp", cls=TjCommand, no_status=True)
@click.pass_context
def cmd_mcp(ctx: click.Context) -> None:
    """Start the MCP server for SDK/API integrations.

    The MCP puts tj in the request path — the right surface for SDK / API
    integrations. Claude Code / Codex subscription users get tj out-of-band via
    the zero-token statusline (`tj statusline`, wired by `tj onboard
    --claude-code`) instead of paying the per-turn MCP token burden.
    """
    from tokenjam.mcp.server import mcp, init

    # The invocation's own config wins: `tj --config PATH mcp` must serve that
    # file, and an explicit override is invisible to rediscovery.
    config_path = resolve_config_path((ctx.obj or {}).get("config_path_override"))
    if config_path is not None:
        config = load_config(str(config_path))
        host = config.api.host
        port = config.api.port

        if _port_open(host, port) or _start_and_wait(host, port):
            # tj serve is running (already up or we just started it)
            serve_url = f"http://{host}:{port}"
            init(ro_conn=None, config=config, serve_url=serve_url)
        else:
            # Could not reach or start tj serve — fall back to read-only DuckDB
            # so MCP read tools still work, though live ingest won't be available.
            #
            # A read-only connection bypasses `DuckDBBackend.__init__` and its
            # `run_migrations`/schema-self-heal call entirely (this file never
            # calls either), so a store last written by an older `tj` build —
            # missing a column or table a newer analyzer/tool depends on —
            # would otherwise surface as a raw duckdb BinderError/
            # CatalogException the first time a tool touches it. Detect the
            # gap up front (both checks are pure SELECTs against
            # information_schema, safe on a read-only connection) and hand it
            # to `init()` so every tool degrades to one clear message instead.
            from tokenjam.core.db import missing_expected_columns, missing_expected_tables

            db_path = str(Path(config.storage.path).expanduser())
            ro_conn = duckdb.connect(db_path, read_only=True)
            schema_gap = missing_expected_tables(ro_conn) + missing_expected_columns(ro_conn)
            init(ro_conn=ro_conn, config=config, serve_url=None, schema_gap=schema_gap)
    # If no config: init is not called; tools return the no-config sentinel.

    mcp.run()
