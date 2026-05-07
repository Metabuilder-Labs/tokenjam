"""tj mcp — start the stdio MCP server."""
from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import click
import duckdb

from tj.core.config import find_config_file, load_config


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
    ocw_bin = shutil.which("tj") or sys.argv[0]
    popen_kwargs: dict = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.DETACHED_PROCESS
    else:
        popen_kwargs["start_new_session"] = True

    try:
        subprocess.Popen([ocw_bin, "serve"], **popen_kwargs)
    except (FileNotFoundError, OSError):
        return False

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(0.25)
        if _port_open(host, port):
            return True
    return False


@click.command("mcp")
@click.pass_context
def cmd_mcp(ctx: click.Context) -> None:
    """Start the OCW MCP server (stdio transport for Claude Code)."""
    from tj.mcp.server import mcp, init

    config_path = find_config_file()
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
            db_path = str(Path(config.storage.path).expanduser())
            ro_conn = duckdb.connect(db_path, read_only=True)
            init(ro_conn=ro_conn, config=config, serve_url=None)
    # If no config: init is not called; tools return the no-config sentinel.

    mcp.run()
