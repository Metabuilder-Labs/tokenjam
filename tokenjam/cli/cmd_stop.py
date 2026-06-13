from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

import click

from tokenjam.utils.formatting import console


def stop_tj_serve(*, quiet: bool = False) -> tuple[bool, list[str]]:
    """Stop tj serve (launchd/systemd + orphan foreground processes).

    Callable from onboard/backfill without requiring ``tj`` on PATH.
    Returns ``(stopped, stopped_via)`` where *stopped* is True when anything
    was stopped.
    """
    plist_path = Path.home() / "Library/LaunchAgents/com.tokenjam.serve.plist"
    systemd_path = Path.home() / ".config/systemd/user/tokenjam.service"

    stopped_via: list[str] = []

    # Try launchd first (macOS).
    if plist_path.exists():
        result = subprocess.run(
            ["launchctl", "unload", "-w", str(plist_path)],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            stopped_via.append("launchd daemon unloaded")

    # Try systemd (Linux).
    if systemd_path.exists():
        result = subprocess.run(
            ["systemctl", "--user", "disable", "--now", "tokenjam"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            stopped_via.append("systemd service stopped")

    # Sweep orphan foreground `uv run tj serve` / `tj serve` processes.
    signaled: set[int] = set()
    for _ in range(20):
        pid = _find_serve_pid()
        if not pid or pid in signaled:
            break
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            break
        signaled.add(pid)
        stopped_via.append(f"PID {pid}")

    if not quiet:
        if stopped_via:
            console.print(
                f"[green]tj serve stopped.[/green] ({', '.join(stopped_via)})"
            )
        else:
            console.print("[dim]tj serve is not running.[/dim]")
    return bool(stopped_via), stopped_via


@click.command("stop")
@click.pass_context
def cmd_stop(ctx: click.Context) -> None:
    """Stop the tj serve daemon or background process."""
    stop_tj_serve()


def _find_serve_pid() -> int | None:
    """Find the PID of a running `tj serve`. process."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "tokenjam\\.serve|tj serve|uv run tj serve"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                pid = int(line.strip())
                # Don't return our own PID
                if pid != os.getpid():
                    return pid
    except (FileNotFoundError, ValueError):
        pass
    return None
