from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Callable

import click

from tokenjam.core.server_state import find_own_serve_pid
from tokenjam.utils.formatting import console

# How long to wait for a signaled PID to actually exit before concluding the
# signal didn't land. SIGTERM gets the longer window (graceful shutdown --
# flushing the pipeline, closing DuckDB); SIGKILL is immediate at the kernel
# level, so a short window is just there to observe the exit, not to wait on
# any handler.
_TERM_WAIT_S = 2.0
_KILL_WAIT_S = 1.0
_POLL_INTERVAL_S = 0.05


def stop_tj_serve(*, quiet: bool = False) -> tuple[bool, list[str]]:
    """Stop tj serve (launchd/systemd + orphan foreground processes).

    Callable from onboard/backfill without requiring ``tj`` on PATH.
    Returns ``(stopped, stopped_via)`` where *stopped* is True only once
    every process this call signaled has been confirmed to have exited --
    never on a signal simply having been sent.
    """
    plist_path = Path.home() / "Library/LaunchAgents/com.tokenjam.serve.plist"
    systemd_path = Path.home() / ".config/systemd/user/tokenjam.service"

    stopped_via: list[str] = []
    failed_via: list[str] = []

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

    # Sweep the `tj serve` daemon belonging to THIS install (see
    # _find_serve_pid -- PID-file scoped to $HOME, not a machine-wide pgrep).
    # This also catches a launchd/systemd-managed daemon that's slow to exit,
    # since it writes the same state file regardless of how it was launched.
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

        if _wait_for_exit(pid, timeout_s=_TERM_WAIT_S):
            stopped_via.append(f"PID {pid}")
            continue

        # SIGTERM didn't land in time -- escalate rather than silently
        # reporting success for a process that's still alive.
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            stopped_via.append(f"PID {pid}")
            continue

        if _wait_for_exit(pid, timeout_s=_KILL_WAIT_S):
            stopped_via.append(f"PID {pid} (SIGKILL)")
        else:
            failed_via.append(f"PID {pid}")

    if not quiet:
        if failed_via:
            console.print(
                f"[red]tj serve did not stop.[/red] Still running: "
                f"{', '.join(failed_via)}"
            )
        if stopped_via:
            console.print(
                f"[green]tj serve stopped.[/green] ({', '.join(stopped_via)})"
            )
        elif not failed_via:
            console.print("[dim]tj serve is not running.[/dim]")

    return (bool(stopped_via) and not failed_via), stopped_via


@click.command("stop")
@click.pass_context
def cmd_stop(ctx: click.Context) -> None:
    """Stop the tj serve daemon or background process."""
    stop_tj_serve()


def _find_serve_pid() -> int | None:
    """Find the PID of the `tj serve` daemon started under THIS install
    (i.e. this invocation's $HOME) -- see `tokenjam.core.server_state`.

    A thin wrapper (rather than calling `find_own_serve_pid()` directly from
    `stop_tj_serve`) so tests can patch discovery independently of the
    signal-and-verify logic above.
    """
    return find_own_serve_pid()


def _pid_alive(pid: int) -> bool:
    """`os.kill(pid, 0)` sends no signal -- it only checks the PID exists.
    A thin, separately-patchable wrapper so tests can simulate exit timing
    without mocking every `os.kill` call (including the SIGTERM/SIGKILL
    sends above) into a no-op."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_for_exit(
    pid: int,
    *,
    timeout_s: float,
    interval_s: float = _POLL_INTERVAL_S,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> bool:
    """Poll until `pid` exits or `timeout_s` elapses.

    Returns True once the process is confirmed gone, False if it's still
    alive when we give up. `sleep`/`monotonic` are injectable so tests never
    wait real seconds.
    """
    start = monotonic()
    while True:
        if not _pid_alive(pid):
            return True
        if monotonic() - start >= timeout_s:
            return False
        sleep(interval_s)
