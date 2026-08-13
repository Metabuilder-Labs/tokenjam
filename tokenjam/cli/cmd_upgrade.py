"""`tj upgrade`: upgrade the installed tokenjam package AND restart the
`tj serve` daemon so it actually reflects the new version.

Why this exists: `tokenjam/__init__.py` resolves `__version__` once at
import time. `tj serve` is a long-lived process (often supervised by
launchd/systemd), so a package upgrade done outside `tj` (`pipx upgrade
tokenjam`, `uv tool upgrade tokenjam`, ...) leaves the RUNNING daemon
serving the OLD version out of process memory until something restarts it.
Before this command, nothing did that automatically -- the only guidance was
a manual `tj stop && tj serve &` in the CLI epilog and docs/installation.md.

Three phases, each gated on the previous succeeding:
1. Detect how THIS running `tj` is installed and upgrade it in place.
   Ephemeral/unmanaged installs (uvx/npx/pipx-run cache, or a read-only
   Python) are refused up front -- never half-upgraded.
2. Only on a SUCCESSFUL install, restart the daemon: `launchctl kickstart`
   when launchd supervises it, otherwise reuse `stop_tj_serve()` +
   relaunch `tj serve` in the background. No daemon running is a no-op,
   not a failure.
3. Poll `/api/v1/version` on the daemon's own port until it reports the
   newly installed version, or fail loudly on timeout -- never claim
   success without observing it.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import click

from tokenjam import __version__
from tokenjam.cli.cmd_stop import stop_tj_serve
from tokenjam.cli.cmd_uninstall import (
    _installed_via_pipx,
    _installed_via_uv_tool,
    _is_ephemeral_runner,
)
from tokenjam.cli.tj_status import TjCommand
from tokenjam.core.server_state import (
    ServerState,
    is_pid_alive,
    is_serve_process,
    read_server_state,
)
from tokenjam.utils.formatting import console

_LAUNCHD_LABEL = "com.tokenjam.serve"

# How long to poll the daemon's /api/v1/version for the new version to show
# up after a restart. Generous enough to cover a cold uvicorn boot (imports,
# scheduler start, DB open) without hanging the CLI on a genuine failure.
_VERSION_POLL_TIMEOUT_S = 15.0
_VERSION_POLL_INTERVAL_S = 0.5
_VERSION_HTTP_TIMEOUT_S = 2.0

# How long to wait after relaunching `tj serve` before checking whether the
# child is still alive -- long enough for an immediate failure (bad config,
# port already in use) to have exited, short enough not to meaningfully
# delay the CLI for the common case of a healthy restart.
_RESTART_CHILD_CHECK_S = 0.5


@dataclass(frozen=True)
class UpgradePlan:
    """How to upgrade the persistent install THIS `tj` process is running
    from. `argv` is always safe to run non-interactively -- unmanaged/
    ephemeral installs never produce a plan (see `detect_upgrade_plan`)."""

    manager: str  # "pipx" | "uv-tool" | "pip"
    argv: list[str]
    display: str


def _pip_target_writable() -> bool:
    """Best-effort check that this interpreter's site-packages dir is
    writable, so a system-managed/read-only Python is refused UP FRONT
    rather than attempted and left in a half-upgraded state on failure.

    Walks up to the nearest existing ancestor directory since the exact
    package dir may not exist yet, but its writability is what determines
    whether `pip install --upgrade` can actually write there.
    """
    try:
        import sysconfig

        purelib = sysconfig.get_paths().get("purelib")
    except Exception:
        return False
    if not purelib:
        return False
    target = Path(purelib)
    while not target.exists():
        parent = target.parent
        if parent == target:
            return False
        target = parent
    return os.access(target, os.W_OK)


def detect_upgrade_plan() -> UpgradePlan | None:
    """Return the upgrade plan for how THIS running `tj` is installed, or
    None when it must NOT be auto-upgraded in place (ephemeral uvx/npx/
    pipx-run runner, or an unwritable pip target).

    Derived from `sys.executable` / the resolved install path -- the same
    detection `tj uninstall` already uses (`cmd_uninstall._installed_via_*`)
    -- rather than probing the whole machine, since upgrade targets THIS
    process's own install, not every install that happens to exist.
    """
    if _is_ephemeral_runner():
        return None
    if _installed_via_pipx():
        return UpgradePlan(
            manager="pipx",
            argv=["pipx", "upgrade", "tokenjam"],
            display="pipx upgrade tokenjam",
        )
    if _installed_via_uv_tool():
        return UpgradePlan(
            manager="uv-tool",
            argv=["uv", "tool", "upgrade", "tokenjam"],
            display="uv tool upgrade tokenjam",
        )
    if not _pip_target_writable():
        return None
    return UpgradePlan(
        manager="pip",
        argv=[sys.executable, "-m", "pip", "install", "--upgrade", "tokenjam"],
        display="pip install --upgrade tokenjam",
    )


def _print_refusal() -> None:
    console.print(
        "[yellow]tj is running from an ephemeral or unmanaged install "
        "(uvx / npx / pipx run, or a read-only Python) -- there is no "
        "persistent install here to upgrade in place.[/yellow]"
    )
    console.print("  Install tokenjam persistently first, then re-run `tj upgrade`:")
    console.print("    [bold]uv tool install tokenjam[/bold]  (or)")
    console.print("    [bold]pipx install tokenjam[/bold]")


def run_package_upgrade(plan: UpgradePlan) -> tuple[bool, str]:
    """Run the upgrade command. Never raises -- a subprocess failure to even
    launch (missing binary, etc.) is reported the same way as a non-zero
    exit."""
    console.print(f"Running: [bold]{plan.display}[/bold]")
    try:
        result = subprocess.run(
            plan.argv, capture_output=True, text=True, timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        return False, detail
    return True, (result.stdout.strip() or result.stderr.strip())


def detect_new_version() -> str | None:
    """Best-effort re-read of the installed `tokenjam` version straight from
    disk, after a package upgrade completed in this same process. Distinct
    from the process-level `tokenjam.__version__`, which was resolved once
    at import time and stays pinned to the OLD version for the lifetime of
    this process.
    """
    importlib.invalidate_caches()
    try:
        return importlib.metadata.version("tokenjam")
    except importlib.metadata.PackageNotFoundError:
        return None


def _launchd_loaded() -> bool:
    try:
        result = subprocess.run(
            ["launchctl", "list", _LAUNCHD_LABEL],
            capture_output=True, text=True,
        )
    except OSError:
        return False
    return result.returncode == 0


def _restart_via_launchd() -> tuple[bool, str]:
    """`launchctl kickstart -k` restarts an already-loaded job in place --
    verified live to actually pick up the new binary/version, unlike
    `unload`/`load`, which can race a Disabled flag left over from `tj stop`
    (see cmd_stop.py's `_launchd_label_loaded` note)."""
    uid = os.getuid()
    target = f"gui/{uid}/{_LAUNCHD_LABEL}"
    try:
        result = subprocess.run(
            ["launchctl", "kickstart", "-k", target],
            capture_output=True, text=True,
        )
    except OSError as exc:
        return False, str(exc)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        return False, detail
    return True, f"restarted via `launchctl kickstart -k {target}`"


def _restart_via_stop_serve(config_path: str | None) -> tuple[bool, str]:
    """Fallback for installs not supervised by launchd: reuse `stop_tj_serve`
    (cmd_stop.py) to stop the current daemon, then relaunch `tj serve` as a
    detached background process -- deliberately NOT re-implementing either
    command's own logic.

    `config_path` is the ORIGINAL daemon's config (from the `ServerState`
    read before this restart), so a daemon started with `tj --config <path>
    serve` or `TJ_CONFIG` comes back up against the same config, not
    whatever `tj --config` would resolve by default (a different DB/address/
    port than the one the caller polls afterward).

    `stop_tj_serve`'s result is honored, not discarded: if the old daemon
    could not be confirmed stopped, launching a replacement anyway would run
    two daemons against the same port/DB, so this refuses rather than
    silently doubling up. The relaunched child's exit status is also
    checked after a short delay -- a `Popen` call succeeding only means the
    OS accepted the exec, not that the process is still alive a moment
    later (a bad config or a port already in use exits almost immediately),
    and reporting that as a successful restart would surface as a
    misleading version mismatch instead of the real cause.
    """
    from tokenjam.cli.cmd_onboard import _resolve_tj_binary

    stopped, stopped_via = stop_tj_serve(quiet=True)
    if not stopped:
        detail = "; ".join(stopped_via) if stopped_via else "no confirmation of shutdown"
        return False, f"could not confirm the running daemon stopped ({detail}) -- refusing to start a second one alongside it"

    tj_bin = _resolve_tj_binary()
    argv = [tj_bin]
    if config_path:
        argv += ["--config", config_path]
    argv += ["serve"]

    log_dir = Path.home() / ".local" / "share" / "tj"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        out_f = open(log_dir / "serve-restart.out", "ab")
        err_f = open(log_dir / "serve-restart.err", "ab")
    except OSError as exc:
        return False, str(exc)
    try:
        proc = subprocess.Popen(
            argv,
            stdout=out_f, stderr=err_f, stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        return False, str(exc)
    finally:
        out_f.close()
        err_f.close()

    time.sleep(_RESTART_CHILD_CHECK_S)
    if proc.poll() is not None:
        return False, (
            f"`{' '.join(argv)}` exited immediately (code {proc.returncode}) -- "
            f"see {log_dir / 'serve-restart.err'}"
        )

    return True, f"restarted via `{' '.join(argv)}` in the background"


def restart_daemon(state: ServerState | None) -> tuple[str, bool, str]:
    """Restart the daemon described by `state` (the server.state read BEFORE
    the upgrade ran). Returns (method, success, detail):
    - method="none": no daemon was running -- nothing to restart, not a
      failure.
    - method="launchd": restarted via `launchctl kickstart`.
    - method="stop-serve": restarted via the stop + relaunch fallback.
    """
    daemon_running = (
        state is not None
        and is_pid_alive(state.pid)
        and is_serve_process(state.pid)
    )
    if not daemon_running:
        return "none", True, "no daemon running -- nothing to restart"

    if _launchd_loaded():
        success, detail = _restart_via_launchd()
        return "launchd", success, detail

    success, detail = _restart_via_stop_serve(state.config_path if state else None)
    return "stop-serve", success, detail


def fetch_daemon_version(port: int) -> str | None:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/v1/version", timeout=_VERSION_HTTP_TIMEOUT_S,
        ) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return data.get("version")


def poll_daemon_version(
    port: int,
    expected_version: str,
    *,
    timeout_s: float = _VERSION_POLL_TIMEOUT_S,
    interval_s: float = _VERSION_POLL_INTERVAL_S,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    fetch: Callable[[int], str | None] | None = None,
) -> tuple[bool, str | None]:
    """Poll until the daemon reports `expected_version` or `timeout_s`
    elapses. Returns (verified, last_seen_version). `sleep`/`monotonic`/
    `fetch` are injectable so tests never wait real time or hit a real
    daemon."""
    fetch = fetch or fetch_daemon_version
    start = monotonic()
    seen: str | None = None
    while True:
        seen = fetch(port)
        if seen == expected_version:
            return True, seen
        if monotonic() - start >= timeout_s:
            return False, seen
        sleep(interval_s)


@click.command("upgrade", cls=TjCommand, status_message="Upgrading tj…")
@click.pass_context
def cmd_upgrade(ctx: click.Context) -> None:
    """Upgrade tj and restart the daemon."""
    old_version = __version__

    plan = detect_upgrade_plan()
    if plan is None:
        _print_refusal()
        raise SystemExit(1)

    console.print(f"Current version: [bold]{old_version}[/bold]")

    success, detail = run_package_upgrade(plan)
    if not success:
        console.print(f"[red]Upgrade failed:[/red] {detail}")
        console.print("[dim]Daemon left untouched.[/dim]")
        raise SystemExit(1)

    new_version = detect_new_version()
    console.print(
        f"[green]Package upgraded.[/green] {old_version} -> {new_version or 'unknown'}"
    )

    # Read BEFORE restarting -- the daemon's port doesn't change across a
    # restart, and this is the one snapshot that's guaranteed to describe
    # the daemon we're about to restart (a stop-serve restart rewrites the
    # state file's pid, so reading again afterward would race the new
    # process's own lifespan write).
    state = read_server_state()
    method, restarted, restart_detail = restart_daemon(state)

    if method == "none":
        console.print("[dim]tj serve is not running -- nothing to restart.[/dim]")
        return

    if not restarted:
        console.print(f"[red]Daemon restart failed ({method}):[/red] {restart_detail}")
        raise SystemExit(1)

    console.print(f"[dim]{restart_detail}[/dim]")

    if new_version is None:
        console.print(
            "[yellow]Could not determine the newly installed version -- "
            "skipping restart verification. Check `tj --version` "
            "manually.[/yellow]"
        )
        raise SystemExit(1)

    if state is None or state.port is None:
        console.print(
            "[yellow]Could not determine the daemon's port -- skipping "
            "restart verification. Check `tj --version` manually.[/yellow]"
        )
        raise SystemExit(1)

    verified, seen = poll_daemon_version(state.port, new_version)
    if not verified:
        console.print(
            f"[red]Daemon still reports {seen or 'no version'} after "
            f"restart (expected {new_version}).[/red] Try "
            "`tj stop && tj serve &` manually."
        )
        raise SystemExit(1)

    console.print(f"[green]Daemon verified running {new_version}.[/green]")
