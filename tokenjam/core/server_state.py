"""The `tj serve` PID file: `~/.local/share/tj/server.state`.

Written by `tj serve` (see `cmd_serve.py`) after uvicorn binds the port, so
other subcommands can find the daemon's config regardless of CWD. `tj stop`
(and, through it, `tj reset` / `tj uninstall`) reads it to locate the daemon
belonging to THIS install -- instead of a bare `pgrep -f`, which matches
every `tj serve` process on the machine, including ones started by a
different install/worktree with a different $HOME.

`Path.home()` is what actually scopes this: it resolves through the
invoking process's $HOME, so two installs with different $HOME never see
each other's state file, and never see each other's daemon.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path


def server_state_path() -> Path:
    return Path.home() / ".local" / "share" / "tj" / "server.state"


@dataclass(frozen=True)
class ServerState:
    pid: int
    port: int | None
    config_path: str | None


@dataclass(frozen=True)
class DaemonUnitState:
    """Observed registration state of TokenJam's per-user service unit."""

    manager: str | None
    path: Path | None
    installed: bool
    loaded: bool | None
    active: bool | None
    error: str | None = None


@dataclass(frozen=True)
class ServeProcess:
    """One machine-visible process whose argv is a ``tj serve`` invocation."""

    pid: int
    command: str


_SYSTEMD_ENABLED_STATES = {
    "enabled", "enabled-runtime",
}
# `linked`/`linked-runtime`/`alias` are deliberately excluded: systemd reports
# those for a unit that is registered but NOT attached to the login target, so
# it will not autostart at the next login. We only ever install via `enable
# --now` (cmd_onboard.py's `_install_systemd`), so this can't fire today, but
# treating them as enabled would tell the user something untrue if the unit
# ever ends up in one of those states by other means.
_SYSTEMD_LIVE_STATES = {"active", "activating", "reloading"}


def inspect_daemon_unit(
    *,
    system: str | None = None,
    home: Path | None = None,
) -> DaemonUnitState:
    """Inspect the service manager, not merely the unit file on disk.

    A launchd plist or systemd unit is only an installation artifact. It does
    not prove the job was loaded/enabled, which is the distinction that leaves
    a machine looking configured while no daemon will start at login.
    """
    current_system = system or platform.system()
    home_dir = home or Path.home()

    if current_system == "Darwin":
        path = home_dir / "Library/LaunchAgents/com.tokenjam.serve.plist"
        if not path.exists():
            return DaemonUnitState("launchd", path, False, False, None)
        try:
            result = subprocess.run(
                ["launchctl", "list", "com.tokenjam.serve"],
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            return DaemonUnitState("launchd", path, True, None, None, str(exc))
        return DaemonUnitState(
            "launchd", path, True, result.returncode == 0, None,
        )

    if current_system == "Linux":
        path = home_dir / ".config/systemd/user/tokenjam.service"
        if not path.exists():
            return DaemonUnitState("systemd", path, False, False, False)
        try:
            enabled = subprocess.run(
                ["systemctl", "--user", "is-enabled", "tokenjam"],
                capture_output=True,
                text=True,
            )
            active = subprocess.run(
                ["systemctl", "--user", "is-active", "tokenjam"],
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            return DaemonUnitState("systemd", path, True, None, None, str(exc))
        enabled_state = enabled.stdout.strip()
        is_enabled = (
            enabled.returncode == 0
            and (not enabled_state or enabled_state in _SYSTEMD_ENABLED_STATES)
        )
        return DaemonUnitState(
            "systemd",
            path,
            True,
            is_enabled,
            active.stdout.strip() in _SYSTEMD_LIVE_STATES,
        )

    return DaemonUnitState(None, None, False, None, None)


def list_serve_processes() -> list[ServeProcess] | None:
    """Return every machine-visible ``tj serve`` process.

    ``server.state`` deliberately scopes destructive commands to one install,
    but doctor needs the inverse, read-only view: a second venv or foreground
    server is exactly the conflict it must disclose. ``None`` means the process
    table could not be inspected; an empty list is a successful observation.
    """
    try:
        result = subprocess.run(
            ["ps", "-ww", "-axo", "pid=,command="],
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None

    processes: list[ServeProcess] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        command = parts[1]
        if pid != os.getpid() and _looks_like_serve(command):
            processes.append(ServeProcess(pid=pid, command=command))
    return processes


def read_server_state(path: Path | None = None) -> ServerState | None:
    """Read + parse the state file. Returns None if it's missing or
    malformed -- a corrupt/partial write (e.g. a crash mid-write) just means
    "nothing found", never a crash here."""
    state_path = path if path is not None else server_state_path()
    try:
        raw = state_path.read_text()
    except OSError:
        return None
    try:
        data = json.loads(raw)
        pid = int(data["pid"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    return ServerState(pid=pid, port=data.get("port"), config_path=data.get("config_path"))


def is_pid_alive(pid: int) -> bool:
    """`os.kill(pid, 0)` sends no signal -- it only checks the PID exists."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Owned by someone else, but it exists.
        return True
    return True


def is_serve_process(pid: int) -> bool:
    """`pid`'s command line still looks like a `tj serve` process.

    Guards against PID reuse: a stale state file's PID may since have been
    recycled by an unrelated process, and we must never signal that.
    """
    proc_cmdline = Path(f"/proc/{pid}/cmdline")
    if proc_cmdline.exists():
        # Prefer /proc on Linux: it's the exact, NUL-separated argv, never
        # truncated by a display width. GNU `ps`'s COMMAND column truncates
        # long lines -- a long interpreter path (e.g. a hostedtoolcache
        # Python) can push "tj serve" past the cut, a real miss on Linux CI.
        try:
            raw = proc_cmdline.read_bytes()
        except OSError:
            return False
        cmdline = " ".join(part for part in raw.decode(errors="replace").split("\0") if part)
        return _looks_like_serve(cmdline)

    try:
        # `-ww`: unlimited width, so this fallback (macOS/BSD, no /proc)
        # can't be truncated the same way the bare form was.
        result = subprocess.run(
            ["ps", "-ww", "-p", str(pid), "-o", "command="],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        # No `ps` on this platform -- fail open on identity (liveness above
        # is still checked) rather than refuse to ever stop anything.
        return True
    if result.returncode != 0:
        return False
    return _looks_like_serve(result.stdout.strip())


def _looks_like_serve(cmdline: str) -> bool:
    """Structural match: is this cmdline a `tj`/`tokenjam` `serve` invocation?

    A plain substring check for `"tj serve"` misses the REAL installed
    daemon: `_daemon_program_args()` (cmd_onboard.py) launches it as
    `tj --config <config_path> serve`, i.e. with `--config <path>` sitting
    between `tj` and `serve` -- no contiguous `"tj serve"` substring ever
    appears in that cmdline. The old synthetic test (a `python -c ... tj
    serve` argv) happened to satisfy the substring check, which is why CI
    never caught this.

    Tokenize instead and check the SHAPE: `serve` must appear as a bare
    argv token (the subcommand, not part of a longer word/flag), and the
    cmdline must otherwise look like a `tj`/`tokenjam` invocation -- either
    the module form (`tokenjam.serve` appears anywhere, e.g. `python -m
    tokenjam.serve`) or some token's basename is exactly `tj`/`tokenjam`
    (covers a direct path to the binary, the plain `tj` program name, and
    wrapper forms like `uv run tj serve` / `uvx --from tokenjam tj
    --config <path> serve` / `pipx run --spec tokenjam tj --config <path>
    serve`, since those all still carry a bare `tj` or `tokenjam` token
    alongside `serve`).
    """
    if "tokenjam.serve" in cmdline:
        return True
    tokens = cmdline.split()
    if "serve" not in tokens:
        return False
    return any(token.rsplit("/", 1)[-1] in ("tj", "tokenjam") for token in tokens)


def find_own_serve_pid() -> int | None:
    """The PID of the `tj serve` daemon belonging to THIS install, or None.

    Validates the PID is alive AND still looks like a `tj serve` process
    before returning it -- so a stale state file (process already dead, or
    its PID recycled by something else) is handled as "nothing running"
    rather than a false positive.
    """
    state = read_server_state()
    if state is None:
        return None
    if state.pid == os.getpid():
        return None
    if not is_pid_alive(state.pid):
        return None
    if not is_serve_process(state.pid):
        return None
    return state.pid
