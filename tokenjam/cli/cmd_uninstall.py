from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import click

from tokenjam.utils.formatting import console


def _installed_via_pipx() -> bool:
    """True when this tj lives in a pipx-managed venv.

    A pipx install lives at ~/.local/share/pipx/venvs/tokenjam/... — detecting
    that is how we tell pipx installs apart from a plain pip / venv install.
    """
    return "pipx/venvs/" in sys.executable.replace("\\", "/")


def _installed_via_uv_tool() -> bool:
    """True when this tj lives in a `uv tool install`-managed venv.

    A persistent uv tool install lives at .../uv/tools/tokenjam/... (verified:
    `uv tool install cowsay` puts its venv at ~/.local/share/uv/tools/cowsay/).
    That's distinct from the ephemeral venvs `uvx`/`uv tool run` spin up, which
    live under uv's *cache* dir instead — see `_is_ephemeral_runner()`.
    """
    return "/uv/tools/" in sys.executable.replace("\\", "/")


def _is_ephemeral_runner() -> bool:
    """True when this process is a throwaway venv from `uvx`/`uv tool run` or
    `pipx run` — there is no persistent tokenjam install to remove.

    Verified empirically: `uvx --from tokenjam tj ...` materializes its venv
    under uv's cache dir (.../uv/archive-<rev>/<hash>/...), never under
    .../uv/tools/. `pipx run` lands in the same place when pipx delegates to
    uv (its default backend whenever uv is on PATH); pipx's own non-uv
    fallback instead uses its run-cache at .../pipx/.cache/<hash>/.

    Deliberately narrow: matching on a bare "/uv/" substring also caught
    uv-managed Python *runtimes* (.../uv/python/cpython-.../bin/python3) and
    uv-tool installs (.../uv/tools/...) — both persistent installs, not
    ephemeral ones — which silently no-opped `tj uninstall`'s package-removal
    step for those users. Only uv's cache dirs (archive/builds, or the
    generic .cache/uv/ prefix) count as ephemeral; uv doesn't pin the
    "archive-v0" suffix forever, so match the prefix rather than the exact
    revision.
    """
    if _installed_via_pipx() or _installed_via_uv_tool():
        return False
    exe = sys.executable.replace("\\", "/")
    if "/uv/python/" in exe or "/uv/tools/" in exe:
        return False
    is_uv_cache = "/uv/archive-" in exe or "/uv/builds-" in exe or ".cache/uv/" in exe
    return is_uv_cache or "/pipx/.cache/" in exe


def _package_uninstall_hint() -> str:
    """Return the right uninstall command for how the user installed.

    pipx and `uv tool` are both isolated, canonical install paths (per
    README and docs/installation.md); otherwise fall back to `pip uninstall`
    for venv / pip installs.
    """
    if _installed_via_pipx():
        return "pipx uninstall tokenjam"
    if _installed_via_uv_tool():
        return "uv tool uninstall tokenjam"
    return "pip uninstall tokenjam"


def _package_reinstall_hint() -> str:
    """Return the command that actually reinstalls FRESH.

    A plain `pipx install tokenjam` / `uv tool install tokenjam` /
    `pip install tokenjam` no-ops when the package is already present — the
    source of the confusing "fresh install" that silently does nothing. Point
    pipx users at `pipx upgrade` (or `pipx install --force`), uv tool users at
    `uv tool upgrade` (or `uv tool install --force`), and pip users at
    `pip install --upgrade`.
    """
    if _installed_via_pipx():
        return "pipx upgrade tokenjam  (or: pipx install --force tokenjam)"
    if _installed_via_uv_tool():
        return "uv tool upgrade tokenjam  (or: uv tool install --force tokenjam)"
    return "pip install --upgrade tokenjam"


def _package_fresh_install_hint(manager: str | None = None) -> str:
    """Return the command that reinstalls after the package was fully removed.

    Once the venv is gone, `pipx upgrade` / `uv tool upgrade` fail ("not
    installed") — the right command is the plain install form (a fresh
    install, no venv to upgrade). This is distinct from
    `_package_reinstall_hint()`, which is only correct while the package
    REMAINS in place (a plain install would no-op then, so it points at
    upgrade/--force).

    `manager` overrides the current-process auto-detection (one of
    `PersistentInstall.manager`'s values: "pipx" / "uv-tool" / "pip") — pass
    the manager of the install that was just removed via
    `_find_persistent_install()`. This matters because `tj uninstall` almost
    always runs from an ephemeral `uvx`/`pipx run` process (the npx wrapper's
    default), whose OWN `sys.executable` never matches the persistent
    pipx/uv-tool install it just removed on the user's behalf.
    """
    if manager is None:
        if _installed_via_pipx():
            manager = "pipx"
        elif _installed_via_uv_tool():
            manager = "uv-tool"
        else:
            manager = "pip"
    if manager == "pipx":
        return "pipx install tokenjam"
    if manager == "uv-tool":
        return "uv tool install tokenjam"
    return "pip install tokenjam"


@dataclass(frozen=True)
class PersistentInstall:
    """One persistent tokenjam install found somewhere on the machine —
    independent of how the CURRENT `tj` process happens to be running (see
    `_find_persistent_install()`)."""

    manager: str              # "pipx" | "uv-tool" | "pip"
    auto: bool                # True: safe to auto-run `argv` non-interactively
    argv: list[str] | None    # subprocess argv when auto=True, else None
    display: str              # human-readable command, for prompts + printing


_MANAGER_LABEL = {"pipx": "pipx", "uv-tool": "uv tool", "pip": "pip"}


def _pipx_home() -> Path:
    """pipx's data dir — `PIPX_HOME` overrides the default `~/.local/pipx`."""
    override = os.environ.get("PIPX_HOME")
    return Path(override) if override else Path.home() / ".local" / "pipx"


def _uv_data_home() -> Path:
    """uv's data dir — `XDG_DATA_HOME` overrides the default `~/.local/share`."""
    override = os.environ.get("XDG_DATA_HOME")
    base = Path(override) if override else Path.home() / ".local" / "share"
    return base / "uv"


def _pipx_has_tokenjam() -> bool:
    """Authoritative: `pipx list --json` reports a `tokenjam` venv. Falls
    back to a dir probe (`${PIPX_HOME:-~/.local/pipx}/venvs/tokenjam`) only
    when the `pipx` binary is missing or the command errors — e.g. probing
    from an ephemeral `uvx`/`pipx run` process that has no `pipx` on its own
    PATH, even though a persistent pipx install exists elsewhere on the
    machine."""
    pipx_bin = shutil.which("pipx")
    if pipx_bin:
        try:
            result = subprocess.run(
                [pipx_bin, "list", "--json"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return "tokenjam" in data.get("venvs", {})
        except Exception:
            pass
    return (_pipx_home() / "venvs" / "tokenjam").exists()


def _uv_tool_has_tokenjam() -> bool:
    """Authoritative: `uv tool list` reports a `tokenjam` tool. Falls back to
    a dir probe (`${XDG_DATA_HOME:-~/.local/share}/uv/tools/tokenjam`) only
    when `uv` is missing or the command errors. `uv tool list` has no --json
    output today, so text-parse it — only unindented lines that don't start
    with `-` are package names (indented `- <entrypoint>` lines list each
    tool's installed scripts, not other packages)."""
    uv_bin = shutil.which("uv")
    if uv_bin:
        try:
            result = subprocess.run(
                [uv_bin, "tool", "list"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if not line or line[0].isspace() or line.lstrip().startswith("-"):
                        continue
                    if line.split()[0] == "tokenjam":
                        return True
                return False
        except Exception:
            pass
    return (_uv_data_home() / "tools" / "tokenjam").exists()


def _pip_tj_on_path() -> str | None:
    """Return a persistent `tj` on PATH that is a plain pip / editable-dev
    install — i.e. neither pipx- nor uv-tool-managed (those are already
    covered by `_pipx_has_tokenjam()`/`_uv_tool_has_tokenjam()`) nor an
    ephemeral cache shim (`uvx`/`pipx run` — nothing to remove there).
    None if `tj` isn't found on PATH, or it resolves to one of those other
    categories."""
    tj_path = shutil.which("tj")
    if not tj_path:
        return None
    norm = tj_path.replace("\\", "/")
    if "pipx/venvs/" in norm or "/uv/tools/" in norm:
        return None
    if (
        "/uv/archive-" in norm or "/uv/builds-" in norm
        or ".cache/uv/" in norm or "/pipx/.cache/" in norm
    ):
        return None
    return tj_path


def _find_persistent_install() -> list[PersistentInstall]:
    """Probe the ENVIRONMENT — not the current process — for every
    persistent tokenjam install on the machine.

    `npx tokenjam uninstall` always runs `tj` via an ephemeral `uvx`/
    `pipx run` venv (see npm-wrapper/bin/tj.js's runner preference), so the
    `sys.executable`-based helpers above (`_installed_via_pipx()`,
    `_is_ephemeral_runner()`, etc.) always report "this process is
    ephemeral" on that path — the WRONG signal for what `tj uninstall`
    should remove. A user who separately `pipx install`ed tokenjam still has
    that install sitting on disk; only probing the environment (not
    `sys.executable`) finds it, so it can actually be removed from within an
    ephemeral `npx tokenjam uninstall` run instead of silently leaving it
    behind while claiming success.

    Checks, in order: `pipx list --json` (dir-probe fallback), `uv tool
    list` (dir-probe fallback), then a plain `tj` on PATH that is neither of
    those managers nor an ephemeral cache shim (a plain pip install or an
    editable dev checkout). Returns EVERY match, not just the first — pipx
    and uv-tool installs can coexist on one machine, and `tj uninstall`
    removes every auto-removable one it finds.
    """
    installs: list[PersistentInstall] = []
    if _pipx_has_tokenjam():
        installs.append(PersistentInstall(
            manager="pipx", auto=True,
            argv=["pipx", "uninstall", "tokenjam"],
            display="pipx uninstall tokenjam",
        ))
    if _uv_tool_has_tokenjam():
        installs.append(PersistentInstall(
            manager="uv-tool", auto=True,
            argv=["uv", "tool", "uninstall", "tokenjam"],
            display="uv tool uninstall tokenjam",
        ))
    if _pip_tj_on_path():
        installs.append(PersistentInstall(
            manager="pip", auto=False, argv=None,
            display="pip uninstall tokenjam",
        ))
    return installs


def _uninstall_confirm_prompt(installs: list[PersistentInstall]) -> str:
    """Build the `tj uninstall` confirm prompt so it states EXACTLY what will
    happen. Detection (`_find_persistent_install()`) runs BEFORE this is
    called (see `cmd_uninstall`) — the prompt must never promise a package
    removal that won't actually happen, or fail to mention one that will
    (Greptile P1 on #443)."""
    base = (
        "This will delete all TokenJam data (config, telemetry history, "
        "daemon, shell wiring)"
    )
    auto = [i for i in installs if i.auto]
    manual = [i for i in installs if not i.auto]
    clauses = []
    if auto:
        labels = " and ".join(_MANAGER_LABEL[i.manager] for i in auto)
        clauses.append(f"and uninstall the tokenjam package (via {labels})")
    if manual:
        cmds = " / ".join(i.display for i in manual)
        clauses.append(f"your package install will need `{cmds}` (shown after)")
    if clauses:
        base += ", " + "; ".join(clauses)
    return base + ". Continue?"


@click.command("uninstall")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def cmd_uninstall(ctx: click.Context, yes: bool) -> None:
    """Remove TokenJam entirely: config/daemon/wiring AND the tokenjam package.

    The full symmetric counterpart to `tj onboard`. For a config-only reset
    that leaves the tokenjam CLI installed (e.g. to reconfigure or pause),
    use `tj reset` instead.

    Package detection is environment-wide (`_find_persistent_install()`),
    not based on the current process — `tj uninstall` almost always runs via
    an ephemeral `uvx`/`pipx run` venv (the npx wrapper's default), and that
    process's own `sys.executable` says nothing about a persistent pipx/
    uv-tool install sitting elsewhere on the machine.
    """
    # Detect FIRST so the confirmation prompt states exactly what will
    # happen — never promising a package removal that won't occur, or vice
    # versa.
    installs = _find_persistent_install()

    if not yes:
        confirmed = click.confirm(_uninstall_confirm_prompt(installs), default=False)
        if not confirmed:
            console.print("[dim]Cancelled.[/dim]")
            return

    _teardown_side_effects(ctx)

    if not installs:
        console.print(
            "[dim]No persistent tokenjam install found on this machine — "
            "nothing to remove; the config/wiring cleanup above is already "
            "done.[/dim]"
        )
        return

    console.print()
    console.print("[dim]Removing the tokenjam package...[/dim]")
    for install in installs:
        _remove_persistent_install(install)


def _teardown_side_effects(ctx: click.Context) -> None:
    """Reverse everything `tj onboard` wires up — stop/unregister the daemon,
    deregister the Claude Code MCP server, delete TokenJam's config/DB/temp
    files, and strip the tj-managed `~/.zshrc` OTEL block + `claude()`
    wrapper + per-project `settings.json` env vars. Does NOT touch the
    tokenjam package itself.

    Shared by `tj uninstall` (this, plus removing the package) and `tj reset`
    (this only — package stays installed, ready for `tj onboard` again).
    """
    # 1. Stop tj serve if running
    from tokenjam.cli.cmd_stop import cmd_stop
    ctx.invoke(cmd_stop)

    # 2. Deregister MCP server from Claude Code (Gap #13)
    if shutil.which("claude"):
        subprocess.run(
            ["claude", "mcp", "remove", "tj", "--scope", "user"],
            capture_output=True, text=True,
        )
        console.print("  Removed tj MCP server from Claude Code.")

    # 3. Unload and delete launchd plist
    plist_path = Path.home() / "Library/LaunchAgents/com.tokenjam.serve.plist"
    if plist_path.exists():
        subprocess.run(
            ["launchctl", "unload", str(plist_path)],
            capture_output=True, text=True,
        )
        plist_path.unlink()
        console.print(f"  Removed {plist_path}")

    # 4. Delete systemd service if present
    systemd_path = Path.home() / ".config/systemd/user/tokenjam.service"
    if systemd_path.exists():
        subprocess.run(
            ["systemctl", "--user", "disable", "--now", "tokenjam"],
            capture_output=True, text=True,
        )
        systemd_path.unlink()
        console.print(f"  Removed {systemd_path}")

    # 5. Delete ~/.tj/ (telemetry DB)
    tj_dir = Path.home() / ".tj"
    if tj_dir.exists():
        shutil.rmtree(tj_dir)
        console.print(f"  Removed {tj_dir}")

    # 6. Read projects index BEFORE deleting the global config dir.
    global_config_dir = Path.home() / ".config" / "tj"
    project_paths: list[Path] = []
    projects_index = global_config_dir / "projects.json"
    try:
        if projects_index.exists():
            paths = json.loads(projects_index.read_text())
            project_paths = [Path(p) for p in paths if isinstance(p, str)]
    except Exception:
        pass

    # 7. Delete global config ~/.config/tj/
    if global_config_dir.exists():
        shutil.rmtree(global_config_dir)
        console.print(f"  Removed {global_config_dir}")

    # 8. Delete local .tj/ if present
    local_tj = Path(".tj")
    if local_tj.exists():
        shutil.rmtree(local_tj)
        console.print(f"  Removed {local_tj}")

    # 9. Delete temp files
    for tmp_file in ["/tmp/tj-serve.out", "/tmp/tj-serve.err"]:
        p = Path(tmp_file)
        if p.exists():
            p.unlink()
            console.print(f"  Removed {tmp_file}")

    # 10. Remove TokenJam env vars from ~/.claude/settings.json
    _GLOBAL_TJ_KEYS = {
        "CLAUDE_CODE_ENABLE_TELEMETRY",
        "OTEL_LOGS_EXPORTER",
        "OTEL_EXPORTER_OTLP_PROTOCOL",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_HEADERS",
    }
    global_settings_path = Path.home() / ".claude" / "settings.json"
    if global_settings_path.exists():
        try:
            gs = json.loads(global_settings_path.read_text())
            env = gs.get("env", {})
            removed = [k for k in _GLOBAL_TJ_KEYS if k in env]
            for k in removed:
                del env[k]
            if removed:
                gs["env"] = env
            # Remove the tj-managed SessionStart resume-brief hook and the
            # opt-in PostToolUse output-cap hook (both idempotent,
            # non-destructive — foreign hooks of either kind are preserved).
            from tokenjam.cli.cmd_onboard import (
                _unwire_claude_output_cap_hook,
                _unwire_claude_resume_brief_hook,
            )
            hook_removed = _unwire_claude_resume_brief_hook(gs)
            cap_removed = _unwire_claude_output_cap_hook(gs)
            if removed or hook_removed or cap_removed:
                global_settings_path.write_text(json.dumps(gs, indent=2) + "\n")
            if removed:
                console.print(f"  Cleaned {len(removed)} TokenJam env vars from {global_settings_path}")
            if hook_removed:
                console.print(f"  Removed tj resume-brief SessionStart hook from {global_settings_path}")
            if cap_removed:
                console.print(f"  Removed tj output-cap PostToolUse hook from {global_settings_path}")
        except Exception as exc:
            console.print(f"  [yellow]Could not clean {global_settings_path}: {exc}[/yellow]")

    # 11. Remove OTEL_RESOURCE_ATTRIBUTES from all onboarded project .claude/settings.json files.
    # project_paths was read from projects.json before the global config dir was deleted above.
    # Always include CWD so running uninstall from a project dir works even without the index
    cwd = Path.cwd()
    if cwd not in project_paths:
        project_paths.append(cwd)

    for proj in project_paths:
        proj_settings = proj / ".claude" / "settings.json"
        if not proj_settings.exists():
            continue
        try:
            ps = json.loads(proj_settings.read_text())
            env = ps.get("env", {})
            if "OTEL_RESOURCE_ATTRIBUTES" in env:
                del env["OTEL_RESOURCE_ATTRIBUTES"]
                ps["env"] = env
                proj_settings.write_text(json.dumps(ps, indent=2) + "\n")
                console.print(f"  Removed OTEL_RESOURCE_ATTRIBUTES from {proj_settings}")
        except Exception as exc:
            console.print(f"  [yellow]Could not clean {proj_settings}: {exc}[/yellow]")

    # 11. Remove tj-managed OTEL block(s) from ~/.zshrc — current sentinel AND
    # any legacy marker (#118), so an install that accumulated blocks under an
    # older marker (pre-rebrand or pre-sentinel) gets fully cleaned up, not
    # just the current-marker block.
    zshrc = Path.home() / ".zshrc"
    if zshrc.exists():
        try:
            from tokenjam.cli.cmd_onboard import _strip_zshrc_otel_blocks
            text = zshrc.read_text()
            cleaned = _strip_zshrc_otel_blocks(text)
            if cleaned != text:
                zshrc.write_text(cleaned)
                console.print(f"  Removed TokenJam env block from {zshrc}")
        except Exception as exc:
            console.print(f"  [yellow]Could not clean {zshrc}: {exc}[/yellow]")

    # Remove the `claude()` shell wrapper (`# tj per-terminal naming` ..
    # `# end tj per-terminal naming`) from ~/.zshrc / ~/.bashrc. This is the
    # counterpart to `_install_claude_wrapper()` in cmd_onboard.py — without
    # it, a leftover `claude()` function calls `tj` on every launch, which
    # errors once the package is removed (#117).
    from tokenjam.cli.cmd_onboard import _unwire_claude_wrapper
    try:
        wrapper_removed = _unwire_claude_wrapper()
        for rc_path in wrapper_removed:
            console.print(f"  Removed claude() shell wrapper from {rc_path}")
    except Exception as exc:
        console.print(f"  [yellow]Could not clean claude() shell wrapper: {exc}[/yellow]")

    console.print()
    console.print("[green]TokenJam data, config, and wiring removed.[/green]")


def _run_auto_uninstall(argv: list[str], uninstall_cmd: str, manager: str) -> None:
    """Shared runner for the pipx/uv-tool auto-remove branches: run `argv`,
    then report success (with the fresh-install hint for `manager`) or fall
    back to the manual command on failure."""
    console.print()
    console.print(f"Running: [bold]{uninstall_cmd}[/bold]")
    result = subprocess.run(argv, capture_output=True, text=True)
    if result.returncode == 0:
        console.print("[green]tokenjam package removed.[/green]")
        # The venv is gone now, so upgrade would fail ("not installed") —
        # point at the plain fresh install, not the upgrade/--force reinstall
        # hint (#430). `manager` is the install that was JUST removed, not
        # necessarily what the current (often ephemeral) process would guess.
        console.print(
            f"  [dim]To reinstall FRESH: "
            f"{_package_fresh_install_hint(manager)}[/dim]"
        )
    else:
        console.print(
            f"  [yellow]Could not remove the package automatically: "
            f"{result.stderr.strip() or result.stdout.strip()}[/yellow]"
        )
        console.print(f"  Run manually: [bold]{uninstall_cmd}[/bold]")


def _remove_persistent_install(install: PersistentInstall) -> None:
    """Execute (pipx/uv-tool) or print (pip/editable) removal for one
    detected persistent install (`_find_persistent_install()`).

    Only pipx and `uv tool` are auto-run: both are isolated, canonical
    install paths safe to invoke non-interactively. For pip / editable-dev
    installs we print the command instead of guessing — a wrong
    `pip uninstall` in the wrong (possibly shared/live) environment is worse
    than a copy-paste."""
    if install.auto and install.argv:
        _run_auto_uninstall(install.argv, install.display, install.manager)
        return

    console.print()
    console.print(
        "  [yellow]Can't safely auto-remove this install "
        "(not a pipx- or uv-tool-managed venv).[/yellow]"
    )
    console.print(f"  Remove the package with: [bold]{install.display}[/bold]")
