from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import click

from tokenjam.utils.formatting import console


def _installed_via_pipx() -> bool:
    """True when this tj lives in a pipx-managed venv.

    A pipx install lives at ~/.local/share/pipx/venvs/tokenjam/... — detecting
    that is how we tell pipx installs apart from a plain pip / venv install.
    """
    return "pipx/venvs/" in sys.executable.replace("\\", "/")


def _package_uninstall_hint() -> str:
    """Return the right uninstall command for how the user installed.

    pipx is the canonical install path (per README and docs/installation.md);
    otherwise fall back to `pip uninstall` for venv / pip installs.
    """
    return "pipx uninstall tokenjam" if _installed_via_pipx() else "pip uninstall tokenjam"


def _package_reinstall_hint() -> str:
    """Return the command that actually reinstalls FRESH.

    A plain `pipx install tokenjam` / `pip install tokenjam` no-ops when the
    package is already present — the source of the confusing "fresh install"
    that silently does nothing. Point pipx users at `pipx upgrade` (or
    `pipx install --force`) and pip users at `pip install --upgrade`.
    """
    if _installed_via_pipx():
        return "pipx upgrade tokenjam  (or: pipx install --force tokenjam)"
    return "pip install --upgrade tokenjam"


def _package_fresh_install_hint() -> str:
    """Return the command that reinstalls after the package was fully removed.

    Once the venv is gone, `pipx upgrade tokenjam` fails ("not installed") — the
    right command is the plain `pipx install tokenjam` (a fresh install, no venv
    to upgrade). This is distinct from `_package_reinstall_hint()`, which is only
    correct while the package REMAINS in place (a plain install would no-op then,
    so it points at upgrade/--force).
    """
    return (
        "pipx install tokenjam" if _installed_via_pipx()
        else "pip install tokenjam"
    )


@click.command("uninstall")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
@click.option(
    "--remove-package",
    is_flag=True,
    help="Also remove the tokenjam package: runs the uninstall for "
    "pipx-managed installs, or prints the command to run for pip/venv "
    "installs. With --yes, proceeds without a second prompt.",
)
@click.pass_context
def cmd_uninstall(ctx: click.Context, yes: bool, remove_package: bool) -> None:
    """Remove all TokenJam data, config, and daemon."""
    if not yes:
        confirmed = click.confirm(
            "This will delete all TokenJam data including telemetry history. Continue?",
            default=False,
        )
        if not confirmed:
            console.print("[dim]Cancelled.[/dim]")
            return

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
    console.print("[dim]The tokenjam package itself is still installed.[/dim]")

    # 12. Optionally remove the package too (safe, opt-in, default No).
    uninstall_cmd = _package_uninstall_hint()
    do_remove = remove_package
    if not do_remove and not yes:
        # pipx installs are auto-run; pip/venv installs only get the command
        # printed (a wrong `pip uninstall` in the wrong env is worse), so word
        # the prompt to match what will actually happen (#430).
        prompt_action = (
            f"runs {uninstall_cmd}" if _installed_via_pipx()
            else f"prints {uninstall_cmd} for you to run"
        )
        do_remove = click.confirm(
            f"Also remove the tokenjam package now? ({prompt_action})",
            default=False,
        )

    if do_remove:
        _remove_package(uninstall_cmd)
        return

    # Package left in place: spell out the two-step so a later reinstall isn't
    # a silent no-op (a plain `pipx/pip install` no-ops when already present).
    console.print()
    console.print("Next steps:")
    console.print(f"  Fully remove the package:  [bold]{uninstall_cmd}[/bold]")
    console.print(f"  Reinstall FRESH:           [bold]{_package_reinstall_hint()}[/bold]")
    console.print(
        "  [dim]A plain `install` no-ops when tokenjam is already present — "
        "use the upgrade/--force form above to reinstall.[/dim]"
    )


def _remove_package(uninstall_cmd: str) -> None:
    """Run the package uninstall when we can, else print the exact command.

    Only pipx is auto-run: it's the canonical install path and safe to invoke
    non-interactively. For pip / venv installs we print the command instead of
    guessing (a wrong `pip uninstall` in the wrong environment is worse than a
    copy-paste)."""
    if _installed_via_pipx() and shutil.which("pipx"):
        console.print()
        console.print(f"Running: [bold]{uninstall_cmd}[/bold]")
        result = subprocess.run(
            ["pipx", "uninstall", "tokenjam"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            console.print("[green]tokenjam package removed.[/green]")
            # The venv is gone now, so `pipx upgrade` would fail — point at the
            # plain fresh install, not the upgrade/--force reinstall hint (#430).
            console.print(
                f"  [dim]To reinstall FRESH: {_package_fresh_install_hint()}[/dim]"
            )
        else:
            console.print(
                f"  [yellow]Could not remove the package automatically: "
                f"{result.stderr.strip() or result.stdout.strip()}[/yellow]"
            )
            console.print(f"  Run manually: [bold]{uninstall_cmd}[/bold]")
        return

    # Not a pipx install (or pipx missing) — print the right command, don't guess.
    console.print()
    console.print(
        "  [yellow]Can't safely auto-remove this install "
        "(not a pipx-managed venv).[/yellow]"
    )
    console.print(f"  Remove the package with: [bold]{uninstall_cmd}[/bold]")
