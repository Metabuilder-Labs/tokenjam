from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import click

from tokenjam.utils.formatting import console


@click.command("uninstall")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def cmd_uninstall(ctx: click.Context, yes: bool) -> None:
    """Remove all OCW data, config, and daemon."""
    if not yes:
        confirmed = click.confirm(
            "This will delete all OCW data including telemetry history. Continue?",
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
    ocw_dir = Path.home() / ".tj"
    if ocw_dir.exists():
        shutil.rmtree(ocw_dir)
        console.print(f"  Removed {ocw_dir}")

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
    local_ocw = Path(".tj")
    if local_ocw.exists():
        shutil.rmtree(local_ocw)
        console.print(f"  Removed {local_ocw}")

    # 9. Delete temp files
    for tmp_file in ["/tmp/tj-serve.out", "/tmp/tj-serve.err"]:
        p = Path(tmp_file)
        if p.exists():
            p.unlink()
            console.print(f"  Removed {tmp_file}")

    # 10. Remove OCW env vars from ~/.claude/settings.json
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
                global_settings_path.write_text(json.dumps(gs, indent=2) + "\n")
                console.print(f"  Cleaned {len(removed)} OCW env vars from {global_settings_path}")
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

    # 11. Remove # tj harness observability block from ~/.zshrc
    zshrc = Path.home() / ".zshrc"
    if zshrc.exists():
        try:
            text = zshrc.read_text()
            # Match the marker line plus all following export lines (any count)
            cleaned = re.sub(
                r"# tj harness observability\n(?:export [^\n]+\n)*",
                "",
                text,
            )
            if cleaned != text:
                zshrc.write_text(cleaned)
                console.print(f"  Removed OCW env block from {zshrc}")
        except Exception as exc:
            console.print(f"  [yellow]Could not clean {zshrc}: {exc}[/yellow]")

    console.print()
    console.print("[green]TokenJam data and config removed.[/green]")
    console.print("To remove the package itself, run: [bold]pip uninstall tokenjam[/bold]")
