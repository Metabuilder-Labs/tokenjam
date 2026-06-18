"""``tj proxy`` — lifecycle for the optional enforcement-plane proxy (#219).

Suggest mode only. Subcommands:

- ``enable``     — turn the proxy on in config + wire provider base-URLs at it.
- ``disable``    — turn it off + remove the base-URL wiring.
- ``status``     — show config, killswitch, and detected base-URL wiring.
- ``killswitch`` — flip to pass-through-everything (``--off`` to release).

These read/write only config (+ Claude Code's settings env), so ``proxy`` is in
``no_db_commands`` and never opens the DB.
"""
from __future__ import annotations

import json
from pathlib import Path

import click

from tokenjam.core.config import write_config
from tokenjam.proxy.wiring import (
    apply_env_wiring,
    detect_wiring,
    find_orphaned_wiring,
    proxy_base_url,
    remove_env_wiring,
)
from tokenjam.utils.formatting import console


def _config_write_path(config) -> Path:
    """Where to persist proxy config — the active file, else the global config."""
    if getattr(config, "config_path", None):
        return Path(config.config_path)
    return Path.home() / ".config" / "tj" / "config.toml"


@click.group("proxy")
def cmd_proxy() -> None:
    """Manage the optional enforcement-plane proxy (suggest mode)."""


@cmd_proxy.command("enable")
@click.pass_context
def proxy_enable(ctx: click.Context) -> None:
    """Enable the proxy and point provider base-URLs at it."""
    config = ctx.obj["config"]
    config.proxy.enabled = True
    config.proxy.killswitch = False
    path = _config_write_path(config)
    write_config(config, path)
    wired = apply_env_wiring(config)

    url = proxy_base_url(config)
    console.print(f"[green]Proxy enabled[/green] (suggest mode) — will listen on {url}")
    console.print(f"  Config:  {path}")
    console.print(f"  Wired:   {', '.join(wired)} → {url} (~/.claude/settings.json)")
    console.print("[dim]Subscription/unknown traffic is forwarded unmodified "
                  "(observe-only). Restart `tj serve` to start the listener.[/dim]")


@cmd_proxy.command("disable")
@click.pass_context
def proxy_disable(ctx: click.Context) -> None:
    """Disable the proxy and remove the base-URL wiring."""
    config = ctx.obj["config"]
    config.proxy.enabled = False
    config.proxy.killswitch = False
    path = _config_write_path(config)
    write_config(config, path)
    removed = remove_env_wiring(config)

    console.print("[yellow]Proxy disabled[/yellow]")
    console.print(f"  Config:  {path}")
    if removed:
        console.print(f"  Unwired: {', '.join(removed)} (~/.claude/settings.json)")
    console.print("[dim]Restart `tj serve` to stop the listener.[/dim]")


@cmd_proxy.command("killswitch")
@click.option("--off", "release", is_flag=True, help="Release the killswitch.")
@click.pass_context
def proxy_killswitch(ctx: click.Context, release: bool) -> None:
    """Flip the proxy to pass-through-everything (listener stays alive)."""
    config = ctx.obj["config"]
    config.proxy.killswitch = not release
    path = _config_write_path(config)
    write_config(config, path)
    if release:
        console.print("[green]Killswitch released[/green] — normal classification resumes.")
    else:
        console.print("[red]Killswitch engaged[/red] — ALL traffic forwarded "
                      "unmodified (pass-through). Listener stays alive.")
    console.print(f"  Config:  {path}")
    console.print("[dim]Restart `tj serve` for the running listener to pick this up.[/dim]")


@cmd_proxy.command("status")
@click.pass_context
def proxy_status(ctx: click.Context) -> None:
    """Show proxy config + base-URL wiring state."""
    config = ctx.obj["config"]
    p = config.proxy
    wiring = detect_wiring(config)
    orphaned = find_orphaned_wiring(config)
    payload = {
        "enabled": p.enabled,
        "host": p.host,
        "port": p.port,
        "mode": p.mode,
        "killswitch": p.killswitch,
        "base_url": proxy_base_url(config),
        "anthropic_base_url": p.anthropic_base_url,
        "openai_base_url": p.openai_base_url,
        "wiring": wiring,
        "orphaned_wiring": orphaned,
    }
    if ctx.obj.get("output_json"):
        click.echo(json.dumps(payload, indent=2))
        return

    state = "[green]enabled[/green]" if p.enabled else "[dim]disabled[/dim]"
    ks = " [red](killswitch engaged)[/red]" if p.killswitch else ""
    console.print(f"tj proxy: {state}{ks}")
    console.print(f"  Listen:    http://{p.host}:{p.port}  (mode: {p.mode})")
    console.print(f"  Upstreams: anthropic={p.anthropic_base_url}  openai={p.openai_base_url}")
    if wiring:
        console.print(f"  Wiring:    {', '.join(wiring)} → {proxy_base_url(config)}")
    else:
        console.print("  Wiring:    none detected (~/.claude/settings.json)")
    if orphaned:
        console.print(f"  [yellow]Orphaned wiring:[/yellow] {', '.join(orphaned)} "
                      "points at the proxy but it is disabled. "
                      "Run `tj proxy enable` or `tj proxy disable`.")
