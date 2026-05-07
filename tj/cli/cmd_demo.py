"""tj demo — Agent Incident Library CLI command."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import click

from tj.utils.formatting import console

# incidents/ lives two levels above this file (tj/cli/ -> tj/ -> repo/site-packages root)
_INCIDENTS_DIR = Path(__file__).parent.parent.parent / "incidents"


def _discover_scenarios() -> dict[str, ModuleType]:
    """
    Scan incidents/*/scenario.py for modules exposing a `run` callable.
    Returns a dict mapping scenario slug to loaded module.
    """
    scenarios: dict[str, ModuleType] = {}
    if not _INCIDENTS_DIR.exists():
        return scenarios
    for scenario_file in sorted(_INCIDENTS_DIR.glob("*/scenario.py")):
        slug = scenario_file.parent.name
        spec = importlib.util.spec_from_file_location(
            f"incidents.{slug}.scenario", scenario_file
        )
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        if callable(getattr(mod, "run", None)):
            scenarios[slug] = mod
    return scenarios


@click.command("demo")
@click.argument("scenario", required=False, default=None)
@click.option("--json", "output_json", is_flag=True, help="Output JSON instead of Rich panels")
@click.pass_context
def cmd_demo(ctx: click.Context, scenario: str | None, output_json: bool) -> None:
    """Run a reproducible AI agent incident scenario.

    \b
    tj demo                     List available scenarios
    tj demo retry-loop          Run a specific scenario
    tj demo retry-loop --json   Machine-readable output
    """
    scenarios = _discover_scenarios()

    if scenario is None:
        _list_scenarios(scenarios)
        return

    if scenario not in scenarios:
        click.echo(
            f"Unknown scenario '{scenario}'. Run `tj demo` to see available scenarios.",
            err=True,
        )
        raise SystemExit(1)

    scenarios[scenario].run()


def _list_scenarios(scenarios: dict[str, ModuleType]) -> None:
    from rich import box
    from rich.table import Table

    console.print()
    console.print(
        "[bold]OCW Agent Incident Library[/bold]\n"
        "Reproducible AI agent failures — no API keys, no config needed.\n"
    )
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Scenario", style="cyan", no_wrap=True)
    table.add_column("Description")
    for slug, mod in scenarios.items():
        table.add_row(slug, getattr(mod, "DESCRIPTION", ""))
    console.print(table)
    console.print("[dim]Usage:[/dim] tj demo <scenario>  [dim]|[/dim]  tj demo <scenario> --json")
    console.print()
