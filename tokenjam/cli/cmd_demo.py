"""tj demo — Agent Incident Library CLI command."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import click

from tokenjam.utils.formatting import console

# The scenarios live at repo-root `incidents/` in the dev tree, but ship INSIDE
# the package at `tokenjam/incidents/` in a built wheel (via the hatchling
# force-include in pyproject.toml — repo-root `incidents/` is outside the
# `tokenjam/` package, so it is never wheeled on its own). Resolve both and use
# whichever exists: installed location first, dev-tree fallback (#291). Scenarios
# load by file path (spec_from_file_location), so the physical location doesn't
# affect the import — both candidates work.
def _candidate_incidents_dirs() -> list[Path]:
    here = Path(__file__).resolve().parent  # …/tokenjam/cli
    return [
        here.parent / "incidents",         # installed: …/tokenjam/incidents (force-included)
        here.parent.parent / "incidents",  # dev tree: repo-root incidents/
    ]


def _incidents_dir() -> Path | None:
    """The first existing scenarios dir, or None when none are present."""
    for candidate in _candidate_incidents_dirs():
        if candidate.exists():
            return candidate
    return None


def _discover_scenarios() -> dict[str, ModuleType]:
    """
    Scan incidents/*/scenario.py for modules exposing a `run` callable.
    Returns a dict mapping scenario slug to loaded module.
    """
    scenarios: dict[str, ModuleType] = {}
    base = _incidents_dir()
    if base is None:
        return scenarios
    for scenario_file in sorted(base.glob("*/scenario.py")):
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
        "[bold]TokenJam Agent Incident Library[/bold]\n"
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
