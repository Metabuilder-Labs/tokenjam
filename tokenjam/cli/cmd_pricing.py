import json

import click

from tokenjam.core.pricing import (
    PRICING_FILE,
    _override_raw_sources,
    _split_pricing_raw,
    load_pricing_table,
)
from tokenjam.utils.formatting import console, make_table

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]


def _source_map() -> dict[tuple[str, str], str]:
    """Map each (provider, model) to where its rate resolved from.

    Mirrors the precedence in pricing._build_pricing(): override layers win
    over the packaged models.toml, which wins over the built-in default. A
    model present in an override layer is labelled "override"; one present
    only in the packaged table is "packaged"; anything else falls through to
    "default" (handled by the caller for models not seen in either layer).
    """
    sources: dict[tuple[str, str], str] = {}

    # Packaged table (the base layer) -> "packaged".
    with open(PRICING_FILE, "rb") as fh:
        packaged_providers, _ = _split_pricing_raw(tomllib.load(fh))
    for provider, models in packaged_providers.items():
        for model_name in models:
            sources[(provider, model_name)] = "packaged"

    # Override layers win over the packaged table -> "override".
    for raw in _override_raw_sources():
        override_providers, _ = _split_pricing_raw(raw)
        for provider, models in override_providers.items():
            for model_name in models:
                sources[(provider, model_name)] = "override"

    return sources


@click.group("pricing", invoke_without_command=False)
def cmd_pricing() -> None:
    """Inspect the resolved model pricing table (read-only)."""


@cmd_pricing.command("list")
@click.option("--model", default=None,
              help="Filter to model names containing this substring.")
@click.option("--json", "output_json_flag", is_flag=True,
              help="Emit machine-readable JSON.")
@click.pass_context
def cmd_pricing_list(ctx: click.Context, model: str | None,
                     output_json_flag: bool) -> None:
    """List the resolved per-model rates (input/output/cache, in $/MTok)."""
    # Honour either `tj --json pricing list` or `tj pricing list --json`.
    output_json: bool = output_json_flag or ctx.obj.get("output_json", False)

    table = load_pricing_table()  # {provider: {model: ModelRates}}
    source_map = _source_map()

    # Flatten the nested table into one row per (provider, model), optionally
    # filtered by a case-insensitive substring match on the model name.
    rows = []
    for provider in sorted(table):
        for model_name in sorted(table[provider]):
            if model and model.lower() not in model_name.lower():
                continue
            rates = table[provider][model_name]
            rows.append({
                "provider": provider,
                "model": model_name,
                "input_per_mtok": rates.input_per_mtok,
                "output_per_mtok": rates.output_per_mtok,
                "cache_read_per_mtok": rates.cache_read_per_mtok,
                "cache_write_per_mtok": rates.cache_write_per_mtok,
                "source": source_map.get((provider, model_name), "default"),
            })

    if output_json:
        click.echo(json.dumps(rows, indent=2))
        return

    if not rows:
        msg = (f"No pricing entries match '{model}'." if model
               else "No pricing entries found.")
        console.print(f"[dim]{msg}[/dim]")
        return

    t = make_table("PROVIDER", "MODEL", "INPUT", "OUTPUT",
                   "CACHE READ", "CACHE WRITE", "SOURCE")
    for row in rows:
        t.add_row(
            row["provider"],
            row["model"],
            f"{row['input_per_mtok']:.2f}",
            f"{row['output_per_mtok']:.2f}",
            f"{row['cache_read_per_mtok']:.2f}",
            f"{row['cache_write_per_mtok']:.2f}",
            row["source"],
        )
    console.print(t)
    console.print("[dim]Rates are USD per million tokens (MTok).[/dim]")
