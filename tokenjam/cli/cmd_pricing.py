import json

import click

from tokenjam.core.pricing import load_pricing_sources, load_pricing_table
from tokenjam.utils.formatting import console, make_table


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
    sources = load_pricing_sources()  # {(provider, model): "override"|"packaged"}

    # Flatten the nested table into one row per (provider, model), optionally
    # filtered by a case-insensitive substring match on the model name. Every
    # listed model is in the resolved table, so its source is "packaged" or
    # "override" (the built-in default only applies to models absent from the
    # table, which never appear here); "packaged" is a defensive fallback.
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
                "source": sources.get((provider, model_name), "packaged"),
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
