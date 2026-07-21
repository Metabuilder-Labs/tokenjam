"""`tj relearn` — the self-improve loop's CLI surface.

The group's write verbs (``list`` / ``apply <proposal-id> [--go]`` / ``enable``
/ ``revert`` / ``verify``) live in ``cli/relearn_write_verbs.py`` and are
attached at the bottom of this file; this file owns the group itself plus the
read-only ``receipts`` and ``eval-case`` commands.

``receipts`` needs no DB connection — it reads the two on-disk ledgers
(``relearn_apply.list_applied`` / ``cost_apply.list_applied``) the same way
the Review inbox's ``GET /relearn/applied`` and ``GET /relearn/cost-applied``
endpoints do, then combines them via ``core.optimize.receipts`` — the exact
function ``GET /relearn/receipts`` calls server-side, so the CLI and the web
UI can never disagree about the number.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from tokenjam.core.config import TjConfig
from tokenjam.utils.formatting import console, format_cost


@click.group("relearn", invoke_without_command=False)
def cmd_relearn() -> None:
    """Self-improve loop: review, apply, and verify recurring-mistake fixes."""


# --------------------------------------------------------------------------- #
# Read-only receipts. The write verbs are attached from
# ``cli/relearn_write_verbs.py`` at the bottom of this file.
# --------------------------------------------------------------------------- #

@cmd_relearn.command("receipts")
@click.option("--json", "output_json", is_flag=True, help="Emit machine-readable JSON.")
@click.pass_context
def cmd_relearn_receipts(ctx: click.Context, output_json: bool) -> None:
    """Cumulative verified-saved receipts across the relearn + cost ledgers.

    The measured twin of the Review inbox's estimated-recoverable rollup
    (Component E). Regressed / no-change / insufficient-data fixes are shown
    here, not hidden — the honesty is the feature.
    """
    from tokenjam.core.optimize import cost_apply, receipts, relearn_apply

    config: TjConfig = ctx.obj["config"]
    relearn_records = relearn_apply.list_applied(config)
    cost_records = cost_apply.list_applied(config)
    summary = receipts.verified_saved_summary(relearn_records, cost_records)

    if output_json or ctx.obj.get("output_json", False):
        click.echo(json.dumps(summary, indent=2))
        return

    _render_receipts(summary)


@cmd_relearn.command("eval-case")
@click.argument("proposal_id")
@click.option("--out", "out_path", default=None,
              help="Write the JSON here instead of printing it.")
@click.pass_context
def cmd_relearn_eval_case(ctx: click.Context, proposal_id: str, out_path: str | None) -> None:
    """Emit the eval-case JSON artifact for a stored PROPOSAL_ID.

    The advise lane's hand-off. tokenjam cannot apply a fix into an agent it
    has no workspace for, so an advise-only proposal has no apply path at all;
    this hands back the same clustered evidence in a plain JSON shape you can
    feed your own eval tooling as a regression case. Read-only: it writes
    nothing but the file you name.
    """
    from tokenjam.core.optimize import relearn_proposals
    from tokenjam.core.optimize.relearn_otel import to_eval_case

    config: TjConfig = ctx.obj["config"]
    stored = relearn_proposals.get_proposal(proposal_id, config=config)
    if stored is None:
        raise click.ClickException(
            f"no stored proposal {proposal_id}. Run `tj relearn list` for the "
            f"IDs the detector actually produced."
        )
    case = to_eval_case(relearn_proposals.relearn_cluster_from(stored))
    payload = json.dumps(case, indent=2, default=str)

    if out_path:
        Path(out_path).write_text(payload + "\n", encoding="utf-8")
        console.print(f"[green]wrote[/] {out_path}")
        return
    click.echo(payload)


def _render_receipts(summary: dict[str, Any]) -> None:
    if summary["verified_count"] == 0:
        console.print(
            "[dim]No fixes verified yet. Apply a relearn fix or mark a cost "
            "proposal applied, then check back once enough post-apply "
            "exposure accumulates.[/dim]"
        )
        return

    console.print(
        f"[bold]{format_cost(summary['verified_saved_usd'])}[/bold] verified saved to date "
        f"[dim](measured · {summary['improved_count']} improved fix(es))[/dim]"
    )
    console.print(
        f"[dim]+ {summary['verified_saved_tokens']:,} tok saved "
        f"({summary['relearn_tokens_saved']:,} from relearn fixes, "
        f"{summary['cost_tokens_saved']:,} from cost fixes)[/dim]"
    )
    console.print(
        f"[dim]{summary['verified_count']} checked · {summary['improved_count']} improved · "
        f"{summary['regressed_count']} regressed · {summary['no_change_count']} no change · "
        f"{summary['enforcement_disabled_count']} awaiting enforcement · "
        f"{summary['insufficient_data_count']} still measuring[/dim]"
    )
    console.print(f"[dim]{summary['estimate_basis']}[/dim]")


# --------------------------------------------------------------------------- #
# Write verbs (list / apply / enable / revert / verify) — defined in their own
# module, attached here so `tj relearn` exposes the whole group.
# --------------------------------------------------------------------------- #

from tokenjam.cli.relearn_write_verbs import register_write_verbs  # noqa: E402

register_write_verbs(cmd_relearn)
