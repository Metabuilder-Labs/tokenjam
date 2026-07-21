"""`tj relearn` — the self-improve loop's CLI surface.

The group's write verbs (``list`` / ``apply <proposal-id> [--go]`` / ``enable``
/ ``revert``) live in ``cli/relearn_write_verbs.py`` and are attached at the
bottom of this file; this file owns the group itself plus the read-only
``eval-case`` command.
"""
from __future__ import annotations

import json
from pathlib import Path

import click

from tokenjam.core.config import TjConfig
from tokenjam.utils.formatting import console


@click.group("relearn", invoke_without_command=False)
def cmd_relearn() -> None:
    """Self-improve loop: review and apply recurring-mistake fixes."""


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


# --------------------------------------------------------------------------- #
# Write verbs (list / apply / enable / revert) — defined in their own module,
# attached here so `tj relearn` exposes the whole group.
# --------------------------------------------------------------------------- #

from tokenjam.cli.relearn_write_verbs import register_write_verbs  # noqa: E402

register_write_verbs(cmd_relearn)
