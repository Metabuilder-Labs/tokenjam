"""``tj relearn`` write verbs -- the terminal write path for the loop.

Until now the only way to approve a proposal was the Lens Review inbox: a
browser, on a machine with a display, talking to a running ``tj serve``. A
terminal-first or headless install could detect its recurring failures and
then do nothing about them. These verbs close that gap.

They live in their own module, apart from the ``tj relearn`` group itself, so
the write side and the read side of the group can be developed without one
owning the other's file. ``cmd_relearn`` attaches them with::

    from tokenjam.cli.relearn_write_verbs import register_write_verbs
    register_write_verbs(cmd_relearn)

Thin by construction. Every command here is a wrapper over the SAME functions
the API route calls (``core.optimize.relearn_proposals`` for the stored
proposal, ``core.optimize.relearn_apply`` for the write, backup, git commit
and revert, ``core.optimize.relearn_verify`` / ``cost_verify`` for the receipt
recompute). No detection, no rung routing, no ledger logic lives in this file,
so the CLI and the UI can never drift into two different meanings of "apply".

The human gate is unchanged and unconditional:

  * ``apply`` is a DRY RUN unless you pass ``--go``; the dry run prints the
    exact diff that would be written.
  * ``enable`` needs ``--yes``, because wiring a hook into settings.json means
    it starts intercepting tool calls.
  * every write is reversible with ``revert``.
"""
from __future__ import annotations

import json

import click

from tokenjam.core.optimize import relearn_apply, relearn_proposals
from tokenjam.utils.formatting import console, make_table


def _config(ctx: click.Context):
    config = ctx.obj.get("config")
    if config is None:
        raise click.ClickException("no config loaded.")
    return config


def _conn(ctx: click.Context):
    """The live DuckDB connection, or None when the daemon holds the lock and
    the CLI fell back to the HTTP backend. Everything here degrades gracefully
    without it; only the active-session guard and the verify recompute read it.
    """
    db = ctx.obj.get("db")
    return getattr(db, "conn", None) if db is not None else None


def _emit(ctx: click.Context, payload: dict) -> bool:
    """Echo ``payload`` as JSON when ``--json`` is set. Returns True when it
    did, so callers can skip their human rendering."""
    if ctx.obj.get("output_json"):
        click.echo(json.dumps(payload, default=str))
        return True
    return False


# --------------------------------------------------------------------------- #
# F1: list / apply / enable / revert
# --------------------------------------------------------------------------- #

@click.command("list")
@click.pass_context
def list_cmd(ctx: click.Context) -> None:
    """List the stored proposals, with the IDs `tj relearn apply` takes."""
    proposals = relearn_proposals.list_proposals(_config(ctx))
    if _emit(ctx, {"proposals": proposals, "count": len(proposals)}):
        return
    if not proposals:
        console.print(
            "[dim]No proposals stored yet. The detector runs on a schedule "
            "inside `tj serve`; give it a pass over your sessions first.[/dim]"
        )
        return
    table = make_table("ID", "TITLE", "SESSIONS", "RUNG", "SCOPE", "APPLY")
    for p in proposals:
        table.add_row(
            str(p.get("proposal_id") or ""),
            str(p.get("title") or p.get("signature") or ""),
            str(p.get("sessions") or 0),
            str(p.get("rung") or ""),
            str(p.get("scope") or ""),
            "advise-only" if p.get("advise_only") else "workspace fix",
        )
    console.print(table)
    console.print(
        "[dim]Preview one with [bold]tj relearn apply <id>[/bold]; "
        "write it with [bold]--go[/bold].[/dim]"
    )
    # State the seam rather than leaving the user to infer it from an apply
    # that refuses. An advise-only proposal has no file to be written into.
    if any(p.get("advise_only") for p in proposals):
        console.print(f"[dim]{relearn_proposals.ADVISE_ONLY_REASON}[/dim]")


@click.command("apply")
@click.argument("proposal_id")
@click.option("--go", is_flag=True, help="Actually write the fix (default is a dry run).")
@click.option("--target", "target_path", default=None,
              help="Where to write it. Defaults to the proposal's suggested target.")
@click.option("--scope", type=click.Choice(["project", "user-global"]), default=None,
              help="Override the proposal's scope.")
@click.option("--force", is_flag=True,
              help="Apply even though a session looks live in the target repo.")
@click.pass_context
def apply_cmd(ctx, proposal_id, go, target_path, scope, force):
    """Preview (default) or write the fix for a stored PROPOSAL_ID."""
    config = _config(ctx)
    stored = relearn_proposals.get_proposal(proposal_id, config=config)
    if stored is None:
        raise click.ClickException(
            f"no stored proposal {proposal_id}. Run `tj relearn list` for the "
            f"IDs the detector actually produced."
        )
    target = (target_path or stored.get("suggested_target") or "").strip()
    if not target:
        # An advise-only proposal has no target because there is no workspace,
        # not because the detector failed to guess one. Say which.
        raise click.ClickException(
            relearn_proposals.ADVISE_ONLY_REASON
            if stored.get("advise_only")
            else "this proposal has no suggested target path. Pass one with --target."
        )
    cluster = relearn_proposals.cluster_for_apply(stored)
    try:
        result = relearn_apply.apply_relearn_fix(
            config, cluster,
            target_path=target,
            scope=scope or stored.get("scope") or "project",
            go=go, conn=_conn(ctx), force=force,
        )
    except relearn_apply.RelearnApplyRefused as exc:
        raise click.ClickException(str(exc)) from exc

    if _emit(ctx, result):
        return
    if result.get("dry_run"):
        console.print(f"[bold]{stored.get('title') or proposal_id}[/]")
        console.print(f"[dim]would write {result['kind']} to {result['target_path']}[/dim]")
        if result.get("diff"):
            console.print(result["diff"])
        console.print("[dim]Dry run. Nothing was written; re-run with --go to apply.[/dim]")
        return
    rec = result["record"]
    console.print(
        f"[green]applied[/] {rec['kind']} to {rec['target_path']} "
        f"[dim](fix {rec['id']})[/dim]"
    )
    if rec.get("git_commit"):
        console.print(f"[dim]committed as {rec['git_commit'][:12]}[/dim]")
    if rec.get("enforcement"):
        console.print(
            f"[dim]Staged disabled. Arm it with "
            f"[bold]tj relearn enable {rec['id']} --yes[/bold].[/dim]"
        )
    console.print(f"[dim]Undo with [bold]tj relearn revert {rec['id']}[/bold].[/dim]")


@click.command("enable")
@click.argument("fix_id")
@click.option("--yes", is_flag=True,
              help="Confirm wiring this hook into settings.json.")
@click.pass_context
def enable_cmd(ctx, fix_id, yes):
    """Wire an applied enforcement fix into settings.json (needs --yes)."""
    try:
        rec = relearn_apply.enable_enforcement(_config(ctx), fix_id, confirm=yes)
    except relearn_apply.RelearnApplyRefused as exc:
        raise click.ClickException(str(exc)) from exc
    if _emit(ctx, rec):
        return
    console.print(
        f"[green]enabled[/] {rec['title']} "
        f"[dim](disable again with tj relearn revert {fix_id})[/dim]"
    )


@click.command("revert")
@click.argument("fix_id")
@click.pass_context
def revert_cmd(ctx, fix_id):
    """Undo an applied fix: unwire it if live, then restore the file."""
    try:
        rec = relearn_apply.revert_applied_fix(_config(ctx), fix_id)
    except relearn_apply.RelearnApplyRefused as exc:
        raise click.ClickException(str(exc)) from exc
    if _emit(ctx, rec):
        return
    console.print(f"[green]reverted[/] {rec.get('title') or fix_id}")
    if rec.get("revert_commit"):
        console.print(f"[dim]committed as {rec['revert_commit'][:12]}[/dim]")


# --------------------------------------------------------------------------- #
# G2: on-demand verify. The verify pass otherwise runs only on the daemon's
# six-hour schedule, so a fresh apply had no way to produce its receipt on
# demand. This recomputes both ledgers now, against the same functions the
# scheduled pass calls.
# --------------------------------------------------------------------------- #

def _verify_otel(ctx, proposal_id, expectation_id, *, record: bool) -> None:
    """The OTel lane of verify: measure one advise-only proposal's failure
    signature against stored spans, before and after the moment its fix went
    live, and write the outcome into the same loop ledger the rest of the loop
    reads.

    Why an expectation supplies the marker: a workspace-less agent has no apply
    path, so tokenjam never wrote the fix and cannot know when it shipped. The
    expectation's created_at IS that declaration, and its agent_id scopes the
    measurement. Create one first with `tj loop expect`.
    """
    from tokenjam.core.loop import get_expectation
    from tokenjam.core.optimize.relearn_otel_verify import verify_otel_expectation

    if not proposal_id or not expectation_id:
        raise click.ClickException(
            "--otel and --expectation go together: name the proposal to "
            "measure and the expectation whose created_at marks when its fix "
            "went live."
        )
    conn = _conn(ctx)
    if conn is None:
        raise click.ClickException(
            "the span measurement needs the database, and the daemon is "
            "holding it. Stop it with `tj stop` and re-run."
        )
    stored = relearn_proposals.get_proposal(proposal_id, config=_config(ctx))
    if stored is None:
        raise click.ClickException(
            f"no stored proposal {proposal_id}. Run `tj relearn list` for the "
            f"IDs the detector actually produced."
        )
    expectation = get_expectation(ctx.obj.get("db"), expectation_id)
    if expectation is None:
        raise click.ClickException(
            f"no expectation {expectation_id}. Create one at the moment the "
            f"fix goes live with `tj loop expect`."
        )

    result = verify_otel_expectation(
        ctx.obj.get("db"), expectation,
        signature=stored.get("signature"),
        family_key=stored.get("family_key"),
        record=record,
    )
    result["proposal_id"] = proposal_id
    result["expectation_id"] = expectation_id
    if _emit(ctx, result):
        return

    console.print(
        f"[bold]{result.get('verdict', 'unknown')}[/bold] "
        f"{stored.get('title') or stored.get('signature') or proposal_id}"
    )
    console.print(f"[dim]{result.get('reason', '')}[/dim]")
    if result.get("fix_marker_at"):
        console.print(
            f"[dim]{result.get('pre_occurrences', 0)} occurrence(s) over "
            f"{result.get('pre_sessions', 0)} session(s) before "
            f"{result['fix_marker_at']}; "
            f"{result.get('recurrence_since_apply', 0)} over "
            f"{result.get('post_sessions_since_apply', 0)} after.[/dim]"
        )
    if result.get("run_ledger_id"):
        console.print(f"[dim]recorded in the loop ledger as {result['run_ledger_id']}[/dim]")
    elif record:
        console.print(
            "[dim]Nothing written to the ledger: only a measured drop or a "
            "measured non-drop is decisive enough to record.[/dim]"
        )
    if result.get("estimate_basis"):
        console.print(f"[dim]{result['estimate_basis']}[/dim]")


@click.command("verify")
@click.option("--otel", "otel_proposal_id", default=None,
              help="Verify one advise-only proposal against stored spans "
                   "instead of recomputing the on-disk receipts.")
@click.option("--expectation", "expectation_id", default=None,
              help="The expectation whose created_at marks when the fix went "
                   "live. Required with --otel.")
@click.option("--no-record", is_flag=True,
              help="Measure without writing the outcome to the loop ledger.")
@click.pass_context
def verify_cmd(ctx, otel_proposal_id, expectation_id, no_record):
    """Recompute the verify receipts now instead of waiting for the schedule."""
    from tokenjam.core.optimize import cost_verify, relearn_verify

    config = _config(ctx)
    if otel_proposal_id or expectation_id:
        _verify_otel(ctx, otel_proposal_id, expectation_id, record=not no_record)
        return
    conn = _conn(ctx)
    fixes = relearn_verify.rescan_all(config, conn)
    costs = {"checked": 0, "updated": 0}
    if conn is not None:
        costs = cost_verify.rescan_all(ctx.obj.get("db"), config)
    payload = {"fixes": fixes, "cost_fixes": costs, "measured_against_db": conn is not None}
    if _emit(ctx, payload):
        return
    console.print(
        f"[green]verified[/] {fixes['updated']}/{fixes['checked']} applied fixes, "
        f"{costs['updated']}/{costs['checked']} cost fixes."
    )
    if conn is None:
        console.print(
            "[dim]The daemon holds the database, so the cost receipts were "
            "skipped. Stop it with `tj stop` and re-run for those.[/dim]"
        )
    console.print("[dim]Read the results with [bold]tj relearn list[/bold] or the Review inbox.[/dim]")


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #

#: Every verb this module contributes to the ``tj relearn`` group, in the order
#: they should appear.
WRITE_VERBS = (list_cmd, apply_cmd, enable_cmd, revert_cmd, verify_cmd)


def register_write_verbs(group: click.Group) -> click.Group:
    """Attach this module's verbs to the ``tj relearn`` group and return it.

    The group itself is defined in ``cli/cmd_relearn.py``; this keeps the write
    side out of that file so the two halves of the group stay independently
    editable. Returns the group so a caller can chain or assert on it.
    """
    for command in WRITE_VERBS:
        group.add_command(command)
    return group
