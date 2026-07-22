"""``tj relearn cost-proposals`` (+ ``cost-apply`` / ``cost-mark-applied`` /
``cost-revert``) -- the terminal view of the cost-analyzer proposals
(downsize/cache/trim/subagent/deadweight/script/reuse/verbosity) that,
before this file, had no CLI renderer at all.

``cmd_optimize.py`` renders each analyzer's raw FINDING (a different, older
shape); it never touches the proposal layer at all. The proposal layer
(``core.optimize.cost_proposals`` -> ``core.optimize.relearn_proposals
.list_cost_proposals``) is what the web Review inbox's cards are built from
-- evidence, a recommendation, and (for most analyzers) a copy-pasteable
``suggestion`` snippet that is the actual fix. A terminal user got the
diagnosis and never saw the fix. These verbs close that gap by reading the
SAME store the API route (``GET /api/v1/relearn/cost-proposals``) serves,
never duplicating its logic.

Flat verbs, siblings of ``tj relearn list`` / ``apply`` / ``enable`` /
``revert`` in the same group -- not a nested subgroup, matching this
codebase's existing "one level deep" CLI shape. Thin by construction: every
write here is a wrapper over ``core.optimize.cost_apply`` (mark/revert) or
``core.optimize.relearn_apply`` (the same workspace-write machinery
``tj relearn apply`` uses), never a second copy of that logic.

Two kinds of cost proposal, rendered honestly as two different things:

  * ``apply_capable`` (subagent / script / reuse / verbosity, some of the
    time) -- tokenjam CAN write the fix into a workspace file. ``cost-apply``
    does that, dry-run by default, exactly like ``tj relearn apply``.
  * advise-only (most cards: downsize, cache, trim, deadweight) -- the fix
    lives in the user's OWN application code. There is no apply path; the
    ``suggestion`` snippet printed with the card IS the fix. Presenting it as
    a consolation ("sorry, no auto-apply") would misstate what happened here
    -- for these analyzers a hand-edit was always the intended fix.
    ``cost-mark-applied`` just records that the user made the change, so the
    delta-verify pass has a "changed at T" marker to measure against.
"""
from __future__ import annotations

import json

import click

from tokenjam.core.framing import (
    Framing,
    WindowSummary,
    compute_framing,
    plan_determination_mix,
    render_savings,
)
from tokenjam.core.optimize import cost_apply, relearn_apply, relearn_proposals, relearn_store
from tokenjam.utils.formatting import console


def _config(ctx: click.Context):
    config = ctx.obj.get("config")
    if config is None:
        raise click.ClickException("no config loaded.")
    return config


def _conn(ctx: click.Context):
    """The live DuckDB connection, or None when the daemon holds the lock and
    the CLI fell back to the HTTP backend."""
    db = ctx.obj.get("db")
    return getattr(db, "conn", None) if db is not None else None


def _emit(ctx: click.Context, payload: dict) -> bool:
    if ctx.obj.get("output_json"):
        click.echo(json.dumps(payload, default=str))
        return True
    return False


def _framing_for(ctx: click.Context) -> Framing:
    """The same plan-tier framing chain ``GET /relearn/cost-proposals`` uses
    (``core.framing.compute_framing`` off a window-INDEPENDENT plan mix,
    since cost-proposal figures are cumulative-to-date, not scoped to a
    --since window) -- so a dollar figure rendered here never disagrees with
    the web Review inbox's card for the same proposal. Degrades to the
    config-declared plan when there is no direct DB connection, exactly as
    ``compute_framing`` already handles an empty mix."""
    config = _config(ctx)
    conn = _conn(ctx)
    mix = plan_determination_mix(conn) if conn is not None else {}
    return compute_framing(config, WindowSummary(plan_tier_mix=mix, sessions=sum(mix.values())))


def _stored_cost_proposal(config, proposal_id: str) -> dict | None:
    """The stored cost proposal with this ID, scoped to the cost-proposal
    store only -- never a relearn cluster ID accepted here by accident.
    Mirrors ``api/routes/relearn.py``'s ``_stored_cost_proposal`` without
    importing across the CLI/API boundary (the two are kept in sync only by
    both reading ``relearn_proposals.list_cost_proposals``)."""
    for proposal in relearn_proposals.list_cost_proposals(config):
        if proposal.get("proposal_id") == proposal_id:
            return proposal
    return None


# --------------------------------------------------------------------------- #
# cost-proposals: the list/render verb.
# --------------------------------------------------------------------------- #

@click.command("cost-proposals")
@click.pass_context
def cost_proposals_cmd(ctx: click.Context) -> None:
    """List cost-saving fixes (downsize/cache/trim/subagent/deadweight/
    script/reuse/verbosity), each with the evidence and the copy-pasteable
    snippet or workspace-apply path it carries."""
    config = _config(ctx)
    block = relearn_store.read_cost_proposals(config=config)
    proposals = relearn_proposals.list_cost_proposals(config)
    framing = _framing_for(ctx)

    if _emit(ctx, {
        "status": "ready" if block is not None else "never_run",
        "computed_at": (block or {}).get("cost_computed_at"),
        "proposals": proposals,
        "framing": framing.to_dict(),
    }):
        return

    if block is None:
        console.print(
            "[dim]Cost proposals have never been computed. Run "
            "[bold]tj optimize[/bold] to compute them.[/dim]"
        )
        return

    if not proposals:
        computed_at = str(block.get("cost_computed_at") or "")[:16].replace("T", " ")
        console.print(
            f"[dim]No cost-saving fixes found as of {computed_at or 'the last pass'}. "
            f"Either usage doesn't match any analyzer's pattern yet, or there "
            f"isn't enough history. Run [bold]tj optimize[/bold] again after "
            f"more usage accrues.[/dim]"
        )
        return

    applied_sigs = {
        rec.get("signature") for rec in cost_apply.list_applied(config)
        if rec.get("state") != "reverted"
    }
    open_proposals = [p for p in proposals if p.get("signature") not in applied_sigs]

    if open_proposals:
        from tokenjam.core.optimize.cost_proposals import estimated_recoverable_rollup
        rollup = estimated_recoverable_rollup(open_proposals)
        headline = render_savings(
            rollup.get("estimated_recoverable_usd"),
            rollup.get("estimated_recoverable_tokens"),
            framing,
        )
        if headline != "—":
            console.print(
                f"[bold green]~{headline}[/bold green] estimated recoverable across "
                f"{rollup.get('proposal_count', 0)} of "
                f"{rollup.get('deduplicated_proposal_count', 0)} open proposal(s) "
                f"[dim](estimated, correlational)[/dim]\n"
            )

    for i, p in enumerate(proposals, start=1):
        _render_cost_proposal(p, framing, i, applied=p.get("signature") in applied_sigs)
        console.print()

    if any(not p.get("apply_capable") for p in proposals):
        console.print(
            "[dim]Advise-only proposals have no apply path: the fix lives in "
            "your own application code, which tokenjam has no workspace to "
            "write into. The snippet above IS the fix -- copy it in, then "
            "[bold]tj relearn cost-mark-applied <id>[/bold] to record "
            "it.[/dim]"
        )


def _render_cost_proposal(
    p: dict, framing: Framing, index: int, *, applied: bool = False,
) -> None:
    pid = p.get("proposal_id") or ""
    title = p.get("title") or p.get("signature") or pid
    analyzer = p.get("analyzer") or ""
    if applied:
        badge = "applied"
    elif p.get("apply_capable"):
        badge = "workspace fix"
    else:
        badge = "advise-only"

    console.print(
        f"[bold]{index}.[/bold] [bold]{title}[/bold] "
        f"[dim]({analyzer} · {badge})[/dim]  [dim]{pid}[/dim]"
    )
    if p.get("evidence"):
        console.print(f"     {p['evidence']}")

    savings = render_savings(
        p.get("estimated_recoverable_usd"), p.get("estimated_recoverable_tokens"), framing,
    )
    if savings != "—":
        console.print(
            f"     [green]~{savings}[/green] estimated recoverable "
            f"[dim](estimate: {p.get('estimate_basis') or 'correlational'})[/dim]"
        )

    if p.get("advise_text"):
        console.print(f"     [dim]Recommendation:[/dim] {p['advise_text']}")

    # The copy-pasteable fix: its own line, no markup/highlighting applied and
    # no wrapping, so a command like `claude mcp remove foo --scope user` can
    # be selected cleanly out of the terminal.
    snippet = p.get("suggestion") or p.get("one_paste_fix") or ""
    if snippet:
        console.print("     [dim]Fix:[/dim]")
        console.print(snippet, markup=False, highlight=False, soft_wrap=True)

    if applied:
        pass
    elif p.get("apply_capable"):
        console.print(
            f"     [yellow]→[/yellow] workspace fix available: "
            f"[bold]tj relearn cost-apply {pid}[/bold] "
            f"[dim](dry run; add --go to write)[/dim]"
        )
    else:
        reason = p.get("apply_blocked_reason") or (
            "no workspace tokenjam can write this fix into -- apply the "
            "snippet above yourself"
        )
        console.print(f"     [dim]{reason}[/dim]")

    if p.get("caveat"):
        console.print(f"     [yellow]![/yellow] [italic]{p['caveat']}[/italic]")


# --------------------------------------------------------------------------- #
# cost-apply: the workspace-write verb, apply_capable proposals only.
# --------------------------------------------------------------------------- #

@click.command("cost-apply")
@click.argument("proposal_id")
@click.option("--go", is_flag=True, help="Actually write the fix (default is a dry run).")
@click.option("--target", "target_path", default=None,
              help="Where to write it. Defaults to the proposal's target path.")
@click.option("--scope", type=click.Choice(["project", "user-global"]), default=None,
              help="Override the proposal's scope.")
@click.option("--force", is_flag=True,
              help="Apply even though a session looks live in the target repo.")
@click.pass_context
def cost_apply_cmd(
    ctx: click.Context, proposal_id: str, go: bool,
    target_path: str | None, scope: str | None, force: bool,
) -> None:
    """Preview (default) or write the workspace fix for an apply-capable
    stored cost PROPOSAL_ID (subagent / script / reuse / verbosity).

    Most cost proposals (downsize, cache, trim, deadweight) are advise-only
    and have no workspace fix for this command to write -- their fix is the
    snippet `tj relearn cost-proposals` prints; apply it yourself, then
    record it with `tj relearn cost-mark-applied`.
    """
    config = _config(ctx)
    stored = _stored_cost_proposal(config, proposal_id)
    if stored is None:
        raise click.ClickException(
            f"no stored cost proposal {proposal_id}. Run "
            f"`tj relearn cost-proposals` for the IDs the detector actually produced."
        )
    if not stored.get("apply_capable"):
        raise click.ClickException(
            stored.get("apply_blocked_reason")
            or "this proposal is advise-only: it has no workspace tokenjam "
               "can write into. Apply its snippet yourself, then run "
               f"`tj relearn cost-mark-applied {proposal_id}`."
        )
    target = (target_path or stored.get("target_path") or "").strip()
    if not target:
        raise click.ClickException(
            "this proposal has no suggested target path. Pass one with --target."
        )

    analyzer = str(stored.get("analyzer") or "")
    baseline = dict(stored.get("baseline") or {})
    # The cluster shape `relearn_apply.apply_relearn_fix` renders a rung-1/2
    # note/skill from, built the same way `POST
    # /relearn/cost-proposals/apply-workspace` builds it
    # (api/routes/relearn.py) -- duplicated here rather than imported,
    # because the CLI must not import across into the API layer.
    cluster = {
        "signature": str(stored.get("signature") or ""),
        "family_key": f"cost_{analyzer}" if analyzer else "cost_proposal",
        "title": str(stored.get("title") or "") or str(stored.get("signature") or ""),
        "proposed_fix": str(stored.get("proposed_fix") or ""),
        "rung": int(stored.get("rung") or 1),
        "sessions": int(
            baseline.get("apply_sessions", baseline.get("flagged_subagents", 0)) or 0
        ),
        "repos": list(baseline.get("apply_repos") or []),
        "examples": list(baseline.get("apply_examples") or []),
    }
    try:
        result = relearn_apply.apply_relearn_fix(
            config, cluster, target_path=target,
            scope=scope or stored.get("scope") or "project",
            go=go, conn=_conn(ctx), force=force,
        )
    except relearn_apply.RelearnApplyRefused as exc:
        raise click.ClickException(str(exc)) from exc

    # Real write happened: drop the cost marker so the realized delta is
    # measured against this moment, mirroring the API route's
    # apply-then-mark sequencing.
    cost_record = None
    if go and not result.get("dry_run"):
        db = ctx.obj.get("db")
        if db is not None:
            try:
                cost_record = cost_apply.mark_applied(db, config, stored)
            except cost_apply.CostApplyRefused:
                cost_record = None

    if _emit(ctx, {"applied": result, "cost_record": cost_record}):
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
    if cost_record:
        console.print(
            f"[dim]cost proposal marked applied (record {cost_record['id']}); "
            f"undo with `tj relearn cost-revert {cost_record['id']}`.[/dim]"
        )
    console.print(f"[dim]Undo the workspace write with `tj relearn revert {rec['id']}`.[/dim]")


# --------------------------------------------------------------------------- #
# cost-mark-applied / cost-revert: the advise-only bookkeeping verbs.
# --------------------------------------------------------------------------- #

@click.command("cost-mark-applied")
@click.argument("proposal_id")
@click.pass_context
def cost_mark_applied_cmd(ctx: click.Context, proposal_id: str) -> None:
    """Record that you applied an advise-only cost PROPOSAL_ID yourself.

    No code write -- cost proposals are advise-only by default; this only
    creates the fix marker the delta-verify pass measures the realized
    change against.
    """
    config = _config(ctx)
    stored = _stored_cost_proposal(config, proposal_id)
    if stored is None:
        raise click.ClickException(
            f"no stored cost proposal {proposal_id}. Run "
            f"`tj relearn cost-proposals` for the IDs the detector actually produced."
        )
    db = ctx.obj.get("db")
    if db is None or _conn(ctx) is None:
        raise click.ClickException(
            "cost-mark-applied requires a direct database connection "
            "(stop `tj serve` first, or mark it from the web Review inbox instead)."
        )
    try:
        rec = cost_apply.mark_applied(db, config, stored)
    except cost_apply.CostApplyRefused as exc:
        raise click.ClickException(str(exc)) from exc
    if _emit(ctx, rec):
        return
    console.print(
        f"[green]marked applied[/] {rec.get('title') or proposal_id} "
        f"[dim](record {rec['id']})[/dim]"
    )
    console.print(f"[dim]Undo with `tj relearn cost-revert {rec['id']}`.[/dim]")


@click.command("cost-revert")
@click.argument("record_id")
@click.pass_context
def cost_revert_cmd(ctx: click.Context, record_id: str) -> None:
    """Undo a marked/applied cost fix RECORD_ID.

    Ledger-only for advise-only proposals (there's no file to restore); a
    workspace fix written by `cost-apply` should be undone with
    `tj relearn revert` instead, which restores the actual file.
    """
    config = _config(ctx)
    try:
        rec = cost_apply.revert_applied(config, record_id)
    except cost_apply.CostApplyRefused as exc:
        raise click.ClickException(str(exc)) from exc
    if _emit(ctx, rec):
        return
    console.print(f"[green]reverted[/] {rec.get('title') or record_id}")


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #

#: Every verb this module contributes to the ``tj relearn`` group, in the
#: order they should appear.
COST_PROPOSAL_VERBS = (
    cost_proposals_cmd, cost_apply_cmd, cost_mark_applied_cmd, cost_revert_cmd,
)


def register_cost_proposal_verbs(group: click.Group) -> click.Group:
    """Attach this module's verbs to the ``tj relearn`` group and return it."""
    for command in COST_PROPOSAL_VERBS:
        group.add_command(command)
    return group
