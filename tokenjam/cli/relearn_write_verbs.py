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
and revert). No detection, no rung routing, no ledger logic lives in this file,
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
    without it; only the active-session guard reads it.
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
# Registration
# --------------------------------------------------------------------------- #

#: Every verb this module contributes to the ``tj relearn`` group, in the order
#: they should appear.
WRITE_VERBS = (list_cmd, apply_cmd, enable_cmd, revert_cmd)


def register_write_verbs(group: click.Group) -> click.Group:
    """Attach this module's verbs to the ``tj relearn`` group and return it.

    The group itself is defined in ``cli/cmd_relearn.py``; this keeps the write
    side out of that file so the two halves of the group stay independently
    editable. Returns the group so a caller can chain or assert on it.
    """
    for command in WRITE_VERBS:
        group.add_command(command)
    return group
