"""`tj rules` — the permanent rules the analyzers are offering, and where they go.

Four analyzers (`downsize`, `resend`, `subagent`, `relearn`) end in the same
artifact: a block appended to a CLAUDE.md. `tj rules list` shows each one with
the files it would land in; `stage` renders one diff per destination; `check`
re-verifies them against the files as they stand now; `apply` writes (default
**dry-run**, `--go` writes, gzip backup first); `undo` reverts.

Config-only — it reads the proposal cache the optimize pass already wrote, so
it is in `no_db_commands` and works while `tj serve` holds the DuckDB lock.
Answering "what rules are on offer" must never trigger an analyzer sweep.
"""
from __future__ import annotations

import json

import click
from rich.markup import escape

from tokenjam.cli.json_option import json_option, resolve_output_json
from tokenjam.core.config import TjConfig
from tokenjam.core.rulewrite import (
    RuleWriteRefused,
    apply_staged,
    check_staged,
    find_rule,
    list_rule_writes,
    stage_rule,
    undo,
)
from tokenjam.core.rulewrite import store
from tokenjam.core.rulewrite.delivery import DEFAULT_DELIVERY, DELIVERY_KINDS
from tokenjam.core.rulewrite.legacy import UNRESOLVED_DELIVERY_LABEL
from tokenjam.utils.formatting import console, format_tokens

# Honesty discipline (Critical Rule 14). A rule is a candidate an agent will
# read, never a guaranteed change in behaviour, and the figure beside it is the
# analyzer's own past-tense observation — not a promise about what writing the
# rule returns.
RULES_NOTE = (
    "Each figure is what the flagged behaviour already cost over the analyzed "
    "window, observed and estimated. Writing a rule is a change to what your "
    "agent reads, not a guaranteed saving. Review the diff before applying."
)


def _print_diff(diff: str) -> None:
    """Unified diff, +/- coloured. The dry-run preview of a staged write."""
    for line in diff.splitlines():
        if line.startswith(("+++", "---", "@@")):
            console.print(f"[muted]{escape(line)}[/muted]")
        elif line.startswith("+"):
            console.print(f"[ok]{escape(line)}[/ok]")
        elif line.startswith("-"):
            console.print(f"[warn]{escape(line)}[/warn]")
        else:
            console.print(f"[muted]{escape(line)}[/muted]")


@click.group("rules", invoke_without_command=False)
def cmd_rules() -> None:
    """Propose permanent CLAUDE.md rules."""


@cmd_rules.command("list")
@json_option
@click.pass_context
def cmd_rules_list(ctx: click.Context, output_json_flag: bool) -> None:
    """Every rule on offer, with the files it would be written into."""
    config: TjConfig = ctx.obj["config"]
    rules = list_rule_writes(config)
    if resolve_output_json(ctx, output_json_flag):
        console.print_json(json.dumps({
            "rules": [r.to_dict() for r in rules], "note": RULES_NOTE,
        }))
        return
    if not rules:
        console.print(
            "No permanent rules on offer. Run [accent]tj optimize[/accent] to "
            "refresh the analyzers, then re-run this.",
        )
        return

    from rich.table import Table

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("ANALYZER")
    table.add_column("RULE")
    table.add_column("HOW")
    table.add_column("WHERE")
    table.add_column("STATE")
    table.add_column("PAST OVERSPEND", justify="right")
    for rule in rules:
        if rule.destinations:
            where = (
                rule.destinations[0].path if len(rule.destinations) == 1
                else f"{len(rule.destinations)} project files"
            )
        else:
            where = "not resolved"
        # `None` renders an em dash, never $0.00: a rule with no priced figure
        # is not a rule worth nothing.
        figure = (
            "—" if rule.past_overspend_usd is None
            else f"${rule.past_overspend_usd:,.2f}"
        )
        kind = DELIVERY_KINDS.get(rule.delivery or DEFAULT_DELIVERY)
        # An applied rule keeps its row and its figure and says so. A row that
        # simply vanished would leave a user who just applied it unable to tell
        # "done" from "broken".
        # "in your files" is a THIRD way to be done, and the one no ledger knew
        # about: the user wrote this rule themselves. Without it such a rule read
        # as "deferred", i.e. as something tokenjam declined — the inversion the
        # presence check exists to fix.
        #
        # "deferred" survives as a fallback only. The write budget's refusals no
        # longer reach this list at all (`rulewrite/plan._is_listable`), so a row
        # showing it means something new is gating the offer and should be looked
        # at rather than quietly labelled.
        state = (
            "applied" if rule.already_applied
            else "in your files" if rule.already_present
            else "dismissed" if rule.dismissed
            else ("on offer" if rule.offered else "deferred")
        )
        table.add_row(
            rule.analyzer,
            escape(rule.title or rule.signature),
            escape(kind.label if kind else (rule.delivery or "unknown")),
            escape(where),
            state,
            figure,
        )
    console.print(table)
    console.print()
    # The per-rule "not on offer: <reason>" trailer is GONE. It existed because
    # every declined rule was listed and had to explain itself; the gate now sits
    # below this surface, so there is nothing in `rules` left to explain. What
    # replaces it is a count of the rules that are already in place, since a list
    # that is short because the work is done reads identically to one that is
    # short because nothing was found.
    in_place = [r for r in rules if r.already_applied or r.already_present]
    if in_place:
        by_us = sum(1 for r in in_place if r.already_applied)
        by_you = len(in_place) - by_us
        parts = []
        if by_us:
            parts.append(f"{by_us} applied by tokenjam")
        if by_you:
            parts.append(f"{by_you} already in your instruction files")
        console.print(f"[muted]In place: {escape(', '.join(parts))}.[/muted]")
    console.print(f"[muted]{escape(RULES_NOTE)}[/muted]")
    console.print(
        "[muted]stage one with [accent]tj rules stage <signature>[/accent], "
        "then [accent]tj rules apply[/accent] (dry-run) and "
        "[accent]--go[/accent] to write.[/muted]",
    )


@cmd_rules.command("show")
@click.argument("signature")
@json_option
@click.pass_context
def cmd_rules_show(
    ctx: click.Context, signature: str, output_json_flag: bool,
) -> None:
    """One rule in full: its text, its destinations, and why they were chosen."""
    config: TjConfig = ctx.obj["config"]
    rule = find_rule(config, signature)
    if rule is None:
        raise click.ClickException(f"no rule with signature {signature!r}")
    if resolve_output_json(ctx, output_json_flag):
        console.print_json(json.dumps(rule.to_dict()))
        return
    console.print(f"[heading]{escape(rule.title or rule.signature)}[/heading]")
    kind = DELIVERY_KINDS.get(rule.delivery or DEFAULT_DELIVERY)
    console.print(
        f"[muted]{escape(rule.signature)} · "
        f"{escape(kind.label if kind else UNRESOLVED_DELIVERY_LABEL)}[/muted]",
    )
    console.print()
    console.print(escape(rule.artifact_text))
    console.print()
    for destination in rule.destinations:
        sessions = (
            f" · {destination.sessions} session(s)" if destination.sessions else ""
        )
        console.print(
            f"  [accent]{escape(destination.path)}[/accent] "
            f"[muted]({destination.scope}{sessions})[/muted]",
        )
    if rule.placement_basis:
        console.print()
        console.print(f"[muted]{escape(rule.placement_basis)}[/muted]")
    if rule.placement_coverage_note:
        console.print(f"[muted]{escape(rule.placement_coverage_note)}[/muted]")
    if rule.already_applied:
        console.print()
        console.print(f"[ok]✓[/ok] Already applied. {escape(rule.blocked_reason)}")
    elif rule.already_present:
        console.print()
        console.print(
            f"[ok]✓[/ok] Already in your instruction files. "
            f"{escape(rule.blocked_reason)}",
        )
        if rule.presence_path:
            console.print(f"[muted]found in {escape(rule.presence_path)}[/muted]")
        # The model's own evidence, printed rather than summarised: this verdict is
        # an inference, so a reader needs the file's words to check it against.
        if rule.presence_evidence:
            console.print(f"[muted]{escape(rule.presence_evidence)}[/muted]")
    elif rule.dismissed:
        console.print()
        console.print(f"[muted]Dismissed.[/muted] {escape(rule.blocked_reason)}")
        console.print(
            f"[muted]bring it back: [accent]tj rules undismiss "
            f"{escape(rule.signature)}[/accent][/muted]",
        )
    elif not rule.offered:
        console.print()
        console.print(f"[warn]Not on offer:[/warn] {escape(rule.blocked_reason)}")


@cmd_rules.command("stage")
@click.argument("signature")
@json_option
@click.pass_context
def cmd_rules_stage(
    ctx: click.Context, signature: str, output_json_flag: bool,
) -> None:
    """Render one diff per destination and stage them for review. Writes nothing."""
    config: TjConfig = ctx.obj["config"]
    rule = find_rule(config, signature)
    if rule is None:
        raise click.ClickException(f"no rule with signature {signature!r}")
    try:
        staged = stage_rule(config, rule)
    except RuleWriteRefused as exc:
        raise click.ClickException(str(exc)) from exc
    if resolve_output_json(ctx, output_json_flag):
        console.print_json(json.dumps({"staged": [e.to_dict() for e in staged]}))
        return
    for entry in staged:
        console.print(
            f"[ok]✓[/ok] staged [accent]{escape(entry.path)}[/accent] "
            f"[muted](+{format_tokens(entry.standing_tokens_per_session)} "
            f"tok/session)[/muted]",
        )
    console.print()
    console.print(
        "[muted]review with [accent]tj rules apply[/accent] (dry-run prints "
        "each diff), then [accent]--go[/accent] to write.[/muted]",
    )


@cmd_rules.command("check")
@json_option
@click.pass_context
def cmd_rules_check(ctx: click.Context, output_json_flag: bool) -> None:
    """Re-verify every staged write against the files as they stand now."""
    config: TjConfig = ctx.obj["config"]
    rows = check_staged(config)
    if resolve_output_json(ctx, output_json_flag):
        console.print_json(json.dumps({"staged": rows}))
        return
    if not rows:
        console.print("Nothing staged.")
        return
    for row in rows:
        if row["applyable"]:
            console.print(f"[ok]✓[/ok] {escape(row['path'])}")
        else:
            console.print(
                f"[warn]![/warn] {escape(row['path'])} — {escape(row['reason'])}",
            )


@cmd_rules.command("apply")
@click.argument("signature", required=False)
@click.option("--go", is_flag=True, help="Actually write. Without it this is a dry-run.")
@json_option
@click.pass_context
def cmd_rules_apply(
    ctx: click.Context, signature: str | None, go: bool, output_json_flag: bool,
) -> None:
    """Apply staged writes (all, or one rule's). Default dry-run; --go writes."""
    config: TjConfig = ctx.obj["config"]
    try:
        result = apply_staged(config, signature, go=go)
    except RuleWriteRefused as exc:
        raise click.ClickException(str(exc)) from exc
    if resolve_output_json(ctx, output_json_flag):
        console.print_json(json.dumps(result))
        return
    for row in result["applied"]:
        if result["dry_run"]:
            console.print(f"[heading]{escape(row['path'])}[/heading]")
            _print_diff(row["diff"])
            console.print()
        else:
            console.print(f"[ok]✓[/ok] wrote {escape(row['path'])}")
    for row in result["skipped"]:
        console.print(
            f"[warn]![/warn] skipped {escape(row['path'])} — {escape(row['reason'])}",
        )
    if result["dry_run"] and result["applied"]:
        console.print(
            "[muted]dry-run — nothing written. Re-run with "
            "[accent]--go[/accent] to apply.[/muted]",
        )
    if not result["applied"] and not result["skipped"]:
        console.print("Nothing staged.")


@cmd_rules.command("undo")
@click.argument("signature")
@click.option("--path", default=None, help="Undo one destination only.")
@click.option("--go", is_flag=True, help="Actually restore. Without it this is a dry-run.")
@json_option
@click.pass_context
def cmd_rules_undo(
    ctx: click.Context, signature: str, path: str | None, go: bool,
    output_json_flag: bool,
) -> None:
    """Revert an applied rule write, per destination. Default dry-run."""
    config: TjConfig = ctx.obj["config"]
    try:
        result = undo(config, signature, path, go=go)
    except RuleWriteRefused as exc:
        raise click.ClickException(str(exc)) from exc
    if resolve_output_json(ctx, output_json_flag):
        console.print_json(json.dumps(result))
        return
    for row in result["restored"]:
        verb = "would remove" if result["dry_run"] else "removed"
        if not row["removed_file"]:
            verb = "would restore" if result["dry_run"] else "restored"
        console.print(f"[ok]✓[/ok] {verb} {escape(row['path'])}")
    for row in result["skipped"]:
        console.print(
            f"[warn]![/warn] {escape(row['path'])} — {escape(row['reason'])}",
        )
    if result["dry_run"]:
        console.print(
            "[muted]dry-run — nothing changed. Re-run with "
            "[accent]--go[/accent] to restore.[/muted]",
        )


@cmd_rules.command("applied")
@json_option
@click.pass_context
def cmd_rules_applied(ctx: click.Context, output_json_flag: bool) -> None:
    """Rule writes that have been applied and can still be undone."""
    config: TjConfig = ctx.obj["config"]
    rows = store.list_backups(config)
    if resolve_output_json(ctx, output_json_flag):
        console.print_json(json.dumps({"applied": rows}))
        return
    if not rows:
        console.print("Nothing applied yet.")
        return
    for row in rows:
        state = "" if row["undoable"] else f" [muted]({escape(row['reason'])})[/muted]"
        console.print(
            f"  [accent]{escape(row['source_path'])}[/accent] "
            f"[muted]{escape(row['signature'])} · {escape(row['applied_at'])}"
            f"[/muted]{state}",
        )


@cmd_rules.command("dismiss")
@click.argument("signature")
@click.option("--reason", default="", help="Optional note, stored verbatim.")
@json_option
@click.pass_context
def cmd_rules_dismiss(
    ctx: click.Context, signature: str, reason: str, output_json_flag: bool,
) -> None:
    """Stop offering a rule. Reversible, and the figure is untouched."""
    from tokenjam.core.optimize import dismissals

    config: TjConfig = ctx.obj["config"]
    record = dismissals.dismiss(config, signature, reason=reason)
    if resolve_output_json(ctx, output_json_flag):
        console.print_json(json.dumps(record))
        return
    console.print(f"[ok]✓[/ok] dismissed [accent]{escape(signature)}[/accent]")
    console.print(
        "[muted]What the behaviour already cost is still reported. Bring it "
        f"back with [accent]tj rules undismiss {escape(signature)}[/accent]."
        "[/muted]",
    )


@cmd_rules.command("undismiss")
@click.argument("signature")
@json_option
@click.pass_context
def cmd_rules_undismiss(
    ctx: click.Context, signature: str, output_json_flag: bool,
) -> None:
    """Bring a dismissed rule back."""
    from tokenjam.core.optimize import dismissals

    config: TjConfig = ctx.obj["config"]
    record = dismissals.undismiss(config, signature)
    if resolve_output_json(ctx, output_json_flag):
        console.print_json(json.dumps(record or {}))
        return
    if record is None:
        console.print(f"[muted]{escape(signature)} was not dismissed.[/muted]")
        return
    console.print(f"[ok]✓[/ok] restored [accent]{escape(signature)}[/accent]")
