"""``tj loop`` — close the loop on a run (#53).

Annotate a run with a human verdict/note, promote a bad run into a stored
expectation, and record whether later runs pass or regress against it — the
loop-closing half that pairs with tokenjam's capture half (``tj trace``).

Dual-path like the read commands: when ``tj serve`` holds the DB lock the CLI
runs in ``api_mode`` (``ctx.obj["db"]`` is an ``ApiBackend``), so these commands
route writes through the running server's HTTP API; otherwise they call
``core.loop`` directly against the local DuckDB. Both paths hit the exact same
storage, so ``tj loop`` and the Lens "Loop" tab stay consistent.
"""
from __future__ import annotations

import json

import click

from tokenjam.core import loop
from tokenjam.utils.formatting import console, make_table


def _api_mode(ctx: click.Context) -> bool:
    return bool(ctx.obj.get("api_mode"))


def _client(ctx: click.Context):
    """The ApiBackend's httpx client (api_mode only)."""
    return ctx.obj["db"].client


def _emit(ctx: click.Context, payload: dict) -> None:
    """Echo a payload as JSON when ``--json`` is set (shared by subcommands)."""
    if ctx.obj.get("output_json"):
        click.echo(json.dumps(payload, default=str))


def _verdict_style(verdict: str | None) -> str:
    return {
        "good": "green", "bad": "red", "mixed": "yellow", "unknown": "dim",
    }.get(verdict or "", "dim")


def _outcome_style(outcome: str) -> str:
    return {"pass": "green", "regress": "red", "unknown": "dim"}.get(outcome, "dim")


@click.group("loop")
def cmd_loop() -> None:
    """Close the loop: annotate runs, promote expectations, track fix-history."""


@cmd_loop.command("annotate")
@click.argument("session_id")
@click.option("--note", required=True, help="Human note on this run.")
@click.option(
    "--verdict",
    type=click.Choice(sorted(loop.VALID_VERDICTS)),
    default=None,
    help="Optional verdict for the run.",
)
@click.pass_context
def annotate(ctx, session_id, note, verdict):
    """Leave a note (+ optional verdict) on a run."""
    if _api_mode(ctx):
        body = {"note": note}
        if verdict:
            body["verdict"] = verdict
        resp = _client(ctx).post(
            f"/api/v1/sessions/{session_id}/annotations", json=body
        )
        if resp.status_code >= 400:
            raise click.ClickException(_err(resp))
        ann = resp.json()
    else:
        try:
            ann = loop.add_annotation(
                ctx.obj["db"], session_id, note=note, verdict=verdict
            ).to_dict()
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
    _emit(ctx, ann)
    if not ctx.obj.get("output_json"):
        v = ann.get("verdict")
        tag = f"[{_verdict_style(v)}]{v}[/] " if v else ""
        console.print(f"{tag}annotation saved on [bold]{session_id}[/].")


@cmd_loop.command("annotations")
@click.argument("session_id")
@click.pass_context
def list_annotations_cmd(ctx, session_id):
    """List annotations on a run, newest first."""
    if _api_mode(ctx):
        resp = _client(ctx).get(f"/api/v1/sessions/{session_id}/annotations")
        if resp.status_code >= 400:
            raise click.ClickException(_err(resp))
        rows = resp.json().get("annotations", [])
    else:
        rows = [a.to_dict() for a in loop.list_annotations(ctx.obj["db"], session_id)]
    _emit(ctx, {"annotations": rows, "count": len(rows)})
    if ctx.obj.get("output_json"):
        return
    if not rows:
        console.print("[dim]No annotations on this run yet.[/dim]")
        return
    table = make_table("WHEN", "VERDICT", "NOTE")
    for a in rows:
        v = a.get("verdict") or "-"
        table.add_row(
            _short_ts(a.get("created_at")),
            f"[{_verdict_style(a.get('verdict'))}]{v}[/]",
            a.get("note") or "",
        )
    console.print(table)


@cmd_loop.command("expect")
@click.argument("session_id", required=False)
@click.option("--name", required=True, help="Short name for the expectation.")
@click.option("--desc", "description", default=None, help="What's expected.")
@click.option("--agent", default=None, help="Agent id this expectation scopes to.")
@click.pass_context
def expect(ctx, session_id, name, description, agent):
    """Promote a run into a stored expectation (SESSION_ID optional)."""
    body = {"name": name}
    if description:
        body["description"] = description
    if session_id:
        body["origin_session_id"] = session_id
    if agent:
        body["agent_id"] = agent
    if _api_mode(ctx):
        resp = _client(ctx).post("/api/v1/expectations", json=body)
        if resp.status_code >= 400:
            raise click.ClickException(_err(resp))
        exp = resp.json()
    else:
        try:
            exp = loop.create_expectation(
                ctx.obj["db"], name=name, description=description,
                origin_session_id=session_id, agent_id=agent,
            ).to_dict()
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
    _emit(ctx, exp)
    if not ctx.obj.get("output_json"):
        console.print(
            f"Expectation [bold]{exp['name']}[/] created "
            f"([dim]{exp['expectation_id']}[/]).\n"
            f"Record a rerun with: [bold]tj loop record {exp['expectation_id']} "
            "<session_id> --outcome pass|regress[/]"
        )


@cmd_loop.command("expectations")
@click.pass_context
def expectations(ctx):
    """List all expectations, newest first."""
    if _api_mode(ctx):
        resp = _client(ctx).get("/api/v1/expectations")
        if resp.status_code >= 400:
            raise click.ClickException(_err(resp))
        rows = resp.json().get("expectations", [])
    else:
        rows = [e.to_dict() for e in loop.list_expectations(ctx.obj["db"])]
    _emit(ctx, {"expectations": rows, "count": len(rows)})
    if ctx.obj.get("output_json"):
        return
    if not rows:
        console.print("[dim]No expectations yet. Promote a run with `tj loop expect`.[/dim]")
        return
    table = make_table("ID", "NAME", "ORIGIN", "CREATED")
    for e in rows:
        origin = (e.get("origin_session_id") or "-")
        table.add_row(
            e["expectation_id"][:12],
            e.get("name") or "",
            origin[:12] if origin != "-" else "-",
            _short_ts(e.get("created_at")),
        )
    console.print(table)


@cmd_loop.command("record")
@click.argument("expectation_id")
@click.argument("session_id", required=False)
@click.option(
    "--outcome",
    required=True,
    type=click.Choice(sorted(loop.VALID_OUTCOMES)),
    help="Did this run pass or regress against the expectation?",
)
@click.option("--note", default=None, help="Optional note (what changed).")
@click.pass_context
def record(ctx, expectation_id, session_id, outcome, note):
    """Record a rerun's outcome against an expectation (fix-history ledger)."""
    body = {"outcome": outcome}
    if session_id:
        body["session_id"] = session_id
    if note:
        body["note"] = note
    if _api_mode(ctx):
        resp = _client(ctx).post(
            f"/api/v1/expectations/{expectation_id}/runs", json=body
        )
        if resp.status_code >= 400:
            raise click.ClickException(_err(resp))
        entry = resp.json()
    else:
        try:
            entry = loop.record_expectation_run(
                ctx.obj["db"], expectation_id,
                outcome=outcome, session_id=session_id, note=note,
            ).to_dict()
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
    _emit(ctx, entry)
    if not ctx.obj.get("output_json"):
        console.print(
            f"Recorded [{_outcome_style(outcome)}]{outcome}[/] against "
            f"expectation [dim]{expectation_id[:12]}[/]."
        )


@cmd_loop.command("history")
@click.argument("expectation_id")
@click.pass_context
def history(ctx, expectation_id):
    """Show an expectation and its fix-history (pass/regress over time)."""
    if _api_mode(ctx):
        resp = _client(ctx).get(f"/api/v1/expectations/{expectation_id}")
        if resp.status_code == 404:
            raise click.ClickException(f"Expectation {expectation_id} not found.")
        if resp.status_code >= 400:
            raise click.ClickException(_err(resp))
        payload = resp.json()
        exp = payload.get("expectation")
        runs = payload.get("runs", [])
    else:
        exp_obj = loop.get_expectation(ctx.obj["db"], expectation_id)
        if exp_obj is None:
            raise click.ClickException(f"Expectation {expectation_id} not found.")
        exp = exp_obj.to_dict()
        runs = [r.to_dict() for r in loop.list_expectation_runs(ctx.obj["db"], expectation_id)]
    _emit(ctx, {"expectation": exp, "runs": runs, "run_count": len(runs)})
    if ctx.obj.get("output_json"):
        return
    console.print(f"[bold]{exp['name']}[/]  [dim]{exp['expectation_id']}[/]")
    if exp.get("description"):
        console.print(f"  {exp['description']}")
    if exp.get("origin_session_id"):
        console.print(f"  [dim]promoted from run {exp['origin_session_id']}[/]")
    if not runs:
        console.print("[dim]No reruns recorded yet.[/dim]")
        return
    table = make_table("WHEN", "OUTCOME", "RUN", "NOTE")
    for r in runs:
        sid = r.get("session_id") or "-"
        table.add_row(
            _short_ts(r.get("created_at")),
            f"[{_outcome_style(r.get('outcome'))}]{r.get('outcome')}[/]",
            sid[:12] if sid != "-" else "-",
            r.get("note") or "",
        )
    console.print(table)


def _err(resp) -> str:
    """Best-effort error message from a non-2xx API response."""
    try:
        return resp.json().get("error") or f"API {resp.status_code}"
    except Exception:
        return f"API {resp.status_code}"


def _short_ts(ts: str | None) -> str:
    """Trim an ISO timestamp to minute precision for table display."""
    if not ts:
        return "-"
    return ts.replace("T", " ")[:16]
