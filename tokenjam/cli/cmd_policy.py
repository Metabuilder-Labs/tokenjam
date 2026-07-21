"""
`tj policy` — read-only view of the unified + enforcement-plane policy surface.

`tj policy list` consolidates existing alerts / drift / schema / budget /
sensitive-actions configuration under the unified "policy" framing AND the
data-driven `[[policies]]` enforcement-plane policies (#220) the proxy engine
loads. Each row points back to the TOML section it was read from.

`tj policy decisions` shows recent policy decisions (what each policy WOULD do)
from a running `tj serve` proxy.

`tj policy add | edit | apply | remove | test` remain out of scope this sprint.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import click
from rich.markup import escape

from tokenjam.cli.json_option import json_option, resolve_output_json
from tokenjam.core.config import (
    AgentConfig,
    AlertChannelConfig,
    AlertsConfig,
    BudgetConfig,
    CaptureConfig,
    DriftConfig,
    PolicyConfig,
    ProviderBudget,
    TjConfig,
)
from tokenjam.proxy.engine import UNVALIDATED_LABEL
from tokenjam.utils.formatting import console


PREVIEW_NOTE = (
    "Note: this is a read-only preview. The unified "
    "`tj policy add|edit|apply` surface lands next sprint."
)

# Every OSS enforcement-plane policy runs unvalidated — there is no certification
# engine in the open tree, so a suggestion is never implied to be validated safe.
UNVALIDATED_NOTE = (
    f"Enforcement-plane policies ([[policies]]) run '{UNVALIDATED_LABEL}' "
    "(suggest mode only — they record what they WOULD do; nothing is enforced)."
)


@dataclass(frozen=True)
class PolicyRow:
    policy: str
    setting: str
    source: str

    def to_dict(self) -> dict[str, str]:
        return {"policy": self.policy, "setting": self.setting, "source": self.source}


@click.group("policy", invoke_without_command=False)
def cmd_policy() -> None:
    """Unified view of policy-adjacent configuration (read-only preview)."""


@cmd_policy.command("list")
@json_option
@click.pass_context
def cmd_policy_list(ctx: click.Context, output_json_flag: bool) -> None:
    """List existing alerts, drift, schema, and budget configuration."""
    config: TjConfig = ctx.obj["config"]
    output_json = resolve_output_json(ctx, output_json_flag)

    rows = _collect_rows(config)
    has_engine_policies = bool(config.policies)

    if output_json:
        payload: dict[str, Any] = {
            "policies": [r.to_dict() for r in rows],
            "note": PREVIEW_NOTE,
        }
        if has_engine_policies:
            payload["unvalidated_note"] = UNVALIDATED_NOTE
        click.echo(json.dumps(payload, indent=2))
        return

    if not rows:
        console.print("[dim]No policy-adjacent configuration found.[/dim]")
        console.print()
        console.print(f"[dim]{PREVIEW_NOTE}[/dim]")
        return

    from rich.table import Table

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("POLICY")
    table.add_column("SETTING")
    table.add_column("SOURCE", style="dim")

    for row in rows:
        table.add_row(escape(row.policy), escape(row.setting), escape(row.source))

    console.print(table)
    console.print()
    if has_engine_policies:
        console.print(f"[yellow]{escape(UNVALIDATED_NOTE)}[/yellow]")
    console.print(f"[dim]{PREVIEW_NOTE}[/dim]")


def _fetch_proxy_json(config: TjConfig, path: str) -> dict | None:
    """GET a tj-internal read endpoint from the running proxy (None if down).

    The proxy keeps recent decisions in memory, so `tj policy decisions` reads
    them from the live `tj serve` proxy. Best-effort: any failure (proxy not
    running, connection refused) returns None and the caller renders a hint.
    """
    import httpx
    url = f"http://{config.proxy.host}:{config.proxy.port}{path}"
    try:
        resp = httpx.get(url, timeout=2.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:  # noqa: BLE001 — proxy may simply be down
        return None
    return None


def _read_decisions_from_db(config: TjConfig, since, limit: int):
    """Read PERSISTED decisions + savings from the DB (#221).

    Returns (decisions, savings_dict) or None when the DB can't be opened (e.g.
    `tj serve` holds the write lock) so the caller falls back to the live proxy.
    """
    from tokenjam.core.db import open_db
    from tokenjam.core.models import PolicyDecisionFilters
    from tokenjam.proxy.audit import decision_to_display_dict, reconcile_savings
    try:
        db = open_db(config.storage)
    except Exception:  # noqa: BLE001 — locked / missing → caller falls back
        return None
    try:
        recs = db.get_policy_decisions(PolicyDecisionFilters(since=since, limit=limit))
        savings = reconcile_savings(db, since=since).to_dict()
        return [decision_to_display_dict(r) for r in recs], savings
    finally:
        db.close()


@cmd_policy.command("decisions")
@json_option
@click.option("--limit", default=20, show_default=True,
              help="Max number of recent decisions to show.")
@click.option("--since", default=None,
              help="Window for the savings summary (e.g. 7d, 30d).")
@click.pass_context
def cmd_policy_decisions(ctx: click.Context, output_json_flag: bool, limit: int,
                         since: str | None) -> None:
    """Show recent persisted policy decisions + the estimated-recoverable meter."""
    config: TjConfig = ctx.obj["config"]
    output_json = resolve_output_json(ctx, output_json_flag)

    since_dt = None
    if since:
        from tokenjam.utils.time_parse import parse_since
        since_dt = parse_since(since)

    # Persisted DB read is the richer source (history + savings). If the DB is
    # locked by a running daemon, fall back to the live proxy's in-memory ring.
    db_result = _read_decisions_from_db(config, since_dt, limit)
    savings = None
    source = "db"
    if db_result is not None:
        decisions, savings = db_result
    else:
        source = "proxy"
        payload = _fetch_proxy_json(config, "/__tj/policy/decisions")
        raw = (payload or {}).get("decisions", [])
        # Normalise the live-proxy envelope shape to the display shape.
        decisions = [{
            "ts": d.get("ts", ""), "provider": d.get("provider", ""),
            "path": d.get("path", ""),
            "would_action": (d.get("policy") or {}).get("overall_action", "-"),
            "policy_name": ", ".join(
                e.get("policy_name", "") for e in (d.get("policy") or {}).get("evaluations", [])
            ) or None,
            "label": (d.get("policy") or {}).get("label", UNVALIDATED_LABEL),
        } for d in raw][-limit:]
        if payload is None:
            decisions = None  # signals "unreachable" below

    if output_json:
        click.echo(json.dumps({
            "source": source,
            "decisions": decisions or [],
            "savings": savings,
            "label": UNVALIDATED_LABEL,
        }, indent=2))
        return

    if decisions is None:
        console.print(
            "[dim]No persisted decisions and no running proxy reachable at "
            f"http://{config.proxy.host}:{config.proxy.port}. "
            "Start it with `tj proxy enable` + `tj serve`.[/dim]"
        )
        return
    if not decisions:
        console.print("[dim]No policy decisions recorded yet "
                      "(suggest mode — eligible api traffic only).[/dim]")
        console.print()
        if savings is not None:
            _print_savings_summary(savings)
        console.print(f"[yellow]{escape(UNVALIDATED_NOTE)}[/yellow]")
        return

    from rich.table import Table

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    for col in ("TIME", "PROVIDER", "PATH", "WOULD-DO", "POLICY", "LABEL"):
        table.add_column(col, style="dim" if col in ("TIME", "LABEL") else None)
    for d in decisions:
        table.add_row(
            escape(str(d.get("ts", ""))[:19]),
            escape(str(d.get("provider") or "")),
            escape(str(d.get("path") or "")),
            escape(str(d.get("would_action") or "-")),
            escape(str(d.get("policy_name") or "-")),
            escape(str(d.get("label", UNVALIDATED_LABEL))),
        )
    console.print(table)
    console.print()
    if savings is not None:
        _print_savings_summary(savings)
    console.print(f"[yellow]{escape(UNVALIDATED_NOTE)}[/yellow]")


def _print_savings_summary(savings: dict) -> None:
    """Render the savings meter — always 'estimated recoverable', NEVER 'saved'."""
    est = savings.get("estimated_recoverable_usd", 0.0)
    spend = savings.get("actual_spend_usd", 0.0)
    pct = savings.get("estimated_recoverable_pct")
    pct_str = f" ({pct:.1f}% of actual spend)" if pct is not None else ""
    console.print(
        f"[bold]Estimated recoverable:[/bold] ~${est:.4f}{pct_str} "
        f"vs actual spend ${spend:.4f} "
        f"[dim]({savings.get('decisions', 0)} decisions, label={savings.get('label')})[/dim]"
    )
    console.print(f"[dim]{escape(savings.get('disclaimer', ''))}[/dim]")


def _collect_rows(config: TjConfig) -> list[PolicyRow]:
    rows: list[PolicyRow] = []

    rows.extend(_policy_engine_rows(config.policies))
    rows.extend(_alerts_rows(config.alerts))
    rows.extend(_defaults_budget_rows(config.defaults.budget))
    rows.extend(_provider_budget_rows(config.budgets))
    rows.extend(_agents_rows(config.agents))
    rows.extend(_capture_rows(config.capture))

    return rows


def _policy_engine_rows(policies: list[PolicyConfig]) -> list[PolicyRow]:
    """Rows for the data-driven `[[policies]]` enforcement-plane policies (#220).

    Each carries the explicit `unvalidated` label so the surface never implies a
    policy has been certified safe.
    """
    rows: list[PolicyRow] = []
    for idx, p in enumerate(policies):
        parts = [f"kind={p.kind}", f"mode={p.mode}", f"label={UNVALIDATED_LABEL}"]
        if not p.enabled:
            parts.append("enabled=false")
        if p.target_provider:
            parts.append(f"provider={p.target_provider}")
        if p.target_agent:
            parts.append(f"agent={p.target_agent}")
        rows.append(PolicyRow(
            f"policies.{p.name}",
            ", ".join(parts),
            f"[[policies]][{idx}]",
        ))
    return rows


def _alerts_rows(alerts: AlertsConfig) -> list[PolicyRow]:
    rows: list[PolicyRow] = []
    parts = [
        f"cooldown_seconds={alerts.cooldown_seconds}",
        f"include_captured_content={str(alerts.include_captured_content).lower()}",
        f"channels={len(alerts.channels)}",
    ]
    rows.append(PolicyRow("alerts", ", ".join(parts), "[alerts]"))
    for idx, ch in enumerate(alerts.channels):
        rows.append(PolicyRow(
            f"alerts.channels[{idx}]",
            _channel_summary(ch),
            "[[alerts.channels]]",
        ))
    return rows


def _channel_summary(ch: AlertChannelConfig) -> str:
    parts = [f"type={ch.type}", f"min_severity={ch.min_severity}"]
    if ch.type == "file" and ch.path:
        parts.append(f"path={ch.path}")
    elif ch.type == "ntfy" and ch.topic:
        parts.append(f"topic={ch.topic}")
    elif ch.type == "webhook" and ch.url:
        parts.append(f"url={ch.url}")
    elif ch.type == "discord" and ch.webhook_url:
        parts.append("webhook_url=<set>")
    elif ch.type == "telegram" and ch.chat_id:
        parts.append(f"chat_id={ch.chat_id}")
    return ", ".join(parts)


def _defaults_budget_rows(budget: BudgetConfig) -> list[PolicyRow]:
    if budget.daily_usd is None and budget.session_usd is None:
        return []
    return [PolicyRow(
        "defaults.budget",
        _budget_summary(budget),
        "[defaults.budget]",
    )]


def _budget_summary(budget: BudgetConfig) -> str:
    parts: list[str] = []
    if budget.daily_usd is not None:
        parts.append(f"daily_usd={budget.daily_usd:g}")
    if budget.session_usd is not None:
        parts.append(f"session_usd={budget.session_usd:g}")
    return ", ".join(parts) if parts else "[unset]"


def _provider_budget_rows(budgets: dict[str, ProviderBudget]) -> list[PolicyRow]:
    rows: list[PolicyRow] = []
    for provider, pb in sorted(budgets.items()):
        rows.append(PolicyRow(
            f"budget.{provider}",
            _provider_budget_summary(pb),
            f"[budget.{provider}]",
        ))
    return rows


def _provider_budget_summary(pb: ProviderBudget) -> str:
    parts: list[str] = []
    if pb.usd is not None:
        parts.append(f"usd={pb.usd:g}")
    if pb.plan is not None:
        parts.append(f"plan={pb.plan}")
    parts.append(f"cycle_start_day={pb.cycle_start_day}")
    if pb.applies_to_services:
        parts.append(f"applies_to_services={','.join(pb.applies_to_services)}")
    return ", ".join(parts)


def _agents_rows(agents: dict[str, AgentConfig]) -> list[PolicyRow]:
    rows: list[PolicyRow] = []
    for agent_id, agent in sorted(agents.items()):
        if agent.budget.daily_usd is not None or agent.budget.session_usd is not None:
            rows.append(PolicyRow(
                f"agents.{agent_id}.budget",
                _budget_summary(agent.budget),
                f"[agents.{agent_id}.budget]",
            ))
        if _drift_overrides_default(agent.drift):
            rows.append(PolicyRow(
                f"agents.{agent_id}.drift",
                _drift_summary(agent.drift),
                f"[agents.{agent_id}.drift]",
            ))
        if agent.sensitive_actions:
            names = ", ".join(sa.name for sa in agent.sensitive_actions)
            rows.append(PolicyRow(
                f"agents.{agent_id}.sensitive_actions",
                f"block: {names}",
                f"[agents.{agent_id}]",
            ))
        if agent.output_schema:
            rows.append(PolicyRow(
                f"agents.{agent_id}.schema",
                f"output_schema={agent.output_schema}",
                f"[agents.{agent_id}]",
            ))
    return rows


def _drift_overrides_default(drift: DriftConfig) -> bool:
    default = DriftConfig()
    return (
        drift.enabled != default.enabled
        or drift.baseline_sessions != default.baseline_sessions
        or drift.token_threshold != default.token_threshold
        or drift.tool_sequence_diff != default.tool_sequence_diff
    )


def _drift_summary(drift: DriftConfig) -> str:
    return (
        f"enabled={str(drift.enabled).lower()}, "
        f"baseline_sessions={drift.baseline_sessions}, "
        f"token_threshold={drift.token_threshold:g}, "
        f"tool_sequence_diff={drift.tool_sequence_diff:g}"
    )


def _capture_rows(capture: CaptureConfig) -> list[PolicyRow]:
    # Always emit the row — capture is a policy choice even when all four
    # toggles are off (the default). Suppressing it hid the section from
    # users who'd explicitly verified their privacy settings (#71 finding 7).
    parts = [
        f"prompts={str(capture.prompts).lower()}",
        f"completions={str(capture.completions).lower()}",
        f"tool_inputs={str(capture.tool_inputs).lower()}",
        f"tool_outputs={str(capture.tool_outputs).lower()}",
    ]
    return [PolicyRow("capture", ", ".join(parts), "[capture]")]


def _exposed_for_tests() -> dict[str, Any]:
    """Expose internals to unit tests without polluting the public surface."""
    return {
        "collect_rows": _collect_rows,
        "PolicyRow": PolicyRow,
        "PREVIEW_NOTE": PREVIEW_NOTE,
    }
