"""
`tj policy` — read-only preview of the unified policy surface.

This sprint ships `tj policy list` only: a consolidated view of existing
alerts / drift / schema / budget / sensitive-actions configuration under
the unified "policy" framing. The underlying config structure is NOT
migrated — each row points back to the TOML section it was read from.

`tj policy add | edit | apply | remove | test` land next sprint.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import click

from tokenjam.core.config import (
    AgentConfig,
    AlertChannelConfig,
    AlertsConfig,
    BudgetConfig,
    CaptureConfig,
    DriftConfig,
    ProviderBudget,
    TjConfig,
)
from tokenjam.utils.formatting import console


PREVIEW_NOTE = (
    "Note: this is a read-only preview. The unified "
    "`tj policy add|edit|apply` surface lands next sprint."
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
@click.pass_context
def cmd_policy_list(ctx: click.Context) -> None:
    """List existing alerts, drift, schema, and budget configuration."""
    config: TjConfig = ctx.obj["config"]
    output_json: bool = ctx.obj.get("output_json", False)

    rows = _collect_rows(config)

    if output_json:
        payload = {
            "policies": [r.to_dict() for r in rows],
            "note": PREVIEW_NOTE,
        }
        click.echo(json.dumps(payload, indent=2))
        return

    if not rows:
        console.print("[dim]No policy-adjacent configuration found.[/dim]")
        console.print()
        console.print(f"[dim]{PREVIEW_NOTE}[/dim]")
        return

    from rich.markup import escape
    from rich.table import Table

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("POLICY")
    table.add_column("SETTING")
    table.add_column("SOURCE", style="dim")

    for row in rows:
        table.add_row(escape(row.policy), escape(row.setting), escape(row.source))

    console.print(table)
    console.print()
    console.print(f"[dim]{PREVIEW_NOTE}[/dim]")


def _collect_rows(config: TjConfig) -> list[PolicyRow]:
    rows: list[PolicyRow] = []

    rows.extend(_alerts_rows(config.alerts))
    rows.extend(_defaults_budget_rows(config.defaults.budget))
    rows.extend(_provider_budget_rows(config.budgets))
    rows.extend(_agents_rows(config.agents))
    rows.extend(_capture_rows(config.capture))

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
    if not any([capture.prompts, capture.completions, capture.tool_inputs, capture.tool_outputs]):
        return []
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
