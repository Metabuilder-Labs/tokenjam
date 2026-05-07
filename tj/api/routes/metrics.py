"""GET /metrics — Prometheus text format metrics from DB aggregation."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import PlainTextResponse

from tj.api.deps import require_api_key
from tj.core.models import AlertFilters, CostFilters

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/metrics")
async def prometheus_metrics(request: Request) -> PlainTextResponse:
    """
    Generate Prometheus text format metrics by querying the DB.
    Regenerated on each request so data is accurate after restarts.
    """
    db = request.app.state.db
    lines: list[str] = []

    # -- Cost per agent --
    _add_header(lines, "ocw_cost_usd_total", "gauge", "Running cost total per agent")
    cost_rows = db.get_cost_summary(CostFilters(group_by="agent"))
    for row in cost_rows:
        agent = row.agent_id or "unknown"
        lines.append(f'ocw_cost_usd_total{{agent_id="{_escape(agent)}"}} {row.cost_usd}')

    # -- Tokens per agent and type --
    _add_header(lines, "ocw_tokens_total", "counter", "Token usage by type")
    for row in cost_rows:
        agent = row.agent_id or "unknown"
        lines.append(f'ocw_tokens_total{{agent_id="{_escape(agent)}",type="input"}} {row.input_tokens}')
        lines.append(f'ocw_tokens_total{{agent_id="{_escape(agent)}",type="output"}} {row.output_tokens}')

    # -- Tool calls per agent --
    tool_rows = db.get_tool_calls(None, None, None)
    _add_header(lines, "ocw_tool_calls_total", "counter", "Total tool calls per agent and tool")
    for row in tool_rows:
        agent = row.get("agent_id") or "unknown"
        tool = row.get("tool_name") or "unknown"
        count = row.get("call_count", 0)
        lines.append(f'ocw_tool_calls_total{{agent_id="{_escape(agent)}",tool_name="{_escape(tool)}"}} {count}')

    # -- Alerts per agent, type, severity --
    _add_header(lines, "ocw_alerts_total", "counter", "Total alerts fired")
    alerts = db.get_alerts(AlertFilters(limit=10000))
    alert_counts: dict[tuple[str, str, str], int] = {}
    for a in alerts:
        key = (a.agent_id or "unknown", a.type.value, a.severity.value)
        alert_counts[key] = alert_counts.get(key, 0) + 1
    for (agent, atype, sev), count in alert_counts.items():
        lines.append(
            f'ocw_alerts_total{{agent_id="{_escape(agent)}",'
            f'type="{_escape(atype)}",severity="{_escape(sev)}"}} {count}'
        )

    # -- Session duration (latest completed per agent) --
    _add_header(lines, "ocw_session_duration_seconds", "gauge", "Duration of last completed session")
    # Collect unique agent_ids from cost rows
    agent_ids = {row.agent_id for row in cost_rows if row.agent_id}
    for agent_id in sorted(agent_ids):
        sessions = db.get_completed_sessions(agent_id, limit=1)
        if sessions and sessions[0].duration_seconds is not None:
            lines.append(
                f'ocw_session_duration_seconds{{agent_id="{_escape(agent_id)}"}} '
                f'{sessions[0].duration_seconds:.1f}'
            )

    lines.append("")  # trailing newline
    return PlainTextResponse("\n".join(lines), media_type="text/plain; version=0.0.4")


def _add_header(lines: list[str], name: str, mtype: str, help_text: str) -> None:
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} {mtype}")


def _escape(value: str) -> str:
    """Escape label values for Prometheus text format."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
