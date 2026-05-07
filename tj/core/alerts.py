"""
Alert engine — evaluates per-span and per-session alert rules, dispatches to channels.

Called as a post-ingest hook by IngestPipeline.process().
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

import httpx

from tj.core.config import AlertChannelConfig, TjConfig, resolve_effective_budget
from tj.core.models import Alert, AlertType, Severity
from tj.otel.semconv import TjAttributes
from tj.utils.formatting import console, severity_colour
from tj.utils.ids import new_uuid
from tj.utils.time_parse import utcnow

if TYPE_CHECKING:
    from tj.core.db import StorageBackend
    from tj.core.models import NormalizedSpan, SessionRecord

logger = logging.getLogger(__name__)

SENSITIVE_DETAIL_KEYS = frozenset({
    "prompt_content", "completion_content", "tool_input", "tool_output",
})

# Sandbox event value -> AlertType mapping
_SANDBOX_EVENT_MAP: dict[str, AlertType] = {
    "network_blocked":    AlertType.NETWORK_EGRESS_BLOCKED,
    "fs_denied":          AlertType.FILESYSTEM_ACCESS_DENIED,
    "syscall_denied":     AlertType.SYSCALL_DENIED,
    "inference_rerouted": AlertType.INFERENCE_REROUTED,
}

# Default thresholds
_RETRY_LOOP_WINDOW = 6
_RETRY_LOOP_THRESHOLD = 4
_FAILURE_RATE_WINDOW = 20
_FAILURE_RATE_THRESHOLD = 0.20
_FAILURE_RATE_CHECK_INTERVAL = 5
_SESSION_DURATION_DEFAULT = 3600  # seconds


# ── Cooldown tracker ───────────────────────────────────────────────────────

class CooldownTracker:
    """
    Prevents alert storms by suppressing repeat alerts of the same type
    for the same agent within the cooldown window.
    Stored in-memory — resets when the process restarts.
    """

    def __init__(self, cooldown_seconds: int = 60) -> None:
        self.cooldown_seconds = cooldown_seconds
        self._last_fired: dict[tuple[str, str], datetime] = {}

    def is_suppressed(self, agent_id: str | None, alert_type: AlertType) -> bool:
        key = (agent_id or "", alert_type.value)
        last = self._last_fired.get(key)
        if last is None:
            return False
        return (utcnow() - last).total_seconds() < self.cooldown_seconds

    def record(self, agent_id: str | None, alert_type: AlertType) -> None:
        key = (agent_id or "", alert_type.value)
        self._last_fired[key] = utcnow()


# ── Alert engine ───────────────────────────────────────────────────────────

class AlertEngine:
    """
    Post-ingest hook. Evaluates all alert rules after each span is written.
    Called by IngestPipeline.process() after the span is in the DB.
    """

    def __init__(self, db: StorageBackend, config: TjConfig) -> None:
        self.db = db
        self.config = config
        self.cooldown = CooldownTracker(config.alerts.cooldown_seconds)
        self.dispatcher = AlertDispatcher(config)
        # Tracks the error_count value at which the failure-rate check last fired per session.
        # Prevents non-deterministic re-firing when the sliding-window count oscillates.
        self._last_failure_rate_check: dict[str, int] = {}

    def evaluate(self, span: NormalizedSpan) -> None:
        """Evaluate all per-span alert rules against this span."""
        self._check_sensitive_action(span)
        self._check_retry_loop(span)
        self._check_failure_rate(span)
        self._check_sandbox_events(span)

    def evaluate_session_end(self, session: SessionRecord) -> None:
        """
        Evaluate per-session alert rules when a session ends.
        DRIFT_DETECTED and TOKEN_ANOMALY are fired from drift.py, not here.
        """
        self._check_cost_budgets(session)
        self._check_session_duration(session)

    def fire(
        self,
        alert_type: AlertType,
        span_or_session: NormalizedSpan | SessionRecord,
        detail: dict[str, Any],
        severity: Severity | None = None,
    ) -> None:
        """
        External entry point for other modules (SchemaValidator, DriftDetector)
        to fire alerts they detect.
        """
        from tj.core.models import NormalizedSpan

        if severity is None:
            severity = Severity.WARNING

        if isinstance(span_or_session, NormalizedSpan):
            agent_id = span_or_session.agent_id
            session_id = span_or_session.session_id
            span_id = span_or_session.span_id
        else:
            agent_id = span_or_session.agent_id
            session_id = span_or_session.session_id
            span_id = None

        alert = Alert(
            alert_id=new_uuid(),
            fired_at=utcnow(),
            type=alert_type,
            severity=severity,
            title=f"{alert_type.value} — {agent_id or 'unknown'}",
            detail=detail,
            agent_id=agent_id,
            session_id=session_id,
            span_id=span_id,
        )
        self._fire(alert)

    # ── Per-span checks ────────────────────────────────────────────────────

    def _check_sensitive_action(self, span: NormalizedSpan) -> None:
        """Fire SENSITIVE_ACTION if span.tool_name matches the agent's sensitive_actions."""
        if not span.tool_name or not span.agent_id:
            return
        agent_cfg = self.config.agents.get(span.agent_id)
        if not agent_cfg:
            return
        for sa in agent_cfg.sensitive_actions:
            if sa.name == span.tool_name:
                sev = Severity(sa.severity) if sa.severity in ("critical", "warning", "info") else Severity.WARNING
                alert = Alert(
                    alert_id=new_uuid(),
                    fired_at=utcnow(),
                    type=AlertType.SENSITIVE_ACTION,
                    severity=sev,
                    title=f"sensitive_action — {span.agent_id}",
                    detail={
                        "tool_name": span.tool_name,
                        "message": f"{span.tool_name} called",
                    },
                    agent_id=span.agent_id,
                    session_id=span.session_id,
                    span_id=span.span_id,
                )
                self._fire(alert)
                return

    def _check_retry_loop(self, span: NormalizedSpan) -> None:
        """
        Fetch last 6 spans for this session. If same tool_name appears 4+ times,
        fire RETRY_LOOP.
        """
        if not span.session_id or not span.tool_name:
            return
        recent = self.db.get_recent_spans(span.session_id, _RETRY_LOOP_WINDOW)
        tool_counts: dict[str, int] = {}
        for s in recent:
            if s.tool_name:
                tool_counts[s.tool_name] = tool_counts.get(s.tool_name, 0) + 1
        count = tool_counts.get(span.tool_name, 0)
        if count >= _RETRY_LOOP_THRESHOLD:
            alert = Alert(
                alert_id=new_uuid(),
                fired_at=utcnow(),
                type=AlertType.RETRY_LOOP,
                severity=Severity.WARNING,
                title=f"retry_loop — {span.agent_id or 'unknown'}",
                detail={
                    "tool_name": span.tool_name,
                    "count": count,
                    "window": _RETRY_LOOP_WINDOW,
                    "message": f"{span.tool_name} called {count} times in last {_RETRY_LOOP_WINDOW} spans",
                },
                agent_id=span.agent_id,
                session_id=span.session_id,
                span_id=span.span_id,
            )
            self._fire(alert)

    def _check_failure_rate(self, span: NormalizedSpan) -> None:
        """
        In a rolling window of last 20 spans, fire FAILURE_RATE if error rate > 20%.
        Only check when error_count reaches a new multiple of the check interval to
        avoid firing on every single error and to avoid re-firing when the sliding
        window count oscillates.
        """
        if not span.session_id or span.status_code.value != "error":
            return
        recent = self.db.get_recent_spans(span.session_id, _FAILURE_RATE_WINDOW)
        total = len(recent)
        if total < _FAILURE_RATE_CHECK_INTERVAL:
            return
        error_count = sum(1 for s in recent if s.status_code.value == "error")
        session_key = span.session_id
        last_checked = self._last_failure_rate_check.get(session_key, 0)
        if error_count < _FAILURE_RATE_CHECK_INTERVAL or error_count <= last_checked:
            return
        self._last_failure_rate_check[session_key] = error_count
        rate = error_count / total
        if rate > _FAILURE_RATE_THRESHOLD:
            alert = Alert(
                alert_id=new_uuid(),
                fired_at=utcnow(),
                type=AlertType.FAILURE_RATE,
                severity=Severity.WARNING,
                title=f"failure_rate — {span.agent_id or 'unknown'}",
                detail={
                    "error_count": error_count,
                    "total": total,
                    "rate": round(rate, 3),
                    "message": f"Failure rate {rate:.0%} exceeds {_FAILURE_RATE_THRESHOLD:.0%} threshold",
                },
                agent_id=span.agent_id,
                session_id=span.session_id,
                span_id=span.span_id,
            )
            self._fire(alert)

    def _check_sandbox_events(self, span: NormalizedSpan) -> None:
        """Check for NemoClaw/OpenShell sandbox event attributes."""
        event = span.attributes.get(TjAttributes.SANDBOX_EVENT)
        if not event:
            return
        alert_type = _SANDBOX_EVENT_MAP.get(event)
        if not alert_type:
            return
        detail: dict[str, Any] = {"sandbox_event": event}
        if event == "network_blocked":
            detail["host"] = span.attributes.get(TjAttributes.EGRESS_HOST, "unknown")
            detail["port"] = span.attributes.get(TjAttributes.EGRESS_PORT)
            detail["message"] = f"Network egress blocked to {detail['host']}"
        elif event == "fs_denied":
            detail["path"] = span.attributes.get(TjAttributes.FILESYSTEM_PATH, "unknown")
            detail["message"] = f"Filesystem access denied: {detail['path']}"
        elif event == "syscall_denied":
            detail["syscall"] = span.attributes.get(TjAttributes.SYSCALL_NAME, "unknown")
            detail["message"] = f"Syscall denied: {detail['syscall']}"
        elif event == "inference_rerouted":
            detail["message"] = "Inference endpoint changed from expected"

        alert = Alert(
            alert_id=new_uuid(),
            fired_at=utcnow(),
            type=alert_type,
            severity=Severity.CRITICAL,
            title=f"{alert_type.value} — {span.agent_id or 'unknown'}",
            detail=detail,
            agent_id=span.agent_id,
            session_id=span.session_id,
            span_id=span.span_id,
        )
        self._fire(alert)

    # ── Per-session checks ─────────────────────────────────────────────────

    def _check_cost_budgets(self, session: SessionRecord) -> None:
        """Check daily and session cost thresholds against the agent's budget config."""
        from tj.core.config import BudgetConfig
        budget = resolve_effective_budget(session.agent_id, self.config)
        if budget == BudgetConfig():
            return

        # Session budget
        if budget.session_usd is not None and session.total_cost_usd is not None:
            if session.total_cost_usd > budget.session_usd:
                alert = Alert(
                    alert_id=new_uuid(),
                    fired_at=utcnow(),
                    type=AlertType.COST_BUDGET_SESSION,
                    severity=Severity.CRITICAL,
                    title=f"cost_budget_session — {session.agent_id}",
                    detail={
                        "session_cost": session.total_cost_usd,
                        "budget": budget.session_usd,
                        "message": f"Session cost ${session.total_cost_usd:.4f} exceeds budget ${budget.session_usd:.4f}",
                    },
                    agent_id=session.agent_id,
                    session_id=session.session_id,
                )
                self._fire(alert)

        # Daily budget
        if budget.daily_usd is not None:
            today = utcnow().date()
            daily_cost = self.db.get_daily_cost(session.agent_id, today)
            if daily_cost > budget.daily_usd:
                alert = Alert(
                    alert_id=new_uuid(),
                    fired_at=utcnow(),
                    type=AlertType.COST_BUDGET_DAILY,
                    severity=Severity.CRITICAL,
                    title=f"cost_budget_daily — {session.agent_id}",
                    detail={
                        "daily_cost": daily_cost,
                        "budget": budget.daily_usd,
                        "message": f"Daily cost ${daily_cost:.4f} exceeds budget ${budget.daily_usd:.4f}",
                    },
                    agent_id=session.agent_id,
                    session_id=session.session_id,
                )
                self._fire(alert)

    def _check_session_duration(self, session: SessionRecord) -> None:
        """Fire SESSION_DURATION if session wall time exceeds threshold."""
        duration = session.duration_seconds
        if duration is None:
            return
        if duration > _SESSION_DURATION_DEFAULT:
            alert = Alert(
                alert_id=new_uuid(),
                fired_at=utcnow(),
                type=AlertType.SESSION_DURATION,
                severity=Severity.WARNING,
                title=f"session_duration — {session.agent_id}",
                detail={
                    "duration_seconds": duration,
                    "threshold_seconds": _SESSION_DURATION_DEFAULT,
                    "message": f"Session lasted {duration:.0f}s, exceeding {_SESSION_DURATION_DEFAULT}s threshold",
                },
                agent_id=session.agent_id,
                session_id=session.session_id,
            )
            self._fire(alert)

    # ── Internal dispatch ──────────────────────────────────────────────────

    def _fire(self, alert: Alert) -> None:
        """Persist alert to DB and dispatch. Suppressed alerts are persisted but not dispatched."""
        if self.cooldown.is_suppressed(alert.agent_id, alert.type):
            alert.suppressed = True
            self.db.insert_alert(alert)
            return
        self.db.insert_alert(alert)
        self.cooldown.record(alert.agent_id, alert.type)
        self.dispatcher.dispatch(alert)


# ── Alert dispatcher ───────────────────────────────────────────────────────

class AlertDispatcher:
    """Routes a fired alert to all configured output channels."""

    def __init__(self, config: TjConfig) -> None:
        self.channels: list[AlertChannel] = [
            _build_channel(ch_config, config.alerts.include_captured_content)
            for ch_config in config.alerts.channels
        ]

    def dispatch(self, alert: Alert) -> None:
        for channel in self.channels:
            # Enforce min_severity gate centrally so every channel type honours it.
            min_sev = getattr(channel, "min_severity", None)
            if min_sev is not None and _severity_rank(alert.severity) < _severity_rank(min_sev):
                continue
            try:
                channel.send(alert)
            except Exception as exc:
                logger.warning("Alert channel %s failed: %s", channel, exc)


def _build_channel(
    config: AlertChannelConfig, include_captured_content: bool
) -> AlertChannel:
    """Factory: return the correct channel instance for the config type."""
    match config.type:
        case "stdout":
            return StdoutChannel(min_severity=Severity(config.min_severity))
        case "file":
            return FileChannel(
                config.path or "alerts.jsonl",
                include_captured_content,
                min_severity=Severity(config.min_severity),
            )
        case "ntfy":
            return NtfyChannel(config, include_captured_content)
        case "webhook":
            return WebhookChannel(config, include_captured_content)
        case "discord":
            return DiscordChannel(config, include_captured_content)
        case "telegram":
            return TelegramChannel(config, include_captured_content)
        case _:
            raise ValueError(f"Unknown alert channel type: {config.type!r}")


def _strip_sensitive(detail: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of detail with captured content keys removed."""
    return {k: v for k, v in detail.items() if k not in SENSITIVE_DETAIL_KEYS}


def _format_detail_text(alert: Alert, strip: bool) -> str:
    """Format alert detail as a human-readable text block."""
    detail = _strip_sensitive(alert.detail) if strip else alert.detail
    return detail.get("message", json.dumps(detail, default=str))


def _alert_to_dict(alert: Alert, strip: bool) -> dict[str, Any]:
    """Serialise an alert to a dict, optionally stripping sensitive content."""
    detail = _strip_sensitive(alert.detail) if strip else alert.detail
    return {
        "alert_id": alert.alert_id,
        "fired_at": alert.fired_at.isoformat(),
        "type": alert.type.value,
        "severity": alert.severity.value,
        "title": alert.title,
        "detail": detail,
        "agent_id": alert.agent_id,
        "session_id": alert.session_id,
        "span_id": alert.span_id,
    }


# ── Channel base ──────────────────────────────────────────────────────────

class AlertChannel:
    """Base class for alert output channels."""

    def send(self, alert: Alert) -> None:
        raise NotImplementedError


# ── Channel implementations ───────────────────────────────────────────────

class StdoutChannel(AlertChannel):
    """
    Prints to stdout using Rich.
    Format: [HH:MM:SS]  icon SEVERITY  type  agent  message
    Always includes full detail (no content stripping).
    """

    def __init__(self, min_severity: Severity = Severity.INFO) -> None:
        self.min_severity = min_severity

    def send(self, alert: Alert) -> None:
        time_str = alert.fired_at.strftime("%H:%M:%S")
        sev = alert.severity.value.upper()
        colour = severity_colour(alert.severity.value)
        icon = "\u26a0" if alert.severity in (Severity.CRITICAL, Severity.WARNING) else "\u2139"
        message = alert.detail.get("message", alert.title)
        agent = alert.agent_id or "unknown"
        console.print(
            f"[dim]{time_str}[/dim]  {icon} [{colour} bold]{sev}[/]  "
            f"[cyan]{alert.type.value}[/]  [dim]{agent}[/]  {message}"
        )


class FileChannel(AlertChannel):
    """
    Appends a JSON line to the configured log file path.
    Always includes full detail (no content stripping).
    """

    def __init__(self, path: str, include_captured_content: bool,
                 min_severity: Severity = Severity.INFO) -> None:
        self.path = path
        # File channels always get full payload regardless of config
        self._include_captured_content = True
        self.min_severity = min_severity

    def send(self, alert: Alert) -> None:
        from pathlib import Path

        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        record = _alert_to_dict(alert, strip=False)
        with open(p, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")


class NtfyChannel(AlertChannel):
    """Sends push notifications via ntfy.sh or self-hosted ntfy."""

    def __init__(self, config: AlertChannelConfig, include_captured_content: bool) -> None:
        self.server = config.server
        self.topic = config.topic or ""
        self.token = config.token
        self.min_severity = Severity(config.min_severity)
        self._include_captured_content = include_captured_content

    def send(self, alert: Alert) -> None:
        if not self.topic:
            return
        if _severity_rank(alert.severity) < _severity_rank(self.min_severity):
            return
        url = f"{self.server.rstrip('/')}/{self.topic}"
        headers: dict[str, str] = {"Title": alert.title}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        body = _format_detail_text(alert, strip=not self._include_captured_content)
        with httpx.Client(timeout=5.0) as client:
            client.post(url, content=body, headers=headers)


class WebhookChannel(AlertChannel):
    """HTTP POST to configured URL with JSON payload."""

    def __init__(self, config: AlertChannelConfig, include_captured_content: bool) -> None:
        self.url = config.url or ""
        self.method = config.method
        self.headers = config.headers
        self._include_captured_content = include_captured_content
        self.min_severity = Severity(config.min_severity)

    def send(self, alert: Alert) -> None:
        if not self.url:
            return
        payload = _alert_to_dict(alert, strip=not self._include_captured_content)
        with httpx.Client(timeout=5.0) as client:
            client.request(
                self.method,
                self.url,
                json=payload,
                headers=self.headers,
            )


class DiscordChannel(AlertChannel):
    """POST to Discord webhook URL with an embed coloured by severity."""

    def __init__(self, config: AlertChannelConfig, include_captured_content: bool) -> None:
        self.webhook_url = config.webhook_url or ""
        self._include_captured_content = include_captured_content
        self.min_severity = Severity(config.min_severity)

    def send(self, alert: Alert) -> None:
        if not self.webhook_url:
            return
        colour_map = {
            Severity.CRITICAL: 0xFF0000,
            Severity.WARNING:  0xFFAA00,
            Severity.INFO:     0x3498DB,
        }
        description = _format_detail_text(alert, strip=not self._include_captured_content)
        payload = {
            "embeds": [{
                "title": alert.title,
                "description": description,
                "color": colour_map.get(alert.severity, 0x3498DB),
                "fields": [
                    {"name": "Type", "value": alert.type.value, "inline": True},
                    {"name": "Severity", "value": alert.severity.value, "inline": True},
                    {"name": "Agent", "value": alert.agent_id or "unknown", "inline": True},
                ],
                "timestamp": alert.fired_at.isoformat(),
            }],
        }
        with httpx.Client(timeout=5.0) as client:
            client.post(self.webhook_url, json=payload)


class TelegramChannel(AlertChannel):
    """POST to Telegram Bot API sendMessage with Markdown formatting."""

    def __init__(self, config: AlertChannelConfig, include_captured_content: bool) -> None:
        self.bot_token = config.bot_token or ""
        self.chat_id = config.chat_id or ""
        self._include_captured_content = include_captured_content
        self.min_severity = Severity(config.min_severity)

    def send(self, alert: Alert) -> None:
        if not self.bot_token or not self.chat_id:
            return
        message = _format_detail_text(alert, strip=not self._include_captured_content)
        text = (
            f"*{_escape_markdown(alert.title)}*\n"
            f"Severity: {alert.severity.value}\n"
            f"Type: {alert.type.value}\n"
            f"Agent: {alert.agent_id or 'unknown'}\n\n"
            f"{_escape_markdown(message)}"
        )
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }
        with httpx.Client(timeout=5.0) as client:
            client.post(url, json=payload)


# ── Helpers ────────────────────────────────────────────────────────────────

_SEVERITY_RANK = {Severity.INFO: 0, Severity.WARNING: 1, Severity.CRITICAL: 2}


def _severity_rank(sev: Severity) -> int:
    return _SEVERITY_RANK.get(sev, 0)


def _escape_markdown(text: str) -> str:
    """Escape Telegram Markdown v1 special characters."""
    for ch in ("_", "*", "[", "`"):
        text = text.replace(ch, f"\\{ch}")
    return text
