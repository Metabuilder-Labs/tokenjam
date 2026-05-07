"""
Shared demo environment: wires IngestPipeline + InMemoryBackend directly,
bypassing the SDK/OTel TracerProvider so scenarios have zero setup friction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tj.core.config import AgentConfig
    from tj.core.models import NormalizedSpan


@dataclass
class DemoResult:
    agent_id: str
    span_count: int
    alert_count: int
    alert_types: list[str]
    total_cost_usd: float
    trace_count: int


class DemoEnvironment:
    """
    Self-contained observability stack for demo scenarios.
    No API keys, no OTel global state, no config files needed.
    Alerts are silenced (no stdout/file channels) so scenarios run cleanly.
    """

    def __init__(
        self,
        agent_configs: "dict[str, AgentConfig] | None" = None,
    ) -> None:
        from tj.core.alerts import AlertEngine
        from tj.core.config import AlertsConfig, TjConfig, SecurityConfig
        from tj.core.cost import CostEngine
        from tj.core.db import InMemoryBackend
        from tj.core.drift import DriftDetector
        from tj.core.ingest import IngestPipeline

        self.db = InMemoryBackend()

        self.config = TjConfig(
            version="1",
            security=SecurityConfig(ingest_secret="demo"),
            alerts=AlertsConfig(channels=[], cooldown_seconds=0),
            agents=agent_configs or {},
        )

        cost_engine = CostEngine(db=self.db)
        alert_engine = AlertEngine(db=self.db, config=self.config)
        drift_detector = DriftDetector(
            db=self.db, alert_engine=alert_engine, config=self.config
        )

        self.pipeline = IngestPipeline(
            db=self.db,
            config=self.config,
            cost_engine=cost_engine,
            alert_engine=alert_engine,
            drift_detector=drift_detector,
        )

    def process(self, span: "NormalizedSpan") -> None:
        self.pipeline.process(span)

    def get_alerts(self) -> list:
        from tj.core.models import AlertFilters
        return self.db.get_alerts(AlertFilters(limit=1000))

    def total_cost_usd(self) -> float:
        row = self.db.conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0.0) FROM spans"
        ).fetchone()
        return float(row[0]) if row else 0.0

    def span_count(self) -> int:
        row = self.db.conn.execute("SELECT COUNT(*) FROM spans").fetchone()
        return int(row[0]) if row else 0

    def trace_count(self) -> int:
        from tj.core.models import TraceFilters
        return len(self.db.get_traces(TraceFilters()))

    def build_result(self, agent_id: str) -> DemoResult:
        alerts = self.get_alerts()
        return DemoResult(
            agent_id=agent_id,
            span_count=self.span_count(),
            alert_count=len(alerts),
            alert_types=[a.type.value for a in alerts],
            total_cost_usd=self.total_cost_usd(),
            trace_count=self.trace_count(),
        )
