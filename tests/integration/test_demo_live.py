"""
End-to-end test for `tj demo <scenario> --live`.

Drives the real CLI command with a real `tj serve` app behind it (the demo
scenario's spans are POSTed through the actual /api/v1/spans receive path into a
real IngestPipeline), then asserts the spans + alert landed in the server's DB —
the same DB the dashboard reads. This is the regression guard for the ticket's
"a documented flag replays a demo scenario into a live tj serve" contract.
"""
from __future__ import annotations

import httpx
import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from tokenjam.api.app import create_app
from tokenjam.cli.main import cli
from tokenjam.core.config import (
    AlertsConfig,
    ApiAuthConfig,
    ApiConfig,
    SecurityConfig,
    TjConfig,
)
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest import build_default_pipeline
from tokenjam.core.models import AlertFilters


@pytest.fixture
def live_serve(monkeypatch):
    """A real serve app; route the demo live sink's httpx calls into it."""
    db = InMemoryBackend()
    # Auth disabled so the CLI's own (unrelated) config secret doesn't have to
    # match — the receive path itself is exercised in full regardless.
    # channels=[] silences the alert stdout channel — real `tj serve` runs in a
    # separate process so its channels never touch the demo command's stdout;
    # in-process here they would, so mute them (DB alerts still get written).
    config = TjConfig(
        version="1",
        security=SecurityConfig(ingest_secret=""),
        api=ApiConfig(auth=ApiAuthConfig(enabled=False)),
        alerts=AlertsConfig(channels=[]),
    )
    # Wire the pipeline exactly as `tj serve` does (build_default_pipeline) so
    # cost + alert + drift hooks fire on the receive path, not a bare pipeline.
    pipeline = build_default_pipeline(db, config)
    app = create_app(config=config, db=db, ingest_pipeline=pipeline)
    client = TestClient(app)

    def fake_post(url, json=None, headers=None, timeout=None):
        return client.post(httpx.URL(url).path, json=json, headers=headers)

    def fake_get(url, timeout=None):
        return client.get(httpx.URL(url).path)

    monkeypatch.setattr("tokenjam.demo.live.httpx.post", fake_post)
    monkeypatch.setattr("tokenjam.demo.live.httpx.get", fake_get)
    try:
        yield db
    finally:
        db.close()


def _spans_for(db, agent_id: str) -> int:
    row = db.conn.execute(
        "SELECT COUNT(*) FROM spans WHERE agent_id = $1", [agent_id]
    ).fetchone()
    return int(row[0]) if row else 0


def _alert_types(db) -> list[str]:
    return [a.type.value for a in db.get_alerts(AlertFilters(limit=1000))]


def test_demo_retry_loop_live_lands_spans_and_alert(live_serve):
    db = live_serve
    result = CliRunner().invoke(cli, ["demo", "retry-loop", "--live"])
    assert result.exit_code == 0, result.output

    # Spans reached the real server DB (the one the dashboard reads).
    assert _spans_for(db, "demo-retry-loop") >= 5
    # And the round-tripped tool input let the server re-detect the loop.
    assert "retry_loop" in _alert_types(db)
    assert "Replayed" in result.output


def test_demo_surprise_cost_live_lands_costed_spans(live_serve):
    db = live_serve
    result = CliRunner().invoke(cli, ["demo", "surprise-cost", "--live"])
    assert result.exit_code == 0, result.output

    assert _spans_for(db, "demo-surprise-cost") == 8
    # The server recomputes cost from the round-tripped token counts.
    row = db.conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0) FROM spans WHERE agent_id = $1",
        ["demo-surprise-cost"],
    ).fetchone()
    assert float(row[0]) > 0.0


def test_demo_live_json_keeps_stdout_clean(live_serve):
    import json as _json

    db = live_serve
    result = CliRunner().invoke(cli, ["demo", "retry-loop", "--live", "--json"])
    assert result.exit_code == 0, result.output
    # stdout must stay a single parseable JSON object; the replay summary
    # is routed to stderr and never corrupts it.
    data = _json.loads(result.stdout)
    assert data["scenario"] == "retry-loop"
    assert _spans_for(db, "demo-retry-loop") >= 5


def test_demo_live_without_serve_fails_fast(monkeypatch):
    # No serve reachable → the health probe returns falsey and we exit non-zero
    # BEFORE running the scenario, with an actionable message.
    monkeypatch.setattr(
        "tokenjam.demo.live.check_serve_alive", lambda config: False
    )
    result = CliRunner().invoke(cli, ["demo", "retry-loop", "--live"])
    assert result.exit_code != 0
    assert "tj serve" in result.output
