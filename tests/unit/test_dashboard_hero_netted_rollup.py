"""The Dashboard hero and its total-opportunity tile read the netted,
cross-analyzer rollup `GET /relearn/cost-proposals` already publishes
(`past_overspend`, `cost_proposals.past_overspend_rollup`) — never a plain
client-side sum of each analyzer's own `past_overspend_usd`.

`reuse` and `script` both cluster on the identical repeated-tool-sequence
shape (a planning call plus a deterministic tool sequence) and can genuinely
claim the SAME sessions once both are enabled for one persona — see
`_net_cross_analyzer_session_overlap` (`core/optimize/cost_proposals.py`),
which reduces the lower-priority claim by the proportion of its own sessions
a higher-priority analyzer already claimed. On a `claude-code`-only corpus
neither the JS naive sum nor this test could ever see the difference,
because `PERSONA_DISABLED_ANALYZERS["claude-code"]` (`core/optimize/
runner.py`) disables both `reuse` and `script` for that persona. The Lens
redesign ships a persona toggle that reaches `sdk`, where BOTH stay enabled
(`disabled_analyzers_for_persona("sdk")` carries neither), which is exactly
the condition under which the old naive sum double-counted.

This seeds an sdk-dominant window with the same overlapping-cluster shape
`test_cross_analyzer_recoverable_invariant.py` uses (20 sessions sharing one
planning-call + tool-sequence skeleton) and proves two things: the RAW,
un-netted per-analyzer figures a naive client-side sum would have added
together, and the netted total the wire actually serves via
`GET /relearn/cost-proposals`, are not the same number — the wire figure is
strictly smaller.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from tokenjam.api.app import create_app
from tokenjam.core.config import (
    ApiAuthConfig,
    ApiConfig,
    CaptureConfig,
    OptimizeConfig,
    StorageConfig,
    TjConfig,
)
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest import build_default_pipeline
from tokenjam.core.optimize import cost_proposals as cost_proposals_mod
from tokenjam.core.optimize import runner
from tokenjam.otel.semconv import GenAIAttributes
from tests.factories import make_llm_span, make_session, make_tool_span

UTC = timezone.utc
BASE = datetime(2026, 5, 10, tzinfo=UTC)
SINCE = datetime(2026, 5, 1, tzinfo=UTC)
UNTIL = datetime(2026, 5, 30, tzinfo=UTC)
MODEL = "claude-haiku-4-5"


def _seed_overlapping_cluster(db, *, count: int = 20) -> None:
    """`count` sessions sharing one planning-call + tool-sequence skeleton —
    the exact shape both `reuse` (cache-reuse over the repeated plan) and
    `script` (deterministic tool-call cluster) key off, mirroring
    `test_cross_analyzer_recoverable_invariant.py::_seed_repeated_skeleton_cluster`.
    Session `agent_id` is left at the factory default (not `claude-code-*`),
    so the window resolves to the `sdk` persona and neither analyzer is
    persona-gated off."""
    tool_names = ["read", "edit", "run"]
    for i in range(count):
        sid = f"cluster-{i}"
        db.upsert_session(make_session(
            session_id=sid, plan_tier="api",
            input_tokens=1_000, output_tokens=200, total_cost_usd=0.20,
        ))
        t0 = BASE + timedelta(hours=2, minutes=i)
        plan = make_llm_span(
            model=MODEL, provider="anthropic",
            session_id=sid, start_time=t0, cost_usd=0.20,
            input_tokens=1_000, output_tokens=200,
            extra_attributes={
                GenAIAttributes.PROMPT_CONTENT: "cut release on monday",
            },
        )
        db.insert_span(plan)
        for j, tn in enumerate(tool_names):
            ts = make_tool_span(tool_name=tn)
            ts.session_id = sid
            ts.start_time = t0 + timedelta(seconds=j + 1)
            ts.attributes = {GenAIAttributes.TOOL_INPUT: {"path": f"/repo/f{j}.py"}}
            db.insert_span(ts)


def _config(tmp_path) -> TjConfig:
    return TjConfig(
        version="1",
        storage=StorageConfig(path=str(tmp_path / "t.duckdb")),
        api=ApiConfig(auth=ApiAuthConfig(enabled=False)),
        # A background scan on app startup would race this test's own
        # explicit recompute below (see the identical guard in
        # test_scan_cycle_provenance.py's `client` fixture).
        optimize=OptimizeConfig(scan_enabled=False),
        capture=CaptureConfig(prompts=True, tool_inputs=True),
    )


def test_wire_total_is_netted_not_the_naive_per_analyzer_sum(tmp_path):
    config = _config(tmp_path)
    db = InMemoryBackend()
    _seed_overlapping_cluster(db)

    # The RAW, un-netted figures — exactly what the Dashboard's old
    # `totalOpportunityFigure(tiles)` summed, one `past_overspend_usd` per
    # analyzer straight off `/optimize`'s findings, before any cross-analyzer
    # netting is applied.
    report = runner.build_report(
        db=db, config=config, since=SINCE, until=UNTIL,
        findings=["reuse", "script"],
    )
    # Wave-2 analyzers attach to the generic `findings` dict, keyed by their
    # registration name — the same field the Dashboard's JS reads
    # (`opt.findings`, `recoverableTiles()` in ui/index.html).
    reuse_raw = getattr(report.findings.get("reuse"), "past_overspend_usd", None) or 0.0
    script_raw = getattr(report.findings.get("script"), "past_overspend_usd", None) or 0.0
    assert reuse_raw > 0 and script_raw > 0, (
        "fixture must trip both reuse and script with a nonzero figure, or "
        "this test cannot tell netted from naive"
    )
    naive_sum = reuse_raw + script_raw

    # The netted figure the wire actually serves: recompute (background-job
    # shape) writes the adapted, netted proposals to the store, and
    # GET /relearn/cost-proposals reads them back — the same path the
    # Dashboard hero and total tile now fetch.
    app = create_app(config=config, db=db, ingest_pipeline=build_default_pipeline(db, config))
    cost_proposals_mod.recompute_cost_proposals(db, config, until=UNTIL)
    with TestClient(app) as client:
        body = client.get("/api/v1/relearn/cost-proposals").json()

    rollup = body["past_overspend"]
    by_analyzer = {row["analyzer"]: row for row in rollup["by_analyzer"]}
    assert "reuse" in by_analyzer and "script" in by_analyzer, (
        "fixture must trip both reuse and script on the wire too, or this "
        "test cannot tell netted from naive"
    )
    netted_reuse_script = by_analyzer["reuse"]["usd"] + by_analyzer["script"]["usd"]
    # The whole point: reuse and script both claim the SAME 20 sessions here,
    # so a plain sum double-counts them. The netted pair the wire serves must
    # be strictly less than what a client-side sum of the raw figures would
    # have produced.
    assert netted_reuse_script < naive_sum
    # And the netted rollup's OWN published total can never exceed the
    # naive per-analyzer sum either — it is the number the Dashboard hero and
    # total tile must render instead of re-deriving one client-side.
    assert rollup["past_overspend_usd"] <= naive_sum
