"""The Review inbox's headline total covers EVERY row the inbox lists.

The inbox is one list merging two endpoints: ``/relearn/cost-proposals``
(cost proposals, plus the headline aggregate) and ``/relearn/proposals``
(relearn clusters). The headline was summed over the cost feed only, so relearn's
rows sat outside it and every sentence derived from the whole list was false for
the part it could not see.

These tests talk through the real ASGI app because the invariant is a property of
the two responses TOGETHER: the aggregate one endpoint publishes has to equal the
sum of the ``inbox_contribution_usd`` field carried by every row of both. A unit
test cannot catch the two routes drifting onto different window bases; this can.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest

from tokenjam.core.rulewrite.kinds import DELIVERY_CLAUDE_MD_RULE

from tokenjam.api.app import create_app
from tokenjam.core.config import ApiAuthConfig, ApiConfig, StorageConfig, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest import IngestPipeline
from tokenjam.core.optimize import relearn_store
from tokenjam.core.optimize.analyzers.relearn import (
    RelearnCluster,
    RelearnFinding,
)
from tokenjam.core.optimize.relearn_window import (
    WINDOWED_BASIS,
    RelearnWindowedObservation,
    sum_windowed,
)
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span

ANCHOR = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


@pytest.fixture
def config(tmp_path):
    return TjConfig(
        version="1",
        api=ApiConfig(auth=ApiAuthConfig(enabled=False)),
        storage=StorageConfig(path=str(tmp_path / "telemetry.duckdb")),
    )


@pytest.fixture
def app(config, db):
    # Low-cache-efficacy spans so a real cost proposal exists to sum beside
    # relearn's clusters. Without a cost row the invariant would be trivially
    # true over one population.
    now = utcnow()
    for i in range(12):
        db.insert_span(make_llm_span(
            agent_id="svc-a", provider="anthropic", model="claude-sonnet-5",
            billing_account="anthropic", input_tokens=15_000, output_tokens=200,
            cache_tokens=400, session_id=f"s-{i}",
            start_time=now - timedelta(days=2, minutes=i),
        ))
    # The corpus has to be 30 days WIDE, not merely 30 days' worth of rows. The
    # inbox window is the analysis span bounded by how far back this store's
    # oldest row actually sits, so a two-day corpus resolves to a two-day
    # window — and the hand-built relearn buckets below, which are the point of
    # these tests, are labelled `30d`. One row at the far edge makes the two
    # sides name the same window without changing what any analyzer finds.
    db.insert_span(make_llm_span(
        agent_id="svc-a", provider="anthropic", model="claude-sonnet-5",
        billing_account="anthropic", input_tokens=15_000, output_tokens=200,
        cache_tokens=400, session_id="s-oldest",
        start_time=now - timedelta(days=29),
    ))
    pipeline = IngestPipeline(db=db, config=config)
    return create_app(config=config, db=db, ingest_pipeline=pipeline)


@pytest.fixture
def client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def _observation(
    label: str, days: float, *, usd: float, reread_usd: float,
    tokens: int, reread_tokens: int, occurrences: int,
) -> RelearnWindowedObservation:
    """A detector-shaped bounded bucket. Built directly rather than by running
    the detector: pricing a real scan needs a rate profile off ingested spans for
    the cluster's own sessions, and what these tests are about is the two ROUTES
    agreeing on one basis, not the detector's arithmetic (``tests/unit/
    test_relearn_window.py`` owns that)."""
    return RelearnWindowedObservation(
        label=label, window_days=days,
        window_start=(ANCHOR - timedelta(days=days)).isoformat(),
        window_end=ANCHOR.isoformat(),
        occurrences=occurrences, sessions=3, detour_turns=6.0,
        undated_occurrences=0, tail_calls_median=2, tail_multiplier=1.3,
        past_overspend_tokens=tokens, past_overspend_usd=usd,
        past_reread_tokens=reread_tokens, past_reread_usd=reread_usd,
        capped_at_unbounded=False, basis=WINDOWED_BASIS,
    )


def _cluster(
    signature: str, *, unbounded_usd: float, unbounded_tokens: int,
    windows: dict[str, RelearnWindowedObservation] | None,
) -> RelearnCluster:
    return RelearnCluster(
        signature=signature, family_key=signature, title=f"{signature} recurs",
        sessions=3, occurrences=9, repos=["demo"], delivery=DELIVERY_CLAUDE_MD_RULE, scope="project",
        proposed_fix="Do the thing that stops this.",
        past_overspend_usd=unbounded_usd, past_overspend_tokens=unbounded_tokens,
        past_reread_usd=round(unbounded_usd * 0.1, 6),
        past_reread_tokens=unbounded_tokens // 10,
        past_overspend_windows=windows,
    )


def _seed_relearn_cache(config, *, windowed: bool = True):
    """A cached finding with a big cluster and a below-floor one.

    The small cluster is the case the below-floor disclosure sentence was false
    about: its money has to reach the headline even though no row for it renders.
    """
    big_windows = {
        "24h": _observation("24h", 1.0, usd=2.0, reread_usd=0.4,
                            tokens=200, reread_tokens=40, occurrences=1),
        "30d": _observation("30d", 30.0, usd=30.0, reread_usd=5.0,
                            tokens=3_000, reread_tokens=500, occurrences=7),
    } if windowed else None
    small_windows = {
        # Quiet in the last day: a known zero for that window, not an unknown.
        "24h": _observation("24h", 1.0, usd=0.0, reread_usd=0.0,
                            tokens=0, reread_tokens=0, occurrences=0),
        "30d": _observation("30d", 30.0, usd=1.20, reread_usd=0.20,
                            tokens=120, reread_tokens=20, occurrences=2),
    } if windowed else None
    clusters = [
        _cluster("cwd_confusion", unbounded_usd=40.0, unbounded_tokens=400_000,
                 windows=big_windows),
        _cluster("read_before_write", unbounded_usd=9.0, unbounded_tokens=90_000,
                 windows=small_windows),
    ]
    per_cluster = [c.past_overspend_windows for c in clusters]
    totals = None
    if windowed:
        totals = {
            label: sum_windowed(
                per_cluster, label,
                anchor_start=(ANCHOR - timedelta(days=1)).isoformat(),
                anchor_end=ANCHOR.isoformat(),
            )
            for label in ("24h", "30d")
        }
    finding = RelearnFinding(
        clusters=clusters, sessions_scanned=6, failures_examined=18,
        past_overspend_usd=49.0, past_overspend_tokens=490_000,
        past_reread_usd=4.9, past_reread_tokens=49_000,
        past_overspend_windows=totals,
    )
    relearn_store.write_cache(finding, config=config)
    return finding


async def _refresh_cost(app, c):
    r = await c.post(
        "/api/v1/relearn/cost-proposals/refresh",
        headers={"X-TJ-Local-Token": app.state.relearn_write_token},
    )
    assert r.status_code == 200, r.text


@pytest.mark.asyncio
async def test_the_headline_equals_the_sum_of_every_inbox_rows_contribution(
    app, client, config,
):
    _seed_relearn_cache(config)
    async with client as c:
        await _refresh_cost(app, c)
        cost = (await c.get("/api/v1/relearn/cost-proposals")).json()
        relearn = (await c.get("/api/v1/relearn/proposals")).json()

    rollup = cost["past_overspend"]
    rows = list(cost["proposals"]) + list(relearn["finding"]["clusters"])
    assert len(cost["proposals"]) >= 1
    assert len(relearn["finding"]["clusters"]) == 2

    # The field is PRESENT on every row, so a renderer never has to guess which
    # rows it applies to. Its value may be None: an unpriced row is unpriced, not
    # cheap, and it contributes nothing to a total rather than a zero.
    assert all("inbox_contribution_usd" in r for r in rows), rows
    contributions = [
        r["inbox_contribution_usd"] for r in rows
        if r["inbox_contribution_usd"] is not None
    ]
    assert rollup["past_overspend_usd"] == pytest.approx(sum(contributions), abs=1e-6)
    # Relearn's share is attributable in the breakdown, never an unexplained delta.
    assert "relearn" in {e["analyzer"] for e in rollup["by_analyzer"]}
    # Nothing was pushed into the excluded channel: every row made it in.
    assert "relearn" not in rollup["excluded"]


@pytest.mark.asyncio
async def test_every_row_publishes_its_contribution_under_the_headlines_window(
    app, client, config,
):
    _seed_relearn_cache(config)
    async with client as c:
        await _refresh_cost(app, c)
        cost = (await c.get("/api/v1/relearn/cost-proposals")).json()
        relearn = (await c.get("/api/v1/relearn/proposals")).json()

    window_days = cost["past_overspend"]["window_days"]
    labels = {
        r["inbox_contribution_window"]
        for r in list(cost["proposals"]) + list(relearn["finding"]["clusters"])
    }
    # ONE label across both feeds, and it is the one the headline names.
    assert labels == {f"{int(window_days)}d"}


@pytest.mark.asyncio
async def test_a_relearn_rows_contribution_is_net_of_its_reread_share(client, config):
    _seed_relearn_cache(config)
    async with client as c:
        relearn = (await c.get("/api/v1/relearn/proposals")).json()

    for cluster in relearn["finding"]["clusters"]:
        bucket = cluster["past_overspend_windows"]["30d"]
        expected = round(
            max(bucket["past_overspend_usd"] - bucket["past_reread_usd"], 0.0), 6)
        assert cluster["inbox_contribution_usd"] == pytest.approx(expected)
        # The unbounded figures the write budget nets against are untouched.
        assert cluster["past_overspend_usd"] >= cluster["inbox_contribution_usd"]


@pytest.mark.asyncio
async def test_a_readers_own_since_does_not_move_the_headlines_basis(client, config):
    """``since`` bounds the ROWS a reader sees. The contribution field stays on
    the headline's window, because the floor, the tail sum and the headline have
    to be one quantity however the reader is filtering."""
    _seed_relearn_cache(config)
    async with client as c:
        wide = (await c.get("/api/v1/relearn/proposals")).json()
        narrow = (await c.get(
            "/api/v1/relearn/proposals", params={"since": "24h"})).json()

    assert narrow["window"]["applied"] == "24h"
    by_sig = {c["signature"]: c for c in wide["finding"]["clusters"]}
    for cluster in narrow["finding"]["clusters"]:
        assert cluster["inbox_contribution_window"] == "30d"
        assert cluster["inbox_contribution_usd"] == pytest.approx(
            by_sig[cluster["signature"]]["inbox_contribution_usd"])


@pytest.mark.asyncio
async def test_a_cache_without_bounded_figures_discloses_relearn_as_excluded(
    app, client, config,
):
    """Unknown is never zero. A pre-windowing cache cannot put relearn's money on
    the headline's window, so it is stated through ``excluded`` and every relearn
    row reports an absent contribution rather than a free cluster."""
    _seed_relearn_cache(config, windowed=False)
    async with client as c:
        await _refresh_cost(app, c)
        cost = (await c.get("/api/v1/relearn/cost-proposals")).json()
        relearn = (await c.get("/api/v1/relearn/proposals")).json()

    rollup = cost["past_overspend"]
    excluded = rollup["excluded"]["relearn"]
    assert excluded["clusters"] == 2
    assert excluded["past_overspend_usd"] > 0
    assert "relearn" not in {e["analyzer"] for e in rollup["by_analyzer"]}
    # Stated, and summed into nothing.
    cost_only = sum(
        p["inbox_contribution_usd"] for p in cost["proposals"]
        if p["inbox_contribution_usd"] is not None
    )
    assert rollup["past_overspend_usd"] == pytest.approx(cost_only, abs=1e-6)

    for cluster in relearn["finding"]["clusters"]:
        assert cluster["inbox_contribution_usd"] is None
        assert cluster["inbox_contribution_tokens"] is None
        assert "unknown, not zero" in cluster["inbox_contribution_basis"]


@pytest.mark.asyncio
async def test_an_applied_cluster_leaves_the_total_as_it_leaves_the_list(
    app, client, config,
):
    """The headline answers "what is still outstanding", for both feeds."""
    finding = _seed_relearn_cache(config)
    approved = finding.clusters[0].signature
    async with client as c:
        await _refresh_cost(app, c)
        before = (await c.get("/api/v1/relearn/cost-proposals")).json()["past_overspend"]

    # The ledger is JSON on disk; write the record the apply path would have
    # written rather than driving a real file write through the whole apply
    # stack, which is not what this test is about.
    import json

    from tokenjam.core.optimize.relearn_apply import _storage_base_dir

    ledger = _storage_base_dir(config) / "applied_fixes.json"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(json.dumps([
        {"id": "f1", "signature": approved, "state": "applied"},
    ]), encoding="utf-8")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as c:
        after = (await c.get("/api/v1/relearn/cost-proposals")).json()["past_overspend"]

    assert after["past_overspend_usd"] < before["past_overspend_usd"]
