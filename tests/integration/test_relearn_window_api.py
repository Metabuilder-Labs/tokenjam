"""`since` on the two endpoints that had no window at all.

``/api/v1/relearn/proposals`` and ``/api/v1/drift`` were the only window-shaped
reads on the Dashboard that took no window parameter, so the window selector
visibly governed three tiles and silently did nothing to the other two: the same
42 clusters came back for ``?since=24h`` and ``?since=90d``. A control that
governs some of the page and not the rest is worse than no control, because a
reader has no way to tell which figures moved.

Both now accept ``since``, parsed by the SAME ``parse_since`` helper the other
routes use and rejected with a 400 (not a 500) when malformed, and both report
back the window they ACTUALLY applied rather than the one that was asked for.
The two differ because the underlying data differs: relearn serves a cached
finding whose bounded figures were precomputed by the detector, so a `since`
with no matching bucket resolves to the nearest one and says so; drift reads
live rows, so it applies the caller's window exactly.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest

from tokenjam.api.app import create_app
from tokenjam.core.config import ApiAuthConfig, ApiConfig, StorageConfig, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest import IngestPipeline
from tokenjam.core.models import DriftBaseline
from tokenjam.core.optimize import relearn_store
from tokenjam.core.optimize.analyzers.relearn import FailureEpisode, analyze_relearns
from tokenjam.core.optimize.relearn_window import RELEARN_WINDOW_LABELS
from tests.factories import make_session

ANCHOR = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)


def utc_noon(days_ago: int) -> datetime:
    """A seed instant that cannot straddle a day boundary, in any timezone.

    The day-span measure counts DISTINCT UTC days and discards anything dated
    later than today, so a fixture seeded at an offset from ``utcnow()`` puts
    its rows wherever the wall clock happens to be: rows land on one calendar
    day for part of the day and two for the rest, and a row seeded at "now" can
    read as tomorrow under a database timezone that runs ahead of UTC. Both
    made the assertions below go red for a few hours out of every day and green
    again afterwards, which reads as "whoever pushed last broke it".

    Noon UTC is the furthest any instant can be from both boundaries, so the
    day a row belongs to is fixed no matter when or where the suite runs.
    """
    today = datetime.now(tz=timezone.utc).date()
    return datetime(
        today.year, today.month, today.day, 12, 0, tzinfo=timezone.utc,
    ) - timedelta(days=days_ago)


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
    pipeline = IngestPipeline(db=db, config=config)
    return create_app(config=config, db=db, ingest_pipeline=pipeline)


@pytest.fixture
def client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def _failure(session: str, days_ago: float, *, text: str) -> FailureEpisode:
    return FailureEpisode(
        session_id=session, repo="repo-a",
        ts=(ANCHOR - timedelta(days=days_ago)).isoformat().replace("+00:00", "Z"),
        tool_name="Bash", label="", error_text=text,
        kind="act", is_retry=False, depth=0, detour_turns=2.0,
    )


def _seed_relearn_cache(config, *, window_labels=RELEARN_WINDOW_LABELS, anchor=ANCHOR):
    """A cached finding with one RECENT cluster and one OLD one.

    Two distinct failure signatures so they cluster separately: the recent one
    is inside every window, the old one only inside the widest.
    """
    recent = [
        _failure(f"recent{i}", days_ago, text="(eval):cd:1: no such file or directory: x")
        for i, days_ago in enumerate([0.1, 0.2, 0.3])
    ]
    old = [
        _failure(f"old{i}", days_ago, text="File has not been read yet. Read it first.")
        for i, days_ago in enumerate([40, 45, 50])
    ]
    finding = analyze_relearns(
        [], extra_failures=recent + old, distill_enabled=False, min_sessions=3,
        window_labels=window_labels, window_anchor=anchor,
    )
    assert len(finding.clusters) == 2, "fixture must produce two clusters"
    relearn_store.write_cache(finding, config=config)
    return finding


# --- /relearn/proposals ----------------------------------------------------- #

@pytest.mark.asyncio
async def test_no_since_is_todays_unbounded_behaviour(client, config):
    _seed_relearn_cache(config)
    async with client as c:
        body = (await c.get("/api/v1/relearn/proposals")).json()
    assert body["status"] == "ready"
    assert len(body["finding"]["clusters"]) == 2
    assert body["window"]["applied"] is None
    assert body["window"]["since_requested"] is None
    # No window was applied, so there is no windowed figure to publish. Absent,
    # not zero.
    assert body.get("past_overspend_windowed") is None


@pytest.mark.asyncio
async def test_since_changes_the_result_and_is_reported_back(client, config):
    _seed_relearn_cache(config)
    async with client as c:
        narrow = (await c.get("/api/v1/relearn/proposals", params={"since": "24h"})).json()
        wide = (await c.get("/api/v1/relearn/proposals", params={"since": "90d"})).json()

    # The old cluster has no occurrence in the last 24 hours, so it is not part
    # of what that window observed.
    assert len(narrow["finding"]["clusters"]) == 1
    assert len(wide["finding"]["clusters"]) == 2

    assert narrow["window"]["applied"] == "24h"
    assert narrow["window"]["since_requested"] == "24h"
    assert narrow["window"]["clusters_in_window"] == 1
    assert narrow["window"]["clusters_omitted_outside_window"] == 1
    # The anchor is when the DETECTOR ran, not when the page was opened.
    assert narrow["window"]["window_end"] == ANCHOR.isoformat()

    # And the figures move with it, in the only direction a filter can move them.
    assert narrow["past_overspend_windowed"]["past_overspend_tokens"] < \
        wide["past_overspend_windowed"]["past_overspend_tokens"]
    assert wide["past_overspend_windowed"]["past_overspend_tokens"] <= \
        wide["finding"]["past_overspend_tokens"]


@pytest.mark.asyncio
async def test_every_returned_row_carries_the_same_windowed_figure_the_total_sums(
    client, config,
):
    """The floor note and the headline must read one quantity.

    ``belowInboxFloor``/``BelowFloorNote`` test and sum each ROW's own figure to
    say "N smaller items are hidden, $X combined, still counted in the total
    above". That sentence is only true while the total is the sum of exactly the
    rows' own figures, so the windowed total is asserted against the rows here
    rather than trusted.
    """
    _seed_relearn_cache(config)
    async with client as c:
        body = (await c.get("/api/v1/relearn/proposals", params={"since": "90d"})).json()
    rows = body["finding"]["clusters"]
    assert all(r["window"]["label"] == "90d" for r in rows)
    assert body["past_overspend_windowed"]["past_overspend_tokens"] == sum(
        r["window"]["past_overspend_tokens"] for r in rows
    )


@pytest.mark.asyncio
async def test_a_since_with_no_precomputed_bucket_snaps_and_says_which(client, config):
    _seed_relearn_cache(config)
    async with client as c:
        body = (await c.get("/api/v1/relearn/proposals", params={"since": "45d"})).json()
    # Asked for 45d, no such bucket exists; 30d was applied and the response
    # names it rather than labelling 30d figures "45d".
    assert body["window"]["since_requested"] == "45d"
    assert body["window"]["applied"] == "30d"
    assert body["window"]["window_days"] == pytest.approx(30.0)


@pytest.mark.asyncio
async def test_malformed_since_is_a_400_not_a_500(client, config):
    _seed_relearn_cache(config)
    async with client as c:
        resp = await c.get("/api/v1/relearn/proposals", params={"since": "banana"})
    assert resp.status_code == 400
    assert "since" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_a_cache_without_windowed_figures_says_so_rather_than_faking_one(
    client, config,
):
    """A cache written by an older build carries no bounded figures at all.

    The window cannot be honored, and neither wrong answer is acceptable: an
    empty list would claim the window observed nothing, and silently returning
    the unbounded figures under the requested label would publish a corpus-wide
    total as a 24-hour one. So the rows come back unfiltered and the response
    states that no window was applied and why.
    """
    _seed_relearn_cache(config, window_labels=None)
    async with client as c:
        body = (await c.get("/api/v1/relearn/proposals", params={"since": "24h"})).json()
    assert len(body["finding"]["clusters"]) == 2
    assert body["window"]["applied"] is None
    assert body["window"]["unavailable_reason"]
    assert body.get("past_overspend_windowed") is None


@pytest.mark.asyncio
async def test_the_days_of_data_available_measure_is_served(client, config, db):
    _seed_relearn_cache(config)
    db.upsert_session(make_session(
        session_id="s1", agent_id="claude-code-x", started_at=utc_noon(0),
    ))
    async with client as c:
        body = (await c.get("/api/v1/relearn/proposals")).json()
    span = body["data_span"]
    assert span["available_days"] == 1
    assert span["days_with_data"] == 1


@pytest.mark.asyncio
async def test_an_ancient_row_does_not_inflate_the_served_span(client, config, db):
    """The route must serve the ROBUST measure, not newest-minus-oldest.

    One 2020-dated row is enough to make the naive span read in the thousands of
    days; the served figure must not move by more than the one day that row
    actually carries data on.
    """
    _seed_relearn_cache(config)
    db.upsert_session(make_session(
        session_id="s1", agent_id="a", started_at=utc_noon(0),
    ))
    db.upsert_session(make_session(
        session_id="ancient", agent_id="a",
        started_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        ended_at=datetime(2020, 1, 1, 1, tzinfo=timezone.utc),
    ))
    async with client as c:
        body = (await c.get("/api/v1/relearn/proposals")).json()
    span = body["data_span"]
    assert span["available_days"] == 1
    assert span["days_with_data"] == 2
    assert span["ignored_days_before_block"] == 1


# --- /drift ----------------------------------------------------------------- #

def _seed_drift(db, *, agent_id: str, session_age_days: float) -> None:
    now = datetime.now(tz=timezone.utc)
    ended = now - timedelta(days=session_age_days)
    db.upsert_baseline(DriftBaseline(
        agent_id=agent_id, sessions_sampled=10, computed_at=now,
        avg_input_tokens=1000.0, stddev_input_tokens=10.0,
    ))
    db.upsert_session(make_session(
        session_id=f"{agent_id}-s1", agent_id=agent_id,
        started_at=ended - timedelta(minutes=1), ended_at=ended,
        input_tokens=99_000,
    ))


@pytest.mark.asyncio
async def test_drift_without_since_is_unchanged(client, db):
    _seed_drift(db, agent_id="stale-agent", session_age_days=60)
    async with client as c:
        body = (await c.get("/api/v1/drift")).json()
    agents = body["agents"]
    assert len(agents) == 1
    assert agents[0]["latest_session"] is not None
    assert body["window"]["since"] is None


@pytest.mark.asyncio
async def test_drift_since_excludes_a_session_outside_the_window(client, db):
    _seed_drift(db, agent_id="stale-agent", session_age_days=60)
    _seed_drift(db, agent_id="live-agent", session_age_days=0.1)
    async with client as c:
        wide = (await c.get("/api/v1/drift", params={"since": "90d"})).json()
        narrow = (await c.get("/api/v1/drift", params={"since": "24h"})).json()

    def with_latest(body):
        return sorted(a["agent_id"] for a in body["agents"] if a["latest_session"])

    assert with_latest(wide) == ["live-agent", "stale-agent"]
    # A drift SIGNAL needs a session in the window to compare the baseline
    # against. The stale agent's baseline is still listed; what it has no
    # in-window observation of is said explicitly rather than left as a bare
    # null a reader would read as "never ran".
    assert with_latest(narrow) == ["live-agent"]
    stale = next(a for a in narrow["agents"] if a["agent_id"] == "stale-agent")
    assert stale["baseline"] is not None
    assert stale["latest_session_outside_window"] is True
    assert stale["latest_session_at"]
    assert narrow["window"]["since"] == "24h"
    assert narrow["window"]["start"]


@pytest.mark.asyncio
async def test_drift_since_applies_to_a_single_agent_read_too(client, db):
    _seed_drift(db, agent_id="stale-agent", session_age_days=60)
    async with client as c:
        body = (await c.get(
            "/api/v1/drift", params={"agent_id": "stale-agent", "since": "24h"},
        )).json()
    assert body["latest_session"] is None
    assert body["latest_session_outside_window"] is True
    assert body["window"]["since"] == "24h"


@pytest.mark.asyncio
async def test_drift_malformed_since_is_a_400_not_a_500(client, db):
    _seed_drift(db, agent_id="a", session_age_days=1)
    async with client as c:
        resp = await c.get("/api/v1/drift", params={"since": "banana"})
    assert resp.status_code == 400
    assert "since" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_drift_serves_the_same_days_available_measure(client, db):
    _seed_drift(db, agent_id="a", session_age_days=1)
    async with client as c:
        body = (await c.get("/api/v1/drift")).json()
    assert body["data_span"]["available_days"] is not None
