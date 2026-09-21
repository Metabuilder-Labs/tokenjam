"""The bridge, sender side (ledger W5; contracts §3, §6, §9, §12).

Every forwarding test runs the real `run_sync` over a REAL DuckDB store
against a recorded fake Cloud (`httpx.MockTransport`), so the bodies asserted
here are the bytes that would leave the machine. The OTLP body is round-tripped
through `parse_otlp_span`, which is the parser Cloud runs, so "the receiver
rebuilds the span" is proved rather than assumed.
"""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from tokenjam.core import cloud_sync as cs
from tokenjam.core.config import (
    CaptureConfig,
    CloudConfig,
    StorageConfig,
    TjConfig,
    load_config,
    write_config,
)
from tokenjam.core.db import DuckDBBackend
from tokenjam.core.models import SessionCommit
from tokenjam.otel.otlp_parsing import iter_otlp_spans, parse_otlp_span
from tokenjam.utils.time_parse import utcnow

from tests.factories import make_llm_span, make_session, make_tool_span

T0 = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
KEY = "tj_live_" + "k" * 43
ORG = "org_abc123"
SECRET = "TOP SECRET PROMPT TEXT"


# --- Fixtures ----------------------------------------------------------------------

@pytest.fixture
def db(tmp_path) -> DuckDBBackend:
    backend = DuckDBBackend(StorageConfig(path=str(tmp_path / "t.duckdb")))
    yield backend
    backend.close()


@pytest.fixture
def state_file(tmp_path) -> Path:
    return tmp_path / "cloud_sync.json"


class FakeCloud:
    """Records every request; answers per a scripted status list, then 200."""

    def __init__(self, statuses: list[int] | None = None, *, bad_span: str | None = None,
                 accept_key: str = KEY):
        self.requests: list[httpx.Request] = []
        self.accepted: list[dict] = []
        self.statuses = list(statuses or [])
        self.bad_span = bad_span
        self.accept_key = accept_key

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.headers.get("Authorization") != f"Bearer {self.accept_key}":
            return httpx.Response(401, json={"detail": "Invalid organization or ingest key"})
        if self.statuses:
            status = self.statuses.pop(0)
            if status != 200:
                headers = {"Retry-After": "0"} if status in (429, 503) else {}
                return httpx.Response(status, json={"error": f"scripted {status}"}, headers=headers)
        body = json.loads(request.content)
        if self.bad_span and any(
            s.get("spanId") == self.bad_span for s in self.spans(body)
        ):
            return httpx.Response(400, json={"error": "scripted refusal"})
        self.accepted.append(body)
        return httpx.Response(200, json={"ingested": 1, "rejected": 0, "rejections": []})

    @staticmethod
    def spans(body: dict) -> list[dict]:
        return [raw for raw, _ in iter_otlp_spans(body)]

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)

    def bodies(self, path: str) -> list[dict]:
        return [json.loads(r.content) for r in self.requests if r.url.path == path]


def _config(**cloud) -> TjConfig:
    defaults = dict(org_id=ORG, ingest_key=KEY, endpoint="https://cloud.test")
    defaults.update(cloud)
    return TjConfig(version="1", cloud=CloudConfig(**defaults))


def _client(config: TjConfig, cloud: FakeCloud, *, sleeps: list[float] | None = None) -> cs.CloudClient:
    return cs.CloudClient(
        config, transport=cloud.transport,
        sleep=(sleeps.append if sleeps is not None else (lambda _s: None)),
    )


def _seed(db: DuckDBBackend, *, sessions: int = 1, spans_per: int = 2, commits: bool = True,
          at: datetime = T0, prompt: str | None = SECRET) -> None:
    for i in range(sessions):
        sid = f"sess-{i}"
        start = at + timedelta(hours=i)
        db.upsert_session(replace(
            make_session(session_id=sid, agent_id="claude-code-widgets", started_at=start,
                         total_cost_usd=1.0 + i, input_tokens=100, output_tokens=10,
                         plan_tier="max_20x"),
            ended_at=start + timedelta(minutes=5),
            repo_remote="https://github.com/Acme/widgets", repo_root="/w/widgets",
            branch_start="main", branch_end="feat/x", head_sha_start="a" * 40,
            user_email="dev@example.com", developer_id="deadbeefcafe0000",
            bridge_session_id="cse_1",
        ))
        for j in range(spans_per):
            extra = {"gen_ai.prompt.content": prompt, "llm.prompts": prompt} if prompt else {}
            db.insert_span(make_llm_span(
                agent_id="claude-code-widgets", session_id=sid,
                start_time=start + timedelta(seconds=j), cost_usd=0.25,
                input_tokens=10, output_tokens=5, cache_tokens=3, cache_write_tokens=2,
                extra_attributes=extra,
            ))
        db.insert_span(make_tool_span(
            agent_id="claude-code-widgets", tool_name="Bash", session_id=sid,
            start_time=start + timedelta(seconds=30),
            tool_input={"command": "git commit -m 'x'"},
        ))
        if commits:
            db.upsert_session_commits([SessionCommit(
                session_id=sid, commit_sha=("%040x" % (i + 1)), confidence="inferred",
                source="trailer_window", repo_remote="https://github.com/Acme/widgets",
                author_email="dev@example.com", committed_at=start + timedelta(seconds=31),
                matched_at=start + timedelta(minutes=20), match_delta_s=None,
            )])


def _run(db, config, cloud, state_file, **kw) -> cs.SyncReport:
    return cs.run_sync(db, config, client=_client(config, cloud), state_file=state_file,
                       install_id="inst-1", host_name="devbox", **kw)


# --- Config round trip + key parsing (scope 1) ------------------------------------------

def test_cloud_block_round_trips_and_is_absent_until_configured(tmp_path):
    path = tmp_path / "c.toml"
    write_config(TjConfig(version="1"), path)
    assert "[cloud]" not in path.read_text()
    loaded = load_config(str(path))
    assert loaded.cloud.configured is False and loaded.cloud.is_active is False

    cfg = TjConfig(version="1", cloud=CloudConfig(org_id=ORG, ingest_key=KEY, forward_content=True))
    write_config(cfg, path)
    text = path.read_text()
    assert "[cloud]" in text and f'ingest_key = "{KEY}"' in text
    back = load_config(str(path)).cloud
    assert back == CloudConfig(enabled=True, endpoint=CloudConfig.endpoint,
                               org_id=ORG, ingest_key=KEY, forward_content=True)
    assert back.is_active

    cfg.cloud.enabled = False
    write_config(cfg, path)
    back = load_config(str(path)).cloud
    assert back.configured and not back.is_active


def test_parse_ingest_key_accepts_cloud_shaped_keys_only():
    assert cs.parse_ingest_key(f"  {KEY}\n") == KEY
    with pytest.raises(ValueError, match="starts with"):
        cs.parse_ingest_key("sk-ant-nope")
    with pytest.raises(ValueError, match="truncated"):
        cs.parse_ingest_key("tj_live_short")
    with pytest.raises(ValueError):
        cs.parse_ingest_key("")


def test_auth_headers_are_exactly_the_two_cloud_reads():
    headers = cs.auth_headers(_config(org_id=" org_x ", ingest_key=f" {KEY} "))
    assert headers["X-TokenJam-Org"] == "org_x"
    assert headers["Authorization"] == f"Bearer {KEY}"
    assert headers["Content-Type"] == "application/json"


# --- Nothing leaves without [cloud] (pinned) -------------------------------------------

def test_nothing_is_sent_when_cloud_is_absent_or_disabled(db, state_file):
    _seed(db)
    cloud = FakeCloud()
    for config in (TjConfig(version="1"), _config(enabled=False),
                   _config(org_id=""), _config(ingest_key="")):
        report = _run(db, config, cloud, state_file)
        assert report.skipped_reason and not report.sent_anything
    assert cloud.requests == []
    assert not state_file.exists()
    assert cs.start_sync(lambda: db, TjConfig(version="1"), state_file=state_file) is None
    assert cloud.requests == []


# --- Wire format (contracts §6) ---------------------------------------------------------

def test_sessions_commits_and_spans_reach_the_two_routes_with_the_contract_shape(db, state_file):
    _seed(db)
    config = _config()
    cloud = FakeCloud()
    report = _run(db, config, cloud, state_file)
    assert (report.sessions_sent, report.commits_sent, report.spans_sent) == (1, 1, 3)
    assert report.stopped is None

    paths = [r.url.path for r in cloud.requests]
    assert paths == [cs.LEDGER_SESSIONS_PATH, cs.LEDGER_SESSIONS_PATH, cs.SPANS_PATH]
    for r in cloud.requests:
        assert r.headers["X-TokenJam-Org"] == ORG
        assert r.headers["Authorization"] == f"Bearer {KEY}"
        assert r.url.host == "cloud.test"

    sessions_body, commits_body = cloud.bodies(cs.LEDGER_SESSIONS_PATH)
    assert set(sessions_body) == {"sessions", "session_commits"}
    assert sessions_body["session_commits"] == []
    (row,) = sessions_body["sessions"]
    assert row == {
        "session_id": "sess-0", "agent_id": "claude-code-widgets",
        "conversation_id": row["conversation_id"],
        "started_at": "2026-09-01T10:00:00+00:00", "ended_at": "2026-09-01T10:05:00+00:00",
        "status": "completed", "total_cost_usd": 1.0,
        "input_tokens": 100, "output_tokens": 10, "cache_tokens": 0, "cache_write_tokens": 0,
        "tool_call_count": 0, "error_count": 0,
        "plan_tier": "max_20x", "pricing_mode": "subscription", "source": None,
        "service_namespace": None, "service_instance_id": None,
        "run_id": None, "parent_session_id": None, "bridge_session_id": "cse_1",
        "install_id": "inst-1",
        "repo_remote": "https://github.com/Acme/widgets", "repo_root": "/w/widgets",
        "branch_start": "main", "branch_end": "feat/x",
        "head_sha_start": "a" * 40, "head_sha_end": None,
        "developer_id": "deadbeefcafe0000", "user_email": "dev@example.com",
    }
    assert commits_body["sessions"] == []
    (commit,) = commits_body["session_commits"]
    assert commit == {
        "session_id": "sess-0", "commit_sha": "%040x" % 1,
        "repo_remote": "https://github.com/Acme/widgets",
        "confidence": "inferred", "source": "trailer_window",
        "author_email": "dev@example.com",
        "committed_at": "2026-09-01T10:00:31+00:00",
        "matched_at": "2026-09-01T10:20:00+00:00",
        "match_delta_s": None,
    }

    (spans_body,) = cloud.bodies(cs.SPANS_PATH)
    (resource,) = spans_body["resourceSpans"]
    res_attrs = {a["key"]: a["value"] for a in resource["resource"]["attributes"]}
    assert res_attrs == {
        "service.name": {"stringValue": "claude-code-widgets"},
        "tokenjam.install_id": {"stringValue": "inst-1"},
        "host.name": {"stringValue": "devbox"},
    }
    parsed = [parse_otlp_span(raw, res) for raw, res in iter_otlp_spans(spans_body)]
    assert len(parsed) == 3
    llm = next(p for p in parsed if p.model)
    assert (llm.agent_id, llm.provider, llm.model) == ("claude-code-widgets", "anthropic", "claude-haiku-4-5")
    assert (llm.input_tokens, llm.output_tokens, llm.cache_tokens, llm.cache_write_tokens) == (10, 5, 3, 2)
    assert llm.cost_usd == 0.25 and llm.session_id == "sess-0" and llm.billing_account == "anthropic"
    assert llm.attributes["tokenjam.plan_tier"] == "max_20x"
    ctx = llm.session_context
    assert ctx is not None
    assert (ctx.repo_remote, ctx.repo_root, ctx.branch_start, ctx.branch_end) == (
        "https://github.com/Acme/widgets", "/w/widgets", "main", "feat/x")
    assert (ctx.head_sha_start, ctx.developer_id, ctx.user_email) == (
        "a" * 40, "deadbeefcafe0000", "dev@example.com")
    assert llm.attributes["vcs.repository.name"] == "Acme/widgets"
    tool = next(p for p in parsed if p.tool_name)
    assert tool.tool_name == "Bash" and tool.session_id == "sess-0"


def test_content_is_stripped_by_default_even_when_captured_locally(db, state_file):
    _seed(db)
    config = _config()
    config.capture = CaptureConfig(prompts=True, completions=True, tool_inputs=True, tool_outputs=True)
    assert config.cloud.forward_content is False
    cloud = FakeCloud()
    _run(db, config, cloud, state_file)
    wire = b"".join(r.content for r in cloud.requests).decode()
    assert SECRET not in wire
    assert "git commit" not in wire
    assert "gen_ai.tool.input" not in wire and "llm.prompts" not in wire
    (spans_body,) = cloud.bodies(cs.SPANS_PATH)
    tool = next(p for p, _ in iter_otlp_spans(spans_body) if p["name"] == "gen_ai.tool.call")
    keys = {a["key"] for a in tool["attributes"]}
    assert "gen_ai.tool.name" in keys and "tokenjam.tool_arg_sig" in keys


def test_content_crosses_only_when_capture_and_forward_content_both_allow(db, state_file):
    _seed(db)
    cloud = FakeCloud()
    # forward_content on, but local capture off: still nothing.
    config = _config(forward_content=True)
    config.capture = CaptureConfig(prompts=False, completions=False, tool_inputs=False, tool_outputs=False)
    _run(db, config, cloud, state_file)
    assert SECRET not in b"".join(r.content for r in cloud.requests).decode()

    # Both on for prompts and tool inputs: the governed prompt and tool input
    # cross; a vendor key no toggle names (`llm.prompts`, `output.value`)
    # stays home while any toggle is off, since nothing can vouch for what
    # kind of content it holds.
    db.insert_span(make_llm_span(agent_id="claude-code-widgets", session_id="sess-0",
                                 start_time=T0 + timedelta(minutes=1),
                                 extra_attributes={"output.value": "VENDOR COMPLETION"}))
    cloud = FakeCloud()
    config = _config(forward_content=True)
    config.capture = CaptureConfig(prompts=True, completions=False, tool_inputs=True, tool_outputs=False)
    _run(db, config, cloud, state_file=state_file.with_name("s2.json"))
    wire = b"".join(r.content for r in cloud.requests).decode()
    assert SECRET in wire and "git commit" in wire
    assert "gen_ai.prompt.content" in wire and "llm.prompts" not in wire
    assert "VENDOR COMPLETION" not in wire

    # Every toggle on: the user has said everything may cross, and it does.
    cloud = FakeCloud()
    config.capture = CaptureConfig(prompts=True, completions=True, tool_inputs=True, tool_outputs=True)
    _run(db, config, cloud, state_file=state_file.with_name("s3.json"))
    wire = b"".join(r.content for r in cloud.requests).decode()
    assert "VENDOR COMPLETION" in wire and "llm.prompts" in wire


# --- Batching, resume, idempotency ----------------------------------------------------------

def test_batches_are_capped_and_a_second_pass_sends_nothing(db, state_file, monkeypatch):
    monkeypatch.setattr(cs, "BATCH_SIZE", 4)
    _seed(db, sessions=3, spans_per=3)  # 3 sessions, 3 commits, 12 spans
    config = _config()
    cloud = FakeCloud()
    report = _run(db, config, cloud, state_file)
    assert (report.sessions_sent, report.commits_sent, report.spans_sent) == (3, 3, 12)
    for body in cloud.bodies(cs.LEDGER_SESSIONS_PATH):
        assert len(body["sessions"]) <= 4 and len(body["session_commits"]) <= 4
    span_batches = [len(FakeCloud.spans(b)) for b in cloud.bodies(cs.SPANS_PATH)]
    assert span_batches == [4, 4, 4]
    # Sessions before commits before spans: a commit never precedes its session.
    paths = [r.url.path for r in cloud.requests]
    assert paths.index(cs.SPANS_PATH) > paths.index(cs.LEDGER_SESSIONS_PATH)

    state = cs.load_state(state_file)
    assert (state.spans_sent, state.sessions_sent, state.commits_sent) == (12, 3, 3)
    # The mark is ARRIVAL time (migration 25), not the span's own time.
    assert state.spans.hwm and state.spans.hwm > "2026-09-01T12:00:30+00:00"
    assert state.last_success_at and state.last_error is None

    again = FakeCloud()
    report = _run(db, config, again, state_file)
    assert not report.sent_anything and again.requests == []


def test_resume_picks_up_only_what_arrived_after_the_mark(db, state_file):
    _seed(db)
    config = _config()
    _run(db, config, FakeCloud(), state_file)
    # A new session, two new spans and a confidence upgrade land later.
    later = T0 + timedelta(days=1)
    db.upsert_session(replace(make_session(session_id="sess-new", agent_id="a", started_at=later), ended_at=later))
    db.insert_span(make_llm_span(agent_id="a", session_id="sess-new", start_time=later))
    db.insert_span(make_llm_span(agent_id="a", session_id="sess-new", start_time=later))
    db.upsert_session_commits([SessionCommit(
        session_id="sess-0", commit_sha="%040x" % 1, confidence="deterministic",
        source="tool_span_git_log", matched_at=later,
    )])
    cloud = FakeCloud()
    report = _run(db, config, cloud, state_file)
    assert (report.sessions_sent, report.commits_sent, report.spans_sent) == (1, 1, 2)
    sessions_body, commits_body = cloud.bodies(cs.LEDGER_SESSIONS_PATH)
    assert [s["session_id"] for s in sessions_body["sessions"]] == ["sess-new"]
    (commit,) = commits_body["session_commits"]
    assert commit["confidence"] == "deterministic" and commit["commit_sha"] == "%040x" % 1
    assert {p["spanId"] for p in FakeCloud.spans(cloud.bodies(cs.SPANS_PATH)[0])} == {
        s.span_id for s in db.get_recent_spans("sess-new", 10)
    }


def test_rows_that_arrive_late_with_old_timestamps_are_still_forwarded(db, state_file):
    """A backfill or the daemon's transcript catch-up inserts spans and
    sessions OLDER than everything already forwarded. The mark is arrival
    order, so they are picked up; an event-time mark would skip them forever."""
    _seed(db, at=T0 + timedelta(days=10))
    config = _config()
    _run(db, config, FakeCloud(), state_file)
    old = T0 - timedelta(days=30)
    db.upsert_session(replace(make_session(session_id="ancient", agent_id="a", started_at=old), ended_at=old))
    db.insert_span(make_llm_span(agent_id="a", session_id="ancient", start_time=old))
    cloud = FakeCloud()
    report = _run(db, config, cloud, state_file)
    assert (report.sessions_sent, report.spans_sent) == (1, 1)
    (body,) = cloud.bodies(cs.LEDGER_SESSIONS_PATH)
    assert body["sessions"][0]["session_id"] == "ancient"


def test_a_status_only_close_and_a_plan_stamp_are_re_sent(db, state_file):
    """Every session writer stamps `updated_at`, including the ones that
    move no timestamp: a stale-active sweep and the plan-tier stamp."""
    from tokenjam.core.config import ProviderBudget
    from tokenjam.core.framing import apply_declared_plans_to_sessions

    _seed(db)
    db.upsert_session(replace(make_session(session_id="zombie", agent_id="a", started_at=T0,
                                           status="active", plan_tier="unknown"), ended_at=T0))
    db.insert_span(make_llm_span(agent_id="a", session_id="zombie", start_time=T0))
    config = _config()
    _run(db, config, FakeCloud(), state_file)

    db.mark_sessions_completed(["zombie"])
    cloud = FakeCloud()
    assert _run(db, config, cloud, state_file).sessions_sent == 1
    (body,) = cloud.bodies(cs.LEDGER_SESSIONS_PATH)
    assert (body["sessions"][0]["session_id"], body["sessions"][0]["status"]) == ("zombie", "completed")

    config.budgets = {"anthropic": ProviderBudget(plan="max_5x")}
    apply_declared_plans_to_sessions(db.conn, config)
    cloud = FakeCloud()
    assert _run(db, config, cloud, state_file).sessions_sent == 1
    (body,) = cloud.bodies(cs.LEDGER_SESSIONS_PATH)
    assert body["sessions"][0]["plan_tier"] == "max_5x"
    assert body["sessions"][0]["pricing_mode"] == "subscription"

    db.increment_session_cost("sess-0", 0.5)
    cloud = FakeCloud()
    assert _run(db, config, cloud, state_file).sessions_sent == 1
    assert cloud.bodies(cs.LEDGER_SESSIONS_PATH)[0]["sessions"][0]["total_cost_usd"] == 1.5


def test_a_session_whose_totals_grew_is_re_sent(db, state_file):
    _seed(db)
    config = _config()
    _run(db, config, FakeCloud(), state_file)
    grown = replace(db.get_session("sess-0"), total_cost_usd=9.5, ended_at=T0 + timedelta(hours=3))
    db.upsert_session(grown)
    cloud = FakeCloud()
    report = _run(db, config, cloud, state_file)
    assert report.sessions_sent == 1 and report.spans_sent == 0
    (body,) = cloud.bodies(cs.LEDGER_SESSIONS_PATH)
    assert body["sessions"][0]["total_cost_usd"] == 9.5


def test_rows_sharing_the_boundary_timestamp_are_neither_skipped_nor_resent(db, state_file, monkeypatch):
    monkeypatch.setattr(cs, "BATCH_SIZE", 3)
    # Seven spans that all ARRIVED at the same instant (one bulk insert
    # shares one default evaluation), so every one sits on the boundary.
    db.bulk_insert_spans([
        make_llm_span(agent_id="a", session_id="s", start_time=T0, span_id=f"{i:016x}")
        for i in range(7)
    ])
    stamps = db.conn.execute("SELECT COUNT(DISTINCT ingested_at) FROM spans").fetchone()[0]
    assert stamps == 1
    config = _config()
    cloud = FakeCloud()
    report = _run(db, config, cloud, state_file)
    assert report.spans_sent == 7
    sent = [p["spanId"] for b in cloud.bodies(cs.SPANS_PATH) for p in FakeCloud.spans(b)]
    assert sorted(sent) == [f"{i:016x}" for i in range(7)] and len(sent) == 7
    again = FakeCloud()
    assert not _run(db, config, again, state_file).sent_anything and again.requests == []


def test_a_crash_between_send_and_ack_re_sends_at_most_one_batch(db, state_file, monkeypatch):
    """State is written after each acknowledged batch, so a process killed
    between the receiver's ack and the state write resends that one batch
    on the next run, which the receiver dedupes by primary key."""
    monkeypatch.setattr(cs, "BATCH_SIZE", 2)
    _seed(db, sessions=1, spans_per=5, commits=False)  # 6 spans: three batches
    config = _config()
    cloud = FakeCloud()
    real_save = cs.save_state
    dead = {"is": False}

    def _die_on_the_third_span_batch(state, path=None):
        # The third batch was acked and the mark advanced in memory; the
        # process dies before (and on every attempt at) writing that down.
        if state.spans_sent == 6:
            dead["is"] = True
        if dead["is"]:
            raise RuntimeError("killed")
        real_save(state, path)

    monkeypatch.setattr(cs, "save_state", _die_on_the_third_span_batch)
    with pytest.raises(RuntimeError):
        _run(db, config, cloud, state_file)
    monkeypatch.setattr(cs, "save_state", real_save)
    assert len(cloud.bodies(cs.SPANS_PATH)) == 3  # all three left the machine
    assert cs.load_state(state_file).spans_sent == 4  # two are on disk
    resumed = FakeCloud()
    report = _run(db, config, resumed, state_file)
    assert report.spans_sent == 2 and report.sessions_sent == 0
    (body,) = resumed.bodies(cs.SPANS_PATH)
    assert {p["spanId"] for p in FakeCloud.spans(body)} == {
        p["spanId"] for p in FakeCloud.spans(cloud.bodies(cs.SPANS_PATH)[2])
    }
    assert cs.load_state(state_file).spans_sent == 6


# --- Failure modes ---------------------------------------------------------------------------

def test_401_disables_the_bridge_with_a_reason_and_stops_every_later_pass(db, state_file):
    _seed(db)
    config = _config(ingest_key="tj_live_" + "z" * 43)
    cloud = FakeCloud()
    report = _run(db, config, cloud, state_file)
    assert report.stopped == cs.Outcome.UNAUTHORIZED and not report.sent_anything
    assert len(cloud.requests) == 1  # no retry on a dead key
    state = cs.load_state(state_file)
    assert state.disabled_reason and "401" in state.disabled_reason
    assert "tj init --cloud" in state.disabled_reason and state.disabled_at
    later = FakeCloud()
    report = _run(db, config, later, state_file)
    assert report.skipped_reason == state.disabled_reason and later.requests == []
    line = cs.status_line(config, state_file=state_file)
    assert line.startswith("Cloud: disabled:") and "401" in line
    # A fresh init clears it and keeps the marks for the same org.
    cs.save_state(replace(cs.load_state(state_file), spans_sent=7), state_file)
    cs.reset_state(org_id=ORG, path=state_file)
    cleared = cs.load_state(state_file)
    assert cleared.disabled_reason is None and cleared.spans_sent == 7
    assert cs.reset_state(org_id="org_other", path=state_file).spans_sent == 0


def test_5xx_and_network_failures_back_off_then_resume_next_pass(db, state_file):
    _seed(db)
    config = _config()
    cloud = FakeCloud(statuses=[503, 500, 502, 500])  # exhausts the three retries
    sleeps: list[float] = []
    report = cs.run_sync(db, config, client=_client(config, cloud, sleeps=sleeps),
                         state_file=state_file, install_id="i", host_name="h")
    assert report.stopped == cs.Outcome.UNAVAILABLE and not report.sent_anything
    assert sleeps == list(cs.RETRY_BACKOFF_S)
    assert len(cloud.requests) == 4
    state = cs.load_state(state_file)
    assert state.last_error and "will retry" in state.last_error
    assert state.sessions.hwm is None and state.disabled_reason is None
    assert cs.status_line(config, state_file=state_file).startswith("Cloud: unreachable")

    # Next pass: one 500 then healthy; everything drains.
    cloud = FakeCloud(statuses=[500])
    report = _run(db, config, cloud, state_file)
    assert report.stopped is None and (report.sessions_sent, report.commits_sent, report.spans_sent) == (1, 1, 3)
    assert cs.load_state(state_file).last_error is None
    assert cs.status_line(config, state_file=state_file).startswith("Cloud: connected · 3 spans, 1 sessions, 1 commits sent")

    def _boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    down = cs.CloudClient(config, transport=httpx.MockTransport(_boom), sleep=lambda _s: None)
    assert down.post(cs.SPANS_PATH, {"resourceSpans": []}).outcome == cs.Outcome.UNAVAILABLE


def test_a_refused_record_is_isolated_skipped_and_counted(db, state_file, monkeypatch):
    monkeypatch.setattr(cs, "BATCH_SIZE", 8)
    _seed(db, spans_per=5, commits=False)  # 6 spans in one batch
    bad = db.get_recent_spans("sess-0", 10)[2].span_id
    config = _config()
    cloud = FakeCloud(bad_span=bad)
    report = _run(db, config, cloud, state_file)
    assert report.spans_sent == 5 and report.rejected == 1 and report.stopped is None
    accepted = {p["spanId"] for b in cloud.accepted if "resourceSpans" in b for p in FakeCloud.spans(b)}
    assert bad not in accepted and len(accepted) == 5
    refused = [b for b in cloud.bodies(cs.SPANS_PATH) if bad in {p["spanId"] for p in FakeCloud.spans(b)}]
    assert len(refused[-1]["resourceSpans"][0]["scopeSpans"][0]["spans"]) == 1  # bisected down to the one row
    assert cs.load_state(state_file).rejected == 1
    assert "1 refused by the receiver" in cs.status_line(config, state_file=state_file)
    assert not _run(db, config, FakeCloud(), state_file).sent_anything


def test_a_different_org_or_store_resets_the_marks(db, state_file, tmp_path):
    _seed(db)
    _run(db, _config(), FakeCloud(), state_file)
    other = FakeCloud()
    report = _run(db, _config(org_id="org_other"), other, state_file)
    assert report.sent_anything and cs.load_state(state_file).org_id == "org_other"
    assert other.requests[0].headers["X-TokenJam-Org"] == "org_other"

    # Same org, a second database: its rows are older than the marks the
    # first one left, and they must still all be forwarded.
    second = DuckDBBackend(StorageConfig(path=str(tmp_path / "second.duckdb")))
    try:
        _seed(second, at=T0 - timedelta(days=5))
        config2 = _config(org_id="org_other")
        config2.storage = StorageConfig(path=str(tmp_path / "second.duckdb"))
        cloud2 = FakeCloud()
        report = _run(second, config2, cloud2, state_file)
        assert (report.sessions_sent, report.commits_sent, report.spans_sent) == (1, 1, 3)
        assert cs.load_state(state_file).db_path == str((tmp_path / "second.duckdb").resolve())
    finally:
        second.close()


def test_partial_rejections_inside_a_2xx_are_counted_not_folded_into_sent(db, state_file):
    _seed(db, spans_per=3, commits=False)  # 4 spans

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == cs.SPANS_PATH:
            return httpx.Response(200, json={
                "ingested": 3, "rejected": 1,
                "rejections": [{"span_id": "x", "reason": "span carries no startTimeUnixNano"}],
            })
        return httpx.Response(200, json={"ingested": 1, "rejected": 0})

    config = _config()
    client = cs.CloudClient(config, transport=httpx.MockTransport(_handler), sleep=lambda _s: None)
    report = cs.run_sync(db, config, client=client, state_file=state_file, install_id="i", host_name="h")
    assert (report.spans_sent, report.rejected, report.stopped) == (3, 1, None)
    state = cs.load_state(state_file)
    assert state.spans_sent == 3 and state.rejected == 1
    assert "refused 1 record" in state.last_error and "startTimeUnixNano" in state.last_error
    # Advanced past, not retried: the same body would be refused again.
    assert not cs.run_sync(db, config, client=client, state_file=state_file,
                           install_id="i", host_name="h").sent_anything


def test_passes_never_overlap(db, state_file):
    import threading

    _seed(db)
    config = _config()
    gate = threading.Event()
    inside = threading.Event()

    def _slow(request: httpx.Request) -> httpx.Response:
        inside.set()
        gate.wait(timeout=10)
        return httpx.Response(200, json={"ingested": 1, "rejected": 0})

    slow = cs.CloudClient(config, transport=httpx.MockTransport(_slow), sleep=lambda _s: None)
    first: list[cs.SyncReport] = []
    t = threading.Thread(target=lambda: first.append(cs.run_sync(
        db, config, client=slow, state_file=state_file, install_id="i", host_name="h")))
    t.start()
    assert inside.wait(timeout=10)
    second = _run(db, config, FakeCloud(), state_file)
    assert second.skipped_reason == "a forwarding pass is already running"
    gate.set()
    t.join(timeout=30)
    assert first and first[0].sent_anything


def test_sdk_event_timestamps_survive_the_wire(db, state_file):
    at = T0 + timedelta(minutes=1)
    span = make_llm_span(agent_id="a", session_id="s", start_time=at)
    span.events = [
        {"name": "sdk", "timestamp": at.isoformat(), "attributes": {}},
        {"name": "otlp", "time": str(int(at.timestamp() * 1e9) + 5), "attributes": {}},
        {"name": "none", "attributes": {}},
    ]
    db.insert_span(span)
    cloud = FakeCloud()
    _run(db, _config(), cloud, state_file)
    (raw,) = FakeCloud.spans(cloud.bodies(cs.SPANS_PATH)[0])
    times = {e["name"]: e["timeUnixNano"] for e in raw["events"]}
    assert times == {"sdk": str(int(at.timestamp() * 1e9)),
                     "otlp": str(int(at.timestamp() * 1e9) + 5), "none": "0"}


def test_a_corrupt_state_file_costs_one_resend_never_the_pass(db, state_file):
    _seed(db)
    state_file.write_text("{not json")
    report = _run(db, _config(), FakeCloud(), state_file)
    assert report.sent_anything and cs.load_state(state_file).spans_sent == 3


def test_start_sync_runs_on_its_own_backend_and_never_raises(tmp_path, state_file, monkeypatch):
    storage = StorageConfig(path=str(tmp_path / "d.duckdb"))
    seed = DuckDBBackend(storage)
    _seed(seed)
    seed.close()
    config = _config()
    cloud = FakeCloud()
    real_client = cs.CloudClient
    monkeypatch.setattr(cs, "CloudClient", lambda cfg, **kw: real_client(
        cfg, transport=cloud.transport, sleep=lambda _s: None))
    done: list[cs.SyncReport] = []
    thread = cs.start_sync(lambda: DuckDBBackend(storage), config, on_done=done.append,
                           state_file=state_file)
    assert thread is not None
    thread.join(timeout=30)
    assert done and done[0].spans_sent == 3

    def _broken_factory():
        raise RuntimeError("no db")

    thread = cs.start_sync(_broken_factory, config, state_file=state_file)
    thread.join(timeout=30)
    assert "no db" in (cs.load_state(state_file).last_error or "")


# --- Surfaces ---------------------------------------------------------------------------------

def test_status_line_and_summary_cover_every_state(state_file):
    assert cs.status_line(TjConfig(version="1"), state_file=state_file) is None
    assert cs.status_summary(TjConfig(version="1"), state_file=state_file)["state"] == "not_configured"
    config = _config()
    assert cs.status_line(config, state_file=state_file) == (
        "Cloud: configured, first pass pending · 0 spans, 0 sessions, 0 commits sent")
    assert cs.status_line(_config(enabled=False), state_file=state_file).startswith("Cloud: off")
    state = cs.SyncState(spans_sent=15, sessions_sent=3, commits_sent=3,
                         last_success_at=(utcnow() - timedelta(minutes=2)).isoformat())
    cs.save_state(state, state_file)
    assert cs.status_line(config, state_file=state_file) == (
        "Cloud: connected · 15 spans, 3 sessions, 3 commits sent · last 2m ago")
    summary = cs.status_summary(config, state_file=state_file)
    assert summary["state"] == "connected" and summary["org_id"] == ORG
    assert summary["forward_content"] is False and "ingest_key" not in summary


def test_probe_sends_an_empty_body_and_classifies_the_answer():
    seen: list[dict] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        if request.headers["Authorization"] != f"Bearer {KEY}":
            return httpx.Response(401, json={"detail": "nope"})
        return httpx.Response(200, json={"ingested": 0, "rejected": 0})

    ok = cs.probe(_config(), transport=httpx.MockTransport(_handler))
    assert ok.outcome == cs.Outcome.OK and seen == [{"resourceSpans": []}]
    bad = cs.probe(_config(ingest_key="tj_live_" + "q" * 43), transport=httpx.MockTransport(_handler))
    assert bad.outcome == cs.Outcome.UNAUTHORIZED and bad.detail == "nope"


def test_emission_list_is_the_contract_verbatim():
    assert cs.EMISSION_LEAVES == (
        "token counts", "model names", "cost", "timestamps", "tool names",
        "file paths touched", "session / repo / branch / commit identifiers",
        "hashed developer id", "git author email",
    )
    assert cs.EMISSION_NEVER_BY_DEFAULT == (
        "prompt text", "completions", "tool outputs", "file contents", "diffs", "secrets",
    )
