"""The default `tj status` screen — the capped agent overview.

`tj status` printed one ~9-line card per tracked agent_id with no cap, no
pointer at the rest, and no cross-agent totals until the final line. The
agent_id set grows monotonically (roughly one per project directory tj has
ever seen), so the screen got longer the longer you used tj.

These pin the overview's own rails: totals first, rows ranked by recency
(live sessions above everything), and no cell that asserts more than the data
supports — no bare `0` where "none" is meant, no invented timestamp for an
agent that has never run.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from tokenjam.cli.cmd_status import _elide_left, _fmt_age, _print_overview
from tokenjam.utils.time_parse import utcnow

UTC = timezone.utc


def _flat(out: str) -> str:
    return " ".join(out.split())


class _Session:
    def __init__(self, started_at, ended_at=None):
        self.started_at = started_at
        self.ended_at = ended_at


def _entry(agent_id, *, status="completed", cost=0.0, alerts=0, age_hours=1.0):
    data = {
        "agent_id": agent_id,
        "status": status,
        "session_id": None,
        "cost_today": cost,
        "daily_limit": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "tool_call_count": 0,
        "error_count": 0,
        "active_alerts": alerts,
        "duration_seconds": None,
        "active_seconds": None,
    }
    session = _Session(utcnow() - timedelta(hours=age_hours))
    return (data, [], session)


def test_totals_render_before_any_row(capsys):
    entries = [_entry(f"agent-{i}", cost=1.0) for i in range(3)]
    _print_overview(entries)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]

    assert "3 agents" in _flat(lines[0])
    assert "$3.00 today" in _flat(lines[0])
    assert all("AGENT" not in ln for ln in lines[:1])


def test_rows_rank_by_recency_with_live_sessions_first(capsys):
    entries = [
        _entry("stale", age_hours=100),
        _entry("recent", age_hours=1),
        _entry("live", status="active", age_hours=50),
    ]
    _print_overview(entries)
    out = capsys.readouterr().out
    order = [ln.split()[0] for ln in out.splitlines()
             if ln.split() and ln.split()[0] in ("stale", "recent", "live")]

    assert order == ["live", "recent", "stale"]


def test_no_alerts_renders_the_null_marker_not_a_zero(capsys):
    _print_overview([_entry("quiet", alerts=0)])
    row = next(ln for ln in capsys.readouterr().out.splitlines() if "quiet" in ln)

    assert " 0 " not in row
    assert "-" in row


def test_alert_count_is_called_out_above_the_table(capsys):
    entries = [_entry("noisy", alerts=3), _entry("quiet")]
    _print_overview(entries)
    out = _flat(capsys.readouterr().out)

    assert "1 agent with active alerts" in out


def test_agent_with_no_session_shows_unknown_not_a_fabricated_time(capsys):
    data, alerts, _session = _entry("never-ran")
    _print_overview([(data, alerts, None)])
    row = next(ln for ln in capsys.readouterr().out.splitlines() if "never-ran" in ln)

    assert "ago" not in row


def test_fmt_age_marks_unknown_rather_than_zero():
    assert _fmt_age(None) == "-"
    assert _fmt_age(utcnow() - timedelta(seconds=5)) == "just now"
    assert _fmt_age(utcnow() - timedelta(minutes=5)) == "5m ago"
    assert _fmt_age(utcnow() - timedelta(hours=5)) == "5h ago"
    assert _fmt_age(utcnow() - timedelta(days=5)) == "5d ago"


def test_fmt_age_prefers_the_end_of_a_finished_session():
    """A resumed session can have started days before it ended; the age the
    user means is when the agent last did something."""
    started = datetime(2026, 5, 1, tzinfo=UTC)
    ended = utcnow() - timedelta(hours=2)
    from tokenjam.cli.cmd_status import _last_activity

    assert _last_activity(_Session(started, ended)) == ended


def test_agent_ids_elide_from_the_left_keeping_the_distinguishing_tail():
    """Agent ids share a long leading `claude-code-` run, so right-truncation
    renders a whole screen of identical rows."""
    ids = ["claude-code-tokenjam-website", "claude-code-tokenjam-harness"]
    short = [_elide_left(i, 20) for i in ids]

    assert short[0] != short[1]
    assert all(s.startswith("…") for s in short)
    assert _elide_left("short-id", 20) == "short-id"
