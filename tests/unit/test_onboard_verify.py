"""First-signal onboarding verification (#80): the poll primitive, per-persona
cause strings, and the lock-aware read-backend resolver."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from tokenjam.core.onboard_verify import (
    VerifyResult,
    not_confirmed_cause,
    open_read_backend,
    poll_for_first_span,
)


class _FakeBackend:
    """Returns empty until ``arrive_after`` calls, then one trace."""

    def __init__(self, arrive_after: int = 0, trace_id: str = "trace-1"):
        self.arrive_after = arrive_after
        self.trace_id = trace_id
        self.calls = 0
        self.seen_filters: list = []

    def get_traces(self, filters):
        self.calls += 1
        self.seen_filters.append(filters)
        if self.calls > self.arrive_after:
            return [SimpleNamespace(trace_id=self.trace_id)]
        return []


# --- poll_for_first_span ----------------------------------------------------


def test_confirms_when_a_span_arrives_immediately():
    backend = _FakeBackend(arrive_after=0)
    result = poll_for_first_span(
        backend, since=None, sleep=lambda _s: None, monotonic=lambda: 0.0,
    )
    assert isinstance(result, VerifyResult)
    assert result.confirmed is True
    assert result.first_trace_id == "trace-1"
    assert result.error is None


def test_confirms_after_a_few_empty_polls():
    backend = _FakeBackend(arrive_after=2)
    ticks = iter([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    result = poll_for_first_span(
        backend, since=None, interval_s=1.0,
        sleep=lambda _s: None, monotonic=lambda: next(ticks),
    )
    assert result.confirmed is True
    assert backend.calls == 3  # two empties + one hit


def test_times_out_when_no_span_arrives():
    backend = _FakeBackend(arrive_after=99)  # never arrives
    # start, the >=timeout check, then the elapsed_s read on the timeout return.
    ticks = iter([0.0, 100.0, 100.0])
    result = poll_for_first_span(
        backend, since=None, timeout_s=60.0,
        sleep=lambda _s: None, monotonic=lambda: next(ticks),
    )
    assert result.confirmed is False
    assert result.error is None  # a clean timeout is not an error


def test_read_error_is_surfaced_not_raised():
    class _Boom:
        def get_traces(self, filters):
            raise RuntimeError("db went away")

    result = poll_for_first_span(
        _Boom(), since=None, sleep=lambda _s: None, monotonic=lambda: 0.0,
    )
    assert result.confirmed is False
    assert "db went away" in (result.error or "")


def test_agent_id_is_passed_through_to_the_filter():
    backend = _FakeBackend(arrive_after=0)
    poll_for_first_span(
        backend, since=None, agent_id="my-agent",
        sleep=lambda _s: None, monotonic=lambda: 0.0,
    )
    assert backend.seen_filters[0].agent_id == "my-agent"


# --- not_confirmed_cause ----------------------------------------------------


def test_cause_is_persona_specific():
    assert "restart" in not_confirmed_cause("claude_code").lower()
    assert "Claude Code" in not_confirmed_cause("claude_code")
    assert "Codex" in not_confirmed_cause("codex")
    assert "tj ping" in not_confirmed_cause("sdk")
    # Unknown persona covers both failure modes.
    generic = not_confirmed_cause("???")
    assert "Claude Code" in generic and "SDK" in generic


# --- open_read_backend ------------------------------------------------------


def _config(host="127.0.0.1", port=7391, auth_enabled=False):
    return SimpleNamespace(
        api=SimpleNamespace(
            host=host, port=port,
            auth=SimpleNamespace(enabled=auth_enabled, api_key="k"),
        ),
        storage=SimpleNamespace(path="/tmp/does-not-matter.db"),
    )


def test_prefers_the_running_daemon_over_the_db_lock(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        "tokenjam.core.api_backend.probe_api",
        lambda host, port, api_key: sentinel,
    )
    backend, mode, error = open_read_backend(_config())
    assert backend is sentinel
    assert mode == "api"
    assert error is None


def test_falls_back_to_direct_when_no_daemon(monkeypatch):
    monkeypatch.setattr(
        "tokenjam.core.api_backend.probe_api",
        lambda host, port, api_key: None,
    )
    sentinel = object()
    monkeypatch.setattr("tokenjam.core.db.open_db", lambda storage: sentinel)
    backend, mode, error = open_read_backend(_config())
    assert backend is sentinel
    assert mode == "direct"


def test_reports_error_when_db_locked_and_no_daemon(monkeypatch):
    monkeypatch.setattr(
        "tokenjam.core.api_backend.probe_api",
        lambda host, port, api_key: None,
    )

    def _locked(storage):
        raise RuntimeError("Could not set lock on file: Conflicting lock held")

    monkeypatch.setattr("tokenjam.core.db.open_db", _locked)
    backend, mode, error = open_read_backend(_config())
    assert backend is None
    assert error is not None and "locked" in error.lower()
