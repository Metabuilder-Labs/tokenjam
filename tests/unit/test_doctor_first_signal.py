"""Doctor's onboarded-but-silent diagnosis (#80): the after-the-fact check that
turns a blank Status page into an actionable per-persona cause."""
from __future__ import annotations

from types import SimpleNamespace

from tokenjam.cli.cmd_doctor import (
    _check_onboarding_first_signal,
    _detect_onboarded_persona,
)
from tokenjam.core.db import InMemoryBackend
from tests.factories import make_llm_span


def _config(agent_ids=()):
    agents = {aid: SimpleNamespace() for aid in agent_ids}
    return SimpleNamespace(agents=agents)


# --- persona detection ------------------------------------------------------


def test_detects_claude_code_from_agent_id():
    assert _detect_onboarded_persona(_config(["claude-code-myproj"])) == "claude_code"


def test_detects_codex_from_agent_id():
    assert _detect_onboarded_persona(_config(["codex_exec"])) == "codex"


def test_defaults_to_sdk_when_no_runtime_marker(monkeypatch):
    monkeypatch.setattr("tokenjam.cli.cmd_doctor._tj_statusline_wired", lambda: False)
    assert _detect_onboarded_persona(_config([])) == "sdk"
    assert _detect_onboarded_persona(_config(["my-agent"])) == "sdk"


def test_detects_claude_code_from_statusline_when_no_agents(monkeypatch):
    monkeypatch.setattr("tokenjam.cli.cmd_doctor._tj_statusline_wired", lambda: True)
    assert _detect_onboarded_persona(_config([])) == "claude_code"


# --- the check itself -------------------------------------------------------


def test_flags_info_when_onboarded_but_zero_spans(monkeypatch):
    # Info, not warning: doctor has no onboarding timestamp, so it can't tell a
    # fresh setup from a silent one — a warning would false-alarm and flip exit 1.
    monkeypatch.setattr("tokenjam.cli.cmd_doctor._tj_statusline_wired", lambda: False)
    db = InMemoryBackend()
    check = _check_onboarding_first_signal(_config(["claude-code-myproj"]), db)
    assert check["level"] == "info"
    assert "no spans" in check["message"].lower()
    assert "Claude Code" in check["message"]  # per-persona cause


def test_ok_once_a_live_span_exists():
    db = InMemoryBackend()
    db.insert_span(make_llm_span(agent_id="my-agent"))
    check = _check_onboarding_first_signal(_config(["my-agent"]), db)
    assert check["level"] == "ok"
    assert "flowing" in check["message"].lower()
    assert "live" in check["message"].lower()


def test_backfill_only_is_not_reported_as_flowing(monkeypatch):
    # The #102 contradiction: a DB with only backfilled spans must NOT read
    # "telemetry is flowing" (which contradicts `--verify` polling for a live
    # span). It reports the honest backfill-vs-live state instead.
    monkeypatch.setattr("tokenjam.cli.cmd_doctor._tj_statusline_wired", lambda: False)
    db = InMemoryBackend()
    db.insert_span(make_llm_span(
        agent_id="claude-code-myproj",
        extra_attributes={"source": "backfill.claude_code"},
    ))
    check = _check_onboarding_first_signal(_config(["claude-code-myproj"]), db)
    assert check["level"] == "info"
    assert "flowing" not in check["message"].lower()
    assert "backfilled" in check["message"].lower()
    assert "live" in check["message"].lower()


def test_live_span_counts_as_flowing_even_alongside_backfill():
    db = InMemoryBackend()
    db.insert_span(make_llm_span(
        agent_id="my-agent",
        extra_attributes={"source": "backfill.claude_code"},
    ))
    db.insert_span(make_llm_span(agent_id="my-agent"))  # one genuinely live span
    check = _check_onboarding_first_signal(_config(["my-agent"]), db)
    assert check["level"] == "ok"
    assert "flowing" in check["message"].lower()
    assert "1 live" in check["message"]


def test_sdk_persona_cause_points_at_ping(monkeypatch):
    monkeypatch.setattr("tokenjam.cli.cmd_doctor._tj_statusline_wired", lambda: False)
    db = InMemoryBackend()
    check = _check_onboarding_first_signal(_config([]), db)
    assert check["level"] == "info"
    assert "tj ping" in check["message"]


def test_skips_in_api_fallback_mode():
    # A backend without a `.conn` (the ApiBackend HTTP fallback) can't run the
    # raw count query — the check must degrade to info, never error.
    fake_db = SimpleNamespace()  # no `conn` attribute
    check = _check_onboarding_first_signal(_config(["my-agent"]), fake_db)
    assert check["level"] == "info"
    assert "fallback" in check["message"].lower()
