"""Track A summarize analyzer — filesystem-derived recoverable finding.

The analyzer reasons over the summarize scan (filesystem), not telemetry, so the
scan is monkeypatched to a controlled ScanResult. Asserts the #111 recoverable
contract: `past_overspend_tokens` and `past_overspend_usd` are on
the SAME window-priced basis — both None when no loading session is
observed, both populated together when one is — while the one-time per-call
aggregate lives in `file_reduction_tokens`. Also checks an explicit basis and a
clean report_to_dict/report_from_dict round-trip.
"""
from __future__ import annotations

import logging
from datetime import timedelta

import pytest

from tokenjam.core.config import TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize import build_report
from tokenjam.core.optimize.runner import report_from_dict, report_to_dict
from tokenjam.core.summarize.candidates import Candidate, ScanResult
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_session


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def _window():
    return utcnow() - timedelta(days=30), utcnow() + timedelta(hours=1)


def _cand(path: str, saved: int, *, is_prompt: bool = True, scope: str = "repo",
          load_class: str = "always", resident: int | None = None,
          on_demand: int = 0) -> Candidate:
    """A scan candidate with its load-semantics split filled in.

    ``resident``/``on_demand`` are the two halves of ``saved`` the real scan
    measures on the file's text (see `core/summarize/load_semantics`); by
    default the whole reduction is always-resident, which is what a
    `CLAUDE.md` looks like.
    """
    from tokenjam.core.summarize.load_semantics import invocation_key

    return Candidate(
        path=path, prose_words=saved * 3, total_chars=saved * 12,
        protected_blocks=0, est_tokens_saved=saved, pricing_mode="api",
        scope=scope, is_prompt=is_prompt,
        load_class=load_class,
        invocation_key=invocation_key(path, load_class),
        always_resident_tokens_saved=saved if resident is None else resident,
        on_demand_tokens_saved=on_demand,
        always_resident_chars=(saved if resident is None else resident) * 12,
    )


def _skill_cand(path: str, *, resident: int, on_demand: int,
                scope: str = "global") -> Candidate:
    """A `.claude/skills/<slug>/SKILL.md`: frontmatter always resident, body
    delivered only when the skill is invoked."""
    return _cand(path, resident + on_demand, scope=scope, load_class="skill",
                 resident=resident, on_demand=on_demand)


def _patch_scan(monkeypatch, cands: list[Candidate]) -> None:
    result = ScanResult(candidates=cands, root=".", recursive=False,
                        globals_checked=0, walk_capped=False, note="")
    monkeypatch.setattr(
        "tokenjam.core.summarize.candidates.list_candidates",
        lambda **kw: result,
    )


#: The window must classify as `claude-code`, or the persona gate true-skips
#: `summarize` before dispatch and every test here reads an absent finding.
#: That is not test scaffolding: `summarize` scans AGENT INSTRUCTION FILES
#: (`core/summarize/agent_files.toml`), which an SDK window does not have, so a
#: coding-agent window is the only shape in which this analyzer runs at all.
#: `tests.factories`' default `agent_id="test-agent"` classifies as `sdk`.
_CC_AGENT_ID = "claude-code-proj"


def _seed_window(db) -> None:
    """One qualifying LLM call in the window so the analyzer runs — it
    window-guards on a dead window (no telemetry → no per-call saving to attach).
    Content is irrelevant beyond the agent id; the finding is filesystem-derived."""
    db.upsert_session(make_session(agent_id=_CC_AGENT_ID, session_id="s0"))
    db.insert_span(make_llm_span(agent_id=_CC_AGENT_ID, session_id="s0",
                                 start_time=utcnow() - timedelta(days=1)))


def _run(db) -> object:
    _seed_window(db)
    since, until = _window()
    report = build_report(db=db, config=TjConfig(version="1"),
                          since=since, until=until, findings=["summarize"])
    return report.findings["summarize"]


def test_sums_per_call_tokens_drops_zero_saving(db, monkeypatch):
    _patch_scan(monkeypatch, [
        _cand("./CLAUDE.md", 410),
        _cand("./AGENTS.md", 300),
        _cand("./docs/x.md", 0, is_prompt=False),   # no saving → dropped
    ])
    f = _run(db)
    assert f.files == 2
    # One-time, per-call aggregate — always available regardless of session
    # evidence; feeds the curate/diff UI and reduction_pct.
    assert f.file_reduction_tokens == 710
    # No session in this window is observed loading a repo-scoped "./CLAUDE.md"
    # (its ancestor names match no agent_id-derived repo), so NEITHER window
    # figure is attached — tokens and dollars are on the same basis:
    # never a zero, never a rate borrowed from a file that did resolve.
    assert f.past_overspend_tokens is None
    assert f.past_overspend_usd is None
    assert f.estimate_confidence == "heuristic"
    assert f.estimate_basis                          # explicit basis required by contract
    assert {c.path for c in f.candidates} == {"./CLAUDE.md", "./AGENTS.md"}
    # Mandatory honesty caveat carried as the field default (Rule 14).
    assert "meaning may change" in f.caveat
    # Prose reduction %s computed server-side (_cand sets total_chars = saved*12,
    # so source tokens = saved*3 → every file reduces ~33%). Per-file + aggregate.
    assert all(c.reduction_pct == 33 for c in f.candidates)
    assert f.reduction_pct == 33
    assert f.avg_reduction_pct == 33


def test_dead_window_contributes_nothing(db, monkeypatch):
    # No telemetry in the window → no per-call saving to attach; the analyzer must
    # NOT scan the filesystem and must emit no recoverable figure (#211 invariant).
    def must_not_run(**kw):
        raise AssertionError("summarize scan ran on a dead telemetry window")
    monkeypatch.setattr("tokenjam.core.summarize.candidates.list_candidates", must_not_run)
    since, until = _window()   # empty db → total_tokens == 0
    report = build_report(db=db, config=TjConfig(version="1"),
                          since=since, until=until, findings=["summarize"])
    f = report.findings["summarize"]
    assert f.files == 0
    assert f.past_overspend_tokens is None
    assert f.reduction_pct is None


def test_empty_scan_yields_no_tokens_but_keeps_basis(db, monkeypatch):
    _patch_scan(monkeypatch, [])
    f = _run(db)
    assert f.files == 0
    assert f.past_overspend_tokens is None
    assert f.past_overspend_usd is None
    assert f.estimate_basis


def test_scan_error_never_breaks_the_report(db, monkeypatch, caplog):
    def boom(**kw):
        raise OSError("disk gone")
    monkeypatch.setattr("tokenjam.core.summarize.candidates.list_candidates", boom)
    with caplog.at_level(logging.DEBUG, logger="tokenjam.core.optimize.analyzers.summarize"):
        f = _run(db)
    assert f.files == 0 and f.past_overspend_tokens is None
    # the swallow must leave a trail (not silent) so a real regression is diagnosable
    assert any("scan failed" in r.message for r in caplog.records)


# --- CLI text-view rendering regression --------------------------------------
# Same class of defect deadweight/relearn hit before: `summarize` was registered
# in ANALYZER_REGISTRY and ran on every report, but had no entry in cmd_optimize's
# _FINDING_RENDERERS dispatch table, so `_rank_findings` silently dropped it and
# plain-text `tj optimize` never showed it -- only `--json` did.

def test_summarize_in_click_choices_and_renderer():
    from tokenjam.cli.cmd_optimize import (
        _FINDING_RENDERERS,
        _MINOR_FINDING_LABELS,
        cmd_optimize,
    )

    findings_param = next(
        p for p in cmd_optimize.params if getattr(p, "name", None) == "findings"
    )
    assert "summarize" in findings_param.type.choices
    assert "summarize" in _FINDING_RENDERERS
    assert "summarize" in _MINOR_FINDING_LABELS


def test_render_summarize_lists_candidates_and_points_to_summarize_list(db, monkeypatch, capsys):
    """The finding renders through the CLI dispatch path, names the files, the
    per-call token saving, and points at `tj summarize`.

    These candidates resolve to no loading session, so there is no dollar
    figure to show and none is fabricated. The inverse — that a resolvable file
    DOES surface its dollars — is pinned by
    `test_render_summarize_shows_the_window_dollar_figure_when_priced`.
    """
    from tokenjam.cli.cmd_optimize import _render_summarize

    _patch_scan(monkeypatch, [_cand("./CLAUDE.md", 410), _cand("./AGENTS.md", 300)])
    finding = _run(db)
    assert finding.past_overspend_usd is None

    for mode in ("api", "subscription", "local", "unknown"):
        _render_summarize(finding, pricing_mode=mode, marker="①")
    out = capsys.readouterr().out

    assert "CLAUDE.md" in out
    assert "AGENTS.md" in out
    assert "710" in out or "710" in out.replace(",", "")  # aggregate tokens saved
    assert "tj summarize list" in out
    assert "tj summarize prep" in out
    assert "$" not in out  # no fabricated dollar figure


# --- Always-on files save REPEATEDLY, so the figure is priced per window ------

def test_global_scope_file_is_priced_across_every_session_in_the_window(
    db, monkeypatch,
):
    """A `~/.claude` file is loaded by every session and re-read on every call
    within one, so its reduction is worth reduction x sessions x (input rate +
    later calls at the cache-read rate) — not a one-time figure."""
    from tokenjam.core.pricing import get_rates

    db.upsert_session(make_session(agent_id=_CC_AGENT_ID, session_id="s1"))
    db.upsert_session(make_session(agent_id=_CC_AGENT_ID, session_id="s2"))
    for sid in ("s1", "s2"):
        for _ in range(3):
            db.insert_span(make_llm_span(
                session_id=sid, provider="anthropic", model="claude-haiku-4-5",
                input_tokens=100, output_tokens=10,
                start_time=utcnow() - timedelta(days=1),
            ))
    _patch_scan(monkeypatch, [_cand("~/.claude/CLAUDE.md", 1_000, scope="global")])

    since, until = _window()
    report = build_report(db=db, config=TjConfig(version="1"),
                          since=since, until=until, findings=["summarize"])
    f = report.findings["summarize"]

    rates = get_rates("anthropic", "claude-haiku-4-5")
    # 2 sessions x 3 calls each: one send at the input rate, two re-reads.
    expected = (
        1_000
        * (rates.input_per_mtok / 1_000_000)
        * (1 + (rates.cache_read_per_mtok / rates.input_per_mtok) * 2)
        * 2
    )
    assert f.sessions_examined == 2
    assert f.calls_per_session == 3.0
    assert f.candidates[0].sessions_loading == 2
    assert f.past_overspend_usd == pytest.approx(round(expected, 6))
    assert f.rate_basis
    # Strictly more than a single send: the saving recurs, it is not one-time.
    assert f.past_overspend_usd > 1_000 * rates.input_per_mtok / 1_000_000


def test_token_and_dollar_aggregates_describe_the_same_quantity(db, monkeypatch):
    """Basis-coherence guard: dividing the dollar aggregate by the token
    aggregate must land on the blended per-token rate the basis advertises.

    Before the fix `past_overspend_tokens` was the un-multiplied one-time
    file reduction while `past_overspend_usd` was per-call-multiplied, so
    the implied rate came out orders of magnitude above any real price — a card
    whose own tokens and dollars described different quantities. Both fields are
    now the same event count (reduction x reads x loading sessions), one counted
    and one priced.
    """
    from tokenjam.core.pricing import get_rates

    for sid in ("s1", "s2"):
        db.upsert_session(make_session(agent_id=_CC_AGENT_ID, session_id=sid))
        for _ in range(3):
            db.insert_span(make_llm_span(
                session_id=sid, provider="anthropic", model="claude-haiku-4-5",
                input_tokens=100, output_tokens=10,
                start_time=utcnow() - timedelta(days=1),
            ))
    _patch_scan(monkeypatch, [_cand("~/.claude/CLAUDE.md", 1_000, scope="global")])

    since, until = _window()
    f = build_report(db=db, config=TjConfig(version="1"),
                     since=since, until=until, findings=["summarize"]).findings["summarize"]

    # 1,000 tokens x 3 reads per session (one send + two re-reads) x 2 sessions.
    assert f.past_overspend_tokens == 6_000
    # The one-time per-call reduction keeps its own field and its own basis.
    assert f.file_reduction_tokens == 1_000

    rates = get_rates("anthropic", "claude-haiku-4-5")
    input_per_token = rates.input_per_mtok / 1_000_000
    cache_read_per_token = rates.cache_read_per_mtok / 1_000_000
    implied = f.past_overspend_usd / f.past_overspend_tokens
    # One send at the input rate + two re-reads at the cache-read rate, over 3
    # token-events: the exact blend the basis string states.
    assert implied == pytest.approx((input_per_token + 2 * cache_read_per_token) / 3)
    # And it stays inside the real price band — the property that fails loudly if
    # either field ever drifts back onto its own basis.
    assert cache_read_per_token <= implied <= input_per_token


def test_project_scope_file_is_priced_against_its_own_repos_calls_per_session(
    db, monkeypatch,
):
    """A project-scope candidate must price against its OWN repo's average
    calls-per-session -- never the window-wide blend across every other
    repo/agent in the window, which can differ sharply from this repo's own
    session behavior.

    Repo A: one session, 5 calls (a heavy call-count repo). Repo B: four
    sessions, one call each (a light one). The window-wide average
    (9 calls / 5 sessions = 1.8, rounds to 2) is nowhere near repo A's own
    average (5.0) -- pricing repo A's file against the blend would understate
    its recoverable figure.
    """
    from tokenjam.core.pricing import get_rates

    db.upsert_session(make_session(session_id="a1", agent_id="claude-code-repo-a"))
    for i in range(5):
        db.insert_span(make_llm_span(
            session_id="a1", agent_id="claude-code-repo-a",
            provider="anthropic", model="claude-haiku-4-5",
            input_tokens=100, output_tokens=10,
            start_time=utcnow() - timedelta(days=1, minutes=i),
        ))
    for i in range(4):
        sid = f"b{i}"
        db.upsert_session(make_session(session_id=sid, agent_id="claude-code-repo-b"))
        db.insert_span(make_llm_span(
            session_id=sid, agent_id="claude-code-repo-b",
            provider="anthropic", model="claude-haiku-4-5",
            input_tokens=100, output_tokens=10,
            start_time=utcnow() - timedelta(days=1),
        ))

    _patch_scan(monkeypatch, [_cand("repo-a/CLAUDE.md", 1_000, scope="repo")])
    since, until = _window()
    report = build_report(db=db, config=TjConfig(version="1"),
                          since=since, until=until, findings=["summarize"])

    f = report.findings["summarize"]
    candidate = f.candidates[0]
    rates = get_rates("anthropic", "claude-haiku-4-5")
    ratio = rates.cache_read_per_mtok / rates.input_per_mtok

    assert candidate.sessions_loading == 1  # only repo-a's own session matches
    # Repo A's own average is 5 calls/session -> 4 rereads, never the
    # window-wide blend's rounded 2 calls/session -> 1 reread.
    expected = 1_000 * (rates.input_per_mtok / 1_000_000) * (1 + ratio * 4)
    wrong_window_blend = 1_000 * (rates.input_per_mtok / 1_000_000) * (1 + ratio * 1)
    assert candidate.est_usd_saved == pytest.approx(round(expected, 6))
    assert candidate.est_usd_saved != pytest.approx(round(wrong_window_blend, 6))
    assert candidate.est_usd_saved > wrong_window_blend


def test_render_summarize_shows_the_window_dollar_figure_when_priced(
    db, monkeypatch, capsys,
):
    from tokenjam.cli.cmd_optimize import _render_summarize

    db.upsert_session(make_session(agent_id=_CC_AGENT_ID, session_id="s1"))
    db.insert_span(make_llm_span(
        session_id="s1", provider="anthropic", model="claude-haiku-4-5",
        input_tokens=100, output_tokens=10, start_time=utcnow() - timedelta(days=1),
    ))
    _patch_scan(monkeypatch, [_cand("~/.claude/CLAUDE.md", 5_000_000, scope="global")])
    since, until = _window()
    report = build_report(db=db, config=TjConfig(version="1"),
                          since=since, until=until, findings=["summarize"])
    finding = report.findings["summarize"]
    assert finding.past_overspend_usd

    _render_summarize(finding, pricing_mode="api", marker="①")
    out = capsys.readouterr().out
    assert "$" in out
    # Rich soft-wraps, so match on a fragment that survives a line break.
    assert "session(s) in this" in out


def test_render_summarize_empty_state_names_the_reason(db, monkeypatch, capsys):
    from tokenjam.cli.cmd_optimize import _render_summarize

    _patch_scan(monkeypatch, [])
    finding = _run(db)

    _render_summarize(finding, pricing_mode="api", marker="①")
    out = capsys.readouterr().out

    assert "No catalog prompt files" in out
    assert "tj summarize list" not in out  # no remedy pointer on an empty finding


def test_render_report_surfaces_summarize_instead_of_generic_empty(db, monkeypatch, capsys):
    """End-to-end: a report whose only finding is a populated summarize set
    must not fall through to the generic "No candidates flagged" empty state."""
    from tokenjam.cli.cmd_optimize import _render_report

    _seed_window(db)
    _patch_scan(monkeypatch, [_cand("./CLAUDE.md", 410)])
    since, until = _window()
    report = build_report(db=db, config=TjConfig(version="1"),
                          since=since, until=until, findings=["summarize"])
    assert report.findings["summarize"].candidates  # sanity

    _render_report(report, agent=None, requested=["summarize"], pricing_mode="local")
    out = capsys.readouterr().out

    assert "No candidates flagged" not in out
    assert "CLAUDE.md" in out


def test_finding_round_trips(db, monkeypatch):
    _seed_window(db)
    _patch_scan(monkeypatch, [_cand("./CLAUDE.md", 410)])
    since, until = _window()
    report = build_report(db=db, config=TjConfig(version="1"),
                          since=since, until=until, findings=["summarize"])
    payload = report_to_dict(report)
    sd = payload["findings"]["summarize"]
    # No session in this window loads a repo-scoped "./CLAUDE.md", so neither
    # window figure resolves; the one-time aggregate still does.
    assert sd["file_reduction_tokens"] == 410
    assert sd["past_overspend_tokens"] is None
    assert sd["past_overspend_usd"] is None
    assert "meaning may change" in sd["caveat"]       # caveat survives serialization
    assert sd["reduction_pct"] == 33 and sd["avg_reduction_pct"] == 33
    back = report_from_dict(payload).findings["summarize"]
    assert back.files == 1
    assert back.file_reduction_tokens == 410
    assert back.past_overspend_tokens is None
    assert back.candidates[0].path == "./CLAUDE.md"
    assert back.candidates[0].reduction_pct == 33     # per-file % survives the round-trip
    assert "meaning may change" in back.caveat        # and the caveat survives the ctor
    assert back.reduction_pct == 33 and back.avg_reduction_pct == 33


# --- Load semantics: on-demand files are not always-on context ----------------
# The defect: every catalog file's WHOLE body was priced as if it were resident
# in every session, on every call. On a real 30-day corpus that made a skill
# library nobody had invoked the single most expensive prompt file a user owns,
# and pushed the cross-analyzer token rollup ABOVE the window's billed tokens.

def _seed_calls(db, sessions: int, calls: int, *, agent_id: str = "claude-code-repo"):
    for i in range(sessions):
        sid = f"s{i}"
        db.upsert_session(make_session(session_id=sid, agent_id=agent_id))
        for j in range(calls):
            db.insert_span(make_llm_span(
                session_id=sid, agent_id=agent_id,
                provider="anthropic", model="claude-haiku-4-5",
                input_tokens=100, output_tokens=10,
                start_time=utcnow() - timedelta(days=1, minutes=j),
            ))


def _corpus(tmp_path, monkeypatch, records_by_session: dict[str, list] | None = None):
    """Point the invocation scan at a controlled transcript corpus."""
    import json as _json

    root = tmp_path / "projects"
    (root / "-repo").mkdir(parents=True)
    for session_id, records in (records_by_session or {}).items():
        (root / "-repo" / f"{session_id}.jsonl").write_text(
            "".join(_json.dumps(r) + "\n" for r in records), encoding="utf-8",
        )
    monkeypatch.setenv("TJ_CLAUDE_PROJECTS_ROOT", str(root))
    monkeypatch.setenv("TJ_TRANSCRIPT_CACHE_DIR", str(tmp_path / "tcache"))
    return root


def _skill_use(slug: str):
    return {"message": {"content": [
        {"type": "tool_use", "id": "t", "name": "Skill", "input": {"skill": slug}},
    ]}}


def test_uninvoked_skill_body_is_not_priced_as_always_on_context(
    db, monkeypatch, tmp_path,
):
    """A skill invoked ZERO times in the window costs only its frontmatter.

    Its body is real context when it IS invoked -- see the next test -- but it
    is not in context on a single call of a session that never invoked it, and
    pricing it as if it were is what produced the indefensible figure.
    """
    _seed_calls(db, sessions=4, calls=5)
    _corpus(tmp_path, monkeypatch, {"s0": [_skill_use("other")]})
    _patch_scan(monkeypatch, [
        _skill_cand("~/.claude/skills/ship/SKILL.md", resident=10, on_demand=9_990),
    ])

    since, until = _window()
    f = build_report(db=db, config=TjConfig(version="1"),
                     since=since, until=until, findings=["summarize"]).findings["summarize"]
    c = f.candidates[0]

    assert f.invocations_observed is True
    assert c.load_class == "skill" and c.invocations == 0
    # 10 frontmatter tokens x 5 reads x 4 sessions -- the body contributes
    # nothing because it was never delivered.
    assert f.past_overspend_tokens == 200
    # Not the 10,000-token body on every read of every session.
    assert f.past_overspend_tokens < 10_000 * 5 * 4
    # The one-time per-call reduction is untouched: the file IS still worth
    # compressing, and the curate/diff surface still says so.
    assert f.file_reduction_tokens == 10_000


def test_an_invoked_skill_body_is_priced_once_per_observed_invocation(
    db, monkeypatch, tmp_path,
):
    """The opposite error -- charging an invoked body zero -- is also wrong: a
    long skill that really is run costs real tokens each time."""
    _seed_calls(db, sessions=4, calls=5)
    _corpus(tmp_path, monkeypatch, {
        "s0": [_skill_use("ship"), _skill_use("ship")],
        "s1": [_skill_use("ship")],
    })
    _patch_scan(monkeypatch, [
        _skill_cand("~/.claude/skills/ship/SKILL.md", resident=10, on_demand=9_990),
    ])

    since, until = _window()
    f = build_report(db=db, config=TjConfig(version="1"),
                     since=since, until=until, findings=["summarize"]).findings["summarize"]
    c = f.candidates[0]

    assert c.invocations == 3
    # frontmatter (10 x 5 reads x 4 sessions) + body (9,990 x 3 invocations)
    assert f.past_overspend_tokens == 200 + 9_990 * 3


def test_no_transcript_corpus_degrades_both_fields_for_an_on_demand_file(
    db, monkeypatch, tmp_path,
):
    """Critical Rule 28 corollary (a): where the invocation count could not be
    OBSERVED at all, an on-demand file carries neither a token figure nor a
    dollar one -- never a number on one side and a zero on the other. An
    always-resident file in the same scan is unaffected: its basis is intact.
    """
    _seed_calls(db, sessions=4, calls=5)
    monkeypatch.setenv("TJ_CLAUDE_PROJECTS_ROOT", str(tmp_path / "absent"))
    monkeypatch.setenv("TJ_TRANSCRIPT_CACHE_DIR", str(tmp_path / "tcache"))
    _patch_scan(monkeypatch, [
        _skill_cand("~/.claude/skills/ship/SKILL.md", resident=10, on_demand=9_990),
        _cand("~/.claude/CLAUDE.md", 1_000, scope="global"),
    ])

    since, until = _window()
    f = build_report(db=db, config=TjConfig(version="1"),
                     since=since, until=until, findings=["summarize"]).findings["summarize"]
    by_path = {c.path: c for c in f.candidates}
    skill = by_path["~/.claude/skills/ship/SKILL.md"]
    always = by_path["~/.claude/CLAUDE.md"]

    assert f.invocations_observed is False
    assert skill.invocations is None
    assert skill.est_usd_saved is None and skill.est_tokens_saved_window is None
    assert always.est_usd_saved is not None and always.est_tokens_saved_window is not None
    # The finding still reports the file it COULD price -- 1,000 x 5 x 4.
    assert f.past_overspend_tokens == 20_000
    # ...and the basis says why the other one carries nothing.
    assert "none available" in f.estimate_basis


def test_basis_names_the_invocation_evidence_it_used(db, monkeypatch, tmp_path):
    """Critical Rule 14: the basis states the arithmetic truthfully, including
    where the invocation multiplier came from."""
    _seed_calls(db, sessions=2, calls=2)
    _corpus(tmp_path, monkeypatch, {"s0": [_skill_use("ship")]})
    _patch_scan(monkeypatch, [
        _skill_cand("~/.claude/skills/ship/SKILL.md", resident=10, on_demand=90),
    ])
    since, until = _window()
    f = build_report(db=db, config=TjConfig(version="1"),
                     since=since, until=until, findings=["summarize"]).findings["summarize"]

    assert "Claude Code transcripts" in f.estimate_basis
    assert "1 invocation(s) observed" in f.estimate_basis
    assert f.invocations_total == 1 and f.transcripts_examined == 1
    # No claim is strengthened anywhere in it.
    assert "saves you" not in f.estimate_basis
    assert "estimated" in f.caveat.lower() or "review" in f.caveat.lower()


def test_a_session_touching_two_agent_ids_counts_once(db, monkeypatch, tmp_path):
    """`sessions_loading` for a global file is DISTINCT sessions in the window.

    Summing the per-agent_id `COUNT(DISTINCT session_id)` groups counted such a
    session twice -- 22% inflation on a real corpus, applied to every global
    candidate.
    """
    db.upsert_session(make_session(session_id="s0", agent_id="claude-code-repo-a"))
    for agent in ("claude-code-repo-a", "claude-code-repo-b"):
        db.insert_span(make_llm_span(
            session_id="s0", agent_id=agent,
            provider="anthropic", model="claude-haiku-4-5",
            input_tokens=100, output_tokens=10,
            start_time=utcnow() - timedelta(days=1),
        ))
    _corpus(tmp_path, monkeypatch, {})
    _patch_scan(monkeypatch, [_cand("~/.claude/CLAUDE.md", 1_000, scope="global")])

    since, until = _window()
    f = build_report(db=db, config=TjConfig(version="1"),
                     since=since, until=until, findings=["summarize"]).findings["summarize"]

    assert f.sessions_examined == 1                  # not 2
    assert f.candidates[0].sessions_loading == 1
    # 2 calls in that one session -> 1 send + 1 re-read.
    assert f.calls_per_session == 2.0
    assert f.past_overspend_tokens == 2_000


def test_mixed_finding_stays_inside_the_real_price_band(db, monkeypatch, tmp_path):
    """Critical Rule 28, restated for the two-term model: the always-resident
    term bills at a blend of input + cache-read rates and the on-demand term at
    the input rate, so the implied rate over BOTH must still land between the
    cache-read rate and the input rate. A basis mismatch on either term throws
    it orders of magnitude out of band; a hardcoded-number assertion would not
    notice.
    """
    from tokenjam.core.pricing import get_rates

    _seed_calls(db, sessions=4, calls=5)
    _corpus(tmp_path, monkeypatch, {"s0": [_skill_use("ship"), _skill_use("ship")]})
    _patch_scan(monkeypatch, [
        _skill_cand("~/.claude/skills/ship/SKILL.md", resident=120, on_demand=9_880),
        _cand("~/.claude/CLAUDE.md", 3_000, scope="global"),
    ])

    since, until = _window()
    f = build_report(db=db, config=TjConfig(version="1"),
                     since=since, until=until, findings=["summarize"]).findings["summarize"]

    rates = get_rates("anthropic", "claude-haiku-4-5")
    input_per_token = rates.input_per_mtok / 1_000_000
    cache_read_per_token = rates.cache_read_per_mtok / 1_000_000

    assert f.past_overspend_usd is not None and f.past_overspend_tokens
    implied = f.past_overspend_usd / f.past_overspend_tokens
    assert cache_read_per_token <= implied <= input_per_token
    # Strictly inside, not pinned to an endpoint: both terms really contribute.
    assert cache_read_per_token < implied < input_per_token


def test_reduction_pct_keeps_its_one_time_numerator(db, monkeypatch, tmp_path):
    """The basis change must not leak into the percentage: `reduction_pct` is
    saved / source tokens and would read as far over 100% the moment sessions,
    calls or invocations multiplied into its numerator."""
    _seed_calls(db, sessions=4, calls=5)
    _corpus(tmp_path, monkeypatch, {"s0": [_skill_use("ship")]})
    _patch_scan(monkeypatch, [
        _skill_cand("~/.claude/skills/ship/SKILL.md", resident=10, on_demand=990),
    ])
    since, until = _window()
    f = build_report(db=db, config=TjConfig(version="1"),
                     since=since, until=until, findings=["summarize"]).findings["summarize"]

    assert 0 < f.reduction_pct <= 100
    assert 0 < f.avg_reduction_pct <= 100
    assert f.file_reduction_tokens == 1_000       # one-time, never window-priced


def test_render_summarize_labels_on_demand_files(db, monkeypatch, tmp_path, capsys):
    """Critical Rule 14 on the CLI surface: the copy must not tell the reader
    a skill body is re-sent on every call, and it must show the invocation
    evidence behind the figure it does print."""
    from tokenjam.cli.cmd_optimize import _render_summarize

    _seed_calls(db, sessions=2, calls=4)
    _corpus(tmp_path, monkeypatch, {"s0": [_skill_use("ship")]})
    _patch_scan(monkeypatch, [
        _skill_cand("~/.claude/skills/ship/SKILL.md", resident=500, on_demand=5_000),
        _cand("~/.claude/CLAUDE.md", 900, scope="global"),
    ])
    since, until = _window()
    finding = build_report(db=db, config=TjConfig(version="1"),
                           since=since, until=until,
                           findings=["summarize"]).findings["summarize"]

    _render_summarize(finding, pricing_mode="api", marker="①")
    out = capsys.readouterr().out.replace("\n", " ")

    assert "on demand" in out
    assert "always-on" in out
    assert "invocation(s) observed" in out
    # The old, now-false claim that every one of these files is re-sent on
    # every call must not come back.
    assert "re-send these files on every call" not in out


def test_a_file_measured_to_cost_nothing_is_not_listed_at_all(
    db, monkeypatch, tmp_path,
):
    """Critical Rule 22: a `.claude/commands/x.md` with no frontmatter that was
    never invoked is resident in no session and delivered on no call. It is not
    a candidate worth `$0.00` — rendering that zero would read as "we looked for
    a saving and found none", when the truth is there is nothing to summarize
    here this window. It drops out of `files`, the aggregate and the %s too.
    """
    _seed_calls(db, sessions=3, calls=4)
    _corpus(tmp_path, monkeypatch, {"s0": [_skill_use("ship")]})
    _patch_scan(monkeypatch, [
        # No frontmatter (resident=0), never invoked.
        _cand("~/.claude/commands/dormant.md", 700, scope="global",
              load_class="command", resident=0, on_demand=700),
        _skill_cand("~/.claude/skills/ship/SKILL.md", resident=40, on_demand=600),
    ])

    since, until = _window()
    f = build_report(db=db, config=TjConfig(version="1"),
                     since=since, until=until, findings=["summarize"]).findings["summarize"]

    assert [c.path for c in f.candidates] == ["~/.claude/skills/ship/SKILL.md"]
    assert f.files == 1
    # The suppressed file's one-time reduction leaves the aggregate with it.
    assert f.file_reduction_tokens == 640
    assert f.past_overspend_usd is not None and f.past_overspend_usd > 0


def test_an_unmeasured_file_is_kept_even_though_it_carries_no_figure(
    db, monkeypatch, tmp_path,
):
    """The inverse of the rule above, and the reason it is keyed on a MEASURED
    zero: a candidate whose window figure degraded to None must NOT be
    suppressed. "Not measured" is not "worth nothing", and dropping it would
    hide a file the analyzer simply failed to price."""
    _seed_calls(db, sessions=3, calls=4)
    monkeypatch.setenv("TJ_CLAUDE_PROJECTS_ROOT", str(tmp_path / "absent"))
    monkeypatch.setenv("TJ_TRANSCRIPT_CACHE_DIR", str(tmp_path / "tcache"))
    _patch_scan(monkeypatch, [
        _cand("~/.claude/commands/dormant.md", 700, scope="global",
              load_class="command", resident=0, on_demand=700),
    ])

    since, until = _window()
    f = build_report(db=db, config=TjConfig(version="1"),
                     since=since, until=until, findings=["summarize"]).findings["summarize"]

    assert f.files == 1
    assert f.candidates[0].est_usd_saved is None
    assert f.candidates[0].est_tokens_saved_window is None


def test_basis_keeps_the_two_terms_distinguishable(db, monkeypatch, tmp_path):
    """The two terms must stay separable by a reader: collapsing them back into
    one is how both the always-on and the frontmatter-only errors get made."""
    _seed_calls(db, sessions=3, calls=4)
    _corpus(tmp_path, monkeypatch, {"s0": [_skill_use("ship")]})
    _patch_scan(monkeypatch, [
        _skill_cand("~/.claude/skills/ship/SKILL.md", resident=40, on_demand=600),
    ])
    since, until = _window()
    f = build_report(db=db, config=TjConfig(version="1"),
                     since=since, until=until, findings=["summarize"]).findings["summarize"]

    c = f.candidates[0]
    assert c.always_resident_tokens_saved == 40
    assert c.on_demand_tokens_saved == 600
    assert "always_resident_tokens_saved" in f.estimate_basis
    assert "on_demand_tokens_saved" in f.estimate_basis


# --------------------------------------------------------------------------- #
# File population: many repo roots, and one charge per file however many
# checkouts of it exist on disk.
# --------------------------------------------------------------------------- #

def _repo_window(db, repo: str, sessions: int, calls: int = 2) -> None:
    """`sessions` sessions in `repo`, each making `calls` LLM calls."""
    agent_id = f"claude-code-{repo}"
    for n in range(sessions):
        sid = f"{repo}-{n}"
        db.upsert_session(make_session(session_id=sid, agent_id=agent_id))
        for _ in range(calls):
            db.insert_span(make_llm_span(
                session_id=sid, agent_id=agent_id, provider="anthropic",
                model="claude-haiku-4-5", input_tokens=100, output_tokens=10,
                start_time=utcnow() - timedelta(days=1),
            ))


def _rooted(path: str, root: str, saved: int) -> Candidate:
    """A project-scope candidate that remembers the root it was scanned under."""
    from dataclasses import replace

    return replace(_cand(path, saved, scope="project"), scan_root=root)


def test_one_charge_per_file_however_many_checkouts_hold_it(db, monkeypatch):
    """A git worktree carries its repo's directory name, so every checkout of one
    `CLAUDE.md` matches the SAME sessions by ancestor name. Charging each would
    multiply one file's cost by however many checkouts happen to exist — money
    the user cannot act on. Copies drift, so this must hold for copies whose
    CONTENT differs, not just byte-identical ones."""
    _repo_window(db, "myrepo", sessions=4)
    _patch_scan(monkeypatch, [
        _rooted("/code/myrepo/CLAUDE.md", "/code/myrepo", 1_000),
        # Same repo label, same slot, DIFFERENT size (a drifted checkout).
        _rooted("/code/wt/branch-a/myrepo/CLAUDE.md", "/code/wt/branch-a/myrepo", 1_400),
        _rooted("/code/wt/branch-b/myrepo/CLAUDE.md", "/code/wt/branch-b/myrepo", 900),
    ])
    f = _run(db)

    assert f.files == 1
    assert f.duplicate_copies_collapsed == 2
    # The retained copy is the shallowest path — the working copy, not a
    # worktree cut from it — so the file named for the fix is the actionable
    # one, and an incidental larger checkout cannot inflate the figure.
    assert f.candidates[0].path == "/code/myrepo/CLAUDE.md"
    assert f.candidates[0].sessions_loading == 4


def test_same_slot_in_different_repos_is_charged_twice(db, monkeypatch):
    """The collapse keys on the LOADING POPULATION, not the file's content: two
    repos' `CLAUDE.md` are loaded by different sessions and genuinely cost twice,
    even when their bytes are identical."""
    _repo_window(db, "alpha", sessions=3)
    _repo_window(db, "beta", sessions=3)
    _patch_scan(monkeypatch, [
        _rooted("/code/alpha/CLAUDE.md", "/code/alpha", 1_000),
        _rooted("/code/beta/CLAUDE.md", "/code/beta", 1_000),
    ])
    f = _run(db)

    assert f.files == 2
    assert f.duplicate_copies_collapsed == 0
    assert {c.path for c in f.candidates} == {
        "/code/alpha/CLAUDE.md", "/code/beta/CLAUDE.md",
    }


def test_different_slots_in_one_repo_are_charged_separately(db, monkeypatch):
    """Same repo, different files — a `CLAUDE.md` and a command are two real
    files, and the collapse must not merge them just because they share a
    session population."""
    _repo_window(db, "myrepo", sessions=3)
    _patch_scan(monkeypatch, [
        _rooted("/code/myrepo/CLAUDE.md", "/code/myrepo", 1_000),
        _rooted("/code/myrepo/AGENTS.md", "/code/myrepo", 800),
    ])
    f = _run(db)

    assert f.files == 2
    assert f.duplicate_copies_collapsed == 0


def test_global_files_are_never_collapsed(db, monkeypatch):
    """Two distinct `~/.claude` files share no repo label and are each real."""
    _repo_window(db, "myrepo", sessions=2)
    _patch_scan(monkeypatch, [
        _cand("~/.claude/CLAUDE.md", 900, scope="global"),
        _cand("~/.claude/rules/style.md", 700, scope="global"),
    ])
    f = _run(db)

    assert f.files == 2
    assert f.duplicate_copies_collapsed == 0


def test_unmatched_project_file_is_never_collapsed(db, monkeypatch):
    """A file whose loading sessions could not be identified carries no window
    figure. "Not measured" is not evidence of sameness, so two of them must not
    silently merge into one."""
    _repo_window(db, "myrepo", sessions=2)
    _patch_scan(monkeypatch, [
        _rooted("/elsewhere/one/CLAUDE.md", "/elsewhere/one", 900),
        _rooted("/elsewhere/two/CLAUDE.md", "/elsewhere/two", 900),
    ])
    f = _run(db)

    assert f.files == 2
    assert f.duplicate_copies_collapsed == 0
    assert all(c.est_usd_saved is None for c in f.candidates)
    assert all(c.est_tokens_saved_window is None for c in f.candidates)


def test_basis_states_the_scanned_population(db, monkeypatch):
    """Critical Rule 14: the basis must say which files were even looked at —
    how many roots, and that vanished ones are excluded rather than guessed."""
    from tokenjam.core.optimize.analyzers.summarize import _estimate_basis
    from tokenjam.core.summarize.invocations import InvocationCounts
    from tokenjam.core.summarize.repo_roots import ResolvedRoots

    roots = ResolvedRoots(roots=(), recorded=9, vanished=3)
    basis = _estimate_basis(InvocationCounts(observed=True), roots, roots_scanned=6)
    assert "6 project root(s)" in basis
    assert "3" in basis and "no longer exist on disk" in basis

    # No root observed at all: say so, rather than implying the whole corpus
    # was scanned when only the process's own directory was.
    cwd_only = _estimate_basis(InvocationCounts(observed=True), roots, roots_scanned=0)
    assert "only in the directory this process runs from" in cwd_only

    # Every root present: no vanished-root claim is made.
    clean = _estimate_basis(
        InvocationCounts(observed=True), ResolvedRoots(recorded=4), roots_scanned=4,
    )
    assert "no longer exist on disk" not in clean


def test_nested_roots_stack_but_parallel_checkouts_collapse(db, monkeypatch):
    """A meta-repo's `CLAUDE.md` and the `CLAUDE.md` of a sub-repo checked out
    inside it are BOTH loaded — the harness reads the working directory's file
    and its ancestors'. A worktree cut from that sub-repo lives in a parallel
    tree and is the same file seen twice. Same repo label and same slot for all
    three, so only the nesting tells them apart."""
    _repo_window(db, "myrepo", sessions=3)
    _patch_scan(monkeypatch, [
        _rooted("/code/myrepo/CLAUDE.md", "/code/myrepo", 500),           # meta
        _rooted("/code/myrepo/myrepo/CLAUDE.md", "/code/myrepo/myrepo", 1_200),
        _rooted("/wt/branch/myrepo/CLAUDE.md", "/wt/branch/myrepo", 1_100),
    ])
    f = _run(db)

    assert {c.path for c in f.candidates} == {
        "/code/myrepo/CLAUDE.md", "/code/myrepo/myrepo/CLAUDE.md",
    }
    assert f.duplicate_copies_collapsed == 1


def test_basis_says_the_unmeasured_figure_uses_a_prior_from_other_machines(db, monkeypatch):
    """Critical Rule 14 + 30(c). With no verified sample here, the figure rests
    on tokenjam's own measurement of OTHER people's files, so the basis must say
    that, disclose the sample size and spread, and say plainly that it is not
    the target the rewriter is asked for. (Inverted from an earlier version that
    asserted the basis called the number a TARGET — estimating at the ask was
    the defect, and the guard now defends the corrected state.)"""
    from tokenjam.core.optimize.analyzers.summarize import _estimate_basis
    from tokenjam.core.summarize.estimate import (
        UNMEASURED_PRIOR_RANGE,
        UNMEASURED_PRIOR_SAMPLES,
    )
    from tokenjam.core.summarize.invocations import InvocationCounts

    basis = _estimate_basis(InvocationCounts(observed=True))

    assert "No verified rewrite exists on THIS machine yet" in basis
    assert "not your files" in basis or "not yours" in basis or "were not your files" in basis
    assert f"{UNMEASURED_PRIOR_SAMPLES:,} rewrites" in basis      # sample size
    assert f"{UNMEASURED_PRIOR_RANGE[0]:.0%}" in basis            # ...and spread
    assert f"{UNMEASURED_PRIOR_RANGE[1]:.0%}" in basis
    assert "deliberately NOT the 50% target" in basis
    assert "overstated this figure by roughly an order of magnitude" in basis
    assert "tj summarize calibrate" in basis
    assert "Symlinked files are excluded" in basis


def test_basis_switches_to_the_measured_ratio_when_one_exists(db, monkeypatch):
    from tokenjam.core.optimize.analyzers.summarize import _estimate_basis
    from tokenjam.core.summarize.invocations import InvocationCounts

    basis = _estimate_basis(
        InvocationCounts(observed=True), None, 0, 0.82, True, 7,
    )

    assert "ACTUALLY delivered" in basis
    assert "7 structure-checked" in basis
    assert "upper bound" not in basis


def test_basis_names_the_command_that_produces_the_missing_evidence(db):
    """Waiting passively for a user to happen to rewrite files is what made the
    target permanent. The unmeasured basis says how to measure it."""
    from tokenjam.core.optimize.analyzers.summarize import _estimate_basis
    from tokenjam.core.summarize.invocations import InvocationCounts

    basis = _estimate_basis(InvocationCounts(observed=True), None, 0, 0.5, False, 0)

    assert "tj summarize calibrate" in basis


def test_basis_discloses_gate_failures_excluded_from_the_ratio(db):
    """A sample made mostly of failed rewrites must not read as a clean
    measurement, so the excluded attempts are stated rather than implied."""
    from tokenjam.core.optimize.analyzers.summarize import _estimate_basis
    from tokenjam.core.summarize.invocations import InvocationCounts

    clean = _estimate_basis(InvocationCounts(observed=True), None, 0, 0.82, True, 7)
    noisy = _estimate_basis(InvocationCounts(observed=True), None, 0, 0.82, True, 7, 4)

    assert "failed the structure check" not in clean
    assert "4 attempted rewrite(s) here failed the structure check" in noisy
    assert "excluded from the ratio" in noisy


def test_basis_attributes_the_line_target_to_anthropic_and_claims_only_tokens(db):
    """The size target is theirs, the adherence benefit is their rationale and
    not a saving we measure, and the path-scoped-rules alternative is a mention
    rather than something summarize writes (Critical Rule 14)."""
    from tokenjam.core.optimize.analyzers.summarize import _estimate_basis
    from tokenjam.core.summarize.estimate import PUBLISHED_LINE_TARGET
    from tokenjam.core.summarize.invocations import InvocationCounts

    basis = _estimate_basis(InvocationCounts(observed=True), None, 0, 0.5, False, 0)

    assert f"under {PUBLISHED_LINE_TARGET} lines" in basis
    assert "Anthropic's published guidance" in basis
    assert "not tokenjam's" in basis
    assert "only the token reduction is" in basis


def test_basis_refuses_to_present_compression_as_the_only_route(db):
    """An instruction file is usually long because rules ACCUMULATED, so
    compressing it shortens each surviving rule rather than removing any. That
    trades adherence for tokens (Critical Rule 26, gate 3), so the three routes
    that cost no specificity have to be named alongside it."""
    from tokenjam.core.optimize.analyzers.summarize import _estimate_basis
    from tokenjam.core.summarize.invocations import InvocationCounts
    from tokenjam.core.summarize.route import PRUNE_TEST_QUOTE

    basis = _estimate_basis(InvocationCounts(observed=True), None, 0, 0.5, False, 0)

    assert "only ONE of four routes" in basis
    assert "the only one that costs specificity" in basis
    assert PRUNE_TEST_QUOTE in basis                  # prune
    assert "`paths:` frontmatter" in basis            # path-scope
    assert "escalating a must-always-run instruction to a hook" in basis
    assert "summarize performs NONE of them; it names them" in basis
    # The diagnosis is of prose SHAPE, never of which rules are needed.
    assert "never of which rules earn their place" in basis
    assert "withheld rather than guessed" in basis
    # And meaning is not verified by anything automatic.
    assert "does not read the prose" in basis


def test_basis_states_the_figure_prices_only_the_operation_we_perform(db):
    """The coverage statement must stay accurate as the product grows.

    It used to say relocation was NOT counted and its size unmeasured. Relocation
    is now performed and priced, so the same pin is INVERTED rather than deleted
    (Critical Rule 23): it now asserts that relocation is priced in its own named
    fields, that the two figures must not be added, and that the two routes still
    NOT performed are the ones described as unmeasured. Deleting the assertion
    would have left the whole class uncovered at exactly the moment it changed."""
    from tokenjam.core.optimize.analyzers.summarize import _estimate_basis
    from tokenjam.core.summarize.invocations import InvocationCounts

    basis = _estimate_basis(InvocationCounts(observed=True), None, 0, 0.5, False, 0)

    assert "COVERAGE:" in basis
    assert "prices ONLY what compression recovers" in basis
    assert "not the size of the opportunity" in basis
    # Relocation is now performed, priced, and named — no longer "not counted".
    assert "relocation_past_overspend_usd" in basis
    assert "pure MOVE with no semantic loss" in basis
    assert "must never be added together" in basis
    # ...and the routes still unperformed are the ones called unmeasured.
    assert "unmeasured, not zero" in basis
    assert "pruning rules that do not earn their place" in basis


def test_a_prune_route_candidate_keeps_its_full_figure(db, monkeypatch, tmp_path):
    """The route changes what is OFFERED, never what is claimed: the tokens are
    recoverable by whichever route the user picks, so a rule-heavy file is not
    quietly discounted for being a pruning candidate."""
    from tokenjam.core.summarize import route
    from tokenjam.core.summarize.candidates import Candidate

    rules = "\n".join(f"- Never skip step {i}; run its own check first." for i in range(60))
    advice = route.recommend_route(text=rules, load_class="always")
    assert advice.route == route.ROUTE_PRUNE

    c = Candidate(
        path="/x/CLAUDE.md", prose_words=600, total_chars=4_000, protected_blocks=0,
        est_tokens_saved=500, pricing_mode="api", scope="project", is_prompt=True,
        reduction_route=advice.route, directive_share=advice.directive_share,
    )
    assert c.est_tokens_saved == 500                  # undiscounted
    assert c.to_dict()["reduction_route"] == route.ROUTE_PRUNE


def test_measured_ratio_supersedes_the_target_in_the_figure(db, monkeypatch, tmp_path):
    """The measured ratio is fed to the scan, so the reduction shrinks to what
    rewrites really deliver rather than what they are asked for."""
    import json

    from tokenjam.core.config import StorageConfig
    from tokenjam.core.summarize.session import results_dir

    # tmp-scoped storage: `results_dir` hangs off the storage path, and a test
    # must never read or unlink the developer's own staged results.
    cfg = TjConfig(version="1", storage=StorageConfig(path=str(tmp_path / "t.duckdb")))
    seen: dict = {}

    def fake_scan(**kw):
        seen.update(kw)
        return ScanResult(candidates=[], root=".", recursive=False,
                          globals_checked=0, walk_capped=False, note="")

    monkeypatch.setattr(
        "tokenjam.core.summarize.candidates.list_candidates", fake_scan,
    )
    d = results_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    for n in ("a", "b", "c"):
        (d / f"{n}.json").write_text(json.dumps({
            "path": f"/x/{n}.md", "staged": True, "prose_words_before": 1_000,
            "words_before": 1_200, "words_after": 1_100,
        }), encoding="utf-8")
    _seed_window(db)
    since, until = _window()
    report = build_report(db=db, config=cfg, since=since, until=until,
                          findings=["summarize"])

    f = report.findings["summarize"]
    assert seen["ratio"] == pytest.approx(0.9)      # 300 removed of 3,000 prose words
    assert f.prose_ratio == pytest.approx(0.9)
    assert f.prose_ratio_observed is True
    assert f.prose_ratio_samples == 3


# --------------------------------------------------------------------------- #
# Relocation: the second operation, priced beside compression and never added
# to it (Critical Rules 27 + 28)
# --------------------------------------------------------------------------- #

def _reloc_cand(path: str, saved: int, relocatable_chars: int, *, scope: str = "global"):
    import dataclasses
    return dataclasses.replace(
        _cand(path, saved, scope=scope), relocatable_content_chars=relocatable_chars,
    )


def test_relocation_is_priced_on_the_same_multiplier_as_compression(db, monkeypatch):
    """The whole point of routing it through the same per-file session/call
    multiplier: the two operations become directly comparable, and the token and
    dollar fields count the same events (Critical Rule 28)."""
    db.upsert_session(make_session(agent_id=_CC_AGENT_ID, session_id="s1"))
    db.upsert_session(make_session(agent_id=_CC_AGENT_ID, session_id="s2"))
    for sid in ("s1", "s2"):
        for _ in range(3):
            db.insert_span(make_llm_span(
                session_id=sid, provider="anthropic", model="claude-haiku-4-5",
                input_tokens=100, output_tokens=10,
                start_time=utcnow() - timedelta(days=1),
            ))
    # 4_000 content chars relocatable = 1_000 tokens on the shared constant,
    # which is exactly the compression reduction — so the two figures must come
    # out identical, proving they run through one multiplier and not two.
    _patch_scan(monkeypatch, [_reloc_cand("~/.claude/CLAUDE.md", 1_000, 4_000)])

    since, until = _window()
    f = build_report(db=db, config=TjConfig(version="1"), since=since, until=until,
                     findings=["summarize"]).findings["summarize"]

    assert f.relocation_files == 1
    assert f.relocation_file_reduction_tokens == 1_000
    assert f.relocation_past_overspend_tokens == f.past_overspend_tokens
    assert f.relocation_past_overspend_usd == pytest.approx(f.past_overspend_usd)


def test_the_relocation_figure_is_never_summed_into_the_compression_one(db, monkeypatch):
    """A section relocated out of a file is no longer there to be compressed, so
    adding the two would price the same text twice (Critical Rule 27). They are
    alternatives the user picks between, per file."""
    db.upsert_session(make_session(agent_id=_CC_AGENT_ID, session_id="s1"))
    for _ in range(3):
        db.insert_span(make_llm_span(
            session_id="s1", provider="anthropic", model="claude-haiku-4-5",
            input_tokens=100, output_tokens=10,
            start_time=utcnow() - timedelta(days=1),
        ))
    _patch_scan(monkeypatch, [_reloc_cand("~/.claude/CLAUDE.md", 1_000, 4_000)])

    since, until = _window()
    f = build_report(db=db, config=TjConfig(version="1"), since=since, until=until,
                     findings=["summarize"]).findings["summarize"]

    # The compression aggregate is untouched by the relocation figure existing.
    assert f.past_overspend_tokens == f.candidates[0].est_tokens_saved_window
    assert f.relocation_past_overspend_tokens is not None
    assert f.past_overspend_tokens != (
        f.past_overspend_tokens + f.relocation_past_overspend_tokens
    )


def test_the_relocation_implied_rate_lands_inside_a_real_price_band(db, monkeypatch):
    """Critical Rule 28, written as the mechanical check it prescribes: divide
    the dollars by the tokens and the implied per-token rate must sit between
    the cache-read rate and the input rate. A hardcoded-number assertion passes
    happily while both fields drift; a rate assertion cannot, because a basis
    mismatch always throws the implied rate orders of magnitude out of band."""
    from tokenjam.core.pricing import get_rates

    db.upsert_session(make_session(agent_id=_CC_AGENT_ID, session_id="s1"))
    for _ in range(8):
        db.insert_span(make_llm_span(
            session_id="s1", provider="anthropic", model="claude-haiku-4-5",
            input_tokens=100, output_tokens=10,
            start_time=utcnow() - timedelta(days=1),
        ))
    _patch_scan(monkeypatch, [_reloc_cand("~/.claude/CLAUDE.md", 1_000, 4_000)])

    since, until = _window()
    f = build_report(db=db, config=TjConfig(version="1"), since=since, until=until,
                     findings=["summarize"]).findings["summarize"]

    rates = get_rates("anthropic", "claude-haiku-4-5")
    implied = f.relocation_past_overspend_usd / f.relocation_past_overspend_tokens * 1_000_000
    assert rates.cache_read_per_mtok <= implied <= rates.input_per_mtok


def test_a_file_with_nothing_relocatable_carries_no_relocation_figure(db, monkeypatch):
    """Symmetric degrade: a measured zero is not a figure to render, and the
    aggregate is `None` rather than `0` when nothing qualified — "no reference
    section here" is not "relocation is worth nothing"."""
    db.upsert_session(make_session(agent_id=_CC_AGENT_ID, session_id="s1"))
    db.insert_span(make_llm_span(
        session_id="s1", provider="anthropic", model="claude-haiku-4-5",
        input_tokens=100, output_tokens=10,
        start_time=utcnow() - timedelta(days=1),
    ))
    _patch_scan(monkeypatch, [_cand("~/.claude/CLAUDE.md", 1_000, scope="global")])

    since, until = _window()
    f = build_report(db=db, config=TjConfig(version="1"), since=since, until=until,
                     findings=["summarize"]).findings["summarize"]

    assert f.relocation_files == 0
    assert f.relocation_past_overspend_usd is None
    assert f.relocation_past_overspend_tokens is None
    assert f.relocation_file_reduction_tokens is None
    # ...while the compression figure is unaffected.
    assert f.past_overspend_usd is not None


def test_an_unpriceable_window_carries_neither_relocation_figure(db, monkeypatch):
    """Same "no evidence" condition the compression figures degrade on: a file
    no observed session loads gets no window figure at all, never a zero."""
    _patch_scan(monkeypatch, [_reloc_cand("./CLAUDE.md", 400, 4_000, scope="repo")])
    f = _run(db)
    assert f.relocation_past_overspend_usd is None
    assert f.relocation_past_overspend_tokens is None
    # The one-time figure survives, exactly as `file_reduction_tokens` does.
    assert f.relocation_file_reduction_tokens == 1_000
