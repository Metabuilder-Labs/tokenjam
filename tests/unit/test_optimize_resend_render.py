"""`tj optimize resend` — CLI text-view rendering for the context-resend finding.

Same class of gap `relearn`/`deadweight`/`summarize` shipped with before their
renderers landed: `context_resend.py` registers "resend" and runs on every
report (see tests/unit/test_context_resend.py for the analyzer's own tests),
but had no entry in cmd_optimize's `_FINDING_RENDERERS` dispatch table, so
plain-text `tj optimize` showed nothing for it -- only `--json` did. These
tests exercise the renderer added to close that gap, independent of the
analyzer's own correctness tests.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tokenjam.core.config import CaptureConfig, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize import build_report
from tests.factories import make_llm_span, make_session, make_tool_span

UTC = timezone.utc


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def _config(*, tool_inputs=False) -> TjConfig:
    return TjConfig(version="1", capture=CaptureConfig(tool_inputs=tool_inputs))


def _seed_session(db, session_id, sizes, *, provider="anthropic",
                   model="claude-haiku-4-5", cache_ratio=0.0,
                   start=None, cost_usd=0.01):
    """One session with `len(sizes)` LLM turns; `sizes[i]` is that turn's
    prompt_size (input_tokens + cache_tokens). Mirrors the analyzer test's
    own seeding helper (tests/unit/test_context_resend.py) without importing
    across test files."""
    start = start or datetime(2026, 5, 10, tzinfo=UTC)
    db.upsert_session(make_session(session_id=session_id, plan_tier="api"))
    for i, size in enumerate(sizes):
        cache_tok = int(size * cache_ratio)
        input_tok = size - cache_tok
        db.insert_span(make_llm_span(
            session_id=session_id, provider=provider, model=model,
            input_tokens=input_tok, cache_tokens=cache_tok, output_tokens=50,
            cost_usd=cost_usd, start_time=start + timedelta(minutes=i),
        ))


def _window():
    return datetime(2026, 5, 1, tzinfo=UTC), datetime(2026, 5, 30, tzinfo=UTC)


def _flat(out: str) -> str:
    """Collapse Rich's terminal-width line wrapping to a single line so a
    long fixed string (a caveat, a fix sentence) can be matched by substring
    regardless of where the console happened to wrap it."""
    return " ".join(out.split())


def _run(db, config=None, *, tool_inputs=False):
    config = config or _config(tool_inputs=tool_inputs)
    since, until = _window()
    report = build_report(db=db, config=config, since=since, until=until,
                          findings=["resend"])
    return report, report.findings["resend"]


def _seed_heavy_resend(db, *, cache_ratio=0.0, provider="anthropic",
                        model="claude-haiku-4-5"):
    """A window that clears both signal thresholds and carries a real
    repeat-share, so the renderer has a populated finding to work with."""
    _seed_session(db, "heavy", [1000, 1000, 1000, 1000],
                  cache_ratio=cache_ratio, provider=provider, model=model)
    _seed_session(db, "pad1", [500])
    _seed_session(db, "pad2", [500])


# --------------------------------------------------------------------------- #
# Reachability: registered choice + dispatch table + minor-findings label
# --------------------------------------------------------------------------- #

def test_resend_in_click_choices_and_renderer():
    from tokenjam.cli.cmd_optimize import (
        _FINDING_RENDERERS,
        _MINOR_FINDING_LABELS,
        cmd_optimize,
    )

    findings_param = next(
        p for p in cmd_optimize.params if getattr(p, "name", None) == "findings"
    )
    assert "resend" in findings_param.type.choices
    assert "resend" in _FINDING_RENDERERS
    assert "resend" in _MINOR_FINDING_LABELS


# --------------------------------------------------------------------------- #
# Empty state / below-threshold: never a bare "nothing found"
# --------------------------------------------------------------------------- #

def test_render_resend_empty_state_names_the_reason(db, capsys):
    from tokenjam.cli.cmd_optimize import _render_resend

    # No sessions at all in the window -> repeat_share stays None.
    _, finding = _run(db)
    assert finding.repeat_share is None

    _render_resend(finding, pricing_mode="api", marker="①")
    out = capsys.readouterr().out

    assert "No LLM turns" in out
    assert "None" not in out.replace("No LLM turns in this window.", "")


# --------------------------------------------------------------------------- #
# Populated finding: headline, distribution, examples, recurring, fix, caveat
# --------------------------------------------------------------------------- #

def test_render_resend_shows_headline_examples_and_caveat(db, capsys):
    from tokenjam.cli.cmd_optimize import _render_resend

    _seed_heavy_resend(db)
    _, finding = _run(db)
    assert finding.repeat_share is not None

    for mode in ("api", "subscription", "local", "unknown"):
        _render_resend(finding, pricing_mode=mode, marker="①", persona="unknown")
    out = _flat(capsys.readouterr().out)

    assert "heavy" in out                        # heaviest example named
    assert "conservative lower bound" in out      # caveat renders verbatim
    assert finding.caveat in out
    assert "No candidates flagged" not in out
    # Compaction fix always present regardless of persona/pricing mode.
    assert "compact" in out.lower()


def test_render_resend_below_threshold_shows_no_examples_or_fix(db, capsys):
    """Below the data threshold, only the reason should print -- no headline
    figures, no examples section, no fix CTA (there is nothing to fix yet)."""
    from tokenjam.cli.cmd_optimize import _render_resend

    _seed_session(db, "s1", [100, 200, 300])
    _seed_session(db, "s2", [100, 200, 300])
    _, finding = _run(db)
    assert finding.repeat_share is None

    _render_resend(finding, pricing_mode="api", marker="①")
    out = capsys.readouterr().out

    assert "too few sessions" in out
    assert "Fix:" not in out
    # No examples/heaviest-sessions section prints below the threshold.
    assert "Heaviest sessions" not in out


def test_render_resend_recurring_examples_reuse_context_labels(db, capsys):
    """The 'why' section reuses `tj context`'s own inclusion-tag lookup
    (`_INCLUSION_LABELS`) rather than re-deriving a second translation table."""
    from tokenjam.cli.cmd_context import _INCLUSION_LABELS
    from tokenjam.cli.cmd_optimize import _render_resend
    from tokenjam.core.context_diagnostic import INCLUSION_FILE_READ

    base = datetime(2026, 5, 10, tzinfo=UTC)
    for i, sid in enumerate(["s1", "s2", "s3"]):
        _seed_session(db, sid, [100, 200, 300], start=base + timedelta(hours=i))
        ts = make_tool_span(tool_name="Read", tool_input={"file_path": "/repo/schema.py"})
        ts.session_id = sid
        ts.start_time = base + timedelta(hours=i, minutes=1)
        db.insert_span(ts)
    _, finding = _run(db, tool_inputs=True)
    assert finding.recurring_examples

    _render_resend(finding, pricing_mode="api", marker="①")
    out = _flat(capsys.readouterr().out)

    assert "/repo/schema.py" in out
    tag = _INCLUSION_LABELS[INCLUSION_FILE_READ]
    assert f"[{tag}]" in out


def test_render_resend_no_dollar_figure_when_no_priced_example(db, capsys):
    """A fully-cached session recovers no USD (already captured by
    cache_efficacy's own figure) but still recovers tokens -- api mode must
    say why the dollar figure is absent rather than silently print nothing."""
    from tokenjam.cli.cmd_optimize import _render_resend

    _seed_heavy_resend(db, cache_ratio=1.0)
    _, finding = _run(db)
    assert finding.estimated_recoverable_usd is None
    assert finding.estimated_recoverable_tokens

    _render_resend(finding, pricing_mode="api", marker="①")
    out = capsys.readouterr().out

    assert "No dollar figure" in out
    # Never suppressed silently: the reason names the actual mechanism (the
    # cache_control lever specifically), not a bare "no data" shrug.
    assert "cache_control" in out


# --------------------------------------------------------------------------- #
# Snippet hygiene: copy-pasteable, unwrapped, no markup mangling
# --------------------------------------------------------------------------- #

def test_render_resend_cache_control_snippet_is_unmangled(db, capsys):
    from tokenjam.cli.cmd_optimize import _render_resend

    _seed_heavy_resend(db)
    _, finding = _run(db)
    assert finding.fix_cache_control

    _render_resend(finding, pricing_mode="api", marker="①", persona="sdk")
    out = capsys.readouterr().out

    assert '"cache_control"' in out
    assert '"type": "ephemeral"' in out


# --------------------------------------------------------------------------- #
# Persona branching: compaction (agent-harness) vs cache_control (SDK)
# --------------------------------------------------------------------------- #

def test_render_resend_claude_code_persona_leads_with_compaction(db, capsys):
    from tokenjam.cli.cmd_optimize import _render_resend

    _seed_heavy_resend(db)
    _, finding = _run(db)

    _render_resend(finding, pricing_mode="subscription", marker="①", persona="claude-code")
    out = _flat(capsys.readouterr().out)

    assert "Fix:" in out
    assert finding.fix_compaction in out
    # Secondary aside for the cache_control lever, not the lead.
    assert "If you also run SDK agents" in out


def test_render_resend_sdk_persona_leads_with_cache_control_snippet(db, capsys):
    from tokenjam.cli.cmd_optimize import _render_resend

    _seed_heavy_resend(db)
    _, finding = _run(db)

    _render_resend(finding, pricing_mode="api", marker="①", persona="sdk")
    out = capsys.readouterr().out

    assert "cache_control adoption" in out
    assert '"cache_control"' in out


def test_render_resend_mixed_persona_shows_both_labeled(db, capsys):
    from tokenjam.cli.cmd_optimize import _render_resend

    _seed_heavy_resend(db)
    _, finding = _run(db)

    _render_resend(finding, pricing_mode="api", marker="①", persona="mixed")
    out = _flat(capsys.readouterr().out)

    assert "Agent-harness sessions" in out
    assert "SDK sessions" in out
    assert finding.fix_compaction in out
    assert '"cache_control"' in out


# --------------------------------------------------------------------------- #
# End-to-end through _render_report: full dispatch, no generic empty state
# --------------------------------------------------------------------------- #

def test_render_report_surfaces_resend_instead_of_generic_empty(db, capsys):
    from tokenjam.cli.cmd_optimize import _render_report

    _seed_heavy_resend(db)
    report, finding = _run(db)
    assert finding.repeat_share is not None

    _render_report(report, agent=None, requested=["resend"], pricing_mode="api")
    out = capsys.readouterr().out

    assert "No candidates flagged" not in out
    assert "Context resend" in out
    assert "heavy" in out
