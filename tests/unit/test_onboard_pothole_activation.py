"""Onboarding tail: pothole surfacing + first-fix-enable CTA (#179).

Exercises ``_run_pothole_first_fix`` directly (mirrors the direct-call +
capsys pattern in ``test_onboard_first_run.py``'s nudge tests) rather than
driving the full onboard wizard — the apply/enable/revert mechanics
themselves are already covered by ``test_pothole_apply.py``; this file only
tests onboarding's orchestration: what gets shown, when the enable ask fires,
and that enforcement is never armed without an explicit confirm.
"""
from __future__ import annotations

import pytest

from tokenjam.cli import cmd_onboard
from tokenjam.core.optimize import pothole_apply
from tokenjam.core.optimize.analyzers.pothole import (
    PotholeCluster,
    PotholeExample,
    PotholeFinding,
)


def _cluster(**overrides) -> PotholeCluster:
    base = dict(
        signature="cwd_confusion", family_key="cwd_confusion",
        title="cwd / relative-path confusion", sessions=12, occurrences=324,
        repos=["demo"], rung=3, scope="project",
        proposed_fix="PostToolUseFailure hook: inject the real cwd on failure.",
        examples=[
            PotholeExample(session_id="s1", repo="demo", ts=None, snippet="no such file"),
        ],
        estimated_recoverable_tokens=486_000,
        suggested_target="/tmp/does-not-matter/.claude/hooks/cwd-confusion.py",
    )
    base.update(overrides)
    return PotholeCluster(**base)


def _finding(clusters, **overrides) -> PotholeFinding:
    base = dict(
        clusters=clusters, sessions_scanned=42,
        estimated_recoverable_tokens=sum(c.estimated_recoverable_tokens for c in clusters) or None,
    )
    base.update(overrides)
    return PotholeFinding(**base)


@pytest.fixture
def _no_op_apply(monkeypatch):
    """Fail loudly if a test that shouldn't reach apply/enable does anyway."""
    def _boom(*a, **k):
        raise AssertionError("apply_pothole_fix should not have been called")

    monkeypatch.setattr(pothole_apply, "apply_pothole_fix", _boom)
    monkeypatch.setattr(pothole_apply, "enable_enforcement", _boom)


def test_thin_history_shows_watching_note_and_never_applies(monkeypatch, capsys, _no_op_apply):
    import tokenjam.core.optimize.analyzers.pothole as pothole_mod

    monkeypatch.setattr(pothole_mod, "compute_pothole_finding", lambda **k: _finding([]))
    cmd_onboard._run_pothole_first_fix(object(), port=7391, want_daemon=True)
    out = capsys.readouterr().out
    assert "still watching" in out
    assert "The mistakes your agent keeps making" not in out


def test_no_hook_quality_candidate_routes_to_review_later(monkeypatch, capsys, _no_op_apply):
    import tokenjam.core.optimize.analyzers.pothole as pothole_mod

    low_confidence = _cluster(rung=1, family_key="edit_before_read", title="Edit before Read")
    distilled = _cluster(
        rung=3, family_key="distilled:some-slug", title="a distilled guess",
    )
    monkeypatch.setattr(
        pothole_mod, "compute_pothole_finding",
        lambda **k: _finding([low_confidence, distilled]),
    )
    monkeypatch.setattr(cmd_onboard, "_is_interactive", lambda: True)
    cmd_onboard._run_pothole_first_fix(object(), port=7391, want_daemon=True)
    out = capsys.readouterr().out
    assert "The mistakes your agent keeps making" in out
    assert "No day-1 hook-quality fix yet" in out
    assert "Enable this fix now" not in out


def test_non_interactive_shows_cta_but_never_prompts(monkeypatch, capsys, _no_op_apply):
    import tokenjam.core.optimize.analyzers.pothole as pothole_mod

    monkeypatch.setattr(pothole_mod, "compute_pothole_finding", lambda **k: _finding([_cluster()]))
    monkeypatch.setattr(cmd_onboard, "_is_interactive", lambda: False)
    cmd_onboard._run_pothole_first_fix(object(), port=7391, want_daemon=True)
    out = capsys.readouterr().out
    assert "cwd / relative-path confusion" in out
    assert "324" in out  # occurrences shown in the shock-stat list
    assert "Re-run `tj onboard --claude-code`" in out
    assert "Enable this fix now" not in out


def test_interactive_decline_skips_apply(monkeypatch, capsys, _no_op_apply):
    import tokenjam.core.optimize.analyzers.pothole as pothole_mod

    monkeypatch.setattr(pothole_mod, "compute_pothole_finding", lambda **k: _finding([_cluster()]))
    monkeypatch.setattr(cmd_onboard, "_is_interactive", lambda: True)
    monkeypatch.setattr(cmd_onboard.click, "confirm", lambda *a, **k: False)
    cmd_onboard._run_pothole_first_fix(object(), port=7391, want_daemon=True)
    out = capsys.readouterr().out
    assert "Your #1 fix:" in out
    assert "Skipped" in out


def test_interactive_confirm_applies_and_enables_with_explicit_confirm(
    monkeypatch, capsys,
):
    import tokenjam.core.optimize.analyzers.pothole as pothole_mod

    cluster = _cluster()
    monkeypatch.setattr(pothole_mod, "compute_pothole_finding", lambda **k: _finding([cluster]))
    monkeypatch.setattr(cmd_onboard, "_is_interactive", lambda: True)
    monkeypatch.setattr(cmd_onboard.click, "confirm", lambda *a, **k: True)

    calls = {}

    def _fake_apply(config, cluster_dict, *, target_path, scope, go, force):
        calls["apply"] = {
            "target_path": target_path, "scope": scope, "go": go, "force": force,
        }
        return {"dry_run": False, "record": {"id": "fix123"}}

    def _fake_enable(config, fix_id, *, confirm):
        calls["enable"] = {"fix_id": fix_id, "confirm": confirm}

    monkeypatch.setattr(pothole_apply, "apply_pothole_fix", _fake_apply)
    monkeypatch.setattr(pothole_apply, "enable_enforcement", _fake_enable)

    cmd_onboard._run_pothole_first_fix(object(), port=7391, want_daemon=True)
    out = capsys.readouterr().out

    # Evidence + the hook explainer print before the (mocked) confirm ask.
    assert "Evidence" in out
    assert "s1" in out  # repro session id from the example
    assert "What enabling does" in out
    assert out.index("Evidence") < out.index("What enabling does")

    # Applied with go=True and no silent force; enabled only via explicit confirm=True.
    assert calls["apply"]["go"] is True
    assert calls["apply"]["force"] is False
    assert calls["apply"]["target_path"] == cluster.suggested_target
    assert calls["enable"]["fix_id"] == "fix123"
    assert calls["enable"]["confirm"] is True

    assert "Enabled: cwd / relative-path confusion" in out
    assert "One-click revert" in out
    assert "#/review" in out


def test_apply_refused_never_silently_forces(monkeypatch, capsys):
    import tokenjam.core.optimize.analyzers.pothole as pothole_mod

    cluster = _cluster()
    monkeypatch.setattr(pothole_mod, "compute_pothole_finding", lambda **k: _finding([cluster]))
    monkeypatch.setattr(cmd_onboard, "_is_interactive", lambda: True)
    monkeypatch.setattr(cmd_onboard.click, "confirm", lambda *a, **k: True)

    def _refuse(*a, **k):
        raise pothole_apply.PotholeApplyRefused("an active session was seen")

    monkeypatch.setattr(pothole_apply, "apply_pothole_fix", _refuse)
    called_enable = []
    monkeypatch.setattr(
        pothole_apply, "enable_enforcement",
        lambda *a, **k: called_enable.append((a, k)),
    )

    cmd_onboard._run_pothole_first_fix(object(), port=7391, want_daemon=True)
    out = capsys.readouterr().out
    assert "Could not enable yet" in out
    assert "an active session was seen" in out
    assert not called_enable


def test_no_daemon_points_at_tj_serve_first(monkeypatch, capsys, _no_op_apply):
    import tokenjam.core.optimize.analyzers.pothole as pothole_mod

    low_confidence = _cluster(rung=1, family_key="edit_before_read")
    monkeypatch.setattr(pothole_mod, "compute_pothole_finding", lambda **k: _finding([low_confidence]))
    monkeypatch.setattr(cmd_onboard, "_is_interactive", lambda: True)
    cmd_onboard._run_pothole_first_fix(object(), port=7391, want_daemon=False)
    out = capsys.readouterr().out
    assert "run `tj serve`" in out
