"""Onboarding tail: the web dashboard review-inbox pointer.

Onboard used to run the relearn scan inline and drive an interactive
"enable this fix now?" ask. That whole section is gone: the scan was
compute-for-display only (it called ``compute_relearn_finding`` directly,
bypassing ``core.optimize.relearn_store``, so nothing was persisted for any
other surface to read), and the review inbox in the web dashboard is where
fixes are reviewed and applied now.

These tests pin the replacement: a couple of pointer lines, no scan, no
findings dump, and no prompt.
"""
from __future__ import annotations

import pytest

from tokenjam.cli import cmd_onboard


def test_pointer_sends_user_to_review_inbox(capsys):
    cmd_onboard._print_review_inbox_pointer(port=7391, want_daemon=True)
    out = capsys.readouterr().out
    assert "web dashboard" in out
    assert "http://127.0.0.1:7391/#/review" in out


def test_pointer_is_short(capsys):
    """A pointer, not a report: a couple of lines."""
    cmd_onboard._print_review_inbox_pointer(port=7391, want_daemon=True)
    out = capsys.readouterr().out
    assert len([ln for ln in out.splitlines() if ln.strip()]) <= 3


def test_no_daemon_points_at_tj_serve_first(capsys):
    cmd_onboard._print_review_inbox_pointer(port=7391, want_daemon=False)
    out = capsys.readouterr().out
    assert "run `tj serve`" in out
    assert "#/review" in out


def test_no_daemon_never_claims_tj_is_already_watching(capsys):
    """Regression guard: the relearn job that watches sessions only runs
    under `tj serve` -- under `--no-daemon` nothing is watching yet, and
    claiming otherwise would contradict the very next line telling the user
    to start the server."""
    cmd_onboard._print_review_inbox_pointer(port=7391, want_daemon=False)
    out = capsys.readouterr().out
    assert "keeps watching your sessions" not in out


def test_daemon_running_does_claim_tj_is_already_watching(capsys):
    cmd_onboard._print_review_inbox_pointer(port=7391, want_daemon=True)
    out = capsys.readouterr().out
    assert "keeps watching your sessions" in out


def test_pointer_never_prompts(monkeypatch, capsys):
    """The blocking "Enable this fix now?" ask is gone; onboard must not
    consult click.confirm from this tail at all."""
    def _boom(*a, **k):
        raise AssertionError("the onboard tail must not prompt")

    monkeypatch.setattr(cmd_onboard.click, "confirm", _boom)
    cmd_onboard._print_review_inbox_pointer(port=7391, want_daemon=True)
    out = capsys.readouterr().out
    assert "Enable this fix now" not in out


def test_pointer_never_scans_or_applies(monkeypatch, capsys):
    """No relearn compute, no apply/enable: the daemon's own background job
    owns both now."""
    import tokenjam.core.optimize.analyzers.relearn as relearn_mod
    from tokenjam.core.optimize import relearn_apply

    def _boom(*a, **k):
        raise AssertionError("the onboard tail must not compute or apply")

    monkeypatch.setattr(relearn_mod, "compute_relearn_finding", _boom)
    monkeypatch.setattr(relearn_apply, "apply_relearn_fix", _boom)
    monkeypatch.setattr(relearn_apply, "enable_enforcement", _boom)
    cmd_onboard._print_review_inbox_pointer(port=7391, want_daemon=True)


@pytest.mark.parametrize(
    "gone",
    [
        "Scanning your history for recurring mistakes",
        "The mistakes your agent keeps making",
        "estimated recoverable tokens across",
        "Your #1 fix:",
        "Evidence",
        "What enabling does",
        "Enable this fix now",
        "sessions scanned",
    ],
)
def test_removed_strings_are_gone_from_onboard(gone):
    import inspect

    src = inspect.getsource(cmd_onboard)
    assert gone not in src, gone
