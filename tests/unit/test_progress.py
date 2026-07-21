"""Unit tests for the shared delayed-start progress indicator.

`tj optimize` (and other analytics commands routed through
`tokenjam.cli.data_access`) can run 40-90s against a real Claude Code
history and printed nothing the whole time — this covers the suppression
matrix (non-TTY, --json, quiet, CI) that must never let a spinner frame leak
into machine-readable output, plus the sub-threshold silence case that keeps
every fast command exactly as quiet as it is today.
"""
from __future__ import annotations

import io
import time

import pytest
from rich.console import Console

from tokenjam.cli.progress import progress_disabled, progress_indicator


def _console(*, terminal: bool) -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return Console(file=buf, force_terminal=terminal, width=80), buf


# ─────────────────────────── progress_disabled matrix ──────────────────────

def test_disabled_when_output_json():
    console, _ = _console(terminal=True)
    assert progress_disabled(output_json=True, console=console) is True


def test_disabled_when_quiet():
    console, _ = _console(terminal=True)
    assert progress_disabled(quiet=True, console=console) is True


def test_disabled_when_ci_env_set(monkeypatch):
    monkeypatch.setenv("CI", "true")
    console, _ = _console(terminal=True)
    assert progress_disabled(console=console) is True


def test_disabled_when_stderr_not_a_tty(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    console, _ = _console(terminal=False)
    assert progress_disabled(console=console) is True


def test_not_disabled_when_terminal_and_no_suppression_flags(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    console, _ = _console(terminal=True)
    assert progress_disabled(output_json=False, quiet=False, console=console) is False


# ─────────────────────────── progress_indicator behavior ───────────────────

def test_disabled_indicator_renders_nothing_even_past_the_delay():
    console, buf = _console(terminal=True)
    with progress_indicator("Scanning...", disabled=True, console=console, delay=0.01):
        time.sleep(0.05)
    assert buf.getvalue() == ""


def test_sub_threshold_work_stays_silent():
    # Delay is longer than the work — the overwhelming majority of commands —
    # so nothing should ever render, matching today's silent behavior.
    console, buf = _console(terminal=True)
    with progress_indicator("Scanning...", disabled=False, console=console, delay=1.0):
        time.sleep(0.05)
    assert buf.getvalue() == ""


def test_slow_work_past_the_delay_renders_the_label():
    console, buf = _console(terminal=True)
    with progress_indicator("Scanning transcripts...", disabled=False, console=console, delay=0.02):
        time.sleep(0.1)
    assert "Scanning transcripts..." in buf.getvalue()


def test_update_changes_the_rendered_label():
    console, buf = _console(terminal=True)
    with progress_indicator("Step one...", disabled=False, console=console, delay=0.02) as handle:
        time.sleep(0.1)
        handle.update("Step two...")
        time.sleep(0.1)
    assert "Step two..." in buf.getvalue()


def test_update_before_delay_fires_is_a_safe_noop():
    # update() must not raise even when called before the spinner has ever
    # rendered anything (delay not yet elapsed, or the indicator is disabled).
    console, buf = _console(terminal=True)
    with progress_indicator("Scanning...", disabled=True, console=console, delay=1.0) as handle:
        handle.update("Something else...")
    assert buf.getvalue() == ""


def test_teardown_on_exception_clears_the_line_and_restores_cursor():
    console, buf = _console(terminal=True)
    with pytest.raises(ValueError):
        with progress_indicator("Scanning...", disabled=False, console=console, delay=0.02):
            time.sleep(0.1)
            raise ValueError("boom")
    out = buf.getvalue()
    assert "Scanning..." in out
    # Rich's transient teardown: show-cursor sequence present, meaning the
    # spinner was stopped cleanly rather than left stranded.
    assert "?25h" in out


def test_teardown_on_keyboard_interrupt_clears_the_line_and_restores_cursor():
    console, buf = _console(terminal=True)
    with pytest.raises(KeyboardInterrupt):
        with progress_indicator("Scanning...", disabled=False, console=console, delay=0.02):
            time.sleep(0.1)
            raise KeyboardInterrupt()
    out = buf.getvalue()
    assert "?25h" in out
