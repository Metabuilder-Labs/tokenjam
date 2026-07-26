"""Spend-guard tests. Zero API calls throughout: the unit tests exercise
`SpendGuard` in isolation, and the one end-to-end test runs a real workload
script with `--dry-run`, which never imports `openai` or touches the network
(see `_shared.build_client`).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from _shared import DEFAULT_MAX_SPEND_USD, SpendCeilingExceeded, SpendGuard

WORKLOADS_DIR = Path(__file__).resolve().parents[1]


def test_default_ceiling_is_low() -> None:
    """The mandatory spend guard defaults to a couple of dollars, not a
    number a runaway loop could burn through unnoticed."""
    assert 0 < DEFAULT_MAX_SPEND_USD <= 5.0


def test_check_before_call_allows_a_call_under_the_ceiling() -> None:
    guard = SpendGuard(ceiling_usd=1.0)
    guard.check_before_call(0.50)  # must not raise
    assert guard.total_spent_usd == 0.0  # check alone doesn't record spend
    assert guard.calls_made == 0


def test_check_before_call_blocks_before_the_call_is_made() -> None:
    """The guard must refuse BEFORE the call, not after; this is the
    difference between a spend guard and a spend logger."""
    guard = SpendGuard(ceiling_usd=0.01)
    with pytest.raises(SpendCeilingExceeded):
        guard.check_before_call(0.02)
    # A blocked call is never recorded as made.
    assert guard.calls_made == 0
    assert guard.total_spent_usd == 0.0


def test_cumulative_spend_across_calls_eventually_blocks() -> None:
    """Simulates several cheap calls that individually clear the ceiling but
    collectively exceed it; the guard must catch the CUMULATIVE total, not
    just judge each call in isolation."""
    guard = SpendGuard(ceiling_usd=1.00)

    for _ in range(3):
        guard.check_before_call(0.30)
        guard.record_actual(0.30)
    assert guard.total_spent_usd == pytest.approx(0.90)
    assert guard.calls_made == 3

    # A 4th call of the same size would push cumulative to $1.20 > $1.00.
    with pytest.raises(SpendCeilingExceeded) as excinfo:
        guard.check_before_call(0.30)
    assert "0.9" in str(excinfo.value) or "$0.9" in str(excinfo.value)
    # The blocked attempt must not be recorded as spent or made.
    assert guard.calls_made == 3
    assert guard.total_spent_usd == pytest.approx(0.90)


def test_ceiling_must_be_positive() -> None:
    with pytest.raises(ValueError):
        SpendGuard(ceiling_usd=0)
    with pytest.raises(ValueError):
        SpendGuard(ceiling_usd=-1.0)


def test_report_reflects_ceiling_and_spend() -> None:
    guard = SpendGuard(ceiling_usd=2.00)
    guard.check_before_call(0.10)
    guard.record_actual(0.10)
    text = guard.report()
    assert "0.10" in text or "0.1000" in text
    assert "2.00" in text
    assert "1" in text  # one call made


def test_workload_script_actually_aborts_when_the_guard_trips(tmp_path: Path) -> None:
    """End-to-end: run a real workload script with --dry-run and a ceiling so
    low the very first call must be blocked. Zero API calls: --dry-run never
    imports openai. This proves the guard is wired into the real call path,
    not just correct in isolation.

    A `@watch()`-wrapped run still creates one session-only span before the
    guard trips on the first LLM call, so this test points TJ_CONFIG/HOME at
    an isolated scratch dir (same technique runner.py uses) rather than let
    that span land in whatever tj store the test machine happens to have.
    """
    import os

    import runner as runner_module

    toml_path = runner_module._write_scratch_config(tmp_path, max_spend=0.0000001)
    env = os.environ.copy()
    env["TJ_CONFIG"] = str(toml_path)
    env["HOME"] = str(tmp_path / "home")

    script = WORKLOADS_DIR / "oversized_model.py"
    result = subprocess.run(
        [sys.executable, str(script), "--dry-run", "--max-spend", "0.0000001"],
        cwd=str(WORKLOADS_DIR), capture_output=True, text=True, timeout=60, env=env,
    )
    assert result.returncode == 2, (
        f"expected the spend guard to abort with exit code 2; "
        f"got {result.returncode}\nstderr:\n{result.stderr}"
    )
    assert "spend-guard" in result.stderr.lower()
    assert "aborting before call" in result.stderr.lower()
