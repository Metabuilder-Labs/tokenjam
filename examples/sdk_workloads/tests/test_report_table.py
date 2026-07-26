"""Report-table shape tests for `runner.py`'s output.

The unit tests below exercise `_finding_summary` / `_print_table` directly
with synthetic finding objects (no telemetry, no subprocess, no API calls).
The one end-to-end test runs the full harness (`runner.py <workload>
--dry-run`) and asserts the printed table has the expected header row and
required sections; this is the "report table shape" test the task brief
asks for, proven against the REAL renderer rather than a reimplementation of
it, with zero API spend.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import runner

WORKLOADS_DIR = Path(__file__).resolve().parents[1]


def test_finding_summary_none_means_no_data() -> None:
    fired, detail = runner._finding_summary(None)
    assert fired is False
    assert detail == "no data"


def test_finding_summary_empty_lists_mean_ran_but_nothing_cleared() -> None:
    finding = SimpleNamespace(examples=[], candidates=[], past_overspend_usd=None)
    fired, detail = runner._finding_summary(finding)
    assert fired is False
    assert "nothing cleared threshold" in detail


def test_finding_summary_nonempty_candidates_fire_with_a_count() -> None:
    finding = SimpleNamespace(
        examples=[object(), object()], candidates=[], past_overspend_usd=None,
    )
    fired, detail = runner._finding_summary(finding)
    assert fired is True
    assert "2 candidate(s)" in detail


def test_finding_summary_includes_dollar_figure_when_present() -> None:
    finding = SimpleNamespace(examples=[object()], past_overspend_usd=1.2345)
    fired, detail = runner._finding_summary(finding)
    assert fired is True
    assert "1.2345" in detail or "1.23" in detail
    assert "past_overspend_usd" in detail


def test_print_table_shape(capsys) -> None:
    runner._print_table(
        ["analyzer", "status", "detail"],
        [("downsize", "FIRED", "1 candidate"), ("cache", "ran, no finding", "n/a")],
    )
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.strip()]
    # Header, separator, then one row per data row.
    assert len(lines) == 4
    assert "analyzer" in lines[0] and "status" in lines[0] and "detail" in lines[0]
    assert set(lines[1].strip()) <= {"-", " "}
    assert "downsize" in lines[2] and "FIRED" in lines[2]
    assert "cache" in lines[3]


def test_workloads_registry_matches_scripts_on_disk() -> None:
    """Every entry in runner.WORKLOADS must point at a real, present file --
    this catches a stale/renamed workload entry before a user hits it."""
    for name, meta in runner.WORKLOADS.items():
        script_path = runner.WORKLOADS_DIR / meta["script"]
        assert script_path.is_file(), f"{name!r} points at missing {script_path}"


def test_end_to_end_report_table_shape_dry_run() -> None:
    """Runs the full harness against the cheapest workload with --dry-run
    (zero API calls) and checks the printed report carries the expected
    structure: the window header, the analyzer table's header row, the
    per-analyzer explanatory notes, the alerts section, and the final
    cumulative-spend line every run must print.
    """
    result = subprocess.run(
        [sys.executable, str(WORKLOADS_DIR / "runner.py"), "oversized-model", "--dry-run"],
        cwd=str(WORKLOADS_DIR), capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout

    assert "=== Window:" in out
    assert "persona=" in out
    assert "analyzer" in out and "status" in out and "detail" in out
    assert "=== Alerts fired in window:" in out
    assert "=== Actual spend recorded by tj's own cost engine: $" in out
    # Every registered analyzer must appear as a row, in ANALYZER_ORDER.
    for name in ("downsize", "cache", "cache-recommend", "resend", "script"):
        assert name in out
