"""Every workload in the corpus must run clean through the full harness in
--dry-run, end to end, with zero API calls. This is the corpus-wide
regression test: a workload that raises, hangs, or silently drops telemetry
in dry-run would be caught here before ever costing anyone real money.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import runner

WORKLOADS_DIR = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("workload_name", sorted(runner.WORKLOADS))
def test_workload_dry_runs_clean_through_the_harness(workload_name: str) -> None:
    result = subprocess.run(
        [sys.executable, str(WORKLOADS_DIR / "runner.py"), workload_name, "--dry-run"],
        cwd=str(WORKLOADS_DIR), capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, (
        f"{workload_name} exited {result.returncode} in --dry-run\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "cumulative spend: $0.0" in result.stdout or "cumulative spend: $" in result.stdout
    assert "Actual spend recorded by tj's own cost engine: $" in result.stdout


@pytest.mark.parametrize("script_name", [w["script"] for w in runner.WORKLOADS.values()])
def test_workload_script_help_never_touches_the_network(script_name: str) -> None:
    """`--help` must exit 0 with zero setup; a sanity check that argparse
    wiring alone (no client construction) doesn't accidentally require a key
    or attempt any I/O.
    """
    result = subprocess.run(
        [sys.executable, str(WORKLOADS_DIR / script_name), "--help"],
        cwd=str(WORKLOADS_DIR), capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0
    assert "--dry-run" in result.stdout
    assert "--max-spend" in result.stdout
