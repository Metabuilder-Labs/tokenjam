"""Test-collection plumbing for the SDK workload corpus's own test suite.

Every workload module (and `_shared.py`) is written to run as a standalone
script; `python examples/sdk_workloads/repeated_prefix.py`; and imports its
shared plumbing with a bare `from _shared import ...`, which only resolves
when the script's own directory is on `sys.path` (true automatically when
Python runs it directly, since it adds the script's directory to
`sys.path[0]`). Pytest does not do that for us, so tests that import these
modules in-process need the workloads directory on `sys.path` explicitly.

This suite lives OUTSIDE `tests/unit` / `tests/synthetic` / `tests/agents` /
`tests/integration`; the repo's `[tool.pytest.ini_options] testpaths` in
pyproject.toml deliberately scopes CI's default `pytest` invocation to those
four core-product directories, so this additive corpus never becomes a
mandatory core-CI dependency. Run it explicitly:

    pytest examples/sdk_workloads/tests/
"""
from __future__ import annotations

import sys
from pathlib import Path

WORKLOADS_DIR = Path(__file__).resolve().parents[1]
if str(WORKLOADS_DIR) not in sys.path:
    sys.path.insert(0, str(WORKLOADS_DIR))
