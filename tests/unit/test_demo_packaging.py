"""Regression tests for `tj demo` scenario packaging (#291).

The bug: `incidents/` lived outside the `tokenjam/` package, so it never made it
into the wheel, and cmd_demo's discovery path resolved to a nonexistent
`site-packages/incidents/`. CI ran from the repo tree (where the dev-tree path
worked), so it never caught the broken install.

These tests assert the two halves of the fix:
1. the hatchling force-include mapping ships `incidents/` INTO the package, and
2. discovery resolves the PACKAGED location (not only the repo tree).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

from tokenjam.cli import cmd_demo

_EXPECTED_SLUGS = {"retry-loop", "surprise-cost", "hallucination-drift"}
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_pyproject_force_includes_incidents_into_package():
    """The wheel must ship `incidents/` as `tokenjam/incidents/` (#291)."""
    with open(_REPO_ROOT / "pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)
    force_include = (
        pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    )
    assert force_include.get("incidents") == "tokenjam/incidents", (
        "incidents/ must be force-included into the package so `tj demo` scenarios "
        "ship in pip/pipx installs (#291)."
    )


def test_discover_scenarios_returns_all_three():
    """Discovery finds every shipped scenario (here, via the repo-tree fallback)."""
    assert set(cmd_demo._discover_scenarios()) == _EXPECTED_SLUGS


def test_installed_candidate_is_tried_first():
    """The packaged path (`tokenjam/incidents`) is candidate #0, repo-root is #1."""
    candidates = cmd_demo._candidate_incidents_dirs()
    pkg_root = Path(cmd_demo.__file__).resolve().parent.parent  # …/tokenjam
    assert candidates[0] == pkg_root / "incidents"             # installed/wheel
    assert candidates[1] == pkg_root.parent / "incidents"      # dev-tree fallback


def test_discovery_resolves_packaged_layout(tmp_path, monkeypatch):
    """Simulate an installed wheel: scenarios under `tokenjam/incidents/`, with NO
    repo-root `incidents/`. This is the path CI never exercised (#291)."""
    source = cmd_demo._incidents_dir()
    assert source is not None  # the live scenarios dir (repo tree in CI)

    # Copy the scenarios into a packaged-style location: <tmp>/tokenjam/incidents.
    packaged = tmp_path / "tokenjam" / "incidents"
    shutil.copytree(source, packaged)

    # Point discovery ONLY at the packaged dir — the repo-root fallback is absent,
    # exactly like a real site-packages install.
    monkeypatch.setattr(cmd_demo, "_candidate_incidents_dirs", lambda: [packaged])

    assert cmd_demo._incidents_dir() == packaged
    assert set(cmd_demo._discover_scenarios()) == _EXPECTED_SLUGS


def test_discovery_empty_when_no_incidents_dir(monkeypatch, tmp_path):
    """No scenarios dir anywhere → empty (no crash), the graceful degrade path."""
    monkeypatch.setattr(
        cmd_demo, "_candidate_incidents_dirs", lambda: [tmp_path / "nope"]
    )
    assert cmd_demo._discover_scenarios() == {}
