"""Regression tests for the `tj --help` command short-descriptions (issue #644).

The command-list short help is what a first-run user sees on `tj --help`. It
must be concise (fits a standard terminal without `...` truncation) and read
in plain user-facing language, not internal jargon. These tests pin a few of
the rewritten descriptions and guard the length budget so a future edit can't
silently reintroduce a truncated, jargon-y line.
"""
from __future__ import annotations

import re

from click.testing import CliRunner

from tokenjam.cli.main import cli

# A sampling of the rewritten short-helps that must render verbatim.
EXPECTED_SHORT_HELPS = {
    "serve": "Run the local web UI (Lens).",
    "optimize": "Find cost-saving opportunities.",
    "onboard": "Set up tj (interactive).",
    "backfill": "Import past sessions from your agents.",
    "doctor": "Health-check your tj setup.",
}


def _command_help_lines() -> dict[str, str]:
    """Parse the `Commands:` block of `tj --help` into {name: short_help}."""
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("Commands:"))
    mapping: dict[str, str] = {}
    for ln in lines[start + 1:]:
        m = re.match(r"^  (\S+)\s\s+(\S.*)$", ln)
        if not m:
            break  # first non-command line ends the block
        mapping[m.group(1)] = m.group(2).strip()
    return mapping


def test_key_short_helps_render_verbatim() -> None:
    helps = _command_help_lines()
    for name, expected in EXPECTED_SHORT_HELPS.items():
        assert helps.get(name) == expected, f"{name}: {helps.get(name)!r}"


def test_no_short_help_is_truncated_by_click() -> None:
    # Click appends a literal '...' only when it truncates an over-long line.
    # Our `export` help intentionally ends in a literal ellipsis, so exclude it.
    helps = _command_help_lines()
    for name, short in helps.items():
        if name == "export":
            continue
        assert not short.endswith("..."), f"{name} help was truncated: {short!r}"
