"""Unit tests for the pure trim policy (`tokenjam.core.output_cap`).

The policy is pure — no I/O, no config tree — so these tests use a lightweight
local stub config (the `_Cfg` dataclass) rather than the full `TjConfig`, in the
style of tests/unit/test_framing.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from tokenjam.core.output_cap import (
    CHARS_PER_TOKEN,
    est_tokens,
    is_test_build_command,
    trim,
)


@dataclass
class _Cfg:
    """Duck-typed stand-in for CapOutputConfig with small budgets for tests."""
    enabled:           bool = True
    budget_tokens:     int = 100          # ~400 bytes — tiny so fixtures stay small
    head_lines:        int = 5
    tail_lines:        int = 5
    smart_errors:      bool = True
    min_saving_tokens: int = 10
    killswitch:        bool = False
    tools:             list = field(default_factory=lambda: ["Bash", "Grep", "Glob", "WebFetch"])


def _big(nlines: int, width: int = 40) -> str:
    return "\n".join(f"line {i:04d} " + "x" * width for i in range(nlines))


# --- est_tokens / command classifier --------------------------------------

def test_est_tokens_is_char_over_four():
    assert est_tokens("x" * 400) == 100
    assert est_tokens("") == 0
    assert est_tokens("abc") == 0  # floor division


@pytest.mark.parametrize("cmd", [
    "pytest tests/",
    "python -m pytest -q",
    "npm run build",
    "go test ./...",
    "cargo build --release",
    "make all",
    "tsc --noEmit",
    "ruff check .",
    "mypy tokenjam/",
])
def test_is_test_build_command_true(cmd):
    assert is_test_build_command(cmd) is True


@pytest.mark.parametrize("cmd", [
    "ls -la",
    "grep -rn foo .",
    "echo hello",
    "cat file.txt",
    "",
])
def test_is_test_build_command_false(cmd):
    assert is_test_build_command(cmd) is False


# --- pass-through cases ----------------------------------------------------

def test_under_budget_passes_through():
    cfg = _Cfg()
    small = "hello world\n" * 3
    assert trim("Bash", {"command": "echo hi"}, small, cfg) is None


def test_ineligible_tool_passes_through():
    cfg = _Cfg()
    out = _big(500)
    assert trim("Read", {}, out, cfg) is None
    assert trim("Task", {}, out, cfg) is None


def test_disabled_passes_through():
    out = _big(500)
    assert trim("Bash", {"command": "x"}, out, _Cfg(enabled=False)) is None


def test_killswitch_passes_through():
    out = _big(500)
    assert trim("Bash", {"command": "x"}, out, _Cfg(killswitch=True)) is None


def test_empty_or_nonstring_output_passes_through():
    cfg = _Cfg()
    assert trim("Bash", {}, "", cfg) is None
    assert trim("Bash", {}, None, cfg) is None  # type: ignore[arg-type]


def test_min_saving_floor_blocks_marginal_trim():
    # Output just over budget by a hair, with a huge min-saving requirement →
    # not worth trimming.
    cfg = _Cfg(budget_tokens=100, min_saving_tokens=100_000)
    out = _big(500)
    assert trim("Bash", {"command": "echo x"}, out, cfg) is None


# --- the trimming path -----------------------------------------------------

def test_over_budget_keeps_head_tail_and_marker():
    cfg = _Cfg(head_lines=5, tail_lines=5)
    out = _big(500)
    res = trim("Bash", {"command": "echo x"}, out, cfg)
    assert res is not None
    assert res.saved_bytes > 0
    assert res.saved_tokens > 0
    assert res.kept_bytes < res.orig_bytes
    # first and last real lines survive
    assert "line 0000" in res.kept_text
    assert "line 0499" in res.kept_text
    # a middle line is gone
    assert "line 0250" not in res.kept_text
    # the transparency marker is present
    assert "tj cap-output" in res.kept_text
    assert "reclaimed" in res.kept_text


def test_marker_reports_trimmed_line_count():
    cfg = _Cfg(head_lines=5, tail_lines=5)
    out = _big(500)
    res = trim("Bash", {"command": "echo x"}, out, cfg)
    assert res is not None
    assert res.trimmed_lines == 500 - 5 - 5


def test_preserved_ref_appears_in_marker():
    cfg = _Cfg()
    out = _big(500)
    res = trim("Bash", {"command": "echo x"}, out, cfg,
               preserved_ref="/home/u/.local/share/tj/hooks/outputs/abc.txt")
    assert res is not None
    assert "/home/u/.local/share/tj/hooks/outputs/abc.txt" in res.kept_text


def test_no_preserved_ref_gives_rerun_hint():
    cfg = _Cfg()
    out = _big(500)
    res = trim("Bash", {"command": "echo x"}, out, cfg, preserved_ref=None)
    assert res is not None
    assert "re-run narrower" in res.kept_text


# --- smart-error mode ------------------------------------------------------

def _pytest_log(n_noise: int = 400) -> str:
    """Synthetic pytest log: lots of passing noise + a few failure lines."""
    noise = [f"tests/test_mod_{i}.py::test_{i} PASSED" for i in range(n_noise)]
    # bury failures in the MIDDLE (they'd be trimmed by plain head/tail)
    noise[200] = "tests/test_auth.py::test_login FAILED"
    noise[201] = "E   AssertionError: expected 200 got 401"
    noise[202] = "tests/test_db.py::test_query ERROR"
    return "\n".join(noise)


def test_smart_error_mode_keeps_failure_lines_from_middle():
    cfg = _Cfg(head_lines=5, tail_lines=5, smart_errors=True, budget_tokens=100)
    out = _pytest_log()
    res = trim("Bash", {"command": "pytest -q"}, out, cfg)
    assert res is not None
    # failures buried in the middle survive because it's a test command
    assert "test_login FAILED" in res.kept_text
    assert "AssertionError" in res.kept_text
    assert "test_query ERROR" in res.kept_text


def test_smart_error_off_drops_middle_failures():
    cfg = _Cfg(head_lines=5, tail_lines=5, smart_errors=False, budget_tokens=100)
    out = _pytest_log()
    res = trim("Bash", {"command": "pytest -q"}, out, cfg)
    assert res is not None
    # with smart-errors off, the buried middle failure is trimmed away
    assert "test_login FAILED" not in res.kept_text


def test_smart_error_not_applied_to_non_test_command():
    # Same log, but the command is not a test/build run → plain head/tail.
    cfg = _Cfg(head_lines=5, tail_lines=5, smart_errors=True, budget_tokens=100)
    out = _pytest_log()
    res = trim("Bash", {"command": "cat results.txt"}, out, cfg)
    assert res is not None
    assert "test_login FAILED" not in res.kept_text


def test_smart_error_not_applied_to_non_bash_tool():
    cfg = _Cfg(head_lines=5, tail_lines=5, smart_errors=True, budget_tokens=100)
    out = _pytest_log()
    # Grep is eligible but smart-error is Bash-only → plain head/tail
    res = trim("Grep", {}, out, cfg)
    assert res is not None
    assert "test_login FAILED" not in res.kept_text


# --- char-cap fallback (few huge lines) ------------------------------------

def test_char_cap_fallback_for_one_giant_line():
    cfg = _Cfg(budget_tokens=100, head_lines=5, tail_lines=5)
    out = "START" + "z" * 5000 + "END"   # one line, way over budget
    res = trim("WebFetch", {}, out, cfg)
    assert res is not None
    assert res.saved_bytes > 0
    assert res.kept_bytes < res.orig_bytes
    assert res.kept_text.startswith("START")
    assert res.kept_text.rstrip().endswith("END")
    assert "tj cap-output" in res.kept_text


# --- non-Bash eligible tool uses plain cap ---------------------------------

def test_grep_and_glob_and_webfetch_are_eligible():
    cfg = _Cfg()
    out = _big(500)
    for tool in ("Grep", "Glob", "WebFetch"):
        res = trim(tool, {}, out, cfg)
        assert res is not None, tool
        assert res.tool == tool
