"""Unit tests for the append-only savings sink (`tokenjam.core.savings_log`)."""
from __future__ import annotations

from dataclasses import dataclass

from tokenjam.core.savings_log import (
    append_saving,
    hooks_dir,
    persist_output,
    read_savings,
    savings_path,
    summarize_savings,
)


@dataclass
class _Storage:
    path: str


@dataclass
class _Cfg:
    storage: _Storage


def _cfg(tmp_path):
    return _Cfg(storage=_Storage(path=str(tmp_path / "tj.duckdb")))


def test_sink_path_derives_from_storage_parent(tmp_path):
    cfg = _cfg(tmp_path)
    assert hooks_dir(cfg) == tmp_path / "hooks"
    assert savings_path(cfg) == tmp_path / "hooks" / "cap_output.jsonl"


def test_append_and_read_roundtrip(tmp_path):
    cfg = _cfg(tmp_path)
    append_saving(cfg, {"session_id": "a", "tool": "Bash", "saved_tok_est": 100,
                        "orig_tok_est": 500})
    append_saving(cfg, {"session_id": "b", "tool": "Grep", "saved_tok_est": 50,
                        "orig_tok_est": 200})
    events = read_savings(cfg)
    assert len(events) == 2
    assert events[0]["ts"]  # stamped automatically
    # session scoping
    assert len(read_savings(cfg, session_id="a")) == 1


def test_read_missing_sink_returns_empty(tmp_path):
    assert read_savings(_cfg(tmp_path)) == []


def test_read_tolerates_partial_line(tmp_path):
    cfg = _cfg(tmp_path)
    append_saving(cfg, {"session_id": "a", "tool": "Bash", "saved_tok_est": 10})
    # append a truncated/garbage line (append-only can race a reader)
    with open(savings_path(cfg), "a") as f:
        f.write('{"session_id": "b", "tool":\n')  # broken JSON
    events = read_savings(cfg)
    assert len(events) == 1  # broken line skipped, good one kept


def test_summarize_aggregates_by_tool_and_total(tmp_path):
    events = [
        {"tool": "Bash", "saved_tok_est": 100, "orig_tok_est": 400, "session_id": "s1"},
        {"tool": "Bash", "saved_tok_est": 50, "orig_tok_est": 200, "session_id": "s1"},
        {"tool": "Grep", "saved_tok_est": 30, "orig_tok_est": 90, "session_id": "s2"},
    ]
    s = summarize_savings(events)
    assert s["trims"] == 3
    assert s["saved_tok_est"] == 180
    assert s["by_tool"]["Bash"]["trims"] == 2
    assert s["by_tool"]["Bash"]["saved_tok_est"] == 150
    assert s["by_session"]["s1"] == 150


def test_persist_output_writes_full_text(tmp_path):
    cfg = _cfg(tmp_path)
    text = "hello\n" * 1000
    p = persist_output(cfg, "Bash", "sess-xyz", text)
    assert p is not None
    assert p.exists()
    assert p.read_text() == text
    assert p.parent == tmp_path / "hooks" / "outputs"


def test_append_saving_is_fail_safe_on_bad_config():
    # A config whose storage.path can't be resolved must not raise.
    @dataclass
    class _Bad:
        pass
    append_saving(_Bad(), {"tool": "Bash"})  # should silently no-op, not raise
