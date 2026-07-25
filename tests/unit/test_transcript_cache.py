"""Unit tests for the persistent transcript parse cache
(core/transcript_cache.py).

No I/O beyond a ``tmp_path`` cache dir and ``tmp_path``-rooted transcript
files — mirrors test_deadweight.py / test_relearn.py's fixture style.
"""
from __future__ import annotations

import json
from pathlib import Path

from tokenjam.core import transcript_cache as tc


def _write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")


def test_cold_cache_parses_and_returns_records(tmp_path):
    src = tmp_path / "session.jsonl"
    _write(src, [{"type": "user", "message": {"content": "hi"}}])
    cache_dir = tmp_path / "cache"

    records = tc.cached_read_records(src, cache_dir)

    assert records == [{"type": "user", "message": {"content": "hi"}}]
    # A cache entry was actually written (not just an in-memory shortcut).
    assert list(cache_dir.glob("*.json"))


def test_warm_cache_skips_reparsing_the_source_file(tmp_path, monkeypatch):
    src = tmp_path / "session.jsonl"
    _write(src, [{"type": "user", "message": {"content": "hi"}}])
    cache_dir = tmp_path / "cache"

    first = tc.cached_read_records(src, cache_dir)

    # Any second call against an UNCHANGED file must be served from the cache
    # entry, never re-invoking the real parser — patch it to blow up if it's
    # called again so a regression here fails loudly instead of just slowly.
    def _boom(_path):
        raise AssertionError("_parse_records was called on a warm cache hit")

    monkeypatch.setattr("tokenjam.core.transcript._parse_records", _boom)

    second = tc.cached_read_records(src, cache_dir)
    assert second == first


def test_cache_invalidates_when_the_file_changes(tmp_path):
    src = tmp_path / "session.jsonl"
    _write(src, [{"type": "user", "message": {"content": "v1"}}])
    cache_dir = tmp_path / "cache"

    first = tc.cached_read_records(src, cache_dir)
    assert first[0]["message"]["content"] == "v1"

    # Mutate the source: both size and mtime change (a real edit/append, not
    # a no-op rewrite), so the cached (size, mtime) pair no longer matches.
    _write(src, [{"type": "user", "message": {"content": "v2-longer-content"}}])

    second = tc.cached_read_records(src, cache_dir)
    assert second[0]["message"]["content"] == "v2-longer-content"
    assert second != first


def test_cache_invalidates_on_mtime_change_even_at_same_size(tmp_path):
    """Two distinct edits can coincidentally leave the file the same size —
    the cache must still catch that via mtime, not just size."""
    src = tmp_path / "session.jsonl"
    _write(src, [{"type": "user", "message": {"content": "aaa"}}])
    cache_dir = tmp_path / "cache"

    tc.cached_read_records(src, cache_dir)

    import os

    st = src.stat()
    _write(src, [{"type": "user", "message": {"content": "bbb"}}])
    # Force a distinct mtime (filesystem mtime resolution can be coarse, and
    # the rewrite above may otherwise land in the same tick).
    os.utime(src, (st.st_atime, st.st_mtime + 5))
    st2 = src.stat()
    assert st.st_size == st2.st_size  # same size, different content

    second = tc.cached_read_records(src, cache_dir)
    assert second[0]["message"]["content"] == "bbb"


def test_cache_invalidates_on_equal_size_rewrite_with_unchanged_mtime(tmp_path):
    """An in-place rewrite that keeps both the byte count AND the mtime
    identical (e.g. two edits landing in the same filesystem mtime tick)
    must still be caught — size+mtime alone can't see it, so the content
    fingerprint is the only thing that can."""
    src = tmp_path / "session.jsonl"
    _write(src, [{"type": "user", "message": {"content": "aaa"}}])
    cache_dir = tmp_path / "cache"

    first = tc.cached_read_records(src, cache_dir)
    assert first[0]["message"]["content"] == "aaa"

    import os

    st = src.stat()
    _write(src, [{"type": "user", "message": {"content": "bbb"}}])
    assert src.stat().st_size == st.st_size  # same size by construction
    os.utime(src, (st.st_atime, st.st_mtime))  # force mtime back to identical

    second = tc.cached_read_records(src, cache_dir)
    assert second[0]["message"]["content"] == "bbb"
    assert second != first


def test_cache_invalidates_on_middle_only_rewrite_of_a_large_transcript(tmp_path):
    """The rewritten bytes can sit far from BOTH ends of the file: a large
    transcript whose middle record changes (identical size, identical mtime,
    identical first/last few KB) must still invalidate — an edge-sampling
    fingerprint would serve stale records here."""
    src = tmp_path / "session.jsonl"
    # Padding well past any head/tail sampling window on either side, so the
    # only differing bytes are deep in the middle of the file.
    pad = [{"type": "user", "message": {"content": "x" * 200}} for _ in range(100)]
    _write(src, pad + [{"type": "user", "message": {"content": "aaa"}}] + pad)
    assert src.stat().st_size > 2 * 8192
    cache_dir = tmp_path / "cache"

    first = tc.cached_read_records(src, cache_dir)
    assert first[100]["message"]["content"] == "aaa"

    import os

    st = src.stat()
    _write(src, pad + [{"type": "user", "message": {"content": "bbb"}}] + pad)
    assert src.stat().st_size == st.st_size  # same size by construction
    os.utime(src, (st.st_atime, st.st_mtime))  # force mtime back to identical

    second = tc.cached_read_records(src, cache_dir)
    assert second[100]["message"]["content"] == "bbb"


def test_unreadable_file_still_reuses_its_cache_entry(tmp_path, monkeypatch):
    """A file that stats fine but can't be read has no fingerprint to compare.
    The stored ``None`` must still validate, otherwise every lookup re-parses
    (to the same empty result) and rewrites the same unusable entry."""
    src = tmp_path / "session.jsonl"
    _write(src, [{"type": "user", "message": {"content": "hi"}}])
    cache_dir = tmp_path / "cache"

    monkeypatch.setattr(tc, "_fingerprint", lambda _path: None)
    first = tc.cached_read_records(src, cache_dir)

    def _boom(_path):
        raise AssertionError("_parse_records was called on an unchanged file")

    monkeypatch.setattr("tokenjam.core.transcript._parse_records", _boom)

    assert tc.cached_read_records(src, cache_dir) == first


def test_legacy_cache_entry_without_a_fingerprint_is_reparsed(tmp_path, monkeypatch):
    """Entries written before fingerprinting existed carry no ``fingerprint``
    key. They must NOT be mistaken for the unreadable-file case above just
    because a missing key reads back as ``None``."""
    src = tmp_path / "session.jsonl"
    _write(src, [{"type": "user", "message": {"content": "hi"}}])
    cache_dir = tmp_path / "cache"

    tc.cached_read_records(src, cache_dir)
    entry = next(cache_dir.glob("*.json"))
    legacy = json.loads(entry.read_text())
    legacy.pop("fingerprint")
    legacy["records"] = [{"stale": True}]
    entry.write_text(json.dumps(legacy), encoding="utf-8")

    # Even with a readable file whose fingerprint is unavailable, a keyless
    # legacy entry must lose to a fresh parse.
    monkeypatch.setattr(tc, "_fingerprint", lambda _path: None)
    assert tc.cached_read_records(src, cache_dir) == [
        {"type": "user", "message": {"content": "hi"}}
    ]


def test_missing_source_returns_empty_list_and_no_cache_write(tmp_path):
    src = tmp_path / "gone.jsonl"
    cache_dir = tmp_path / "cache"

    assert tc.cached_read_records(src, cache_dir) == []
    assert not cache_dir.exists() or not list(cache_dir.glob("*.json"))


def test_corrupt_cache_entry_falls_back_to_reparsing(tmp_path):
    src = tmp_path / "session.jsonl"
    _write(src, [{"type": "user", "message": {"content": "hi"}}])
    cache_dir = tmp_path / "cache"

    tc.cached_read_records(src, cache_dir)
    entry = next(cache_dir.glob("*.json"))
    entry.write_text("not json{{{", encoding="utf-8")

    # Must degrade to a fresh parse, not raise.
    records = tc.cached_read_records(src, cache_dir)
    assert records == [{"type": "user", "message": {"content": "hi"}}]


def test_prune_orphaned_entries_removes_only_dead_sources(tmp_path):
    alive = tmp_path / "alive.jsonl"
    dying = tmp_path / "dying.jsonl"
    _write(alive, [{"type": "user"}])
    _write(dying, [{"type": "user"}])
    cache_dir = tmp_path / "cache"

    tc.cached_read_records(alive, cache_dir)
    tc.cached_read_records(dying, cache_dir)
    assert len(list(cache_dir.glob("*.json"))) == 2

    dying.unlink()
    removed = tc.prune_orphaned_entries(cache_dir)

    assert removed == 1
    remaining = list(cache_dir.glob("*.json"))
    assert len(remaining) == 1
    assert json.loads(remaining[0].read_text())["path"] == str(alive)


def test_prune_orphaned_entries_on_missing_cache_dir_is_a_noop(tmp_path):
    assert tc.prune_orphaned_entries(tmp_path / "never-created") == 0


def test_default_cache_dir_honors_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("TJ_TRANSCRIPT_CACHE_DIR", str(tmp_path / "custom"))
    assert tc.default_cache_dir() == tmp_path / "custom"


def test_default_cache_dir_falls_back_to_home_tj_without_config(monkeypatch):
    monkeypatch.delenv("TJ_TRANSCRIPT_CACHE_DIR", raising=False)
    assert tc.default_cache_dir() == Path.home() / ".tj" / "transcript_cache"


def test_concurrent_writers_never_corrupt_the_cache_file(tmp_path):
    """Two 'processes' (simulated by calling the private writer twice with
    different pids-in-name) racing the same entry must never leave a
    partially-written / unparseable file behind — the atomic rename
    guarantees the last writer's COMPLETE payload wins."""
    src = tmp_path / "session.jsonl"
    _write(src, [{"type": "user", "message": {"content": "hi"}}])
    cache_dir = tmp_path / "cache"
    cache_path = cache_dir / tc._cache_key(src)

    tc._store(cache_path, src, 100, 1.0, "fp1", [{"a": 1}])
    tc._store(cache_path, src, 100, 1.0, "fp1", [{"a": 2}])

    loaded = tc._load(cache_path)
    assert loaded is not None
    assert loaded["records"] == [{"a": 2}]
