"""Unit tests for the title distiller (core/distill.py).

These never invoke the real ``claude`` CLI — ``subprocess.run`` and the binary
resolution are monkeypatched. They cover: parsing a realistic ``claude`` JSON
envelope (including a fenced ``result``), graceful ``{}`` when ``claude`` is
missing or the call fails, and the session-keyed disk cache hit/miss behaviour.
"""
from __future__ import annotations

import json
import subprocess

from tokenjam.core import distill


def _envelope(result: str) -> str:
    """Build a realistic ``claude --output-format json`` stdout envelope."""
    return json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": result,
            "session_id": "abc123",
            "total_cost_usd": 0.03,
        }
    )


def _fenced(obj: dict) -> str:
    """Wrap a JSON object in a ```json fence, as the model often does."""
    return "```json\n" + json.dumps(obj) + "\n```"


def _patch_run(monkeypatch, *, returncode: int, stdout: str):
    """Patch ``subprocess.run`` to return a fixed CompletedProcess; record calls."""
    calls: list[dict] = []

    def fake_run(argv, **kwargs):
        calls.append({"argv": argv, "kwargs": kwargs})
        return subprocess.CompletedProcess(
            args=argv, returncode=returncode, stdout=stdout, stderr=""
        )

    monkeypatch.setattr(distill.subprocess, "run", fake_run)
    return calls


def _patch_claude_found(monkeypatch):
    """Make ``claude`` resolve to a fixed fake path."""
    monkeypatch.setattr(distill.shutil, "which", lambda _: "/fake/bin/claude")


# --- 1. parse a fenced result --------------------------------------------------


def test_distill_parses_fenced_json_result(monkeypatch):
    # Arrange
    _patch_claude_found(monkeypatch)
    result = _fenced({"1": "set up auth", "2": "fixed the test"})
    _patch_run(monkeypatch, returncode=0, stdout=_envelope(result))
    asks = [
        {"n": 1, "outcome": "I added the authentication middleware and wired it up."},
        {"n": 2, "outcome": "The failing unit test now passes after the fix."},
    ]

    # Act
    titles = distill.distill_titles(asks)

    # Assert
    assert titles == {1: "set up auth", 2: "fixed the test"}


def test_distill_invocation_uses_pinned_recipe(monkeypatch):
    # Arrange
    _patch_claude_found(monkeypatch)
    result = _fenced({"1": "did a thing"})
    calls = _patch_run(monkeypatch, returncode=0, stdout=_envelope(result))

    # Act
    distill.distill_titles([{"n": 1, "outcome": "did a thing in detail"}], model="haiku")

    # Assert — exact argv + stdin + neutral cwd, not a shell arg.
    assert len(calls) == 1
    argv = calls[0]["argv"]
    assert argv == [
        "/fake/bin/claude",
        "-p",
        "--output-format",
        "json",
        "--model",
        "haiku",
        "--disallowed-tools",
        "*",
    ]
    kwargs = calls[0]["kwargs"]
    assert "did a thing in detail" in kwargs["input"]
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["cwd"]  # neutral temp dir, non-empty


def test_distill_parses_unfenced_result(monkeypatch):
    # Arrange
    _patch_claude_found(monkeypatch)
    result = json.dumps({"3": "refactored module"})
    _patch_run(monkeypatch, returncode=0, stdout=_envelope(result))

    # Act
    titles = distill.distill_titles([{"n": 3, "outcome": "refactored the big module"}])

    # Assert
    assert titles == {3: "refactored module"}


def test_distill_drops_unknown_numbers(monkeypatch):
    # Arrange — model hallucinates an extra key the caller never asked about.
    _patch_claude_found(monkeypatch)
    result = _fenced({"1": "real title", "99": "phantom"})
    _patch_run(monkeypatch, returncode=0, stdout=_envelope(result))

    # Act
    titles = distill.distill_titles([{"n": 1, "outcome": "some outcome"}])

    # Assert
    assert titles == {1: "real title"}


# --- 2. claude missing ---------------------------------------------------------


def test_distill_returns_empty_when_claude_missing(monkeypatch, tmp_path):
    # Arrange — not on PATH and none of the fallback paths exist.
    monkeypatch.setattr(distill.shutil, "which", lambda _: None)
    monkeypatch.setattr(distill, "_CLAUDE_FALLBACK_PATHS", (tmp_path / "nope" / "claude",))

    def _should_not_run(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("subprocess.run must not be called when claude is missing")

    monkeypatch.setattr(distill.subprocess, "run", _should_not_run)

    # Act
    titles = distill.distill_titles([{"n": 1, "outcome": "x"}])

    # Assert
    assert titles == {}


# --- 3. failure modes ----------------------------------------------------------


def test_distill_returns_empty_on_nonzero_exit(monkeypatch):
    # Arrange
    _patch_claude_found(monkeypatch)
    _patch_run(monkeypatch, returncode=1, stdout="")

    # Act / Assert
    assert distill.distill_titles([{"n": 1, "outcome": "x"}]) == {}


def test_distill_returns_empty_on_garbage_stdout(monkeypatch):
    # Arrange — exit 0 but stdout isn't a JSON envelope.
    _patch_claude_found(monkeypatch)
    _patch_run(monkeypatch, returncode=0, stdout="not json at all")

    # Act / Assert
    assert distill.distill_titles([{"n": 1, "outcome": "x"}]) == {}


def test_distill_returns_empty_on_unparseable_result(monkeypatch):
    # Arrange — valid envelope, but result has no JSON object inside.
    _patch_claude_found(monkeypatch)
    _patch_run(monkeypatch, returncode=0, stdout=_envelope("just some prose, no object"))

    # Act / Assert
    assert distill.distill_titles([{"n": 1, "outcome": "x"}]) == {}


def test_distill_returns_empty_on_timeout(monkeypatch):
    # Arrange
    _patch_claude_found(monkeypatch)

    def _timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="claude", timeout=1)

    monkeypatch.setattr(distill.subprocess, "run", _timeout)

    # Act / Assert
    assert distill.distill_titles([{"n": 1, "outcome": "x"}]) == {}


def test_distill_empty_asks_skips_call(monkeypatch):
    # Arrange
    def _should_not_run(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("subprocess.run must not be called for empty asks")

    monkeypatch.setattr(distill.subprocess, "run", _should_not_run)

    # Act / Assert
    assert distill.distill_titles([]) == {}


# --- 4. caching ----------------------------------------------------------------


def test_cache_hit_avoids_second_call(monkeypatch, tmp_path):
    # Arrange — count distiller invocations.
    calls = {"n": 0}

    def fake_distill(asks, *, model="haiku", timeout=distill.DEFAULT_TIMEOUT):
        calls["n"] += 1
        return {1: "set up auth", 2: "fixed the test"}

    monkeypatch.setattr(distill, "distill_titles", fake_distill)
    asks = [{"n": 1, "outcome": "added auth"}, {"n": 2, "outcome": "fixed it"}]

    # Act — first call populates the cache, second call should hit it.
    first = distill.distill_titles_cached("sess-1", asks, cache_dir=tmp_path)
    second = distill.distill_titles_cached("sess-1", asks, cache_dir=tmp_path)

    # Assert
    assert first == {1: "set up auth", 2: "fixed the test"}
    assert second == first
    assert calls["n"] == 1
    assert (tmp_path / "sess-1.json").exists()


def test_peek_returns_cached_without_calling_claude(monkeypatch, tmp_path):
    calls = {"n": 0}

    def fake_distill(asks, *, model="haiku", timeout=distill.DEFAULT_TIMEOUT):
        calls["n"] += 1
        return {1: "set up auth"}

    monkeypatch.setattr(distill, "distill_titles", fake_distill)
    asks = [{"n": 1, "outcome": "added auth"}]

    # Nothing cached yet -> peek returns {} and never calls claude.
    assert distill.peek_cached_titles("s", asks, cache_dir=tmp_path) == {}
    assert calls["n"] == 0

    # Populate the cache, then peek hits it (still zero extra claude calls).
    distill.distill_titles_cached("s", asks, cache_dir=tmp_path)
    assert calls["n"] == 1
    assert distill.peek_cached_titles("s", asks, cache_dir=tmp_path) == {1: "set up auth"}
    assert calls["n"] == 1


def test_peek_miss_on_changed_outcome(monkeypatch, tmp_path):
    monkeypatch.setattr(distill, "distill_titles", lambda *a, **k: {1: "t"})
    distill.distill_titles_cached("s", [{"n": 1, "outcome": "original"}], cache_dir=tmp_path)
    # A changed outcome no longer matches the cache hash -> peek returns {}.
    assert distill.peek_cached_titles(
        "s", [{"n": 1, "outcome": "CHANGED"}], cache_dir=tmp_path) == {}


def test_cache_miss_on_changed_outcome_reinvokes(monkeypatch, tmp_path):
    # Arrange
    calls = {"n": 0}

    def fake_distill(asks, *, model="haiku", timeout=distill.DEFAULT_TIMEOUT):
        calls["n"] += 1
        return {1: f"title v{calls['n']}"}

    monkeypatch.setattr(distill, "distill_titles", fake_distill)

    # Act
    distill.distill_titles_cached("sess-1", [{"n": 1, "outcome": "original"}], cache_dir=tmp_path)
    distill.distill_titles_cached("sess-1", [{"n": 1, "outcome": "original"}], cache_dir=tmp_path)
    third = distill.distill_titles_cached(
        "sess-1", [{"n": 1, "outcome": "CHANGED outcome"}], cache_dir=tmp_path
    )

    # Assert — hash mismatch on the third call forces a re-invocation.
    assert calls["n"] == 2
    assert third == {1: "title v2"}


def test_cache_empty_result_not_persisted(monkeypatch, tmp_path):
    # Arrange — distiller fails (returns {}).
    monkeypatch.setattr(distill, "distill_titles", lambda *a, **k: {})

    # Act
    result = distill.distill_titles_cached("sess-1", [{"n": 1, "outcome": "x"}], cache_dir=tmp_path)

    # Assert — empty returned, nothing written.
    assert result == {}
    assert not (tmp_path / "sess-1.json").exists()


def test_cache_empty_result_does_not_clobber_good_cache(monkeypatch, tmp_path):
    # Arrange — first a good result is cached.
    monkeypatch.setattr(distill, "distill_titles", lambda *a, **k: {1: "good title"})
    distill.distill_titles_cached("sess-1", [{"n": 1, "outcome": "x"}], cache_dir=tmp_path)

    # Now the distiller starts failing, but inputs changed (forces a miss).
    monkeypatch.setattr(distill, "distill_titles", lambda *a, **k: {})

    # Act
    result = distill.distill_titles_cached(
        "sess-1", [{"n": 1, "outcome": "DIFFERENT"}], cache_dir=tmp_path
    )

    # Assert — empty returned, but the prior good cache file is untouched.
    assert result == {}
    cached = json.loads((tmp_path / "sess-1.json").read_text())
    assert cached["titles"] == {"1": "good title"}


def test_cache_corrupt_file_treated_as_miss(monkeypatch, tmp_path):
    # Arrange — a garbage cache file on disk.
    (tmp_path / "sess-1.json").write_text("{ this is not valid json")
    monkeypatch.setattr(distill, "distill_titles", lambda *a, **k: {1: "fresh"})

    # Act
    result = distill.distill_titles_cached("sess-1", [{"n": 1, "outcome": "x"}], cache_dir=tmp_path)

    # Assert — corrupt file ignored, distiller runs, cache overwritten.
    assert result == {1: "fresh"}
    cached = json.loads((tmp_path / "sess-1.json").read_text())
    assert cached["titles"] == {"1": "fresh"}
