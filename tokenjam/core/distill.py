"""Distill crisp titles for a session's asks via the user's local ``claude`` CLI.

A session's work map (``core.workmap`` / ``core.transcript``) gives each ask a
deterministic *outcome* — the last assistant narration of that exchange. That
text is faithful but often long and verbose. This module turns each outcome into
a short, scannable title ("set up auth", "fixed the failing test") so the UI can
label a long run at a glance.

It deliberately holds **no API key of its own**. Instead it shells out to the
user's already-authenticated ``claude`` CLI, reusing their Claude subscription /
key. The invocation is pinned to the cheapest path that works:

  * ``claude -p --output-format json --model haiku --disallowed-tools '*'``
  * run from a neutral temp ``cwd`` so it loads no project ``CLAUDE.md`` / MCP
    (keeps the input tiny and the call cheap — roughly $0.03 / ~5s),
  * prompt passed on **stdin** (never as a shell arg).

``claude`` returns a JSON envelope on stdout; its ``result`` field is the model's
answer, which may be wrapped in a ```` ```json … ``` ```` fence. We extract the
first ``{…}`` block from that and parse it.

Everything is best-effort: every entry point returns ``{}`` on any failure
(``claude`` missing, non-zero exit, timeout, unparseable output) and never
raises. Titles are advisory chrome, not load-bearing data.

Pure module: no I/O beyond the subprocess + cache file; never imports
``tokenjam.api`` / ``tokenjam.cli``.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

#: Max words a distilled title may contain (instructed to the model).
MAX_TITLE_WORDS = 6

#: Each outcome is truncated to this many chars before being sent, to bound the
#: prompt size (and therefore cost) regardless of how chatty a run was.
MAX_OUTCOME_CHARS = 400

#: Default wall-clock budget for the ``claude`` call. Generous: a cold CLI start
#: plus a Haiku round-trip is ~5s, but launchd / first-run paths can be slower.
DEFAULT_TIMEOUT = 180

#: Probed in order when ``claude`` is not on ``PATH`` — the tj daemon runs under
#: launchd / systemd with a minimal ``PATH``, so the bare name often misses.
_CLAUDE_FALLBACK_PATHS = (
    Path.home() / ".claude" / "local" / "claude",
    Path("/opt/homebrew/bin/claude"),
    Path("/usr/local/bin/claude"),
    Path.home() / ".local" / "bin" / "claude",
    Path.home() / ".npm-global" / "bin" / "claude",
)

#: Matches the first ``{...}`` block in a string (``re.S`` so it spans newlines).
#: The model may wrap its JSON answer in a ```` ```json … ``` ```` fence; this
#: pulls the object out without caring about the fence.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.S)


def _resolve_claude() -> str | None:
    """Return an absolute path to the ``claude`` CLI, or ``None`` if not found.

    Tries ``PATH`` first, then a handful of common install locations so the call
    still works from a minimal-``PATH`` daemon environment.
    """
    found = shutil.which("claude")
    if found:
        return found
    for candidate in _CLAUDE_FALLBACK_PATHS:
        if candidate.is_file():
            return str(candidate)
    return None


def _build_prompt(asks: list[dict]) -> str:
    """Build the distillation prompt from numbered ask outcomes.

    Each outcome is truncated to :data:`MAX_OUTCOME_CHARS`. The model is asked to
    return only a JSON object mapping the (string) number to a crisp title.
    """
    lines = []
    for ask in asks:
        outcome = str(ask.get("outcome", "")).strip()
        if len(outcome) > MAX_OUTCOME_CHARS:
            outcome = outcome[:MAX_OUTCOME_CHARS] + "…"
        lines.append(f"{ask['n']}. {outcome}")
    numbered = "\n".join(lines)
    return (
        "Below are numbered outcomes, each describing what an AI coding agent did "
        "in one exchange of a session.\n"
        f"For each number, write a crisp title of AT MOST {MAX_TITLE_WORDS} words "
        "describing WHAT THE AGENT DID. Use past tense, plain text only — no "
        "markdown, no surrounding quotes.\n"
        "Return ONLY a JSON object mapping the number (as a string) to its title, "
        'e.g. {"1": "set up auth", "2": "fixed the failing test"}. No prose, no '
        "code fence required.\n\n"
        f"{numbered}"
    )


def _extract_titles(result: str, valid_ns: set[int]) -> dict[int, str]:
    """Parse the model's ``result`` string into ``{n: title}`` int-keyed dict.

    Extracts the first ``{...}`` block (tolerating a code fence), parses it, and
    keeps only entries whose key is a known ask number with a non-empty title.
    Returns ``{}`` if nothing usable can be parsed.
    """
    match = _JSON_OBJECT_RE.search(result)
    if not match:
        return {}
    try:
        raw = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}

    titles: dict[int, str] = {}
    for key, value in raw.items():
        try:
            n = int(key)
        except (TypeError, ValueError):
            continue
        if n not in valid_ns:
            continue
        title = str(value).strip().strip('"').strip()
        if title:
            titles[n] = title
    return titles


def _invoke_claude(prompt: str, *, model: str, timeout: int) -> str | None:
    """Shell out to the local ``claude`` CLI with ``prompt`` on stdin; return its
    ``result`` string, or ``None`` on any failure (missing CLI, non-zero exit,
    timeout, unparseable envelope). Never raises.

    Shared by every distill entry point (title distillation, relearn-cluster
    naming, …) so the pinned invocation recipe — cheapest model path, neutral
    cwd, stdin-not-argv prompt — lives in exactly one place.
    """
    claude_bin = _resolve_claude()
    if claude_bin is None:
        return None

    argv = [
        claude_bin,
        "-p",
        "--output-format",
        "json",
        "--model",
        model,
        "--disallowed-tools",
        "*",
    ]

    try:
        # Neutral cwd so the CLI loads no project CLAUDE.md / MCP config, keeping
        # the input tiny and the call cheap.
        proc = subprocess.run(
            argv,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
            cwd=tempfile.gettempdir(),
        )
    except (subprocess.TimeoutExpired, OSError):
        return None

    if proc.returncode != 0 or not proc.stdout:
        return None

    try:
        envelope = json.loads(proc.stdout)
    except (json.JSONDecodeError, ValueError):
        return None

    result = envelope.get("result") if isinstance(envelope, dict) else None
    return result if isinstance(result, str) else None


def distill_titles(
    asks: list[dict],
    *,
    model: str = "haiku",
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[int, str]:
    """Distill a crisp title for each ask outcome via the local ``claude`` CLI.

    Args:
        asks: ``[{"n": int, "outcome": str}, ...]`` — only work asks with
            non-empty outcomes; the caller is responsible for that filtering.
        model: ``claude`` model alias to use (default ``"haiku"`` — cheapest).
        timeout: wall-clock budget in seconds for the CLI call.

    Returns:
        ``{n: title}`` (int keys). Returns ``{}`` on any failure — ``claude`` not
        found, non-zero exit, timeout, or unparseable output. Never raises.
    """
    if not asks:
        return {}

    prompt = _build_prompt(asks)
    valid_ns = {int(ask["n"]) for ask in asks}

    result = _invoke_claude(prompt, model=model, timeout=timeout)
    if result is None:
        return {}

    return _extract_titles(result, valid_ns)


def _cache_signature(asks: list[dict], model: str) -> str:
    """SHA-256 of a stable serialization of the asks + model.

    The cache hit/miss decision keys on ``(n, outcome)`` pairs and the model, so
    a changed outcome (or model) re-invokes the distiller.
    """
    payload = {
        "model": model,
        "asks": [[int(ask["n"]), str(ask.get("outcome", ""))] for ask in asks],
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _default_cache_dir() -> Path:
    """Default on-disk cache location: ``~/.tj/distill_cache``."""
    return Path.home() / ".tj" / "distill_cache"


def distill_titles_cached(
    session_id: str,
    asks: list[dict],
    *,
    model: str = "haiku",
    cache_dir: Path | None = None,
) -> dict[int, str]:
    """Cached wrapper around :func:`distill_titles`, keyed by session + inputs.

    The cache file ``<cache_dir>/<session_id>.json`` stores
    ``{"hash": <signature>, "titles": {n: title}}``. On a hash match the cached
    titles are returned without calling the CLI. On a miss the distiller runs;
    a non-empty result is written back, while an empty result returns ``{}``
    without overwriting an existing good cache.

    Missing or corrupt cache files are treated as a miss. Never raises.
    """
    if cache_dir is None:
        cache_dir = _default_cache_dir()

    signature = _cache_signature(asks, model)
    cache_file = cache_dir / f"{session_id}.json"

    cached = _read_cache(cache_file)
    if cached is not None and cached.get("hash") == signature:
        return _coerce_titles(cached.get("titles"))

    titles = distill_titles(asks, model=model)
    if not titles:
        # Don't clobber a previously-good cache with an empty (failed) result.
        return {}

    _write_cache(cache_file, signature, titles)
    return titles


def peek_cached_titles(
    session_id: str,
    asks: list[dict],
    *,
    model: str = "haiku",
    cache_dir: Path | None = None,
) -> dict[int, str]:
    """Return cached titles for these asks **without** ever calling the CLI.

    Like :func:`distill_titles_cached` but cache-only: on a hash match it returns
    the cached titles, and on a miss (no cache, stale cache, or unreadable) it
    returns ``{}`` — it never shells out to ``claude``. Used to auto-apply an
    already-distilled session on load, so the user presses the button once and it
    sticks, at zero cost. Never raises.
    """
    if cache_dir is None:
        cache_dir = _default_cache_dir()
    signature = _cache_signature(asks, model)
    cached = _read_cache(cache_dir / f"{session_id}.json")
    if cached is not None and cached.get("hash") == signature:
        return _coerce_titles(cached.get("titles"))
    return {}


def _read_cache(cache_file: Path) -> dict | None:
    """Load a cache file, returning ``None`` on any read/parse failure."""
    try:
        raw = json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    return raw if isinstance(raw, dict) else None


def _write_cache(cache_file: Path, signature: str, titles: dict[int, str]) -> None:
    """Write the cache file; swallow I/O errors (cache is best-effort)."""
    payload = {
        "hash": signature,
        "titles": {str(n): title for n, title in titles.items()},
    }
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
    except OSError:
        pass


def _coerce_titles(raw: object) -> dict[int, str]:
    """Coerce a cached ``titles`` mapping back to ``{int: str}``."""
    if not isinstance(raw, dict):
        return {}
    titles: dict[int, str] = {}
    for key, value in raw.items():
        try:
            n = int(key)
        except (TypeError, ValueError):
            continue
        title = str(value).strip()
        if title:
            titles[n] = title
    return titles
