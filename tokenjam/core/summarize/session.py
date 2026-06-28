"""Prep / check lifecycle for structure-aware summarization (no scratch, no LLM here).

`prepare(path)` reads a prompt, wraps its structure (`wrap.protect`), and returns the
wrapped prompt + the summarizer contract + a `source_sha256` of the file — it persists
**nothing** (DEC-024). `check(config, path, summary, source_hash)` re-reads the file,
**refuses if it changed or vanished since prep** (the hash guard), re-runs the
deterministic wrap to regenerate the structure map, restores the model's summary, and —
when structure survives — **stages** the result under `~/.tj/summary/results/` for review.

No hidden handle/scratch and no LLM call: the model runs in the client, the file on disk
is the source of truth, and the only persisted artifact is the *visible* staged result
(plus, in PR3, the backup). Applying the staged results is PR3 (DEC-025/026).
"""
from __future__ import annotations

import difflib
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from tokenjam.core.config import TjConfig
from tokenjam.core.summarize import detect, wrap
from tokenjam.core.summarize.estimate import DEFAULT_TARGET_RATIO


class SummarizeRefused(Exception):
    """File changed or vanished since prep — re-prep before check/apply (carries the house-voice text)."""


CHECK_NOTE = (
    "Structure is restored verbatim; only the prose was rewritten. Review the diff "
    "before adopting — must-keep word changes are flagged, not blocked."
)
GATE_FAIL_NOTE = (
    "Structure check failed — not staged. The prompt can't be safely compressed as summarized."
)


@dataclass(frozen=True)
class PrepResult:
    """What `prepare()` hands the client — persists nothing. `wrapped_prompt` is empty
    when the prompt is below the worth-it prose gate (`note` says so)."""

    path: str
    source_sha256: str
    wrapped_prompt: str
    system_rules: str
    prose_words: int
    target_prose_words: int
    protected_blocks: int
    plan: list[dict]
    note: str

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "source_sha256": self.source_sha256,
            "wrapped_prompt": self.wrapped_prompt,
            "system_rules": self.system_rules,
            "prose_words": self.prose_words,
            "target_prose_words": self.target_prose_words,
            "protected_blocks": self.protected_blocks,
            "plan": self.plan,
            "note": self.note,
        }


@dataclass(frozen=True)
class CheckVerdict:
    """The headless verdict. Structure is the only hard gate; must-keep word movement is
    reported for review, never used to reject. `staged` is True iff it was written to the
    results dir (only structure-ok results are staged)."""

    path: str
    structure_ok: bool
    reason: str
    integrity: dict
    words_before: int
    words_after: int
    est_tokens_saved: int
    must_keep_removed: list[str]
    must_keep_added: list[str]
    diff: str
    restored: str
    staged: bool
    produced_by: str
    note: str

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "structure_ok": self.structure_ok,
            "reason": self.reason,
            "integrity": self.integrity,
            "words_before": self.words_before,
            "words_after": self.words_after,
            "est_tokens_saved": self.est_tokens_saved,
            "must_keep_removed": self.must_keep_removed,
            "must_keep_added": self.must_keep_added,
            "diff": self.diff,
            "restored": self.restored,
            "staged": self.staged,
            "produced_by": self.produced_by,
            "note": self.note,
        }


# --------------------------------------------------------------------------- #
# Hashing + file IO (prep and check MUST read identically so the hash matches)
# --------------------------------------------------------------------------- #

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read(path: str) -> str:
    return Path(path).expanduser().read_text(encoding="utf-8")


def _refuse_symlink(path: str) -> None:
    """Summarize won't rewrite through a symlink — the write could land outside where you expect."""
    if Path(path).expanduser().is_symlink():
        raise SummarizeRefused(
            f"{path} is a symlink — summarize won't rewrite through links (the write could land "
            f"outside where you expect). Point it at the real file.")


# --------------------------------------------------------------------------- #
# Results staging dir — DEC-026: <storage-parent>/summary/results/ (default ~/.tj/summary/)
# --------------------------------------------------------------------------- #

def summary_root(config: TjConfig) -> Path:
    """The durable summarize anchor next to the storage DB (`~/.tj/summary/` by default)."""
    sp = config.storage.path
    base = Path.home() / ".tj" if sp in ("", ":memory:") else Path(sp).expanduser().parent
    return base / "summary"


def results_dir(config: TjConfig) -> Path:
    return summary_root(config) / "results"


def stage_key(path: str) -> str:
    """Collision-free key for a source file: sha256 of its resolved absolute path."""
    return sha256(str(Path(path).expanduser().resolve()))


def stage(config: TjConfig, verdict: CheckVerdict, source_hash: str) -> Path:
    """Write a verified result + metadata under the results dir; return the staged file."""
    d = results_dir(config)
    d.mkdir(parents=True, exist_ok=True)
    payload = verdict.to_dict()
    payload["source_sha256"] = source_hash            # apply re-checks the original against this
    f = d / f"{stage_key(verdict.path)}.json"
    f.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return f


def list_staged(config: TjConfig) -> list[dict]:
    """Every staged result (for the take-all CLI and the future Lens results picker)."""
    d = results_dir(config)
    if not d.exists():
        return []
    out: list[dict] = []
    for f in sorted(d.glob("*.json")):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue                                  # skip a half-written/corrupt entry
    return out


def read_staged(config: TjConfig, path: str) -> dict | None:
    f = results_dir(config) / f"{stage_key(path)}.json"
    if not f.exists():
        return None
    parsed: dict = json.loads(f.read_text(encoding="utf-8"))
    return parsed


def clear(config: TjConfig, path: str | None = None) -> int:
    """Remove one staged result (by source path) or all; return the count removed."""
    d = results_dir(config)
    if not d.exists():
        return 0
    files = [d / f"{stage_key(path)}.json"] if path is not None else list(d.glob("*.json"))
    removed = 0
    for f in files:
        try:
            f.unlink()
            removed += 1
        except FileNotFoundError:
            continue
    return removed


# --------------------------------------------------------------------------- #
# prep / check
# --------------------------------------------------------------------------- #

def prepare(
    *,
    path: str,
    ratio: float = DEFAULT_TARGET_RATIO,
    hide_if_chars: int = wrap.HIDE_IF_CHARS,
    min_prose_words: int = detect.MIN_PROSE_WORDS,
) -> PrepResult:
    """Wrap a prompt's structure; return the wrapped prompt + rules + source hash. Persists nothing."""
    _refuse_symlink(path)
    content = _read(path)
    source_hash = sha256(content)
    breakdown = detect.analyze(content)
    n_prose = breakdown.prose_words
    if n_prose < min_prose_words:
        return PrepResult(
            path=path, source_sha256=source_hash, wrapped_prompt="", system_rules="",
            prose_words=n_prose, target_prose_words=0,
            protected_blocks=breakdown.protected_blocks, plan=[],
            note=f"Only {n_prose} prose words (< {min_prose_words}-word gate) — not worth summarizing.",
        )
    wrapped, _saved, _order, plan = wrap.protect(content, hide_if_chars)
    target = max(8, int(ratio * n_prose))
    return PrepResult(
        path=path, source_sha256=source_hash, wrapped_prompt=wrapped,
        system_rules=wrap.WRAP_SUMM_SYS.format(n=target),
        prose_words=n_prose, target_prose_words=target,
        protected_blocks=breakdown.protected_blocks, plan=plan, note="",
    )


def check(config: TjConfig, path: str, summary: str, source_hash: str,
          *, produced_by: str = "manual") -> CheckVerdict:
    """Hash-guard the file, re-derive the map, restore the summary; stage it when structure holds.

    ``produced_by`` records who made the summary (manual / claude-p / api / in-session) on the staged
    result — the seam the Lens UI uses for mode-aware gating later (DEC-028). Raises
    ``SummarizeRefused`` (house-voice) if the file vanished or changed since prep — the summary was
    built against a different version, so there's nothing safe to do.
    """
    _refuse_symlink(path)
    p = Path(path).expanduser()
    if not p.is_file():
        raise SummarizeRefused(
            f"{path} not found (moved or deleted since prep) — re-run `tj summarize prep`.")
    p = p.resolve()   # canonical absolute — `apply` must not reinterpret a relative path against its cwd
    current = p.read_text(encoding="utf-8")
    if sha256(current) != source_hash:
        raise SummarizeRefused(
            f"{path} changed since `tj summarize prep` — re-run prep, then check.")

    # File matches prep → `protect` is deterministic, so this regenerates the exact map.
    _wrapped, saved, order, _plan = wrap.protect(current)
    prose = detect.prose_text(current)        # the unprotected text the model was given to rewrite
    source_sentinels = prose.count("<tj-keep") + prose.count("</tj-keep>")
    restored, integ = wrap.restore(summary, saved, order, source_sentinels=source_sentinels)
    ok = wrap.is_structure_ok(integ)
    removed, added = wrap.crit_delta(current, restored)
    wb, wa = wrap.word_count(current), wrap.word_count(restored)
    est_saved = max(0, (len(current) - len(restored)) // detect.CHARS_PER_TOKEN)
    diff = "".join(difflib.unified_diff(
        current.splitlines(keepends=True), restored.splitlines(keepends=True),
        fromfile="original", tofile="summarized", n=2))

    verdict = CheckVerdict(
        path=str(p), structure_ok=ok, reason="" if ok else wrap.integrity_reason(integ),
        integrity=integ, words_before=wb, words_after=wa, est_tokens_saved=est_saved,
        must_keep_removed=removed, must_keep_added=added, diff=diff, restored=restored,
        staged=ok, produced_by=produced_by, note=CHECK_NOTE if ok else GATE_FAIL_NOTE,
    )
    if ok:
        stage(config, verdict, source_hash)
    return verdict
