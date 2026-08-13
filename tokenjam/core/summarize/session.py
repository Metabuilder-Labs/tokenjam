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
from tokenjam.core.summarize import detect, load_semantics, wrap
from tokenjam.core.summarize import route as route_mod
from tokenjam.core.summarize.estimate import (
    DEFAULT_TARGET_RATIO,
    PUBLISHED_LINE_TARGET,
    PUBLISHED_LINE_TARGET_QUOTE,
    line_target_prose_words,
)
from tokenjam.core.summarize.route import recommend_route


class SummarizeRefused(Exception):
    """Refusing to summarize/apply a file (carries the house-voice text): it changed or
    vanished since prep, isn't valid UTF-8, is a symlink, or otherwise can't be processed.
    Callers surface it as a 409 / re-prep prompt."""


CHECK_NOTE = (
    "Structure is restored verbatim; only the prose was rewritten. Review the diff "
    "before adopting — must-keep word changes are flagged, not blocked."
)
GATE_FAIL_NOTE = (
    "Structure check failed — not staged. The prompt can't be safely compressed as summarized."
)

#: Why the rewriter is being asked for THIS word budget, for a file the
#: published size guidance does NOT cover. States plainly that the ratio is
#: tokenjam's own and that nothing enforces it. For an always-resident
#: instruction file the basis comes from `route.recommend_route` instead, which
#: names all four routes to the published target and the quality tax that makes
#: compression the worst of them for such a file.
TARGET_BASIS_RATIO = (
    "Aiming to keep about {pct:.0f}% of the prose. That ratio is tokenjam's own "
    "target, not published guidance, and nothing enforces it: whatever the "
    "rewrite returns is accepted so long as the structure survives. Only the "
    "token reduction is claimed as a saving; whether the rewrite preserved your "
    "meaning is not checked by anything automatic, so read the diff."
)
#: Prefixed to the route advice when the line target is what set the budget, so
#: the number in `target_prose_words` is traceable to the target it came from.
TARGET_BASIS_LINE_PREFIX = (
    "Aiming to bring this file from {lines_before:,} lines to under {lines} by "
    "rewriting prose, which is what the {words:,}-word budget above encodes. "
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
    #: Lines the source held, and the published line target when it applies to
    #: this file (``None`` when it does not — see
    #: ``estimate.line_target_prose_words`` for the three reasons). Defaulted so
    #: every existing construction of this dataclass keeps working.
    lines_before: int = 0
    line_target: int | None = None
    #: Why ``target_prose_words`` is what it is, in one user-facing paragraph.
    #: Load-bearing: a reader must be able to tell a published target from ours.
    target_basis: str = ""
    #: Per-call envelope nonce. ``wrapped_prompt`` is fenced with it and
    #: ``system_rules`` names it, so `check` can require the rewrite to come back
    #: inside the same envelope. Carry it from prep to check (the automated path
    #: does so in one transaction; the manual paths pass it back explicitly).
    #: Empty when the file was below the prose gate and no envelope was built.
    source_nonce: str = ""

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
            "lines_before": self.lines_before,
            "line_target": self.line_target,
            "target_basis": self.target_basis,
            "source_nonce": self.source_nonce,
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
    #: Content tokens this rewrite removed, whitespace-normalized so reflow
    #: cannot be booked as compression. ``None`` when ``structure_ok`` is False:
    #: a rewrite the gate refused produced no usable output, so it has no
    #: saving — never a number computed off the refused text.
    est_tokens_saved: int | None
    must_keep_removed: list[str]
    must_keep_added: list[str]
    diff: str
    restored: str
    staged: bool
    produced_by: str
    note: str
    #: Prose words the source held before the rewrite. Recorded because it is the
    #: denominator that turns this verdict into evidence: structure is preserved
    #: verbatim, so ``words_before - words_after`` is prose words actually removed,
    #: and dividing by this gives the reduction the rewriter ACHIEVED — the number
    #: `core/summarize/estimate.observed_prose_ratio` reads back so the estimate can
    #: stop assuming what a rewrite will deliver. Defaulted so an older staged
    #: result (written before this field existed) still loads; such a result simply
    #: carries no evidence and is skipped rather than counted as a zero.
    prose_words_before: int = 0

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
            "prose_words_before": self.prose_words_before,
        }


# --------------------------------------------------------------------------- #
# Hashing + file IO (prep and check MUST read identically so the hash matches)
# --------------------------------------------------------------------------- #

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read(path: str) -> str:
    try:
        return Path(path).expanduser().read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        # A non-UTF-8 prompt file can't be summarized — refuse cleanly (callers map
        # SummarizeRefused → 409) instead of letting a ValueError (not an OSError)
        # crash the read. Missing files still raise FileNotFoundError (→ 404).
        raise SummarizeRefused(f"{path} isn't valid UTF-8 — cannot summarize it.") from exc


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


#: Filename suffix for a rewrite that was attempted and FAILED the structure
#: gate. Kept in the SAME directory as the staged results rather than in a
#: sibling store — there is one place a rewrite outcome lives, and the two kinds
#: are told apart by name so `list_staged` (and therefore `apply`, and
#: `estimate.observed_prose_ratio`) still sees only usable outcomes.
ATTEMPT_SUFFIX = ".attempt.json"


def _load_json(f: Path) -> dict | None:
    try:
        parsed = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None                                   # half-written/corrupt/unreadable
    return parsed if isinstance(parsed, dict) else None


def list_staged(config: TjConfig) -> list[dict]:
    """Every staged result (for the take-all CLI and the future Lens results picker).

    Failed-gate ATTEMPT records live in the same directory and are excluded
    here: they were never staged, there is nothing to apply, and they are not
    evidence of what a rewrite delivers. Read them with `list_attempts`.
    """
    d = results_dir(config)
    if not d.exists():
        return []
    out: list[dict] = []
    for f in sorted(d.glob("*.json")):
        if f.name.endswith(ATTEMPT_SUFFIX):
            continue
        parsed = _load_json(f)
        if parsed is not None:
            out.append(parsed)
    return out


def attempt_file(config: TjConfig, path: str) -> Path:
    return results_dir(config) / f"{stage_key(path)}{ATTEMPT_SUFFIX}"


def stage_attempt(config: TjConfig, verdict: CheckVerdict, source_hash: str) -> Path:
    """Record a rewrite that failed the structure gate; return the record file.

    Deliberately metadata only — the rewritten text and its diff are dropped,
    because a rejected rewrite is not something anyone will apply and keeping
    the file's content on disk buys nothing. What is kept is the finding: this
    file was attempted, at this cost, and could not be safely compressed. A run
    that produced only failures must not be indistinguishable from a run that
    never happened.
    """
    d = results_dir(config)
    d.mkdir(parents=True, exist_ok=True)
    payload = verdict.to_dict()
    for heavy in ("restored", "diff"):
        payload.pop(heavy, None)
    payload["source_sha256"] = source_hash
    payload["record_kind"] = "attempt"
    f = attempt_file(config, verdict.path)
    f.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return f


def list_attempts(config: TjConfig) -> list[dict]:
    """Every recorded failed-gate attempt. Never raises on a corrupt entry."""
    d = results_dir(config)
    if not d.exists():
        return []
    out: list[dict] = []
    for f in sorted(d.glob(f"*{ATTEMPT_SUFFIX}")):
        parsed = _load_json(f)
        if parsed is not None:
            out.append(parsed)
    return out


def clear_attempt(config: TjConfig, path: str) -> bool:
    """Drop the attempt record for ``path`` (it succeeded, or was applied)."""
    try:
        attempt_file(config, path).unlink()
    except (FileNotFoundError, OSError):
        return False
    return True


def read_staged(config: TjConfig, path: str) -> dict | None:
    f = results_dir(config) / f"{stage_key(path)}.json"
    if not f.exists():
        return None
    try:
        parsed: dict = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None                                   # corrupt/unreadable → treat as no record
    return parsed


def clear(config: TjConfig, path: str | None = None) -> int:
    """Remove one staged result (by source path) or all; return the count removed."""
    d = results_dir(config)
    if not d.exists():
        return 0
    # A path clears BOTH its staged result and its failed-attempt record; a bare
    # clear globs the dir, which already covers both.
    files = (
        [d / f"{stage_key(path)}.json", attempt_file(config, path)]
        if path is not None else list(d.glob("*.json"))
    )
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
    """Wrap a prompt's structure; return the wrapped prompt + rules + source hash. Persists nothing.

    The word budget in the rules is the MORE aggressive of two asks: ``ratio`` of
    the prose, and (for an always-resident file over the published size target)
    whatever reaching that target requires. The line target only ever tightens
    the ask; it never widens an estimate — `estimate.tokens_saved` prices off
    ``ratio`` alone, so a target the rewrite may not hit cannot inflate a figure.
    """
    _refuse_symlink(path)
    content = _read(path)
    source_hash = sha256(content)
    breakdown = detect.analyze(content)
    lines = detect.line_breakdown(content)
    n_prose = breakdown.prose_words
    if n_prose < min_prose_words:
        return PrepResult(
            path=path, source_sha256=source_hash, wrapped_prompt="", system_rules="",
            prose_words=n_prose, target_prose_words=0,
            protected_blocks=breakdown.protected_blocks, plan=[],
            note=f"Only {n_prose} prose words (< {min_prose_words}-word gate) — not worth summarizing.",
            lines_before=lines.total_lines,
        )
    wrapped, _saved, _order, plan = wrap.protect(content, hide_if_chars)
    load_class = load_semantics.classify(path, content)
    ratio_target = max(8, int(ratio * n_prose))
    line_budget = line_target_prose_words(
        text=content, load_class=load_class, prose_words=n_prose,
    )
    if line_budget is None:
        target = ratio_target
        rules = wrap.WRAP_SUMM_SYS.format(n=target)
    else:
        target = max(8, min(ratio_target, line_budget))
        rules = wrap.WRAP_SUMM_SYS.format(n=target) + wrap.WRAP_SUMM_SYS_LINE_GOAL.format(
            lines=PUBLISHED_LINE_TARGET, quote=PUBLISHED_LINE_TARGET_QUOTE,
        )
    # Fence the source as data. The rules say what to do; the envelope says which
    # of the two texts is the job — without it the file's own imperatives compete
    # with the ask to rewrite them, and on a large instruction file they win.
    nonce = wrap.new_nonce()
    rules += wrap.WRAP_SUMM_SYS_ENVELOPE.format(tag=wrap.SOURCE_TAG, nonce=nonce)
    wrapped = wrap.envelope(wrapped, nonce)
    # For an always-resident instruction file the basis is the ROUTE advice, not
    # a bare target: compression is one of four routes to the published size
    # target and the only one that costs specificity, so the offer must not
    # present it as the default. Everything else keeps the ratio statement.
    advice = recommend_route(text=content, load_class=load_class)
    if advice.route == route_mod.ROUTE_NOT_INSTRUCTION:
        basis = TARGET_BASIS_RATIO.format(pct=ratio * 100)
    elif line_budget is None:
        basis = advice.advice
    else:
        basis = TARGET_BASIS_LINE_PREFIX.format(
            lines_before=lines.total_lines, lines=PUBLISHED_LINE_TARGET, words=target,
        ) + advice.advice
    return PrepResult(
        path=path, source_sha256=source_hash, wrapped_prompt=wrapped,
        system_rules=rules,
        prose_words=n_prose, target_prose_words=target,
        protected_blocks=breakdown.protected_blocks, plan=plan, note="",
        lines_before=lines.total_lines,
        line_target=PUBLISHED_LINE_TARGET if line_budget is not None else None,
        target_basis=basis,
        source_nonce=nonce,
    )


#: Floor of the length band, as a fraction of the prose-word budget the rewriter was
#: given. A hijacked call answers the file instead of compressing it, and an answer
#: ("I am ready to help. What would you like to work on?") is orders of magnitude
#: under any real budget — so a very loose floor separates the two while leaving an
#: unusually eager but legitimate compression alone. Deliberately one-sided: a
#: rewrite that under-compresses is a quality question the diff already surfaces,
#: not a sign the model stopped rewriting.
MIN_PROSE_FRACTION_OF_TARGET = 0.25


def check(config: TjConfig, path: str, summary: str, source_hash: str,
          *, produced_by: str = "manual", source_nonce: str | None = None,
          target_prose_words: int | None = None) -> CheckVerdict:
    """Hash-guard the file, re-derive the map, restore the summary; stage it when structure holds.

    ``produced_by`` records who made the summary (manual / claude-p / api / in-session) on the staged
    result — the seam the Lens UI uses for mode-aware gating later (DEC-028). Raises
    ``SummarizeRefused`` (house-voice) if the file vanished or changed since prep — the summary was
    built against a different version, so there's nothing safe to do.

    ``source_nonce`` is `prepare`'s ``source_nonce``: pass it and the rewrite must come back inside
    the same envelope, which is the one check that works on a file with no protected blocks.
    ``target_prose_words`` is `prepare`'s budget; pass it and a rewrite far under it is refused as a
    non-rewrite. Both are optional so the manual and in-session paths, which don't carry a prep
    transaction, keep working — they still cannot pass vacuously, because `wrap.is_structure_ok`
    then requires at least one protected block.
    """
    _refuse_symlink(path)
    p = Path(path).expanduser()
    if not p.is_file():
        raise SummarizeRefused(
            f"{path} not found (moved or deleted since prep) — re-run `tj summarize prep`.")
    p = p.resolve()   # canonical absolute — `apply` must not reinterpret a relative path against its cwd
    try:
        current = p.read_text(encoding="utf-8")
    except FileNotFoundError as exc:   # vanished between the is_file check above and here (TOCTOU)
        raise SummarizeRefused(
            f"{path} not found (moved or deleted since prep) — re-run `tj summarize prep`.") from exc
    except UnicodeDecodeError as exc:
        raise SummarizeRefused(f"{path} isn't valid UTF-8 — cannot summarize it.") from exc
    if sha256(current) != source_hash:
        raise SummarizeRefused(
            f"{path} changed since `tj summarize prep` — re-run prep, then check.")

    # File matches prep → `protect` is deterministic, so this regenerates the exact map.
    _wrapped, saved, order, _plan = wrap.protect(current)
    prose = detect.prose_text(current)        # the unprotected text the model was given to rewrite
    source_sentinels = prose.count("<tj-keep") + prose.count("</tj-keep>")
    restored, integ = wrap.restore(summary, saved, order, source_sentinels=source_sentinels,
                                   nonce=source_nonce or None)
    ok = wrap.is_structure_ok(integ)
    reason = "" if ok else wrap.integrity_reason(integ)
    # Length band. The envelope catches a hijack that answers in prose; this catches one that
    # echoes the envelope but has nothing in it — a refusal, a one-line greeting, an apology.
    if ok and target_prose_words:
        floor = max(1, int(MIN_PROSE_FRACTION_OF_TARGET * target_prose_words))
        got = detect.analyze(restored).prose_words
        if got < floor:
            ok = False
            reason = (f"the rewrite is {got:,} prose words against a {target_prose_words:,}-word "
                      f"budget (floor {floor:,}) — that is not a compression of this file")
    removed, added = wrap.crit_delta(current, restored)
    wb, wa = wrap.word_count(current), wrap.word_count(restored)
    # Measured on whitespace-normalized CONTENT, never raw length: a `len()`
    # delta books reflow as compression, and on a hard-wrapped file that
    # artifact was the majority of the reported saving (see
    # `detect.content_chars`). And `None` — not a number — when the structure
    # gate refused the rewrite: a figure derived from output that failed the
    # gate is not a conservative estimate, it is a fabricated one. Observed in
    # the wild: a CLAUDE.md whose prose the rewriter answered as a live prompt
    # instead of summarizing, returning a chat greeting that dropped all 583
    # protected blocks — refused by the gate, yet still credited with a
    # five-figure token saving off the wreckage.
    est_saved: int | None = (
        max(0, (detect.content_chars(current) - detect.content_chars(restored))
            // detect.CHARS_PER_TOKEN)
        if ok else None
    )
    diff = "".join(difflib.unified_diff(
        current.splitlines(keepends=True), restored.splitlines(keepends=True),
        fromfile="original", tofile="summarized", n=2))

    verdict = CheckVerdict(
        path=str(p), structure_ok=ok, reason=reason,
        integrity=integ, words_before=wb, words_after=wa, est_tokens_saved=est_saved,
        prose_words_before=detect.analyze(current).prose_words,
        must_keep_removed=removed, must_keep_added=added, diff=diff, restored=restored,
        staged=ok, produced_by=produced_by, note=CHECK_NOTE if ok else GATE_FAIL_NOTE,
    )
    if ok:
        stage(config, verdict, source_hash)
        clear_attempt(config, str(p))     # an earlier failure on this file is superseded
    else:
        # Not staged (nothing here is applyable), but recorded: "this file could
        # not be safely compressed" is a finding, and a silently dropped attempt
        # makes a run that spent money look like a run that never happened.
        stage_attempt(config, verdict, source_hash)
    return verdict
