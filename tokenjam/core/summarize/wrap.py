"""Wrap → restore mechanism for structure-aware summarization (stdlib only).

Pure algorithm — no IO, no config. PROTECT each structured span behind an id'd
``<tj-keep>`` marker (copying the original out by id), let a frontier model
rewrite only the PROSE, then RESTORE each original verbatim by id. Structure is
a HARD guarantee (restore-by-id, mechanical); must-keep WORDS are TRACKED, never
gated — a good summary may rephrase "never X" → "avoid X" (TokenJam DEC-009/010).

Ports the research source of truth (``trim_pipeline.py`` / the skill's
``verify.py``). The protected-span detectors are shared with
``detect.protected_spans`` so the wrap stays consistent with what the scan
counts as structure — keep the tag format and restore semantics in sync.
"""
from __future__ import annotations

import re
from collections import Counter

from tokenjam.core.summarize.detect import protected_spans

# Spans larger than this are HIDDEN (self-closing marker) to save summarizer call
# tokens; smaller spans stay VISIBLE so the model keeps full context. One
# mechanism — both are reinserted verbatim by id. Threshold tuning is DEF-003.
HIDE_IF_CHARS = 800

# Load-bearing words whose movement we TRACK (never gate on). Mirror of the
# research allowlist (structure_detect.ALLOWLIST / the skill verifier).
ALLOWLIST: dict[str, frozenset[str]] = {
    "negation":   frozenset({"no", "not", "nor", "never", "neither", "none", "without", "cannot",
                             "can't", "cant", "don't", "dont", "doesn't", "didn't", "won't",
                             "wouldn't", "shouldn't", "mustn't", "couldn't", "isn't", "aren't",
                             "wasn't", "weren't", "haven't", "hasn't", "hadn't", "nothing",
                             "nobody", "nowhere"}),
    "scope":      frozenset({"all", "every", "each", "always", "only", "solely", "exclusively",
                             "entirely", "completely", "fully", "wholly", "everyone", "everything",
                             "everywhere"}),
    "obligation": frozenset({"must", "shall", "required", "require", "mandatory", "obligatory",
                             "forbidden", "prohibited", "prohibit", "banned", "disallowed"}),
    "exception":  frozenset({"except", "unless", "otherwise", "instead"}),
    "emphasis":   frozenset({"strictly", "exactly", "precisely", "verbatim", "explicitly",
                             "specifically", "ever"}),
}
CRITICAL_WORDS: frozenset[str] = frozenset().union(*ALLOWLIST.values())

# The summarizer contract. ``{n}`` is the target PROSE word budget — structure
# (the markers and their blocks) is excluded from the count.
WRAP_SUMM_SYS = (
    "You compress AI system prompts. Rewrite the prose to about {n} words while preserving the MEANING "
    "of every instruction, constraint, tool, and persona detail — especially negations, scope, and "
    "conditions (\"never\", \"only\", \"always\", \"unless\") — even if you rephrase them. Preserve EXACT "
    "tool/function names and any SPECIFIC required action or procedure (e.g. \"call reference_check on the "
    "link\", \"flag it as phishing\", \"long-press to copy, do not click\") — never generalize a concrete "
    "required step into something vague like \"analyze it\".\n\n"
    "The text contains markers <tj-keep id=\"K\" ...> in two forms: with content "
    "(<tj-keep id=K ...>…</tj-keep>) and self-closing (<tj-keep id=K .../>, which stands for a large "
    "block omitted here). Both are reinserted verbatim from the original. For every marker:\n"
    "- Keep it exactly — same id, each exactly once, in the same order.\n"
    "- Never edit, describe, expand, merge, or split a marker or its contents; for self-closing "
    "markers, do not guess what they contain.\n"
    "- Place each marker so the surrounding text still refers to it correctly.\n\n"
    "Your {n}-word budget is for the prose only — markers and their blocks don't count. Output ONLY "
    "the rewritten prompt — no commentary, no code fences."
)

_WORD_RE = re.compile(r"\S+")
_CRIT_STRIP = ".,;:!?\"'()[]{}*_`"
_KEEP_RE = re.compile(r'<tj-keep id="(\d+)"[^>]*?(?:/>|>.*?</tj-keep>)', re.DOTALL)


def word_count(text: str) -> int:
    """Whitespace-delimited word count (matches the detector's prose basis)."""
    return len(_WORD_RE.findall(text))


def critical_words(text: str) -> Counter[str]:
    """Multiset of load-bearing words present in ``text`` (lowercased, depunctuated)."""
    out: list[str] = []
    for w in _WORD_RE.findall(text):
        t = w.strip(_CRIT_STRIP).lower()
        if t in CRITICAL_WORDS:
            out.append(t)
    return Counter(out)


def protect(
    text: str, hide_if_chars: int = HIDE_IF_CHARS
) -> tuple[str, dict[str, str], list[int], list[dict]]:
    """Wrap structured spans in id'd keep-tags; copy the originals out by id.

    Returns ``(wrapped, saved, order, plan)`` where ``saved`` maps ``str(id) ->
    verbatim content`` (JSON-friendly), ``order`` is the id sequence, and
    ``plan`` describes each block ``{id, kind, chars, mode}`` for transparency.
    """
    parts: list[str] = []
    saved: dict[str, str] = {}
    order: list[int] = []
    plan: list[dict] = []
    cur = 0
    nid = 0
    for start, end, kind in protected_spans(text):
        parts.append(text[cur:start])
        nid += 1
        content = text[start:end]
        saved[str(nid)] = content
        order.append(nid)
        # A span containing a literal "tj-keep" must be hidden — a visible marker
        # would nest and break the (non-recursive) restore regex.
        hidden = len(content) > hide_if_chars or "tj-keep" in content
        if hidden:
            parts.append(f'<tj-keep id="{nid}" kind="{kind}"/>')
        else:
            parts.append(f'<tj-keep id="{nid}" kind="{kind}">{content}</tj-keep>')
        plan.append({"id": nid, "kind": kind, "chars": len(content),
                     "mode": "hidden" if hidden else "visible"})
        cur = end
    parts.append(text[cur:])
    return "".join(parts), saved, order, plan


def restore(
    model_output: str, saved: dict[str, str], order: list[int], *, source_sentinels: int = 0,
) -> tuple[str, dict]:
    """Swap each saved original back into its id'd slot; strip the tags.

    Hallucinated keep-tags (ids the model invented) are stripped to empty. ``source_sentinels``
    is how many literal ``<tj-keep`` / ``</tj-keep>`` strings the SOURCE prose already held (a
    prompt may document the markers); residue is flagged only when it EXCEEDS that baseline, so a
    legitimate prose mention isn't a false failure.
    """
    # Canonical STRING ids throughout (matching `saved`'s keys). Integrity and substitution MUST
    # agree on the key: int-parsing for the set checks while substituting by raw string let a
    # non-canonical id like "01" pass (int("01")==1 looks present) yet miss saved["01"] and silently
    # drop the block. As strings, "01" surfaces as missing "1" + extra "01" and the gate refuses.
    found = _KEEP_RE.findall(model_output)
    counts = Counter(found)
    seen: set[str] = set()
    first_order: list[str] = []
    for i in found:
        if i not in seen:
            seen.add(i)
            first_order.append(i)
    expected_ids = [str(i) for i in order]
    missing = [i for i in expected_ids if i not in seen]
    duplicated = sorted((i for i, c in counts.items() if c > 1), key=int)
    extra = sorted((i for i in seen if i not in saved), key=int)
    expected = [i for i in expected_ids if i in seen]
    reordered = first_order != expected

    def _sub(m: re.Match[str]) -> str:
        key = m.group(1)
        return saved[key] if key in saved else ""   # invented / non-canonical ids → stripped (in `extra`)

    # Any `<tj-keep` / `</tj-keep>` the well-formed matcher didn't consume is marker residue
    # (malformed, invented, id-less, or a stray close). The model may legitimately echo sentinels
    # the SOURCE prose contained, so only residue BEYOND that baseline is a real failure.
    residue = _KEEP_RE.sub("", model_output)
    residue_sentinels = residue.count("<tj-keep") + residue.count("</tj-keep>")
    malformed = residue_sentinels > source_sentinels
    restored = _KEEP_RE.sub(_sub, model_output)
    return restored, {"missing": missing, "duplicated": duplicated, "extra": extra,
                      "reordered": reordered, "malformed": malformed, "n_blocks": len(order)}


def is_structure_ok(integrity: dict) -> bool:
    """Structure is intact iff no block was dropped, duplicated, invented, or moved."""
    return not (integrity["missing"] or integrity["duplicated"] or integrity["extra"]
                or integrity["reordered"] or integrity["malformed"])


def integrity_reason(integrity: dict) -> str:
    """Human-readable explanation of a failed structure check ("" when OK)."""
    parts: list[str] = []
    if integrity["missing"]:
        parts.append(f"dropped blocks {integrity['missing']}")
    if integrity["duplicated"]:
        parts.append(f"duplicated blocks {integrity['duplicated']}")
    if integrity["extra"]:
        parts.append(f"invented blocks {integrity['extra']}")
    if integrity["reordered"]:
        parts.append("blocks reordered")
    if integrity["malformed"]:
        parts.append("malformed markers (unrestored tj-keep tag residue)")
    return "; ".join(parts)


def crit_delta(original: str, restored: str) -> tuple[list[str], list[str]]:
    """TRACKED must-keep word movement: ``(removed, added)``. NOT a gate.

    A good summary may validly rephrase ("never X" → "avoid X"), so we record the
    load-bearing words it dropped or introduced as a review signal and never
    reject a summary over them.
    """
    oc, rc = critical_words(original), critical_words(restored)
    removed = sorted(w for w in oc if rc[w] < oc[w])
    added = sorted(w for w in rc if rc[w] > oc[w])
    return removed, added
