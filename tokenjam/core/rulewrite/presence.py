"""Is this rule's guidance ALREADY in the user's instruction files?

THE DEFECT THIS EXISTS FOR. ``rulewrite/plan._mark_applied`` resolves "already
dealt with" from tokenjam's OWN ledgers and nothing else. So a rule the user (or
their own harness) already wrote by hand is invisible to it: the analyzer measures
the recurrence, finds the guidance already present, and then reports the rule as
*too expensive to write* rather than as *already in place*.

That inversion is worst on exactly the machines that need the product most. The
write budget's ceiling reason says "your instruction files already carry more
standing per-session context than the budget allows to grow" — and on a corpus
whose instruction files are large BECAUSE they already contain these fixes, the
reason the budget is saturated IS that the work is done. The user reads a refusal
where the honest answer is a checkmark.

WHY A MODEL AND NOT A MATCHER. "Is this guidance already present" is a semantic
question. tokenjam's own marker comments only ever find tokenjam's own writes,
which is precisely the population that was already handled. Keyword overlap fails
both ways: it misses paraphrases (the common case, since a human writes the same
rule in their own words) and it false-positives on shared vocabulary, which would
route a real gap into "Applied" and silently drop a fix the user never made. So
this asks the user's own local ``claude`` CLI, reusing the subscription they
already pay for — the same mechanism, and the same pinned invocation recipe, that
``core.distill`` uses for title distillation.

WHAT IT COSTS AND HOW THAT IS BOUNDED. One call per DESTINATION FILE, not per
rule: the rules bound for one file are asked about together, so a corpus with
dozens of clusters over a handful of instruction files costs a handful of calls.
Results are cached against the file's content hash, so an unchanged file is never
re-asked, and the whole pass runs inside the analyzer cycle rather than on a
request (an analyzer never runs on a user-facing route).

THE SAFE DIRECTION IS "NOT PRESENT". Every failure — no ``claude`` on PATH, a
timeout, an unparseable answer, an unreadable file — resolves to absent, which
leaves the rule on offer. That can waste the user's attention on a rule they
already have. The opposite error hides a fix they never made, and a missing rule
is invisible in a way a duplicate one is not.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

if TYPE_CHECKING:
    from tokenjam.core.config import TjConfig
    from tokenjam.core.rulewrite.types import RuleWrite

#: Cheapest model that can do the comparison. Same default as ``core.distill``.
DEFAULT_MODEL = "haiku"

#: Wall-clock budget for one file's question. Generous — a cold CLI start plus a
#: long instruction file is not fast, and this runs on a background pass where
#: waiting costs nothing a user can see.
DEFAULT_TIMEOUT_SECONDS = 120

#: Instruction files longer than this are truncated before being sent. A rule is
#: either stated in a file or it is not, and the head of the file carries the
#: standing guidance; sending a megabyte to answer a yes/no is the sort of waste
#: this product exists to find.
_MAX_FILE_CHARS = 60_000

#: The rule text sent for comparison. The full artifact can be a long markdown
#: block; its opening states the guidance, and the rest is rationale.
_MAX_RULE_CHARS = 1_200

#: What the model is asked to return, and the only shape parsed back.
_ANSWER_RE = re.compile(
    r"^\s*(?P<idx>\d+)\s*[:.]\s*(?P<verdict>present|absent)\b(?P<rest>.*)$",
    re.IGNORECASE,
)


def presence_path(config: Any) -> Path:
    """``<storage-parent>/rule_presence.json``.

    Resolved through the same helper the apply ledgers use, so an in-memory or
    ``--config``-scoped install never writes into a real ``~/.tj``.
    """
    from tokenjam.core.optimize.relearn_apply import _storage_base_dir

    return _storage_base_dir(config) / "rule_presence.json"


def _read_store(config: Any) -> dict[str, Any]:
    """The stored verdicts, or ``{}``. Never raises — a corrupt store reads as
    "nothing known", which leaves every rule on offer (the safe direction)."""
    try:
        raw = json.loads(presence_path(config).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _write_store(config: Any, store: dict[str, Any]) -> None:
    """Temp-file + rename. Best-effort: a read-only filesystem degrades to a
    no-op, which costs a re-ask next pass and nothing else."""
    p = presence_path(config)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(store, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(p)
    except OSError:
        pass


def load_presence(config: Any) -> dict[str, dict[str, Any]]:
    """Signature → the stored presence verdict. Cheap; no model call.

    This is what a route or the CLI reads. Only records whose verdict is
    ``present`` are returned, so a caller cannot accidentally treat "we asked and
    it is absent" as a positive.
    """
    out: dict[str, dict[str, Any]] = {}
    for sig, rec in _read_store(config).items():
        if isinstance(rec, dict) and rec.get("present") is True:
            out[str(sig)] = rec
    return out


def _file_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def _rule_digest(rule: RuleWrite) -> str:
    return _file_digest(str(rule.artifact_text or ""))[:12]


def _cache_key(rule: RuleWrite, file_digest: str) -> str:
    """What must be unchanged for a stored verdict to still be valid.

    Both halves matter: edit the FILE and the answer may flip (the user just
    added the rule); change the RULE TEXT and it is a different question. Keyed
    on both so neither change is silently ignored.
    """
    return f"{_rule_digest(rule)}:{file_digest}"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n…[truncated]"


def _build_prompt(file_path: str, file_text: str, rules: Sequence[RuleWrite]) -> str:
    """One file, N candidate rules, a fixed answer shape.

    The prompt is deliberately blunt about the asymmetry: answering "present"
    when the guidance is not really there hides a fix the user never made, so it
    is told to require the substance and not merely a related topic.
    """
    lines = [
        "You are checking whether guidance is ALREADY PRESENT in an instruction "
        "file that an AI coding agent reads on every session.",
        "",
        f"FILE: {file_path}",
        "--- BEGIN FILE ---",
        _truncate(file_text, _MAX_FILE_CHARS),
        "--- END FILE ---",
        "",
        "For each candidate rule below, decide whether the file ALREADY tells the "
        "agent to do (or avoid) substantially the same thing. Wording will differ; "
        "judge the substance, not the phrasing.",
        "",
        "Answer PRESENT only if acting on the file alone would already produce the "
        "behaviour the candidate asks for. A file that merely mentions the same "
        "topic, or states a weaker or unrelated rule about it, is ABSENT. When "
        "genuinely unsure, answer ABSENT.",
        "",
    ]
    for i, rule in enumerate(rules, start=1):
        lines.append(f"CANDIDATE {i}: {rule.title or rule.signature}")
        lines.append(_truncate(str(rule.artifact_text or ""), _MAX_RULE_CHARS))
        lines.append("")
    lines += [
        "Reply with one line per candidate and nothing else:",
        "<number>: PRESENT — <the file's own words or heading that covers it>",
        "<number>: ABSENT",
    ]
    return "\n".join(lines)


def _parse_answer(result: str, count: int) -> dict[int, tuple[bool, str]]:
    """``{1-based index: (present, evidence)}`` for the lines that parsed.

    Unparseable or missing lines are simply absent from the result, and the
    caller treats a missing index as "not present" — the safe direction. A model
    that answers in prose therefore changes nothing rather than being guessed at.
    """
    out: dict[int, tuple[bool, str]] = {}
    for line in (result or "").splitlines():
        m = _ANSWER_RE.match(line)
        if not m:
            continue
        idx = int(m.group("idx"))
        if not 1 <= idx <= count:
            continue
        present = m.group("verdict").lower() == "present"
        evidence = m.group("rest").strip().lstrip("-—:").strip()
        out[idx] = (present, evidence[:300])
    return out


def _destination_paths(rule: RuleWrite) -> list[str]:
    return [d.path for d in (rule.destinations or ()) if getattr(d, "path", "")]


def detect_presence(
    config: TjConfig,
    rules: Iterable[RuleWrite],
    *,
    model: str = DEFAULT_MODEL,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, dict[str, Any]]:
    """Ask the local ``claude`` CLI which of ``rules`` are already in place.

    Groups by destination file so one call covers every rule bound for that file,
    skips anything whose verdict is already cached against the file's current
    content, merges the answers into the store and returns the full
    signature → record map (cached entries included).

    Never raises. Runs on the analyzer cycle, never on a request.
    """
    from tokenjam.core import distill

    rules = [r for r in rules if str(getattr(r, "artifact_text", "") or "")]
    store = _read_store(config)
    by_file: dict[str, list[RuleWrite]] = {}
    file_text: dict[str, str] = {}
    file_hash: dict[str, str] = {}

    for rule in rules:
        for path in _destination_paths(rule):
            if path not in file_text:
                try:
                    file_text[path] = Path(path).read_text(encoding="utf-8", errors="replace")
                except OSError:
                    # An unreadable destination cannot answer the question. Left
                    # out entirely rather than recorded as absent, so it is
                    # re-asked once the file is readable again.
                    continue
                file_hash[path] = _file_digest(file_text[path])
            if path not in file_text:
                continue
            key = _cache_key(rule, file_hash[path])
            cached = store.get(rule.signature)
            if isinstance(cached, dict) and cached.get("key") == key:
                continue          # already answered for this exact file+rule
            by_file.setdefault(path, []).append(rule)

    for path, pending in by_file.items():
        prompt = _build_prompt(path, file_text[path], pending)
        result = distill.invoke_claude(prompt, model=model, timeout=timeout)
        if result is None:
            # No CLI, a timeout, a non-zero exit. Nothing is recorded, so this
            # is retried next pass and every rule stays on offer meanwhile.
            continue
        answers = _parse_answer(result, len(pending))
        for i, rule in enumerate(pending, start=1):
            present, evidence = answers.get(i, (False, ""))
            store[rule.signature] = {
                "key": _cache_key(rule, file_hash[path]),
                "present": bool(present),
                "source_path": path,
                "evidence": evidence,
                "model": model,
            }

    if by_file:
        _write_store(config, store)
    return {
        str(sig): rec for sig, rec in store.items()
        if isinstance(rec, dict) and rec.get("present") is True
    }
