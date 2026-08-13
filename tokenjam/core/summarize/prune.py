"""Prune and expire — the two routes that remove tokens without making any
surviving instruction vaguer, now with write paths.

``route.py`` names six verdicts and, until this module, exactly one of them
(``compress``) could reach a write. Prune and expire were the largest unrealised
recovery in the tool for a defensible reason: compression is reversible in
principle, deletion is not, so the tool advised and stopped. What makes acting
safe is not refusing to act — it is the quarantine underneath
(``core/summarize/quarantine``), which holds every removed fragment verbatim and
can put it back.

**The two routes select differently, and the difference is the honesty line.**

* **expire** is MECHANICALLY decidable. An append-only log's entries open with a
  date (``detect._DATED_UNIT_RE``, the same detector ``route`` uses to call a
  file a log at all), and "older than N days" is a fact about a date, not a
  judgement about a rule. So this module can propose the set itself.
* **prune** is NOT. ``route.py`` says so outright: the shape measurement
  separates a rule-heavy file from a prose-heavy one and is "emphatically NOT a
  measurement of whether any individual rule is needed — that is the pruning
  question, and only the file's owner can answer it." So prune takes the
  sections the USER named, exactly the way ``relocate`` takes them, and proposes
  nothing on its own. A heuristic that guessed which rules earn their place
  would be this product asserting more than its data supports, on the one
  surface where being wrong deletes something.

**The human gate is unchanged.** Both produce a plan, the plan renders the exact
lines and the reason, and nothing is written until ``--go``. Quarantine is the
net under that gate, not a replacement for it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tokenjam.core.config import TjConfig
from tokenjam.core.summarize import backup, quarantine
from tokenjam.core.summarize.detect import CHARS_PER_TOKEN, _DATED_UNIT_RE
from tokenjam.core.summarize.route import ROUTE_EXPIRE, ROUTE_PRUNE
from tokenjam.core.summarize.sections import Section, parse_sections, remove_sections
from tokenjam.core.summarize.session import SummarizeRefused, sha256
from tokenjam.utils.time_parse import utcnow

#: Default age above which a dated log entry is offered for expiry. Deliberately
#: generous: the cost of keeping a stale entry one more quarter is a few tokens
#: a session, and the cost of expiring a still-live one is a lost decision.
DEFAULT_EXPIRE_DAYS = 180


@dataclass(frozen=True)
class PrunedFragment:
    """One fragment a plan would remove, with everything the diff needs."""

    title: str
    text: str
    start: int
    end: int
    #: 1-indexed, ``end_line`` exclusive — what the quarantine records and what
    #: the CLI prints, so "which lines would be cut" is answerable exactly.
    start_line: int
    end_line: int
    reason: str

    @property
    def chars(self) -> int:
        return len(self.text)

    @property
    def tokens(self) -> int:
        return max(0, round(len(self.text) / CHARS_PER_TOKEN))

    def to_dict(self) -> dict:
        return {
            "title": self.title, "start_line": self.start_line,
            "end_line": self.end_line, "chars": self.chars,
            "tokens": self.tokens, "reason": self.reason,
        }


@dataclass(frozen=True)
class PrunePlan:
    """What would be removed, from where, and what the file would become."""

    source_path: str
    route: str
    source_before: str
    source_after: str
    fragments: list[PrunedFragment]
    #: ``(title, why)`` for anything considered and not selected, so a user can
    #: see that the plan looked at it — silence would read as "not there".
    declined: list[tuple[str, str]] = field(default_factory=list)

    @property
    def tokens_freed(self) -> int:
        return sum(f.tokens for f in self.fragments)

    @property
    def chars_freed(self) -> int:
        return sum(f.chars for f in self.fragments)

    def to_dict(self) -> dict:
        return {
            "source_path": self.source_path,
            "route": self.route,
            "fragments": [f.to_dict() for f in self.fragments],
            "declined": [{"title": t, "why": w} for t, w in self.declined],
            "tokens_freed": self.tokens_freed,
            "chars_freed": self.chars_freed,
        }


def _line_range(text: str, start: int, end: int) -> tuple[int, int]:
    """1-indexed ``(start_line, end_line)`` for the ``[start:end)`` slice."""
    start_line = text.count("\n", 0, start) + 1
    end_line = start_line + text.count("\n", start, end)
    return start_line, end_line


def _fragment(text: str, section: Section, reason: str) -> PrunedFragment:
    start_line, end_line = _line_range(text, section.start, section.end)
    return PrunedFragment(
        title=section.title, text=section.text, start=section.start,
        end=section.end, start_line=start_line, end_line=end_line, reason=reason,
    )


def _build(source_path: str, text: str, route: str, chosen: list[PrunedFragment],
           declined: list[tuple[str, str]]) -> PrunePlan:
    sections = [
        Section(level=0, title=f.title, start=f.start, body_start=f.start,
                end=f.end, text=f.text)
        for f in chosen
    ]
    after = remove_sections(text, sections, [""] * len(sections)) if sections else text
    plan = PrunePlan(
        source_path=str(Path(source_path).expanduser()), route=route,
        source_before=text, source_after=after, fragments=chosen, declined=declined,
    )
    _assert_gates(plan)
    return plan


def _assert_gates(plan: PrunePlan) -> None:
    """Every fragment must be gone from the result, and nothing else may move.

    Checked at planning time AND again at apply time against exactly what is
    about to be written, because a plan can be built, held, and applied after
    the source changed underneath it — the same reason
    ``relocate._assert_gates`` is called twice.
    """
    for fragment in plan.fragments:
        if fragment.text in plan.source_after:
            raise SummarizeRefused(
                f'prune refused: the "{fragment.title}" fragment is still present '
                f"in the proposed result, so nothing would actually be removed. "
                f"Nothing was written."
            )
    removed = len(plan.source_before) - len(plan.source_after)
    if removed != plan.chars_freed:
        raise SummarizeRefused(
            f"prune refused: the result is {removed} characters shorter but the "
            f"selected fragments total {plan.chars_freed}. Something outside the "
            f"selection would have changed. Nothing was written."
        )


def plan_prune(
    *,
    source_path: str,
    source_text: str,
    titles: list[str],
    level: int = 2,
    reason: str = "",
) -> PrunePlan:
    """Remove the sections the USER named. Nothing is proposed automatically.

    ``route.py``'s shape measurement can say a file is long because rules
    accumulated; it explicitly cannot say WHICH rules earn their place, and this
    module does not pretend otherwise. An unmatched title is an error rather
    than a silent skip: a user who typo'd a heading must not get a successful
    "pruned 0 sections" and believe it worked.
    """
    if not titles:
        raise SummarizeRefused(
            "prune needs at least one --section: which rules earn their place is "
            "a question only the file's owner can answer, so nothing is selected "
            "automatically."
        )
    sections = parse_sections(source_text, level=level)
    # A LIST per title, not one section. `{s.title: s for s in sections}` keeps
    # only the LAST section of any repeated heading, and a file with two
    # `## Notes` then prunes the second while leaving the first — silently, and
    # while reporting that "Notes" was pruned. Worse, the survivor was invisible
    # in `declined` too, because that list was filtered by TITLE membership, so
    # the remaining duplicate was masked by the name of the one that went. The
    # user is told the section is gone and half of it is still there.
    by_title: dict[str, list[Section]] = {}
    for section in sections:
        by_title.setdefault(section.title.strip().lower(), []).append(section)

    chosen: list[PrunedFragment] = []
    missing: list[str] = []
    ambiguous: list[tuple[str, int]] = []
    for title in titles:
        matches = by_title.get(title.strip().lower(), [])
        if not matches:
            missing.append(title)
            continue
        if len(matches) > 1:
            ambiguous.append((title, len(matches)))
            continue
        section = matches[0]
        chosen.append(_fragment(
            source_text, section,
            reason or f'section "{section.title}" selected for pruning',
        ))
    if missing:
        available = ", ".join(sorted(s.title for s in sections)) or "(none)"
        raise SummarizeRefused(
            f"no level-{level} section named {', '.join(repr(m) for m in missing)} "
            f"in {source_path}. Available: {available}."
        )
    if ambiguous:
        # The same reasoning this function already applies to a typo'd title: a
        # request the tool cannot satisfy exactly must be an error, never a
        # partial success reported as a whole one. Refusing names the ambiguity
        # so the user can disambiguate or rename; guessing removes content they
        # did not point at, or leaves content they think is gone.
        detail = "; ".join(
            f"{title!r} matches {count} sections at level {level}"
            for title, count in ambiguous
        )
        raise SummarizeRefused(
            f"ambiguous --section in {source_path}: {detail}. Prune refuses "
            f"rather than removing one of them and reporting the whole title as "
            f"pruned. Give the duplicates distinct headings, or prune at a "
            f"level where they are unique."
        )
    # Declined by IDENTITY, not by title, so a section that survives because a
    # same-named sibling was chosen still shows up as kept.
    picked = {(f.start, f.end) for f in chosen}
    declined = [
        (s.title, "not named on the command line")
        for s in sections if (s.start, s.end) not in picked
    ]
    return _build(source_path, source_text, ROUTE_PRUNE, chosen, declined)


def _entry_date(title: str) -> datetime | None:
    """The date a dated-log entry's heading opens with, or ``None``.

    Uses the SAME detector ``detect``/``route`` use to call a file a log, so a
    heading that made the file diagnose as ``expire`` is a heading this can act
    on. A day-less ``2024-06`` heading is anchored to the first of the month,
    which is the reading that expires it LATEST — the conservative direction
    when the entry's exact day is unknown.
    """
    match = _DATED_UNIT_RE.match(title.strip())
    if not match:
        return None
    year, month, day = match.group(1), match.group(2), match.group(3)
    try:
        return datetime(
            int(year), int(month), int(day[1:]) if day else 1, tzinfo=timezone.utc,
        )
    except ValueError:
        return None


def plan_expire(
    *,
    source_path: str,
    source_text: str,
    older_than_days: int = DEFAULT_EXPIRE_DAYS,
    level: int = 2,
    now: datetime | None = None,
) -> PrunePlan:
    """Remove dated log entries older than ``older_than_days``.

    The one route this module selects for itself, because the selection is a
    fact rather than a judgement: an entry's heading carries a date, and older
    than a cutoff is arithmetic. Entries whose heading carries NO date are
    declined and named, never swept along — an undated section in a log is
    exactly where its standing content lives.
    """
    cutoff = (now or utcnow()) - timedelta(days=max(0, int(older_than_days)))
    chosen: list[PrunedFragment] = []
    declined: list[tuple[str, str]] = []
    for section in parse_sections(source_text, level=level):
        entry_date = _entry_date(section.title)
        if entry_date is None:
            declined.append((
                section.title,
                "no date in the heading — an undated section in a log is standing "
                "content, not an entry",
            ))
            continue
        if entry_date >= cutoff:
            declined.append((
                section.title,
                f"dated {entry_date.date().isoformat()}, newer than the "
                f"{older_than_days}-day cutoff",
            ))
            continue
        chosen.append(_fragment(
            source_text, section,
            f"dated {entry_date.date().isoformat()}, older than the "
            f"{older_than_days}-day cutoff",
        ))
    return _build(source_path, source_text, ROUTE_EXPIRE, chosen, declined)


def _result_anchors(plan: PrunePlan) -> list[tuple[str, str]]:
    """Per fragment, the text either side of the hole it leaves IN THE RESULT.

    Computed here rather than in the quarantine because only the plan knows how
    several removals compose. A fragment's neighbour in the ORIGINAL is often
    another fragment being removed in the same apply, so an anchor taken from
    the original names text the restored file will never contain — which is what
    makes every multi-fragment restore refuse. In the RESULT the removals have
    already collapsed, so each hole's neighbours are text that really survives.
    """
    after = plan.source_after
    ordered = sorted(plan.fragments, key=lambda f: f.start)
    cuts: dict[int, int] = {}
    shift = 0
    for fragment in ordered:
        cuts[fragment.start] = fragment.start - shift
        shift += len(fragment.text)
    out: list[tuple[str, str]] = []
    for fragment in plan.fragments:
        cut = cuts[fragment.start]
        out.append((
            after[max(0, cut - quarantine.ANCHOR_CHARS):cut],
            after[cut:cut + quarantine.ANCHOR_CHARS],
        ))
    return out


def apply_prune(config: TjConfig, plan: PrunePlan, *, go: bool = False) -> dict:
    """Write ``plan`` to disk. Default dry-run; ``go`` writes.

    ORDER IS THE SAFETY PROPERTY. Every fragment is committed to the quarantine
    — written, fsync'd, and read back from disk — BEFORE a single byte of the
    source changes. A quarantine write that fails raises, and it raises before
    the source is touched, so the file is left exactly as it was. There is no
    path through this function that removes content without a verified way back.

    Every guard ``apply.apply_staged`` applies to a rewrite applies here too:
    symlink, ownership, and drift against the text the plan was built from.
    """
    from tokenjam.core.summarize.apply import _owned_by_current_user, _write

    source = Path(plan.source_path).expanduser()
    skipped: list[dict] = []
    if source.is_symlink():
        skipped.append({"path": str(source), "reason": "symlink — refusing to write through it"})
    elif not source.is_file():
        skipped.append({"path": str(source), "reason": "file not found"})
    elif not _owned_by_current_user(source):
        skipped.append({"path": str(source), "reason": "owned by another user — refusing to write"})
    elif sha256(source.read_text(encoding="utf-8")) != sha256(plan.source_before):
        skipped.append({
            "path": str(source),
            "reason": "changed since the plan was built — re-plan it",
        })
    if skipped:
        return {
            "applied": False, "dry_run": not go, "skipped": skipped,
            "tokens_freed": 0, "quarantined": [], "plan": plan.to_dict(),
        }

    # Re-run the gates against exactly what is about to be written.
    _assert_gates(plan)

    quarantined: list[str] = []
    if go:
        # RE-CHECK IMMEDIATELY BEFORE THE WRITE, NOT ONLY BEFORE THE WORK.
        #
        # The hash check above happens, then N fsync'd quarantine writes and a
        # gzip backup happen, and only then does the source get rewritten. An
        # editor that touched the file during that window used to have its work
        # destroyed three ways at once: gone from the file (overwritten by text
        # derived from the older read), absent from the quarantine (which only
        # holds the fragments being removed), and absent from the backup —
        # because the backup was saved with the PLAN's text rather than a fresh
        # read, so `undo` restored the file to a state that never contained the
        # edit. Unrecoverable by any of the three rails.
        #
        # The re-check itself is below, immediately before the write — see
        # there. This block only does the quarantine writes.
        for fragment, (before_anchor, after_anchor) in zip(
            plan.fragments, _result_anchors(plan),
        ):
            entry = quarantine.record(
                config,
                source_path=plan.source_path,
                removed_text=fragment.text,
                start_line=fragment.start_line,
                end_line=fragment.end_line,
                source_before=plan.source_before,
                source_after=plan.source_after,
                route=plan.route,
                reason=fragment.reason,
                before_anchor=before_anchor,
                after_anchor=after_anchor,
            )
            quarantined.append(entry.entry_id)

        # THE LAST-MOMENT RE-CHECK. The quarantine writes above are fsync'd, so
        # they take real time, and an editor can land inside exactly that
        # window. Re-read here — not only at the top — because a check that runs
        # before the slow part does not cover the slow part.
        current = source.read_text(encoding="utf-8")
        if sha256(current) != sha256(plan.source_before):
            # The entries just written describe a removal that is not going to
            # happen. Drop them rather than leaving records of a cut nobody
            # made, which would read as recoverable history for a file that was
            # never touched.
            for entry_id in quarantined:
                quarantine.forget(config, entry_id)
            return {
                "applied": False, "dry_run": not go, "skipped": [{
                    "path": str(source),
                    "reason": (
                        "changed while this apply was preparing the quarantine — "
                        "re-plan it. Nothing was written, and the quarantine "
                        "entries for this attempt were discarded."
                    ),
                }],
                "tokens_freed": 0, "quarantined": [], "plan": plan.to_dict(),
            }

        # Back up WHAT WAS JUST READ, never `plan.source_before`. They are equal
        # here by the check immediately above, and that is the point: the backup
        # has to be a fresh read so it can never record a state the file was not
        # actually in. `apply.apply_staged` backs up its `current` for the same
        # reason. The whole-file backup is in addition to the quarantine — one
        # reverses the operation in a single `tj summarize undo`, the other
        # holds each fragment individually, and neither replaces the other.
        backup.save(
            config, str(source), original=current,
            output=plan.source_after, est_tokens_saved=plan.tokens_freed,
        )
        _write(source, plan.source_after)

    return {
        "applied": go, "dry_run": not go, "skipped": [],
        "tokens_freed": plan.tokens_freed, "quarantined": quarantined,
        "plan": plan.to_dict(),
    }


__all__ = [
    "DEFAULT_EXPIRE_DAYS",
    "PrunePlan",
    "PrunedFragment",
    "apply_prune",
    "plan_expire",
    "plan_prune",
]
