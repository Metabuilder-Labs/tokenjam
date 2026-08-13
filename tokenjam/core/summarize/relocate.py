"""Relocate a reference section out of an always-loaded file into a linked one.

**Relocation is lossless or it is not relocation.** The content moves; nothing
is rewritten and nothing is deleted. That is the whole reason this operation is
worth more than compression despite being simpler: compression can silently
erode a modifier (an ``only``, an ``unless``, a ``never``) and no structural
check would see it, whereas a move can be verified by substring identity.
Anything that REMOVES content is a prune, which is a different operation, is not
performed here, and needs a human's judgement rather than an automatic apply.

The shape:

1. Pick the sections a validated classifier calls REFERENCE
   (``core/summarize/classify``) — what EXISTS, not what to do.
2. Append them verbatim to a non-loaded target (``docs/ARCHITECTURE.md`` by
   default), under a provenance line saying where each came from.
3. Leave the heading behind in the source with a POINTER to the target.

**The stub must be followable or the move is a loss.** A pointer nobody can
follow converts a lossless move into a real one, so the stub keeps the original
heading (so an existing link to ``#the-anchor`` still resolves), states the
target as a relative path from the source's own directory (so it works in an
editor, on GitHub, and in a terminal), and names the section's own heading
inside that target.

**Three gates run before anything is staged**, and any failure means nothing is
written — no partial application, no "wrote the source but not the target":

* the **losslessness gate** — every moved section's exact bytes must be
  findable in the new target text, and the source must have lost exactly the
  ranges the plan named;
* the **never-renumber gate** (``core/summarize/numbering``) — the multiset of
  numbered list items across BOTH files must be identical before and after;
* the **write guards** already used by ``apply.py`` — owner check, symlink
  refusal, content-hash drift check, and ``atomic_write``.

Dry-run is the default; ``go=True`` writes, exactly as ``apply.apply_staged``
does, and both files are backed up through the same store before either is
touched.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from tokenjam.core.atomic_write import AtomicWriteRefused, atomic_write
from tokenjam.core.config import TjConfig
from tokenjam.core.summarize import backup
from tokenjam.core.summarize.classify import SectionClassification, classify_section
from tokenjam.core.summarize.detect import CHARS_PER_TOKEN, content_chars
from tokenjam.core.summarize.numbering import describe_drift, numbering_drift
from tokenjam.core.summarize.sections import Section, parse_sections, remove_sections
from tokenjam.core.summarize.session import SummarizeRefused, sha256

#: Where relocated reference material goes when the caller names no target.
#: Deliberately under ``docs/`` and deliberately NOT any name the harness loads
#: automatically — the whole saving comes from the target not being resident.
DEFAULT_TARGET = "docs/ARCHITECTURE.md"

#: Header written once at the top of a freshly created target file.
_TARGET_HEADER = (
    "# Reference\n\n"
    "Reference material relocated out of always-loaded instruction files by "
    "`tj summarize relocate`. Nothing here was rewritten or shortened — each "
    "section is the verbatim text that used to live in the file named above "
    "it, moved so it is read when it is needed rather than re-sent at the head "
    "of every session.\n"
)
#: Provenance line above each relocated section in the target.
_PROVENANCE = "<!-- relocated from {source} -->\n"
#: What is left behind in the source. Keeps the original heading (so existing
#: anchors still resolve) and gives a followable relative path plus the heading
#: to look for once there.
_STUB = (
    "{heading}"
    "Moved to [`{link}`]({link}) — see the \"{title}\" section there. Read it "
    "when working on this area; it is reference material, not an instruction, "
    "so it does not need to be resident in every session. Nothing was "
    "rewritten or removed in the move.\n"
)


def _owned_by_current_user(p: Path) -> bool:
    if not hasattr(os, "getuid"):          # non-POSIX — no ownership model to honour
        return True
    return p.stat().st_uid == os.getuid()


def _write(p: Path, text: str) -> None:
    """``atomic_write``, translated to the house ``SummarizeRefused`` on refusal."""
    try:
        atomic_write(p, text)
    except AtomicWriteRefused as exc:
        raise SummarizeRefused(str(exc)) from exc


def _relative_link(source: Path, target: Path) -> str:
    """``target`` as a path relative to ``source``'s directory, POSIX-style.

    Falls back to the absolute path when the two share no walkable ancestor
    (different drives, or a target outside the source's tree): an absolute path
    is uglier but still followable, and a pointer that cannot be followed is
    the one failure this stub exists to prevent.
    """
    try:
        return os.path.relpath(target, source.parent).replace(os.sep, "/")
    except ValueError:
        return target.as_posix()


@dataclass(frozen=True)
class PlannedSection:
    """One section the plan proposes to move, with the evidence for moving it."""

    title: str
    #: Content characters (whitespace-normalized — see ``detect.content_chars``)
    #: removed from the always-loaded file, NET of the stub left behind. A raw
    #: character delta would book the stub as free.
    content_chars_freed: int
    classification: SectionClassification

    @property
    def tokens_freed(self) -> int:
        return max(0, round(self.content_chars_freed / CHARS_PER_TOKEN))

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "content_chars_freed": self.content_chars_freed,
            "tokens_freed": self.tokens_freed,
            "classification": self.classification.to_dict(),
        }


@dataclass(frozen=True)
class RelocationPlan:
    """A complete, verified proposal to move N sections from one file to another.

    Holds the FULL new text of both files rather than a diff, so the gates can
    run on exactly what would be written and the apply step has no second
    chance to derive something different.
    """

    source_path: str
    target_path: str
    sections: list[PlannedSection]
    source_before: str
    target_before: str
    source_after: str
    target_after: str
    #: Sections the classifier examined and declined to move, with its reason —
    #: carried because "what was left alone and why" is the half of this
    #: operation a reviewer most needs to see.
    declined: list[tuple[str, SectionClassification]] = field(default_factory=list)

    @property
    def tokens_freed(self) -> int:
        return sum(s.tokens_freed for s in self.sections)

    @property
    def content_chars_freed(self) -> int:
        return sum(s.content_chars_freed for s in self.sections)

    def to_dict(self) -> dict:
        return {
            "source_path": self.source_path,
            "target_path": self.target_path,
            "sections": [s.to_dict() for s in self.sections],
            "tokens_freed": self.tokens_freed,
            "content_chars_freed": self.content_chars_freed,
            "declined": [
                {"title": t, "classification": c.to_dict()} for t, c in self.declined
            ],
        }


def _stub_for(section: Section, link: str) -> str:
    heading = section.heading
    if not heading.endswith("\n"):
        heading += "\n"
    return _STUB.format(heading=heading + "\n", link=link, title=section.title)


def relocatable_sections(text: str, *, level: int = 2) -> list[tuple[Section, SectionClassification]]:
    """Every section of ``text`` paired with the classifier's verdict on it.

    Returned for ALL sections, not only the reference ones: a caller that wants
    to show what was left alone (and the relocate CLI does) needs the declined
    ones too, and a caller that only wants the movable ones filters on
    ``classification.is_reference``.
    """
    return [(s, classify_section(s.title, s.body)) for s in parse_sections(text, level=level)]


def relocatable_content_chars(text: str, *, level: int = 2) -> int:
    """Content characters ``text`` would shed by relocating its reference sections.

    NET of the pointer stubs left behind, and whitespace-normalized
    (``detect.content_chars``) for the same reason every other figure in this
    package is: a raw delta books reflow as a saving.

    Zero — never an error and never a guess — when the file has no sections,
    none of them classify as reference, or the stubs would cost more than the
    move frees. Never raises: this runs inside the read-only catalog scan, which
    must not be breakable by one unusual file.
    """
    try:
        total = 0
        for section, verdict in relocatable_sections(text, level=level):
            if not verdict.is_reference:
                continue
            stub = _stub_for(section, DEFAULT_TARGET)
            total += max(0, content_chars(section.text) - content_chars(stub))
        return total
    except Exception:       # pragma: no cover - defensive; the scan must not break
        return 0


def plan_relocation(
    *,
    source_path: str,
    source_text: str,
    target_path: str,
    target_text: str = "",
    level: int = 2,
    titles: list[str] | None = None,
) -> RelocationPlan | None:
    """Build a verified plan, or ``None`` when there is nothing safe to move.

    ``titles`` restricts the move to named sections (still subject to the
    classifier — an explicitly named section that classifies as instruction is
    NOT moved, because the point of the gate is that it cannot be talked out of
    by a caller). Raises ``SummarizeRefused`` if a gate fails on the candidate
    output, which is a bug rather than a user condition: the gates exist so a
    plan that reaches a caller is already known to be lossless.
    """
    pairs = relocatable_sections(source_text, level=level)
    if not pairs:
        return None
    wanted = set(titles) if titles is not None else None
    chosen: list[tuple[Section, SectionClassification]] = []
    declined: list[tuple[str, SectionClassification]] = []
    for section, verdict in pairs:
        if verdict.is_reference and (wanted is None or section.title in wanted):
            chosen.append((section, verdict))
        else:
            declined.append((section.title, verdict))
    if not chosen:
        return None

    source = Path(source_path)
    target = Path(target_path)
    link = _relative_link(source, target)

    stubs = [_stub_for(section, link) for section, _ in chosen]
    source_after = remove_sections(source_text, [s for s, _ in chosen], stubs)

    appended = target_text
    if not appended.strip():
        appended = _TARGET_HEADER
    if not appended.endswith("\n"):
        appended += "\n"
    for section, _ in chosen:
        body = section.text
        if not body.endswith("\n"):
            body += "\n"
        appended += "\n" + _PROVENANCE.format(source=source_path) + body
    target_after = appended

    planned = [
        PlannedSection(
            title=section.title,
            content_chars_freed=max(
                0, content_chars(section.text) - content_chars(stub),
            ),
            classification=verdict,
        )
        for (section, verdict), stub in zip(chosen, stubs)
    ]
    plan = RelocationPlan(
        source_path=source_path, target_path=target_path, sections=planned,
        source_before=source_text, target_before=target_text,
        source_after=source_after, target_after=target_after, declined=declined,
    )
    _assert_gates(plan, [s for s, _ in chosen])
    return plan


def _assert_gates(plan: RelocationPlan, moved: list[Section]) -> None:
    """Run the losslessness and never-renumber gates on the candidate output.

    Raises ``SummarizeRefused`` on any failure. Called from ``plan_relocation``
    so a plan is verified at construction, and AGAIN from ``apply_relocation``
    against the text about to be written — the second call is not redundant,
    because a plan can be built, held, and applied after the source changed
    underneath it.
    """
    for section in moved:
        if section.text not in plan.target_after:
            raise SummarizeRefused(
                f'relocation refused: the "{section.title}" section\'s exact text '
                "is not present in the proposed target, so the move would not be "
                "lossless. Nothing was written."
            )
        if section.text in plan.source_after:
            raise SummarizeRefused(
                f'relocation refused: the "{section.title}" section is still '
                "present in the proposed source, so it would be DUPLICATED "
                "rather than moved. Nothing was written."
            )
    drift = numbering_drift(
        source_before=plan.source_before, target_before=plan.target_before,
        source_after=plan.source_after, target_after=plan.target_after,
    )
    if drift:
        raise SummarizeRefused(f"relocation refused: {describe_drift(drift)}. Nothing was written.")


def apply_relocation(config: TjConfig, plan: RelocationPlan, *, go: bool = False) -> dict:
    """Write ``plan`` to disk. Default dry-run; ``go`` writes.

    Returns ``{"applied": bool, "dry_run": bool, "skipped": [...], ...}``. Every
    guard ``apply.apply_staged`` applies to a rewrite applies here to both
    files, and the source is checked for drift against the text the plan was
    built from — a file edited since the plan was made is skipped, never
    written over.
    """
    source = Path(plan.source_path).expanduser()
    target = Path(plan.target_path).expanduser()
    skipped: list[dict] = []

    for p, label, expected in (
        (source, "source", plan.source_before), (target, "target", plan.target_before),
    ):
        if p.is_symlink():
            skipped.append({"path": str(p), "reason": "symlink — refusing to write through it"})
        elif p.is_file() and not _owned_by_current_user(p):
            skipped.append({"path": str(p), "reason": "owned by another user — refusing to write"})
        elif p.is_file() and sha256(p.read_text(encoding="utf-8")) != sha256(expected):
            skipped.append({"path": str(p), "reason": f"{label} changed since the plan was built — re-plan it"})
        elif label == "source" and not p.is_file():
            skipped.append({"path": str(p), "reason": "file not found"})

    if skipped:
        return {"applied": False, "dry_run": not go, "skipped": skipped,
                "tokens_freed": 0, "plan": plan.to_dict()}

    # Re-run the gates against exactly what is about to be written. A plan can
    # outlive the state it was built from, and a gate that only ran at planning
    # time is a gate that does not run at apply time.
    _assert_gates(plan, [
        s for s in parse_sections(plan.source_before)
        if any(ps.title == s.title for ps in plan.sections)
    ])

    if go:
        backup.save(config, str(source), original=plan.source_before,
                    output=plan.source_after, est_tokens_saved=plan.tokens_freed)
        backup.save(config, str(target), original=plan.target_before,
                    output=plan.target_after, est_tokens_saved=0)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Target first: if the source write then fails, the reference material
        # exists in two places, which is recoverable. The reverse order can
        # lose it outright.
        if target.exists():
            _write(target, plan.target_after)
        else:
            target.write_text(plan.target_after, encoding="utf-8")
        _write(source, plan.source_after)

    return {
        "applied": go, "dry_run": not go, "skipped": [],
        "tokens_freed": plan.tokens_freed, "plan": plan.to_dict(),
    }
