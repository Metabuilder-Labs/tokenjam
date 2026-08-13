"""The shapes the rule-write lifecycle passes around.

One rule, N destinations. Everything downstream — staging, the diff, apply,
undo — is per DESTINATION, because that is the granularity a user reviews at:
"this rule, into these 4 of your 11 projects, here is each diff, apply
selectively" is a sentence a Review-inbox card structurally cannot say.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tokenjam.core.rulewrite.legacy import delivery_from_legacy_record


@dataclass(frozen=True)
class RuleDestination:
    """One file a rule would land in, and the evidence that put it there."""

    path: str
    #: ``project`` or ``user-global``.
    scope: str = "user-global"
    #: How many of the window's sessions load this file. The standing cost is
    #: charged against exactly this count, which is the whole point of placing
    #: a rule rather than defaulting it to the user-global file.
    sessions: int = 0
    #: This destination's share of the rule's finding, split by session weight.
    #: ``usd`` is ``None`` — never 0.0 — when the finding carried no dollars.
    tokens: int = 0
    usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path, "scope": self.scope, "sessions": self.sessions,
            "tokens": self.tokens, "usd": self.usd,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RuleDestination:
        usd = raw.get("usd")
        return cls(
            path=str(raw.get("path", "")),
            scope=str(raw.get("scope", "user-global")),
            sessions=int(raw.get("sessions", 0) or 0),
            tokens=int(raw.get("tokens", 0) or 0),
            usd=None if usd is None else float(usd),
        )


@dataclass(frozen=True)
class RuleWrite:
    """One permanent rule an analyzer is offering, with where it would go.

    ``signature`` is the proposal's own stable identity, so a rule listed here,
    a card in the Review inbox and a staged diff on disk are the same thing
    under one name.
    """

    signature: str
    analyzer: str
    title: str
    artifact_text: str
    #: HOW this rule reaches the agent, and therefore WHAT gets written — see
    #: ``core/rulewrite/delivery``. A markdown block in a CLAUDE.md is one
    #: mechanism, not the only one, and for several of these analyzers not the
    #: best one: guidance delivered at the moment of the decision beats guidance
    #: at the top of a long context. Empty means the default, which is what
    #: every rule written before this was a field carries.
    delivery: str = ""
    #: Path globs this rule is confined to, when its observation is confined to
    #: identifiable paths. Non-empty is what makes a path-scoped delivery
    #: legitimate: a rule that must hold on every turn regardless of what is
    #: being touched belongs in a CLAUDE.md, and scoping it to globs would make
    #: it silently stop applying. DERIVED from the observation, never guessed.
    paths: tuple[str, ...] = ()
    destinations: tuple[RuleDestination, ...] = ()
    #: Mirrors the proposal's own verdict. A rule the write budget did not
    #: offer is still LISTED — its text is copyable and its reason is stated —
    #: it simply cannot be staged. Hiding it would make a deliberate deferral
    #: indistinguishable from an analyzer that found nothing.
    offered: bool = True
    blocked_reason: str = ""
    #: The user already dealt with this one — applied it through the product,
    #: or marked it applied by hand. Read from the apply ledgers via
    #: ``cost_apply.applied_signatures`` / ``relearn_apply.applied_signatures``
    #: and matched by ``cost_apply.signature_is_applied``, never by a second
    #: matcher of this module's own.
    #:
    #: **This suppresses the OFFER and NOTHING ELSE (Critical Rule 32).**
    #: ``past_overspend_usd`` / ``_tokens`` are untouched, deliberately and
    #: permanently: the waste genuinely happened inside the analyzed window,
    #: and the user having since fixed it does not un-spend the money. A gate
    #: on whether WE still have an action available may never reach back and
    #: edit what a behaviour already cost — that conflation is exactly what
    #: rule 32 exists to stop, and it is the easiest thing to get wrong here
    #: because "it is fixed now" feels like it should zero the number.
    #:
    #: The row is still LISTED, carrying this flag, rather than vanishing: a
    #: user who applies something and then sees nothing cannot tell "done"
    #: from "broken".
    already_applied: bool = False
    #: The user dismissed this card. Same discipline as ``already_applied``
    #: and for the same reason (Critical Rule 32): it withdraws the OFFER and
    #: leaves ``past_overspend_*`` untouched. The behaviour still happened and
    #: still cost what it cost; "not this one" is a statement about our
    #: recommendation, not about their bill.
    #:
    #: Reversible by construction — the row stays listed carrying this flag so
    #: there is something to un-dismiss. A durable dismissal with no way back
    #: is worse than the browser-local one it replaces.
    dismissed: bool = False
    #: The guidance is ALREADY in the user's instruction file — but written by
    #: them (or their own harness), not by tokenjam, so no apply ledger knows
    #: about it. Resolved by ``core/rulewrite/presence``, which asks the user's
    #: local ``claude`` because this is a semantic question: a human writes the
    #: same rule in their own words, and tokenjam's own markers only ever find
    #: tokenjam's own writes.
    #:
    #: Same discipline as ``already_applied``, and for the same reason (Critical
    #: Rule 32): it withdraws the OFFER and leaves ``past_overspend_*`` exactly
    #: as measured. The recurrence happened and cost what it cost; the guidance
    #: having been present the whole time does not un-spend it, and may say
    #: something about the rule's wording rather than about the money.
    #:
    #: Why this is the field that mattered most: without it, a rule already
    #: present is reported as too EXPENSIVE to write, because the same files that
    #: contain it are what saturate the write budget. The user reads a refusal
    #: where the honest answer is a checkmark.
    already_present: bool = False
    #: The file's own words that covered it, as quoted back by the model, and
    #: which file they were found in. Evidence, not a claim — it is what lets a
    #: reader check a verdict they did not make themselves.
    presence_evidence: str = ""
    presence_path: str = ""
    #: Why the rule is going where it is going, and what the placement could not
    #: cover. Carried verbatim from ``core/optimize/rule_placement``; never
    #: re-derived here, so the CLI, the UI and the payload cannot disagree.
    placement_basis: str = ""
    placement_coverage_note: str = ""
    #: The finding's own figures, for ordering and for the card. Not recomputed.
    past_overspend_tokens: int | None = None
    past_overspend_usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature,
            "analyzer": self.analyzer,
            "title": self.title,
            "artifact_text": self.artifact_text,
            "delivery": self.delivery,
            "paths": list(self.paths),
            "destinations": [d.to_dict() for d in self.destinations],
            "offered": self.offered,
            "blocked_reason": self.blocked_reason,
            "already_applied": self.already_applied,
            "dismissed": self.dismissed,
            "already_present": self.already_present,
            "presence_evidence": self.presence_evidence,
            "presence_path": self.presence_path,
            "placement_basis": self.placement_basis,
            "placement_coverage_note": self.placement_coverage_note,
            "past_overspend_tokens": self.past_overspend_tokens,
            "past_overspend_usd": self.past_overspend_usd,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RuleWrite:
        tokens = raw.get("past_overspend_tokens")
        usd = raw.get("past_overspend_usd")
        return cls(
            signature=str(raw.get("signature", "")),
            analyzer=str(raw.get("analyzer", "")),
            title=str(raw.get("title", "")),
            artifact_text=str(raw.get("artifact_text", "")),
            # A record cached by a build that named the artifact with a ladder
            # NUMBER instead of a mechanism is mapped on read. Where the number
            # does not establish which mechanism it was, this resolves to a
            # sentinel no renderer answers to, so the record is refused rather
            # than guessed at. Never written back — see ``rulewrite/legacy``.
            delivery=delivery_from_legacy_record(raw),
            paths=tuple(str(x) for x in (raw.get("paths") or []) if x),
            destinations=tuple(
                RuleDestination.from_dict(d) for d in (raw.get("destinations") or [])
            ),
            offered=bool(raw.get("offered", True)),
            blocked_reason=str(raw.get("blocked_reason", "") or ""),
            already_applied=bool(raw.get("already_applied", False)),
            dismissed=bool(raw.get("dismissed", False)),
            already_present=bool(raw.get("already_present", False)),
            presence_evidence=str(raw.get("presence_evidence", "") or ""),
            presence_path=str(raw.get("presence_path", "") or ""),
            placement_basis=str(raw.get("placement_basis", "") or ""),
            placement_coverage_note=str(raw.get("placement_coverage_note", "") or ""),
            past_overspend_tokens=None if tokens is None else int(tokens),
            past_overspend_usd=None if usd is None else float(usd),
        )


@dataclass
class StagedRuleWrite:
    """One rule's staged result for ONE destination: what would be written,
    the diff a reviewer reads, and the hash apply re-checks the file against.

    Staging persists the rendered OUTPUT rather than a recipe, so what a user
    approved in the diff is byte-for-byte what apply writes. The hash is the
    guard that makes that promise keepable: a file edited between staging and
    applying is skipped and reported, never merged into.
    """

    signature: str
    path: str
    scope: str
    title: str
    analyzer: str
    #: The mechanism this entry was rendered by. Persisted so apply re-resolves
    #: the SAME one that produced the diff a reviewer approved, rather than
    #: whatever the default happens to be by then.
    delivery: str
    #: sha256 of the file as it stood when this was staged.
    source_sha256: str
    #: The full file content to write.
    rendered: str
    diff: str
    standing_tokens_per_session: int = 0
    sessions: int = 0
    #: True when the file did not exist at staging time and would be created.
    creates_file: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature, "path": self.path, "scope": self.scope,
            "title": self.title, "analyzer": self.analyzer,
            "delivery": self.delivery, "source_sha256": self.source_sha256, "rendered": self.rendered,
            "diff": self.diff,
            "standing_tokens_per_session": self.standing_tokens_per_session,
            "sessions": self.sessions, "creates_file": self.creates_file,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> StagedRuleWrite:
        return cls(
            signature=str(raw.get("signature", "")),
            path=str(raw.get("path", "")),
            scope=str(raw.get("scope", "")),
            title=str(raw.get("title", "")),
            analyzer=str(raw.get("analyzer", "")),
            # An entry staged by a pre-delivery build carries a ladder number
            # rather than a mechanism; resolved on read, never rewritten. An
            # entry that cannot be resolved carries the unresolved sentinel and
            # is REFUSED at apply time by ``resolve_delivery`` rather than
            # rendered as a guess — this is the path that writes a real file to
            # a real location on a user's machine.
            delivery=delivery_from_legacy_record(raw),
            source_sha256=str(raw.get("source_sha256", "")),
            rendered=str(raw.get("rendered", "")),
            diff=str(raw.get("diff", "")),
            standing_tokens_per_session=int(
                raw.get("standing_tokens_per_session", 0) or 0
            ),
            sessions=int(raw.get("sessions", 0) or 0),
            creates_file=bool(raw.get("creates_file", False)),
            notes=list(raw.get("notes") or []),
        )


class RuleWriteRefused(Exception):
    """Refusing to stage, apply or undo a rule write.

    Carries the house-voice text; callers surface it as a 409 / a re-stage
    prompt. Every guard in this package raises this rather than returning a
    falsy value, so no caller can accidentally treat a refusal as a no-op.
    """
