"""Reading records written before the delivery kind replaced the ladder rung.

Cached proposals and the applied/dismissed ledgers on a user's disk still carry
a numeric ``rung``. Nothing writes one any more, and nothing here ever will —
this module is a READ shim and only a read shim.

**Why it refuses rather than guesses.** The rung was ambiguous exactly where it
mattered most. Rung 3 covered two mechanisms with opposite cost behaviour: a
guard hook that executes and is genuinely free of prompt cost, and a reactive
hook that injects text and is not. Mapping every legacy 3 to one of them would
price a real user's applied fix wrongly in one of the two directions, silently
and permanently. So a rung-3 record is resolved only from evidence the record
itself carries — what was actually wired, or which family it belongs to — and
reports it UNRESOLVED when neither is present. A caller that gets that says the
kind is unknown; it must never fill the gap with a plausible-looking default.

Rungs 4 and 5 (wrapper, config) were never produced by any shipped build, so
there is nothing to migrate and no mapping is invented for them.
"""
from __future__ import annotations

from typing import Any, Mapping

from tokenjam.core.rulewrite.kinds import (
    DELIVERY_CLAUDE_MD_RULE,
    DELIVERY_EXECUTING_HOOK,
    DELIVERY_INJECTING_HOOK,
    DELIVERY_PATH_SCOPED_RULE,
    DELIVERY_SKILL,
    UNRESOLVED_DELIVERY,
)

#: What to SHOW for a record resolved to :data:`UNRESOLVED_DELIVERY`. An honest
#: gap: a wrong kind prices a real user's fix wrongly, which is worse than a
#: blank.
UNRESOLVED_DELIVERY_LABEL = "unknown (legacy record)"

#: Every mechanism this build knows by name. A record whose ``kind`` is one of
#: these needs no migration at all — it already names its mechanism.
_LIVE_DELIVERY_NAMES = frozenset({
    DELIVERY_CLAUDE_MD_RULE,
    DELIVERY_PATH_SCOPED_RULE,
    DELIVERY_SKILL,
    DELIVERY_EXECUTING_HOOK,
    DELIVERY_INJECTING_HOOK,
})

#: The unambiguous half of the old ladder. A rung-1 artifact was always a
#: markdown block in a ``CLAUDE.md``; a rung-2 artifact was always a skill file.
_UNAMBIGUOUS_BY_RUNG = {1: DELIVERY_CLAUDE_MD_RULE, 2: DELIVERY_SKILL}

#: The same mapping keyed on the old ``kind`` word, which some records carry
#: instead of (or alongside) the number.
_UNAMBIGUOUS_BY_KIND = {"note": DELIVERY_CLAUDE_MD_RULE, "skill": DELIVERY_SKILL}

#: Hook events, as they appear in a staged settings patch, and what each one
#: proves about the artifact. ``PreToolUse`` can block and injects nothing;
#: ``PostToolUseFailure`` exists to hand the model ``additionalContext``.
_DELIVERY_BY_HOOK_EVENT = {
    "PreToolUse": DELIVERY_EXECUTING_HOOK,
    "PostToolUseFailure": DELIVERY_INJECTING_HOOK,
}


def _hook_events(record: Mapping[str, Any]) -> list[str]:
    """Hook events named by this record's staged settings patch, if any.

    This is the strongest evidence available: it is the wiring that was
    actually written next to the hook, not an inference about it.
    """
    enforcement = record.get("enforcement")
    if not isinstance(enforcement, Mapping):
        return []
    patch = enforcement.get("patch")
    if not isinstance(patch, Mapping):
        return []
    hooks = patch.get("hooks")
    if not isinstance(hooks, Mapping):
        return []
    return [str(event) for event in hooks]


def _delivery_from_family(family_key: str) -> str | None:
    """Which hook kind this family's matcher produces, from the live tables.

    Read off ``relearn_apply``'s own guard/reactive registries rather than a
    copy, so a family that changes shape cannot leave this shim asserting the
    old one. Imported lazily: ``relearn_apply`` imports this package's names.
    """
    if not family_key:
        return None
    from tokenjam.core.optimize.relearn_apply import _GUARD_FAMILIES, _REACTIVE_SPECS

    if family_key in _GUARD_FAMILIES:
        return DELIVERY_EXECUTING_HOOK
    if family_key in _REACTIVE_SPECS:
        return DELIVERY_INJECTING_HOOK
    return None


def delivery_from_legacy_record(record: Mapping[str, Any]) -> str:
    """The delivery kind a pre-delivery record describes. Three outcomes.

    * A mechanism name — the record established which artifact was written.
    * :data:`UNRESOLVED_DELIVERY` — the record describes SOME artifact but not
      which mechanism produced it (a hook with neither wiring nor a known
      family; a rung 4 or 5 no build ever produced). Callers show it as
      unknown, and any attempt to render it refuses.
    * ``""`` — the record carries no artifact information at all, so there is
      nothing to migrate and the caller's own default applies.

    The middle case is the whole point of this module. Collapsing it into
    either of the others would either hide a real gap or invent a mechanism.
    """
    delivery = str(record.get("delivery") or "")
    if delivery:
        return delivery

    kind = str(record.get("kind") or "").strip().lower()
    # The applied-fix ledger records the artifact under ``kind``, and for a
    # record written by THIS build that word already IS a mechanism name. Check
    # that before any legacy mapping, so a current record is read as current
    # rather than falling through to the ambiguous branch below.
    if kind in _LIVE_DELIVERY_NAMES:
        return kind
    if kind in _UNAMBIGUOUS_BY_KIND:
        return _UNAMBIGUOUS_BY_KIND[kind]

    try:
        rung = int(record.get("rung") or 0)
    except (TypeError, ValueError):
        rung = 0
    if rung in _UNAMBIGUOUS_BY_RUNG:
        return _UNAMBIGUOUS_BY_RUNG[rung]

    if kind == "hook" or rung == 3:
        for event in _hook_events(record):
            resolved = _DELIVERY_BY_HOOK_EVENT.get(event)
            if resolved is not None:
                return resolved
        from_family = _delivery_from_family(str(record.get("family_key") or ""))
        return from_family or UNRESOLVED_DELIVERY

    # A rung outside the ladder's produced range, or a `kind` word this build
    # does not know: the record asserts an artifact and does not say which.
    if rung or kind:
        return UNRESOLVED_DELIVERY
    return ""


__all__ = ["UNRESOLVED_DELIVERY_LABEL", "delivery_from_legacy_record"]
