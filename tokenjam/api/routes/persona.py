"""/api/v1/persona — which persona the corpus looks like, and which personas
have any data at all.

The UI's persona selector is a VIEW filter, not a stored preference: it decides
which lenses and pages are offered, defaults to whatever the corpus actually
looks like, and resets on reload. Nothing here is persisted, so this route is a
read of the ingested data and nothing else.

Thin by convention: the classification itself is
:func:`tokenjam.core.framing.dominant_persona` over
:func:`tokenjam.core.framing.agent_persona_mix`, the same derivation the CLI and
the Review inbox use. Re-deriving "which persona is this" a second way here is
exactly how the codebase ended up with two persona vocabularies once already.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from tokenjam.api.deps import require_api_key
from tokenjam.core.framing import (
    PERSONAS,
    agent_persona_mix,
    config_declared_plan,
    dominant_persona,
)

router = APIRouter()

#: Personas the selector offers. Every user is shown BOTH, whether or not they
#: have data for each: someone evaluating tokenjam for SDK work should be able
#: to look at the SDK view and be told plainly that this machine has no SDK
#: workflows, rather than have the option hidden and be left wondering whether
#: the product supports it. `mixed` / `unknown` are classifier outputs, not
#: things a human picks, so they are not offered here even though
#: `dominant_persona` can return them.
SELECTABLE_PERSONAS = ("claude-code", "sdk")


def _conn(request: Request) -> Any | None:
    db = getattr(request.app.state, "db", None)
    return getattr(db, "conn", None) if db is not None else None


@router.get("/persona", dependencies=[Depends(require_api_key)])
async def get_persona(request: Request) -> dict:
    """Detected persona plus per-persona session counts.

    ``detected`` is what the selector defaults to. ``available`` says which
    personas this machine has any ingested sessions for, so the UI can tell
    "you have no SDK workflows here" apart from "your SDK workflows cost
    nothing", which are entirely different statements and must never render
    the same way.

    ``counts`` is returned as-is rather than pre-formatted; a caller that wants
    to say "0 sessions" should be able to, and one that wants to say "not
    measured" should be able to tell the two apart. A missing/failed read
    returns ``known: false`` rather than zeros, for the same reason: a zero is
    a measurement, and we do not have one.
    """
    conn = _conn(request)
    if conn is None:
        return {
            "known": False,
            "detected": "unknown",
            "selectable": list(SELECTABLE_PERSONAS),
            "personas": list(PERSONAS),
            "counts": {},
            "available": [],
        }
    try:
        mix = agent_persona_mix(conn)
        detected = dominant_persona(
            mix, declared_plan=config_declared_plan(request.app.state.config)
        )
    except Exception:
        return {
            "known": False,
            "detected": "unknown",
            "selectable": list(SELECTABLE_PERSONAS),
            "personas": list(PERSONAS),
            "counts": {},
            "available": [],
        }

    # `agent_persona_mix` buckets sessions as coding-agent vs everything else;
    # "everything else" IS the SDK population for this question. Naming it here
    # rather than inventing a second query keeps one derivation.
    counts = {
        "claude-code": int(mix.get("claude-code", 0)),
        "sdk": int(mix.get("other", 0)),
    }
    return {
        "known": True,
        "detected": detected,
        "selectable": list(SELECTABLE_PERSONAS),
        "personas": list(PERSONAS),
        "counts": counts,
        "available": [p for p in SELECTABLE_PERSONAS if counts.get(p, 0) > 0],
    }
