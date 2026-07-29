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

from datetime import timedelta
from typing import Any

from fastapi import APIRouter, Depends, Request

from tokenjam.api.deps import require_api_key
from tokenjam.core.framing import (
    PERSONAS,
    agent_persona_mix,
    config_declared_plan,
    dominant_persona,
)
from tokenjam.core.optimize.report_window import report_window_days
from tokenjam.utils.time_parse import utcnow

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
    config = request.app.state.config
    try:
        # TWO QUESTIONS, TWO BASES — and they are deliberately different.
        #
        # `detected` is a CLASSIFICATION, and classification gates which
        # analyzers run (`PERSONA_DISABLED_ANALYZERS`) and whether relearn may
        # offer a workspace write. Every surface must resolve that gate over the
        # same window or two screens disagree about which findings exist, so it
        # goes through the one window seam (`core/optimize/report_window`) that
        # the Dashboard and the Review inbox already share.
        window_days = report_window_days(config, conn)
        until = utcnow()
        since = until - timedelta(days=window_days)
        detected = dominant_persona(
            agent_persona_mix(conn, since, until),
            declared_plan=config_declared_plan(config),
        )
        # `counts`/`available` answer a different question — "has this machine
        # ever ingested anything for this persona" — and the UI renders it as
        # "No SDK workflows on this machine. Nothing has been ingested for this
        # persona." That claim is about the corpus, not about a window, so it is
        # derived over ALL history on purpose: windowing it would report "never
        # ingested" for a persona whose sessions are merely older than the
        # window, which is a surface asserting more than its data supports.
        mix = agent_persona_mix(conn, None, None)
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
        # The window `detected` was classified over. Published rather than
        # implied, because the two figures on this payload sit on different
        # bases and a reader must be able to see which is which.
        "detected_window_days": window_days,
        "selectable": list(SELECTABLE_PERSONAS),
        "personas": list(PERSONAS),
        "counts": counts,
        "available": [p for p in SELECTABLE_PERSONAS if counts.get(p, 0) > 0],
    }
