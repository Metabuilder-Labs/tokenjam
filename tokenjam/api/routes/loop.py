"""Close-the-loop routes (#53) — annotations, expectations, fix-history.

The read/write surface for the loop-closing primitive in ``core/loop.py``. All
local-first: these back the Lens "Loop" tab so a user can, in the dashboard,
leave a verdict/note on a run, promote a bad run into an expectation, and record
whether a later run passed or regressed against it.

Write routes are gated by ``require_api_key`` (the UI's ``apiPost`` sends the
Bearer key), consistent with the ``POST /sessions/{id}/label`` rename — they are
NOT the ingest-secret PROTECTED_PATHS (that's for span ingest only). Reads are
guarded by ``require_api_key`` like every other GET.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from tokenjam.api.deps import require_api_key
from tokenjam.core import loop

router = APIRouter()


async def _json_body(request: Request) -> dict | JSONResponse:
    """Parse a JSON object body, or return a 400 JSONResponse."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})
    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400, content={"error": "Expected a JSON object"}
        )
    return body


# --- Annotations -------------------------------------------------------------

@router.post(
    "/sessions/{session_id}/annotations",
    dependencies=[Depends(require_api_key)],
)
async def add_annotation_endpoint(
    request: Request, session_id: str
) -> JSONResponse:
    """Append a human note + optional verdict to a run.

    Body: ``{"note": "<str>", "verdict": "good|bad|mixed|unknown"?}``. ``note``
    is required; ``verdict`` optional. Returns the created annotation, or 400 on
    a missing note / invalid verdict.
    """
    body = await _json_body(request)
    if isinstance(body, JSONResponse):
        return body
    try:
        ann = loop.add_annotation(
            request.app.state.db,
            session_id,
            note=body.get("note") or "",
            verdict=body.get("verdict"),
        )
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    return JSONResponse(status_code=201, content=ann.to_dict())


@router.get(
    "/sessions/{session_id}/annotations",
    dependencies=[Depends(require_api_key)],
)
async def list_annotations_endpoint(request: Request, session_id: str) -> dict:
    """Every annotation on a run, newest first."""
    anns = loop.list_annotations(request.app.state.db, session_id)
    return {"annotations": [a.to_dict() for a in anns], "count": len(anns)}


# --- Expectations ------------------------------------------------------------

@router.post("/expectations", dependencies=[Depends(require_api_key)])
async def create_expectation_endpoint(request: Request) -> JSONResponse:
    """Promote a run into a stored expectation/case.

    Body: ``{"name": "<str>", "description": "<str>"?,
    "origin_session_id": "<str>"?, "agent_id": "<str>"?}``. ``name`` required.
    """
    body = await _json_body(request)
    if isinstance(body, JSONResponse):
        return body
    try:
        exp = loop.create_expectation(
            request.app.state.db,
            name=body.get("name") or "",
            description=body.get("description"),
            origin_session_id=body.get("origin_session_id"),
            agent_id=body.get("agent_id"),
        )
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    return JSONResponse(status_code=201, content=exp.to_dict())


@router.get("/expectations", dependencies=[Depends(require_api_key)])
async def list_expectations_endpoint(
    request: Request, session_id: str | None = None
) -> dict:
    """All expectations, or (with ``?session_id=``) those promoted from a run."""
    db = request.app.state.db
    if session_id:
        exps = loop.expectations_for_session(db, session_id)
    else:
        exps = loop.list_expectations(db)
    return {"expectations": [e.to_dict() for e in exps], "count": len(exps)}


@router.get(
    "/expectations/{expectation_id}",
    response_model=None,
    dependencies=[Depends(require_api_key)],
)
async def get_expectation_endpoint(request: Request, expectation_id: str):
    """An expectation plus its fix-history ledger (newest first).

    404 (JSONResponse) when the id is unknown — hence ``response_model=None``.
    """
    db = request.app.state.db
    exp = loop.get_expectation(db, expectation_id)
    if exp is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"Expectation {expectation_id} not found"},
        )
    runs = loop.list_expectation_runs(db, expectation_id)
    return {
        "expectation": exp.to_dict(),
        "runs": [r.to_dict() for r in runs],
        "run_count": len(runs),
    }


@router.post(
    "/expectations/{expectation_id}/runs",
    response_model=None,
    dependencies=[Depends(require_api_key)],
)
async def record_run_endpoint(request: Request, expectation_id: str):
    """Record a rerun's outcome against an expectation (the fix-history ledger).

    Body: ``{"outcome": "pass|regress|unknown", "session_id": "<str>"?,
    "note": "<str>"?}``. 400 on a bad outcome, 404 on an unknown expectation.
    """
    body = await _json_body(request)
    if isinstance(body, JSONResponse):
        return body
    try:
        entry = loop.record_expectation_run(
            request.app.state.db,
            expectation_id,
            outcome=body.get("outcome") or "",
            session_id=body.get("session_id"),
            note=body.get("note"),
        )
    except ValueError as exc:
        # Unknown expectation id -> 404; a bad outcome value -> 400.
        msg = str(exc)
        status = 404 if "unknown expectation_id" in msg else 400
        return JSONResponse(status_code=status, content={"error": msg})
    return JSONResponse(status_code=201, content=entry.to_dict())
