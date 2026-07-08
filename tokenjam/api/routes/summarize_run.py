"""/api/v1/summarize/{run,prep,check} — the OUTBOUND summarize run surface (Track B).

Split from the read/apply surface (routes/summarize.py) on purpose: `run` makes
`tj serve` an OUTBOUND LLM caller for the FIRST time — a posture shift the
maintainer reserved (where the key/outbound config lives, auth-to-spend, sync vs
async). Isolating it here lets the reads/apply layer merge while this surface
waits on that design call.

  run   = OUTBOUND (api | claude-p): prep -> deliver -> check -> stage, one file.
  prep  = manual step 1 (NO outbound): wrap structure -> the copy-paste prompt.
  check = manual step 2 (NO outbound): verify the pasted-back summary -> stage.

Per-file by design — the UI loops `run` with a progress bar and streams results
into the review box, so no async-job infra is needed. All call core
(session.prepare / delivery.summarize_via / session.check) directly; nothing
shells out to `tj`. Routes are sync `def` (blocking I/O -> Starlette threadpool).

`_config` is duplicated from routes/summarize.py so this module stands alone; a
shared helper can fold the two once both land.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from tokenjam.api.deps import require_api_key

router = APIRouter()


def _config(request: Request):
    config = request.app.state.config
    if config is None:
        raise HTTPException(
            status_code=503, detail="Server not fully initialised (config missing)."
        )
    return config


class RunRequest(BaseModel):
    path: str
    mode: str                  # "api" | "claude-p" ("claude_p" accepted); manual uses /prep+/check
    ratio: float = 0.5


class PrepRequest(BaseModel):
    path: str
    ratio: float = 0.5


class CheckRequest(BaseModel):
    path: str
    summary: str
    source_hash: str           # the prep's source_sha256 — check hash-guards against it


@router.post("/summarize/run", dependencies=[Depends(require_api_key)])
def post_summarize_run(request: Request, body: RunRequest) -> dict[str, Any]:
    """Run the summarizer on ONE file server-side (the only OUTBOUND route).

    `api` bills the user's `TJ_ANTHROPIC_API_KEY`; `claude-p` spawns the host's
    `claude` (subscription limits). Per-file by design — the UI loops it with a
    progress bar; results stream into the review box. `manual` is NOT here.
    """
    from tokenjam.core.summarize.delivery import DeliveryError, summarize_via
    from tokenjam.core.summarize.session import SummarizeRefused

    mode = body.mode.replace("_", "-")
    if mode not in ("api", "claude-p"):
        raise HTTPException(
            status_code=400,
            detail="mode must be 'api' or 'claude-p' (manual uses /summarize/prep + /summarize/check).",
        )
    try:
        result = summarize_via(_config(request), body.path, mode, ratio=body.ratio)
    except SummarizeRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DeliveryError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        "verdict": result.verdict.to_dict() if result.verdict is not None else None,
        "amortization": result.amortization.to_dict() if result.amortization is not None else None,
        "skipped_note": result.skipped_note,
        "cost_unknown": result.cost_unknown,
    }


@router.post("/summarize/prep", dependencies=[Depends(require_api_key)])
def post_summarize_prep(request: Request, body: PrepRequest) -> dict[str, Any]:
    """Manual step 1 (no outbound): wrap structure → the prompt to summarize in a new
    session. `wrapped_prompt` is empty (with a note) when the file is below the prose gate."""
    from tokenjam.core.summarize.session import SummarizeRefused, prepare

    try:
        return prepare(path=body.path, ratio=body.ratio).to_dict()
    except SummarizeRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=404, detail=f"cannot read {body.path}: {exc}") from exc


@router.post("/summarize/check", dependencies=[Depends(require_api_key)])
def post_summarize_check(request: Request, body: CheckRequest) -> dict[str, Any]:
    """Manual step 2 (no outbound): verify the pasted-back summary + stage it if structure
    holds. 409 if the file changed since prep (the summary was built against another version)."""
    from tokenjam.core.summarize.session import SummarizeRefused, check

    try:
        return check(_config(request), body.path, body.summary, body.source_hash).to_dict()
    except SummarizeRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
