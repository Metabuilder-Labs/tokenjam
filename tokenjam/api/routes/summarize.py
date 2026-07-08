"""/api/v1/summarize/* — the in-process summarize surface (Track B).

Reads for the Lens Summarize screen, plus the guarded apply/undo writes:
  * /summarize/capabilities — which run engines actually work on THIS host, so the
    UI offers only live paths (never a dead "API" when no key is set).
  * /summarize/candidates   — the read-only scan (core `list_candidates`).
  * /summarize/staged       — staged rewrite records (source hash, restored text,
    diff, produced_by, est_tokens_saved) for the review/apply cockpit.
  * /summarize/backups      — applied files that still have a gzip backup (Undo).
  * /summarize/apply, /undo — guarded writes via core (owner + hash-drift +
    symlink-refuse + gzip backup); DRY-RUN by default, go=true to write.

All call core in-process (like `/optimize` → `build_report`); nothing shells out
to `tj`. Routes are sync `def` so their filesystem/DB work runs in Starlette's
threadpool rather than blocking the event loop. The OUTBOUND run/prep/check
endpoints live in their own route module.
"""
from __future__ import annotations

import os
import shutil
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
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


@router.get("/summarize/capabilities", dependencies=[Depends(require_api_key)])
def get_summarize_capabilities() -> dict[str, Any]:
    """Which run engines are usable on this host — the UI greys out dead paths.

    `manual` is always available (no outbound). `api` needs the user's own
    `TJ_ANTHROPIC_API_KEY` in the server env (its presence is the authorization).
    `claude_p` needs the `claude` CLI on the host PATH.
    """
    has_key = bool(os.environ.get("TJ_ANTHROPIC_API_KEY"))
    has_claude = shutil.which("claude") is not None
    return {
        "api": {
            "available": has_key,
            "reason": "" if has_key else "set TJ_ANTHROPIC_API_KEY to enable",
        },
        "claude_p": {
            "available": has_claude,
            "reason": "" if has_claude else "the `claude` CLI is not on this host's PATH",
        },
        "manual": {"available": True, "reason": ""},
    }


@router.get("/summarize/candidates", dependencies=[Depends(require_api_key)])
def get_summarize_candidates(
    request: Request,
    path: str | None = Query(None, description="Explicit path/target to scan (widens to all .md)."),
    recursive: bool = Query(False, description="Walk the repo/PATH for all .md."),
    repo: bool = Query(False, description="Scope to the detected git repo root."),
) -> dict[str, Any]:
    """Read-only summarize scan (core `list_candidates`) — never writes."""
    from tokenjam.core.summarize.candidates import list_candidates

    scan = list_candidates(path=path, config=_config(request), recursive=recursive, repo=repo)
    return scan.to_dict()


@router.get("/summarize/staged", dependencies=[Depends(require_api_key)])
def get_summarize_staged(
    request: Request,
    path: str | None = Query(None, description="Read one staged record by source path; omit for all."),
) -> dict[str, Any]:
    """Read staged rewrite records — all, or one by source path."""
    from tokenjam.core.summarize import session

    config = _config(request)
    if path:
        rec = session.read_staged(config, path)
        return {"staged": [rec] if rec is not None else []}
    return {"staged": session.list_staged(config)}


@router.get("/summarize/backups", dependencies=[Depends(require_api_key)])
def get_summarize_backups(request: Request) -> dict[str, Any]:
    """Applied files that still have a gzip backup — the UI's persistent undo surface.

    Each record carries an ``undoable`` flag (+ ``reason``) so the UI can offer Undo for
    files applied in any earlier session, not just the one that applied them."""
    from tokenjam.core.summarize.backup import list_backups

    return {"backups": list_backups(_config(request))}


# --------------------------------------------------------------------------- #
# Mutating routes — the deliberate departure: Lens's first file-writing action.
# Both call core directly (never re-implement the guards) so every write keeps
# the shipped guarantees: owner-check + content-hash drift-refuse + symlink-refuse
# + gzip backup. Default is DRY-RUN (go=false); the UI sends go=true only on an
# explicit per-file Apply — no flag bypasses a guard.
# --------------------------------------------------------------------------- #
class ApplyRequest(BaseModel):
    path: str | None = None   # one staged file, or all staged when omitted
    go: bool = False          # false = dry-run (returns the plan, writes nothing)


class UndoRequest(BaseModel):
    path: str
    go: bool = False


@router.post("/summarize/apply", dependencies=[Depends(require_api_key)])
def post_summarize_apply(request: Request, body: ApplyRequest) -> dict[str, Any]:
    """Apply staged rewrite(s) to disk via core `apply_staged` (per-file guards inside).

    Dry-run by default: returns `{applied, skipped, dry_run}`; drifted/unowned/symlink
    files are skipped-with-reason, never forced.
    """
    from tokenjam.core.summarize.apply import apply_staged

    return apply_staged(_config(request), body.path, go=body.go)


@router.post("/summarize/undo", dependencies=[Depends(require_api_key)])
def post_summarize_undo(request: Request, body: UndoRequest) -> dict[str, Any]:
    """Restore a file from its backup via core `undo`; refuses (409) on drift/missing."""
    from tokenjam.core.summarize.apply import undo
    from tokenjam.core.summarize.session import SummarizeRefused

    try:
        return undo(_config(request), body.path, go=body.go)
    except SummarizeRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
