"""The self-improve loop's Verify stage (SPEC.md §4 step 6, §10, §11).

Phase 2 (Apply) writes each approved fix to ``applied_fixes.json`` with a
``verify`` scaffold — baseline occurrence/session counts captured at apply
time, and empty slots for what this module fills in: does the signature
actually recur LESS after the fix than before it?

Honesty (SPEC §10): this is a correlational, conservative measurement, never
a causal claim. Three things keep it honest:

  1. **Normalize by exposure.** Raw occurrence counts grow with the size of
     the post-apply window, so a naive "9 before, 2 after" comparison is
     meaningless once the after-window has run for months. Every rate here
     is occurrences-per-session (``count_sessions_in_scope`` is the same
     cheap directory-walk-and-count used for both the baseline and the
     post-apply denominator, so the two sides are comparable).
  2. **Enforcement-aware.** A rung-3/4/5 fix whose hook is still DISABLED
     was never actually live — it cannot have reduced recurrence yet, so it
     gets its own ``enforcement_disabled`` verdict instead of a false
     ``no_change``/``regressed``.
  3. **Admit what can't be measured yet.** A distilled (LLM-merged residual)
     family has no reliable re-matcher here (see ``_matcher_for``) — rather
     than silently under-count and claim a false "improved", it's reported
     ``insufficient_data`` with a reason string.

Never raises: every public function here degrades to a conservative
"can't measure this" result rather than raising, so a single malformed
record or unreadable transcript never sinks a whole rescan pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from tokenjam.core.optimize.analyzers.relearn import (
    _FAMILY_BY_KEY,
    _generic_signature,
    _repo_map_from_db,
    classify_known_family,
    extract_failures_for_session,
    FailureEpisode,
    GROUNDED_TOKENS_PER_OCCURRENCE,
)
from tokenjam.core.optimize.relearn_apply import ENFORCEMENT_RUNGS, RUNG_NOTE
from tokenjam.core.transcript import resolve_projects_root

# --- Tunables ------------------------------------------------------------

#: Minimum sessions observed IN SCOPE since apply before a verdict is trusted
#: (else "insufficient_data" — "check back later").
MIN_POST_SESSIONS_FOR_VERDICT = 5
#: post_rate / baseline_rate <= this -> "improved" (>= a 30% relative drop).
IMPROVED_MAX_RATIO = 0.7
#: post_rate / baseline_rate >= this -> "regressed" ("rate same-or-up").
REGRESSED_MIN_RATIO = 1.0

VERDICT_IMPROVED = "improved"
VERDICT_NO_CHANGE = "no_change"
VERDICT_REGRESSED = "regressed"
VERDICT_INSUFFICIENT_DATA = "insufficient_data"
VERDICT_ENFORCEMENT_DISABLED = "enforcement_disabled"

ESTIMATE_BASIS_VERIFY = (
    "realized savings = avoided occurrences (baseline rate x post-apply sessions, "
    "minus what was actually observed) x the same conservative per-occurrence token "
    "cost the detector uses — estimated / correlational, never a causal claim"
)


# --- Pure verdict logic (no I/O — the unit-tested core) --------------------

def _result(
    *,
    verdict: str,
    reason: str,
    baseline_rate: float | None = None,
    post_rate: float | None = None,
    realized_tokens_saved: int | None = None,
    escalate_candidate: bool = False,
    post_occurrences: int = 0,
    post_sessions: int = 0,
) -> dict[str, Any]:
    return {
        "verdict": verdict,
        "reason": reason,
        "baseline_rate": baseline_rate,
        "post_rate": post_rate,
        "realized_tokens_saved": realized_tokens_saved,
        "escalate_candidate": escalate_candidate,
        "recurrence_since_apply": post_occurrences,
        "post_sessions_since_apply": post_sessions,
    }


def compute_verdict(
    *,
    rung: int | None,
    enforcement: dict[str, Any] | None,
    baseline_occurrences: int | None,
    baseline_total_sessions: int | None,
    baseline_sessions: int | None,
    post_occurrences: int,
    post_sessions: int,
    measurable: bool = True,
    unmeasurable_reason: str | None = None,
    tokens_per_occurrence: int = GROUNDED_TOKENS_PER_OCCURRENCE,
    min_post_sessions: int = MIN_POST_SESSIONS_FOR_VERDICT,
    improved_max_ratio: float = IMPROVED_MAX_RATIO,
    regressed_min_ratio: float = REGRESSED_MIN_RATIO,
) -> dict[str, Any]:
    """The honest, correlational verdict (SPEC §10/§11). Pure — no disk/DB
    access; ``measure_recurrence_since`` supplies the counts.

    Branch order matters: enforcement-disabled is checked BEFORE the data
    volume gate, because a disabled hook was never live — it isn't "not
    enough data yet", it's "nothing to measure yet" for a structural reason.
    """
    is_enforcement_rung = rung in ENFORCEMENT_RUNGS
    enforcement_enabled = bool((enforcement or {}).get("enabled"))
    if is_enforcement_rung and not enforcement_enabled:
        return _result(
            verdict=VERDICT_ENFORCEMENT_DISABLED,
            reason=(
                "enforcement hook is disabled — recurrence isn't expected to drop "
                "until it's enabled from the Applied section"
            ),
            post_occurrences=post_occurrences, post_sessions=post_sessions,
        )

    if not measurable:
        return _result(
            verdict=VERDICT_INSUFFICIENT_DATA,
            reason=unmeasurable_reason or "this pattern can't be re-measured yet",
            post_occurrences=post_occurrences, post_sessions=post_sessions,
        )

    if post_sessions < min_post_sessions:
        return _result(
            verdict=VERDICT_INSUFFICIENT_DATA,
            reason=(
                f"only {post_sessions} session(s) observed since apply — check back "
                f"after at least {min_post_sessions}"
            ),
            post_occurrences=post_occurrences, post_sessions=post_sessions,
        )

    baseline_denominator = baseline_total_sessions or baseline_sessions
    if not baseline_occurrences or not baseline_denominator:
        return _result(
            verdict=VERDICT_INSUFFICIENT_DATA,
            reason="no usable pre-apply baseline was captured for this fix",
            post_occurrences=post_occurrences, post_sessions=post_sessions,
        )

    baseline_rate = baseline_occurrences / baseline_denominator
    post_rate = post_occurrences / post_sessions
    if baseline_rate > 0:
        ratio = post_rate / baseline_rate
    else:
        ratio = 0.0 if post_rate <= 0 else float("inf")

    if ratio <= improved_max_ratio:
        verdict = VERDICT_IMPROVED
    elif ratio >= regressed_min_ratio:
        verdict = VERDICT_REGRESSED
    else:
        verdict = VERDICT_NO_CHANGE

    expected_occurrences = baseline_rate * post_sessions
    avoided = max(0.0, expected_occurrences - post_occurrences)
    realized_tokens_saved = round(avoided * tokens_per_occurrence) if verdict == VERDICT_IMPROVED else 0

    escalate_candidate = rung == RUNG_NOTE and verdict in (VERDICT_NO_CHANGE, VERDICT_REGRESSED)

    reason = (
        f"post-apply rate {post_rate:.3f} occurrence(s)/session over {post_sessions} "
        f"session(s) vs baseline {baseline_rate:.3f}/session"
    )
    return _result(
        verdict=verdict, reason=reason,
        baseline_rate=round(baseline_rate, 4), post_rate=round(post_rate, 4),
        realized_tokens_saved=realized_tokens_saved, escalate_candidate=escalate_candidate,
        post_occurrences=post_occurrences, post_sessions=post_sessions,
    )


# --- Exposure counting (I/O — a cheap directory walk, no transcript parse) ---

def _iter_session_paths(root: Path) -> list[Path]:
    return sorted(root.rglob("*.jsonl")) if root.exists() else []


def _session_mtime(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None


def count_sessions_in_scope(
    projects_root: Path | str | None,
    conn: Any | None,
    repo_filter: str | None,
    *,
    before: datetime | None = None,
    after: datetime | None = None,
) -> int:
    """Cheap exposure count: on-disk session transcripts in scope, no
    transcript parsing — just a directory walk + mtime/repo filter. Used as
    the SAME-basis denominator for both the apply-time baseline
    (``before=applied_at``) and a later verify pass (``after=applied_at``),
    so the two rates are comparable (SPEC's "normalize by exposure").
    Never raises — an unreadable root/file just doesn't count towards N.
    """
    try:
        root = resolve_projects_root(projects_root)
    except Exception:
        return 0
    repo_map = _repo_map_from_db(conn) if conn is not None else {}
    count = 0
    for path in _iter_session_paths(root):
        mtime = _session_mtime(path)
        if mtime is None:
            continue
        if before is not None and mtime > before:
            continue
        if after is not None and mtime <= after:
            continue
        if repo_filter is not None:
            repo = repo_map.get(path.stem, "unknown")
            if repo != repo_filter:
                continue
        count += 1
    return count


# --- Signature re-matching (which post-apply failures count) ---------------

def _matcher_for(rec: dict[str, Any]) -> tuple[Callable[[FailureEpisode], bool], bool, str | None]:
    """(matcher, measurable, reason). ``measurable=False`` means this fix's
    signature can't be reliably re-identified in fresh transcripts yet — the
    caller reports ``insufficient_data`` rather than risk a false "improved"
    from an under-counted match (SPEC §10 honesty).

    - A known static family (``cwd_confusion`` etc.) re-matches via the exact
      same ``classify_known_family`` regex the detector clustered on.
    - A plain residual (non-distilled) generic bucket re-matches via exact
      equality of ``_generic_signature`` — the same normalization the
      detector used to build that bucket's signature string in the first
      place.
    - A ``distilled:*`` family (an LLM merged several generic buckets under
      one root cause) has no cheap, deterministic re-matcher here — doing so
      would mean re-running the distill pass on every fresh failure just to
      verify one applied fix, which isn't worth the $ cost. Reported as not
      measurable rather than guessed at.
    """
    family_key = rec.get("family_key")
    signature = rec.get("signature")

    if family_key and str(family_key).startswith("distilled:"):
        return (
            lambda f: False, False,
            "distilled (LLM-merged) pattern family — recurrence re-matching isn't "
            "implemented yet for this fix",
        )

    if family_key and family_key in _FAMILY_BY_KEY:
        def _match_family(f: FailureEpisode, fk: str = family_key) -> bool:
            return classify_known_family(f.tool_name, f.error_text, f.label) == fk

        return _match_family, True, None

    if not family_key and signature:
        def _match_generic(f: FailureEpisode, sig: str = signature) -> bool:
            return _generic_signature(f.tool_name, f.error_text) == sig

        return _match_generic, True, None

    return (
        lambda f: False, False,
        "unrecognized pattern family — recurrence re-matching isn't implemented "
        "yet for this fix",
    )


def _scope_repo_filter(rec: dict[str, Any]) -> str | None:
    """``None`` = unscoped (matches every repo — a user-global fix, or a
    project fix whose repo couldn't be resolved). A project-scoped fix
    filters to the basename of its recorded ``repo_root`` — best-effort:
    the write target's git root is normally the same repo the relearn's
    sessions were seen in."""
    if rec.get("scope") != "project":
        return None
    repo_root = rec.get("repo_root")
    if not repo_root:
        return None
    return Path(repo_root).name


def _parse_applied_at(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


@dataclass
class RecurrenceMeasurement:
    post_occurrences: int
    post_sessions: int
    measurable: bool
    reason: str | None


def measure_recurrence_since(
    rec: dict[str, Any],
    *,
    conn: Any | None = None,
    projects_root: Path | str | None = None,
) -> RecurrenceMeasurement:
    """Walk on-disk sessions newer than ``rec['applied_at']`` (in the fix's
    scope) and count: how many total (the exposure denominator) and how many
    contained a matching-signature occurrence. Never raises — any failure
    degrades to ``measurable=False``."""
    applied_at = _parse_applied_at(rec.get("applied_at"))
    if applied_at is None:
        return RecurrenceMeasurement(0, 0, False, "no applied_at recorded for this fix")

    matcher, measurable, reason = _matcher_for(rec)
    if not measurable:
        return RecurrenceMeasurement(0, 0, False, reason)

    try:
        root = resolve_projects_root(projects_root)
    except Exception:
        return RecurrenceMeasurement(0, 0, False, "couldn't resolve the sessions directory")

    repo_map = _repo_map_from_db(conn) if conn is not None else {}
    repo_filter = _scope_repo_filter(rec)

    post_sessions = 0
    post_occurrences = 0
    for path in _iter_session_paths(root):
        mtime = _session_mtime(path)
        if mtime is None or mtime <= applied_at:
            continue
        session_id = path.stem
        repo = repo_map.get(session_id, "unknown")
        if repo_filter is not None and repo != repo_filter:
            continue
        post_sessions += 1
        try:
            failures = extract_failures_for_session(session_id, repo, root)
        except Exception:
            continue
        post_occurrences += sum(1 for f in failures if matcher(f))

    return RecurrenceMeasurement(post_occurrences, post_sessions, True, None)


# --- Per-record + fleet-wide rescan orchestration --------------------------

def recompute_verify_for_record(
    rec: dict[str, Any],
    *,
    conn: Any | None = None,
    projects_root: Path | str | None = None,
) -> dict[str, Any]:
    """The new ``verify`` fields for one applied (non-reverted) record —
    merges on top of its existing ``verify`` dict; never raises."""
    from tokenjam.utils.time_parse import utcnow

    measurement = measure_recurrence_since(rec, conn=conn, projects_root=projects_root)
    verify = rec.get("verify") or {}
    result = compute_verdict(
        rung=rec.get("rung"),
        enforcement=rec.get("enforcement"),
        baseline_occurrences=verify.get("baseline_occurrences"),
        baseline_total_sessions=verify.get("baseline_total_sessions"),
        baseline_sessions=verify.get("baseline_sessions"),
        post_occurrences=measurement.post_occurrences,
        post_sessions=measurement.post_sessions,
        measurable=measurement.measurable,
        unmeasurable_reason=measurement.reason,
    )
    result["last_checked_at"] = utcnow().isoformat()
    return result


def rescan_all(
    config: Any,
    conn: Any | None = None,
    *,
    projects_root: Path | str | None = None,
) -> dict[str, int]:
    """Recompute verify for every applied (non-reverted) fix in the ledger —
    the entry point the background relearn-rescan job calls on the SAME
    cadence as the detector (SPEC §4 step 6: "on each rescan"). Never raises;
    a single bad record is skipped, not fatal to the whole pass."""
    from tokenjam.core.optimize import relearn_apply

    checked = 0
    updated = 0
    for rec in relearn_apply.list_applied(config):
        if rec.get("state") == "reverted":
            continue
        checked += 1
        try:
            new_fields = recompute_verify_for_record(rec, conn=conn, projects_root=projects_root)
            merged = {**(rec.get("verify") or {}), **new_fields}
            relearn_apply.set_verify(config, rec["id"], merged)
            updated += 1
        except Exception:
            continue
    return {"checked": checked, "updated": updated}


# --- Compound ledger (SPEC §4 step 7 / §11) --------------------------------

def compound_ledger(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The Applied section's top-of-page summary: total realized savings
    across every VERIFIED (not just applied) fix, plus a verdict breakdown.
    Reverted fixes never contribute — their write no longer exists."""
    total_saved = 0
    verified = improved = no_change = regressed = enforcement_disabled = insufficient = 0
    for rec in records:
        if rec.get("state") == "reverted":
            continue
        verify = rec.get("verify") or {}
        verdict = verify.get("verdict")
        if verdict is None:
            continue
        verified += 1
        if verdict == VERDICT_IMPROVED:
            improved += 1
            total_saved += verify.get("realized_tokens_saved") or 0
        elif verdict == VERDICT_NO_CHANGE:
            no_change += 1
        elif verdict == VERDICT_REGRESSED:
            regressed += 1
        elif verdict == VERDICT_ENFORCEMENT_DISABLED:
            enforcement_disabled += 1
        elif verdict == VERDICT_INSUFFICIENT_DATA:
            insufficient += 1
    return {
        "total_realized_tokens_saved": total_saved,
        "verified_count": verified,
        "improved_count": improved,
        "no_change_count": no_change,
        "regressed_count": regressed,
        "enforcement_disabled_count": enforcement_disabled,
        "insufficient_data_count": insufficient,
        "estimate_basis": ESTIMATE_BASIS_VERIFY,
    }
