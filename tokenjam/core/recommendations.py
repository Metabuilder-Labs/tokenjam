"""Recommendation-outcome ledger — did a recommendation get acted on, and what
did it actually recover?

tokenjam is advisory end to end (suggest-mode proxy, manual routing-config
drop-in, dry-run-default summarize). Its delivered value is therefore capped by
which recommendations get *applied* — yet nothing recorded that. This module is
the outcome half of the loop: it records the two actions tokenjam can observe
directly (a `summarize apply --go`, a `tj optimize --export-config`) and, for
the `downsize` analyzer, does **post-hoc adoption detection** — after an export,
it measures whether the following days' span mix actually shifted off the
recommended premium models and marks the recommendation adopted/ignored with a
**measured** (not estimated) delta.

Storage is an append-only JSONL sink next to the DB, mirroring
``core/savings_log.py``. Two forces make this the right home rather than the
``savings_ledger`` table:

* The write paths run where a DuckDB lock is unavailable. ``tj summarize`` is a
  ``no_db_commands`` command and ``tj optimize --export-config`` runs against the
  read-only serve shim when the daemon holds the write lock. A lock-free sink
  writes from any context.
* ``savings_ledger`` (#221) is the proxy would-have-saved meter with the
  invariant ``realized`` is always FALSE — mixing measured/realized outcomes into
  it would break that documented contract.

Honesty discipline (CLAUDE.md Rule 14) is preserved: every stored figure is
either a labelled ESTIMATE (recorded at recommendation time) or a MEASURED shift
carrying an explicit correlation-not-causation basis string. Nothing here ever
says "tokenjam saved you".
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tokenjam.core.savings_log import hooks_dir
from tokenjam.utils.time_parse import utcnow

# Outcome kinds.
KIND_SUMMARIZE_APPLY = "summarize_apply"
KIND_CONFIG_EXPORT = "config_export"
KIND_DOWNSIZE_ADOPTION = "downsize_adoption"

# Adoption-detection tuning. A config export is only resolved once at least
# MIN_OBSERVATION_DAYS of post-export telemetry exist, and the comparison window
# caps at OBSERVATION_DAYS. The recommended premium models' measured spend rate
# must fall by at least ADOPTION_MIN_REL_DROP (relative) for the export to count
# as adopted; anything less is recorded as ignored (measured, delta 0). A
# pre-export baseline shorter than MIN_PRE_WINDOW_DAYS isn't resolved at all —
# a rate estimate over less than a day is too noisy to trust.
OBSERVATION_DAYS = 14
MIN_OBSERVATION_DAYS = 7
ADOPTION_MIN_REL_DROP = 0.25
MIN_PRE_WINDOW_DAYS = 1.0

# The honest basis string on every measured adoption record. It is a correlation
# (the user may have changed routing for unrelated reasons), never a causal claim.
ADOPTION_BASIS = (
    "measured change in spend on the recommended premium models over the "
    "observation window after the export; a correlation with the recommendation, "
    "not proof tokenjam caused the shift"
)


def recommendations_path(config) -> Path:
    """The append-only JSONL of recommendation outcomes (next to the DB)."""
    return hooks_dir(config) / "recommendations.jsonl"


# ---------------------------------------------------------------------------
# Sink I/O (fail-safe, mirroring savings_log)
# ---------------------------------------------------------------------------

def append_outcome(config, record: dict) -> None:
    """Append one outcome to the JSONL sink. Fail-safe (never raises).

    ``ts`` defaults to now (UTC, tz-aware — Rule 9). ``default=str`` keeps the
    write robust against any non-serializable value.
    """
    try:
        p = recommendations_path(config)
        p.parent.mkdir(parents=True, exist_ok=True)
        rec = dict(record)
        rec.setdefault("ts", utcnow().isoformat())
        with open(p, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        pass


def read_outcomes(config) -> list[dict]:
    """Read all outcome records. Tolerant of a missing file and of a
    partially-written trailing line (append-only can race a read)."""
    p = recommendations_path(config)
    out: list[dict] = []
    try:
        if not p.exists():
            return out
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    except Exception:
        return out
    return out


# ---------------------------------------------------------------------------
# Recording the two directly-observable actions
# ---------------------------------------------------------------------------

def record_summarize_apply(config, *, path: str, est_tokens_saved: int) -> None:
    """Record that a `summarize apply --go` rewrote one file on disk.

    The rewrite itself is a real action, but the token recovery is an ESTIMATE
    (a char/4 reduction of the prompt file — the realized per-call saving depends
    on how often the prompt is used). So ``measured`` is False and the figure
    lands in the estimated column.
    """
    append_outcome(config, {
        "outcome_id": f"summarize:{path}:{utcnow().isoformat()}",
        "kind": KIND_SUMMARIZE_APPLY,
        "source": "summarize",
        "status": "applied",
        "target": path,
        "provider": None,
        "pricing_mode": "unknown",
        "measured": False,
        "recovered_usd": 0.0,
        "recovered_tokens": 0,
        "estimated_usd": 0.0,
        "estimated_tokens": int(est_tokens_saved or 0),
        "basis": "per-call token reduction estimated from the file rewrite (char/4)",
        "detail": {},
    })


def record_config_export(
    config, *, target: str, export_path: str, downgrade,
    pricing_mode: str, provider: str | None,
    since: datetime, until: datetime, window_days: float,
) -> str | None:
    """Record a `tj optimize --export-config` and stash the baseline that
    adoption detection needs.

    ``downgrade`` is the ``DowngradeFinding`` (or None). Its ``suggestions``
    (``{premium_model: cheaper_alt}``) plus the analysis window are persisted so
    :func:`detect_downsize_adoption` can later compare the post-export span mix
    against this exact baseline. Returns the outcome_id (or None when there is no
    downgrade recommendation to track — a bare export with nothing to adopt).
    """
    suggestions = dict(getattr(downgrade, "suggestions", {}) or {})
    outcome_id = f"export:{target}:{until.isoformat()}"
    append_outcome(config, {
        "outcome_id": outcome_id,
        "kind": KIND_CONFIG_EXPORT,
        "source": f"export:{target}",
        "status": "exported",
        "target": export_path,
        "provider": provider,
        "pricing_mode": pricing_mode,
        "measured": False,
        "recovered_usd": 0.0,
        "recovered_tokens": 0,
        "estimated_usd": float(getattr(downgrade, "estimated_recoverable_usd", 0.0) or 0.0),
        "estimated_tokens": int(getattr(downgrade, "estimated_recoverable_tokens", 0) or 0),
        "basis": "estimated recoverable over the analysis window (downsize finding)",
        "detail": {
            "suggestions": suggestions,
            "since": since.isoformat(),
            "until": until.isoformat(),
            "window_days": float(window_days),
        },
    })
    return outcome_id if suggestions else None


# ---------------------------------------------------------------------------
# Post-hoc adoption detection (downsize)
# ---------------------------------------------------------------------------

_DATE_SUFFIX = re.compile(r"^(.*)-(\d{8})$")
_TAG_SUFFIX = re.compile(r"^(.*?)\[[^\]]*\]$")


def _norm_model(model: str | None) -> str:
    """Strip a bracketed context tag (``[1m]``) and a trailing ``-YYYYMMDD`` date
    so a stored suggestion key (``claude-opus-4-8``) matches the dated span model
    (``claude-opus-4-8-20260115``). Mirrors the pricing/downgrade normalisation."""
    if not model:
        return ""
    m = model.strip()
    tag = _TAG_SUFFIX.match(m)
    if tag and tag.group(1):
        m = tag.group(1)
    dated = _DATE_SUFFIX.match(m)
    if dated:
        m = dated.group(1)
    return m


def _parse_ts(value: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(value)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _premium_usage(conn, premium_norm: set[str], start: datetime, end: datetime) -> tuple[float, int]:
    """(cost_usd, tokens) spent on the recommended premium models in [start, end).

    Grouped by raw model then matched in Python via :func:`_norm_model` so dated /
    tagged model ids match the normalised suggestion keys. DuckDB timestamp rule:
    ``start_time`` is TIMESTAMPTZ, compared against tz-aware bounds."""
    rows = conn.execute(
        "SELECT model, "
        "COALESCE(SUM(cost_usd), 0.0), "
        "COALESCE(SUM(input_tokens), 0) + COALESCE(SUM(output_tokens), 0) "
        "+ COALESCE(SUM(cache_tokens), 0) "
        "FROM spans WHERE model IS NOT NULL AND start_time >= $1 AND start_time < $2 "
        "GROUP BY model",
        [start, end],
    ).fetchall()
    usd = 0.0
    tok = 0
    for model, cost, tokens in rows:
        if _norm_model(model) in premium_norm:
            usd += float(cost or 0.0)
            tok += int(tokens or 0)
    return usd, tok


def detect_downsize_adoption(
    conn, config, *, now: datetime | None = None,
    observation_days: int = OBSERVATION_DAYS,
    min_observation_days: int = MIN_OBSERVATION_DAYS,
) -> list[dict]:
    """Resolve ripe, unresolved ``config_export`` outcomes into measured
    adoption records and append them to the sink (idempotent).

    For each export older than ``min_observation_days``, compare the recommended
    premium models' spend *rate* in the recommendation's own analysis window
    against their rate over the post-export observation window. A relative drop of
    at least :data:`ADOPTION_MIN_REL_DROP` marks the recommendation adopted, with
    the measured recovered figure = (rate drop) × (observation days). Returns the
    newly-created records (empty when nothing was ripe / new).

    Requires a direct DuckDB connection. Runs server-side on the daemon (which
    owns the connection) and opportunistically from ``tj optimize`` when the
    daemon is down — either way the shared on-disk sink is what ``tj savings``
    reads, so the two triggers never need to coordinate.
    """
    now = now or utcnow()
    outcomes = read_outcomes(config)
    resolved = {
        (o.get("detail") or {}).get("export_outcome_id")
        for o in outcomes
        if o.get("kind") == KIND_DOWNSIZE_ADOPTION
    }
    new: list[dict] = []
    for o in outcomes:
        if o.get("kind") != KIND_CONFIG_EXPORT:
            continue
        oid = o.get("outcome_id")
        if not oid or oid in resolved:
            continue
        detail = o.get("detail") or {}
        suggestions = detail.get("suggestions") or {}
        if not suggestions:
            continue
        export_ts = _parse_ts(o.get("ts", ""))
        since_pre = _parse_ts(detail.get("since", ""))
        until_pre = _parse_ts(detail.get("until", ""))
        if export_ts is None or since_pre is None or until_pre is None:
            continue

        post_end = min(now, export_ts + timedelta(days=observation_days))
        obs_days = (post_end - export_ts).total_seconds() / 86400.0
        if obs_days < min_observation_days:
            continue  # not enough post-export telemetry yet — leave it pending

        premium_norm = {_norm_model(m) for m in suggestions.keys()} - {""}
        if not premium_norm:
            continue
        pre_days = float(detail.get("window_days") or 0.0)
        if pre_days <= 0:
            pre_days = (until_pre - since_pre).total_seconds() / 86400.0
        if pre_days < MIN_PRE_WINDOW_DAYS:
            # Sub-day baseline (missing window_days + a same-day export) makes
            # pre_rate_usd/tok unreliable — dividing by a near-zero window
            # inflates the rate and can spuriously flip the adoption verdict.
            # Leave it pending rather than resolving off a meaningless rate.
            continue

        pre_usd, pre_tok = _premium_usage(conn, premium_norm, since_pre, until_pre)
        post_usd, post_tok = _premium_usage(conn, premium_norm, export_ts, post_end)

        pre_rate_usd, post_rate_usd = pre_usd / pre_days, post_usd / obs_days
        pre_rate_tok, post_rate_tok = pre_tok / pre_days, post_tok / obs_days

        # Relative drop keys on dollars when priced, else on tokens (subscription
        # / local users have zero-cost spans but real token shifts).
        if pre_rate_usd > 0:
            rel = (pre_rate_usd - post_rate_usd) / pre_rate_usd
        elif pre_rate_tok > 0:
            rel = (pre_rate_tok - post_rate_tok) / pre_rate_tok
        else:
            rel = 0.0
        adopted = rel >= ADOPTION_MIN_REL_DROP

        rec_usd = max(0.0, pre_rate_usd - post_rate_usd) * obs_days if adopted else 0.0
        rec_tok = int(max(0.0, pre_rate_tok - post_rate_tok) * obs_days) if adopted else 0

        record = {
            "outcome_id": f"{oid}:adoption",
            "ts": now.isoformat(),
            "kind": KIND_DOWNSIZE_ADOPTION,
            "source": "downsize",
            "status": "adopted" if adopted else "ignored",
            "target": o.get("target"),
            "provider": o.get("provider"),
            "pricing_mode": o.get("pricing_mode", "unknown"),
            "measured": True,
            "recovered_usd": round(rec_usd, 6),
            "recovered_tokens": rec_tok,
            "estimated_usd": float(o.get("estimated_usd", 0.0) or 0.0),
            "estimated_tokens": int(o.get("estimated_tokens", 0) or 0),
            "basis": ADOPTION_BASIS,
            "detail": {
                "export_outcome_id": oid,
                "observation_days": round(obs_days, 1),
                "premium_models": sorted(premium_norm),
                "relative_drop": round(rel, 3),
                "pre_rate_usd": round(pre_rate_usd, 6),
                "post_rate_usd": round(post_rate_usd, 6),
                "pre_rate_tokens": int(pre_rate_tok),
                "post_rate_tokens": int(post_rate_tok),
            },
        }
        append_outcome(config, record)
        new.append(record)
    return new


# ---------------------------------------------------------------------------
# Aggregation for tj savings / Lens
# ---------------------------------------------------------------------------

def summarize_outcomes(outcomes: list[dict]) -> dict:
    """Aggregate outcomes into the two honest columns the UI needs:
    **estimated-recoverable** (recorded at recommendation time) vs
    **measured-recovered** (post-hoc adoption).

    Dollar totals are summed only over ``api`` pricing-mode records (subscription
    / local / unknown $ figures mislead — Rule 14 framing); token totals sum over
    all records. ``rows`` carries each outcome for a detail table.

    Deduped by ``outcome_id`` (first occurrence wins) before accumulating: the
    sink can carry more than one record for the same logical event — e.g. a
    user running ``tj optimize --export-config`` twice inside the same
    analysis window produces two ``config_export`` rows with an identical
    ``outcome_id`` (keyed on the window's ``until``), and a racing daemon +
    CLI can both append an adoption record for the same export (see
    :func:`detect_downsize_adoption`). Without this, either duplicate inflates
    its column by counting the same export/adoption more than once. A record
    with no ``outcome_id`` is never deduped (nothing to key it on).
    """
    seen_ids: set[str] = set()
    deduped: list[dict] = []
    for o in outcomes:
        oid = o.get("outcome_id")
        if oid:
            if oid in seen_ids:
                continue
            seen_ids.add(oid)
        deduped.append(o)
    outcomes = deduped

    est_usd = est_tok = meas_usd = meas_tok = 0.0
    est_count = adopted = ignored = 0
    rows: list[dict] = []
    for o in outcomes:
        api = o.get("pricing_mode") == "api"
        if o.get("measured"):
            meas_tok += int(o.get("recovered_tokens", 0) or 0)
            if api:
                meas_usd += float(o.get("recovered_usd", 0.0) or 0.0)
            if o.get("status") == "adopted":
                adopted += 1
            elif o.get("status") == "ignored":
                ignored += 1
        else:
            est_tok += int(o.get("estimated_tokens", 0) or 0)
            if api:
                est_usd += float(o.get("estimated_usd", 0.0) or 0.0)
            est_count += 1
        rows.append(o)
    return {
        "estimated_recoverable_usd": round(est_usd, 6),
        "estimated_recoverable_tokens": int(est_tok),
        "measured_recovered_usd": round(meas_usd, 6),
        "measured_recovered_tokens": int(meas_tok),
        "actions_recorded": est_count,
        "adopted": adopted,
        "ignored": ignored,
        "rows": rows,
    }
