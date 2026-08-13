"""GET /api/v1/cost — aggregated cost data."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request

from tokenjam.api.deps import require_api_key
from tokenjam.core.cycle import cycle_bounds, effective_cycle_start_day
from tokenjam.core.data_span import available_data_span
from tokenjam.core.framing import (
    PERSONAS,
    WindowSummary,
    compute_framing,
    plan_determination_mix,
)
from tokenjam.core.models import CostFilters
from tokenjam.core.persona_scope import add_persona_clause
from tokenjam.core.pricing_coverage import coverage_note, summarize_pricing_coverage
from tokenjam.utils.time_parse import parse_since, utcnow

router = APIRouter(dependencies=[Depends(require_api_key)])


def _cycle_block(config) -> dict:
    """Current billing-cycle bounds for the run-rate caption (#138).

    Honors `[budget.<provider>] cycle_start_day` when configured, falling back
    to the calendar month (start_day=1). The UI projects the run-rate to
    `cycle.end` instead of assuming a calendar-month boundary, so a user on a
    non-calendar cycle gets an honest "by <cycle end>" caption.
    """
    now = utcnow()
    start_day = effective_cycle_start_day(config)
    cs, ce = cycle_bounds(now, start_day)
    days_remaining = max(1, math.ceil((ce - now).total_seconds() / 86400.0))
    return {
        "start": int(cs.timestamp()),
        "end": int(ce.timestamp()),
        "days_remaining": days_remaining,
        "start_day": start_day,
    }


def _window_series(conn, agent_id, since_dt, until_dt) -> dict:
    """Window-bucketed cost/tokens per (bucket, agent, model) for charting.

    Buckets hourly for short windows (≤ 2 days) and daily otherwise, returning
    epoch-second bucket keys plus the window bounds so the UI can render the
    FULL selected window with zero-fill and window-matched tick density (#133) —
    the grouped `rows` collapse the agent/model dimension and only cover days
    with data. Buckets are UTC (AT TIME ZONE 'UTC', Rule 1).
    """
    start = since_dt
    end = until_dt or utcnow()
    span_days = ((end - start).total_seconds() / 86400.0) if start is not None else None
    bucket = "hour" if (span_days is not None and span_days <= 2) else "day"
    out: dict = {
        "series": [],
        "series_bucket": bucket,
        "window_start": int(start.timestamp()) if start is not None else None,
        "window_end": int(end.timestamp()),
    }
    if conn is None:
        return out
    # $1 is the date_trunc unit (a controlled 'hour'/'day' literal, bound as a
    # parameter — no f-string SQL, Rule 7).
    clauses = ["model IS NOT NULL"]
    params: list = [bucket]
    if agent_id:
        params.append(agent_id)
        clauses.append("agent_id = $" + str(len(params)))
    if since_dt is not None:
        params.append(since_dt)
        clauses.append("start_time >= $" + str(len(params)))
    if until_dt is not None:
        params.append(until_dt)
        clauses.append("start_time <= $" + str(len(params)))
    where = " AND ".join(clauses)
    # Grouped by (bucket, agent, model, provider) with the full token-component
    # split per row. This is the reusable group-by series shape the cost charts
    # (#213 stacked by model/agent) and the future analytics explorer (#210)
    # consume — callers pick which dimension(s) to stack client-side.
    sql = (
        "SELECT CAST(epoch(date_trunc($1, start_time AT TIME ZONE 'UTC')) AS BIGINT) AS b, "
        "agent_id, model, provider, COALESCE(SUM(cost_usd), 0.0), "
        "COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0), "
        "COALESCE(SUM(cache_tokens), 0), COALESCE(SUM(cache_write_tokens), 0) "
        "FROM spans WHERE " + where + " GROUP BY b, agent_id, model, provider ORDER BY b"
    )
    rows = conn.execute(sql, params).fetchall()
    out["series"] = [
        {
            "bucket": int(r[0]), "agent_id": r[1], "model": r[2], "provider": r[3],
            "cost_usd": float(r[4] or 0.0),
            "input_tokens": int(r[5] or 0), "output_tokens": int(r[6] or 0),
            "cache_tokens": int(r[7] or 0), "cache_write_tokens": int(r[8] or 0),
        }
        for r in rows
    ]
    return out


def _framing_block(db, config, agent_id, total_cost, total_tokens) -> dict:
    """Build the plan-tier framing block (single source, #110), window-independent
    plan mix (#177). Shared by /cost and /cost/cache so neither re-derives the
    suppression rules client-side."""
    conn = getattr(db, "conn", None)
    mix = plan_determination_mix(conn, agent_id) if conn is not None else {}
    return compute_framing(
        config,
        WindowSummary(
            total_cost_usd=total_cost,
            total_tokens=total_tokens,
            sessions=sum(mix.values()),
            plan_tier_mix=mix,
        ),
    ).to_dict()


def _cache_series(conn, agent_id, since_dt, until_dt) -> dict:
    """Per-bucket cache hit-rate + cumulative captured cache savings (#212).

    Hit-rate is `cache_read / (cache_read + input)` per bucket — a token ratio,
    always meaningful regardless of plan tier. "Captured" savings is the dollars
    already saved by the cache reads that DID happen, priced per (provider,
    model) as `cache_read_tokens × (input_rate − cache_read_rate)` — a measured
    figure (so it's "captured", not "estimated"). The recoverable *additional*
    estimate is attached by the route from the cache analyzer. Buckets are UTC
    (Rule 1) and mirror `_window_series` so the chart x-axes line up.
    """
    from tokenjam.core.optimize.span_pricing import rates_at

    start = since_dt
    end = until_dt or utcnow()
    span_days = ((end - start).total_seconds() / 86400.0) if start is not None else None
    bucket = "hour" if (span_days is not None and span_days <= 2) else "day"
    out: dict = {
        "series": [],
        "series_bucket": bucket,
        "window_start": int(start.timestamp()) if start is not None else None,
        "window_end": int(end.timestamp()),
    }
    if conn is None:
        return out
    clauses = ["model IS NOT NULL"]
    params: list = [bucket]
    if agent_id:
        params.append(agent_id)
        clauses.append("agent_id = $" + str(len(params)))
    if since_dt is not None:
        params.append(since_dt)
        clauses.append("start_time >= $" + str(len(params)))
    if until_dt is not None:
        params.append(until_dt)
        clauses.append("start_time <= $" + str(len(params)))
    where = " AND ".join(clauses)
    sql = (
        "SELECT CAST(epoch(date_trunc($1, start_time AT TIME ZONE 'UTC')) AS BIGINT) AS b, "
        "provider, model, COALESCE(SUM(input_tokens), 0), "
        "COALESCE(SUM(cache_tokens), 0), COALESCE(SUM(cache_write_tokens), 0) "
        "FROM spans WHERE " + where + " GROUP BY b, provider, model ORDER BY b"
    )
    rows = conn.execute(sql, params).fetchall()

    # Fold per-(provider, model) rows into per-bucket aggregates + priced savings.
    by_bucket: dict[int, dict] = {}
    for b, provider, model, in_tok, cr_tok, cw_tok in rows:
        b = int(b)
        agg = by_bucket.setdefault(
            b, {"input": 0, "cache_read": 0, "cache_write": 0, "captured_usd": 0.0}
        )
        agg["input"] += int(in_tok or 0)
        agg["cache_read"] += int(cr_tok or 0)
        agg["cache_write"] += int(cw_tok or 0)
        # Priced at the bucket's own instant, not today's: the pricing table
        # has a time axis, and this is an OBSERVED figure over a past window.
        # The bucket is already a UTC hour or day boundary, so it can never
        # straddle a rate change (every `valid_from` is a date).
        rates = rates_at(
            provider, model, datetime.fromtimestamp(b, tz=timezone.utc),
        ) if model else None
        # Only claim captured savings when the model has a real discounted
        # cache-read rate; otherwise stay silent (honest — no invented savings).
        if rates is not None and rates.cache_read_per_mtok > 0:
            delta = max(0.0, rates.input_per_mtok - rates.cache_read_per_mtok)
            agg["captured_usd"] += (int(cr_tok or 0) * delta) / 1_000_000.0

    series = []
    for b in sorted(by_bucket):
        a = by_bucket[b]
        denom = a["input"] + a["cache_read"]
        hit_rate = (a["cache_read"] / denom) if denom > 0 else 0.0
        series.append({
            "bucket": b,
            "input_tokens": a["input"],
            "cache_read_tokens": a["cache_read"],
            "cache_write_tokens": a["cache_write"],
            "hit_rate": round(hit_rate, 4),
            "captured_usd": round(a["captured_usd"], 8),
            "captured_tokens": a["cache_read"],
        })
    out["series"] = series
    return out


# Component a given analyzer's recoverable savings primarily acts on. "call"
# means whole-call (a model swap or call elimination), NOT a single token
# component — used for analyzers whose savings can't be honestly pinned to one
# component (downsize / script / reuse). New analyzers default to "call", so the
# overlay stays registry-driven (#211).
_ANALYZER_COMPONENT = {
    "cache": "input",     # raising cache efficacy shrinks the input component
    "trim": "input",      # trims low-significance input/prompt tokens
    "downsize": "call",   # whole-call model swap
    "script": "call",     # eliminates whole deterministic calls
    "reuse": "call",      # eliminates whole repeated-planning calls
}
_ANALYZER_TITLE = {
    "downsize": "Downsize", "cache": "Cache", "script": "Script",
    "trim": "Trim", "reuse": "Reuse",
}

_COMPONENT_LABELS = [
    ("input", "Input"),
    ("output", "Output"),
    ("cache_read", "Cache read"),
    ("cache_write", "Cache write"),
]


def _component_costs(conn, agent_id, since_dt, until_dt, persona=None) -> dict:
    """Split the window's spend into the four token components (#211).

    Computes each component's cost per (provider, model, variant) via the
    pricing table so the split is exact rather than apportioned from the
    aggregate cost_usd. Returns the four components with both cost and token
    volume (the UI shows tokens for subscription/local framing where dollars
    are suppressed).

    THE SPLIT MUST SUM TO THE WINDOW TOTAL ``/cost`` PUBLISHES. That total is
    ``SUM(spans.cost_usd)``, priced once at ingest by ``CostEngine.
    process_span``, and the only way a re-pricing agrees with it is by using the
    same convention for all three of its inputs:

    * the INSTANT — the UTC-day bucket below, since a rate can only change at
      UTC midnight (``SPAN_UTC_DAY_SQL``);
    * the VARIANT — ``core.cost.SPAN_VARIANT_SQL``, the stored-column spelling
      of ``rate_variant_for_span``. This used to be omitted entirely, so every
      span was re-priced at ``STANDARD_VARIANT``: an Anthropic ``speed="fast"``
      call billed at the premium rate in the stored total and at half that here;
    * the FALLBACK — ``core.cost.billing_rates``, which never returns ``None``.
      This used to be ``span_pricing.rates_at`` followed by ``continue``, so a
      model absent from the pricing table contributed real fallback dollars to
      ``/cost`` and exactly ``$0`` here, silently.

    Reconciliation is exact up to per-span 8-dp rounding (``calculate_cost``
    rounds each span; this sums a bucket's tokens first). The one population
    that can still differ is a span with a model but NO provider: ``CostEngine``
    skips those, leaving whatever cost ingest supplied, while the lookup here
    falls back to ``UNKNOWN_PROVIDER``. ``pricing_coverage`` on the payload is
    what discloses fallback-priced traffic either way.
    """
    from tokenjam.core.cost import SPAN_VARIANT_SQL, billing_rates
    from tokenjam.core.optimize.span_pricing import SPAN_UTC_DAY_SQL, UNKNOWN_PROVIDER

    comp = {k: {"cost_usd": 0.0, "tokens": 0} for k, _ in _COMPONENT_LABELS}
    if conn is None:
        return comp
    clauses = ["model IS NOT NULL"]
    params: list = []
    if agent_id:
        params.append(agent_id)
        clauses.append("agent_id = $" + str(len(params)))
    if since_dt is not None:
        params.append(since_dt)
        clauses.append("start_time >= $" + str(len(params)))
    if until_dt is not None:
        params.append(until_dt)
        clauses.append("start_time <= $" + str(len(params)))
    # The persona POPULATION scope. These bars used to be deliberately
    # unscoped, on the reasoning that measured spend is measured spend whatever
    # persona is selected. That held only while nothing ELSE on the surface was
    # scoped. The overlay drawn on top of them is now this persona's, and the
    # ceiling's denominator comes from the same split — so leaving the bars
    # whole-corpus published one persona's overspend as a share of everybody's
    # spend, which rendered as "exceeds measured spend" on a corpus where it
    # did not.
    add_persona_clause(clauses, persona)
    where = " AND ".join(clauses)
    # Grouped by UTC day as well as (provider, model) so each bucket prices at
    # the rate that actually billed it — a UTC day never straddles a rate change
    # (every `valid_from` in the pricing table is a date). The component totals
    # are summed across buckets, which is the price-each-span-then-sum
    # convention in `tokenjam.core.optimize.span_pricing`.
    sql = (
        f"SELECT provider, model, {SPAN_VARIANT_SQL} AS variant, "
        "COALESCE(SUM(input_tokens), 0), "
        "COALESCE(SUM(output_tokens), 0), COALESCE(SUM(cache_tokens), 0), "
        "COALESCE(SUM(cache_write_tokens), 0), MIN(start_time) "
        "FROM spans WHERE " + where
        + f" GROUP BY provider, model, variant, {SPAN_UTC_DAY_SQL}"
    )
    for (provider, model, variant, in_t, out_t, cr_t, cw_t,
         day_start) in conn.execute(sql, params).fetchall():
        in_t, out_t, cr_t, cw_t = int(in_t or 0), int(out_t or 0), int(cr_t or 0), int(cw_t or 0)
        comp["input"]["tokens"] += in_t
        comp["output"]["tokens"] += out_t
        comp["cache_read"]["tokens"] += cr_t
        comp["cache_write"]["tokens"] += cw_t
        # No `continue` on an unresolved rate, and no skip on a missing bucket
        # instant: `start_time` is NOT NULL, so `day_start` is only ever absent
        # for a store that has been corrupted below the schema, and
        # `billing_rates` then prices at the current rate exactly as a live
        # call does rather than dropping the bucket's money to zero.
        rates = billing_rates(
            provider or UNKNOWN_PROVIDER, model, at=day_start, variant=variant,
        )
        comp["input"]["cost_usd"] += in_t * rates.input_per_mtok / 1_000_000.0
        comp["output"]["cost_usd"] += out_t * rates.output_per_mtok / 1_000_000.0
        comp["cache_read"]["cost_usd"] += cr_t * rates.cache_read_per_mtok / 1_000_000.0
        comp["cache_write"]["cost_usd"] += cw_t * rates.cache_write_per_mtok / 1_000_000.0
    return comp


def _collect_recoverable(report, *, persona: str | None = None) -> list[dict]:
    """Registry-driven per-analyzer recoverable list (#211 overlay).

    Iterates the typed downgrade slot + every wave-2 finding carrying the #111
    recoverable contract field, so a new analyzer appears with no code change
    here. Each entry keeps the analyzer's own caveat + estimate_basis verbatim
    (Rule 14) and the component its savings act on. Only positive estimates are
    surfaced (a None/0 estimate is "nothing to recover", not an overlay bar).

    Returned biggest-first (by USD, then tokens) so a caller can render "largest
    opportunity + N more" without re-deriving the order — and so index 0 is
    always the entry `largest_recoverable_*` below is drawn from."""
    out: list[dict] = []

    def add(name: str, finding) -> None:
        if finding is None:
            return
        usd = getattr(finding, "past_overspend_usd", None)
        tok = getattr(finding, "past_overspend_tokens", None)
        if not ((usd is not None and usd > 0) or (tok is not None and tok > 0)):
            return
        out.append({
            "analyzer": name,
            "title": _ANALYZER_TITLE.get(name, name.replace("-", " ").title()),
            "component": _ANALYZER_COMPONENT.get(name, "call"),
            "past_overspend_usd": usd,
            "past_overspend_tokens": tok,
            "estimate_basis": getattr(finding, "estimate_basis", "") or "",
            "caveat": getattr(finding, "caveat", "") or "",
        })

    # PERSONA-SCOPED, because "every finding carrying the field" is no longer the
    # same set as "every finding this reader can act on": the daemon's pass
    # computes the union across personas so one stored report can answer for
    # either side of the persona picker (`runner.build_report`'s `personas`), so
    # a claude-code window's stored report now legitimately carries `reuse` /
    # `verbosity` / `script` findings. Summing those into this overlay would
    # publish recoverable dollars against levers this persona does not have.
    # Same map both halves of the gate read (ANALYZER-PERSONA-MATRIX.md §6).
    from tokenjam.core.optimize import disabled_analyzers_for_persona, findings_for_persona

    # The REQUESTED persona when the caller named one, else the corpus's own —
    # so this overlay covers the same population as the analyzer list rendered
    # beside it on the Optimize screen.
    view_persona = persona or str(getattr(report, "persona", "") or "unknown")
    if "downsize" not in disabled_analyzers_for_persona(view_persona):
        add("downsize", getattr(report, "downgrade", None))
    for name, finding in findings_for_persona(
        getattr(report, "findings", None) or {}, view_persona,
    ).items():
        if name == "downsize":
            continue
        if hasattr(finding, "past_overspend_usd"):
            add(name, finding)
    out.sort(
        key=lambda r: (r["past_overspend_usd"] or 0.0, r["past_overspend_tokens"] or 0),
        reverse=True,
    )
    return out


# A1 (analyzer-audit #482, self-improve-loop): every analyzer above estimates
# waste from its own angle over the SAME underlying spans — a `cache` fix and
# a `downsize` swap can both price the same call's waste, a `reuse` hit can
# double-count a `trim` hit's planning call, and so on. `total_recoverable_usd`
# below is therefore a GROSS ceiling across N overlapping analyzers, not a
# simultaneously-achievable total — summing the list's own figures would be
# summing waste that was measured twice. This is a presentation fix only: no
# individual analyzer's `past_overspend_usd` is touched or reduced here
# (house rule: never quietly deflate a figure the user can act on). The single
# largest entry, in contrast, IS honest on its own — acting on it alone
# recovers at least that much, because it isn't a sum of anything.
def _recoverable_overlap_note(recoverable: list[dict]) -> str:
    if len(recoverable) < 2:
        return ""
    return (
        f"These {len(recoverable)} estimates are computed from overlapping angles on "
        "the same sessions (for example, a cache fix and a model downsize can both "
        "price the same call's waste), so they do not add up to an amount you could "
        "actually recover. The figure above is a ceiling across every analyzer; the "
        "largest single line is the safest number to act on first."
    )


# THE SPEND BAR'S DENOMINATOR. `total_recoverable_usd` is NOT a figure over the
# window the caller asked for, and it never can be: it is read out of the stored
# analyzer report, which is computed on the daemon's own schedule over its own
# window (analyzers must never run per-request). Pairing it with the requested
# window's `total_cost_usd` published two figures over two populations — a 7-day
# total under a 30-day ceiling, which shaded 73% of a week's spend as overspend
# using a month's estimate, and trends past 100% as the window narrows.
#
# The fix is to give the ceiling a denominator drawn from ITS OWN population
# rather than rescaling either figure: same window, same agents. `agent_id` is
# deliberately NOT applied — the stored report is corpus-wide, so filtering the
# denominator to one agent would reintroduce the same mismatch on the other axis.
#
# The denominator IS persona-scoped, and the two sentences that used to sit here
# saying otherwise were left behind by the change that scoped it. They claimed
# the picker selects a lever set only, and that every finding is computed over
# the whole corpus. Both stopped being true when the daemon started storing a
# separately-scoped sub-report per persona: `stored_report_for_persona` serves
# THAT persona's pass, and `_component_costs` below takes the same `persona` so
# the ceiling and its denominator cover one population. A stale comment asserting
# the old contract is worse than none, because the next reader trusts it over the
# code. `recoverable_basis_note` states the live contract on the bar itself.
def _recoverable_window_bounds(scan: dict) -> tuple[datetime | None, datetime | None]:
    """The window the stored report actually observed, as datetimes.

    Prefers the cycle record's own `scan_since`/`scan_until` (the resolved
    bounds the pass sealed). Falls back to `computed_at` minus `window_days` for
    an artifact written before that record existed. Returns ``(None, None)``
    when neither is available — the bounds are never invented, because a
    denominator over a guessed window is the defect this exists to prevent.
    """
    def _parse(value) -> datetime | None:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        if isinstance(value, str) and value:
            try:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        return None

    until = _parse(scan.get("scan_until")) or _parse(scan.get("computed_at"))
    since = _parse(scan.get("scan_since"))
    if since is None:
        days = scan.get("window_days")
        if until is not None and isinstance(days, (int, float)) and days > 0:
            since = until - timedelta(days=float(days))
    if since is None or until is None:
        return None, None
    return since, until


#: How the picker's values read in the note. The bar is prose, so it uses the
#: same words the picker shows rather than the wire key.
_PERSONA_TRAFFIC_LABEL = {
    "claude-code": "Claude Code",
    "sdk": "SDK workflow",
}


def _recoverable_basis_note(
    since_dt: datetime | None, until_dt: datetime | None, window_days,
    persona: str | None = None, lever_persona: str | None = None,
) -> str:
    """What the bar's two figures cover, stated on the bar.

    Prominence is the point: the mismatch this replaces was invisible precisely
    because nothing on screen said which window the shaded region came from.

    TWO OF THE THREE SENTENCES USED TO BE FALSE, and they were false in the same
    way the defect they were written to fix was: a figure asserting more than its
    population supports. The note said "Both figures cover every agent" and "the
    persona picker changes which analyzers contribute to the ceiling, not which
    traffic is measured". Once the daemon began storing a separately-scoped pass
    per persona, the picker moved BOTH halves: on one real corpus the same bar's
    denominator reads 13733.55 as Claude Code and 1.37 as SDK. A disclosure that
    contradicts the payload it travels with is worse than no disclosure. The
    middle sentence (this bar does not follow the range picker; the estimate
    comes from the stored scan) was and is true, and is kept verbatim.

    NO SENTENCE HERE MAY DESCRIBE BOTH FIGURES AT ONCE, because the two do not
    share a population and no short phrase covering both can be true. Measured on
    a real corpus, per-analyzer, over one 30-day window:

    * the SPEND is exactly persona-scoped (`add_persona_clause`). The two
      personas partition the corpus to the cent, so "covers <persona> traffic"
      is precise for this half;
    * the CEILING is a MIX. `resend`, `subagent` and `summarize` move with the
      persona; `downsize` is exactly additive across the two; but `relearn` and
      `deadweight` come back byte-identical scoped and unscoped, because their
      basis is an unbounded corpus scan rather than this window (matrix §7b,
      `relearn.py`'s own note). On that corpus they were roughly a tenth of the
      claude-code total, sitting under a persona label while not being
      persona-scoped at all.

    So both of the obvious wordings are false. "Both figures cover every agent"
    is false for the spend under a selected persona; "both figures cover this
    persona's traffic" is false for the corpus-wide share of the ceiling. The
    note therefore states each figure's coverage separately and says of the
    ceiling only what holds for every contributor: each is measured on its own
    basis, and not every basis is scoped to the persona. Naming WHICH
    contributors are wider needs a per-analyzer disclosure this note cannot
    carry, and `relearn` currently ships an empty `estimate_basis`, so its card
    does not say either.

    Do not "simplify" this back into one sentence about both figures. That is
    the exact edit that produced the wording this replaced.
    """
    if since_dt is None or until_dt is None:
        return (
            "The last analyzer scan did not record the window it observed, so "
            "this ceiling cannot be shown as a share of spend."
        )
    span = f"{since_dt.date().isoformat()} to {until_dt.date().isoformat()}"
    days = f"{int(window_days)} days" if isinstance(window_days, (int, float)) else "window"
    stable = (
        f"over the analyzer scan's own {days}, {span}. This bar does not follow "
        "the range selected above: the recoverable estimate comes from the stored "
        "scan and is never recomputed per request."
    )
    label = _PERSONA_TRAFFIC_LABEL.get(persona or "")
    if label:
        return (
            f"The spend covers {label} traffic {stable} The ceiling counts only "
            "the analyzers that persona can act on, each measured on its own "
            "basis; not every basis is scoped to this persona."
        )
    # NO PERSONA SELECTED, and this case is NOT a clean corpus total, so the note
    # may not imply one. The spend covers every agent, but the ceiling is still
    # filtered by a lever set: `_collect_recoverable` falls back to the corpus's
    # own dominant persona, so a corpus that resolves to Claude Code drops the
    # analyzers only an SDK user could act on while keeping every agent's spend
    # underneath. Saying so is the whole reason this branch is separate.
    lever_label = _PERSONA_TRAFFIC_LABEL.get(lever_persona or "")
    if lever_label:
        return (
            f"The spend covers every agent {stable} The ceiling does not: with no "
            f"persona selected it counts only the analyzers a {lever_label} user "
            "can act on, each measured on its own basis. Select a persona above "
            "to scope the spend to that persona too."
        )
    return (
        f"The spend covers every agent {stable} Every analyzer contributes to "
        "the ceiling, each measured on its own basis."
    )


@router.get("/cost/components")
async def get_cost_components(
    request: Request,
    agent_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
    persona: str | None = None,
) -> dict:
    """Cost-by-component split + per-analyzer recoverable-waste overlay (#211).

    The component bars are MEASURED spend split into input / output / cache-read
    / cache-write. The overlay is each analyzer's *estimated recoverable* waste —
    registry-driven, carrying the analyzer's own caveat verbatim. "Estimated
    recoverable" is never conflated with the measured cost and never called
    "saved" (Critical Rule 14). Every figure routes through the framing block so
    subscription/local users see token-share, not raw dollars.

    `persona` scopes the OVERLAY to the analyzers that persona has a lever for,
    the same way `GET /optimize?persona=` does — the two are rendered on one
    screen and read as one answer. Without it this endpoint named the corpus's
    dominant persona's biggest opportunity ("Biggest single cause: Subagent")
    beside an analyzer list that had correctly dropped Subagent for the selected
    persona: two figures published together over two different populations
    (root anti-pattern 22b). The component BARS are unscoped and stay that way —
    they are measured spend, which the persona picker does not reinterpret.

    `recoverable_basis_*` is the denominator the overlay's total may be drawn
    against, and the ONLY one: same window, same agents as the stored report.
    `total_cost_usd` answers the caller's window and is not a valid denominator
    for it. See `_recoverable_window_bounds`."""
    db = request.app.state.db
    config = request.app.state.config
    if persona is not None and persona not in PERSONAS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown persona {persona!r}. Expected one of {sorted(PERSONAS)}.",
        )
    try:
        since_dt = parse_since(since) if since else None
        until_dt = parse_since(until) if until else None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid --since: {exc}") from exc
    conn = getattr(db, "conn", None)

    comp = _component_costs(conn, agent_id, since_dt, until_dt, persona)
    components = [
        {"key": key, "label": label,
         "cost_usd": round(comp[key]["cost_usd"], 8), "tokens": comp[key]["tokens"]}
        for key, label in _COMPONENT_LABELS
    ]
    total_cost = sum(c["cost_usd"] for c in components)
    total_tokens = sum(c["tokens"] for c in components)

    # The overlay comes from the STORED analyzer report, never a live run: this
    # endpoint used to call `build_report` inline, dispatching every analyzer
    # over the corpus on the request thread and taking tens of seconds to
    # minutes on a real install. `core.optimize.report_store` is kept warm by
    # the daemon (boot / interval / user-pressed rescan). The component bars
    # above are unaffected — they are plain measured-spend queries off live
    # ingest, and ingestion is untouched.
    from tokenjam.core.optimize import report_store

    scan = report_store.stored_report_block(config)
    # THE PERSONA'S OWN SCOPED REPORT, not the corpus-wide one with its analyzer
    # set sliced. Slicing the set drops findings this persona has no lever for;
    # it does nothing about the fact that the surviving findings were computed
    # over everybody's sessions. `stored_report_for_persona` owns that rule,
    # including the `None` an artifact from before per-persona passes yields —
    # which reports the overlay as NOT YET KNOWN (what `recoverable_status`
    # already exists to say) rather than publishing the corpus's figures here.
    stored = report_store.stored_report_for_persona(config, persona)
    recoverable: list[dict] = (
        _collect_recoverable(stored, persona=persona) if stored is not None else []
    )

    # A cold store contributes NO overlay and says so via `recoverable_status`.
    # The totals below stay `None` rather than 0.0 in that case: a `$0.00`
    # recoverable figure reads as "nothing to recover", which is exactly the
    # reassurance an un-run scan cannot support.
    known = stored is not None
    total_rec_usd: float | None = (
        float(sum(r["past_overspend_usd"] or 0.0 for r in recoverable)) if known else None
    )
    total_rec_tokens: int | None = (
        int(sum(r["past_overspend_tokens"] or 0 for r in recoverable)) if known else None
    )
    largest = recoverable[0] if recoverable else None

    # The ceiling's OWN denominator (see _recoverable_window_bounds): measured
    # spend over the stored report's window, across every agent, so the spend
    # bar's two figures cover one population. Only computed when a report exists
    # — with nothing measured there is no ceiling to give a share of, and a
    # denominator published beside an unmeasured numerator is the zero-as-
    # reassurance failure in a second costume.
    rec_since_dt, rec_until_dt = _recoverable_window_bounds(scan)
    basis_cost: float | None = None
    basis_tokens: int | None = None
    if known and rec_since_dt is not None:
        if (
            agent_id is None
            and since_dt == rec_since_dt
            and (until_dt or rec_until_dt) == rec_until_dt
        ):
            # The requested window IS the report's window: reuse the split
            # already computed above rather than running the aggregate twice.
            basis_cost, basis_tokens = total_cost, total_tokens
        else:
            # SAME persona as the numerator above. `agent_id` is deliberately
            # dropped here (the ceiling spans every agent in the scan) but the
            # persona is not: it is the population the ceiling was computed
            # over, so it has to be the population the denominator covers.
            basis = _component_costs(
                conn, None, rec_since_dt, rec_until_dt, persona,
            )
            basis_cost = sum(v["cost_usd"] for v in basis.values())
            basis_tokens = sum(v["tokens"] for v in basis.values())

    return {
        "components": components,
        "total_cost_usd": round(total_cost, 8),
        "total_tokens": total_tokens,
        "recoverable": recoverable,
        # Freshness of the overlay ONLY — the component bars are live.
        "recoverable_status": scan["status"],
        "recoverable_computed_at": scan["computed_at"],
        "recoverable_window_days": scan["window_days"],
        "recoverable_available": known,
        # Gross ceiling, magnitude unchanged (see _recoverable_overlap_note) —
        # NOT a claim that this much is simultaneously recoverable. `None`
        # means "not measured yet" (cold scan), never zero.
        "total_recoverable_usd": (
            round(total_rec_usd, 8) if total_rec_usd is not None else None
        ),
        "total_recoverable_tokens": total_rec_tokens,
        "recoverable_additive": False,
        "recoverable_overlap_note": _recoverable_overlap_note(recoverable),
        # THE ONLY DENOMINATOR `total_recoverable_usd` MAY BE DRAWN AGAINST.
        # Measured spend over the SAME window and the SAME agents the stored
        # report covered, so a caller cannot accidentally pair the ceiling with
        # the requested window's total (which is what the spend bar did). `None`
        # means the share is unknowable, never zero.
        "recoverable_basis_cost_usd": (
            round(basis_cost, 8) if basis_cost is not None else None
        ),
        "recoverable_basis_tokens": basis_tokens,
        "recoverable_basis_since": (
            int(rec_since_dt.timestamp()) if rec_since_dt is not None else None
        ),
        "recoverable_basis_until": (
            int(rec_until_dt.timestamp()) if rec_until_dt is not None else None
        ),
        # Travels WITH the pair, exactly like `recoverable_overlap_note`: the
        # window and scope disclosure is part of the claim, not decoration a
        # renderer may drop.
        # `lever_persona` mirrors the fallback `_collect_recoverable` applies, so
        # the note describes the set that actually built the ceiling rather than
        # the one the caller asked for. It is an argument, not a payload field.
        "recoverable_basis_note": (
            _recoverable_basis_note(
                rec_since_dt, rec_until_dt, scan["window_days"], persona,
                str(getattr(stored, "persona", "") or "") if stored is not None else None,
            )
            if known else ""
        ),
        # The one entry in `recoverable` that is honest as a standalone claim:
        # it isn't a sum of anything, so it's the floor a reader can act on.
        "largest_recoverable_usd": largest["past_overspend_usd"] if largest else None,
        "largest_recoverable_tokens": largest["past_overspend_tokens"] if largest else None,
        "largest_recoverable_analyzer": largest["analyzer"] if largest else None,
        # Same block, same derivation, as `/cost` — now that the component split
        # prices unpriced models through the same non-null fallback the stored
        # total uses, the two endpoints publish the same dollars over the same
        # window, and a reader of EITHER has to be able to tell an estimated
        # figure from a quoted one. Shipping it on one and not the other made
        # the components view the surface where a 5-30x-wrong model stayed
        # invisible.
        "pricing_coverage": _pricing_coverage_block(
            conn, agent_id, since_dt, until_dt,
        ),
        "framing": _framing_block(db, config, agent_id, total_cost, total_tokens),
    }


@router.get("/cost/cache")
async def get_cost_cache(
    request: Request,
    agent_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
) -> dict:
    """Cache-savings time-series (#212): per-bucket hit-rate + cumulative captured
    savings, plus the window-level *estimated recoverable* from the cache analyzer.

    "Captured" is measured (priced from real cache reads); "estimated
    recoverable" is the analyzer's heuristic for additional savings if caching
    were expanded — never conflated, never called "saved" (Critical Rule 14)."""
    db = request.app.state.db
    config = request.app.state.config
    try:
        since_dt = parse_since(since) if since else None
        until_dt = parse_since(until) if until else None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid --since: {exc}") from exc
    conn = getattr(db, "conn", None)

    block = _cache_series(conn, agent_id, since_dt, until_dt)
    total_captured = sum(p["captured_usd"] for p in block["series"])
    total_captured_tokens = sum(p["captured_tokens"] for p in block["series"])

    # Window-level estimated recoverable from the cache-efficacy analyzer (#111
    # recoverable contract) — read from the STORED report, never re-run here.
    # The captured series above is measured spend off live ingest and is
    # unaffected. `None` throughout means "not measured yet", never zero.
    from tokenjam.core.optimize import report_store

    scan = report_store.stored_report_block(config)
    stored = report_store.stored_report(config)
    recoverable_usd: float | None = None
    recoverable_tokens: int | None = None
    estimate_basis = ""
    if stored is not None:
        cache_finding = (stored.findings or {}).get("cache")
        if cache_finding is not None:
            recoverable_usd = getattr(cache_finding, "past_overspend_usd", None)
            recoverable_tokens = getattr(cache_finding, "past_overspend_tokens", None)
            estimate_basis = getattr(cache_finding, "estimate_basis", "") or ""

    return {
        **block,
        "total_captured_usd": round(total_captured, 8),
        "total_captured_tokens": total_captured_tokens,
        "past_overspend_usd": recoverable_usd,
        "past_overspend_tokens": recoverable_tokens,
        "estimate_basis": estimate_basis,
        "recoverable_status": scan["status"],
        "recoverable_computed_at": scan["computed_at"],
        "recoverable_window_days": scan["window_days"],
        "recoverable_available": stored is not None,
        "framing": _framing_block(db, config, agent_id, total_captured, total_captured_tokens),
    }


# Attribution dimensions the Cost view can group/filter by, beyond the
# original agent/model/day/tool — the "which CUSTOMER/TENANT, FEATURE,
# ENVIRONMENT, PROMPT VERSION is spending the money" breakdown. Each maps its
# group_by key to (spans column, the wire attribute a caller sets to populate
# it, human label) so the UI's empty state can tell a user exactly what to
# instrument instead of just showing a blank chart.
ATTRIBUTION_DIMENSIONS: dict[str, tuple[str, str, str]] = {
    "tenant": ("tenant_id", "tokenjam.tenant_id", "Tenant"),
    "feature": ("feature", "tokenjam.feature", "Feature"),
    "environment": ("environment", "deployment.environment.name", "Environment"),
    "prompt_version": (
        "prompt_template_version", "tokenjam.prompt.template_version", "Prompt version",
    ),
}


def _dimension_coverage(conn, agent_id, since_dt, until_dt) -> dict:
    """Whether each attribution dimension carries ANY data in this window
    (#SDK dashboard shape). Powers the honest empty state: a dimension with
    zero coverage renders "set <attribute> at the call site" instead of a
    misleadingly blank chart indistinguishable from "no spend at all".
    """
    out = {
        key: {"has_data": False, "attribute": attr, "label": label}
        for key, (_, attr, label) in ATTRIBUTION_DIMENSIONS.items()
    }
    if conn is None:
        return out
    clauses = ["model IS NOT NULL"]
    params: list = []
    if agent_id:
        params.append(agent_id)
        clauses.append("agent_id = $" + str(len(params)))
    if since_dt is not None:
        params.append(since_dt)
        clauses.append("start_time >= $" + str(len(params)))
    if until_dt is not None:
        params.append(until_dt)
        clauses.append("start_time <= $" + str(len(params)))
    where = " AND ".join(clauses)
    cols = ", ".join(
        f"COUNT(*) FILTER (WHERE {col} IS NOT NULL)"
        for key, (col, _, _) in ATTRIBUTION_DIMENSIONS.items()
    )
    row = conn.execute(f"SELECT {cols} FROM spans WHERE {where}", params).fetchone()
    if row is not None:
        for i, key in enumerate(ATTRIBUTION_DIMENSIONS):
            out[key]["has_data"] = bool(row[i] and row[i] > 0)
    return out


def _pricing_coverage_block(conn, agent_id, since_dt, until_dt) -> dict:
    """The default-rate share of this window, shaped for the UI.

    `measured` distinguishes "asked, nothing unpriced" from "could not ask" so
    the panel can stay silent rather than assert a clean bill it never checked
    (root anti-pattern 22 — a surface must not claim more than its data
    supports).
    """
    coverage = summarize_pricing_coverage(conn, agent_id, since_dt, until_dt)
    return {
        "measured": coverage.measured,
        "unpriced_call_count": coverage.unpriced_call_count,
        "unpriced_cost_usd": coverage.unpriced_cost_usd,
        "unpriced_models": [
            {"provider": provider, "model": model, "call_count": calls}
            for provider, model, calls in coverage.unpriced_models
        ],
        "note": coverage_note(coverage),
    }


@router.get("/cost")
async def get_cost(
    request: Request,
    agent_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
    group_by: str = "day",
    tenant_id: str | None = None,
    feature: str | None = None,
    environment: str | None = None,
    prompt_version: str | None = None,
    persona: str | None = None,
) -> dict:
    """Grouped spend for the window.

    ``persona`` scopes to one side of the "Viewing as" picker, through
    ``CostFilters`` so every figure derived from these filters covers the same
    population. Omitted (and `mixed`/`unknown`) means all spend.
    """
    db = request.app.state.db
    config = request.app.state.config
    if persona is not None and persona not in PERSONAS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown persona {persona!r}. Expected one of {sorted(PERSONAS)}.",
        )
    try:
        since_dt = parse_since(since) if since else None
        until_dt = parse_since(until) if until else None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid --since: {exc}") from exc
    filters = CostFilters(
        agent_id=agent_id,
        since=since_dt,
        until=until_dt,
        group_by=group_by,
        tenant_id=tenant_id,
        feature=feature,
        environment=environment,
        prompt_version=prompt_version,
        persona=persona,
    )
    rows = db.get_cost_summary(filters)
    total = sum(r.cost_usd for r in rows)
    # All four token types, always (Cache token types in aggregates, root
    # CLAUDE.md): omitting cache_tokens/cache_write_tokens understates the
    # true total by an order of magnitude on a cache-heavy corpus.
    total_tokens = sum(
        r.input_tokens + r.output_tokens + r.cache_tokens + r.cache_write_tokens
        for r in rows
    )

    # Plan-tier framing block — single source shared with the CLI (#110). Lets
    # the local web UI render the same suppressed/qualified dollar figures.
    # The mix is window-INDEPENDENT (#177): the pricing mode + qualifier banner
    # are a property of the user's plan, so the chart's tokens-vs-dollars unit
    # stays consistent as the user switches windows. Only the window totals
    # (above) and the `series` (below) are window-scoped.
    conn = getattr(db, "conn", None)
    framing = _framing_block(db, config, agent_id, total, total_tokens)
    return {
        "rows": [
            {
                "group": r.group,
                "agent_id": r.agent_id,
                "model": r.model,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "cache_tokens": r.cache_tokens,
                "cache_write_tokens": r.cache_write_tokens,
                "cost_usd": r.cost_usd,
                "call_count": r.call_count,
            }
            for r in rows
        ],
        "total_cost_usd": total,
        "total_tokens": total_tokens,
        "total_cache_tokens": sum(r.cache_tokens for r in rows),
        "total_cache_write_tokens": sum(r.cache_write_tokens for r in rows),
        **_window_series(conn, agent_id, since_dt, until_dt),
        "cycle": _cycle_block(config),
        "framing": framing,
        # Per-dimension "is there any data to show" flags (#SDK dashboard
        # shape) so the UI can render an honest empty state — independent of
        # group_by, so switching the dropdown to an uninstrumented dimension
        # doesn't require a second round-trip to know it'll be empty.
        "attribution_coverage": _dimension_coverage(conn, agent_id, since_dt, until_dt),
        # Which models in this window were priced at the flat default rate
        # rather than a published one. Without this the UI cannot tell an
        # estimated dollar figure from a quoted one, which is exactly how a
        # missing table row stays invisible while the number is 5-30x wrong.
        "pricing_coverage": _pricing_coverage_block(
            conn, agent_id, since_dt, until_dt,
        ),
        # `available_days` (core/data_span.py) so the Cost view's own window
        # selector can derive its options from what the store actually holds,
        # the same way the Dashboard's does — instead of a fixed 24h/7d/30d/90d
        # list that can offer a window with nothing behind it.
        "data_span": available_data_span(conn).to_dict(),
    }


@router.get("/cost/tenants")
async def get_cost_tenants(
    request: Request,
    agent_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
    top_n: int = 10,
) -> dict:
    """Top-N tenants by spend: concentration view (#SDK dashboard shape).

    Audits of multi-tenant B2B SaaS commonly find a small share of tenants
    driving most of the token spend — this is the view that answers "which
    ones, and how much". Shares are computed against TOTAL window spend
    (including spend with no tenant_id set), never just the attributed subset,
    so concentration can never be silently inflated by excluded unattributed
    spend (Critical Rule 14 — never mislead with an unstated denominator).
    Trend is each top tenant's cost delta vs. the prior equal-length window.
    """
    db = request.app.state.db
    config = request.app.state.config
    try:
        since_dt = parse_since(since) if since else None
        until_dt = parse_since(until) if until else None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid --since: {exc}") from exc
    conn = getattr(db, "conn", None)
    end_dt = until_dt or utcnow()

    empty = {
        "rows": [],
        "total_cost_usd": 0.0,
        "attributed_cost_usd": 0.0,
        "unattributed_cost_usd": 0.0,
        "total_tokens": 0,
        "attributed_tokens": 0,
        "unattributed_tokens": 0,
        "has_data": False,
        "attribute": "tokenjam.tenant_id",
        "window_start": int(since_dt.timestamp()) if since_dt is not None else None,
        "window_end": int(end_dt.timestamp()),
        "framing": _framing_block(db, config, agent_id, 0.0, 0),
    }
    if conn is None:
        return empty

    clauses = ["model IS NOT NULL", "start_time < $1"]
    params: list = [end_dt]
    if since_dt is not None:
        params.append(since_dt)
        clauses.append("start_time >= $" + str(len(params)))
    if agent_id:
        params.append(agent_id)
        clauses.append("agent_id = $" + str(len(params)))
    where = " AND ".join(clauses)

    # Total window spend (denominator) and the attributed/unattributed split,
    # in one scan.
    totals_row = conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0.0), "
        "COALESCE(SUM(cost_usd) FILTER (WHERE tenant_id IS NOT NULL), 0.0), "
        "COALESCE(SUM(input_tokens + output_tokens + cache_tokens + cache_write_tokens), 0), "
        "COALESCE(SUM(input_tokens + output_tokens + cache_tokens + cache_write_tokens) "
        "FILTER (WHERE tenant_id IS NOT NULL), 0) "
        f"FROM spans WHERE {where}",
        params,
    ).fetchone()
    total_cost = float(totals_row[0] or 0.0) if totals_row else 0.0
    attributed_cost = float(totals_row[1] or 0.0) if totals_row else 0.0
    unattributed_cost = max(0.0, total_cost - attributed_cost)
    total_tokens = int(totals_row[2] or 0) if totals_row else 0
    attributed_tokens = int(totals_row[3] or 0) if totals_row else 0
    unattributed_tokens = max(0, total_tokens - attributed_tokens)

    if attributed_cost <= 0.0:
        return {**empty, "total_cost_usd": round(total_cost, 8),
                "unattributed_cost_usd": round(total_cost, 8),
                "total_tokens": total_tokens,
                "unattributed_tokens": total_tokens}

    top_rows = conn.execute(
        "SELECT tenant_id, "
        "COALESCE(SUM(cost_usd), 0.0), COUNT(*), "
        "COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0), "
        "COALESCE(SUM(cache_tokens), 0), COALESCE(SUM(cache_write_tokens), 0) "
        f"FROM spans WHERE {where} AND tenant_id IS NOT NULL "
        "GROUP BY tenant_id ORDER BY 2 DESC LIMIT $" + str(len(params) + 1),
        [*params, max(1, top_n)],
    ).fetchall()

    # Trend: each top tenant's spend in the prior equal-length window.
    prev_by_tenant: dict[str, float] = {}
    if since_dt is not None:
        prev_since = since_dt - (end_dt - since_dt)
        prev_clauses = ["model IS NOT NULL", "tenant_id IS NOT NULL",
                         "start_time >= $1", "start_time < $2"]
        prev_params: list = [prev_since, since_dt]
        if agent_id:
            prev_params.append(agent_id)
            prev_clauses.append("agent_id = $" + str(len(prev_params)))
        prev_where = " AND ".join(prev_clauses)
        for tenant, cost in conn.execute(
            f"SELECT tenant_id, COALESCE(SUM(cost_usd), 0.0) FROM spans WHERE {prev_where} "
            "GROUP BY tenant_id",
            prev_params,
        ).fetchall():
            prev_by_tenant[tenant] = float(cost or 0.0)

    rows = []
    for tenant_id, cost, call_count, in_t, out_t, cr_t, cw_t in top_rows:
        cost = float(cost or 0.0)
        prev_cost = prev_by_tenant.get(tenant_id)
        delta_pct = (
            round((cost - prev_cost) / prev_cost * 100.0, 1)
            if prev_cost else None
        )
        rows.append({
            "tenant_id": tenant_id,
            "cost_usd": round(cost, 8),
            # Share of TOTAL window spend (the honest denominator — see
            # docstring), not just the attributed subset.
            "share_of_total": round(cost / total_cost, 6) if total_cost > 0 else None,
            "call_count": int(call_count or 0),
            "input_tokens": int(in_t or 0),
            "output_tokens": int(out_t or 0),
            "cache_tokens": int(cr_t or 0),
            "cache_write_tokens": int(cw_t or 0),
            "previous_cost_usd": round(prev_cost, 8) if prev_cost is not None else None,
            "delta_pct": delta_pct,
        })

    return {
        "rows": rows,
        "total_cost_usd": round(total_cost, 8),
        "attributed_cost_usd": round(attributed_cost, 8),
        "unattributed_cost_usd": round(unattributed_cost, 8),
        "total_tokens": total_tokens,
        "attributed_tokens": attributed_tokens,
        "unattributed_tokens": unattributed_tokens,
        "has_data": True,
        "attribute": "tokenjam.tenant_id",
        "window_start": int(since_dt.timestamp()) if since_dt is not None else None,
        "window_end": int(end_dt.timestamp()),
        "framing": _framing_block(db, config, agent_id, total_cost, 0),
    }
