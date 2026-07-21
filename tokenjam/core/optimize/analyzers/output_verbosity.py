"""
Verbosity analyzer — the output-side lever the other analyzers miss.

Every other analyzer targets the INPUT side (prompt size, model choice,
caching, repeated shapes). None looks at OUTPUT tokens. An agent that
generates verbose prose burns output tokens — often the pricier per-token
side — and that spend is invisible today. This analyzer surfaces
`(agent, model)` cohorts / sessions whose output runs high relative to a
like-for-like baseline, as REVIEW CANDIDATES only.

Honesty discipline (registry: verbosity — this is the *least-grounded*
analyzer, and it is framed that way):

  Unlike `trim`, which removes provably-redundant INPUT, output length is
  NOT waste — a terse answer can drop information the task needed, and
  "wasteful output" is task-dependent. So verbosity's recoverable estimate
  is inherently softer than the input-side analyzers. Every user-visible
  string says "predicted high-verbosity output — review before constraining
  a response", never "you're wasting X".

Detection baseline (v1 — tune on real data), strongest signal first:

  1. PREFERRED — per-`(tool, arg-shape)` / task-shape MEDIAN. A session's
     task shape is the ordered tuple of its `(tool_name, arg_shape)` tool
     calls (the same signature the `script` analyzer builds). Output tokens
     are compared against the median for the SAME task shape — the only
     signal grounded in like-for-like tasks. A session is a candidate when
     its output exceeds `HIGH_VERBOSITY_MULTIPLE` × its cohort median.
  2. output:input ratio — the WEAKEST signal, deliberately NOT the lead: a
     legitimately long answer to a short prompt is not waste. Carried only
     as a secondary descriptive field on each candidate, never as the flag.

Recoverable (softer than peers, per the issue framing): output tokens ABOVE
the cohort median, priced at OUTPUT rates. It is surfaced as an estimate
under an explicit "predicted / review before constraining" caveat — a
measured number (via `tj optimize --validate`, #477) is the honest one for
this lever, and the estimate_basis says so.

Remedy is SURFACED, not applied (issue #478 out-of-scope): recommend an
output-brevity constraint — a terse system-prompt snippet and/or a
`max_tokens` cap — for the user to apply and then MEASURE.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from typing import Any

from tokenjam.core.optimize.analyzers.workflow_restructure import _arg_signature
from tokenjam.core.optimize.registry import register
from tokenjam.core.optimize.types import AnalyzerContext
from tokenjam.core.pricing import get_rates
from tokenjam.otel.semconv import GenAIAttributes

# A task-shape cohort needs at least this many sessions before its median is a
# meaningful baseline. Below this, "the median" is one or two sessions and any
# multiple is noise — the least-grounded analyzer must be conservative about
# what it calls a baseline at all.
MIN_COHORT_SESSIONS = 5

# A session is a high-verbosity candidate when its output tokens exceed this
# multiple of the cohort median. Deliberately well above 1.0 — output length is
# not waste, so we only flag the clear outliers, not everything above the middle.
HIGH_VERBOSITY_MULTIPLE = 2.0

# Cap on surfaced example candidates (largest over-baseline output first).
MAX_EXAMPLES = 5

# Mandatory caveat (Rule 14), carried as the dataclass default like every other
# recoverable finding's caveat so no surface can drop it. States the ONE thing
# that makes this the least-grounded analyzer: output length is not waste.
VERBOSITY_HONESTY_CAVEAT = (
    "Predicted high-verbosity output — review before constraining a response. "
    "Output length is not waste: a terse answer can drop information the task "
    "needed. This is a candidate to look at, never a claim you are wasting "
    "tokens. Measure a brevity constraint before applying it."
)

# The `estimate_basis` surfaced behind the "estimated recoverable" tag. Names
# the softer basis explicitly and points at measurement (#477).
VERBOSITY_ESTIMATE_BASIS = (
    "output tokens above the per-task-shape median, priced at output rates — a "
    "SOFT upper bound, not a measured saving: a brevity constraint can be "
    "net-negative once its own overhead is counted, so measure before claiming"
)

# Surfaced remedy (not applied). A terse system-prompt snippet the user can add,
# paired with a max_tokens cap suggestion computed per candidate cohort.
VERBOSITY_REMEDY_SNIPPET = (
    "Be concise. Answer in the fewest words that fully address the request; "
    "omit restatement, preamble, and filler. Prefer lists over prose."
)


@dataclass
class VerbosityCandidate:
    """One session whose output runs high vs its task-shape cohort median."""

    session_id: str
    agent_id: str | None
    model: str
    task_shape: list[dict]          # [{"tool": "...", "args": [...]}], may be []
    cohort_sessions: int            # how many sessions share this task shape
    output_tokens: int              # this session's output tokens
    baseline_output_tokens: int     # the cohort median
    over_baseline_tokens: int       # output_tokens - baseline (>= 0)
    over_baseline_multiple: float   # output_tokens / baseline
    # WEAKEST signal, descriptive only — never the flag. A long answer to a
    # short prompt is not waste, so we surface the ratio but don't gate on it.
    output_input_ratio: float | None
    recoverable_usd: float | None   # over_baseline priced at output rates (None: no rates)


@dataclass
class VerbosityFinding:
    """High output-token spend candidates, on the #111 recoverable contract.

    Framed more conservatively than peer analyzers (issue #478): the recoverable
    figure is a soft upper bound under the "review before constraining" caveat,
    and the preferred baseline is the per-task-shape median (the most defensible
    signal), not the output:input ratio (the weakest).
    """

    candidates: list[VerbosityCandidate] = field(default_factory=list)
    # Pre-truncation flagged count. `candidates` is capped at MAX_EXAMPLES for
    # display, but the recoverable totals accumulate over every flagged session —
    # so the renderer reports this, not len(candidates), as the headline count.
    total_candidates: int = 0
    sessions_examined: int = 0
    cohorts_examined: int = 0       # task-shape cohorts with a usable median
    # Surfaced remedy (not applied): a terse system-prompt snippet + a suggested
    # max_tokens cap (the cohort baseline, the point most candidates already
    # sit under). Advisory strings only — the user applies + measures.
    remedy_snippet: str = VERBOSITY_REMEDY_SNIPPET
    suggested_max_tokens: int | None = None
    confidence: str = "structural"
    caveat: str = VERBOSITY_HONESTY_CAVEAT
    # Recoverable-savings contract (#111). See types.DowngradeFinding for field
    # semantics. None when no candidate cleared the threshold.
    estimated_recoverable_usd: float | None = None
    estimated_recoverable_tokens: int | None = None
    estimate_basis: str = ""
    estimate_confidence: str = "heuristic"
    # The effective cohort-size bar this run applied (config-overridable, see
    # core.config.OptimizeConfig.min_cohort_sessions) — carried on the finding
    # so a renderer's empty-state message never hardcodes a number that could
    # be stale against the user's own config.
    min_cohort_sessions: int = MIN_COHORT_SESSIONS


def _extract_tool_input(attrs: Any) -> Any:
    """Pull gen_ai.tool.input from a span's attributes JSON. None when absent."""
    if isinstance(attrs, str):
        try:
            attrs = json.loads(attrs)
        except Exception:
            return None
    if not isinstance(attrs, dict):
        return None
    return attrs.get(GenAIAttributes.TOOL_INPUT)


def _signature_repr(signature: tuple[tuple[str, tuple[str, ...]], ...]) -> list[dict]:
    """Human-readable task-shape signature, mirroring workflow_restructure."""
    out: list[dict] = []
    for tool_name, arg_sig in signature:
        entry: dict[str, Any] = {"tool": tool_name}
        if arg_sig:
            entry["args"] = list(arg_sig)
        out.append(entry)
    return out


@register("verbosity")
def run(ctx: AnalyzerContext) -> None:
    """Attach a VerbosityFinding to ctx.report.findings["verbosity"].

    Groups sessions by their task shape (ordered `(tool, arg-shape)` tuple),
    computes the per-cohort output-token median, and flags the sessions whose
    output exceeds the median by a wide multiple — the like-for-like baseline
    the issue calls the most defensible signal.
    """
    capture = getattr(ctx.config, "capture", None)
    has_tool_inputs = bool(capture and getattr(capture, "tool_inputs", False))
    optimize_cfg = getattr(ctx.config, "optimize", None)
    min_cohort_sessions = getattr(
        optimize_cfg, "min_cohort_sessions", MIN_COHORT_SESSIONS,
    )

    finding = VerbosityFinding(
        estimate_basis=VERBOSITY_ESTIMATE_BASIS,
        min_cohort_sessions=min_cohort_sessions,
    )

    # No calls in the window → nothing to attach a per-call figure to (mirrors
    # the empty-window contract every recoverable finding honours, #211).
    if ctx.summary.total_tokens == 0:
        ctx.report.findings["verbosity"] = finding
        return

    clauses = [
        "start_time >= $1", "start_time < $2", "session_id IS NOT NULL",
    ]
    params: list[Any] = [ctx.since, ctx.until]
    if ctx.agent_id:
        clauses.append(f"agent_id = ${len(params) + 1}")
        params.append(ctx.agent_id)
    where = " AND ".join(clauses)

    # LLM spans: output tokens per session (task shape carries no model on tool
    # spans, so MODE(model) is the session's dominant model for pricing).
    llm_rows = ctx.conn.execute(
        f"SELECT session_id, "
        f"FIRST(agent_id) AS agent_id, "
        f"MIN(provider) AS provider, "
        f"MODE(model) AS model, "
        f"COALESCE(SUM(output_tokens), 0) AS output_tokens, "
        f"COALESCE(SUM(input_tokens), 0) AS input_tokens "
        f"FROM spans WHERE {where} AND model IS NOT NULL "
        f"GROUP BY session_id",
        params,
    ).fetchall()

    if not llm_rows:
        ctx.report.findings["verbosity"] = finding
        return

    # Tool spans per session, ordered, to reconstruct the task-shape signature.
    tool_rows = ctx.conn.execute(
        f"SELECT session_id, tool_name, attributes "
        f"FROM spans WHERE {where} AND tool_name IS NOT NULL "
        f"ORDER BY session_id, start_time",
        params,
    ).fetchall()
    session_signatures: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
    for session_id, tool_name, attrs in tool_rows:
        seq = session_signatures.setdefault(str(session_id), [])
        tool_input = _extract_tool_input(attrs) if has_tool_inputs else None
        arg_sig = _arg_signature(tool_input) if has_tool_inputs else ()
        seq.append((str(tool_name), arg_sig))

    # Per-session facts, keyed by the task-shape signature cohort.
    @dataclass
    class _Session:
        session_id: str
        agent_id: str | None
        provider: str | None
        model: str
        output_tokens: int
        input_tokens: int
        signature: tuple[tuple[str, tuple[str, ...]], ...]

    sessions: list[_Session] = []
    for row in llm_rows:
        session_id, agent_id, provider, model, out_tok, in_tok = row
        if not model:
            continue
        sig = tuple(session_signatures.get(str(session_id), []))
        sessions.append(_Session(
            session_id=str(session_id),
            agent_id=str(agent_id) if agent_id else None,
            provider=str(provider) if provider else None,
            model=str(model),
            output_tokens=int(out_tok or 0),
            input_tokens=int(in_tok or 0),
            signature=sig,
        ))

    finding.sessions_examined = len(sessions)
    if not sessions:
        ctx.report.findings["verbosity"] = finding
        return

    cohorts: dict[tuple, list[_Session]] = {}
    for s in sessions:
        cohorts.setdefault(s.signature, []).append(s)

    candidates: list[VerbosityCandidate] = []
    total_recoverable_tokens = 0
    total_recoverable_usd = 0.0
    have_any_usd = False
    baselines: list[int] = []
    cohorts_examined = 0

    for signature, members in cohorts.items():
        if len(members) < min_cohort_sessions:
            continue
        outputs = [m.output_tokens for m in members]
        median = statistics.median(outputs)
        if median <= 0:
            continue
        cohorts_examined += 1
        baselines.append(int(round(median)))
        threshold = median * HIGH_VERBOSITY_MULTIPLE
        for m in members:
            if m.output_tokens <= threshold:
                continue
            over = int(round(m.output_tokens - median))
            if over <= 0:
                continue
            rates = get_rates(m.provider or "", m.model)
            recoverable_usd: float | None = None
            if rates is not None:
                recoverable_usd = round(
                    (over / 1_000_000) * rates.output_per_mtok, 6
                )
                total_recoverable_usd += recoverable_usd
                have_any_usd = True
            total_recoverable_tokens += over
            candidates.append(VerbosityCandidate(
                session_id=m.session_id,
                agent_id=m.agent_id,
                model=m.model,
                task_shape=_signature_repr(signature),
                cohort_sessions=len(members),
                output_tokens=m.output_tokens,
                baseline_output_tokens=int(round(median)),
                over_baseline_tokens=over,
                over_baseline_multiple=round(m.output_tokens / median, 2),
                # Weakest signal, descriptive only.
                output_input_ratio=(
                    round(m.output_tokens / m.input_tokens, 2)
                    if m.input_tokens > 0 else None
                ),
                recoverable_usd=recoverable_usd,
            ))

    finding.cohorts_examined = cohorts_examined

    if not candidates:
        ctx.report.findings["verbosity"] = finding
        return

    # Largest over-baseline output first — the most worthwhile to review.
    candidates.sort(key=lambda c: c.over_baseline_tokens, reverse=True)
    finding.total_candidates = len(candidates)
    finding.candidates = candidates[:MAX_EXAMPLES]
    finding.estimated_recoverable_tokens = total_recoverable_tokens
    finding.estimated_recoverable_usd = (
        round(total_recoverable_usd, 6) if have_any_usd else None
    )
    # Suggested max_tokens cap = the median cohort baseline (the point most
    # sessions already sit under). Advisory only — the remedy the user MEASURES.
    if baselines:
        finding.suggested_max_tokens = int(round(statistics.median(baselines)))

    ctx.report.findings["verbosity"] = finding
