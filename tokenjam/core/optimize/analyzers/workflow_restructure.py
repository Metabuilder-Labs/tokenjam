"""
Workflow-restructure analyzer.

Clusters sessions by their tool-call signature and flags clusters where a
deterministic agent run could likely be replaced with a deterministic
script. Conservative thresholds — false positives erode user trust here
much faster than missed opportunities, so v1 only flags clusters with
many instances and tight uniformity.

Signature definition (locked decision):
  Signature = ordered tuple of (tool_name, arg_shape) pairs.

  `arg_shape` describes which argument TYPES are present, not values:
    - file_path     argument resolves to a filesystem path
    - command_string argument is a shell command (heuristic)
    - json_object   argument is a JSON-shaped value (dict)
    - array         argument is list-typed
    - number, boolean  primitives
    - string        generic string

Two sessions with identical signatures may have very different argument
*values* — that's intentional. We're clustering by structural shape, not
content. The deterministic-script recommendation only makes sense when
the *structure* is fixed even if the *values* vary.

When `[capture] tool_inputs = false`, the analyzer degrades to
tool-names-only clustering. Recommendations still surface but the
confidence note discloses the degradation.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from tokenjam.core.optimize.registry import register
from tokenjam.core.optimize.types import AnalyzerContext
from tokenjam.otel.semconv import GenAIAttributes

# Conservative threshold: a cluster needs at least this many sessions
# before we'll recommend replacing it with a script. Lower thresholds
# generate noisy recommendations that look like deterministic patterns
# but are actually rare workflows the user genuinely wants flexibility on.
MIN_CLUSTER_INSTANCES = 20

# Patterns for arg_shape classification. Tested in order, first match wins.
_PATH_RE = re.compile(r"^[~/.][^\s]*$|^[a-zA-Z]:[\\/]")  # rough heuristic for paths
_COMMAND_RE = re.compile(r"^(npm|pnpm|yarn|git|pip|cargo|make|sh|bash|cd|ls|cat|grep|rm|mv|cp|docker|kubectl|curl|tj|pytest|ruff|mypy)\b")


def _classify_arg(value: Any) -> str:
    """Map a single tool-call argument value to its arg_shape category."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "json_object"
    if isinstance(value, str):
        if _PATH_RE.match(value):
            return "file_path"
        if _COMMAND_RE.match(value):
            return "command_string"
        return "string"
    return "string"  # null / unknown → generic string slot


def _arg_signature(tool_input: Any) -> tuple[str, ...]:
    """
    Reduce a tool call's input payload to a tuple of arg_shape labels.

    `tool_input` is typically a dict mapping arg-name → value. We sort
    keys so signature ordering doesn't depend on dict iteration order.
    A non-dict input is treated as a single positional arg.
    """
    if tool_input is None:
        return ()
    if not isinstance(tool_input, dict):
        return (_classify_arg(tool_input),)
    return tuple(_classify_arg(tool_input[k]) for k in sorted(tool_input.keys()))


@dataclass
class WorkflowCluster:
    """One cluster of sessions sharing the same signature."""
    signature:     list[dict]   # human-readable: [{"tool": "...", "args": [...]}]
    instances:     int          # number of sessions matching this signature
    avg_cost_usd:  float        # mean cost across the cluster
    avg_duration_seconds: float | None
    example_session_id: str | None
    # Mean input+output tokens per instance, so the UI can render the per-cluster
    # cost as TOKENS for subscription/local users (#260) — "% of cycle"/$ at this
    # per-item granularity is the same category error fixed in #249/#259. Summed
    # server-side (single compute path). Defaulted so older serialized reports
    # round-trip through WorkflowCluster(**c).
    avg_tokens:    int = 0


@dataclass
class WorkflowRestructureFinding:
    """Deterministic-workflow clusters flagged for script-replacement review."""
    clusters:    list[WorkflowCluster] = field(default_factory=list)
    sessions_examined: int = 0
    degraded:    bool = False   # true when arg_shape couldn't be computed
    confidence:  str = "structural"
    caveat:      str = (
        "Conservative cluster detection. Review each cluster before replacing "
        "with a script — value variation that the heuristic can't see may matter."
    )
    # Recoverable-savings contract (#111). See types.DowngradeFinding for field
    # semantics. None when no cluster cleared the threshold.
    estimated_recoverable_usd:    float | None = None
    estimated_recoverable_tokens: int | None   = None
    estimate_basis:               str          = ""
    estimate_confidence:          str          = "heuristic"


def _extract_tool_input(attrs: Any) -> Any:
    """Pull gen_ai.tool.input from a span's attributes JSON. Returns None when absent."""
    if isinstance(attrs, str):
        try:
            attrs = json.loads(attrs)
        except Exception:
            return None
    if not isinstance(attrs, dict):
        return None
    return attrs.get(GenAIAttributes.TOOL_INPUT)


@register("script")
def run(ctx: AnalyzerContext) -> None:
    """Registry entry point. Attaches a WorkflowRestructureFinding to ctx.report.findings."""
    capture = getattr(ctx.config, "capture", None)
    has_tool_inputs = bool(capture and getattr(capture, "tool_inputs", False))

    # Query tool spans within the window, ordered per-session by start_time
    # so we can reconstruct the call sequence in order.
    clauses = [
        "start_time >= $1", "start_time < $2",
        "tool_name IS NOT NULL", "session_id IS NOT NULL",
    ]
    params: list[Any] = [ctx.since, ctx.until]
    if ctx.agent_id:
        clauses.append(f"agent_id = ${len(params) + 1}")
        params.append(ctx.agent_id)
    where = " AND ".join(clauses)
    rows = ctx.conn.execute(
        f"SELECT session_id, tool_name, attributes "
        f"FROM spans WHERE {where} "
        f"ORDER BY session_id, start_time",
        params,
    ).fetchall()

    if not rows:
        ctx.report.findings["script"] = WorkflowRestructureFinding(
            degraded=not has_tool_inputs,
        )
        return

    # Build per-session signature lists.
    session_signatures: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
    for session_id, tool_name, attrs in rows:
        sig_seq = session_signatures.setdefault(str(session_id), [])
        tool_input = _extract_tool_input(attrs) if has_tool_inputs else None
        arg_sig = _arg_signature(tool_input) if has_tool_inputs else ()
        sig_seq.append((str(tool_name), arg_sig))

    # Cluster sessions by full signature (tuple of (tool, arg-shape)) tuples.
    cluster_members: dict[tuple, list[str]] = {}
    for session_id, seq in session_signatures.items():
        key = tuple(seq)
        cluster_members.setdefault(key, []).append(session_id)

    # Build the report rows. Only clusters above the threshold are surfaced.
    interesting: list[WorkflowCluster] = []
    total_cluster_cost = 0.0   # recoverable USD: full cost of clustered sessions
    total_cluster_tokens = 0   # recoverable tokens: replacing the call frees them
    for signature, members in cluster_members.items():
        if len(members) < MIN_CLUSTER_INSTANCES:
            continue

        # Aggregate session-level cost + tokens + duration for the cluster.
        placeholders = ",".join(f"${i + 1}" for i in range(len(members)))
        agg = ctx.conn.execute(
            f"SELECT "
            f"COALESCE(AVG(total_cost_usd), 0.0), "
            f"COALESCE(AVG(EXTRACT(EPOCH FROM (ended_at - started_at))), 0.0), "
            f"COALESCE(SUM(total_cost_usd), 0.0), "
            f"COALESCE(SUM(input_tokens + output_tokens + cache_tokens + cache_write_tokens), 0), "
            f"COALESCE(AVG(input_tokens + output_tokens), 0) "
            f"FROM sessions WHERE session_id IN ({placeholders})",
            members,
        ).fetchone()
        avg_cost = float(agg[0] or 0.0) if agg else 0.0
        avg_duration = float(agg[1] or 0.0) if agg else 0.0
        cluster_cost = float(agg[2] or 0.0) if agg else 0.0
        cluster_tokens = int(agg[3] or 0) if agg else 0
        # Per-instance avg of input+output (the same basis the UI's _costVal uses
        # for every other per-item token figure) so the cluster cell reads
        # consistently with Traces/Status (#260).
        avg_tokens = round(float(agg[4] or 0.0)) if agg else 0
        total_cluster_cost += cluster_cost
        total_cluster_tokens += cluster_tokens

        signature_repr: list[dict] = []
        for tool_name, arg_sig in signature:
            entry: dict[str, Any] = {"tool": tool_name}
            if arg_sig:
                entry["args"] = list(arg_sig)
            signature_repr.append(entry)

        interesting.append(WorkflowCluster(
            signature=signature_repr,
            instances=len(members),
            avg_cost_usd=round(avg_cost, 6),
            avg_duration_seconds=round(avg_duration, 2) if avg_duration > 0 else None,
            example_session_id=members[0],
            avg_tokens=avg_tokens,
        ))

    # Sort by instance count desc — the most common deterministic patterns
    # are the biggest savings opportunities.
    interesting.sort(key=lambda c: c.instances, reverse=True)

    has_clusters = bool(interesting)
    ctx.report.findings["script"] = WorkflowRestructureFinding(
        clusters=interesting,
        sessions_examined=len(session_signatures),
        degraded=not has_tool_inputs,
        estimated_recoverable_usd=(
            round(total_cluster_cost, 6) if has_clusters else None
        ),
        estimated_recoverable_tokens=(
            total_cluster_tokens if has_clusters else None
        ),
        estimate_basis=(
            "total cost of sessions matching a deterministic call-pattern — "
            "replacing the cluster with a script eliminates the LLM call entirely"
        ),
    )
