"""
Empirical validation of an *estimated* recoverable finding (issue #477).

`tj optimize --validate <finding>` turns a heuristic estimate into a MEASURED
result by re-running the candidate against the baseline on a small sample of the
user's OWN recorded calls (via their own API key), then reporting the measured
token/cost delta plus a quality-preservation check.

Honesty discipline (CLAUDE.md Critical Rule 14). The framing is always
"measured on a sample of N calls" — the sample size is reported prominently so
the figure can never read as a fleet-wide promise. This module NEVER emits
"certified" or "guaranteed": that vocabulary is reserved for a separate, paid
layer, and using it here would dilute it. See :data:`VALIDATE_HONESTY_CAVEAT`.

Reuse note (the "one measurement engine" goal). The A/B-run + scoring primitive
is conceptually shared with the standalone ``tokenjam-bench`` (``tjbench``)
project. That project is published and its harness is cleanly factored, but it
depends on ``tokenjam>=0.5`` — so importing it here would make the OSS package
depend on a package that depends back on it (a cycle), and its harness is built
around STANDARD benchmark scenarios (HumanEval / GSM8K) rather than the user's
own telemetry. So v1 keeps a deliberately MINIMAL, self-contained measurement
here (this module + the provider-client protocol below), isolated so it can
later converge with ``tjbench``'s engine once that engine is split from its
``tokenjam`` dependency. Do not grow this into a second full bench harness.

Scope (v1): the ``downsize`` finding only. ``summarize``/``trim`` (candidate =
a trimmed prompt) share the same engine but need a per-finding candidate builder
and a content-presence quality signal; deferred. ``cache`` is deterministic and
out of scope by design.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Protocol

from tokenjam.core.cost import calculate_cost
from tokenjam.core.optimize.analyzers.model_downgrade import (
    SMALL_INPUT_TOKENS,
    SMALL_OUTPUT_TOKENS,
    lookup_downgrade,
)
from tokenjam.otel.semconv import GenAIAttributes

logger = logging.getLogger(__name__)

# Mandatory honesty caveat (Rule 14). Carried as a dataclass default on
# ValidationResult so no surface can drop it. States the ONE thing a small
# empirical sample can and cannot claim: it measured THESE N calls, it does not
# promise the swap holds for the rest of your usage.
VALIDATE_HONESTY_CAVEAT = (
    "Measured on a sample of your own recorded calls, not a guarantee. "
    "A larger sample and human review of the outputs remain worthwhile "
    "before switching models."
)

# Sampling defaults. Small and cost-bounded — this spends the user's own money
# on live API calls, so we keep K low and always confirm the estimate first.
DEFAULT_SAMPLE_SIZE = 5
MAX_SAMPLE_SIZE = 20

# Env var holding the Anthropic key (matches the e2e/backfill convention).
ANTHROPIC_KEY_ENV = "TJ_ANTHROPIC_API_KEY"


# ---------------------------------------------------------------------------
# Provider client seam (mocked in tests — NO real API calls in the test suite)
# ---------------------------------------------------------------------------


@dataclass
class Completion:
    """One provider response, normalized to what the measurement needs."""

    text: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class ProviderClient(Protocol):
    """Minimal seam over a real provider. Tests inject a mock implementing this;
    the live path (:class:`AnthropicProviderClient`) calls the real API."""

    def complete(
        self,
        *,
        provider: str,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
    ) -> Completion:
        ...


class AnthropicProviderClient:
    """Live Anthropic client. Constructed only on the real ``--validate`` path,
    never in tests. Reads the key from ``TJ_ANTHROPIC_API_KEY``.
    """

    def __init__(self, api_key: str) -> None:
        import anthropic  # deferred — only needed on the live path

        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(
        self,
        *,
        provider: str,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
    ) -> Completion:
        resp = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
        )
        # Concatenate text blocks; a tool-only turn yields "".
        text = "".join(
            getattr(block, "text", "") or ""
            for block in getattr(resp, "content", []) or []
        )
        usage = getattr(resp, "usage", None)
        return Completion(
            text=text,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            cache_read_tokens=int(
                getattr(usage, "cache_read_input_tokens", 0) or 0
            ),
            cache_write_tokens=int(
                getattr(usage, "cache_creation_input_tokens", 0) or 0
            ),
        )


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SampledCall:
    """One recorded LLM call selected for replay. Carries the captured request
    messages (present only when [capture] prompts is on) plus the recorded
    provider/model and token counts used for the up-front cost estimate."""

    span_id: str
    session_id: str | None
    provider: str
    model: str
    candidate_model: str
    messages: list[dict[str, Any]]
    max_tokens: int
    recorded_input_tokens: int
    recorded_output_tokens: int


@dataclass
class CallMeasurement:
    """Measured baseline-vs-candidate outcome for a single replayed call."""

    span_id: str
    model: str
    candidate_model: str
    baseline_input_tokens: int
    baseline_output_tokens: int
    baseline_cost_usd: float
    candidate_input_tokens: int
    candidate_output_tokens: int
    candidate_cost_usd: float
    quality_preserved: bool


@dataclass
class ValidationResult:
    """Aggregate measured result over the replayed sample.

    All figures are MEASURED (from live re-runs), not estimated — but only over
    ``sample_size`` calls, which every renderer must surface. Honesty (Rule 14):
    ``caveat`` is mandatory and defaults to :data:`VALIDATE_HONESTY_CAVEAT`.
    """

    finding: str
    sample_size: int
    baseline_tokens: int
    candidate_tokens: int
    baseline_cost_usd: float
    candidate_cost_usd: float
    quality_preserved: int
    measurements: list[CallMeasurement] = field(default_factory=list)
    caveat: str = VALIDATE_HONESTY_CAVEAT

    @property
    def tokens_delta(self) -> int:
        return self.candidate_tokens - self.baseline_tokens

    @property
    def tokens_delta_pct(self) -> float | None:
        if self.baseline_tokens <= 0:
            return None
        return self.tokens_delta / self.baseline_tokens * 100.0

    @property
    def cost_delta_usd(self) -> float:
        return self.candidate_cost_usd - self.baseline_cost_usd

    @property
    def cost_delta_pct(self) -> float | None:
        if self.baseline_cost_usd <= 0:
            return None
        return self.cost_delta_usd / self.baseline_cost_usd * 100.0


class ValidationError(Exception):
    """Raised for an actionable, user-facing validation precondition failure
    (capture off, no samples, no key). The CLI renders the message verbatim."""


# ---------------------------------------------------------------------------
# Sampling — pull the finding's own recorded calls out of telemetry
# ---------------------------------------------------------------------------


def collect_downsize_samples(
    conn,
    since: datetime,
    until: datetime,
    agent_id: str | None,
    sample_size: int,
) -> list[SampledCall]:
    """Select up to ``sample_size`` recorded calls that match the downsize
    heuristic AND carry captured prompt content, ready to replay.

    Mirrors the structural filter in ``analyze_model_downgrade`` (a downgrade
    candidate model + small-shape spans) so we validate the same class of call
    the finding flagged — but here at the SPAN grain, because each replay is one
    provider request. A span with no captured ``gen_ai.prompt.content`` cannot be
    replayed and is skipped (the capture gate is enforced by the caller; this is
    the belt-and-braces per-span check).
    """
    # The attribute key ("gen_ai.prompt.content") itself contains dots, so it
    # must be a QUOTED JSON path segment ($."gen_ai.prompt.content") — an
    # unquoted $.gen_ai.prompt.content reads it as nested keys and always
    # returns NULL. Passed as a bound parameter (never f-string SQL, Rule 7).
    prompt_path = f'$."{GenAIAttributes.PROMPT_CONTENT}"'
    clauses = [
        "start_time >= $1",
        "start_time < $2",
        "model IS NOT NULL",
        "provider IS NOT NULL",
        "json_extract_string(attributes, $3) IS NOT NULL",
    ]
    params: list[Any] = [since, until, prompt_path]
    if agent_id:
        clauses.append(f"agent_id = ${len(params) + 1}")
        params.append(agent_id)
    where = " AND ".join(clauses)

    rows = conn.execute(
        "SELECT span_id, session_id, provider, model, "
        "json_extract_string(attributes, $3) AS prompt, "
        "COALESCE(input_tokens, 0) AS input_tokens, "
        "COALESCE(output_tokens, 0) AS output_tokens "
        f"FROM spans WHERE {where} "
        "ORDER BY start_time DESC",
        params,
    ).fetchall()

    samples: list[SampledCall] = []
    for span_id, session_id, provider, model, prompt, in_tok, out_tok in rows:
        if len(samples) >= sample_size:
            break
        candidate = lookup_downgrade(provider, model)
        if not candidate:
            continue
        # Same small-shape gate the finding uses, at the span grain.
        if not (
            int(in_tok or 0) < SMALL_INPUT_TOKENS
            and int(out_tok or 0) < SMALL_OUTPUT_TOKENS
        ):
            continue
        messages = _parse_messages(prompt)
        if not messages:
            continue
        samples.append(
            SampledCall(
                span_id=str(span_id),
                session_id=str(session_id) if session_id else None,
                provider=str(provider),
                model=str(model),
                candidate_model=candidate,
                messages=messages,
                # Bound the replay's output at a small multiple of what was
                # actually produced, so a runaway generation can't blow the
                # cost estimate. Floor keeps a very short recorded call usable.
                max_tokens=max(int(out_tok or 0) * 2, 256),
                recorded_input_tokens=int(in_tok or 0),
                recorded_output_tokens=int(out_tok or 0),
            )
        )
    return samples


def _parse_messages(raw: str | None) -> list[dict[str, Any]]:
    """Coerce a captured ``gen_ai.prompt.content`` value into a messages list.

    Capture stores ``json.dumps(messages)`` (see _request_capture), usually a
    list of ``{"role", "content"}`` dicts. We accept that shape; anything we
    can't turn into a non-empty messages list yields ``[]`` (skip — unreplayable).
    """
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return []
    if isinstance(parsed, list):
        return [m for m in parsed if isinstance(m, dict) and m.get("content")]
    return []


# ---------------------------------------------------------------------------
# Up-front cost estimate (shown before any spend; confirmation gates the run)
# ---------------------------------------------------------------------------


def estimate_sample_cost(samples: list[SampledCall]) -> float:
    """Rough USD ceiling for replaying the sample BOTH ways (baseline +
    candidate), from the recorded token counts. Deliberately an over-estimate
    (output priced at the replay's ``max_tokens`` cap) so the confirmation figure
    is a ceiling, never an under-promise the actual spend can exceed."""
    total = 0.0
    for s in samples:
        total += calculate_cost(
            s.provider, s.model, s.recorded_input_tokens, s.max_tokens,
        )
        total += calculate_cost(
            s.provider, s.candidate_model, s.recorded_input_tokens, s.max_tokens,
        )
    return round(total, 4)


# ---------------------------------------------------------------------------
# Quality signal (v1: exact-match on normalized text)
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Whitespace/case-insensitive normalization for exact-match comparison —
    so trailing newlines or casing don't count as a quality regression."""
    return _WS_RE.sub(" ", text or "").strip().lower()


def exact_match(baseline_text: str, candidate_text: str) -> bool:
    """v1 quality signal: the candidate output exactly matches the baseline
    output after whitespace/case normalization. Suited to deterministic and
    tool-shaped outputs (the finding's own target class). LLM-judge and
    embedding-similarity for open-ended text are a deliberate later follow-up."""
    return _normalize(baseline_text) == _normalize(candidate_text)


# ---------------------------------------------------------------------------
# The A/B run — the minimal measurement engine
# ---------------------------------------------------------------------------


def run_validation(
    samples: list[SampledCall],
    client: ProviderClient,
    *,
    finding: str = "downsize",
) -> ValidationResult:
    """Re-run each sampled call two ways (baseline model vs candidate model)
    through ``client``, measure ACTUAL tokens + cost for both, and score quality.

    Cost is computed with the same :func:`calculate_cost` the rest of TokenJam
    uses, so the measured dollars are consistent with the estimated ones.
    """
    measurements: list[CallMeasurement] = []
    baseline_tokens = candidate_tokens = 0
    baseline_cost = candidate_cost = 0.0
    preserved = 0

    for s in samples:
        base = client.complete(
            provider=s.provider, model=s.model,
            messages=s.messages, max_tokens=s.max_tokens,
        )
        cand = client.complete(
            provider=s.provider, model=s.candidate_model,
            messages=s.messages, max_tokens=s.max_tokens,
        )
        base_cost = calculate_cost(
            s.provider, s.model,
            base.input_tokens, base.output_tokens,
            base.cache_read_tokens, base.cache_write_tokens,
        )
        cand_cost = calculate_cost(
            s.provider, s.candidate_model,
            cand.input_tokens, cand.output_tokens,
            cand.cache_read_tokens, cand.cache_write_tokens,
        )
        ok = exact_match(base.text, cand.text)
        if ok:
            preserved += 1

        base_tok = base.input_tokens + base.output_tokens
        cand_tok = cand.input_tokens + cand.output_tokens
        baseline_tokens += base_tok
        candidate_tokens += cand_tok
        baseline_cost += base_cost
        candidate_cost += cand_cost

        measurements.append(CallMeasurement(
            span_id=s.span_id,
            model=s.model,
            candidate_model=s.candidate_model,
            baseline_input_tokens=base.input_tokens,
            baseline_output_tokens=base.output_tokens,
            baseline_cost_usd=round(base_cost, 8),
            candidate_input_tokens=cand.input_tokens,
            candidate_output_tokens=cand.output_tokens,
            candidate_cost_usd=round(cand_cost, 8),
            quality_preserved=ok,
        ))

    return ValidationResult(
        finding=finding,
        sample_size=len(samples),
        baseline_tokens=baseline_tokens,
        candidate_tokens=candidate_tokens,
        baseline_cost_usd=round(baseline_cost, 8),
        candidate_cost_usd=round(candidate_cost, 8),
        quality_preserved=preserved,
        measurements=measurements,
    )


# ---------------------------------------------------------------------------
# Serialization (--json)
# ---------------------------------------------------------------------------


def result_to_dict(result: ValidationResult) -> dict[str, Any]:
    """JSON-serialisable view of a :class:`ValidationResult`."""
    return {
        "finding": result.finding,
        "sample_size": result.sample_size,
        "baseline_tokens": result.baseline_tokens,
        "candidate_tokens": result.candidate_tokens,
        "tokens_delta": result.tokens_delta,
        "tokens_delta_pct": (
            round(result.tokens_delta_pct, 1)
            if result.tokens_delta_pct is not None else None
        ),
        "baseline_cost_usd": round(result.baseline_cost_usd, 6),
        "candidate_cost_usd": round(result.candidate_cost_usd, 6),
        "cost_delta_usd": round(result.cost_delta_usd, 6),
        "cost_delta_pct": (
            round(result.cost_delta_pct, 1)
            if result.cost_delta_pct is not None else None
        ),
        "quality_preserved": result.quality_preserved,
        "quality_metric": "exact_match",
        "measurements": [asdict(m) for m in result.measurements],
        "caveat": result.caveat,
        "basis": f"measured on a sample of {result.sample_size} of your recorded calls",
    }
