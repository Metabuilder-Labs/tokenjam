"""Shared plumbing for the SDK workload corpus (examples/sdk_workloads/).

Every workload in this package:
  - reads OPENAI_API_KEY from the environment only (never hardcoded, never
    logged), and refuses to run without it unless --dry-run is passed;
  - checks a SpendGuard BEFORE every real API call and aborts the run the
    instant the projected cost would cross the ceiling;
  - can run entirely offline via --dry-run, which never imports a live
    OpenAI client and makes zero network calls, but still exercises the
    real tokenjam span pipeline (DuckDB, cost engine, optimize analyzers)
    with deterministic canned token counts;
  - tags every span with the SDK cost-attribution dimensions (tenant_id,
    feature, prompt_template_id/version) via tokenjam's attribution()
    context, and with `environment` via the OTEL_RESOURCE_ATTRIBUTES
    process env var (a resource attribute, so it must be set before the
    TracerProvider is built; see set_default_environment() below).

This module has no tokenjam-internal side effects of its own beyond reusing
tokenjam.core.cost.calculate_cost (the same pricing table `tj` itself uses),
so the guard's pre-call estimate and post-call actuals agree with what the
optimize report will later compute from the ingested spans.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

# CHARS_PER_TOKEN mirrors the same rough heuristic already used elsewhere in
# tokenjam (core/optimize/analyzers/deadweight.py, prompt_bloat.py) for a
# pre-call token estimate when no tokenizer is available.
CHARS_PER_TOKEN = 4

DEFAULT_MAX_SPEND_USD = 2.00
# A CURRENT cheap mini-class OpenAI model (tokenjam/pricing/models.toml's own
# comment: "gpt-4o / gpt-4o-mini / o3 / o4-mini are no longer on OpenAI's
# current pricing page but still appear in older telemetry"). Deliberately
# NOT gpt-4o-mini for that reason, even though it is still priced (for
# historical spans) in that table. `oversized_model.py` is the one workload
# that overrides this default on purpose, to gpt-4o; see its own docstring
# for why that specific legacy model is required there.
DEFAULT_MODEL = "gpt-5.4-mini"


def set_default_environment(environment: str) -> None:
    """Declare the `deployment.environment.name` resource attribute.

    Must be called before the first tokenjam span is created in this
    process (`OTEL_RESOURCE_ATTRIBUTES` is read once, when the OTel
    Resource is built; see tokenjam/otel/provider.py::_build_tj_resource).
    Uses setdefault semantics: an operator's own OTEL_RESOURCE_ATTRIBUTES
    always wins over this workload's default.
    """
    existing = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")
    attr = f"deployment.environment.name={environment}"
    if not existing:
        os.environ["OTEL_RESOURCE_ATTRIBUTES"] = attr
    elif "deployment.environment.name=" not in existing:
        os.environ["OTEL_RESOURCE_ATTRIBUTES"] = f"{existing},{attr}"


def require_api_key() -> str:
    """Read OPENAI_API_KEY or exit with a clear message. Never logs the key."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print(
            "ERROR: OPENAI_API_KEY environment variable is required "
            "(this workload makes real OpenAI API calls).\n"
            "  export OPENAI_API_KEY=sk-...\n"
            "Or run with --dry-run to exercise the same code path with zero "
            "API calls and zero spend.",
            file=sys.stderr,
        )
        sys.exit(1)
    return key


class SpendCeilingExceeded(RuntimeError):
    """Raised by SpendGuard.check_before_call() when a call would cross the ceiling."""


@dataclass
class SpendGuard:
    """Enforces a hard USD ceiling across a workload run.

    check_before_call() is the enforcement point: it must be called with a
    conservative cost ESTIMATE for the upcoming call before that call is
    issued. If the running total plus the estimate would exceed the
    ceiling, the call never happens; the guard raises instead of letting
    the workload proceed.

    record_actual() reconciles the estimate against the real cost once a
    response's usage is known (tokenjam's own calculate_cost(), so this
    guard's running total and tj's own ingested cost_usd sum agree).
    """
    ceiling_usd: float = DEFAULT_MAX_SPEND_USD
    total_spent_usd: float = field(default=0.0, init=False)
    calls_made: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.ceiling_usd <= 0:
            raise ValueError("SpendGuard ceiling_usd must be positive")

    def check_before_call(self, estimated_cost_usd: float) -> None:
        projected = self.total_spent_usd + max(estimated_cost_usd, 0.0)
        if projected > self.ceiling_usd:
            raise SpendCeilingExceeded(
                f"aborting before call #{self.calls_made + 1}: projected spend "
                f"${projected:.4f} would exceed the ${self.ceiling_usd:.2f} "
                f"max-spend ceiling (already spent ${self.total_spent_usd:.4f}, "
                f"this call is estimated at ${estimated_cost_usd:.4f}). "
                "Raise --max-spend if this run is intentional."
            )

    def record_actual(self, actual_cost_usd: float) -> None:
        self.total_spent_usd += max(actual_cost_usd, 0.0)
        self.calls_made += 1

    def report(self) -> str:
        return (
            f"cumulative spend: ${self.total_spent_usd:.4f} "
            f"across {self.calls_made} call(s) (ceiling ${self.ceiling_usd:.2f})"
        )


def estimate_tokens(text: str) -> int:
    return max(len(text) // CHARS_PER_TOKEN, 1)


def estimate_messages_tokens(messages: list[dict], tools: list[dict] | None = None) -> int:
    total_chars = sum(len(json.dumps(m)) for m in messages)
    if tools:
        total_chars += sum(len(json.dumps(t)) for t in tools)
    return estimate_tokens_from_chars(total_chars)


def estimate_tokens_from_chars(chars: int) -> int:
    return max(chars // CHARS_PER_TOKEN, 1)


# ---------------------------------------------------------------------------
# Deterministic canned OpenAI-shaped responses for --dry-run.
# ---------------------------------------------------------------------------


@dataclass
class _FakeFunction:
    name: str
    arguments: str


@dataclass
class _FakeToolCall:
    id: str
    function: _FakeFunction
    type: str = "function"


class _FakeMessage:
    def __init__(self, content: str | None, tool_calls: list[_FakeToolCall] | None = None) -> None:
        self.role = "assistant"
        self.content = content
        self.tool_calls = tool_calls or None

    def model_dump(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": (
                [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in self.tool_calls
                ]
                if self.tool_calls
                else None
            ),
        }


class _FakeChoice:
    def __init__(self, message: _FakeMessage, finish_reason: str) -> None:
        self.message = message
        self.finish_reason = finish_reason


class _FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class _FakeResponse:
    def __init__(self, choices: list[_FakeChoice], usage: _FakeUsage) -> None:
        self.choices = choices
        self.usage = usage


_DRY_RUN_TOOL_CALL_COUNTER = 0


def _canned_response(
    messages: list[dict],
    tools: list[dict] | None,
    force_tool_call: dict | None,
    max_tokens: int,
) -> _FakeResponse:
    """Build a deterministic canned response with realistic token counts.

    Input tokens scale with the actual message volume passed in (so a
    growing-context workload still produces growing input_tokens across
    turns in --dry-run, matching the shape a real run would have). Output
    tokens are fixed and small, capped by max_tokens; deterministic across
    runs, which matters for the before/after replay use case (temperature
    0 / fixed seed on the real path; a canned constant here).
    """
    prompt_tokens = estimate_messages_tokens(messages, tools)
    completion_tokens = min(48, max_tokens)

    if force_tool_call is not None:
        global _DRY_RUN_TOOL_CALL_COUNTER
        _DRY_RUN_TOOL_CALL_COUNTER += 1
        tool_call = _FakeToolCall(
            id=f"call_dryrun_{_DRY_RUN_TOOL_CALL_COUNTER}",
            function=_FakeFunction(
                name=force_tool_call["name"],
                arguments=json.dumps(force_tool_call.get("arguments", {})),
            ),
        )
        message = _FakeMessage(content=None, tool_calls=[tool_call])
        finish_reason = "tool_calls"
    else:
        message = _FakeMessage(content="[dry-run canned response]")
        finish_reason = "stop"

    return _FakeResponse(
        choices=[_FakeChoice(message=message, finish_reason=finish_reason)],
        usage=_FakeUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def guarded_chat_completion(
    client: Any,
    guard: SpendGuard,
    *,
    dry_run: bool,
    provider: str = "openai",
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    force_tool_call: dict | None = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
    seed: int = 42,
) -> Any:
    """Spend-guarded chat-completion call. Same return shape (an object with
    `.choices[0].message` and `.usage.prompt_tokens/.completion_tokens`)
    whether `dry_run` is True (canned, zero network) or False (real
    `openai.OpenAI().chat.completions.create`, patched by tokenjam so the
    call also produces a `gen_ai.llm.call` span).

    Determinism on the real path: temperature=0 and a fixed seed (OpenAI's
    best-effort determinism knob) so a before/after replay of the same
    workload varies only in the code under test, not in sampling noise.
    """
    from tokenjam.core.cost import calculate_cost

    estimated_input = estimate_messages_tokens(messages, tools)
    estimated_cost = calculate_cost(provider, model, estimated_input, max_tokens)
    guard.check_before_call(estimated_cost)

    if dry_run:
        response = _canned_response(messages, tools, force_tool_call, max_tokens)
        # The real path gets its span for free from patch_openai()'s monkeypatch
        # on openai.resources.chat.completions.Completions.create. There is no
        # such patched method to intercept a canned call, so record the span
        # manually; same span shape, zero network calls.
        from tokenjam.sdk.agent import record_llm_call

        record_llm_call(
            model=model,
            provider=provider,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
    else:
        kwargs: dict[str, Any] = dict(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
        )
        if tools:
            kwargs["tools"] = tools
        if force_tool_call is not None:
            kwargs["tool_choice"] = {
                "type": "function",
                "function": {"name": force_tool_call["name"]},
            }
        response = client.chat.completions.create(**kwargs)

    actual_cost = calculate_cost(
        provider,
        model,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )
    guard.record_actual(actual_cost)
    return response


def build_client(dry_run: bool):
    """Real openai.OpenAI() (with tokenjam's patch installed) in live mode;
    None in dry-run mode (guarded_chat_completion never touches it then)."""
    if dry_run:
        return None
    import openai

    from tokenjam.sdk.integrations.openai import patch_openai

    patch_openai()
    return openai.OpenAI()


# ---------------------------------------------------------------------------
# CLI plumbing shared by every workload's __main__ block.
# ---------------------------------------------------------------------------


def build_arg_parser(description: str, *, default_model: str = DEFAULT_MODEL) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Exercise the full workload with a stubbed client; zero API calls, zero spend.",
    )
    parser.add_argument(
        "--max-spend",
        type=float,
        default=DEFAULT_MAX_SPEND_USD,
        help=f"Hard USD ceiling enforced before every call (default: ${DEFAULT_MAX_SPEND_USD:.2f}).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help=f"Model to use (default: {default_model}).",
    )
    return parser


def run_workload_main(
    description: str,
    run_fn: Callable[..., None],
    *,
    default_model: str = DEFAULT_MODEL,
    extra_args: Callable[[argparse.ArgumentParser], None] | None = None,
) -> None:
    """Standard __main__ entry point shared by every workload script.

    `run_fn(client, guard, dry_run, model)` does the actual work. Handles:
    arg parsing, the missing-API-key guard, client construction, and
    printing the final cumulative spend so the operator sees it on every run
    (dry-run or real).
    """
    parser = build_arg_parser(description, default_model=default_model)
    if extra_args is not None:
        extra_args(parser)
    args = parser.parse_args()

    if not args.dry_run:
        require_api_key()

    guard = SpendGuard(ceiling_usd=args.max_spend)
    client = build_client(args.dry_run)

    try:
        run_fn(client, guard, args.dry_run, args.model, args)
    except SpendCeilingExceeded as exc:
        print(f"\n[spend-guard] {exc}", file=sys.stderr)
        print(f"[spend-guard] {guard.report()}", file=sys.stderr)
        sys.exit(2)
    finally:
        _flush_tokenjam()

    print(f"\n[spend-guard] {guard.report()}")


def _flush_tokenjam() -> None:
    """Force the batched span exporter to write everything before this
    process exits, so a harness reading the DB right after we return sees
    every span (BatchSpanProcessor otherwise flushes on its own schedule)."""
    try:
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush(timeout_millis=30_000)
    except Exception:
        pass
