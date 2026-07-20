"""
SDK entry points: @watch() decorator, AgentSession context manager,
and manual span recording functions (record_llm_call, record_tool_call,
record_outcome).

IMPORTANT: @watch() alone tracks session start/end only. Individual LLM call
spans require patch_anthropic(), patch_openai(), or equivalent provider patches.
"""
from __future__ import annotations

import functools
import logging
from contextlib import AbstractContextManager
from typing import Any, Callable, TYPE_CHECKING

from opentelemetry import trace

from tokenjam.otel.semconv import GenAIAttributes, TjAttributes
from tokenjam.utils.ids import new_uuid

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_tracer = trace.get_tracer("tokenjam.sdk")


def watch(
    agent_id: str,
    *,
    agent_name: str | None = None,
    agent_version: str | None = None,
    conversation_id: str | None = None,
):
    """
    Decorator that wraps an agent entry function with session tracking.

    Creates an OTel span named "invoke_agent" with agent identity and
    conversation attributes. Tracks session start/end/duration only.

    Individual LLM call spans are NOT created automatically — they require
    patch_anthropic(), patch_openai(), or equivalent provider patches.

    Never crashes the agent — if something goes wrong internally, it logs
    a warning and runs the function unwrapped.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from tokenjam.sdk.bootstrap import ensure_initialised
            ensure_initialised()
            try:
                with AgentSession(
                    agent_id=agent_id,
                    agent_name=agent_name,
                    agent_version=agent_version,
                    conversation_id=conversation_id,
                ):
                    return func(*args, **kwargs)
            except Exception:
                # Re-raise application exceptions, but if AgentSession
                # itself fails to initialise, fall through to unwrapped call
                raise

        return wrapper
    return decorator


class AgentSession:
    """
    Context manager for an agent session. Used by @watch() and can also be
    used directly for more control.

    Usage:
        with AgentSession(agent_id="my-agent") as session:
            result = run_my_agent()
    """

    def __init__(
        self,
        agent_id: str,
        agent_name: str | None = None,
        agent_version: str | None = None,
        conversation_id: str | None = None,
    ):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.agent_version = agent_version
        self.conversation_id = conversation_id or new_uuid()
        self._span: trace.Span | None = None
        self._ctx: AbstractContextManager[trace.Span] | None = None

    def __enter__(self) -> AgentSession:
        self._span = _tracer.start_span(GenAIAttributes.SPAN_INVOKE_AGENT)
        self._span.set_attribute(GenAIAttributes.AGENT_ID, self.agent_id)
        if self.agent_name:
            self._span.set_attribute(GenAIAttributes.AGENT_NAME, self.agent_name)
        if self.agent_version:
            self._span.set_attribute(GenAIAttributes.AGENT_VERSION, self.agent_version)
        self._span.set_attribute(
            GenAIAttributes.CONVERSATION_ID, self.conversation_id,
        )
        self._ctx = trace.use_span(self._span, end_on_exit=False)
        self._ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._span is None:
            return False
        if exc_type is not None:
            self._span.set_status(
                trace.Status(trace.StatusCode.ERROR, str(exc_val))
            )
        else:
            self._span.set_status(trace.Status(trace.StatusCode.OK))
        self._span.end()
        if self._ctx is not None:
            self._ctx.__exit__(exc_type, exc_val, exc_tb)
        return False  # Never suppress exceptions


def record_llm_call(
    model: str,
    provider: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    duration_ms: float | None = None,
    prompt: str | None = None,
    completion: str | None = None,
) -> None:
    """
    Manual instrumentation: record a single LLM call as an OTel span.
    Use this when no provider patch is available.

    Creates a child span under the current active span (typically set by
    @watch() / AgentSession).
    """
    span = _tracer.start_span(GenAIAttributes.SPAN_LLM_CALL)
    parent_span = trace.get_current_span()
    if parent_span and parent_span.is_recording():
        agent_id = parent_span.attributes.get(GenAIAttributes.AGENT_ID)
        if agent_id:
            span.set_attribute(GenAIAttributes.AGENT_ID, agent_id)
        conv_id = parent_span.attributes.get(GenAIAttributes.CONVERSATION_ID)
        if conv_id:
            span.set_attribute(GenAIAttributes.CONVERSATION_ID, conv_id)
    span.set_attribute(GenAIAttributes.REQUEST_MODEL, model)
    span.set_attribute(GenAIAttributes.PROVIDER_NAME, provider)
    span.set_attribute(GenAIAttributes.INPUT_TOKENS, input_tokens)
    span.set_attribute(GenAIAttributes.OUTPUT_TOKENS, output_tokens)
    if cache_read_tokens:
        span.set_attribute(GenAIAttributes.CACHE_READ_TOKENS, cache_read_tokens)
    if prompt is not None:
        span.set_attribute(GenAIAttributes.PROMPT_CONTENT, prompt)
    if completion is not None:
        span.set_attribute(GenAIAttributes.COMPLETION_CONTENT, completion)
    span.set_status(trace.Status(trace.StatusCode.OK))
    if duration_ms is not None:
        # Set explicit end time based on duration
        start_ns = span.start_time
        if start_ns:
            end_ns = start_ns + int(duration_ms * 1_000_000)
            span.end(end_time=end_ns)
            return
    span.end()


def record_tool_call(
    tool_name: str,
    tool_input: dict | None = None,
    tool_output: dict | None = None,
    duration_ms: float | None = None,
    error: str | None = None,
) -> None:
    """
    Manual instrumentation: record a single tool call as an OTel span.

    Creates a child span under the current active span.
    """
    span = _tracer.start_span(GenAIAttributes.SPAN_TOOL_CALL)
    parent_span = trace.get_current_span()
    if parent_span and parent_span.is_recording():
        agent_id = parent_span.attributes.get(GenAIAttributes.AGENT_ID)
        if agent_id:
            span.set_attribute(GenAIAttributes.AGENT_ID, agent_id)
        conv_id = parent_span.attributes.get(GenAIAttributes.CONVERSATION_ID)
        if conv_id:
            span.set_attribute(GenAIAttributes.CONVERSATION_ID, conv_id)
    span.set_attribute(GenAIAttributes.TOOL_NAME, tool_name)
    if tool_input is not None:
        import json
        span.set_attribute(GenAIAttributes.TOOL_INPUT, json.dumps(tool_input))
    if tool_output is not None:
        import json
        span.set_attribute(GenAIAttributes.TOOL_OUTPUT, json.dumps(tool_output))
    if error:
        span.set_status(trace.Status(trace.StatusCode.ERROR, error))
    else:
        span.set_status(trace.Status(trace.StatusCode.OK))
    if duration_ms is not None:
        start_ns = span.start_time
        if start_ns:
            end_ns = start_ns + int(duration_ms * 1_000_000)
            span.end(end_time=end_ns)
            return
    span.end()


def record_outcome(
    outcome_type: str,
    *,
    workflow_id: str | None = None,
    session_id: str | None = None,
    success: bool = True,
    value_usd: float | None = None,
    attributes: dict[str, Any] | None = None,
) -> None:
    """Emit a gen_ai outcome event attaching a business outcome to a workflow.

    This is a thin wrapper that emits one OTel span carrying the emerging
    gen_ai outcome-event attributes (OTel semconv issue #2665). It sits
    alongside record_llm_call / record_tool_call as a manual-instrumentation
    primitive — one line instead of hand-POSTing OTLP JSON.

    Args:
        outcome_type: the kind of outcome, a caller-defined label
            (e.g. "ticket_resolved", "lead_qualified", "pr_merged"). This is the
            marker attribute; without it the event is not recognised downstream.
        workflow_id: an explicit workflow key to attach the outcome to. Optional
            if session_id is given (the outcome is then keyed to the session's
            workflow). At least one of workflow_id / session_id is required.
        session_id: the session (or root session of a fan-out) the outcome
            belongs to. If omitted, the active @watch()/AgentSession span's
            session/conversation is inherited where available.
        success: whether the outcome was achieved (execution succeeded). Defaults
            to True.
        value_usd: an OPTIONAL, SELF-REPORTED business value for the outcome in
            USD. This is a value YOU declare — TokenJam does not measure or
            verify it. Negative values are treated as undeclared. ROI compute
            (declared value ÷ measured cost) is a TokenJam Cloud feature; the OSS
            SDK only emits the event.
        attributes: extra attributes to attach verbatim to the event.

    Raises:
        ValueError: if outcome_type is empty, or neither workflow_id nor
            session_id is provided (mirrors the Cloud OutcomeIn validator).
    """
    if not outcome_type:
        raise ValueError("record_outcome requires a non-empty outcome_type")

    from tokenjam.sdk.bootstrap import ensure_initialised
    ensure_initialised()

    # Resolve session/agent inheritance from the active span BEFORE starting our
    # own span, so a validation failure emits no half-built outcome span.
    parent_span = trace.get_current_span()
    inherited_agent = None
    inherited_conv = None
    inherited_session: str | None = None
    if parent_span and parent_span.is_recording():
        inherited_agent = parent_span.attributes.get(GenAIAttributes.AGENT_ID)
        inherited_conv = parent_span.attributes.get(GenAIAttributes.CONVERSATION_ID)
        # Inherit the active session key. @watch()/AgentSession stamps its stable
        # id as the conversation id (session continuity keys off it); a raw span
        # may instead carry an explicit session.id. Prefer the explicit one.
        explicit = parent_span.attributes.get(TjAttributes.SESSION_ID)
        inherited_session = explicit if explicit is not None else inherited_conv

    resolved_session = session_id or inherited_session
    if not workflow_id and not resolved_session:
        raise ValueError(
            "record_outcome requires at least one of workflow_id or session_id "
            "(an active @watch()/AgentSession session also satisfies this)"
        )

    span = _tracer.start_span(GenAIAttributes.SPAN_OUTCOME)
    if inherited_agent:
        span.set_attribute(GenAIAttributes.AGENT_ID, inherited_agent)
    if inherited_conv:
        span.set_attribute(GenAIAttributes.CONVERSATION_ID, inherited_conv)

    # The marker attribute the Cloud ROI ingest keys off (roi.is_outcome_event),
    # plus the stock OTel event.name so it can also ride the event path.
    span.set_attribute(GenAIAttributes.EVENT_NAME, GenAIAttributes.OUTCOME_EVENT_NAME)
    span.set_attribute(GenAIAttributes.OUTCOME_TYPE, outcome_type)
    span.set_attribute(GenAIAttributes.OUTCOME_SUCCESS, bool(success))
    if workflow_id:
        span.set_attribute(TjAttributes.WORKFLOW_ID, workflow_id)
    if resolved_session:
        span.set_attribute(TjAttributes.SESSION_ID, resolved_session)
    if value_usd is not None:
        span.set_attribute(GenAIAttributes.OUTCOME_VALUE_USD, float(value_usd))
    if attributes:
        for key, value in attributes.items():
            span.set_attribute(key, value)

    span.set_status(trace.Status(trace.StatusCode.OK))
    span.end()
