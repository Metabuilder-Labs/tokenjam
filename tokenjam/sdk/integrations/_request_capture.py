"""Shared full-request capture helpers for provider integrations (issue #209).

Each provider patch calls :func:`record_request_params` and
:func:`record_request_tools` on its LLM-call span with the call kwargs. They set
OTel ``gen_ai.request.*`` / ``tokenjam.request.tools`` attributes; the single
ingest gate (``strip_captured_content``) then decides whether the data is kept,
and ``extract_request_capture`` projects it into the structured NormalizedSpan
fields. Capturing here (vs. in each patch) keeps the kwarg→semconv mapping in
one place across providers, which use different kwarg names for the same param.
"""
from __future__ import annotations

import json
from typing import Any

from tokenjam.otel.semconv import GenAIAttributes, TjAttributes

# Map each semconv sampling-param attribute to the kwarg names providers use for
# it. First present alias wins (e.g. OpenAI's max_completion_tokens vs the older
# max_tokens; Gemini's max_output_tokens; Bedrock's camelCase maxTokens).
_PARAM_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (GenAIAttributes.REQUEST_TEMPERATURE,       ("temperature",)),
    (GenAIAttributes.REQUEST_TOP_P,             ("top_p", "topP")),
    (GenAIAttributes.REQUEST_TOP_K,             ("top_k", "topK")),
    (GenAIAttributes.REQUEST_MAX_TOKENS,
        ("max_tokens", "max_completion_tokens", "max_output_tokens", "maxTokens")),
    (GenAIAttributes.REQUEST_STOP_SEQUENCES,    ("stop", "stop_sequences", "stopSequences")),
    (GenAIAttributes.REQUEST_FREQUENCY_PENALTY, ("frequency_penalty",)),
    (GenAIAttributes.REQUEST_PRESENCE_PENALTY,  ("presence_penalty",)),
    (GenAIAttributes.REQUEST_SEED,              ("seed",)),
)


def _coerce_attr_value(value: Any) -> Any | None:
    """Coerce a value to something an OTel attribute accepts, else None.

    OTel attribute values must be a primitive (str/bool/int/float) or a
    homogeneous sequence of strings. ``stop`` may be a string or list[str].
    Anything else is skipped — full-request capture is best-effort.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int, float)):
        return value
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, str) for v in value):
        return list(value)
    return None


def record_request_params(span: Any, kwargs: dict[str, Any]) -> None:
    """Set ``gen_ai.request.*`` sampling-param attributes from call kwargs."""
    for attr, aliases in _PARAM_ALIASES:
        for alias in aliases:
            if kwargs.get(alias) is not None:
                coerced = _coerce_attr_value(kwargs[alias])
                if coerced is not None:
                    span.set_attribute(attr, coerced)
                break


def record_prompt_content(span: Any, messages: Any) -> None:
    """Set ``gen_ai.prompt.content`` (the request messages) on the span (#320).

    Serialized as ``json.dumps(messages)`` — the SAME shape every other capture
    path uses (``litellm._record_prompt_content``, ``sdk/agent.py``) so a replay
    harness / backfill reading ``gen_ai.prompt.content`` gets one consistent
    shape. Set UNCONDITIONALLY here; ``strip_captured_content`` drops it at
    ingest when the ``[capture] prompts`` toggle is off (same pattern litellm
    uses). ``messages`` is usually a list of message dicts but may be any
    JSON-shaped request body; ``None`` is a no-op.
    """
    if messages is None:
        return
    try:
        content = json.dumps(messages, default=str)
    except (TypeError, ValueError):
        content = str(messages)
    span.set_attribute(GenAIAttributes.PROMPT_CONTENT, content)


def record_completion_content(span: Any, text: Any) -> None:
    """Set ``gen_ai.completion.content`` (the assistant response text) (#320).

    Set UNCONDITIONALLY; ``strip_captured_content`` drops it at ingest when the
    ``[capture] completions`` toggle is off. ``text`` is the assistant text;
    ``None`` (e.g. a tool-only response, or a shape we couldn't extract) is a
    no-op so we never write an empty/placeholder completion.
    """
    if text is None:
        return
    span.set_attribute(
        GenAIAttributes.COMPLETION_CONTENT,
        text if isinstance(text, str) else str(text),
    )


def record_request_tools(span: Any, kwargs: dict[str, Any]) -> None:
    """Set the ``tokenjam.request.tools`` attribute (JSON) from call kwargs.

    Holds ``{"tools": [...], "tool_choice": ...}`` — serialized to a JSON string
    because the payload is nested (OTel attributes must be flat). Gated under
    the [capture] ``tool_inputs`` toggle at ingest.
    """
    tools = kwargs.get("tools")
    tool_choice = kwargs.get("tool_choice")
    if tools is None and tool_choice is None:
        return
    payload: dict[str, Any] = {}
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    try:
        span.set_attribute(TjAttributes.REQUEST_TOOLS, json.dumps(payload, default=str))
    except (TypeError, ValueError):
        # Non-serialisable tool spec — skip rather than break the call.
        pass


def record_full_request(span: Any, kwargs: dict[str, Any]) -> None:
    """Convenience: capture both sampling params and the tools payload."""
    record_request_params(span, kwargs)
    record_request_tools(span, kwargs)


# --- Response completion-text extraction, per provider response shape (#320).
# Defensive getattr/get access — a shape we don't recognise yields None (no
# completion recorded) rather than raising into the patched call.

def extract_anthropic_completion(response: Any) -> str | None:
    """Assistant text from an Anthropic Messages response (``.content`` blocks).

    ``content`` is a list of blocks; text blocks carry ``.text`` (or
    ``["text"]``). Tool-use blocks have no text. Concatenate the text blocks so
    a mixed text+tool response still yields the prose.
    """
    content = getattr(response, "content", None)
    if content is None and isinstance(response, dict):
        content = response.get("content")
    if not isinstance(content, (list, tuple)):
        return None
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts) if parts else None


def extract_openai_completion(response: Any) -> str | None:
    """Assistant text from an OpenAI chat response (``choices[0].message.content``)."""
    choices = getattr(response, "choices", None)
    if not choices:
        return None
    try:
        choice = choices[0]
    except (IndexError, TypeError):
        return None
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None) if message is not None else None
    if content is None:
        content = getattr(choice, "text", None)  # legacy completions shape
    if content is None:
        return None
    return content if isinstance(content, str) else str(content)


def extract_gemini_completion(response: Any) -> str | None:
    """Assistant text from a Gemini ``generate_content`` response (``.text``)."""
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text
    return None


# Gemini fields surfaced from generation_config (dict or GenerationConfig obj).
_GEMINI_GEN_CONFIG_FIELDS = (
    "temperature", "top_p", "top_k", "max_output_tokens", "stop_sequences",
)


def record_full_request_gemini(span: Any, kwargs: dict[str, Any]) -> None:
    """Capture full request for Gemini, whose sampling params live nested under
    the ``generation_config`` kwarg (a dict or a GenerationConfig object)."""
    flat = dict(kwargs)
    gen_config = kwargs.get("generation_config")
    if isinstance(gen_config, dict):
        flat.update(gen_config)
    elif gen_config is not None:
        for field_name in _GEMINI_GEN_CONFIG_FIELDS:
            value = getattr(gen_config, field_name, None)
            if value is not None:
                flat[field_name] = value
    record_full_request(span, flat)


def record_full_request_bedrock(span: Any, kwargs: dict[str, Any]) -> None:
    """Capture full request for Bedrock invoke_model, whose request payload is a
    JSON ``body`` kwarg (str/bytes/dict). The body schema is model-specific —
    Anthropic-on-Bedrock keys (max_tokens, temperature, top_p, top_k,
    stop_sequences, tools, tool_choice) match the aliases; other model schemas
    simply yield no params (best-effort, no behavior change)."""
    body = kwargs.get("body")
    if body is None:
        return
    if isinstance(body, (str, bytes)):
        try:
            body = json.loads(body)
        except (ValueError, TypeError):
            return
    if isinstance(body, dict):
        record_full_request(span, body)
