"""
AWS Bedrock provider integration.

Wraps boto3 client invoke_model and invoke_agent calls. The Bedrock response
body is JSON — parsed to extract token counts.
"""
from __future__ import annotations

import functools
import json
import logging

from opentelemetry import trace

from tokenjam.otel.semconv import GenAIAttributes
from tokenjam.sdk.integrations._request_capture import (
    extract_anthropic_completion,
    record_completion_content,
    record_full_request_bedrock,
    record_prompt_content,
)

logger = logging.getLogger(__name__)


def _bedrock_body_dict(body: object) -> dict | None:
    """Parse a Bedrock request/response ``body`` (str/bytes/dict) to a dict."""
    if isinstance(body, (str, bytes)):
        try:
            body = json.loads(body)
        except (ValueError, TypeError):
            return None
    return body if isinstance(body, dict) else None


def _bedrock_request_messages(kwargs: dict) -> object | None:
    """The request prompt from a Bedrock ``invoke_model`` body. Anthropic-on-
    Bedrock uses ``messages``; other model schemas use ``prompt`` / ``inputText``."""
    body = _bedrock_body_dict(kwargs.get("body"))
    if body is None:
        return None
    for key in ("messages", "prompt", "inputText"):
        if body.get(key) is not None:
            return body[key]
    return None


def _bedrock_completion_text(body_dict: dict) -> str | None:
    """Assistant text from a parsed Bedrock response body. Anthropic-on-Bedrock
    carries ``content`` blocks; other schemas use scalar text keys."""
    text = extract_anthropic_completion(body_dict)  # Anthropic content blocks
    if text:
        return text
    for key in ("completion", "outputText", "generation"):
        value = body_dict.get(key)
        if isinstance(value, str):
            return value
    results = body_dict.get("results")
    if isinstance(results, list) and results and isinstance(results[0], dict):
        out = results[0].get("outputText")
        if isinstance(out, str):
            return out
    return None


class BedrockIntegration:
    name = "aws.bedrock"
    installed = False

    def __init__(self) -> None:
        self._original_invoke_model = None
        self._tracer = None

    def install(self, tracer) -> None:
        if self.installed:
            return
        self._tracer = tracer
        try:
            import botocore.client
        except ImportError:
            logger.warning("boto3/botocore not installed — skipping patch")
            return

        original_api_call = botocore.client.ClientCreator._create_api_method
        integration = self

        def patched_create_api_method(client_creator, py_operation_name, operation_name, service_model):
            method = original_api_call(client_creator, py_operation_name, operation_name, service_model)
            if py_operation_name not in ("invoke_model", "invoke_agent"):
                return method

            @functools.wraps(method)
            def wrapped(self_client, *args, **kwargs):
                span = integration._tracer.start_span(GenAIAttributes.SPAN_LLM_CALL)
                span.set_attribute(GenAIAttributes.PROVIDER_NAME, "aws.bedrock")
                model_id = kwargs.get("modelId", "unknown")
                span.set_attribute(GenAIAttributes.REQUEST_MODEL, model_id)
                record_full_request_bedrock(span, kwargs)
                # Prompt content (#320). Stripped at ingest unless [capture]
                # prompts is on. Completion is set in _extract_bedrock_usage,
                # which already parses the response body (avoids re-reading it).
                record_prompt_content(span, _bedrock_request_messages(kwargs))
                try:
                    response = method(self_client, *args, **kwargs)
                    _extract_bedrock_usage(response, span)
                    span.set_status(trace.Status(trace.StatusCode.OK))
                    return response
                except Exception as exc:
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
                    raise
                finally:
                    span.end()

            return wrapped

        botocore.client.ClientCreator._create_api_method = patched_create_api_method
        self.installed = True
        logger.debug("Bedrock integration installed")

    def uninstall(self) -> None:
        self.installed = False


def _extract_bedrock_usage(response: dict, span) -> None:
    """Extract token counts from a Bedrock response."""
    body = response.get("body")
    if body is None:
        return
    try:
        if hasattr(body, "read"):
            raw = body.read()
            body_dict = json.loads(raw)
            # Reset stream for caller
            import io
            response["body"] = io.BytesIO(raw)
        else:
            body_dict = json.loads(body) if isinstance(body, (str, bytes)) else body
    except (json.JSONDecodeError, TypeError):
        return

    usage = body_dict.get("usage", {})
    if usage:
        if "input_tokens" in usage:
            span.set_attribute(GenAIAttributes.INPUT_TOKENS, usage["input_tokens"])
        if "output_tokens" in usage:
            span.set_attribute(GenAIAttributes.OUTPUT_TOKENS, usage["output_tokens"])

    # Completion content (#320), from the already-parsed body. Stripped at
    # ingest unless [capture] completions is on.
    record_completion_content(span, _bedrock_completion_text(body_dict))


def patch_bedrock() -> None:
    """Convenience function. Instantiates and installs BedrockIntegration."""
    from tokenjam.sdk.bootstrap import ensure_initialised
    ensure_initialised()
    integration = BedrockIntegration()
    integration.install(trace.get_tracer("tokenjam.sdk"))
