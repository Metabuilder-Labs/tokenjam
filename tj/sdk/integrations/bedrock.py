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

from tj.otel.semconv import GenAIAttributes

logger = logging.getLogger(__name__)


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


def patch_bedrock() -> None:
    """Convenience function. Instantiates and installs BedrockIntegration."""
    from tj.sdk.bootstrap import ensure_initialised
    ensure_initialised()
    integration = BedrockIntegration()
    integration.install(trace.get_tracer("tj.sdk"))
