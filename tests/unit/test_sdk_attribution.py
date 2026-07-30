"""Tests for `tokenjam.sdk.attribution` — the ambient cost-attribution
context (tenant/feature/prompt template identity) that lets auto-instrumented
provider-patch spans (patch_anthropic(), patch_openai(), ...) pick up
tenant_id/feature/prompt_template_id/prompt_template_version with no per-call
kwarg, since those patches wrap a THIRD-PARTY client method with no tj-owned
signature to extend.
"""
from __future__ import annotations

import types

import pytest

from tokenjam.otel.semconv import TjAttributes
from tokenjam.sdk.attribution import attribution, stamp_span_attribution


class _FakeSpan:
    """Minimal OTel-span stand-in recording set_attribute (mirrors the pattern
    in test_provider_content_capture.py)."""

    def __init__(self) -> None:
        self.attributes: dict = {}

    def set_attribute(self, key, value) -> None:
        self.attributes[key] = value

    def set_status(self, *_a, **_k) -> None:
        pass

    def end(self, *_a, **_k) -> None:
        pass

    def is_recording(self) -> bool:
        return True


def _recording_tracer():
    spans: list[_FakeSpan] = []

    class _Tracer:
        def start_span(self, _name):
            span = _FakeSpan()
            spans.append(span)
            return span

    return _Tracer(), spans


class TestStampSpanAttribution:
    def test_stamps_nothing_when_no_context_and_no_kwargs(self):
        span = _FakeSpan()
        stamp_span_attribution(span)
        assert span.attributes == {}

    def test_explicit_kwargs_stamp_directly(self):
        span = _FakeSpan()
        stamp_span_attribution(
            span, tenant_id="acme-corp", feature="triage",
            prompt_template_id="tmpl-1", prompt_template_version="2",
        )
        assert span.attributes[TjAttributes.TENANT_ID] == "acme-corp"
        assert span.attributes[TjAttributes.FEATURE] == "triage"
        assert span.attributes[TjAttributes.PROMPT_TEMPLATE_ID] == "tmpl-1"
        assert span.attributes[TjAttributes.PROMPT_TEMPLATE_VERSION] == "2"

    def test_ambient_context_stamps_when_no_explicit_kwargs(self):
        span = _FakeSpan()
        with attribution(tenant_id="ambient-tenant", feature="ambient-feature"):
            stamp_span_attribution(span)
        assert span.attributes[TjAttributes.TENANT_ID] == "ambient-tenant"
        assert span.attributes[TjAttributes.FEATURE] == "ambient-feature"

    def test_explicit_kwarg_wins_over_ambient_context(self):
        span = _FakeSpan()
        with attribution(tenant_id="ambient-tenant"):
            stamp_span_attribution(span, tenant_id="explicit-tenant")
        assert span.attributes[TjAttributes.TENANT_ID] == "explicit-tenant"

    def test_context_reset_after_block_exits(self):
        span_inside, span_after = _FakeSpan(), _FakeSpan()
        with attribution(tenant_id="acme-corp"):
            stamp_span_attribution(span_inside)
        stamp_span_attribution(span_after)
        assert span_inside.attributes[TjAttributes.TENANT_ID] == "acme-corp"
        assert TjAttributes.TENANT_ID not in span_after.attributes


class TestAttributionNesting:
    def test_inner_block_overrides_only_what_it_passes(self):
        """An inner attribution() that omits a kwarg falls through to the
        outer block's value, not to None."""
        span = _FakeSpan()
        with attribution(tenant_id="outer-tenant", feature="outer-feature"):
            with attribution(feature="inner-feature"):
                stamp_span_attribution(span)
        assert span.attributes[TjAttributes.TENANT_ID] == "outer-tenant"
        assert span.attributes[TjAttributes.FEATURE] == "inner-feature"

    def test_outer_context_restored_after_inner_block_exits(self):
        span = _FakeSpan()
        with attribution(feature="outer-feature"):
            with attribution(feature="inner-feature"):
                pass
            stamp_span_attribution(span)
        assert span.attributes[TjAttributes.FEATURE] == "outer-feature"


# ===========================================================================
# Provider-patch integration — ambient attribution() reaches the real span
# creation code path in the anthropic/openai monkey-patches, not just the
# helper in isolation.
# ===========================================================================

class TestProviderPatchStamping:
    def test_anthropic_patch_stamps_ambient_attribution(self):
        pytest.importorskip("anthropic")
        from anthropic.resources import Messages

        from tokenjam.sdk.integrations.anthropic import AnthropicIntegration

        tracer, spans = _recording_tracer()
        integ = AnthropicIntegration()
        integ.install(tracer)
        try:
            fake_resp = types.SimpleNamespace(
                usage=types.SimpleNamespace(input_tokens=10, output_tokens=5),
                content=[types.SimpleNamespace(type="text", text="4")],
            )
            integ._original_create = lambda _self, *a, **kw: fake_resp
            with attribution(tenant_id="acme-corp", feature="support-triage"):
                Messages.create(
                    "self", model="claude-haiku-4-5",
                    messages=[{"role": "user", "content": "hi"}],
                )
            attrs = spans[0].attributes
            assert attrs[TjAttributes.TENANT_ID] == "acme-corp"
            assert attrs[TjAttributes.FEATURE] == "support-triage"
        finally:
            integ.uninstall()

    def test_anthropic_patch_stamps_nothing_outside_attribution_context(self):
        pytest.importorskip("anthropic")
        from anthropic.resources import Messages

        from tokenjam.sdk.integrations.anthropic import AnthropicIntegration

        tracer, spans = _recording_tracer()
        integ = AnthropicIntegration()
        integ.install(tracer)
        try:
            fake_resp = types.SimpleNamespace(
                usage=types.SimpleNamespace(input_tokens=10, output_tokens=5),
                content=[types.SimpleNamespace(type="text", text="4")],
            )
            integ._original_create = lambda _self, *a, **kw: fake_resp
            Messages.create(
                "self", model="claude-haiku-4-5",
                messages=[{"role": "user", "content": "hi"}],
            )
            attrs = spans[0].attributes
            assert TjAttributes.TENANT_ID not in attrs
        finally:
            integ.uninstall()

    def test_langchain_tool_run_stamps_ambient_attribution(self):
        """Regression guard: both LangChain TOOL wrappers (`BaseTool.run`/
        `.arun`) used to omit attribution stamping entirely, unlike the LLM
        wrappers a few lines above them in the same integration -- a tool
        call span carried no tenant/feature even inside an `attribution()`
        block."""
        pytest.importorskip("langchain_core")
        from langchain_core.tools import BaseTool

        from tokenjam.sdk.integrations.langchain import LangChainIntegration

        tracer, spans = _recording_tracer()
        integ = LangChainIntegration()
        integ.install(tracer)
        try:
            integ._original_tool_run = lambda _self, *a, **kw: "tool result"
            fake_tool = types.SimpleNamespace(name="my-tool")
            with attribution(tenant_id="acme-corp", feature="support-triage"):
                BaseTool.run(fake_tool)
            attrs = spans[0].attributes
            assert attrs[TjAttributes.TENANT_ID] == "acme-corp"
            assert attrs[TjAttributes.FEATURE] == "support-triage"
        finally:
            integ.uninstall()

    async def test_langchain_tool_arun_stamps_ambient_attribution(self):
        pytest.importorskip("langchain_core")
        from langchain_core.tools import BaseTool

        from tokenjam.sdk.integrations.langchain import LangChainIntegration

        tracer, spans = _recording_tracer()
        integ = LangChainIntegration()
        integ.install(tracer)
        try:
            if integ._original_tool_arun is None:
                pytest.skip("this langchain_core version has no BaseTool.arun")

            async def _fake_arun(_self, *a, **kw):
                return "tool result"

            integ._original_tool_arun = _fake_arun
            fake_tool = types.SimpleNamespace(name="my-tool")
            with attribution(tenant_id="acme-corp", feature="support-triage"):
                await BaseTool.arun(fake_tool)
            attrs = spans[0].attributes
            assert attrs[TjAttributes.TENANT_ID] == "acme-corp"
            assert attrs[TjAttributes.FEATURE] == "support-triage"
        finally:
            integ.uninstall()

    def test_openai_patch_stamps_ambient_attribution(self):
        pytest.importorskip("openai")
        from openai.resources.chat.completions import Completions

        from tokenjam.sdk.integrations.openai import OpenAIIntegration

        tracer, spans = _recording_tracer()
        integ = OpenAIIntegration()
        integ.install(tracer)
        try:
            fake_resp = types.SimpleNamespace(
                usage=types.SimpleNamespace(prompt_tokens=3, completion_tokens=1),
                choices=[types.SimpleNamespace(
                    message=types.SimpleNamespace(content="pong"))],
            )
            integ._original_create = lambda _self, *a, **kw: fake_resp
            with attribution(prompt_template_id="tmpl-1", prompt_template_version="2"):
                Completions.create(
                    "self", model="gpt-4o-mini", messages=[{"role": "user", "content": "ping"}],
                )
            attrs = spans[0].attributes
            assert attrs[TjAttributes.PROMPT_TEMPLATE_ID] == "tmpl-1"
            assert attrs[TjAttributes.PROMPT_TEMPLATE_VERSION] == "2"
        finally:
            integ.uninstall()


# ===========================================================================
# LiteLLM out-of-process callback path (sdk/client.py) — metadata convention.
# ===========================================================================

class TestLiteLLMClientMetadata:
    def test_build_litellm_span_reads_tj_attribution_metadata(self):
        import datetime as dt

        from tokenjam.sdk.client import _build_litellm_span

        now = dt.datetime.now(dt.timezone.utc)
        kwargs = {
            "model": "gpt-4o",
            "metadata": {
                "tj_tenant_id": "acme-corp",
                "tj_feature": "support-triage",
                "tj_prompt_template_id": "tmpl-1",
                "tj_prompt_template_version": "2",
            },
        }
        response = types.SimpleNamespace(
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )
        span = _build_litellm_span(
            kwargs=kwargs, response_obj=response, start_time=now, end_time=now, success=True,
        )
        attrs = {a["key"]: a["value"] for a in span["attributes"]}
        assert attrs[TjAttributes.TENANT_ID] == {"stringValue": "acme-corp"}
        assert attrs[TjAttributes.FEATURE] == {"stringValue": "support-triage"}
        assert attrs[TjAttributes.PROMPT_TEMPLATE_ID] == {"stringValue": "tmpl-1"}
        assert attrs[TjAttributes.PROMPT_TEMPLATE_VERSION] == {"stringValue": "2"}

    def test_build_litellm_span_without_metadata_stamps_nothing(self):
        import datetime as dt

        from tokenjam.sdk.client import _build_litellm_span

        now = dt.datetime.now(dt.timezone.utc)
        response = types.SimpleNamespace(usage={"prompt_tokens": 10, "completion_tokens": 5})
        span = _build_litellm_span(
            kwargs={"model": "gpt-4o"}, response_obj=response,
            start_time=now, end_time=now, success=True,
        )
        keys = {a["key"] for a in span["attributes"]}
        assert TjAttributes.TENANT_ID not in keys
