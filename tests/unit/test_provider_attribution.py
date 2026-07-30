"""convert_otel_span SDK cost-attribution extraction (the in-process SDK
path), plus the `_build_tj_resource()` service.version caller-wins fix.

Mirrors `test_provider_run_id.py`'s pattern: build a real ended OTel
ReadableSpan carrying the given resource/span attrs and assert what
`convert_otel_span` extracts. Covers the in-process Python SDK path
(TjSpanExporter -> convert_otel_span), alongside the HTTP/OTLP
(otlp_parsing) path covered in test_ingest_otlp.py.
"""
from __future__ import annotations

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider

from tokenjam.otel.provider import _build_tj_resource, convert_otel_span
from tokenjam.otel.semconv import GenAIAttributes, ResourceAttributes, TjAttributes


def _span_with_attrs(resource_attrs: dict, span_attrs: dict | None = None) -> object:
    """Build a real ended OTel ReadableSpan carrying the given resource + span attrs."""
    provider = TracerProvider(resource=Resource.create(resource_attrs))
    tracer = provider.get_tracer("test")
    span = tracer.start_span(GenAIAttributes.SPAN_LLM_CALL, attributes=span_attrs or {})
    span.end()
    return span


def test_convert_otel_span_extracts_deployment_resource_attrs():
    span = _span_with_attrs({
        ResourceAttributes.DEPLOYMENT_ENVIRONMENT_NAME: "production",
        ResourceAttributes.SERVICE_VERSION: "2.3.1",
        ResourceAttributes.VCS_REF_HEAD_REVISION: "abc123",
    })
    ns = convert_otel_span(span)
    assert ns.environment == "production"
    assert ns.service_version == "2.3.1"
    assert ns.commit_sha == "abc123"


def test_convert_otel_span_vcs_commit_sha_deprecated_fallback():
    span = _span_with_attrs({ResourceAttributes.VCS_REPOSITORY_REF_REVISION: "deadbeef"})
    ns = convert_otel_span(span)
    assert ns.commit_sha == "deadbeef"


def test_convert_otel_span_extracts_span_level_attribution():
    """tenant_id/feature/prompt template identity are SPAN attributes (per
    call), not resource attributes — set by record_llm_call() / the provider
    patches via sdk.attribution.stamp_span_attribution()."""
    span = _span_with_attrs({}, {
        TjAttributes.TENANT_ID: "acme-corp",
        TjAttributes.FEATURE: "support-triage",
        TjAttributes.PROMPT_TEMPLATE_ID: "tmpl-1",
        TjAttributes.PROMPT_TEMPLATE_VERSION: "2",
    })
    ns = convert_otel_span(span)
    assert ns.tenant_id == "acme-corp"
    assert ns.feature == "support-triage"
    assert ns.prompt_template_id == "tmpl-1"
    assert ns.prompt_template_version == "2"


def test_convert_otel_span_attribution_absent_is_none():
    span = _span_with_attrs({})
    ns = convert_otel_span(span)
    assert ns.tenant_id is None
    assert ns.feature is None
    assert ns.environment is None
    assert ns.service_version is None
    assert ns.commit_sha is None
    assert ns.prompt_template_id is None
    assert ns.prompt_template_version is None


class TestBuildTjResource:
    """`_build_tj_resource()` — the service.version caller-wins fix. Does NOT
    call `trace.set_tracer_provider()`, so it's safe to exercise directly in
    the shared test process (Critical Rule 11: the global provider is
    set-once)."""

    def test_defaults_to_tokenjam_package_version(self):
        import tokenjam

        resource = _build_tj_resource()
        assert resource.attributes[ResourceAttributes.SERVICE_VERSION] == tokenjam.__version__
        assert resource.attributes["service.name"] == "tokenjam"

    def test_caller_declared_service_version_via_env_wins(self, monkeypatch):
        """A caller's own OTEL_RESOURCE_ATTRIBUTES-declared service.version
        must NOT be clobbered by tokenjam's own package version — that value
        is the SDK cost-attribution 'service version' dimension."""
        monkeypatch.setenv(
            "OTEL_RESOURCE_ATTRIBUTES",
            "service.version=9.9.9,deployment.environment.name=production",
        )
        resource = _build_tj_resource()
        assert resource.attributes[ResourceAttributes.SERVICE_VERSION] == "9.9.9"
        assert resource.attributes[ResourceAttributes.DEPLOYMENT_ENVIRONMENT_NAME] == "production"
        # service.name stays tokenjam's own — that identity isn't overridden.
        assert resource.attributes["service.name"] == "tokenjam"
