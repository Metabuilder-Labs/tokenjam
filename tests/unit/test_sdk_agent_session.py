"""Tests for `tokenjam.sdk.agent.AgentSession` and `watch()`'s cost-attribution
plumbing.

`AgentSession` is a first-class entry point (used directly, not only via
`@watch()`), so `tenant_id`/`feature` passed to it must reach BOTH its own
"invoke_agent" span AND the ambient `sdk.attribution` context that
auto-instrumented provider-patch spans (`patch_anthropic()`, `patch_openai()`,
...) read. Before this fix, only `@watch()`'s wrapper established the ambient
context (via its own separate `attribution(...)` block); a direct
`AgentSession(tenant_id=..., feature=...)` use tagged only its own span, so
the billable LLM call spans created inside it carried no tenant/feature.
"""
from __future__ import annotations

from tokenjam.otel.semconv import TjAttributes
from tokenjam.sdk.agent import AgentSession, watch
from tokenjam.sdk.attribution import stamp_span_attribution


class _FakeSpan:
    """Minimal OTel-span stand-in (mirrors test_sdk_attribution.py's)."""

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


class TestAgentSessionEstablishesAmbientAttribution:
    def test_direct_use_stamps_a_provider_patch_style_span_via_ambient_context(self):
        """The regression this guards: a span created inside a direct
        `AgentSession(...)` block, with NO tenant_id/feature of its own
        (exactly how a provider patch creates one), must still pick up the
        session's tenant/feature from the ambient context."""
        downstream_span = _FakeSpan()
        with AgentSession(agent_id="my-agent", tenant_id="acme-corp", feature="triage"):
            stamp_span_attribution(downstream_span)
        assert downstream_span.attributes[TjAttributes.TENANT_ID] == "acme-corp"
        assert downstream_span.attributes[TjAttributes.FEATURE] == "triage"

    def test_ambient_context_is_reset_after_the_session_exits(self):
        downstream_span = _FakeSpan()
        with AgentSession(agent_id="my-agent", tenant_id="acme-corp"):
            pass
        stamp_span_attribution(downstream_span)
        assert TjAttributes.TENANT_ID not in downstream_span.attributes

    def test_own_session_span_still_carries_tenant_and_feature(self, monkeypatch):
        """The pre-existing behaviour (stamping the session's OWN span) must
        survive this change, not just the new ambient-context plumbing."""
        import tokenjam.sdk.agent as agent_mod

        class _Tracer:
            def start_span(self, _name):
                return _FakeSpan()

        monkeypatch.setattr(agent_mod, "_tracer", _Tracer())

        with AgentSession(agent_id="my-agent", tenant_id="acme-corp", feature="triage") as session:
            span = session._span
        assert span.attributes.get(TjAttributes.TENANT_ID) == "acme-corp"
        assert span.attributes.get(TjAttributes.FEATURE) == "triage"

    def test_no_tenant_or_feature_leaves_ambient_context_untouched(self):
        downstream_span = _FakeSpan()
        with AgentSession(agent_id="my-agent"):
            stamp_span_attribution(downstream_span)
        assert TjAttributes.TENANT_ID not in downstream_span.attributes
        assert TjAttributes.FEATURE not in downstream_span.attributes

    def test_watch_decorator_establishes_the_same_ambient_context(self):
        """`@watch()` must keep working after its wrapper stopped opening a
        SEPARATE `attribution(...)` block of its own -- `AgentSession` now
        does that job."""
        downstream_span = _FakeSpan()
        seen = {}

        @watch("my-agent", tenant_id="acme-corp", feature="triage")
        def run():
            stamp_span_attribution(downstream_span)
            seen["ran"] = True

        run()
        assert seen["ran"] is True
        assert downstream_span.attributes[TjAttributes.TENANT_ID] == "acme-corp"
        assert downstream_span.attributes[TjAttributes.FEATURE] == "triage"
