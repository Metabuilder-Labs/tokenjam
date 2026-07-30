"""Ambient cost-attribution context: tenant/customer, feature, and prompt
template identity for LLM-call spans.

Why this exists: `record_llm_call()` (manual instrumentation) can accept these
as direct keyword arguments, but the provider-patch integrations
(`patch_anthropic()`, `patch_openai()`, ...) create spans by monkey-patching a
THIRD-PARTY client method — there is no call-site hook to pass tenant_id /
feature / prompt_template_id through, because the method signature belongs to
the anthropic/openai/etc SDK, not tokenjam's. A `contextvars`-based context
manager lets a caller declare these dimensions once around a block of code (a
request handler, a per-tenant background task) and have every span created
within it — across every provider/framework integration, with no per-call
plumbing — stamped automatically.

`contextvars.ContextVar` is used (not a plain global) so this is correct under
`asyncio` (each task gets its own copy) and thread pools (each thread inherits
the context active when it was spawned, per the stdlib's context-copy
semantics), matching how OTel's own span context propagation works.
"""
from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Iterator

from tokenjam.otel.semconv import TjAttributes

_tenant_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "tj_attribution_tenant_id", default=None
)
_feature: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "tj_attribution_feature", default=None
)
_prompt_template_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "tj_attribution_prompt_template_id", default=None
)
_prompt_template_version: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "tj_attribution_prompt_template_version", default=None
)


@contextmanager
def attribution(
    *,
    tenant_id: str | None = None,
    feature: str | None = None,
    prompt_template_id: str | None = None,
    prompt_template_version: str | None = None,
) -> Iterator[None]:
    """Attach cost-attribution dimensions to every span created in this block.

    Usage::

        with tokenjam.sdk.attribution.attribution(tenant_id="acme-corp", feature="support-triage"):
            response = anthropic_client.messages.create(...)  # tagged automatically

    Nests: an inner `attribution(...)` call only overrides the dimensions it
    passes explicitly — an omitted kwarg falls through to whatever an outer
    `attribution()` block already set, not to None. Restores the prior value
    (not None) on exit, so nested blocks compose correctly.
    """
    resets: list[tuple[contextvars.ContextVar, contextvars.Token]] = []
    for var, value in (
        (_tenant_id, tenant_id),
        (_feature, feature),
        (_prompt_template_id, prompt_template_id),
        (_prompt_template_version, prompt_template_version),
    ):
        if value is not None:
            resets.append((var, var.set(value)))
    try:
        yield
    finally:
        for var, token in reversed(resets):
            var.reset(token)


def stamp_span_attribution(
    span,
    *,
    tenant_id: str | None = None,
    feature: str | None = None,
    prompt_template_id: str | None = None,
    prompt_template_version: str | None = None,
) -> None:
    """Stamp attribution attributes onto an in-flight OTel span.

    An explicit kwarg (from a manual `record_llm_call()` call, or a caller
    that knows its own tenant right at the LLM-call site) wins; otherwise the
    ambient `attribution()` context value is used. Call sites that have
    neither leave the attribute unset — never stamp an empty string, so a
    later `NULL` check on the span's attributes table stays meaningful.
    """
    values = {
        TjAttributes.TENANT_ID: tenant_id if tenant_id is not None else _tenant_id.get(),
        TjAttributes.FEATURE: feature if feature is not None else _feature.get(),
        TjAttributes.PROMPT_TEMPLATE_ID: (
            prompt_template_id if prompt_template_id is not None
            else _prompt_template_id.get()
        ),
        TjAttributes.PROMPT_TEMPLATE_VERSION: (
            prompt_template_version if prompt_template_version is not None
            else _prompt_template_version.get()
        ),
    }
    for key, value in values.items():
        if value is not None:
            span.set_attribute(key, value)
