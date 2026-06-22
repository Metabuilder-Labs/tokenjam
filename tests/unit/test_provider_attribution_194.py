"""Regression tests for #194 — LiteLLM provider attribution.

The LiteLLM integration used to record provider="litellm" when
custom_llm_provider was absent and the model was bare (no "provider/" prefix).
That bogus provider missed pricing (~42% cost undercount) and left
billing_account NULL, which suppressed plan-tier dollar framing. These tests
cover the resolver and the downstream pricing/billing/plan-tier chain.
"""
from __future__ import annotations

import pytest

from tokenjam.core.config import ProviderBudget, TjConfig
from tokenjam.core.cost import calculate_cost
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest import IngestPipeline
from tokenjam.core.pricing import get_rates, provider_for_model
from tokenjam.otel.provider import _provider_to_billing_account
from tokenjam.sdk.integrations.litellm import _parse_provider
from tests.factories import make_llm_span


# --- provider_for_model ------------------------------------------------------

@pytest.mark.parametrize("model,expected", [
    ("claude-haiku-4-5", "anthropic"),
    ("claude-haiku-4-5-20251001", "anthropic"),
    ("gpt-4o-mini", "openai"),
    ("o3-mini", "openai"),
    ("chatgpt-4o-latest", "openai"),
    ("gemini-2.5-pro", "google"),
    ("llama-3.1-70b", "local.ollama"),
    ("mistral-large", "local.ollama"),
    ("anthropic/claude-haiku-4-5", "anthropic"),   # tolerates leftover prefix
])
def test_provider_for_model_resolves(model, expected):
    assert provider_for_model(model) == expected


@pytest.mark.parametrize("model", ["", None, "some-internal-model-x", "weird"])
def test_provider_for_model_unresolvable_returns_none(model):
    assert provider_for_model(model) is None


# --- _parse_provider (the bug site) ------------------------------------------

class _Resp:
    def __init__(self, hidden):
        self._hidden_params = hidden


def test_parse_provider_bare_model_none_hidden_resolves_anthropic():
    """custom_llm_provider=None + bare claude-* -> anthropic (the Aider case)."""
    resp = _Resp({"custom_llm_provider": None})
    assert _parse_provider("claude-haiku-4-5", resp) == "anthropic"


def test_parse_provider_never_returns_litellm():
    """Unresolvable bare model -> 'unknown', never the bogus 'litellm'."""
    resp = _Resp({})
    result = _parse_provider("some-internal-model-x", resp)
    assert result == "unknown"
    assert result != "litellm"


def test_parse_provider_hidden_param_wins():
    resp = _Resp({"custom_llm_provider": "openai"})
    assert _parse_provider("claude-haiku-4-5", resp) == "openai"


def test_parse_provider_prefix_still_works():
    assert _parse_provider("anthropic/claude-haiku-4-5", _Resp({})) == "anthropic"


# --- downstream: pricing + billing_account + plan_tier -----------------------

def test_resolved_provider_gets_correct_haiku_pricing_not_default():
    """anthropic/claude-haiku-4-5 prices at Haiku rates; 'litellm' would not."""
    rates = get_rates("anthropic", "claude-haiku-4-5")
    assert rates is not None and rates.input_per_mtok == 0.80

    # 1M input tokens at the real Haiku rate = $0.80.
    correct = calculate_cost("anthropic", "claude-haiku-4-5", 1_000_000, 0)
    assert correct == pytest.approx(0.80)

    # The old bogus provider had no pricing table -> default $0.50 fallback.
    bogus = calculate_cost("litellm", "claude-haiku-4-5", 1_000_000, 0)
    assert bogus == pytest.approx(0.50)
    assert correct > bogus  # the ~42%-undercount the fix removes


def test_billing_account_derives_for_resolved_provider():
    assert _provider_to_billing_account("anthropic") == "anthropic"
    # 'litellm' and 'unknown' must NOT masquerade as a real billing account.
    assert _provider_to_billing_account("litellm") is None
    assert _provider_to_billing_account("unknown") is None


def test_plan_tier_propagates_from_declared_config_via_pipeline():
    """A span resolved to anthropic gets plan_tier from declared [budget.anthropic]."""
    db = InMemoryBackend()
    config = TjConfig(version="1", budgets={"anthropic": ProviderBudget(plan="api")})
    pipeline = IngestPipeline(db=db, config=config)

    span = make_llm_span(
        model="claude-haiku-4-5",
        provider="anthropic",
        billing_account="anthropic",
        conversation_id="conv-194",
    )
    pipeline.process(span)

    row = db.conn.execute(
        "SELECT plan_tier FROM sessions WHERE conversation_id = $1", ["conv-194"]
    ).fetchone()
    assert row is not None and row[0] == "api"


def test_plan_tier_unknown_when_provider_unresolved():
    """An 'unknown' provider yields NULL billing_account -> plan_tier stays unknown."""
    db = InMemoryBackend()
    config = TjConfig(version="1", budgets={"anthropic": ProviderBudget(plan="api")})
    pipeline = IngestPipeline(db=db, config=config)

    span = make_llm_span(
        model="some-internal-model-x",
        provider="unknown",
        billing_account=None,
        conversation_id="conv-194-unknown",
    )
    pipeline.process(span)

    row = db.conn.execute(
        "SELECT plan_tier FROM sessions WHERE conversation_id = $1",
        ["conv-194-unknown"],
    ).fetchone()
    assert row is not None and row[0] == "unknown"
