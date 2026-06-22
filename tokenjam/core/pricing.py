from __future__ import annotations
import logging
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]


log = logging.getLogger(__name__)

PRICING_FILE = Path(__file__).parent.parent / "pricing" / "models.toml"

# Optional user-maintained override file. Lets users add or correct rates
# without editing the packaged models.toml (which a pip upgrade overwrites).
# Resolution order, highest priority first:
#   1. The path in the TJ_PRICING_FILE env var, if set.
#   2. ~/.config/tj/pricing.toml, if it exists.
# Entries in the override are merged over the packaged table per
# provider/model, so it can both override existing rates and add new models.
USER_PRICING_ENV = "TJ_PRICING_FILE"

# Default rate used when a model is not in the pricing table.
# 0.50 per MTok input, 2.00 per MTok output — conservative mid-range estimate.
DEFAULT_INPUT_PER_MTOK = 0.50
DEFAULT_OUTPUT_PER_MTOK = 2.00


@dataclass(frozen=True)
class ModelRates:
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float = 0.0
    cache_write_per_mtok: float = 0.0


def _parse_pricing_file(path: Path) -> dict[str, dict[str, ModelRates]]:
    """Parse a pricing TOML file into { provider: { model: ModelRates } }."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    result: dict[str, dict[str, ModelRates]] = {}
    for provider, models in raw.items():
        result[provider] = {}
        for model_name, rates in models.items():
            result[provider][model_name] = ModelRates(
                input_per_mtok=rates.get("input_per_mtok", DEFAULT_INPUT_PER_MTOK),
                output_per_mtok=rates.get("output_per_mtok", DEFAULT_OUTPUT_PER_MTOK),
                cache_read_per_mtok=rates.get("cache_read_per_mtok", 0.0),
                cache_write_per_mtok=rates.get("cache_write_per_mtok", 0.0),
            )
    return result


def _user_pricing_file() -> Path | None:
    """Resolve the optional user override file, or None if not configured.

    TJ_PRICING_FILE (if set) wins and is returned even when the file is
    missing, so a typo'd path surfaces as a warning rather than being
    silently ignored. Otherwise the default ~/.config/tj/pricing.toml is
    returned only when it exists. Path.home() is resolved here (not at
    import) so the lookup honors the current environment.
    """
    override = os.environ.get(USER_PRICING_ENV)
    if override:
        return Path(override).expanduser()
    default = Path.home() / ".config" / "tj" / "pricing.toml"
    return default if default.exists() else None


@lru_cache(maxsize=1)
def load_pricing_table() -> dict[str, dict[str, ModelRates]]:
    """
    Load the packaged pricing/models.toml, then merge an optional user
    override file over it, and return a nested dict:
      { provider: { model_name: ModelRates } }

    The override (see USER_PRICING_ENV / ~/.config/tj/pricing.toml) is
    applied per provider/model, so it can correct a packaged rate or add a
    model the package doesn't ship. Cached after first load — restart the
    process (or call load_pricing_table.cache_clear()) to pick up changes.
    """
    result = _parse_pricing_file(PRICING_FILE)

    user_file = _user_pricing_file()
    if user_file is not None:
        try:
            overrides = _parse_pricing_file(user_file)
        except FileNotFoundError:
            log.warning(
                "Pricing override file %s=%s not found; using packaged rates only.",
                USER_PRICING_ENV,
                user_file,
            )
            overrides = {}
        except (OSError, tomllib.TOMLDecodeError) as exc:
            log.warning(
                "Could not read pricing override file %s (%s); "
                "using packaged rates only.",
                user_file,
                exc,
            )
            overrides = {}
        for provider, models in overrides.items():
            result.setdefault(provider, {}).update(models)

    return result


def get_rates(provider: str, model: str) -> ModelRates | None:
    """
    Return ModelRates for the given provider/model, or None if not found.

    Tries an exact match first, then falls back to stripping a trailing
    YYYYMMDD release-date suffix (Anthropic/OpenAI both ship dated variants
    like `claude-haiku-4-5-20251001`). This keeps pricing/models.toml short
    while still pricing the dated names that flow through Claude Code logs.
    """
    table = load_pricing_table()
    rates = table.get(provider, {}).get(model)
    if rates is not None:
        return rates
    # Strip trailing 8-digit date suffix
    import re as _re
    m = _re.match(r"^(.*)-(\d{8})$", model)
    if m:
        base = m.group(1)
        rates = table.get(provider, {}).get(base)
        if rates is not None:
            return rates
    return None


def provider_for_model(model: str | None) -> str | None:
    """Best-effort provider inference from a bare model name.

    Used when an upstream integration can't tell us the provider directly —
    e.g. LiteLLM >= 1.75 returns ``custom_llm_provider = None`` and the caller
    passed a bare model name like ``claude-haiku-4-5`` (no ``anthropic/``
    prefix). Returns the canonical provider/billing_account identifier
    (``anthropic`` / ``openai`` / ``google``), or ``None`` when the model can't
    be confidently attributed.

    Callers must NOT invent a provider when this returns None — record
    ``"unknown"`` instead, so pricing and billing_account stay honest (#194).

    Open-weight families (llama / qwen / mistral / gemma / deepseek / ...) are
    intentionally left unattributed -> ``"unknown"``. Mapping them to a local
    billing_account would set ``pricing_mode = local``, asserting "no marginal
    cost" — but the same weights run on PAID hosts (Groq / Together / Bedrock),
    so that would over-claim "free". When unsure we hedge ("unknown" -> dollars
    with a "may overstate" qualifier) rather than assert free; a genuinely-local
    user can pin the rate via the user pricing override.

    Note: a parallel, source-specific copy of this knowledge lives in the
    Langfuse adapter (``_model_to_provider``) and the Claude Code backfill
    parser (``_provider_for_model``); those carry adapter-specific defaults and
    are intentionally left in place.
    """
    if not model:
        return None
    m = model.lower()
    # Defensive: strip any leftover "provider/" prefix the caller didn't.
    if "/" in m:
        m = m.rsplit("/", 1)[1]
    if "claude" in m:
        return "anthropic"
    if m.startswith(("gpt-", "gpt", "o1", "o3", "o4", "chatgpt-")):
        return "openai"
    if "gemini" in m:
        return "google"
    return None
