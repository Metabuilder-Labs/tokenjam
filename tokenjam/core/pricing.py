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

# Optional user-maintained pricing override file. Lets users add or correct
# rates without editing the packaged models.toml (which a pip upgrade
# overwrites). Resolution order, highest priority first:
#   1. The path in the TJ_PRICING_FILE env var, if set.
#   2. ~/.config/tj/pricing.toml, if it exists.
# Entries in the override are merged over the packaged table; see
# _build_pricing() for the full source/precedence chain (the main config's
# [pricing] section is also merged, and wins over this file).
USER_PRICING_ENV = "TJ_PRICING_FILE"

# Default rate used when a model is not in the pricing table.
# 0.50 per MTok input, 2.00 per MTok output — conservative mid-range estimate.
DEFAULT_INPUT_PER_MTOK = 0.50
DEFAULT_OUTPUT_PER_MTOK = 2.00

# Reserved section name for *model-keyed* (provider-agnostic) overrides.
# Lives at `[models]` in the standalone pricing file and `[pricing.models]`
# in the main config. Everything else at that level is a provider section
# (`[anthropic]` / `[pricing.anthropic]`), preserving the existing
# `[provider.model]` format. No provider is named "models", so the reserved
# key never collides — see _split_pricing_raw().
MODEL_SECTION_KEY = "models"


@dataclass(frozen=True)
class ModelRates:
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float = 0.0
    cache_write_per_mtok: float = 0.0


def _rates_from(raw: dict) -> ModelRates:
    """Build ModelRates from a raw inline rate table, defaulting absent fields."""
    return ModelRates(
        input_per_mtok=raw.get("input_per_mtok", DEFAULT_INPUT_PER_MTOK),
        output_per_mtok=raw.get("output_per_mtok", DEFAULT_OUTPUT_PER_MTOK),
        cache_read_per_mtok=raw.get("cache_read_per_mtok", 0.0),
        cache_write_per_mtok=raw.get("cache_write_per_mtok", 0.0),
    )


def _split_pricing_raw(
    raw: dict,
) -> tuple[dict[str, dict[str, ModelRates]], dict[str, ModelRates]]:
    """Split a raw pricing dict into (provider_table, model_keyed).

    Two explicit forms, told apart deterministically by section name (no
    value-shape guessing, no ordering dependency):

      [models]                          # reserved model-keyed section ->
      "claude-haiku-4-5" = { ... }      #   keyed by bare model name

      [anthropic]                       # any other section is a provider ->
      "claude-haiku-4-5" = { ... }      #   keyed by (provider, model)

    A model-keyed entry wins regardless of the inferred provider, so it can
    rescue a span whose provider resolved to "unknown" (#194/#200).
    """
    provider_table: dict[str, dict[str, ModelRates]] = {}
    model_keyed: dict[str, ModelRates] = {}
    for key, val in raw.items():
        if not isinstance(val, dict):
            continue
        target = model_keyed if key == MODEL_SECTION_KEY else provider_table.setdefault(key, {})
        for model_name, rates in val.items():
            if isinstance(rates, dict):
                target[model_name] = _rates_from(rates)
    return provider_table, model_keyed


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


def _config_pricing_section() -> dict | None:
    """Return the [pricing] section of the discovered main config, or None.

    Read directly from the config file (not via a full TjConfig parse) so the
    pricing loader stays light and free of the config dataclass tree. Any
    error — no config file, unreadable, malformed — degrades silently to None;
    config problems surface through the normal config-load path elsewhere.
    """
    from tokenjam.core.config import find_config_file

    try:
        path = find_config_file()
    except (FileNotFoundError, OSError):
        return None
    if path is None:
        return None
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    section = raw.get("pricing")
    return section if isinstance(section, dict) else None


def _override_raw_sources() -> list[dict]:
    """Raw override dicts in precedence order (lowest first, later wins).

    1. The user pricing file (TJ_PRICING_FILE / ~/.config/tj/pricing.toml).
    2. The main config's [pricing] section — project-local, so it wins over
       the global user file.
    """
    sources: list[dict] = []

    user_file = _user_pricing_file()
    if user_file is not None:
        try:
            with open(user_file, "rb") as f:
                sources.append(tomllib.load(f))
        except FileNotFoundError:
            log.warning(
                "Pricing override file %s=%s not found; using packaged rates only.",
                USER_PRICING_ENV,
                user_file,
            )
        except (OSError, tomllib.TOMLDecodeError) as exc:
            log.warning(
                "Could not read pricing override file %s (%s); "
                "using packaged rates only.",
                user_file,
                exc,
            )

    section = _config_pricing_section()
    if section:
        sources.append(section)

    return sources


def _build_pricing() -> tuple[dict[str, dict[str, ModelRates]], dict[str, ModelRates]]:
    """Assemble the merged (provider_table, model_keyed) pricing structures.

    Precedence, highest first:
      user model-keyed override  >  user [provider.model] override
        >  packaged models.toml  >  default flat rate (in get_rates)

    The packaged table is the base; each override source (see
    _override_raw_sources) is merged over it per provider/model, and its
    model-keyed entries accumulate into a separate map consulted first by
    get_rates.
    """
    with open(PRICING_FILE, "rb") as f:
        provider_table, model_keyed = _split_pricing_raw(tomllib.load(f))

    for raw in _override_raw_sources():
        prov, mk = _split_pricing_raw(raw)
        for provider, models in prov.items():
            provider_table.setdefault(provider, {}).update(models)
        model_keyed.update(mk)

    return provider_table, model_keyed


@lru_cache(maxsize=1)
def load_pricing_table() -> dict[str, dict[str, ModelRates]]:
    """
    Load the packaged pricing/models.toml, then merge optional user overrides
    (the user pricing file and the main config's [pricing] section) over it,
    and return a nested dict:
      { provider: { model_name: ModelRates } }

    Provider-keyed overrides are applied per provider/model, so they can
    correct a packaged rate or add a model the package doesn't ship. Cached
    after first load — restart the process (or call clear_pricing_cache()) to
    pick up changes. Model-keyed overrides live separately; see
    load_model_pricing_overrides().
    """
    return _build_pricing()[0]


@lru_cache(maxsize=1)
def load_model_pricing_overrides() -> dict[str, ModelRates]:
    """
    Return user-declared rates keyed by **bare model name**, applied
    regardless of the inferred provider (so they price a span even when the
    provider resolved to "unknown" — #194/#200).

    Sourced from the reserved model section of the same overrides as
    load_pricing_table (`[models]` in the standalone pricing file,
    `[pricing.models]` in the main config). Cached — call
    clear_pricing_cache() to reload.
    """
    return _build_pricing()[1]


def clear_pricing_cache() -> None:
    """Clear both pricing caches so the next lookup re-reads from disk.

    Use after editing the packaged table or a user override at runtime
    (otherwise changes are picked up only on process restart). Primarily a
    test hook — both lru_caches must be cleared together to stay consistent.
    """
    load_pricing_table.cache_clear()
    load_model_pricing_overrides.cache_clear()


def _strip_date_suffix(model: str) -> str | None:
    """Return `model` minus a trailing `-YYYYMMDD` suffix, or None if absent."""
    import re as _re

    m = _re.match(r"^(.*)-(\d{8})$", model)
    return m.group(1) if m else None


def get_rates(provider: str, model: str) -> ModelRates | None:
    """
    Return ModelRates for the given provider/model, or None if not found.

    Lookup order (first match wins):
      1. A user **model-keyed** override (bare model name), consulted before
         the provider table so a user-declared rate is attribution-proof —
         it prices the model even when `provider` is "unknown" (#200).
      2. The provider-keyed table (user [provider.model] overrides merged
         over the packaged models.toml).

    Each step tries an exact match first, then falls back to stripping a
    trailing YYYYMMDD release-date suffix (Anthropic/OpenAI both ship dated
    variants like `claude-haiku-4-5-20251001`). This keeps the tables short
    while still pricing the dated names that flow through Claude Code logs.
    """
    base = _strip_date_suffix(model)

    # 1. Model-keyed user override — wins regardless of inferred provider.
    model_keyed = load_model_pricing_overrides()
    rates = model_keyed.get(model)
    if rates is not None:
        return rates
    if base is not None:
        rates = model_keyed.get(base)
        if rates is not None:
            return rates

    # 2. Provider-keyed table (user [provider.model] over packaged).
    table = load_pricing_table()
    rates = table.get(provider, {}).get(model)
    if rates is not None:
        return rates
    if base is not None:
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
