from __future__ import annotations
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

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

# Cache rates for that same unpriced-model fallback. Every priced Anthropic row
# in models.toml holds cache_read_per_mtok at ~10% of input and
# cache_write_per_mtok at ~1.25x input (e.g. claude-opus-4-8: 5.00 input / 0.50
# cache_read / 6.25 cache_write; claude-sonnet-5: 2.00 / 0.20 / 2.50;
# claude-haiku-4-5: 1.00 / 0.10 / 1.25 — the ratio holds across every priced
# tier). Leaving these at 0.0 (the old behavior) silently priced an unpriced
# model's cache tokens at zero — for a model whose traffic is mostly cache
# (reads/writes), that understated true cost by close to 100%, not by the
# modest amount a flat input/output guess implies. These are a guess
# extrapolated from the observed ratio, not a quoted rate: add the model's
# real row to models.toml as soon as it's known, at which point calculate_cost
# stops using this fallback for it entirely.
DEFAULT_CACHE_READ_PER_MTOK = round(DEFAULT_INPUT_PER_MTOK * 0.1, 4)
DEFAULT_CACHE_WRITE_PER_MTOK = round(DEFAULT_INPUT_PER_MTOK * 1.25, 4)

# Reserved section name for *model-keyed* (provider-agnostic) overrides.
# Lives at `[models]` in the standalone pricing file and `[pricing.models]`
# in the main config. Everything else at that level is a provider section
# (`[anthropic]` / `[pricing.anthropic]`), preserving the existing
# `[provider.model]` format. No provider is named "models", so the reserved
# key never collides — see _split_pricing_raw().
MODEL_SECTION_KEY = "models"

# Reserved section name for *variant* definitions — the price axis that is not
# the model and not the time. A variant is a different way of buying the SAME
# model id: `fast` (Anthropic's fast mode bills the same model at a premium),
# `batch` (the Batch API bills a flat fraction of standard), `cache-1h` (the
# 1-hour cache TTL writes at a different multiple of the input rate). Lives at
# `[variants]` in the standalone pricing file and `[pricing.variants]` in the
# main config. No provider is named "variants", so the reserved key never
# collides — see _split_pricing_raw().
VARIANT_SECTION_KEY = "variants"

#: The variant every existing row implicitly declares, and the default for any
#: lookup that doesn't ask for another one.
STANDARD_VARIANT = "standard"

#: Reserved key inside a `[provider.model]` table holding its list of dated /
#: variant rate rows: `[[provider.model.rates]]`.
RATE_ROWS_KEY = "rates"

#: The four per-MTok rate fields a row may carry.
RATE_FIELDS: tuple[str, ...] = (
    "input_per_mtok",
    "output_per_mtok",
    "cache_read_per_mtok",
    "cache_write_per_mtok",
)

# Sort key for a row with no `valid_from` ("has always applied").
_BEGINNING_OF_TIME = datetime.min.replace(tzinfo=timezone.utc)

# Dedupe the "unknown variant" warning to one line per (model, variant) pair.
_UNKNOWN_VARIANT_WARNED: set[tuple[str, str]] = set()


@dataclass(frozen=True)
class ModelRates:
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float = 0.0
    cache_write_per_mtok: float = 0.0
    # True when the source (a models.toml row, a user override, or a real
    # priced entry) actually specified this cache class. False means the
    # numeric value above is a stand-in for "not specified" — either a
    # deliberate, documented omission in models.toml (e.g. OpenAI publishes no
    # cache-write rate) or an unpriced-model fallback guess (see
    # DEFAULT_CACHE_READ_PER_MTOK / DEFAULT_CACHE_WRITE_PER_MTOK below). This
    # is what makes "priced at exactly 0.0" distinguishable from "never priced
    # at all" — the two used to be indistinguishable, which is how an unpriced
    # model's cache tokens silently priced at zero (see calculate_cost).
    cache_read_specified: bool = True
    cache_write_specified: bool = True


@dataclass(frozen=True)
class RateRow:
    """One rate in a model's history: what it costs, from when, in which variant.

    `valid_from is None` means "has always applied" — the shape every existing
    single-rate `[provider.model]` row parses to, which is why adding this axis
    left `models.toml` valid unchanged.
    """
    rates: ModelRates
    valid_from: datetime | None = None
    variant: str = STANDARD_VARIANT


@dataclass(frozen=True)
class VariantSpec:
    """A model-independent variant, expressed relative to the standard rate.

    `absolute` pins a field to a dollar figure; `multipliers` maps a field to
    `(factor, of_field)` meaning `factor * standard.<of_field>` — the cross-field
    form exists because Anthropic's 1-hour cache write is priced as a multiple of
    the model's INPUT rate, not of its 5-minute cache-write rate. A field named
    in neither map keeps the standard value.
    """
    name: str
    absolute: dict[str, float] = field(default_factory=dict)
    multipliers: dict[str, tuple[float, str]] = field(default_factory=dict)


def _rates_from(raw: dict) -> ModelRates:
    """Build ModelRates from a raw inline rate table, defaulting absent fields."""
    return ModelRates(
        input_per_mtok=raw.get("input_per_mtok", DEFAULT_INPUT_PER_MTOK),
        output_per_mtok=raw.get("output_per_mtok", DEFAULT_OUTPUT_PER_MTOK),
        cache_read_per_mtok=raw.get("cache_read_per_mtok", 0.0),
        cache_write_per_mtok=raw.get("cache_write_per_mtok", 0.0),
        cache_read_specified="cache_read_per_mtok" in raw,
        cache_write_specified="cache_write_per_mtok" in raw,
    )


def as_utc(value: datetime) -> datetime:
    """Normalize a datetime to tz-aware UTC (naive input is assumed UTC)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_valid_from(value: Any) -> datetime | None:
    """Coerce a row's `valid_from` to tz-aware UTC, or None when absent/unparseable.

    TOML gives a bare `2026-09-01` as a `date` and `2026-09-01T00:00:00Z` as a
    `datetime`; a user override written through the config may leave it a string.
    An unparseable value degrades to None ("always applied") rather than dropping
    the row — a rate the user declared should still price something.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return as_utc(value)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            return as_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError:
            log.warning("Unparseable pricing valid_from %r; treating the rate as always-applied.",
                        value)
            return None
    return None


def _now() -> datetime:
    """Current time, tz-aware UTC (Critical Rule 9 — never datetime.now())."""
    from tokenjam.utils.time_parse import utcnow

    return utcnow()


def _rate_field_values(raw: dict) -> dict[str, float]:
    """The rate fields explicitly present in a raw row (absent fields omitted)."""
    return {
        name: float(raw[name])
        for name in RATE_FIELDS
        if isinstance(raw.get(name), (int, float)) and not isinstance(raw.get(name), bool)
    }


def _inherit(base: ModelRates, overrides: dict[str, float]) -> ModelRates:
    """A copy of `base` with the named fields replaced.

    This is what makes a partial row legible: a dated row that only changes
    input/output keeps the model's cache rates rather than silently zeroing them.
    """
    values = {name: getattr(base, name) for name in RATE_FIELDS}
    values.update(overrides)
    return ModelRates(**values)


def _select_row(rows: list[RateRow], at: datetime) -> RateRow | None:
    """The row in effect at `at`: the latest one whose window has opened.

    When `at` predates every row (a span older than the earliest rate we know),
    the EARLIEST row is returned rather than None — an old span priced at the
    oldest known rate is better than an unpriced span, and the alternative
    (falling through to the flat default) would be a bigger lie.
    """
    if not rows:
        return None
    started = [r for r in rows if r.valid_from is None or r.valid_from <= at]
    if started:
        return max(started, key=lambda r: r.valid_from or _BEGINNING_OF_TIME)
    return min(rows, key=lambda r: r.valid_from or _BEGINNING_OF_TIME)


def _rows_from_model(raw: dict) -> tuple[RateRow, ...]:
    """Parse one `[provider.model]` table into its ordered rate rows.

    Two forms, and the first is the whole backward-compatibility story:

      [anthropic.claude-haiku-4-5]        # flat -> exactly one row,
      input_per_mtok = 1.00               #   valid_from=None, variant=standard
      output_per_mtok = 5.00

      [[anthropic.claude-sonnet-5.rates]] # optional additional rows
      valid_from = 2026-09-01             #   a rate change, dated
      input_per_mtok = 3.00

      [[anthropic.claude-opus-5.rates]]   # or a variant of the same model
      variant = "fast"
      input_per_mtok = 10.00

    Both may coexist: the flat keys are the model's base standard rate and the
    list adds to it. A row's absent fields inherit from the standard rate in
    effect at that row's own `valid_from`.
    """
    entries = raw.get(RATE_ROWS_KEY)
    entries = [e for e in entries if isinstance(e, dict)] if isinstance(entries, list) else []
    base_fields = _rate_field_values(raw)

    rows: list[RateRow] = []
    # The flat keys are the base standard row. Kept even when they're absent and
    # there are no entries, so a malformed table still resolves to the default
    # flat rate exactly as it did before this axis existed.
    if base_fields or not entries:
        rows.append(RateRow(rates=_rates_from(raw)))

    parsed = [
        (
            str(e.get("variant", STANDARD_VARIANT)),
            _parse_valid_from(e.get("valid_from")),
            _rate_field_values(e),
        )
        for e in entries
    ]

    # Standard rows first, oldest first, each inheriting from its predecessor —
    # so a dated row that only moves input/output keeps the cache rates.
    standard = sorted(
        [p for p in parsed if p[0] == STANDARD_VARIANT],
        key=lambda p: p[1] or _BEGINNING_OF_TIME,
    )
    for _, valid_from, fields in standard:
        prior = _select_row(rows, valid_from or _BEGINNING_OF_TIME)
        base = prior.rates if prior is not None else _rates_from({})
        rows.append(RateRow(rates=_inherit(base, fields), valid_from=valid_from))

    # Then non-standard rows, each inheriting from the standard rate in effect
    # at its own valid_from.
    for variant, valid_from, fields in parsed:
        if variant == STANDARD_VARIANT:
            continue
        prior = _select_row(rows, valid_from or _BEGINNING_OF_TIME)
        base = prior.rates if prior is not None else _rates_from({})
        rows.append(RateRow(
            rates=_inherit(base, fields), valid_from=valid_from, variant=variant,
        ))

    return tuple(rows)


def _parse_variant_spec(name: str, raw: dict) -> VariantSpec:
    """Parse one `[variants.<name>]` table.

    Each field is either a plain number (an absolute per-MTok rate) or an inline
    table `{ multiplier = 0.5 }` / `{ multiplier = 2.0, of = "input_per_mtok" }`.
    """
    absolute: dict[str, float] = {}
    multipliers: dict[str, tuple[float, str]] = {}
    for field_name in RATE_FIELDS:
        value = raw.get(field_name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            absolute[field_name] = float(value)
        elif isinstance(value, dict):
            factor = value.get("multiplier")
            if isinstance(factor, (int, float)) and not isinstance(factor, bool):
                of_field = value.get("of", field_name)
                if of_field in RATE_FIELDS:
                    multipliers[field_name] = (float(factor), str(of_field))
    return VariantSpec(name=name, absolute=absolute, multipliers=multipliers)


def _apply_variant(base: ModelRates, spec: VariantSpec) -> ModelRates:
    """Derive a variant's rates from a model's standard rates."""
    overrides = dict(spec.absolute)
    for field_name, (factor, of_field) in spec.multipliers.items():
        overrides[field_name] = factor * getattr(base, of_field)
    return _inherit(base, overrides)


def _split_pricing_raw(
    raw: dict,
) -> tuple[
    dict[str, dict[str, tuple[RateRow, ...]]],
    dict[str, tuple[RateRow, ...]],
    dict[str, VariantSpec],
]:
    """Split a raw pricing dict into (provider_table, model_keyed, variants).

    Three explicit forms, told apart deterministically by section name (no
    value-shape guessing, no ordering dependency):

      [models]                          # reserved model-keyed section ->
      "claude-haiku-4-5" = { ... }      #   keyed by bare model name

      [variants]                        # reserved variant section ->
      [variants.batch]                  #   model-independent price variants

      [anthropic]                       # any other section is a provider ->
      "claude-haiku-4-5" = { ... }      #   keyed by (provider, model)

    A model-keyed entry wins regardless of the inferred provider, so it can
    rescue a span whose provider resolved to "unknown" (#194/#200).
    """
    provider_table: dict[str, dict[str, tuple[RateRow, ...]]] = {}
    model_keyed: dict[str, tuple[RateRow, ...]] = {}
    variants: dict[str, VariantSpec] = {}
    for key, val in raw.items():
        if not isinstance(val, dict):
            continue
        if key == VARIANT_SECTION_KEY:
            for variant_name, spec in val.items():
                if isinstance(spec, dict):
                    variants[variant_name] = _parse_variant_spec(variant_name, spec)
            continue
        target = model_keyed if key == MODEL_SECTION_KEY else provider_table.setdefault(key, {})
        for model_name, rates in val.items():
            if isinstance(rates, dict):
                target[model_name] = _rows_from_model(rates)
    return provider_table, model_keyed, variants


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

    Config discovery honors ``TJ_CONFIG`` the same way ``load_config`` does,
    so the [pricing] section is read from the same file as the rest of the
    app. Read directly from the config file (not via a full TjConfig parse)
    so the pricing loader stays light and free of the config dataclass tree.
    Any error — no config file, unreadable, malformed — degrades silently to
    None; config problems surface through the normal config-load path
    elsewhere.
    """
    from tokenjam.core.config import resolve_config_path

    try:
        path = resolve_config_path()
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


def _build_pricing() -> tuple[
    dict[str, dict[str, tuple[RateRow, ...]]],
    dict[str, tuple[RateRow, ...]],
    dict[str, VariantSpec],
]:
    """Assemble the merged (provider_table, model_keyed, variants) structures.

    Precedence, highest first:
      user model-keyed override  >  user [provider.model] override
        >  packaged models.toml  >  default flat rate (in get_rates)

    The packaged table is the base; each override source (see
    _override_raw_sources) is merged over it per provider/model, and its
    model-keyed entries accumulate into a separate map consulted first by
    get_rates. A user override replaces a model's WHOLE row list (the same
    replace-the-model semantics the single-rate form always had) and merges
    variant definitions per variant name.
    """
    with open(PRICING_FILE, "rb") as f:
        provider_table, model_keyed, variants = _split_pricing_raw(tomllib.load(f))

    for raw in _override_raw_sources():
        prov, mk, var = _split_pricing_raw(raw)
        for provider, models in prov.items():
            provider_table.setdefault(provider, {}).update(models)
        model_keyed.update(mk)
        variants.update(var)

    return provider_table, model_keyed, variants


def load_pricing_sources() -> dict[tuple[str, str], str]:
    """Map each (provider, model) in the resolved table to its source layer.

    Returns {(provider, model): "override" | "packaged"}, mirroring the
    precedence in _build_pricing(): the packaged models.toml is the base
    ("packaged"), and any provider/model present in an override source
    (see _override_raw_sources) is promoted to "override". This is the
    single source of truth for *where* a listed rate resolved from; callers
    (e.g. `tj pricing list`) read it instead of re-deriving precedence.

    Note: the built-in default flat rate is intentionally not represented
    here -- it applies in get_rates() to models absent from the table, which
    never appear as listed rows.
    """
    sources: dict[tuple[str, str], str] = {}

    with open(PRICING_FILE, "rb") as f:
        packaged_providers, _, _ = _split_pricing_raw(tomllib.load(f))
    for provider, models in packaged_providers.items():
        for model_name in models:
            sources[(provider, model_name)] = "packaged"

    for raw in _override_raw_sources():
        override_providers, _, _ = _split_pricing_raw(raw)
        for provider, models in override_providers.items():
            for model_name in models:
                sources[(provider, model_name)] = "override"

    return sources


@lru_cache(maxsize=1)
def load_pricing_rows() -> dict[str, dict[str, tuple[RateRow, ...]]]:
    """The full provider-keyed table with every dated / variant row preserved:
      { provider: { model_name: (RateRow, ...) } }

    This is the structure `get_rates` resolves against. Callers that just want
    "the rate now" should use load_pricing_table(), which is a view over this.
    """
    return _build_pricing()[0]


@lru_cache(maxsize=1)
def load_model_pricing_row_overrides() -> dict[str, tuple[RateRow, ...]]:
    """Model-keyed (provider-agnostic) overrides, with every row preserved."""
    return _build_pricing()[1]


@lru_cache(maxsize=1)
def load_rate_variants() -> dict[str, VariantSpec]:
    """Model-independent variant definitions from the `[variants]` section.

    These express a variant as a transform of a model's standard rate, so a
    flat-discount variant (the Batch API) or a cache-TTL variant applies to
    every model without enumerating a row per model. A per-model
    `[[provider.model.rates]]` row carrying `variant = "..."` wins over these.
    """
    return _build_pricing()[2]


def load_pricing_table() -> dict[str, dict[str, ModelRates]]:
    """
    Load the packaged pricing/models.toml, then merge optional user overrides
    (the user pricing file and the main config's [pricing] section) over it,
    and return a nested dict:
      { provider: { model_name: ModelRates } }

    Each model resolves to its **standard-variant rate in effect right now** —
    the one-rate-per-model view that predates the time and variant axes, kept
    intact for callers that only need today's list price (`tj pricing list`).
    Reach for load_pricing_rows() when the history matters.

    Provider-keyed overrides are applied per provider/model, so they can
    correct a packaged rate or add a model the package doesn't ship. Cached
    after first load — restart the process (or call clear_pricing_cache()) to
    pick up changes. Model-keyed overrides live separately; see
    load_model_pricing_overrides().
    """
    now = _now()
    return {
        provider: {
            model: row.rates
            for model, rows in models.items()
            if (row := _select_row(
                [r for r in rows if r.variant == STANDARD_VARIANT], now,
            )) is not None
        }
        for provider, models in load_pricing_rows().items()
    }


def load_model_pricing_overrides() -> dict[str, ModelRates]:
    """
    Return user-declared rates keyed by **bare model name**, applied
    regardless of the inferred provider (so they price a span even when the
    provider resolved to "unknown" — #194/#200).

    Sourced from the reserved model section of the same overrides as
    load_pricing_table (`[models]` in the standalone pricing file,
    `[pricing.models]` in the main config). Like load_pricing_table this is the
    standard-variant, in-effect-now view. Cached — call clear_pricing_cache()
    to reload.
    """
    now = _now()
    return {
        model: row.rates
        for model, rows in load_model_pricing_row_overrides().items()
        if (row := _select_row(
            [r for r in rows if r.variant == STANDARD_VARIANT], now,
        )) is not None
    }


def clear_pricing_cache() -> None:
    """Clear every pricing cache so the next lookup re-reads from disk.

    Use after editing the packaged table or a user override at runtime
    (otherwise changes are picked up only on process restart). Primarily a
    test hook — all caches must be cleared together to stay consistent.
    """
    load_pricing_rows.cache_clear()
    load_model_pricing_row_overrides.cache_clear()
    load_rate_variants.cache_clear()
    _UNKNOWN_VARIANT_WARNED.clear()


#: The two shapes a provider stamps a release date onto a model id in:
#: Anthropic's compact `-YYYYMMDD` (`claude-opus-4-20250514`) and OpenAI's
#: dashed `-YYYY-MM-DD` (`gpt-4o-2024-08-06`). Only the compact form was
#: handled originally, so every dated OpenAI id fell through to the flat
#: default rate even when its bare row was sitting in the table.
_DATE_SUFFIX_RE = re.compile(r"^(.*?)-(?:\d{8}|\d{4}-\d{2}-\d{2})$")

#: A routing prefix from a gateway or aggregator — LiteLLM emits
#: `anthropic/claude-opus-4.1`, OpenRouter `openrouter/anthropic/claude-opus-4.1`.
#: One segment is stripped per application; `_lookup_candidates` re-applies the
#: transform, so a multi-segment prefix unwinds one hop at a time.
_PROVIDER_PREFIX_RE = re.compile(r"^[A-Za-z0-9_.-]+/(.+)$")

#: A dotted version segment (`claude-opus-4.1`, `gemini-2.0-flash`) against the
#: table's dashed key form (`claude-opus-4-1`, `gemini-2-0-flash`). Only digits
#: on both sides of the dot are rewritten, so a dotted table key that is itself
#: exact (`gpt-5.5`) is unaffected — the exact match is tried first regardless.
_VERSION_DOT_RE = re.compile(r"(?<=\d)\.(?=\d)")


def _strip_date_suffix(model: str) -> str | None:
    """Return `model` minus a trailing release-date suffix, or None if absent.

    Handles both published shapes — see `_DATE_SUFFIX_RE`.
    """
    m = _DATE_SUFFIX_RE.match(model)
    return m.group(1) if (m and m.group(1)) else None


def _strip_provider_prefix(model: str) -> str | None:
    """Return `model` minus one leading `<provider>/` routing segment, or None."""
    m = _PROVIDER_PREFIX_RE.match(model)
    return m.group(1) if m else None


def _dashed_version(model: str) -> str | None:
    """Return `model` with dotted version segments dashed, or None if unchanged."""
    dashed = _VERSION_DOT_RE.sub("-", model)
    return dashed if dashed != model else None


def _strip_context_tag(model: str) -> str | None:
    """Return `model` minus a trailing context-window tag like `[1m]`, or None.

    Some runtimes stamp the requested context window onto the model name — e.g.
    Claude's 1M-context variant surfaces as ``claude-opus-4-8[1m]``. As of
    Anthropic's 2026-03-13 pricing change the 1M context window bills at
    **standard** per-token rates: the old long-context premium (historically 2x
    input / 1.5x output for requests over 200K tokens) was removed, so a 900K-token
    request bills at the same per-token rate as a 9K-token one. See
    https://platform.claude.com/docs/en/about-claude/pricing → "Long context
    pricing". Stripping the ``[...]`` tag here resolves ``claude-opus-4-8[1m]`` to
    the base ``claude-opus-4-8`` rates without duplicating rows in models.toml
    (duplicate rows would silently drift). Also handles ``[1M]`` and any future
    bracketed context tag. See PR #386.
    """
    import re as _re

    m = _re.match(r"^(.*?)\[[^\]]*\]$", model)
    return m.group(1) if (m and m.group(1)) else None


#: The name transforms tried, in priority order, when the exact model name has
#: no row. Each is `(kind, fn)`; `fn` returns the rewritten name or None when it
#: does not apply. `kind` is provenance only (see `classify_pricing_source`).
#: Order is load-bearing: the date strip must precede the version dash-ing so
#: `gpt-4.1-2025-04-14` is offered as `gpt-4.1` (a real key) before the
#: less-likely `gpt-4-1-2025-04-14`.
_NAME_TRANSFORMS: tuple[tuple[str, Any], ...] = (
    ("provider_prefix", _strip_provider_prefix),
    ("date_stripped", _strip_date_suffix),
    ("context_tag", _strip_context_tag),
    ("version_dots", _dashed_version),
)

#: How many times the transform set is re-applied to names it produced. A
#: routing prefix over a dotted, dated name needs three hops
#: (`openrouter/anthropic/claude-opus-4.1` → … → `claude-opus-4-1`); the cap
#: exists only so a pathological name cannot loop, not as a tuning knob.
_MAX_TRANSFORM_ROUNDS = 4


def _lookup_candidates(model: str) -> list[tuple[str, str]]:
    """Ordered fallback (name, kind) pairs to try for `model` (most specific first).

    The exact name is the caller's job; this returns only the rewrites. Each
    transform in `_NAME_TRANSFORMS` is applied to the original name and then,
    for `_MAX_TRANSFORM_ROUNDS` rounds, to every name produced so far — so
    combinations compose without enumerating them by hand. The forms this
    covers, and why each exists:

      * ``provider_prefix`` — a gateway's routing segment
        (``anthropic/claude-opus-4.1`` from LiteLLM,
        ``openrouter/anthropic/…`` from OpenRouter). One segment per hop.
      * ``date_stripped`` — a release-date suffix in either published shape,
        ``-YYYYMMDD`` (Anthropic) or ``-YYYY-MM-DD`` (OpenAI).
      * ``context_tag`` — a bracketed context-window tag (``claude-opus-4-8[1m]``);
        the 1M window bills at standard rates, see `_strip_context_tag`.
      * ``version_dots`` — a dotted version segment against the table's dashed
        key form (``claude-opus-4.1`` → ``claude-opus-4-1``).

    A name's `kind` is inherited from the name it was derived from, so the
    provenance names the OUTERMOST thing that had to be removed rather than the
    last transform that ran. A plain model like ``claude-opus-4-8`` yields no
    candidates at all, so the common path stays free.
    """
    candidates: list[tuple[str, str]] = []
    seen: set[str] = {model}
    frontier: list[tuple[str, str | None]] = [(model, None)]

    for _round in range(_MAX_TRANSFORM_ROUNDS):
        next_frontier: list[tuple[str, str | None]] = []
        for name, inherited in frontier:
            for kind, transform in _NAME_TRANSFORMS:
                derived = transform(name)
                if not derived or derived in seen:
                    continue
                seen.add(derived)
                derived_kind = inherited or kind
                candidates.append((derived, derived_kind))
                next_frontier.append((derived, derived_kind))
        if not next_frontier:
            break
        frontier = next_frontier

    return candidates


# Live Bedrock spans carry the provider verbatim from their ingest path —
# "aws.bedrock" (direct boto3 via BedrockIntegration), "bedrock" (LiteLLM's
# `bedrock/...` routing), or "aws_bedrock" — but the packaged table keys
# Bedrock rates under [aws.*]. Mapped here for the lookup only; the stored
# span keeps its raw provider string (#373).
_BEDROCK_PROVIDER_ALIASES = {
    "aws.bedrock": "aws",
    "aws_bedrock": "aws",
    "bedrock": "aws",
}


def _normalize_bedrock_model(model: str) -> str | None:
    """Return the [aws.*] table-key form of a raw Bedrock modelId, or None.

    boto3 modelIds look like "us.amazon.nova-micro-v1:0" — dotted, with a
    trailing ":N" version — and LiteLLM-routed ids may keep a "bedrock/"
    prefix. The table keys are dot-flattened and unversioned
    ("us-amazon-nova-micro-v1"). Only a trailing ":<digits>" is stripped;
    other colons are left alone. Returns None when normalization is a no-op
    so callers skip the redundant second lookup.
    """
    import re as _re

    normalized = model.removeprefix("bedrock/")
    normalized = _re.sub(r":\d+$", "", normalized).replace(".", "-")
    return normalized if normalized != model else None


def _resolve_rows(
    rows: tuple[RateRow, ...], at: datetime, variant: str,
) -> ModelRates | None:
    """Pick the rate a model's rows imply for a given time and variant.

    Order: an explicit per-model variant row wins; then a `[variants]`
    definition derived from the standard rate in effect at `at`; then the
    standard rate itself, with a once-per-(model, variant) warning — an
    unpriced span is worse than a span priced at the rate we do know, but the
    caller should hear that the premium wasn't applied.
    """
    if variant != STANDARD_VARIANT:
        row = _select_row([r for r in rows if r.variant == variant], at)
        if row is not None:
            return row.rates

    standard = _select_row([r for r in rows if r.variant == STANDARD_VARIANT], at)
    if standard is None:
        return None
    if variant == STANDARD_VARIANT:
        return standard.rates

    spec = load_rate_variants().get(variant)
    if spec is not None:
        return _apply_variant(standard.rates, spec)
    return standard.rates


def get_rates(
    provider: str,
    model: str,
    *,
    at: datetime | None = None,
    variant: str = STANDARD_VARIANT,
) -> ModelRates | None:
    """
    Return ModelRates for the given provider/model, or None if not found.

    `at` selects along the TIME axis — the rate whose window contains that
    instant, so a span recorded before a price change keeps pricing at the rate
    that actually billed it and a later span picks up the new one. Defaults to
    now. `variant` selects along the VARIANT axis — a different way of buying
    the same model id (`fast`, `batch`, a cache TTL); defaults to `standard`,
    which is what every single-rate row in models.toml declares.

    Lookup order (first match wins):
      1. A user **model-keyed** override (bare model name), consulted before
         the provider table so a user-declared rate is attribution-proof —
         it prices the model even when `provider` is "unknown" (#200).
      2. The provider-keyed table (user [provider.model] overrides merged
         over the packaged models.toml).

    Each step tries an exact match first, then falls back through
    `_lookup_candidates`, which strips a gateway routing prefix
    (`anthropic/claude-opus-4.1`), a release-date suffix in either published
    shape (`-YYYYMMDD` and `-YYYY-MM-DD`), a bracketed context tag
    (`claude-opus-4-8[1m]` → `claude-opus-4-8`; the 1M window bills at standard
    rates, see `_strip_context_tag`), and dotted version segments
    (`claude-opus-4.1` → `claude-opus-4-1`) — in any combination. This keeps the
    tables short while still pricing every name form that flows through Lens.

    Bedrock spans are normalized for the table lookup only (#373): the
    provider aliases in _BEDROCK_PROVIDER_ALIASES map onto the table's
    "aws" key, and the raw boto3 modelId is additionally tried in its
    table-key form (see _normalize_bedrock_model).
    """
    rows = _find_rate_rows(provider, model)
    if rows is None:
        return None

    resolved_at = as_utc(at) if at is not None else _now()
    rates = _resolve_rows(rows, resolved_at, variant)
    if (
        rates is not None
        and variant != STANDARD_VARIANT
        and not any(r.variant == variant for r in rows)
        and variant not in load_rate_variants()
    ):
        key = (model, variant)
        if key not in _UNKNOWN_VARIANT_WARNED:
            _UNKNOWN_VARIANT_WARNED.add(key)
            log.warning(
                "No '%s' variant rate for %s/%s — pricing at the standard rate, "
                "which understates it if the variant bills at a premium. Declare "
                "one under [variants.%s] or as a [[%s.%s.rates]] row.",
                variant, provider, model, variant, provider, model,
            )
    return rates


def get_rates_in_range(
    provider: str,
    model: str,
    since: datetime,
    until: datetime,
    *,
    variant: str = STANDARD_VARIANT,
) -> tuple[ModelRates, ...]:
    """Every distinct rate that could legitimately have billed a call in
    ``[since, until)`` — one entry when the price never moved, more when it did.

    `get_rates` answers "what was the rate at ONE instant". This answers "what
    were the rates across a WINDOW", which is the question anything reasoning
    about a *range* of spans has to ask: an analyzer prices each span at its own
    timestamp (see :mod:`tokenjam.core.optimize.span_pricing`), so the dollars
    it emits for a window that straddles a rate change are a blend of two rates,
    and a caller that assumes a single rate would be checking the wrong number.

    Returned in chronological order of the boundary that produced them. Empty
    when the provider/model is not in the table at all — the same "no data"
    signal `get_rates` gives by returning None.
    """
    rows = _find_rate_rows(provider, model)
    if rows is None:
        return ()

    since_utc, until_utc = as_utc(since), as_utc(until)
    # The rate can only change at a row boundary, so sampling the window start
    # plus every boundary strictly inside it visits every distinct rate exactly
    # once — no scanning, and no dependence on the window's length.
    boundaries = [since_utc]
    for row in rows:
        if row.valid_from is None:
            continue
        edge = as_utc(row.valid_from)
        if since_utc < edge < until_utc:
            boundaries.append(edge)
    boundaries.sort()

    resolved: list[ModelRates] = []
    for edge in boundaries:
        rates = _resolve_rows(rows, edge, variant)
        if rates is not None and rates not in resolved:
            resolved.append(rates)
    return tuple(resolved)


def _find_rate_rows(provider: str, model: str) -> tuple[RateRow, ...] | None:
    """The rate rows for a provider/model, applying the documented lookup order."""
    candidates = _lookup_candidates(model)

    # 1. Model-keyed user override — wins regardless of inferred provider.
    model_keyed = load_model_pricing_row_overrides()
    rows = model_keyed.get(model)
    if rows is not None:
        return rows
    for name, _kind in candidates:
        rows = model_keyed.get(name)
        if rows is not None:
            return rows

    # 2. Provider-keyed table (user [provider.model] over packaged).
    table = load_pricing_rows()
    lookup_provider = _BEDROCK_PROVIDER_ALIASES.get(provider, provider)
    provider_models = table.get(lookup_provider, {})
    rows = provider_models.get(model)
    if rows is not None:
        return rows
    for name, _kind in candidates:
        rows = provider_models.get(name)
        if rows is not None:
            return rows

    # Bedrock: the raw boto3 modelId ("us.amazon.nova-micro-v1:0") never
    # matches the dot-flattened, unversioned [aws.*] keys — try the
    # normalized table-key form (#373).
    if lookup_provider == "aws":
        normalized = _normalize_bedrock_model(model)
        if normalized is not None:
            rows = provider_models.get(normalized)
            if rows is not None:
                return rows

    return None


def classify_pricing_source(provider: str, model: str) -> str:
    """Classify HOW `get_rates(provider, model)` would resolve, for provenance.

    Mirrors get_rates' own lookup order (model-keyed override, then the
    provider table, each tried exact-then-fallback-candidates) and reports
    WHICH step would resolve, rather than re-deriving the rate itself — so the
    two functions can't silently drift apart on what counts as a match.

    Deliberately reads only the two `@lru_cache`-memoized tables
    (`load_model_pricing_overrides`, `load_pricing_table`) — the same ones
    `get_rates` itself reads. This runs on the ingest hot path (once per span,
    see CostEngine.process_span), so it must NOT touch `load_pricing_sources`
    or `_override_raw_sources`: those re-read the user pricing file and the
    main config from disk on every call (they're designed for the occasional
    `tj pricing list`, not a per-span hot path) — an earlier version of this
    function called them here and silently turned every cost computation into
    a disk read, defeating the whole point of `load_pricing_table`'s cache.
    One consequence: a provider-keyed override (`[provider.model]` in
    ~/.config/tj/pricing.toml) reads as "exact", not "override" — it's merged
    into the same cached table as the packaged row and the two are no longer
    distinguishable without the uncached lookup. Model-keyed overrides don't
    have this problem (`load_model_pricing_overrides` is its own cached
    table), so those still classify as "override".

    Returns one of:
      "override"          - resolved via a user **model-keyed** override
                             (the reserved `[models]` / `[pricing.models]`
                             section — see MODEL_SECTION_KEY).
      "exact"              - resolved via an exact row in the merged
                             provider table (packaged models.toml, a
                             provider-keyed user override, or the
                             Bedrock-normalized modelId form — all
                             indistinguishable here, see above).
      "provider_prefix"    - resolved only after stripping a leading gateway
                             routing segment (`anthropic/…`, `openrouter/…`),
                             possibly alongside further rewrites.
      "date_stripped"      - resolved only after stripping a trailing release
                             date, `-YYYYMMDD` or `-YYYY-MM-DD`.
      "context_tag"        - resolved only after stripping a trailing `[...]`
                             context tag (with or without an additional
                             date-suffix strip).
      "version_dots"       - resolved only after rewriting dotted version
                             segments to the table's dashed form
                             (`claude-opus-4.1` → `claude-opus-4-1`).
      "default_fallback"   - nothing matched; calculate_cost falls back to the
                             flat default rates (DEFAULT_INPUT_PER_MTOK /
                             DEFAULT_OUTPUT_PER_MTOK / DEFAULT_CACHE_*).

    Used to stamp `pricing_source` on a span at ingest (see CostEngine in
    core/cost.py) so a cost figure's provenance survives past ingest instead of
    being unrecoverable once the fallback and a real rate look equally
    plausible in the stored dollar amount.
    """
    candidates = _lookup_candidates(model)

    model_keyed = load_model_pricing_overrides()
    if model in model_keyed or any(name in model_keyed for name, _kind in candidates):
        return "override"

    table = load_pricing_table()
    lookup_provider = _BEDROCK_PROVIDER_ALIASES.get(provider, provider)
    provider_models = table.get(lookup_provider, {})

    if model in provider_models:
        return "exact"
    for name, kind in candidates:
        if name in provider_models:
            return kind

    if lookup_provider == "aws":
        normalized = _normalize_bedrock_model(model)
        if normalized is not None and normalized in provider_models:
            return "exact"

    return "default_fallback"


def variant_price_ratio(variant: str) -> float | None:
    """What fraction of the standard price a *uniform* variant bills at.

    Only defined for a variant whose `[variants]` definition multiplies every
    rate field by the same factor and pins none of them absolutely — the Batch
    API's flat half-price is exactly this shape. Returns None for a variant that
    is model-specific (`fast`), partial (`cache-1h` touches only cache writes),
    or undefined, because no single ratio describes it. Callers that need a
    model's actual rates should call get_rates(..., variant=...) instead.
    """
    spec = load_rate_variants().get(variant)
    if spec is None or spec.absolute:
        return None
    if set(spec.multipliers) != set(RATE_FIELDS):
        return None
    factors: set[float] = set()
    for field_name, (factor, of_field) in spec.multipliers.items():
        if of_field != field_name:
            return None  # cross-field: not a uniform scaling of the standard rate
        factors.add(factor)
    if len(factors) != 1:
        return None
    return factors.pop()


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
