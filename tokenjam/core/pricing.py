from __future__ import annotations
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]


PRICING_FILE = Path(__file__).parent.parent / "pricing" / "models.toml"

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


@lru_cache(maxsize=1)
def load_pricing_table() -> dict[str, dict[str, ModelRates]]:
    """
    Load pricing/models.toml and return a nested dict:
      { provider: { model_name: ModelRates } }
    Cached after first load — restart process to pick up changes.
    """
    with open(PRICING_FILE, "rb") as f:
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


def get_rates(provider: str, model: str) -> ModelRates | None:
    """Return ModelRates for the given provider/model, or None if not found."""
    table = load_pricing_table()
    return table.get(provider, {}).get(model)
