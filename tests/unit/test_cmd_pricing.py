"""Unit tests for `tj pricing list`."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from tokenjam.cli.main import cli
from tokenjam.core.config import TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.pricing import ModelRates


@pytest.fixture
def runner():
    return CliRunner()


def _empty_config() -> TjConfig:
    return TjConfig(version="1")


def _invoke(runner, args):
    """Invoke the CLI with a clean config and an in-memory db patched in."""
    db = InMemoryBackend()
    try:
        with patch("tokenjam.cli.main.load_config", return_value=_empty_config()), \
             patch("tokenjam.cli.main.open_db", return_value=db):
            return runner.invoke(cli, args)
    finally:
        db.close()


# A small, deterministic stand-in for the packaged pricing table so the test
# doesn't depend on the exact contents of models.toml (which change over time).
_FAKE_TABLE = {
    "anthropic": {
        "claude-sonnet-4": ModelRates(
            input_per_mtok=3.0,
            output_per_mtok=15.0,
            cache_read_per_mtok=0.3,
            cache_write_per_mtok=3.75,
        ),
    },
    "openai": {
        "gpt-4o": ModelRates(
            input_per_mtok=2.5,
            output_per_mtok=10.0,
        ),
    },
}

# claude-sonnet-4 comes from an override layer; gpt-4o from the packaged table.
_FAKE_SOURCES = {
    ("anthropic", "claude-sonnet-4"): "override",
    ("openai", "gpt-4o"): "packaged",
}


def test_pricing_list_json_outputs_resolved_rates(runner):
    """--json emits one object per model with the resolved rate fields."""
    with patch("tokenjam.cli.cmd_pricing.load_pricing_table",
               return_value=_FAKE_TABLE), \
         patch("tokenjam.cli.cmd_pricing.load_pricing_sources",
               return_value=_FAKE_SOURCES):
        result = _invoke(runner, ["pricing", "list", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) == 2

    by_model = {row["model"]: row for row in data}
    sonnet = by_model["claude-sonnet-4"]
    assert sonnet["provider"] == "anthropic"
    assert sonnet["input_per_mtok"] == 3.0
    assert sonnet["output_per_mtok"] == 15.0
    assert sonnet["cache_read_per_mtok"] == 0.3
    assert sonnet["cache_write_per_mtok"] == 3.75


def test_pricing_list_model_filter_is_case_insensitive_substring(runner):
    """--model narrows rows to a case-insensitive substring match on model."""
    with patch("tokenjam.cli.cmd_pricing.load_pricing_table",
               return_value=_FAKE_TABLE), \
         patch("tokenjam.cli.cmd_pricing.load_pricing_sources",
               return_value=_FAKE_SOURCES):
        result = _invoke(runner, ["pricing", "list", "--model", "CLAUDE", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert [row["model"] for row in data] == ["claude-sonnet-4"]


def test_pricing_list_source_reflects_packaged_and_override(runner):
    """Each row's source reflects the layer it resolved from."""
    with patch("tokenjam.cli.cmd_pricing.load_pricing_table",
               return_value=_FAKE_TABLE), \
         patch("tokenjam.cli.cmd_pricing.load_pricing_sources",
               return_value=_FAKE_SOURCES):
        result = _invoke(runner, ["pricing", "list", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    by_model = {row["model"]: row for row in data}
    assert by_model["claude-sonnet-4"]["source"] == "override"
    assert by_model["gpt-4o"]["source"] == "packaged"
