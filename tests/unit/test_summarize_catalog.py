"""Unit tests for the agent-file catalog loader (packaged + user override + cache)."""
from __future__ import annotations

import pytest

from tokenjam.core.summarize import catalog


@pytest.fixture(autouse=True)
def _clean_cache():
    catalog.clear_catalog_cache()
    yield
    catalog.clear_catalog_cache()


def test_packaged_catalog_has_core_files():
    cat = catalog.load_catalog()
    assert {"CLAUDE.md", "AGENTS.md", "GEMINI.md"} <= cat.project_files
    assert any("CLAUDE.md" in g for g in cat.global_paths)


def test_user_override_merges(tmp_path, monkeypatch):
    override = tmp_path / "agent_files.toml"
    override.write_text('[custom]\nproject_files = ["MYPROMPT.md"]\n')
    monkeypatch.setattr(catalog, "USER_CATALOG", override)
    catalog.clear_catalog_cache()
    cat = catalog.load_catalog()
    assert "MYPROMPT.md" in cat.project_files          # user addition
    assert "CLAUDE.md" in cat.project_files            # packaged entries survive the merge


def test_malformed_override_skipped(tmp_path, monkeypatch):
    override = tmp_path / "agent_files.toml"
    override.write_text("this is { not ] valid toml ===")
    monkeypatch.setattr(catalog, "USER_CATALOG", override)
    catalog.clear_catalog_cache()
    cat = catalog.load_catalog()                       # must not raise
    assert "CLAUDE.md" in cat.project_files            # falls back to the packaged table


def test_cache_invalidation(tmp_path, monkeypatch):
    override = tmp_path / "agent_files.toml"
    override.write_text('[custom]\nproject_files = ["A.md"]\n')
    monkeypatch.setattr(catalog, "USER_CATALOG", override)
    catalog.clear_catalog_cache()
    assert "A.md" in catalog.load_catalog().project_files
    override.write_text('[custom]\nproject_files = ["B.md"]\n')
    assert "B.md" not in catalog.load_catalog().project_files   # still cached
    catalog.clear_catalog_cache()
    assert "B.md" in catalog.load_catalog().project_files       # reflects the edit
