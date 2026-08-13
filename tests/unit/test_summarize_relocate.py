"""The relocate operation: lossless, gated, dry-run by default.

Relocation's entire claim over compression is that it CANNOT lose meaning:
content moves, nothing is rewritten and nothing is deleted. That claim is only
worth anything if it is enforced, so most of what is tested here is the refusal
path rather than the happy one.
"""
from __future__ import annotations

import dataclasses

import pytest

from tokenjam.core.config import StorageConfig, TjConfig
from tokenjam.core.summarize.classify import REFERENCE
from tokenjam.core.summarize.relocate import (
    DEFAULT_TARGET,
    apply_relocation,
    plan_relocation,
    relocatable_content_chars,
)
from tokenjam.core.summarize.session import SummarizeRefused

_INVENTORY = """
- **`app/core/db.py`**: the storage backend, the in-memory backend used by
  tests, and the migration runner. Migrations are `(version, sql)` tuples.
- **`app/core/ingest.py`**: the ingest pipeline, the span sanitizer, and the
  content stripper. Post-ingest hooks are optional and error-tolerant.
- **`app/core/pricing.py`**: the rates dataclass, the loader, and the lookup.
  It falls back to default rates for models absent from the table.
- **`app/core/cost.py`**: the pure cost function and the cost engine that
  updates the span and session rows.
- **`app/otel/semconv.py`**: constants only, with no internal imports.
- **`app/api/routes/`**: one module per route group. Each exposes a router that
  the application factory mounts.
"""

_RULES = """
1. Never import from the CLI package inside core.
2. You must use parameterised SQL; never build a query with an f-string.
3. Always use the shared clock helper, or you will get a naive datetime.
"""

SOURCE = f"# Project\n\nPreamble.\n\n## Key Modules\n{_INVENTORY}\n## Critical Rules\n{_RULES}"


def config_for(tmp_path):
    return TjConfig(version="1", storage=StorageConfig(path=str(tmp_path / "t.duckdb")))


@pytest.fixture
def config(tmp_path):
    return config_for(tmp_path)


def _plan(tmp_path, source_text=SOURCE, target_text=""):
    return plan_relocation(
        source_path=str(tmp_path / "CLAUDE.md"), source_text=source_text,
        target_path=str(tmp_path / DEFAULT_TARGET), target_text=target_text,
    )


# --------------------------------------------------------------------------- #
# Losslessness
# --------------------------------------------------------------------------- #

def test_the_moved_section_arrives_byte_for_byte(tmp_path):
    plan = _plan(tmp_path)
    assert [s.title for s in plan.sections] == ["Key Modules"]
    assert _INVENTORY.strip() in plan.target_after
    # ...and is gone from the source, not duplicated across both.
    assert "app/core/pricing.py" not in plan.source_after


def test_nothing_is_ever_deleted_only_moved(tmp_path):
    """Removing content is a PRUNE — a different operation, not performed here."""
    plan = _plan(tmp_path)
    combined = plan.source_after + plan.target_after
    for line in _INVENTORY.strip().splitlines():
        assert line.strip() in combined


def test_the_rules_section_is_left_exactly_where_it_was(tmp_path):
    plan = _plan(tmp_path)
    assert _RULES.strip() in plan.source_after
    assert _RULES.strip() not in plan.target_after
    assert [t for t, _ in plan.declined] == ["Critical Rules"]


def test_the_stub_keeps_the_heading_and_carries_a_followable_pointer(tmp_path):
    """A pointer nobody can follow converts a lossless move into a real loss."""
    plan = _plan(tmp_path)
    assert "## Key Modules" in plan.source_after            # anchors still resolve
    assert "docs/ARCHITECTURE.md" in plan.source_after      # relative to the source dir
    assert '"Key Modules" section there' in plan.source_after


def test_the_target_records_where_each_section_came_from(tmp_path):
    plan = _plan(tmp_path)
    assert f"<!-- relocated from {tmp_path / 'CLAUDE.md'} -->" in plan.target_after


def test_appending_to_an_existing_target_keeps_what_was_there(tmp_path):
    plan = _plan(tmp_path, target_text="# Reference\n\nExisting content.\n")
    assert "Existing content." in plan.target_after
    assert plan.target_after.startswith("# Reference\n\nExisting content.\n")


# --------------------------------------------------------------------------- #
# The gates
# --------------------------------------------------------------------------- #

def test_a_move_that_would_renumber_is_refused_and_writes_nothing(monkeypatch, tmp_path):
    """The never-renumber gate sits where the structure gate sits: if it fails,
    nothing is staged. Simulated by corrupting the assembled target, which is
    the only way a legitimate move can produce numbering drift."""
    from tokenjam.core.summarize import relocate

    numbered_source = (
        "# P\n\n1. first\n2. second\n3. third\n\n## Key Modules\n" + _INVENTORY
    )
    real_remove = relocate.remove_sections

    def _renumbering_remove(text, sections, replacements):
        return real_remove(text, sections, replacements).replace("3. third", "9. third")

    monkeypatch.setattr(relocate, "remove_sections", _renumbering_remove)
    with pytest.raises(SummarizeRefused) as exc:
        _plan(tmp_path, source_text=numbered_source)
    assert "numbered list items changed" in str(exc.value)
    assert "Nothing was written" in str(exc.value)


def test_a_move_that_would_duplicate_rather_than_move_is_refused(monkeypatch, tmp_path):
    from tokenjam.core.summarize import relocate

    monkeypatch.setattr(relocate, "remove_sections", lambda text, s, r: text)
    with pytest.raises(SummarizeRefused) as exc:
        _plan(tmp_path)
    assert "DUPLICATED rather than moved" in str(exc.value)


def test_a_move_that_would_not_be_lossless_is_refused(tmp_path):
    """A gate that only ran at planning time is a gate that does not run at
    apply time, so `apply_relocation` re-runs it against exactly what is about
    to be written. Simulated by handing it a plan whose target lost a line."""
    source = tmp_path / "CLAUDE.md"
    source.write_text(SOURCE, encoding="utf-8")
    plan = _plan(tmp_path)
    tampered = dataclasses.replace(
        plan, target_after=plan.target_after.replace("app/core/pricing.py", "gone"),
    )
    with pytest.raises(SummarizeRefused) as exc:
        apply_relocation(config_for(tmp_path), tampered, go=True)
    assert "would not be lossless" in str(exc.value)
    assert source.read_text(encoding="utf-8") == SOURCE


# --------------------------------------------------------------------------- #
# Apply: dry-run by default, guarded, backed up
# --------------------------------------------------------------------------- #

def test_apply_is_a_dry_run_until_go(config, tmp_path):
    source = tmp_path / "CLAUDE.md"
    source.write_text(SOURCE, encoding="utf-8")
    plan = _plan(tmp_path)

    result = apply_relocation(config, plan)
    assert result["dry_run"] is True and result["applied"] is False
    assert source.read_text(encoding="utf-8") == SOURCE
    assert not (tmp_path / DEFAULT_TARGET).exists()


def test_apply_with_go_writes_both_files_and_backs_them_up(config, tmp_path):
    source = tmp_path / "CLAUDE.md"
    source.write_text(SOURCE, encoding="utf-8")
    plan = _plan(tmp_path)

    result = apply_relocation(config, plan, go=True)
    assert result["applied"] is True and result["tokens_freed"] > 0
    assert "app/core/pricing.py" not in source.read_text(encoding="utf-8")
    assert "app/core/pricing.py" in (tmp_path / DEFAULT_TARGET).read_text(encoding="utf-8")

    from tokenjam.core.summarize import backup
    assert backup.load_original(config, str(source), source.read_text(encoding="utf-8")) == SOURCE


def test_a_source_edited_since_the_plan_was_built_is_skipped_never_overwritten(config, tmp_path):
    source = tmp_path / "CLAUDE.md"
    source.write_text(SOURCE, encoding="utf-8")
    plan = _plan(tmp_path)
    source.write_text(SOURCE + "\n## Added since\n\nnew content\n", encoding="utf-8")

    result = apply_relocation(config, plan, go=True)
    assert result["applied"] is False
    assert [s["reason"] for s in result["skipped"]] == [
        "source changed since the plan was built — re-plan it"]
    assert "Added since" in source.read_text(encoding="utf-8")


def test_a_symlinked_source_is_refused(config, tmp_path):
    real = tmp_path / "real.md"
    real.write_text(SOURCE, encoding="utf-8")
    link = tmp_path / "CLAUDE.md"
    link.symlink_to(real)

    plan = _plan(tmp_path)
    result = apply_relocation(config, plan, go=True)
    assert result["applied"] is False
    assert "symlink" in result["skipped"][0]["reason"]
    assert real.read_text(encoding="utf-8") == SOURCE


# --------------------------------------------------------------------------- #
# The scan-side measurement
# --------------------------------------------------------------------------- #

def test_relocatable_content_chars_is_net_of_the_stub(tmp_path):
    """A raw delta would book the pointer stub as free."""
    from tokenjam.core.summarize.detect import content_chars

    freed = relocatable_content_chars(SOURCE)
    assert 0 < freed < content_chars(_INVENTORY)


def test_a_file_with_no_reference_section_measures_zero_not_an_error():
    assert relocatable_content_chars(f"# P\n\n## Critical Rules\n{_RULES}") == 0


def test_a_file_with_no_sections_measures_zero():
    assert relocatable_content_chars("just prose, at some length, no headings") == 0


def test_only_a_reference_verdict_is_ever_planned(tmp_path):
    plan = _plan(tmp_path)
    assert all(s.classification.verdict == REFERENCE for s in plan.sections)


def test_naming_a_section_explicitly_cannot_override_the_classifier(tmp_path):
    """The gate must not be talkable-out-of by a caller: an explicitly requested
    section that classifies as instruction is still not moved."""
    plan = plan_relocation(
        source_path=str(tmp_path / "CLAUDE.md"), source_text=SOURCE,
        target_path=str(tmp_path / DEFAULT_TARGET), titles=["Critical Rules"],
    )
    assert plan is None


def test_no_reference_section_means_no_plan_rather_than_an_empty_one(tmp_path):
    assert _plan(tmp_path, source_text=f"# P\n\n## Critical Rules\n{_RULES}") is None


# --------------------------------------------------------------------------- #
# The capability has a name a user can type (Critical Rule 24(d))
# --------------------------------------------------------------------------- #

def test_relocate_is_reachable_as_a_typed_command(tmp_path):
    """A capability with no typeable name is invisible, and pricing an operation
    the user cannot invoke is the error this analyzer's basis exists to avoid.
    `summarize relocate` must therefore exist as a real subcommand."""
    from click.testing import CliRunner

    from tokenjam.cli.cmd_summarize import cmd_summarize

    source = tmp_path / "CLAUDE.md"
    source.write_text(SOURCE, encoding="utf-8")
    result = CliRunner().invoke(
        cmd_summarize, ["relocate", str(source), "--json"],
        obj={"config": config_for(tmp_path)},
    )
    assert result.exit_code == 0, result.output
    payload = __import__("json").loads(result.output)
    assert payload["dry_run"] is True and payload["applied"] is False
    assert [s["title"] for s in payload["plan"]["sections"]] == ["Key Modules"]
    # ...and the file is untouched until --go.
    assert source.read_text(encoding="utf-8") == SOURCE


def test_the_cli_refuses_go_and_dry_run_together(tmp_path):
    from click.testing import CliRunner

    from tokenjam.cli.cmd_summarize import cmd_summarize

    source = tmp_path / "CLAUDE.md"
    source.write_text(SOURCE, encoding="utf-8")
    result = CliRunner().invoke(
        cmd_summarize, ["relocate", str(source), "--go", "--dry-run"],
        obj={"config": config_for(tmp_path)},
    )
    assert result.exit_code != 0
    assert "Choose one of --dry-run or --go" in result.output


def test_the_cli_says_why_when_nothing_is_offered(tmp_path):
    """Silence in response to a typed command reads as a bug. A file with no
    relocatable section gets the reason, not an empty result."""
    from click.testing import CliRunner

    from tokenjam.cli.cmd_summarize import cmd_summarize

    source = tmp_path / "CLAUDE.md"
    source.write_text(f"# P\n\n## Critical Rules\n{_RULES}", encoding="utf-8")
    result = CliRunner().invoke(
        cmd_summarize, ["relocate", str(source)], obj={"config": config_for(tmp_path)},
    )
    assert result.exit_code == 0
    assert "confidently reference" in result.output
