"""Prune and expire now WRITE, and the quarantine is what makes that safe.

Everything is routed at a tmp storage dir, so the quarantine lands under
``tmp/summary/quarantine`` and never the developer's real ``~/.tj`` — same
isolation pattern as ``test_summarize_apply.py``'s ``cfg`` fixture.

The tests that matter here are the ordering and refusal ones. "A prune removes
the right lines" is table stakes; the properties this design exists for are that
content can never be removed without a committed way back, and that a restore
against a file that moved on refuses rather than corrupting it.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tokenjam.core.config import StorageConfig, TjConfig
from tokenjam.core.summarize import prune, quarantine
from tokenjam.core.summarize.session import SummarizeRefused, sha256

_NOW = datetime(2026, 8, 4, tzinfo=timezone.utc)


@pytest.fixture
def cfg(tmp_path):
    return TjConfig(version="1", storage=StorageConfig(path=str(tmp_path / "t.duckdb")))


RULES = """# Project rules

## Keep this one

This rule earns its place and must survive every operation below.

## Cut this one

Standard language conventions Claude already knows, restated at length for no
reason at all.

## Also keep this

Trailing content that must not move.
"""

LOG = """# learnings

## 2024-01-15 an old lesson

Something learned a long time ago that has since become permanent.

## 2024-02-20 another old one

Also old.

## 2026-07-30 a recent one

Still live, still useful.

## Standing notes

Undated, so not an entry at all.
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# --- Planning ---------------------------------------------------------------

def test_prune_removes_only_the_named_section(tmp_path):
    path = _write(tmp_path, "CLAUDE.md", RULES)
    plan = prune.plan_prune(
        source_path=str(path), source_text=RULES, titles=["Cut this one"],
    )
    assert [f.title for f in plan.fragments] == ["Cut this one"]
    assert "Cut this one" not in plan.source_after
    assert "Keep this one" in plan.source_after
    assert "Also keep this" in plan.source_after
    assert plan.tokens_freed > 0
    # Every considered-and-not-chosen section is named, so silence never reads
    # as "there was nothing else there".
    assert {t for t, _ in plan.declined} == {"Keep this one", "Also keep this"}


def test_prune_refuses_to_select_anything_on_its_own(tmp_path):
    """Which rules earn their place is the owner's question, not the tool's.

    ``route.py`` says exactly this about its own shape measurement. A prune that
    guessed would be the product asserting more than its data supports on the
    one surface where being wrong deletes something.
    """
    with pytest.raises(SummarizeRefused, match="at least one --section"):
        prune.plan_prune(source_path="x", source_text=RULES, titles=[])


def test_a_typoed_section_is_an_error_not_a_silent_no_op():
    """A successful "pruned 0 sections" is the worst possible answer here."""
    with pytest.raises(SummarizeRefused, match="No level-2 section|no level-2 section"):
        prune.plan_prune(
            source_path="x", source_text=RULES, titles=["Cut this onee"],
        )


def test_expire_selects_dated_entries_older_than_the_cutoff():
    plan = prune.plan_expire(
        source_path="learnings.md", source_text=LOG, older_than_days=180, now=_NOW,
    )
    assert [f.title for f in plan.fragments] == [
        "2024-01-15 an old lesson", "2024-02-20 another old one",
    ]
    declined = dict(plan.declined)
    assert "2026-07-30 a recent one" in declined
    assert "newer than" in declined["2026-07-30 a recent one"]
    # An UNDATED section in a log is standing content, never swept along.
    assert "Standing notes" in declined
    assert "no date in the heading" in declined["Standing notes"]


def test_expire_leaves_everything_when_nothing_is_old_enough():
    plan = prune.plan_expire(
        source_path="learnings.md", source_text=LOG, older_than_days=100_000, now=_NOW,
    )
    assert plan.fragments == []
    assert plan.source_after == LOG


# --- The ordering property --------------------------------------------------

def test_a_prune_quarantines_before_it_touches_the_file(cfg, tmp_path, monkeypatch):
    """THE safety property, asserted on ORDER rather than on outcome.

    The source write is replaced with one that records what the quarantine held
    at the moment it was called. If the quarantine is empty then, the design is
    wrong however green the other tests are.
    """
    path = _write(tmp_path, "CLAUDE.md", RULES)
    plan = prune.plan_prune(
        source_path=str(path), source_text=RULES, titles=["Cut this one"],
    )
    seen: dict = {}
    import tokenjam.core.summarize.apply as apply_mod

    real_write = apply_mod._write

    def _spy(p, text):
        seen["entries"] = quarantine.list_entries(cfg)
        return real_write(p, text)

    monkeypatch.setattr(apply_mod, "_write", _spy)
    prune.apply_prune(cfg, plan, go=True)

    assert len(seen["entries"]) == 1, "the file was written before anything was quarantined"
    assert "Cut this one" in seen["entries"][0].removed_text


def test_a_failed_quarantine_write_aborts_with_the_file_untouched(cfg, tmp_path, monkeypatch):
    """An unverifiable quarantine entry is the same situation as no quarantine.

    So it must be fatal to the apply, and fatal BEFORE the source changes.
    """
    path = _write(tmp_path, "CLAUDE.md", RULES)
    plan = prune.plan_prune(
        source_path=str(path), source_text=RULES, titles=["Cut this one"],
    )

    def _boom(*_a, **_kw):
        raise OSError("disk full")

    monkeypatch.setattr(quarantine, "_fsync_write", _boom)
    with pytest.raises(SummarizeRefused, match="could not write the quarantine"):
        prune.apply_prune(cfg, plan, go=True)

    assert path.read_text(encoding="utf-8") == RULES
    assert quarantine.list_entries(cfg) == []


def test_a_quarantine_entry_that_cannot_be_read_back_aborts_the_apply(cfg, tmp_path, monkeypatch):
    """A successful write call is not evidence the entry is retrievable."""
    path = _write(tmp_path, "CLAUDE.md", RULES)
    plan = prune.plan_prune(
        source_path=str(path), source_text=RULES, titles=["Cut this one"],
    )
    monkeypatch.setattr(quarantine, "read", lambda *_a, **_kw: None)
    with pytest.raises(SummarizeRefused, match="could not be read back"):
        prune.apply_prune(cfg, plan, go=True)
    assert path.read_text(encoding="utf-8") == RULES


def test_a_dry_run_writes_nothing_at_all(cfg, tmp_path):
    path = _write(tmp_path, "CLAUDE.md", RULES)
    plan = prune.plan_prune(
        source_path=str(path), source_text=RULES, titles=["Cut this one"],
    )
    result = prune.apply_prune(cfg, plan, go=False)
    assert result["dry_run"] is True
    assert result["applied"] is False
    assert result["quarantined"] == []
    assert path.read_text(encoding="utf-8") == RULES
    assert quarantine.list_entries(cfg) == []


def test_a_file_edited_since_the_plan_was_built_is_skipped(cfg, tmp_path):
    path = _write(tmp_path, "CLAUDE.md", RULES)
    plan = prune.plan_prune(
        source_path=str(path), source_text=RULES, titles=["Cut this one"],
    )
    path.write_text(RULES + "\n## Added since\n\nNew.\n", encoding="utf-8")
    result = prune.apply_prune(cfg, plan, go=True)
    assert result["applied"] is False
    assert "changed since the plan was built" in result["skipped"][0]["reason"]
    assert quarantine.list_entries(cfg) == []


# --- Restore ----------------------------------------------------------------

def test_restore_round_trips_a_pruned_fragment_exactly(cfg, tmp_path):
    """Byte-for-byte. A restore that "mostly" restores an instruction file is a
    restore that silently changed one."""
    path = _write(tmp_path, "CLAUDE.md", RULES)
    plan = prune.plan_prune(
        source_path=str(path), source_text=RULES, titles=["Cut this one"],
    )
    result = prune.apply_prune(cfg, plan, go=True)
    assert path.read_text(encoding="utf-8") != RULES

    entry_id = result["quarantined"][0]
    outcome = quarantine.restore(cfg, entry_id, go=True)
    assert outcome["restored"] is True
    assert outcome["source_unchanged"] is True
    assert path.read_text(encoding="utf-8") == RULES
    assert sha256(path.read_text(encoding="utf-8")) == sha256(RULES)


def test_restore_all_round_trips_several_fragments_from_one_apply(cfg, tmp_path):
    """The multi-fragment case, which anchoring against the ORIGINAL breaks.

    A fragment's neighbour in the original is often another fragment being
    removed in the same apply, so an anchor taken from the original names text
    the restored file never contains. Anchors come from the RESULT for exactly
    this test.
    """
    path = _write(tmp_path, "learnings.md", LOG)
    plan = prune.plan_expire(
        source_path=str(path), source_text=LOG, older_than_days=180, now=_NOW,
    )
    assert len(plan.fragments) == 2
    prune.apply_prune(cfg, plan, go=True)
    assert "2024-01-15" not in path.read_text(encoding="utf-8")

    outcome = quarantine.restore_all(cfg, source_path=str(path), go=True)
    assert outcome["restored"] == 2, outcome["refused"]
    assert path.read_text(encoding="utf-8") == LOG


def test_restore_against_a_modified_file_reanchors_without_corrupting(cfg, tmp_path):
    """An edit elsewhere in the file must not stop a safe restore.

    The fragment goes back where its surrounding text still is, and the rest of
    the user's edit survives untouched. Line numbers are never consulted.
    """
    path = _write(tmp_path, "CLAUDE.md", RULES)
    plan = prune.plan_prune(
        source_path=str(path), source_text=RULES, titles=["Cut this one"],
    )
    result = prune.apply_prune(cfg, plan, go=True)

    edited = path.read_text(encoding="utf-8") + "\n## Written since\n\nBrand new rule.\n"
    path.write_text(edited, encoding="utf-8")

    outcome = quarantine.restore(cfg, result["quarantined"][0], go=True)
    assert outcome["restored"] is True
    assert outcome["source_unchanged"] is False
    final = path.read_text(encoding="utf-8")
    assert "Brand new rule." in final          # the user's edit survived
    assert "Cut this one" in final             # the fragment came back
    assert final.index("Cut this one") < final.index("Also keep this")


def test_restore_refuses_when_it_cannot_place_the_fragment(cfg, tmp_path):
    """Silent corruption of an instruction file is worse than a refusal.

    When the surrounding text is gone there is no honest answer but no. The
    message names the entry so the user can read the text and paste it back.
    """
    path = _write(tmp_path, "CLAUDE.md", RULES)
    plan = prune.plan_prune(
        source_path=str(path), source_text=RULES, titles=["Cut this one"],
    )
    result = prune.apply_prune(cfg, plan, go=True)

    rewritten = "# Project rules\n\nEverything here was rewritten from scratch.\n"
    path.write_text(rewritten, encoding="utf-8")

    outcome = quarantine.restore(cfg, result["quarantined"][0], go=True)
    assert outcome["restored"] is False
    assert "can no longer be located" in outcome["reason"]
    assert outcome["entry_id"] in outcome["reason"]
    # THE assertion: the file is untouched, not half-restored.
    assert path.read_text(encoding="utf-8") == rewritten


def test_restore_of_already_present_text_is_declined(cfg, tmp_path):
    path = _write(tmp_path, "CLAUDE.md", RULES)
    plan = prune.plan_prune(
        source_path=str(path), source_text=RULES, titles=["Cut this one"],
    )
    result = prune.apply_prune(cfg, plan, go=True)
    quarantine.restore(cfg, result["quarantined"][0], go=True)
    again = quarantine.restore(cfg, result["quarantined"][0], go=True)
    assert again["restored"] is False
    assert "already present" in again["reason"]


def test_a_restore_dry_run_writes_nothing(cfg, tmp_path):
    path = _write(tmp_path, "CLAUDE.md", RULES)
    plan = prune.plan_prune(
        source_path=str(path), source_text=RULES, titles=["Cut this one"],
    )
    result = prune.apply_prune(cfg, plan, go=True)
    pruned = path.read_text(encoding="utf-8")
    outcome = quarantine.restore(cfg, result["quarantined"][0], go=False)
    assert outcome["restored"] is False
    assert outcome["dry_run"] is True
    assert outcome["preview"]
    assert path.read_text(encoding="utf-8") == pruned


def test_an_unknown_entry_id_refuses_clearly(cfg):
    with pytest.raises(SummarizeRefused, match="no quarantine entry"):
        quarantine.restore(cfg, "nope", go=True)


def test_a_corrupt_entry_reads_as_absent_rather_than_crashing_the_listing(cfg, tmp_path):
    path = _write(tmp_path, "CLAUDE.md", RULES)
    plan = prune.plan_prune(
        source_path=str(path), source_text=RULES, titles=["Cut this one"],
    )
    prune.apply_prune(cfg, plan, go=True)
    for meta in quarantine.quarantine_dir(cfg).glob("*.meta.json"):
        meta.write_text("{not json", encoding="utf-8")
    assert quarantine.list_entries(cfg) == []


def test_the_quarantine_lives_under_the_configured_storage_root(cfg, tmp_path):
    assert quarantine.quarantine_dir(cfg) == tmp_path / "summary" / "quarantine"


def test_forget_drops_an_entry_but_restore_does_not(cfg, tmp_path):
    """A restored fragment's record stays: a user who restores and then decides
    they were right the first time still has what was cut."""
    path = _write(tmp_path, "CLAUDE.md", RULES)
    plan = prune.plan_prune(
        source_path=str(path), source_text=RULES, titles=["Cut this one"],
    )
    result = prune.apply_prune(cfg, plan, go=True)
    entry_id = result["quarantined"][0]
    quarantine.restore(cfg, entry_id, go=True)
    assert len(quarantine.list_entries(cfg)) == 1
    assert quarantine.forget(cfg, entry_id) is True
    assert quarantine.list_entries(cfg) == []


def test_prune_refuses_a_plan_that_would_move_anything_outside_the_selection():
    """The gate that makes "only these lines" a checked claim rather than a hope."""
    plan = prune.plan_prune(
        source_path="x", source_text=RULES, titles=["Cut this one"],
    )
    tampered = prune.PrunePlan(
        source_path=plan.source_path, route=plan.route,
        source_before=plan.source_before,
        source_after=plan.source_after.replace("Also keep this", "Renamed"),
        fragments=plan.fragments, declined=plan.declined,
    )
    with pytest.raises(SummarizeRefused, match="Something outside the selection"):
        prune._assert_gates(tampered)


def test_a_symlink_source_is_never_written_through(cfg, tmp_path):
    real = _write(tmp_path, "real.md", RULES)
    link = tmp_path / "CLAUDE.md"
    link.symlink_to(real)
    plan = prune.plan_prune(source_path=str(link), source_text=RULES, titles=["Cut this one"])
    result = prune.apply_prune(cfg, plan, go=True)
    assert result["applied"] is False
    assert "symlink" in result["skipped"][0]["reason"]
    assert real.read_text(encoding="utf-8") == RULES


def test_expire_treats_a_month_only_heading_as_the_first_of_the_month():
    """The reading that expires it LATEST — conservative when the day is unknown."""
    text = "# log\n\n## 2026-03 a month-only entry\n\nBody.\n"
    kept = prune.plan_expire(
        source_path="x", source_text=text, older_than_days=180,
        now=datetime(2026, 8, 20, tzinfo=timezone.utc),
    )
    assert [f.title for f in kept.fragments] == []
    gone = prune.plan_expire(
        source_path="x", source_text=text, older_than_days=180,
        now=datetime(2026, 3, 1, tzinfo=timezone.utc) + timedelta(days=200),
    )
    assert [f.title for f in gone.fragments] == ["2026-03 a month-only entry"]


# --- The concurrent-edit window (data loss) ---------------------------------

def test_an_edit_during_the_quarantine_window_is_never_destroyed(cfg, tmp_path, monkeypatch):
    """THE data-loss regression.

    The hash check ran, then N fsync'd quarantine writes and a gzip backup ran,
    and only then was the source rewritten. An editor landing in that window had
    its work destroyed three ways at once: overwritten in the file, absent from
    the quarantine (which only holds the removed fragments), and absent from the
    BACKUP — because the backup was saved with the plan's text rather than a
    fresh read, so `undo` restored a state the file was never in. Unrecoverable
    by any of the three rails.

    The concurrent editor is simulated inside the quarantine write, which is
    exactly where the real one lands.
    """
    path = _write(tmp_path, "CLAUDE.md", RULES)
    plan = prune.plan_prune(
        source_path=str(path), source_text=RULES, titles=["Cut this one"],
    )
    real_record = quarantine.record

    def _record_then_someone_edits(*args, **kwargs):
        entry = real_record(*args, **kwargs)
        path.write_text(
            RULES + "\n## Written by someone else\n\nMid-apply edit.\n",
            encoding="utf-8",
        )
        return entry

    monkeypatch.setattr(quarantine, "record", _record_then_someone_edits)
    result = prune.apply_prune(cfg, plan, go=True)

    assert result["applied"] is False
    assert "changed while this apply was preparing" in result["skipped"][0]["reason"]
    # The concurrent edit survived, in the file, untouched.
    assert "Mid-apply edit." in path.read_text(encoding="utf-8")
    assert "Cut this one" in path.read_text(encoding="utf-8")
    # And no orphan records of a cut that never happened.
    assert quarantine.list_entries(cfg) == []


def test_the_backup_records_a_fresh_read_not_the_plans_text(cfg, tmp_path):
    """`undo` must restore a state the file was ACTUALLY in."""
    from tokenjam.core.summarize import backup

    path = _write(tmp_path, "CLAUDE.md", RULES)
    plan = prune.plan_prune(
        source_path=str(path), source_text=RULES, titles=["Cut this one"],
    )
    prune.apply_prune(cfg, plan, go=True)
    assert backup.load_original(cfg, str(path), None) == RULES


# --- Duplicate section titles (data loss + false report) --------------------

_DUPES = """# Notes file

## Notes

First block, and it must not vanish silently.

## Other

Middle.

## Notes

Second block with the same heading.
"""


def test_duplicate_section_titles_refuse_rather_than_pruning_one(cfg, tmp_path):
    """`{s.title: s}` kept only the LAST section of a repeated heading.

    So `--section Notes --go` removed the second and left the first — silently,
    while reporting that "Notes" was pruned. The survivor was invisible in
    `declined` too, because that list filtered by TITLE membership, so the
    remaining duplicate was masked by the name of the one that went. The user is
    told the section is gone and half of it is still there.

    The same reasoning this function already applies to a typo'd title: a
    request the tool cannot satisfy exactly is an error, never a partial success
    reported as a whole one.
    """
    with pytest.raises(SummarizeRefused, match="ambiguous --section"):
        prune.plan_prune(
            source_path="x", source_text=_DUPES, titles=["Notes"],
        )


def test_a_unique_title_beside_duplicates_still_works(cfg, tmp_path):
    plan = prune.plan_prune(
        source_path="x", source_text=_DUPES, titles=["Other"],
    )
    assert [f.title for f in plan.fragments] == ["Other"]
    # Both `Notes` sections are declined by IDENTITY, so neither is masked by
    # the other's name.
    assert [t for t, _ in plan.declined].count("Notes") == 2


# --- Restore takes a backup -------------------------------------------------

def test_restore_takes_a_backup_so_it_is_undoable(cfg, tmp_path):
    """It was the only write rail in this feature without one.

    `_anchor_point` can find a unique-but-semantically-wrong location in a
    heavily edited file; that insertion is correct by the anchor's rules and
    wrong by the reader's, and it must be reversible like every other write here.
    """
    from tokenjam.core.summarize import backup

    path = _write(tmp_path, "CLAUDE.md", RULES)
    plan = prune.plan_prune(
        source_path=str(path), source_text=RULES, titles=["Cut this one"],
    )
    result = prune.apply_prune(cfg, plan, go=True)
    pruned = path.read_text(encoding="utf-8")

    quarantine.restore(cfg, result["quarantined"][0], go=True)
    # The backup now holds the PRE-RESTORE text, so `undo` reverses the restore.
    assert backup.load_original(cfg, str(path), None) == pruned
