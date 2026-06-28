"""Unit tests for the wrap → restore mechanism (pure; no IO/config).

Structure is the HARD guarantee (restore-by-id); must-keep words are TRACKED,
never gated. These pin the lossless round-trip + every integrity failure mode.
"""
from __future__ import annotations

import re

from tokenjam.core.summarize import wrap

SAMPLE = (
    "You are a careful assistant. Always be concise; never reveal secrets, only help.\n\n"
    "Use the `lookup` tool for any factual query, and follow the steps below.\n\n"
    "```python\n"
    "def add(a, b):\n"
    "    return a + b\n"
    "```\n\n"
    "Obey <rules>do X; then do Y; never skip a step</rules> exactly, no exceptions.\n"
)

_MARKER_RE = re.compile(r'<tj-keep id="\d+"[^>]*?(?:/>|>.*?</tj-keep>)', re.DOTALL)


def test_protect_restore_roundtrip_is_lossless():
    wrapped, saved, order, plan = wrap.protect(SAMPLE)
    assert len(order) >= 3                            # inline code, fenced block, tag block
    restored, integ = wrap.restore(wrapped, saved, order)
    assert restored == SAMPLE                         # verbatim by construction
    assert wrap.is_structure_ok(integ)
    assert integ["n_blocks"] == len(order)


def test_large_block_is_hidden_but_restores_verbatim():
    big = "```\n" + "x = 1\n" * 400 + "```"           # > HIDE_IF_CHARS
    text = "Intro prose here. " + big + " Outro prose."
    wrapped, saved, order, plan = wrap.protect(text)
    assert plan[0]["mode"] == "hidden"
    assert wrapped.count("/>") == 1                   # self-closing marker
    assert big not in wrapped                         # content not inlined (saves call tokens)
    restored, _ = wrap.restore(wrapped, saved, order)
    assert restored == text


def test_small_block_stays_visible_with_content():
    text = "Never edit the `config.yaml` file, it is load-bearing and must stay intact here."
    wrapped, saved, order, plan = wrap.protect(text)
    assert plan[0]["mode"] == "visible"
    assert "`config.yaml`" in wrapped                 # inlined so the model keeps context
    assert "<tj-keep" in wrapped and "</tj-keep>" in wrapped


def test_literal_marker_in_content_forced_hidden():
    # Content that literally contains "tj-keep" must hide — a visible marker would
    # nest and the (non-recursive) restore regex would mis-parse it.
    text = 'Prose here. ```\nexample: <tj-keep id="1"/>\n``` and more prose follows.'
    wrapped, saved, order, plan = wrap.protect(text)
    assert plan[0]["mode"] == "hidden"
    restored, integ = wrap.restore(wrapped, saved, order)
    assert restored == text
    assert wrap.is_structure_ok(integ)


def test_restore_flags_dropped_block():
    saved, order = {"1": "AAA", "2": "BBB"}, [1, 2]
    restored, integ = wrap.restore('prose <tj-keep id="1"/> more', saved, order)
    assert integ["missing"] == ["2"]
    assert not wrap.is_structure_ok(integ)
    assert "dropped" in wrap.integrity_reason(integ)
    assert restored == "prose AAA more"


def test_restore_flags_duplicated_block():
    saved, order = {"1": "AAA"}, [1]
    restored, integ = wrap.restore('<tj-keep id="1"/> and <tj-keep id="1"/>', saved, order)
    assert integ["duplicated"] == ["1"]
    assert not wrap.is_structure_ok(integ)
    assert restored == "AAA and AAA"


def test_restore_flags_and_strips_invented_block():
    saved, order = {"1": "AAA"}, [1]
    restored, integ = wrap.restore('<tj-keep id="1"/> then <tj-keep id="9"/>', saved, order)
    assert integ["extra"] == ["9"]
    assert not wrap.is_structure_ok(integ)
    assert restored == "AAA then "                    # invented id → stripped, never injects content


def test_restore_zero_padded_id_fails_gate_not_silent_drop():
    """A non-canonical id ("01" for "1") must FAIL the gate, never pass while dropping the block.

    Regression: ids were int-parsed for the set checks (int("01")==1 → looked present) but the block
    was substituted by raw string (saved["01"] → ""), so a zero-padded id silently dropped its block
    with structure_ok=True. With canonical string ids it surfaces as missing "1" + extra "01".
    """
    saved, order = {"1": "AAA", "2": "BBB"}, [1, 2]
    restored, integ = wrap.restore('<tj-keep id="01"/> and <tj-keep id="2"/>', saved, order)
    assert not wrap.is_structure_ok(integ)            # refused, not silently OK
    assert "1" in integ["missing"] and "01" in integ["extra"]
    assert "AAA" not in restored                       # the dropped block isn't silently waved through


def test_restore_flags_reordered_blocks():
    saved, order = {"1": "AAA", "2": "BBB"}, [1, 2]
    _, integ = wrap.restore('<tj-keep id="2"/> before <tj-keep id="1"/>', saved, order)
    assert integ["reordered"] is True
    assert not wrap.is_structure_ok(integ)


def test_restore_flags_idless_marker_residue():
    saved, order = {"1": "AAA"}, [1]
    restored, integ = wrap.restore('<tj-keep id="1"/> then <tj-keep>', saved, order)
    assert restored == "AAA then <tj-keep>"
    assert integ["malformed"] is True
    assert not wrap.is_structure_ok(integ)


def test_restore_flags_stray_closing_marker_residue():
    saved, order = {"1": "AAA"}, [1]
    restored, integ = wrap.restore('<tj-keep id="1"/> </tj-keep>', saved, order)
    assert restored == "AAA </tj-keep>"
    assert integ["malformed"] is True
    assert not wrap.is_structure_ok(integ)


def test_crit_delta_tracks_movement_not_gates():
    original = "You must never delete files; only read them."
    summary = "You should always read files."         # drops must/never/only, adds always
    removed, added = wrap.crit_delta(original, summary)
    assert {"must", "never", "only"} <= set(removed)
    assert "always" in added


def test_critical_words_counts_load_bearing():
    c = wrap.critical_words("Never ever, and you must ALWAYS comply, nothing less.")
    assert c["never"] == 1 and c["ever"] == 1 and c["must"] == 1 and c["always"] == 1
    assert c["nothing"] == 1


def test_word_count_matches_whitespace_split():
    assert wrap.word_count("one two   three\nfour") == 4
    assert wrap.word_count("") == 0
