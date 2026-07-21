"""`tj optimize trim` — CLI surfacing for the read-only source-file provenance
and the flagged bloat text itself.

`prompt_bloat.py` attributes a bloated prompt to a catalog file it verbatim-
contains (`BloatPrompt.source_path` / `source_basis`) and every flagged region
already carries its own sample text (`BloatRegion.sample_chars`) -- but
`_render_prompt_bloat` only ever printed bloat-char counts, never the
attributed file or the actual text a user would need to see to know what to
cut. These tests exercise the renderer surface added to close that gap,
independent of the analyzer's own provenance-matching tests
(tests/unit/test_prompt_bloat_provenance.py, if present).
"""
from __future__ import annotations

from tokenjam.core.optimize.analyzers.prompt_bloat import (
    BloatPrompt,
    BloatRegion,
    PromptBloatFinding,
)


def _flat(out: str) -> str:
    """Collapse Rich's terminal-width line wrapping to a single line so a
    long fixed string can be matched by substring regardless of where the
    console happened to wrap it."""
    return " ".join(out.split())


def _region(text: str = "some low-signal padding text here", chars: int = 45) -> BloatRegion:
    return BloatRegion(
        start_char=0, end_char=chars, char_length=chars,
        avg_score=0.21, sample_chars=text,
    )


def _prompt(
    *, source_path: str | None = None, source_basis: str = "",
    regions: list[BloatRegion] | None = None,
) -> BloatPrompt:
    regions = regions if regions is not None else [_region()]
    bloat_chars = sum(r.char_length for r in regions)
    return BloatPrompt(
        agent_id="claude-code-tokenjam",
        sample_chars="Here is a captured prompt sample used for identification purposes only.",
        prompt_chars=400,
        significant_chars=400 - bloat_chars,
        bloat_chars=bloat_chars,
        regions=regions,
        estimated_token_reduction=bloat_chars // 4,
        source_path=source_path,
        source_basis=source_basis,
    )


def _finding(*prompts: BloatPrompt) -> PromptBloatFinding:
    total_bloat = sum(p.bloat_chars for p in prompts)
    total_chars = sum(p.prompt_chars for p in prompts)
    return PromptBloatFinding(
        enabled=True, prompts_scored=len(prompts), prompts_skipped=0,
        total_bloat_chars=total_bloat, total_chars=total_chars,
        per_prompt=list(prompts),
        prompts_with_provenance=sum(1 for p in prompts if p.source_path),
    )


# --------------------------------------------------------------------------- #
# Provenance: shown only when it exists, silent otherwise (by design)
# --------------------------------------------------------------------------- #

def test_render_prompt_bloat_shows_attributed_file_and_basis(capsys):
    from tokenjam.cli.cmd_optimize import _render_prompt_bloat

    finding = _finding(_prompt(
        source_path="/repo/CLAUDE.md",
        source_basis="verbatim match: prompt contains /repo/CLAUDE.md's full "
                      "content unchanged (whitespace-normalized).",
    ))

    _render_prompt_bloat(finding, pricing_mode="api", marker="①")
    out = _flat(capsys.readouterr().out)

    assert "/repo/CLAUDE.md" in out
    assert "verbatim match" in out
    assert "Attributed to" in out


def test_render_prompt_bloat_points_at_summarize_list_never_apply(capsys):
    """`trim` has no apply path (analyzer module docstring) -- the Summarize
    pointer must be a read-only navigation verb, never one that mutates."""
    from tokenjam.cli.cmd_optimize import _render_prompt_bloat

    finding = _finding(_prompt(
        source_path="/repo/CLAUDE.md", source_basis="verbatim match.",
    ))

    _render_prompt_bloat(finding, pricing_mode="api", marker="①")
    out = _flat(capsys.readouterr().out)

    assert "tj summarize list /repo/CLAUDE.md" in out
    assert "tj summarize apply" not in out
    assert "tj summarize prep" not in out


def test_render_prompt_bloat_no_attribution_line_when_unattributed(capsys):
    """No catalog file cleared the verbatim-containment bar -- the expected,
    conservative outcome for most prompts (module docstring), not a failure.
    Nothing extra should print, and no stray "None" leaks into the text."""
    from tokenjam.cli.cmd_optimize import _render_prompt_bloat

    finding = _finding(_prompt(source_path=None, source_basis=""))

    _render_prompt_bloat(finding, pricing_mode="api", marker="①")
    out = capsys.readouterr().out

    assert "Attributed to" not in out
    assert "tj summarize" not in out
    assert "None" not in out


def test_render_prompt_bloat_unattributed_still_shows_flagged_text(capsys):
    """A pure-SDK caller (a large static system prompt, no CLAUDE.md-shaped
    catalog file in play) never gets a source_path -- `summarize` scans
    workspace files and has nothing to offer them. Their output must still be
    complete on its own: the flagged text is the whole answer for this
    persona, not a degraded stand-in for the Summarize pointer."""
    from tokenjam.cli.cmd_optimize import _render_prompt_bloat

    finding = _finding(_prompt(
        source_path=None, source_basis="",
        regions=[_region("static system prompt boilerplate padding text")],
    ))

    _render_prompt_bloat(finding, pricing_mode="api", marker="①")
    out = _flat(capsys.readouterr().out)

    assert "Flagged text" in out
    assert "static system prompt boilerplate padding text" in out
    assert "tj summarize" not in out


# --------------------------------------------------------------------------- #
# Flagged text: the actual regions, not just the bloat percentage
# --------------------------------------------------------------------------- #

def test_render_prompt_bloat_shows_flagged_region_text(capsys):
    from tokenjam.cli.cmd_optimize import _render_prompt_bloat

    finding = _finding(_prompt(regions=[
        _region("repeated boilerplate that adds nothing new here", chars=48),
    ]))

    _render_prompt_bloat(finding, pricing_mode="api", marker="①")
    out = _flat(capsys.readouterr().out)

    assert "repeated boilerplate that adds nothing new here" in out
    assert "Flagged text" in out


def test_render_prompt_bloat_caps_regions_with_trailer(capsys):
    from tokenjam.cli.cmd_optimize import _render_prompt_bloat

    regions = [_region(f"region number {i} of low-signal text padding out", chars=10)
               for i in range(5)]
    finding = _finding(_prompt(regions=regions))

    _render_prompt_bloat(finding, pricing_mode="api", marker="①")
    out = _flat(capsys.readouterr().out)

    assert "region number 0" in out
    assert "region number 1" in out
    assert "region number 2" in out
    assert "region number 3" not in out
    assert "region number 4" not in out
    assert "and 2 more region" in out


def test_render_prompt_bloat_no_flagged_text_section_when_no_regions(capsys):
    from tokenjam.cli.cmd_optimize import _render_prompt_bloat

    finding = _finding(_prompt(regions=[]))

    _render_prompt_bloat(finding, pricing_mode="api", marker="①")
    out = capsys.readouterr().out

    assert "Flagged text" not in out


# --------------------------------------------------------------------------- #
# Rich markup bracket hygiene: bloated prompt/region text often IS a config
# file's prose, which can legitimately contain "[section]" headers -- those
# must render literally, not get silently swallowed as invalid style tags.
# --------------------------------------------------------------------------- #

def test_render_prompt_bloat_escapes_brackets_in_sample_and_region_text(capsys):
    from tokenjam.cli.cmd_optimize import _render_prompt_bloat

    prompt = BloatPrompt(
        agent_id="claude-code-tokenjam",
        sample_chars="Enable [capture] prompts = true before running this.",
        prompt_chars=400, significant_chars=350, bloat_chars=50,
        regions=[_region("See the [optimize] section for tunables.", chars=50)],
        estimated_token_reduction=12,
        source_path="/repo/tj.toml",
        source_basis="verbatim match: contains [optimize] block.",
    )
    finding = _finding(prompt)

    _render_prompt_bloat(finding, pricing_mode="api", marker="①")
    out = _flat(capsys.readouterr().out)

    assert "[capture]" in out
    assert "[optimize]" in out
