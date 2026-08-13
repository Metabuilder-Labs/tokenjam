"""Reference-vs-instruction classifier, and the asymmetry it exists to enforce.

**The two error directions do not cost the same.** Calling an instruction
"reference" relocates it out of the always-loaded file that is supposed to carry
it — a silent correctness bug. Calling reference "instruction" merely leaves a
saving on the table. So the property under test is not accuracy, it is that the
expensive direction stays at zero even when that costs recall, and the tests are
written to fail if a future tuning trades one for the other.

The labelled set below is written in the two real styles rather than copied from
anyone's files: the instruction cases carry the deontic/second-person marks that
real rules carry, and the reference cases are the inventory / API-dump / link-list
shapes that real reference sections take. The repo's own committed `CLAUDE.md` is
then used as a live regression pin, because it contains the hardest real case
there is — a module inventory with binding rules interleaved into it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tokenjam.core.summarize.classify import (
    INSTRUCTION,
    REFERENCE,
    UNCLASSIFIED,
    classify_section,
)
from tokenjam.core.summarize.sections import parse_sections

# --------------------------------------------------------------------------- #
# The labelled set: (title, body, gold)
# --------------------------------------------------------------------------- #

_MODULE_INVENTORY = """
- **`app/core/db.py`**: the storage backend, the in-memory backend used by
  tests, and the migration runner. Migrations are `(version, sql)` tuples in a
  list, and the runner keys on the version integer.
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

_ARCHITECTURE_TABLE = """
| Event | Handler | Timeout |
|-------|---------|---------|
| SessionStart | `session-start.mjs` | — |
| SessionEnd | `session-end.mjs` | — |
| PreToolUse | `pre-tool.mjs` | 5s |
| PostToolUse | `post-tool.mjs` | 5s |

| Directory | Purpose |
|-----------|---------|
| `hooks/` | Event handlers |
| `skills/` | Packaged instructions |
| `agents/` | Subagent definitions |
"""

_LINK_LIST = """
- **[docs/architecture.md](docs/architecture.md)** — design principles, system
  overview, data flow, and the semantic-convention extensions.
- **[docs/installation.md](docs/installation.md)** — base install versus the
  optional extras matrix.
- **[docs/configuration.md](docs/configuration.md)** — the full config surface.
- **[docs/api.md](docs/api.md)** — every route, its parameters, and its shape.
- **[docs/pricing.md](docs/pricing.md)** — the rate table and its three axes.
- **[docs/testing.md](docs/testing.md)** — the four test layers.
"""

_HARD_RULES = """
- **Never import from the CLI package inside core.** Core is pure domain logic
  and the dependency must only point one way.
- **You must use parameterised SQL.** Never build a query with an f-string.
- Always use the shared clock helper rather than the standard library call
  directly; you will otherwise get a naive datetime.
- Do not construct span objects directly in a test. Use the factories.
- Before removing any user-visible string, you must grep the tests for it: a
  green suite may be enforcing the defect rather than protecting against it.
- Never bump the version yourself. That is a separate release-cut concern.
"""

_HYBRID_INVENTORY_WITH_RULES = """
- **`app/core/db.py`**: the storage backend and the migration runner.
  Migrations are `(version, sql)` tuples in a list — never modify an existing
  one, only append. When you append a migration that adds a column the code
  reads, you must add that column to the expected-columns map too.
- **`app/core/ingest.py`**: the ingest pipeline and the span sanitizer. Hook
  failures are logged, never propagated.
- **`app/core/alerts.py`**: the alert engine and the dispatcher. Thresholds are
  module-level constants — read the constants, never a copy kept here.
- **`app/core/cost.py`**: the cost function and the engine. Cache-read and
  cache-write are separate fields; you must price each at its own rate.
- **`app/otel/semconv.py`**: constants only. Never hardcode an attribute name
  string; reference these instead.
"""

_TUTORIAL_PROSE = """
The build pipeline resolves template includes at build time. Templates live
beside the files they generate and carry a marker that names the skill and the
heading to pull. Heading extraction is case-insensitive and captures everything
from the heading to the next heading of equal or higher level.

The output files are committed. A check mode verifies that the outputs are up to
date, which is what continuous integration runs.

Two include marker formats exist. One extracts a markdown section by heading and
the other extracts a frontmatter field value. Both resolve against the same
skill directory.
"""

_SHORT = "A sentence. Another sentence.\n"

#: Six units, so the prose features are readable, but nothing descriptive, no
#: artifact, no link and no structure — a reference-sounding TITLE over a
#: narrative. Exactly the section a title-only heuristic relocates by mistake.
_NARRATIVE = """
- The pipeline resolved templates at build time, once per run.
- Extraction happened case-insensitively, capturing to the next heading.
- Output files got committed alongside the sources they came from.
- Check mode verified freshness, which continuous integration then ran.
- Two marker formats existed, resolving against one directory each time.
- Nothing else got pulled into the resulting output at any point.
"""

LABELLED: list[tuple[str, str, str]] = [
    ("Key Modules", _MODULE_INVENTORY, REFERENCE),
    ("Architecture", _ARCHITECTURE_TABLE, REFERENCE),
    ("Further Reading", _LINK_LIST, REFERENCE),
    ("Critical Rules", _HARD_RULES, INSTRUCTION),
    ("Key Modules", _HYBRID_INVENTORY_WITH_RULES, INSTRUCTION),
    ("Coding conventions", _HARD_RULES, INSTRUCTION),
    ("Template Include Engine", _TUTORIAL_PROSE, INSTRUCTION),
    ("Avoid Layout Thrashing", _ARCHITECTURE_TABLE, INSTRUCTION),
    ("Use the shared console", _MODULE_INVENTORY, INSTRUCTION),
    ("Anti-patterns", _MODULE_INVENTORY, INSTRUCTION),
    ("Working agreements", _ARCHITECTURE_TABLE, INSTRUCTION),
    ("Overview", _SHORT, INSTRUCTION),
]


def _scored() -> dict[str, int]:
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for title, body, gold in LABELLED:
        predicted = classify_section(title, body).verdict
        moved = predicted == REFERENCE
        if gold == REFERENCE:
            counts["tp" if moved else "fn"] += 1
        else:
            counts["fp" if moved else "tn"] += 1
    return counts


def test_the_expensive_error_direction_is_zero():
    """No section labelled INSTRUCTION may ever be classified as reference.

    This is the assertion the whole module exists for. A relocation acts only on
    a REFERENCE verdict, so a false positive here is an instruction silently
    removed from the file that carries it — and nothing downstream can detect
    it, because the move is structurally valid and the numbering gate passes.
    """
    assert _scored()["fp"] == 0


def test_the_cheap_error_direction_is_allowed_but_bounded():
    """Recall may be poor; it may not be nil, or nothing is ever relocated and
    the operation is priced at zero while claiming to exist."""
    counts = _scored()
    assert counts["tp"] >= 1
    assert counts["tp"] + counts["fn"] == 3


@pytest.mark.parametrize("title,body,gold", LABELLED, ids=[t for t, _, g in LABELLED])
def test_each_labelled_section(title, body, gold):
    predicted = classify_section(title, body).verdict
    if gold == INSTRUCTION:
        assert predicted != REFERENCE, (
            f"{title!r} would be relocated, which removes an instruction"
        )
    else:
        assert predicted == REFERENCE


# --------------------------------------------------------------------------- #
# Vetoes, individually
# --------------------------------------------------------------------------- #

def test_a_rules_title_vetoes_however_descriptive_the_prose_reads():
    """A section headed "Critical Rules" carries rules whatever its sentences
    look like, and no measurement may overrule that."""
    c = classify_section("Critical Rules", _MODULE_INVENTORY)
    assert c.verdict == INSTRUCTION
    assert c.title_instruction
    assert "never relocated" in c.reason


def test_an_imperative_title_vetoes_even_with_section_numbering():
    """"7.1 Avoid Layout Thrashing" is a rule with worked examples, and its
    example code makes it look structural."""
    assert classify_section("7.1 Avoid Layout Thrashing", _ARCHITECTURE_TABLE).verdict == INSTRUCTION


def test_a_rule_encoded_in_a_table_is_caught_by_the_raw_text_veto():
    """A table-only section has no prose units at all, so every prose feature
    passes vacuously. The raw-text density check is the backstop."""
    table = (
        "| Situation | Requirement |\n|---|---|\n"
        + "".join(f"| case {i} | you must always do the thing, never skip it |\n" for i in range(8))
    )
    c = classify_section("Reference", table)
    assert c.verdict == INSTRUCTION
    assert "per 100 words" in c.reason


def test_one_reference_signal_is_not_enough():
    """A reference-sounding title over prose that is not descriptive, names no
    artifacts and links nowhere is exactly the section a title-only heuristic
    relocates by mistake — and the original census heuristic was title-based."""
    c = classify_section("Architecture", _NARRATIVE)
    assert c.verdict == INSTRUCTION
    assert c.title_reference
    assert c.signals == ()
    assert "independent reference signals" in c.reason


def test_a_section_with_nothing_to_read_is_unclassified_not_reference():
    c = classify_section("Overview", _SHORT)
    assert c.verdict == UNCLASSIFIED
    assert not c.is_reference


def test_inline_code_is_kept_so_a_module_inventory_is_visible():
    """Stripping inline spans (which `detect.prose_text` does, correctly, for
    its own job) deletes exactly the evidence the artifact feature looks for:
    this genre names its modules as `` `app/core/db.py` ``."""
    assert classify_section("Key Modules", _MODULE_INVENTORY).artifact_share >= 0.5


def test_the_reason_names_the_test_that_decided():
    for title, body, _ in LABELLED:
        assert classify_section(title, body).reason.strip()


# --------------------------------------------------------------------------- #
# Live regression pin against this repo's own committed instruction file
# --------------------------------------------------------------------------- #

_REPO_CLAUDE_MD = Path(__file__).resolve().parents[2] / "CLAUDE.md"


@pytest.mark.skipif(not _REPO_CLAUDE_MD.is_file(), reason="repo CLAUDE.md not present")
@pytest.mark.parametrize("title", [
    "Architecture", "Key Modules", "CLI Commands", "Critical Rules", "Pricing",
    "Releases", "Config", "Packaging", "REST API", "Data Flow",
    "PR and commit conventions (for any agent producing a PR)",
])
def test_the_hard_real_sections_of_this_repos_own_claude_md_stay_put(title):
    """The hardest real cases, and they are committed right here.

    `Architecture` and `Key Modules` read as reference by title and by sentence
    shape — a heuristic built on title patterns plus imperative density (which
    is what the original census used) calls them reference and prices the move.
    They are not: binding rules are interleaved into the module descriptions at
    a granularity finer than any heading ("never modify existing ones, only
    append"; "add that column to `EXPECTED_ADDITIVE_COLUMNS` too"). Relocating
    them would remove those rules from the always-loaded file, and nothing
    downstream could detect it — the move is structurally valid and the
    numbering gate passes.

    A section absent from the file is skipped rather than failing: this pins the
    classifier, not the file's table of contents.
    """
    text = _REPO_CLAUDE_MD.read_text(encoding="utf-8")
    sections = [
        s for level in (2, 3) for s in parse_sections(text, level=level)
        if s.title == title
    ]
    if not sections:
        pytest.skip(f"{title!r} is no longer a section of this file")
    for section in sections:
        verdict = classify_section(section.title, section.body)
        assert verdict.verdict != REFERENCE, (
            f"{title!r} would be relocated out of this repo's own CLAUDE.md: "
            f"{verdict.reason}"
        )


@pytest.mark.skipif(not _REPO_CLAUDE_MD.is_file(), reason="repo CLAUDE.md not present")
def test_the_one_genuinely_relocatable_section_of_this_file_is_found():
    """The other direction, on the same real file.

    A classifier that refuses everything is trivially safe and worth nothing, so
    the recall side needs a real pin too. `Further Reading` is a list of links
    out to documents — "detailed API documentation (link to docs instead)" is on
    Anthropic's exclude list by name, and this is the finished form of that
    advice. It should be found.
    """
    text = _REPO_CLAUDE_MD.read_text(encoding="utf-8")
    sections = [s for s in parse_sections(text) if s.title == "Further Reading"]
    if not sections:
        pytest.skip("'Further Reading' is no longer a section of this file")
    verdict = classify_section(sections[0].title, sections[0].body)
    assert verdict.verdict == REFERENCE, verdict.reason
    assert verdict.signals
