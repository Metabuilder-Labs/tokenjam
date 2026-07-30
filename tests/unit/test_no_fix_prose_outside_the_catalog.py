"""No analyzer module may author its own user-facing fix prose.

A standing guard, not a one-off cleanup — the same shape as
``test_rulewrite_legacy``'s guard against the re-introduced ladder rung, and
for the same reason: the cleanup is worthless if the next author can undo it
without anyone noticing.

The defect this prevents is not untidiness. Every one of these shipped, past
readers who were looking, and every one of them was possible only because the
policy had no single home to be checked in:

* the identical sizing-rule contradiction lived in BOTH the subagent rubric and
  the resend right-size template, in different words, so fixing the reported one
  left the other live;
* three analyzers each authored their own wording of "delegate context-heavy
  work to a subagent", any combination of which could land in one file — and
  length plus redundancy REDUCE adherence, so writing a rule three times makes
  it less likely to be followed than writing it once;
* four call sites each wrote their own wording of the cache_control
  instruction;
* a card shipped whose fix text said no action was needed while still occupying
  an inbox slot.

The lint in ``core/fixes/lint`` catches all four classes — but only over what
is CATALOGUED. A green lint over eight records while thirty texts live
elsewhere is worse than no lint, because it reads as "the fix text is checked".
This test is what makes the lint's coverage total.

**Where the line falls.** The catalog owns DURABLE POLICY: the instruction, as
it would read for any user who hit this finding. It does not own GROUNDING —
the sentence naming this row's server, this window's session count, this
model's id. That distinction is mechanical here rather than a matter of taste:
a static string literal says the same thing to everyone and therefore belongs
in one place, while an f-string interpolating the finding is by construction
per-row. So the check is on the longest CONTIGUOUS run of static text, which
lets a grounded sentence stitch short fragments around its evidence while a
rule smuggled into an f-string still trips.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tokenjam.core.fixes.lint import MIN_FIX_CHARS

ROOT = Path(__file__).resolve().parents[2] / "tokenjam"
ANALYZERS = ROOT / "core" / "optimize" / "analyzers"

#: Every module that BUILDS a card, not only the analyzers. Scoping this to
#: `analyzers/` alone would have left the largest holder of fix prose out: the
#: card builders are where an analyzer's finding acquires the words a user
#: reads, and two of the three defects this migration found were there rather
#: than in an analyzer — a fourth copy of the offload rule in a one-paste
#: block, and a sentence restating the subagent rubric a paragraph below it.
_CARD_BUILDERS = (
    ROOT / "core" / "optimize" / "cost_proposals.py",
    ROOT / "core" / "optimize" / "relearn_apply.py",
    ROOT / "core" / "optimize" / "relearn_proposals.py",
    ROOT / "cli" / "cmd_optimize.py",
)

#: Names whose value IS the fix a user is shown or writes into a file. Slot
#: names rather than a guess at prose: what makes a string a fix is where it
#: goes, not how it reads.
_FIX_SLOTS = frozenset({
    "fix", "proposed_fix", "one_paste_fix", "artifact_text",
    "remedy_snippet", "fix_template", "suggestion",
})

#: Constants that carry fix prose, by the naming this codebase already uses.
_FIX_CONSTANT = (
    "FIX", "RUBRIC", "REMEDY", "LEVER", "RULE_TEXT", "SNIPPET", "ADVICE",
)

#: The only calls that may produce fix text. Reaching the catalog through any
#: of these means the text is linted; anything else means it is not.
_CATALOG_CALLS = frozenset({"fix_text", "fix_text_for", "ground", "compound_offload_fix"})


def _is_fix_slot(node: ast.AST) -> bool:
    """Whether this node names a slot whose value is user-facing fix text."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value in _FIX_SLOTS                      # {"fix": ...}
    if isinstance(node, ast.Name):
        return node.id in _FIX_SLOTS or any(
            token in node.id.upper() for token in _FIX_CONSTANT
        )
    return False


def _longest_static_run(node: ast.AST) -> int:
    """The longest contiguous stretch of literal text ``node`` can produce.

    A grounded sentence ("Remove the `{name}` server; {n} sessions") is made of
    short fragments around interpolated evidence, so its longest run is small. A
    rule is one long stretch of prose however it is quoted, so hiding it in an
    f-string does not get it past this.
    """
    longest = 0
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            longest = max(longest, len(child.value.strip()))
    return longest


def _resolves_through_the_catalog(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in _CATALOG_CALLS:
                return True
    return False


def _offenders_in(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[str] = []

    def check(slot: str, value: ast.AST) -> None:
        if _resolves_through_the_catalog(value):
            return
        run = _longest_static_run(value)
        if run < MIN_FIX_CHARS:
            return
        out.append(
            f"{path.name}:{getattr(value, 'lineno', '?')}: {slot} carries "
            f"{run} characters of hardcoded fix prose",
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if _is_fix_slot(target):
                    check(getattr(target, "id", "?"), node.value)
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if key is not None and _is_fix_slot(key):
                    label = getattr(key, "value", None) or getattr(key, "id", "?")
                    check(f'"{label}"', value)
        elif isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg in _FIX_SLOTS:
                    check(f"{kw.arg}=", kw.value)
    return out


@pytest.mark.parametrize(
    "module",
    sorted(p.name for p in ANALYZERS.glob("*.py") if p.name != "__init__.py"),
)
def test_no_analyzer_defines_its_own_fix_prose(module):
    """THE guard. An analyzer names the fix it hands out; it does not author it.

    Failing this means a policy has acquired a second home, and a second
    definition of one policy is two policies that will disagree — which is not
    a prediction, it is what happened four times over. Move the text to
    ``core/fixes/registry.py`` and reference it with ``fix_text``; the record is
    then linted for every property the loose constant was never checked
    against.
    """
    offenders = _offenders_in(ANALYZERS / module)
    assert not offenders, (
        "fix prose defined outside the catalog — move it to "
        "core/fixes/registry.py and read it back with fix_text():\n  "
        + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("module", [p.name for p in _CARD_BUILDERS])
def test_no_card_builder_defines_its_own_fix_prose(module):
    """The same rule where the cards are actually assembled.

    An analyzer produces a finding; these modules turn it into the words a user
    reads and the block a user pastes. A rule authored here is exactly as
    unlinted as one authored in an analyzer, and harder to notice, because the
    surrounding code is legitimately full of per-row evidence prose.
    """
    path = next(p for p in _CARD_BUILDERS if p.name == module)
    offenders = _offenders_in(path)
    assert not offenders, (
        "fix prose defined outside the catalog — move it to "
        "core/fixes/registry.py and read it back with fix_text():\n  "
        + "\n  ".join(offenders)
    )


def test_the_guard_actually_catches_a_reintroduced_constant():
    """The guard's own regression test.

    A structural check that cannot fail is indistinguishable from one that
    passes, and this one is easy to write in a way that never fires — every
    predicate in it is a judgement about what counts as fix prose. So it is
    pointed at a module that reintroduces the defect, and must report it.
    """
    source = '''
FIX_TEMPLATE = (
    "Right-size the workers you dispatch: default every one of them to the "
    "cheapest same-family model that fits the shape of its task."
)
'''
    tmp = ANALYZERS.parent / "_guard_probe.py"
    tmp.write_text(source, encoding="utf-8")
    try:
        assert _offenders_in(tmp), "the guard cannot see a reintroduced constant"
    finally:
        tmp.unlink()


def test_a_grounded_sentence_is_not_mistaken_for_a_rule():
    """The other half, and the more important one to get right.

    A false positive here is worse than a gap: it teaches the next author to
    reach for an exception rather than to move the text, and an exception in a
    structural guard is permanent. A sentence built around this row's evidence
    is grounding — it belongs at the render site, and it must pass.
    """
    source = '''
def build(server, sessions):
    fix = (
        f"Remove the `{server.name}` MCP server ({server.source}); "
        f"zero tool calls across {sessions} session(s) in this window."
    )
    return fix
'''
    tmp = ANALYZERS.parent / "_guard_probe_grounded.py"
    tmp.write_text(source, encoding="utf-8")
    try:
        assert _offenders_in(tmp) == []
    finally:
        tmp.unlink()
