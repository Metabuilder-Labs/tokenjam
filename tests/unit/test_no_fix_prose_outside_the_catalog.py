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
per-row.

**Three ways prose escaped this guard, all of them silent.** The guard read
green over a ~280-character multi-sentence advisory paragraph hardcoded in
``cost_proposals._summarize_to_proposals``, and it took all three to let that
through:

1. ``advise_text`` — the slot that paragraph is assigned to — was not in
   ``_FIX_SLOTS`` at all. A guard scoped by slot name sees nothing outside the
   slots it lists, and a missing slot is indistinguishable from a clean one.
2. The constant-name check matched the token ``ADVICE`` while the variable was
   named ``advise``. ADVISE vs ADVICE, one letter, and the substring test
   returned False forever. That is why this file no longer GUESSES at names:
   it traces what actually flows into a fix slot, so a near-miss spelling
   cannot get anything past it.
3. The length floor was borrowed from ``MIN_FIX_CHARS``, which answers a
   different question (is a CATALOGUED text long enough to be an instruction).
   Prose written as short interpolated fragments was therefore automatically
   exempt however much of it there was.

**Why the floor could not simply be lowered.** Length alone cannot separate the
two cases: the grounded MCP-removal sentence's longest literal run and the
shortest rule fragment are the same size, so any single threshold either misses
the rule or condemns the grounding. The second detector asks a different
question instead — total literal volume paired with SENTENCE COUNT. Grounding
is one sentence about one row; a policy runs to several, however many holes are
punched in it.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from tokenjam.core.fixes.lint import (
    MIN_GUARD_SENTENCES,
    MIN_GUARD_STATIC_RUN,
    MIN_GUARD_STATIC_TOTAL,
)

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

#: EVERY module under `tokenjam/`, derived by walking the tree.
#:
#: There is no root list any more, because a hand-maintained root list is how
#: every gap in this guard arose. `core/summarize/` was outside it — a whole
#: package of user-facing route advice nothing checked. So were
#: `core/agent_config.py` and `core/optimize/mcp_probe.py`, added by this very
#: change. Each omission was invisible in the same way: the guard reported green
#: over files it never opened, and nobody can tell that from green over files it
#: read and cleared.
#:
#: Walking the tree costs one AST parse per module and removes the failure mode
#: outright. A module added tomorrow, anywhere, is covered without anyone
#: remembering anything.
def _every_module() -> list[Path]:
    return sorted(
        p for p in ROOT.rglob("*.py")
        if p.name != "__init__.py"
    )

#: Names whose value IS the fix a user is shown or writes into a file. Slot
#: names rather than a guess at prose: what makes a string a fix is where it
#: goes, not how it reads.
#:
#: ``advise_text`` is the one whose absence let a multi-sentence advisory
#: paragraph live outside the catalog while this file reported 23 passes.
_FIX_SLOTS = frozenset({
    "fix", "proposed_fix", "one_paste_fix", "artifact_text",
    "remedy_snippet", "fix_template", "suggestion",
    "advise_text", "advice_text", "advice", "recommendation", "guidance",
})

#: The only calls that may produce fix text. Reaching the catalog through any
#: of these means the text is linted; anything else means it is not.
_CATALOG_CALLS = frozenset({"fix_text", "fix_text_for", "ground", "compound_offload_fix"})

_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")


def _names_a_slot(name: str) -> bool:
    """Whether an identifier IS one of the fix slots, compared exactly.

    THE fix for the second escape. The old check asked whether an identifier
    CONTAINED any of ``("FIX", "RUBRIC", "REMEDY", "LEVER", "RULE_TEXT",
    "SNIPPET", "ADVICE")``, and a variable called ``advise`` defeated it
    permanently — ADVISE against ADVICE, one letter, no error, no warning, and
    the paragraph behind it shipped past a guard reporting 23 passes.

    Substring guessing fails in the one direction that cannot be noticed: a name
    the guess does not cover is invisible, and invisible is indistinguishable
    from clean. So there is no guessing left. A name matches only by BEING a
    slot (case- and underscore-insensitively, so ``FIX_TEMPLATE`` and
    ``fix_template`` are one name), and everything else is caught by
    :func:`_fix_carrying_locals` tracing where its value actually goes — which
    holds however the name is spelled.
    """
    return name.lower().strip("_") in _FIX_SLOTS


def _fix_carrying_locals(tree: ast.AST) -> tuple[set[str], set[str]]:
    """Local names whose value ends up in a fix slot, found by TRACING.

    This replaces a substring test over constant names, which is the check that
    missed ``advise`` because it was looking for ``ADVICE``. Guessing at names
    fails silently and in the one direction that matters: a name the guess does
    not cover is invisible, and invisible is indistinguishable from clean.

    So the assignment TARGET SET is derived from where values actually go. Any
    bare ``Name`` handed to a fix slot — as a keyword argument, a dict value, or
    assigned into one — makes that name fix-carrying, and its own definition is
    then checked. Transitive, because one rename of an intermediate would
    otherwise reopen the hole.
    """
    carriers: set[str] = set()
    producers: set[str] = set()
    for _ in range(4):  # transitive closure; the real depth is 1-2
        before = len(carriers) + len(producers)

        def _note(value: ast.AST) -> None:
            """Every name and every producer function this value draws text from.

            Walks the WHOLE subtree rather than switching on a handful of node
            types. The type-switch version handled ``Name``/``BinOp``/
            ``JoinedStr``/``IfExp`` and therefore never looked inside a
            ``Call`` — so a helper (``fix = _compose(row)``), a template
            constant (``fix = _ADVICE.format(...)``) and a join
            (``fix = "".join(parts)``) all passed unread, while ``%``-formatting
            was caught purely because ``%`` happens to be a ``BinOp``. That is
            an arbitrary boundary, not a designed one.

            Over-inclusive on purpose, and safely so: this only ever runs on an
            expression whose value REACHES a fix slot, so everything inside it
            genuinely contributes to the text a user is shown.
            """
            for child in ast.walk(value):
                if isinstance(child, ast.Name):
                    carriers.add(child.id)
                elif isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                    # `_compose(row)` — the callee's own returns are fix text.
                    producers.add(child.func.id)
                elif isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                    # `_ADVICE.format(...)`, `"".join(parts)` — the receiver
                    # holds the template; its args hold the pieces. `ast.walk`
                    # above already reaches both, this branch is here so a
                    # method call on a plain constant is not mistaken for a
                    # producer function.
                    pass

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg in _FIX_SLOTS:
                        _note(kw.value)
            elif isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if (
                        isinstance(key, ast.Constant)
                        and isinstance(key.value, str)
                        and key.value in _FIX_SLOTS
                    ):
                        _note(value)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and (
                        target.id in _FIX_SLOTS or target.id in carriers
                    ):
                        _note(node.value)
        # A producer function's RETURN values are fix text too, so they feed
        # the next round the same way an assignment does.
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in producers:
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Return) and inner.value is not None:
                        _note(inner.value)
        if len(carriers) + len(producers) == before:
            break
    return carriers, producers


def _is_catalog_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
    return name in _CATALOG_CALLS


def _static_text(node: ast.AST) -> tuple[int, str]:
    """``(longest contiguous literal run, all literal text joined)`` — EXCLUDING
    anything inside a catalog call.

    Both measures are needed: the run catches a rule quoted whole, and the joined
    text is what the second detector counts sentences in, which is what makes
    prose stitched around interpolated evidence visible at all.

    The exclusion is the fix for the widest escape this guard had. It used to
    ask "does a catalog call appear anywhere in this subtree?" and, if so, return
    before computing any verdict — so ONE ``fix_text(...)`` whitelisted unlimited
    adjacent hardcoded prose, and ``fix = ("<250 chars of policy> " +
    fix_text("a.b"))`` passed clean. This PR introduces exactly that shape in two
    places, legitimately and briefly, and the old rule bounded that half not at
    all. Skipping only the catalog call's own subtree measures what the author
    actually wrote and leaves what the catalog supplied out of it.
    """
    longest = 0
    parts: list[str] = []
    stack: list[ast.AST] = [node]
    while stack:
        current = stack.pop()
        if current is not node and _is_catalog_call(current):
            continue  # catalogued text is linted where it lives
        if isinstance(current, ast.Constant) and isinstance(current.value, str):
            stripped = current.value.strip()
            longest = max(longest, len(stripped))
            parts.append(current.value)
        stack.extend(ast.iter_child_nodes(current))
    return longest, "".join(parts)


def _sentences(text: str) -> int:
    return len([s for s in _SENTENCE_END.split(text) if s.strip()])


def _verdict(node: ast.AST) -> str:
    """Why this value is fix prose, or ``""`` when it is grounding.

    Two independent detectors, because one threshold cannot answer both. See
    the module docstring.
    """
    run, joined = _static_text(node)
    if run >= MIN_GUARD_STATIC_RUN:
        return f"{run} characters of hardcoded fix prose in one contiguous run"
    total = len(joined.strip())
    if total >= MIN_GUARD_STATIC_TOTAL:
        sentences = _sentences(joined)
        if sentences >= MIN_GUARD_SENTENCES:
            return (
                f"{total} characters of hardcoded fix prose across "
                f"{sentences} sentences, stitched around interpolated evidence"
            )
    return ""


def _offenders_in(path: Path) -> list[dict]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    carriers, producers = _fix_carrying_locals(tree)
    out: list[dict] = []
    module = path.relative_to(ROOT).as_posix() if ROOT in path.parents else path.name

    def _is_fix_slot(node: ast.AST) -> bool:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value in _FIX_SLOTS                   # {"fix": ...}
        if isinstance(node, ast.Name):
            return _names_a_slot(node.id) or node.id in carriers
        return False

    def check(slot: str, value: ast.AST) -> None:  # noqa: ANN202
        # No early return for "a catalog call appears somewhere in here" — that
        # was the escape. `_static_text` skips the catalog call's own subtree
        # and measures the rest, so composing catalogued text with a short
        # grounded sentence passes while composing it with a policy paragraph
        # does not.
        verdict = _verdict(value)
        if not verdict:
            return
        run, joined = _static_text(value)
        out.append({
            "module": module,
            "slot": slot,
            "lineno": getattr(value, "lineno", 0),
            "run": run,
            "total": len(joined.strip()),
            "sentences": _sentences(joined),
            "verdict": verdict,
        })

    # A producer function's returns ARE the fix text — check them where they
    # are written, so a helper cannot launder prose out of the guard's sight.
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in producers:
            for inner in ast.walk(node):
                if isinstance(inner, ast.Return) and inner.value is not None:
                    check(f"{node.name}() return", inner.value)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            # AugAssign (`fix += "..."`) was never inspected at all, so a rule
            # appended to a grounded sentence was invisible however long it ran.
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
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


#: Uncatalogued fix prose that EXISTS TODAY, keyed by a fingerprint of each
#: offending text rather than by a count.
#:
#: A RATCHET, not an allowlist, and the difference is the whole point. Closing
#: the guard's escapes made this prose visible for the first time — it was
#: always there; the guard was structurally incapable of reading it, which is
#: the worst possible state for a check to be in. Migrating it changes what a
#: dozen cards say and interacts with the catalog's duplicate lint, so it
#: belongs in its own review.
#:
#: Keyed on CONTENT, not on how many. A count-based pin (`len(offenders) == 18`)
#: passes when one uncatalogued text is swapped for a different one, which is
#: exactly the drift a ratchet exists to stop. The fingerprint below changes if
#: any text changes, so adding one fails, editing one fails, and removing one
#: fails until this set comes down with it. It can only be driven to empty.
#:
#: `core/summarize/route.py`'s entries are a deliberate deferral rather than an
#: oversight: that file's advisory prose is (a) verbatim quotation of Anthropic's
#: published guidance, which a `FixRecord` would subject to lints written for
#: text we author, and (b) diagnosis of a measured prose SHAPE rather than an
#: instruction. Both may belong in the catalog eventually; neither is a
#: mechanical move.
_UNCATALOGUED: dict[str, set[str]] = {}

#: One line per uncatalogued text: `<module>:<slot>:<longest run>:<total
#: static chars>:<sentences>`. Generated, then committed — see _UNCATALOGUED.
_RATCHET_RAW = """
core/context_diagnostic.py:_fix_for() return:130:130:1
core/context_diagnostic.py:_fix_for() return:161:161:1
core/context_diagnostic.py:_fix_for() return:62:62:1
core/context_diagnostic.py:_fix_for() return:79:94:2
core/context_diagnostic.py:_fix_for() return:94:126:1
core/optimize/analyzers/relearn.py:prompt:305:383:5
core/optimize/cost_proposals.py:MODEL_SWAP_QUALITY_CAVEAT:183:183:2
core/optimize/cost_proposals.py:_driver_role_advice() return:63:63:1
core/optimize/cost_proposals.py:advise:103:190:1
core/optimize/cost_proposals.py:advise:106:113:2
core/optimize/cost_proposals.py:advise:133:133:1
core/optimize/cost_proposals.py:advise:134:278:2
core/optimize/cost_proposals.py:advise:165:255:2
core/optimize/cost_proposals.py:advise:172:239:3
core/optimize/cost_proposals.py:advise:180:180:2
core/optimize/cost_proposals.py:advise:205:205:2
core/optimize/cost_proposals.py:advise:44:44:1
core/optimize/cost_proposals.py:advise:55:108:1
core/optimize/cost_proposals.py:advise:68:212:2
core/optimize/cost_proposals.py:advise:77:141:1
core/optimize/cost_proposals.py:advise:88:211:2
core/optimize/cost_proposals.py:advise_extra:162:253:3
core/optimize/cost_proposals.py:advise_extra:360:485:3
core/optimize/cost_proposals.py:advise_text=:190:196:2
core/optimize/cost_proposals.py:advise_text=:53:59:1
core/optimize/cost_proposals.py:one_paste:59:94:2
core/optimize/cost_proposals.py:reversible_note:133:250:2
core/summarize/route.py:BEST_PRACTICES_SOURCE:46:46:1
core/summarize/route.py:HOOK_QUOTE:107:107:1
core/summarize/route.py:PATH_SCOPE_QUOTE:126:126:1
core/summarize/route.py:PRUNE_EXCLUDE_QUOTE:285:285:1
core/summarize/route.py:PRUNE_TEST_QUOTE:88:88:1
core/summarize/route.py:SPECIFICITY_QUOTE:91:91:1
core/summarize/route.py:_ALREADY_SCOPED:154:154:1
core/summarize/route.py:_NOT_INSTRUCTION:228:228:2
core/summarize/route.py:_PRUNE_ADVICE:370:370:2
core/summarize/route.py:_QUALITY_TAX:520:520:4
core/summarize/route.py:_SCOPE_ADVICE:334:334:2
core/summarize/route.py:_TARGET_STATEMENT:499:499:4
mcp/server.py:"recommendation":166:166:2
"""




def _fingerprint(offender: dict) -> str:
    """Identity of one offending text, independent of where it sits.

    Deliberately excludes the line number: prose that merely moved is the same
    prose, and a ratchet that fires on every unrelated edit above it teaches
    people to re-baseline it without reading.
    """
    return (
        f"{offender['module']}:{offender['slot']}:{offender['run']}:"
        f"{offender['total']}:{offender['sentences']}"
    )


def _load_ratchet() -> dict[str, set[str]]:
    if _UNCATALOGUED:
        return _UNCATALOGUED
    for line in _RATCHET_RAW.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        module = line.split(":", 1)[0]
        _UNCATALOGUED.setdefault(module, set()).add(line)
    return _UNCATALOGUED


def test_no_module_defines_fix_prose_outside_the_catalog():
    """THE guard, over EVERY module under `tokenjam/`.

    One derived scan replaces three hand-listed root sets. Failing this means
    either a policy has acquired a second home — move it to
    ``core/fixes/registry.py`` and read it back with ``fix_text`` — or an
    existing one moved into the catalog and the ratchet below has not caught up
    with it. The message says which.
    """
    ratchet = _load_ratchet()
    found: dict[str, set[str]] = {}
    detail: dict[str, list[str]] = {}
    for path in _every_module():
        offenders = _offenders_in(path)
        for offender in offenders:
            key = _fingerprint(offender)
            found.setdefault(offender["module"], set()).add(key)
            detail.setdefault(offender["module"], []).append(
                f"{offender['module']}:{offender['lineno']}: {offender['slot']} "
                f"carries {offender['verdict']}",
            )

    new = {m: sorted(v - ratchet.get(m, set())) for m, v in found.items()}
    new = {m: v for m, v in new.items() if v}
    gone = {m: sorted(v - found.get(m, set())) for m, v in ratchet.items()}
    gone = {m: v for m, v in gone.items() if v}

    assert not new, (
        "NEW fix prose defined outside the catalog — move it to "
        "core/fixes/registry.py and read it back with fix_text():\n  "
        + "\n  ".join(
            line for module in new for line in detail.get(module, [])
        )
    )
    assert not gone, (
        "uncatalogued prose recorded in the ratchet is no longer there. If you "
        "moved it into the catalog, delete its line from _RATCHET_RAW so the "
        "gap that is LEFT stays honest:\n  "
        + "\n  ".join(f"{m}: {k}" for m, ks in gone.items() for k in ks)
    )


def test_the_guard_catches_prose_assigned_to_an_advise_slot():
    """THE regression test for the escape this guard was green over.

    Reproduces the exact shape: a multi-sentence advisory paragraph built in a
    local named ``advise``, then handed to ``advise_text=``. Before the fix all
    three escapes had to hold at once — the slot was unlisted, the constant-name
    check looked for ADVICE while the name was ADVISE, and the floor was
    borrowed from a different question. Any one of them being closed catches
    this, which is why all three were closed rather than the cheapest one.
    """
    source = '''
def build(files, plural):
    advise = (
        f"Review {files} oversized file{plural} in the summarize curate -> "
        "diff -> apply surface (`tj summarize list` / `tj summarize check` / "
        "`tj summarize apply`, or the Summarize screen in the web UI). This "
        "card links there instead of applying inline: the fix is a reviewed "
        "rewrite - structure kept verbatim, prose compressed, one file at a "
        "time - not a one-click removal."
    )
    return CostProposal(advise_text=advise)
'''
    tmp = ANALYZERS.parent / "_guard_probe_advise.py"
    tmp.write_text(source, encoding="utf-8")
    try:
        offenders = _offenders_in(tmp)
        assert offenders, "the guard still cannot see prose behind an advise slot"
        assert offenders[0]["slot"] == "advise"
    finally:
        tmp.unlink()


def test_the_guard_catches_a_rule_stitched_out_of_short_fragments():
    """The length floor's own regression test.

    No fragment here reaches the single-run floor, so the old guard exempted it
    automatically. It is still a policy: several sentences of durable
    instruction with a couple of interpolations dropped in.
    """
    source = '''
def build(name, n):
    fix = (
        f"Remove `{name}` first. "
        f"Then re-run the scan, {n} times if needed. "
        "Do not re-add it without a reason. "
        "Record that reason in the file itself. "
        "Prefer scoping over deleting when unsure."
    )
    return fix
'''
    tmp = ANALYZERS.parent / "_guard_probe_fragments.py"
    tmp.write_text(source, encoding="utf-8")
    try:
        assert _offenders_in(tmp), "short fragments are still automatically exempt"
    finally:
        tmp.unlink()


def test_the_guard_catches_prose_appended_with_augmented_assignment():
    """``fix += "..."`` was never inspected at all."""
    source = '''
def build(name):
    fix = f"Remove `{name}`."
    fix += (
        "Removing only this location stops the tax for the sessions counted "
        "here. The remaining sessions need their own location edited too."
    )
    return fix
'''
    tmp = ANALYZERS.parent / "_guard_probe_augassign.py"
    tmp.write_text(source, encoding="utf-8")
    try:
        assert _offenders_in(tmp), "an appended rule is still invisible"
    finally:
        tmp.unlink()


def test_a_near_miss_constant_spelling_cannot_escape():
    """The ADVISE/ADVICE hole, pinned.

    The old check asked whether a name CONTAINED one of a handful of guessed
    tokens. One letter defeated it, permanently and silently. The replacement
    traces what reaches a fix slot, so the name can be anything at all.
    """
    source = '''
def build():
    wibble = (
        "Right-size the workers you dispatch: default every one of them to "
        "the cheapest same-family model that fits the shape of its task."
    )
    return CostProposal(advise_text=wibble)
'''
    tmp = ANALYZERS.parent / "_guard_probe_wibble.py"
    tmp.write_text(source, encoding="utf-8")
    try:
        assert _offenders_in(tmp), "a name the guard did not guess still escapes"
    finally:
        tmp.unlink()


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
