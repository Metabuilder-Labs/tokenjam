"""Switching persona on Optimize must pass THROUGH not-yet-known, never around it.

The persona picker is a server-side scope: ``/optimize``, ``/cost/components``
and ``/relearn/cost-proposals`` all take it, so every figure on this page is an
answer *about one persona*. ``OptimizeView`` held its last answer across every
reload (``loading: !s.opt``) and carried no generation counter, which cost it
both halves of the contract ``useTriageRead`` and ``DashboardView``'s
recoverable-waste read already keep:

* the previous persona's analyzer rows, dollar figures and gated-analyzer set
  stayed rendered under the NEW persona's label until the reads landed. A held
  answer is stale truth only while the question is unchanged; across a switch it
  is a figure attributed to the wrong persona, which is the defect this whole
  surface exists to prevent (root anti-pattern 22a).
* nothing dropped a settle whose generation had been superseded, so a response
  for the persona the reader had already switched away from could write itself
  into the current view.

The two decisions are pure functions in the served page (``optEmptyState`` /
``optPendingState``, keyed by ``optQuestionKey``) precisely so they can be run
under node here rather than string-matched -- the extraction trick
``test_lens_dashboard_states.py`` and ``test_lens_select_all_behaviour.py`` use.
A source match alone would still pass if the state logic were inverted, and
"inverted" here means publishing one persona's money under another's name.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

_UI = Path(__file__).parent.parent.parent / "tokenjam" / "ui" / "index.html"


@pytest.fixture(scope="module")
def html() -> str:
    return _UI.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def optimize_view(html: str) -> str:
    """Just ``OptimizeView``'s body, so assertions cannot pass on another view."""
    start = html.index("function OptimizeView(")
    # Bounded by the next TOP-LEVEL declaration (column 0), so the slice cannot
    # spill into a neighbouring component and let an assertion pass on its code.
    nxt = re.search(r"\n(?:function|const|class) ", html[start + 10:])
    assert nxt, "no top-level declaration follows OptimizeView; update this extractor"
    return html[start:start + 10 + nxt.start()]


# --------------------------------------------------------------------------- #
# Behavioural: the pending-state deciders, run under node
# --------------------------------------------------------------------------- #
_node = pytest.mark.skipif(
    shutil.which("node") is None, reason="node not available for JS evaluation"
)


def _deciders_source() -> str:
    src = _UI.read_text(encoding="utf-8")
    start = src.index("function optEmptyState()")
    end = src.index("function OptimizeSkeleton", start)
    slice_ = src[start:end]
    assert "function optQuestionKey" in slice_, "optQuestionKey moved; update this extractor"
    assert "function optPendingState" in slice_, "optPendingState moved; update this extractor"
    return slice_


def _run_js(expr: str):
    script = _deciders_source() + "\nconsole.log(JSON.stringify(" + expr + "));"
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True, text=True, check=True,
    )
    return json.loads(proc.stdout.strip())


# A populated answer for the persona the reader is about to leave. Every value
# here is something that must NOT survive onto the next persona's screen.
_HELD = {
    "loading": False,
    "error": None,
    "opt": {"findings": {"downsize": {"savings_usd": 412.5}}, "report_available": True},
    "cmp": {"cost_delta_usd": -18.0},
    "comp": {"total_cost_usd": 900.0, "total_recoverable_usd": 412.5},
    "bk": [{"path": "CLAUDE.md", "est_tokens_saved": 1200}],
    "proposals": [{"analyzer": "downsize", "signature": "sig"}],
    "proposalsScoped": True,
}


@_node
def test_empty_state_states_no_figure_and_no_absence():
    """The not-yet-known state carries no answer of any kind.

    Zero is the worst possible placeholder here: "no waste" reads as
    reassurance, and a reader has no way to tell it from a measured zero.
    """
    st = _run_js("optEmptyState()")
    assert st["loading"] is True
    assert st["error"] is None
    # `rec` was a held figure here too, until the recommendation-outcome panel
    # was removed for mixing measured with never-measurable kinds. The field
    # went with its only consumer, so there is no longer a `rec` to drop.
    for field in ("opt", "cmp", "comp"):
        assert st[field] is None, f"{field} must be null while unknown, got {st[field]!r}"
    # `bk` is a list only because the block that reads it is behind the same
    # gate; it must at least be empty rather than a previous persona's rows.
    assert st["bk"] == []


@_node
def test_persona_switch_drops_every_held_figure():
    st = _run_js("optPendingState(%s, true)" % json.dumps(_HELD))
    assert st["loading"] is True, "a switch must render as not-yet-known"
    for field in ("opt", "cmp", "comp"):
        assert st[field] is None, (
            f"{field} survived a persona switch: the previous persona's figures "
            f"would render under the new persona's label"
        )
    assert st["bk"] == []
    assert st["proposals"] == []


@_node
def test_same_question_keeps_the_held_answer():
    """A poll tick / Refresh / apply-a-fix reload must not blank the page."""
    st = _run_js("optPendingState(%s, false)" % json.dumps(_HELD))
    assert st["opt"] is not None and st["opt"]["findings"]["downsize"]["savings_usd"] == 412.5
    assert st["comp"]["total_cost_usd"] == 900.0
    assert st["loading"] is False, "a refresh over a held answer is not a skeleton"
    assert st["error"] is None


@_node
def test_first_load_with_no_held_answer_is_loading():
    st = _run_js("optPendingState(optEmptyState(), false)")
    assert st["loading"] is True
    assert st["opt"] is None


@_node
def test_question_key_moves_with_persona_and_with_every_other_scope():
    base = _run_js("optQuestionKey('30d', '', '', 'claude-code')")
    same = _run_js("optQuestionKey('30d', '', '', 'claude-code')")
    assert base == same

    for label, expr in (
        ("persona", "optQuestionKey('30d', '', '', 'sdk')"),
        ("since", "optQuestionKey('7d', '', '', 'claude-code')"),
        ("agent_id", "optQuestionKey('30d', 'agent-1', '', 'claude-code')"),
        ("compare", "optQuestionKey('30d', '', 'previous', 'claude-code')"),
    ):
        assert _run_js(expr) != base, f"{label} does not change the question key"


# --------------------------------------------------------------------------- #
# Structural: the generation guard, which has no pure-function form
# --------------------------------------------------------------------------- #
def test_optimize_read_drops_superseded_settles(optimize_view: str):
    """A settle for a superseded question may not write into the current view.

    Both the success path and the failure path need the guard: an error from the
    persona the reader has left would otherwise put an error banner over a view
    whose own read is fine.
    """
    guards = optimize_view.count("if (runRef.current !== g) return;")
    assert guards >= 2, (
        "both the resolve and the reject path of OptimizeView's load must drop a "
        f"settle from a superseded generation; found {guards} guard(s)"
    )
    assert "const g = ++runRef.current;" in optimize_view
    assert "return () => { runRef.current++; };" in optimize_view, (
        "the effect cleanup must invalidate its generation, so the read this "
        "effect asked cannot write into the question that replaced it"
    )


def test_optimize_pending_state_goes_through_the_shared_decider(optimize_view: str):
    """No second, drifting copy of the stale-vs-skeleton rule in the component."""
    assert "setSt(s => optPendingState(s, changed));" in optimize_view
    # Comment lines are excluded: the header above `load` names the old
    # expression to record WHY it went, which is the opposite of reintroducing it.
    code = "\n".join(
        ln for ln in optimize_view.splitlines() if not ln.lstrip().startswith("//")
    )
    assert "loading: !s.opt" not in code, (
        "the raw held-answer expression is the bug: it kept the previous "
        "persona's figures across a switch. It belongs in optPendingState, "
        "which distinguishes a changed question from a refresh"
    )


def test_optimize_skeleton_reserves_the_page(optimize_view: str, html: str):
    """A switch resolves into roughly the space the skeleton held."""
    assert "<${OptimizeSkeleton}" in optimize_view
    assert 'style="height:200px"' not in optimize_view, (
        "a single 200px shimmer standing in for a screen several times taller "
        "yanks the scroll position on every resolution"
    )
    body = html[html.index("function OptimizeSkeleton"):]
    body = body[: body.index("function OptimizeView")]
    assert "opt-cards" in body, "the skeleton should mirror the layout it replaces"


def test_scanbar_distinguishes_unresolved_from_failed(html: str):
    """"scan state unavailable" is a failure claim; an in-flight read is not one.

    ``scanState`` derives everything from the payload, so a null payload cannot
    tell the two apart there -- which is why the caller passes ``loading``.
    """
    bar = html[html.index("function ScanBar("):]
    bar = bar[: bar.index("\n// A rescan button")]
    assert "loading = false" in bar, "ScanBar must default to its previous behaviour"
    assert "checking scan state" in bar
    assert "scan state unavailable" in bar, (
        "the genuine fetch-failed state must stay distinct, not be folded into "
        "the loading state"
    )
    idx_loading = bar.index("loading ? 'checking scan state")
    idx_failed = bar.index("scan.fetchFailed ? 'scan state unavailable'")
    assert idx_loading < idx_failed, "loading must win over fetchFailed, not the reverse"


def test_optimize_passes_its_unresolved_state_to_the_scanbar(optimize_view: str):
    assert "loading=${st.loading && !st.opt}" in optimize_view


def test_gated_analyzers_get_no_chip_but_keep_their_page(optimize_view: str, html: str):
    """Withdrawing the offer is not the same as denying the page exists.

    Operator decision: the breadcrumb strip on an analyzer detail page renders
    only analyzers that RUN for the selected persona. It used to render all of
    them with the gated ones dimmed and clickable.

    The half that must NOT change with it: ``#/optimize/<gated>`` still routes
    and still lands on the persona-disabled explanation, so a deep link, a
    bookmark, or a link from a screen viewed under the other persona keeps
    working and says why.
    """
    crumbs = optimize_view[optimize_view.index("const crumbs = html`"):]
    crumbs = crumbs[: crumbs.index("</div>`;")]
    assert "${order.map(n => {" in crumbs, (
        "the strip must iterate the persona-filtered list, not every analyzer"
    )
    assert "OPT_DETAIL_ORDER.map" not in crumbs
    assert "gated" not in crumbs, "no gated chip, so no gated branch in the chip"
    # `order` is the SAME filtered list the summary's row list uses, so the two
    # surfaces cannot offer different analyzer sets.
    assert "const order = OPT_DETAIL_ORDER.filter(n => !personaGated.has(n));" in optimize_view

    # The page survives the chip.
    assert "if (personaGated.has(detailName)) {" in optimize_view, (
        "the deep-linked persona-disabled page must still render"
    )
    gated_page = optimize_view[optimize_view.index("if (personaGated.has(detailName)) {"):]
    assert "opt-persona-gated" in gated_page[:1200]
    assert "${crumbs}" in gated_page[:1200], "and it still carries the strip"

    # The dimmed-chip style goes with the behaviour it styled (root anti-pattern
    # 23: the prose arguing for the old rule must not outlive it either).
    assert ".opt-crumb-chip.gated {" not in html
    assert "distinct, never hidden" not in html, (
        "the comment arguing gated chips must never be hidden describes "
        "behaviour the code no longer has"
    )


def test_the_optimize_submenu_has_a_not_yet_known_state(html: str):
    """The sidebar's analyzer children are a persona-specific set too.

    Switching persona left the previous persona's analyzer entries ("Batch
    placement", "Cache") in the sidebar under the new persona's Optimize
    section until the read landed. Removing them instead would assert the
    opposite falsehood, so they render as unsettled: still clickable (each
    detail page has its own persona-gated state), visibly not an answer.
    """
    app = html[html.index("const [optNav, setOptNav]"):]
    app = app[: app.index("useEffect(() => {\n    const handler = () => setRoute(getRoute());")]
    assert "showRules: false, pending: true }" in app or "pending: true }" in app
    assert "setOptNav(s => (s.pending ? s : { ...s, pending: true }));" in app, (
        "a persona change must put the submenu back into not-yet-known"
    )
    assert "optNav.pending ? ' nav-child-pending' : ''" in app
    assert "a.setAttribute('aria-busy', 'true')" in app
    # Every settle path must clear it, or the submenu stays dim forever.
    assert app.count("pending: false") >= 3, (
        "the success, the empty-payload and the failure paths each have to "
        f"resolve the pending state; found {app.count('pending: false')}"
    )
    assert ".sidebar a.nav-child.nav-child-pending" in html, "the dim style must exist"
