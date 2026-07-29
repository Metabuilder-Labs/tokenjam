"""The refresh control must not assert more than it knows.

Three claims live in one control (``ScanBar`` plus ``useRescan``), and two of
them could be wrong:

1. **"Scanning…" tracked the WRONG job.** One analyzer cycle refreshes three
   stores sequentially on one thread — report, then relearn, then the cost
   proposals — and each store owns an INDEPENDENT in-flight flag. The control
   read the report's alone, so the moment the report landed it stopped saying
   "Scanning…" and started saying "computed just now" while relearn (the most
   expensive analyzer in the product) and the proposals behind the Review inbox
   were still being built. Seconds on a warm distill cache; minutes on a cold
   one.

2. **A declined rescan was indistinguishable from a successful one.**
   ``POST /optimize/rescan`` answers HTTP 200 with ``started: false`` when a
   guard refuses. ``useRescan`` surfaced only THROWN errors, so the refusal
   resolved normally: spinner, re-enable, reload, identical numbers.

3. **The provenance line vouched for something it did not know.**
   ``computed <age> ago`` answers HOW OLD and readers take it for WHICH BUILD.
   These stores are caches and an upgrade does not invalidate them.

The deciders are pure functions in the served ``index.html`` so they can be run
here rather than grepped for — a string match would still pass if the logic were
inverted, and "inverted" here means telling a user a scan finished when it is
still running. Same extract-and-run-under-node trick
``test_lens_dashboard_states.py`` uses.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_UI = Path(__file__).parent.parent.parent / "tokenjam" / "ui" / "index.html"

_node = pytest.mark.skipif(
    shutil.which("node") is None, reason="node not available for JS evaluation"
)


@pytest.fixture(scope="module")
def html() -> str:
    return _UI.read_text(encoding="utf-8")


def _fn_source(src: str, name: str) -> str:
    """One top-level function, lifted verbatim out of the served page.

    Brace-free slicing on purpose: these are top-level declarations, so the
    first line that is exactly ``}`` closes them. Asserts rather than returns
    empty, so a rename here fails as "the extractor moved" instead of silently
    testing nothing.
    """
    marker = f"function {name}("
    assert marker in src, f"{name} moved or was renamed; update this extractor"
    start = src.index(marker)
    end = src.index("\n}\n", start) + len("\n}\n")
    return src[start:end]


def _run_js(src: str, expr: str):
    script = src + "\nconsole.log(JSON.stringify(" + expr + "));"
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True, text=True, check=True,
    )
    return json.loads(proc.stdout.strip())


# --------------------------------------------------------------------------- #
# 1. "Scanning…" is the CYCLE, not one store
# --------------------------------------------------------------------------- #
@_node
def test_scanning_stays_true_while_a_sibling_store_is_still_building(html: str):
    """The report's leg has landed; relearn and the cost proposals have not.

    This is the exact payload the defect rendered as "computed just now": the
    report store is `ready` with a fresh timestamp, and the pass that wrote it is
    still running. It must read as scanning.
    """
    src = _fn_source(html, "scanState")
    state = _run_js(src, (
        "scanState({status:'ready', computed_at:'2026-07-29T10:00:00Z',"
        " cycle_computing:true, report_available:true})"
    ))
    assert state["computing"] is True


@_node
def test_scanning_is_false_once_the_whole_cycle_has_landed(html: str):
    src = _fn_source(html, "scanState")
    state = _run_js(src, (
        "scanState({status:'ready', computed_at:'2026-07-29T10:00:00Z',"
        " cycle_computing:false, report_available:true})"
    ))
    assert state["computing"] is False


@_node
def test_the_stores_own_flag_still_counts_on_its_own(html: str):
    """A store refreshed by a lone trigger (a test, a direct call) reports
    through its OWN guard and no cycle. The OR has to keep honouring it."""
    src = _fn_source(html, "scanState")
    state = _run_js(src, "scanState({status:'computing', report_available:true})")
    assert state["computing"] is True


@_node
def test_an_older_server_sending_neither_field_still_reads_correctly(html: str):
    src = _fn_source(html, "scanState")
    state = _run_js(src, (
        "scanState({status:'ready', computed_at:'2026-07-29T10:00:00Z',"
        " report_available:true})"
    ))
    assert state["computing"] is False


# --------------------------------------------------------------------------- #
# 2. A declined rescan is not a successful one
# --------------------------------------------------------------------------- #
@_node
def test_declined_rescan_is_surfaced(html: str):
    """`started: false` — the overlap guard refusing, which is what the daemon's
    own startup kick does to an impatient first-run press."""
    src = _fn_source(html, "declinedReason")
    out = _run_js(src, (
        "declinedReason({started:false, reason:'a scan is already running'})"
    ))
    assert out
    assert "not started" in out


@_node
def test_the_declined_notice_cannot_go_stale(html: str):
    """It stays on screen until the next press, so it must describe the PRESS,
    not the world. The server's own reasons are present tense ("a scan IS
    already running") and stop being true the moment that scan lands — rendering
    them verbatim would leave this control asserting a running scan after it
    finished, which is the defect the control is being fixed for.
    """
    src = _fn_source(html, "declinedReason")
    for res in ("{started:false}",
                "{started:false, throttled:true}",
                "{started:false, reason:'a scan is already running'}"):
        out = _run_js(src, f"declinedReason({res})")
        assert out, res
        assert " is " not in out, (
            f"present-tense claim in a notice that outlives its truth: {out!r}"
        )


@_node
def test_declined_distinguishes_a_throttle_from_an_overlap(html: str):
    src = _fn_source(html, "declinedReason")
    throttled = _run_js(src, "declinedReason({started:false, throttled:true})")
    overlap = _run_js(src, "declinedReason({started:false})")
    assert throttled != overlap


@_node
def test_a_started_rescan_is_never_reported_as_declined(html: str):
    src = _fn_source(html, "declinedReason")
    assert _run_js(src, "declinedReason({started:true})") is None
    # No `started` field at all: an older server, or a surface whose action is
    # not this endpoint. Not evidence of a refusal — inventing one would pin a
    # permanent "already running" onto a control that works.
    assert _run_js(src, "declinedReason({})") is None
    assert _run_js(src, "declinedReason(null)") is None


def test_every_scan_bar_call_site_threads_the_declined_state(html: str):
    """The decider is useless if a surface drops it on the way to the control.

    Source-level on purpose: this guards the WIRING, which no pure function can
    observe. Every mounted ScanBar has to receive `declined`, or that surface
    goes back to rendering a refusal as a success.
    """
    mounts = [ln for ln in html.splitlines() if "<${ScanBar}" in ln]
    assert mounts, "ScanBar is no longer mounted anywhere; update this test"
    missing = [ln.strip() for ln in mounts if "declined=" not in ln]
    assert not missing, f"ScanBar mounted without `declined`: {missing}"


def test_the_rescan_hook_does_not_guess_when_the_pass_is_running(html: str):
    """The fixed 1.5s timeout was standing in for "is it running yet" and
    answered it by guessing — it re-enabled the button whether or not a pass was
    in flight. The cycle flag is set before the POST returns, so the reload sees
    it and `scan.computing` owns the indicator."""
    start = html.index("function useRescan(")
    end = html.index("\n}\n", start)
    body = html[start:end]
    assert "1500" not in body, "the fixed rescan timeout is back in useRescan"


# --------------------------------------------------------------------------- #
# 3. The provenance line cannot vouch for a build it does not know
# --------------------------------------------------------------------------- #
@_node
def test_same_build_adds_no_qualifier(html: str):
    """The ordinary case. The timestamp IS the whole truth here, and a warning
    beside every figure on every screen would be noise that trains the user to
    ignore the one time it matters."""
    src = _fn_source(html, "buildQualifier")
    assert _run_js(src, (
        "buildQualifier({computedAt:'2026-07-29T10:00:00Z',"
        " computedBuild:'0.6.3', build:'0.6.3'})"
    )) is None


@_node
def test_a_previous_builds_result_cannot_read_as_merely_recent(html: str):
    """The upgrade case, and the audience is precisely the user who upgraded to
    get a fix and will otherwise conclude it did not work."""
    src = _fn_source(html, "buildQualifier")
    out = _run_js(src, (
        "buildQualifier({computedAt:'2026-07-29T10:00:00Z',"
        " computedBuild:'0.6.2', build:'0.6.3'})"
    ))
    assert out is not None
    assert "0.6.2" in out and "0.6.3" in out


@_node
def test_an_unstamped_result_is_unknown_not_agreement(html: str):
    """A cache written before the stamp existed. "We cannot vouch either way" is
    a different statement from "same build", and rendering it as the latter is
    the whole defect one layer down."""
    src = _fn_source(html, "buildQualifier")
    out = _run_js(src, (
        "buildQualifier({computedAt:'2026-07-29T10:00:00Z',"
        " computedBuild:null, build:'0.6.3'})"
    ))
    assert out is not None
    assert "unknown" in out.lower()


@_node
def test_nothing_computed_yet_has_no_freshness_claim_to_qualify(html: str):
    src = _fn_source(html, "buildQualifier")
    assert _run_js(src, "buildQualifier({computedAt:null, build:'0.6.3'})") is None
    # An older server publishes no `build`, so we have no stance either way and
    # must not manufacture one.
    assert _run_js(src, (
        "buildQualifier({computedAt:'2026-07-29T10:00:00Z', build:null})"
    )) is None


# --------------------------------------------------------------------------- #
# 4. A control that is wrong AND unpressable
# --------------------------------------------------------------------------- #
def test_an_in_flight_pass_is_polled_faster_than_the_idle_cadence(html: str):
    """The Rescan button is disabled for as long as the surface believes a scan
    is running, and the stored read cadence defaults to five minutes. So a pass
    that finished in twenty seconds left the control disabled and asserting
    "Scanning…" for the rest of that window, with no way for the user to act and
    nothing telling them it had landed.

    Source-level because the cadence lives in a hook (`useAutoRefresh`) whose
    body is an effect, not a pure decider. What is pinned is the property that
    matters: the in-flight branch cannot resolve to the idle cadence, and it
    cannot resolve to "never".
    """
    start = html.index("function useAutoRefresh(")
    body = html[start:html.index("\n}\n", start)]
    assert "scan.computing" in body, (
        "useAutoRefresh no longer distinguishes an in-flight pass"
    )
    assert "SCAN_INFLIGHT_POLL_SECONDS" in body
    # `Math.min` against the constant is what makes the in-flight cadence an
    # upper bound rather than a replacement: a surface configured to poll FASTER
    # than the in-flight floor keeps its own cadence.
    assert "Math.min" in body


def test_the_in_flight_cadence_is_short_enough_to_unstick_the_button(html: str):
    marker = "const SCAN_INFLIGHT_POLL_SECONDS = "
    assert marker in html, "the in-flight poll constant moved; update this test"
    value = int(html[html.index(marker) + len(marker):].split(";")[0].strip())
    assert 0 < value <= 15, (
        f"an in-flight cadence of {value}s leaves the disabled button stuck too long"
    )
