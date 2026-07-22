"""Behavioural tests for the Review inbox's select-all transition.

The rest of the Lens UI is guarded by static source assertions
(``test_lens_ui_regression.py``) because there is no JS test runner in the
Python CI job. That is weak for a bulk-action control: a string match on the
source still passes if the surrounding state logic is wrong, and the failure
mode here is dismissing rows the user never saw.

So this module extracts the one pure function the control delegates to
(``nextSelectAllSelection``) straight out of the served ``index.html`` and runs
it under node. Skipped when node is absent, which keeps it honest without
adding a hard dependency: the Python CI job (``.github/workflows/ci.yml``, the
``test`` job) does not set node up, only the separate ``test-ts`` job does.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_UI = Path(__file__).parent.parent.parent / "tokenjam" / "ui" / "index.html"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node not available for JS evaluation"
)


def _fn_source() -> str:
    html = _UI.read_text(encoding="utf-8")
    start = html.index("function nextSelectAllSelection")
    end = html.index("// The table's select-all box.", start)
    return html[start:end]


def _toggle(visible: list[str], checked: list[str]) -> list[str]:
    """Run one header-checkbox click and return the resulting selection."""
    script = (
        _fn_source()
        + "\nconst out = nextSelectAllSelection("
        + json.dumps(visible) + ", new Set(" + json.dumps(checked) + "));"
        + "\nconsole.log(JSON.stringify([...out]));"
    )
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True, text=True, check=True,
    )
    return json.loads(proc.stdout.strip())


def test_selects_every_rendered_row_from_empty():
    assert sorted(_toggle(["a", "b", "c"], [])) == ["a", "b", "c"]


def test_second_click_clears_the_selection():
    once = _toggle(["a", "b", "c"], [])
    assert _toggle(["a", "b", "c"], once) == []


def test_partial_selection_fills_rather_than_clears():
    # The indeterminate case: a click completes the set, it does not empty it.
    assert sorted(_toggle(["a", "b", "c"], ["b"])) == ["a", "b", "c"]


def test_never_selects_a_row_the_list_is_not_showing():
    # "c" is filtered out of the rendered set (locally dismissed, or approved in
    # this session). Select-all must not reach it.
    assert sorted(_toggle(["a", "b"], [])) == ["a", "b"]


def test_never_clears_a_selected_row_the_list_is_not_showing():
    # Clearing is likewise scoped: a stale signature survives untouched rather
    # than being silently dropped from the user's selection.
    assert sorted(_toggle(["a", "b"], ["a", "b", "stale"])) == ["stale"]


def test_empty_list_is_a_no_op():
    assert _toggle([], []) == []
    assert _toggle([], ["stale"]) == ["stale"]
