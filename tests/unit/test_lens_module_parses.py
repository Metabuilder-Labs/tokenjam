"""The dashboard's module script must PARSE. Nothing else in the suite checks.

A syntax error anywhere in ``ui/index.html``'s module script is total: the
browser never executes a line of it, ``#app`` stays empty, and every page of the
dashboard is a blank white rectangle. There is no partial degradation and no
console error a server-side test would see.

Every other UI test in this suite matches STRINGS against the file. A file with
a syntax error still contains all of them, so the whole UI suite stays green
while the product does not render at all. This was not hypothetical: a comment
written in the ``${ /* ... */ '' }`` form that is correct BETWEEN markup nodes
was placed inside a ternary EXPRESSION, where it is a syntax error. Every
existing test passed; the page was blank.

One assertion, no fixtures, runs in about a second. It is the cheapest test in
the suite and the only one that can fail for the most catastrophic reason.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

_UI = Path(__file__).parent.parent.parent / "tokenjam" / "ui" / "index.html"


@pytest.mark.skipif(
    shutil.which("node") is None, reason="node not available for JS parsing"
)
def test_the_dashboard_module_script_is_syntactically_valid() -> None:
    src = _UI.read_text(encoding="utf-8")
    modules = re.findall(r'<script type="module">(.*?)</script>', src, re.S)
    assert modules, "no module script found in index.html — has the page moved?"
    # The app itself is the big one; the page also carries a short module of
    # import-map commentary. Checking the largest is what covers the app.
    app_module = max(modules, key=len)
    assert len(app_module) > 10_000, (
        "the largest module script is suspiciously small — this test is "
        "probably no longer looking at the application code"
    )
    result = subprocess.run(
        ["node", "--input-type=module", "--check"],
        input=app_module, text=True, capture_output=True,
    )
    assert result.returncode == 0, (
        "ui/index.html's module script does not parse, so the dashboard renders "
        "as a blank page. Every string-matching UI test still passes against a "
        f"file in this state.\n\n{result.stderr}"
    )


@pytest.mark.skipif(
    shutil.which("node") is None, reason="node not available for JS parsing"
)
def test_the_inline_scripts_outside_the_module_also_parse() -> None:
    """The theme toggle and friends are classic scripts, not modules.

    They fail more quietly than the module does — the page still renders — but
    a broken one silently kills whatever it was responsible for, and nothing
    else here would notice.
    """
    src = _UI.read_text(encoding="utf-8")
    classic = re.findall(r'<script>(.*?)</script>', src, re.S)
    for i, body in enumerate(classic):
        if not body.strip():
            continue
        result = subprocess.run(
            ["node", "--check", "-"], input=body, text=True, capture_output=True,
        )
        assert result.returncode == 0, (
            f"inline classic script #{i} in ui/index.html does not "
            f"parse:\n\n{result.stderr}"
        )
