"""
Regression test for issue #87.

TokenJam's main pitch is local-first / no data egress. The served web UI
(`tj serve` → http://127.0.0.1:7391/) must work without internet access —
a user running tj in an air-gapped environment to verify our local-first
claims should see a fully functional dashboard.

This test reads the bundled UI HTML and asserts no external URLs are
referenced. Any future contributor adding a CDN font / icon / script
will fail this test and surface the regression before it ships.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_UI_HTML = Path(__file__).parent.parent.parent / "tokenjam" / "ui" / "index.html"

# External hosts that would cause an offline UI to fail at RENDER time.
# Clickable <a href> links that point to external sites (e.g. github.com
# in the footer) are fine — they don't fetch on load — and are excluded
# below via _is_anchor_href.
_KNOWN_RENDER_TIME_HOSTS = (
    "fonts.googleapis.com",
    "fonts.gstatic.com",
    "cdn.jsdelivr.net",
    "cdnjs.cloudflare.com",
    "unpkg.com",
    "esm.sh",
    "tokenjam.dev",          # don't pull our marketing-site assets into the dashboard
    "tokenjam.com",
    "opencla.watch",
)

# Strip HTML comments first so prose docs about offline-only operation
# don't false-positive.
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
# Find every http(s) URL in the document.
_EXTERNAL_HTTP_RE = re.compile(r"\bhttps?://[a-zA-Z0-9.-]+")
# Anchor hrefs: `<a ... href="https://..."`. These are clickable links,
# not render-time fetches — they don't break offline UI.
_ANCHOR_HREF_URL_RE = re.compile(
    r'<a\b[^>]*\bhref=["\'](https?://[^"\']+)["\']',
    re.IGNORECASE,
)


def _stripped_html() -> str:
    """Read the UI HTML and strip comments so prose docs don't false-positive."""
    return _HTML_COMMENT_RE.sub("", _UI_HTML.read_text(encoding="utf-8"))


def _anchor_href_urls(body: str) -> set[str]:
    return set(_ANCHOR_HREF_URL_RE.findall(body))


def _anchor_href_hosts(body: str) -> set[str]:
    """Hosts referenced by clickable <a href> links (path stripped)."""
    return {_EXTERNAL_HTTP_RE.match(u).group() for u in _anchor_href_urls(body)}


def test_ui_html_exists():
    assert _UI_HTML.exists(), f"Bundled UI HTML missing at {_UI_HTML}"


@pytest.mark.parametrize("host", _KNOWN_RENDER_TIME_HOSTS)
def test_no_known_render_time_external_host(host):
    """
    None of these hosts can appear anywhere outside an <a href> link.
    Anchor links are clickable navigations that don't fire on load and
    are checked separately below.
    """
    body = _stripped_html()
    anchor_hosts = _anchor_href_hosts(body)
    # All host occurrences in the document.
    all_hosts = _EXTERNAL_HTTP_RE.findall(body)
    bad = [h for h in all_hosts if host in h and h not in anchor_hosts]
    assert not bad, (
        f"UI loads from external host {host!r} at render time — breaks "
        f"the local-first promise. See issue #87. If you need the asset, "
        f"vendor it locally or inline it as a data: URL.\n"
        f"Offending URL(s): {bad}"
    )


def test_no_render_time_external_http_references_in_served_html():
    """
    Catch-all: any http(s):// URL that the browser would fetch when the
    page renders (anything NOT inside an <a href>) means the dashboard
    would make an external request on load. The dashboard must work
    fully offline.
    """
    body = _stripped_html()
    anchor_hosts = _anchor_href_hosts(body)
    matches = _EXTERNAL_HTTP_RE.findall(body)
    # Exclusions:
    # 1. XML namespace URLs (xmlns="http://www.w3.org/2000/svg") aren't fetched.
    # 2. URLs whose host appears inside an <a href> are clickable navigation,
    #    not render-time fetches.
    bad = [
        m for m in matches
        if not m.startswith("http://www.w3.org")
        and m not in anchor_hosts
    ]
    assert not bad, (
        f"UI HTML references external URL(s) at render time: {bad}. "
        f"The served dashboard must work fully offline (issue #87). "
        f"Inline as a data: URL, vendor under tokenjam/ui/vendor/, or move "
        f"into a clickable <a href> link."
    )


def test_vendor_directory_has_expected_files():
    """
    The vendored ESM modules must exist so the importmap in index.html
    resolves successfully. If the wheel-build excludes them or someone
    deletes them, this test catches it before the dashboard breaks.
    """
    vendor_dir = _UI_HTML.parent / "vendor"
    assert vendor_dir.exists(), "tokenjam/ui/vendor/ directory missing"
    for filename in ("preact.js", "preact-hooks.js", "htm.js", "uplot.js", "uplot.css"):
        f = vendor_dir / filename
        assert f.exists(), f"vendored module missing: {f}"
        assert f.stat().st_size > 100, f"vendored module suspiciously small: {f}"


def test_vendored_css_has_no_external_refs():
    """Vendored CSS (e.g. uplot.css) must not pull external url() assets at
    render time — that would defeat offline-first. Data URLs are fine."""
    vendor_dir = _UI_HTML.parent / "vendor"
    for css in vendor_dir.glob("*.css"):
        body = css.read_text(encoding="utf-8")
        external = re.findall(r"url\(\s*['\"]?(https?:)", body)
        assert not external, f"{css.name} references an external url(): {external}"
