"""Guard against reintroducing bare `find_config_file()` call sites.

A bare `find_config_file()` call silently ignores `TJ_CONFIG`, even when the
surrounding code already assumes a `TJ_CONFIG`-aware config (loaded via
`load_config`, which does honor it). That shape recurred at 8+ call sites
across the CLI, the API, and the MCP server before this guard was added —
each one independently "forgot" to pass the env override through. The fix:
`resolve_config_path()` in `core/config.py` is now the single source of
truth for "which config file is this process using"; everything except
`core/config.py` itself (which defines both functions) and `cli/home.py`
(whose bare `tj` landing screen must stay resilient to an invalid TJ_CONFIG —
see the `_tj_config_env_path` docstring there) must go through it instead of
calling `find_config_file` directly.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "tokenjam"

# Files allowed to call find_config_file() directly.
#   - core/config.py: defines both find_config_file and resolve_config_path;
#     resolve_config_path's own implementation calls find_config_file.
#   - cli/home.py: the bare `tj` landing screen renders before any config is
#     validated, so it needs a non-raising TJ_CONFIG check (see
#     _tj_config_env_path) rather than resolve_config_path's fail-loud
#     contract. Both of its bare calls are documented, safety-motivated
#     exceptions, not oversights.
_ALLOWED_RELATIVE_PATHS = {
    Path("core/config.py"),
    Path("cli/home.py"),
}

_CALL_PATTERN = re.compile(r"\bfind_config_file\s*\(")


def _iter_source_files():
    for path in PACKAGE_ROOT.rglob("*.py"):
        yield path


def test_no_bare_find_config_file_outside_allowlist():
    offenders: list[str] = []
    for path in _iter_source_files():
        rel = path.relative_to(PACKAGE_ROOT)
        if rel in _ALLOWED_RELATIVE_PATHS:
            continue
        text = path.read_text()
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if _CALL_PATTERN.search(line):
                offenders.append(f"{rel}:{lineno}: {stripped}")

    assert not offenders, (
        "Found bare find_config_file() call(s) outside the allowlist — these "
        "silently ignore TJ_CONFIG. Use resolve_config_path() instead (or, "
        "for a call site that must stay resilient to an invalid TJ_CONFIG "
        "like cli/home.py, add it to _ALLOWED_RELATIVE_PATHS with a comment "
        "explaining why):\n" + "\n".join(offenders)
    )
