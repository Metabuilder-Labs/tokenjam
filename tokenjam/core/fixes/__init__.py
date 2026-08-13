"""One record per fix type, plus the lint that keeps them honest.

Fix text used to live as constants scattered across analyzer modules, which is
why the same sizing contradiction could ship twice in different words and be
fixed in only one place. A catalog gives a policy one home; the lint makes the
properties it must hold mechanical rather than review-dependent — including the
one that matters most here, that a fix may not re-license the behaviour its own
analyzer bills for.
"""
from __future__ import annotations

from tokenjam.core.fixes.catalog import (
    FIX_CATALOG,
    LEVER_AWARENESS,
    LEVER_EFFORT,
    LEVER_MODEL,
    LEVER_OFFLOAD,
    LEVER_ROUTING,
    PERSONA_ANY,
    PERSONA_CLAUDE_CODE,
    PERSONA_SDK,
    FixRecord,
    fix_for,
    fix_text,
    fix_text_for,
    fixes_for,
    register,
)
from tokenjam.core.fixes.lint import MAX_FIX_LINES, lint_catalog, lint_fix

# Imported for its side effect: registering every catalogued fix. Kept last so
# the machinery above is defined before the data lands on it.
from tokenjam.core.fixes import registry as _registry  # noqa: E402,F401

__all__ = [
    "FIX_CATALOG",
    "LEVER_AWARENESS",
    "LEVER_EFFORT",
    "LEVER_MODEL",
    "LEVER_OFFLOAD",
    "LEVER_ROUTING",
    "MAX_FIX_LINES",
    "PERSONA_ANY",
    "PERSONA_CLAUDE_CODE",
    "PERSONA_SDK",
    "FixRecord",
    "fix_for",
    "fix_text",
    "fix_text_for",
    "fixes_for",
    "lint_catalog",
    "lint_fix",
    "register",
]
