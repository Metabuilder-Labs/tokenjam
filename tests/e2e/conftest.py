"""e2e test configuration — skips all tests if no API key is set."""
from __future__ import annotations

import os

import pytest

# Auto-skip every test in e2e/ if the key is absent
pytestmark = pytest.mark.skipif(
    not os.environ.get("TJ_ANTHROPIC_API_KEY"),
    reason="TJ_ANTHROPIC_API_KEY not set — skipping e2e tests",
)
