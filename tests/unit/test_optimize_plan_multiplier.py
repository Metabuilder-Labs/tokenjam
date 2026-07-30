"""Unit tests for subscription-plan multiplier formatting in cmd_optimize."""
from __future__ import annotations

from tokenjam.cli.cmd_optimize import _format_plan_multiplier


def test_format_plan_multiplier_below_threshold_uses_floor_label():
    assert _format_plan_multiplier(0.0) == "<0.1×"
    assert _format_plan_multiplier(0.049) == "<0.1×"
    assert _format_plan_multiplier(0.099) == "<0.1×"


def test_format_plan_multiplier_at_or_above_threshold_keeps_one_decimal():
    assert _format_plan_multiplier(0.1) == "0.1×"
    assert _format_plan_multiplier(0.12) == "0.1×"
    assert _format_plan_multiplier(1.34) == "1.3×"
    assert _format_plan_multiplier(21.2) == "21.2×"
