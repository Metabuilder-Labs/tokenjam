"""Regression guard for the Analytics `tokens` metric aggregate.

The `tokens` metric (`_METRIC_EXPR["tokens"]` in
tokenjam/api/routes/analytics.py) has silently dropped a token column four
times in this repo (see root CLAUDE.md, "Cache token types in aggregates").
`_TOKENS_EXPR` — used for both the per-row token column and the KPI totals —
must stay byte-for-byte aligned with `_METRIC_EXPR["tokens"]` so a future
edit to one can't quietly regress the other.
"""
from __future__ import annotations

from tokenjam.api.routes import analytics

_REQUIRED_COLUMNS = ("input_tokens", "output_tokens", "cache_tokens", "cache_write_tokens")


def test_tokens_metric_covers_all_four_token_columns():
    expr, unit = analytics._METRIC_EXPR["tokens"]
    for col in _REQUIRED_COLUMNS:
        assert col in expr, f"tokens metric expression is missing {col}: {expr}"
    assert unit == "tokens"


def test_tokens_expr_constant_covers_all_four_token_columns():
    """`_TOKENS_EXPR` backs the per-row `tokens` column AND the KPI totals
    (`_kpi_cols`) — it must carry the same four columns as the metric."""
    for col in _REQUIRED_COLUMNS:
        assert col in analytics._TOKENS_EXPR, (
            f"_TOKENS_EXPR is missing {col}: {analytics._TOKENS_EXPR}"
        )


def test_tokens_metric_and_tokens_expr_constant_stay_in_sync():
    """The metric expression and the shared _TOKENS_EXPR constant must be
    identical — they represent the same quantity computed in two SELECT
    clauses (grouped rows and KPI totals) and must never drift apart."""
    metric_expr, _ = analytics._METRIC_EXPR["tokens"]
    assert metric_expr == analytics._TOKENS_EXPR
