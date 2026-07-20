"""Reusable guard: a token aggregate must sum ALL FOUR token types.

A span bills across four buckets: ``input_tokens``, ``output_tokens``,
``cache_tokens`` (cache reads) and ``cache_write_tokens`` (cache creation).
Writing an aggregate that adds up some of them and forgets ``cache_write_
tokens`` under-reports the most expensive bucket, and that exact omission has
shipped three separate times in this package. Reviews keep missing it because
the broken form looks complete: ``SUM(input_tokens + output_tokens +
cache_tokens)`` reads like it covers everything.

Use this from any test that owns a new aggregate::

    from tests.token_aggregate_guard import assert_module_token_sums_are_complete
    from tokenjam.core.optimize.analyzers import my_new_analyzer

    def test_my_analyzer_sums_all_four_token_types():
        assert_module_token_sums_are_complete(my_new_analyzer)

Rule enforced: any single ``SUM(...)`` expression that adds up TWO OR MORE
token columns must contain all four. A deliberate single-bucket sum
(``SUM(cache_tokens)`` for a cache-read ratio, say) is untouched, because it
isn't summing token types together in the first place.
"""
from __future__ import annotations

import inspect
import re
from types import ModuleType

TOKEN_COLUMNS = frozenset(
    {"input_tokens", "output_tokens", "cache_tokens", "cache_write_tokens"}
)

_SUM_START = re.compile(r"\bSUM\s*\(", re.IGNORECASE)


def _sum_expressions(sql: str) -> list[str]:
    """Every ``SUM(...)`` body in ``sql``, paren-balanced so a nested
    ``COALESCE(...)`` inside the sum is captured whole."""
    bodies: list[str] = []
    for match in _SUM_START.finditer(sql):
        depth = 1
        start = match.end()
        for i in range(start, len(sql)):
            if sql[i] == "(":
                depth += 1
            elif sql[i] == ")":
                depth -= 1
                if depth == 0:
                    bodies.append(sql[start:i])
                    break
    return bodies


def _columns_in(expression: str) -> set[str]:
    """The token columns named in one SUM body. ``cache_tokens`` is matched on
    a word boundary so it never swallows ``cache_write_tokens``."""
    found = set()
    for column in TOKEN_COLUMNS:
        if re.search(rf"\b{column}\b", expression):
            found.add(column)
    return found


def find_incomplete_token_sums(sql: str) -> list[tuple[str, set[str]]]:
    """Every mixed token sum in ``sql`` that is missing a bucket, as
    ``(expression, missing columns)``. Empty list means the text is clean."""
    offenders = []
    for body in _sum_expressions(sql):
        columns = _columns_in(body)
        if len(columns) >= 2 and columns != TOKEN_COLUMNS:
            offenders.append((body.strip(), TOKEN_COLUMNS - columns))
    return offenders


def assert_sql_token_sums_are_complete(sql: str, *, context: str = "sql") -> None:
    """Fail when ``sql`` contains a mixed token sum missing a bucket."""
    offenders = find_incomplete_token_sums(sql)
    assert not offenders, (
        f"{context}: token aggregate is missing a token type. "
        + "; ".join(
            f"SUM({expr}) omits {', '.join(sorted(missing))}" for expr, missing in offenders
        )
    )


def assert_module_token_sums_are_complete(module: ModuleType) -> None:
    """Fail when any mixed token sum in ``module``'s source misses a bucket.

    Reads the module's own source, so it covers every query the module builds,
    including ones assembled at runtime from f-strings.
    """
    assert_sql_token_sums_are_complete(
        inspect.getsource(module), context=module.__name__,
    )
