"""
Analyzer registry. Each analyzer module self-registers via @register.

Auto-discovery in analyzers/__init__.py walks the analyzers/ directory and
imports every .py file, triggering registration as a side effect. Adding a
new analyzer = drop a file under analyzers/ with @register("name") on a
function taking AnalyzerContext.
"""
from __future__ import annotations

from typing import Callable

from tokenjam.core.optimize.types import AnalyzerContext

# Analyzer signature: takes the shared context, mutates ctx.report in place.
Analyzer = Callable[[AnalyzerContext], None]

ANALYZER_REGISTRY: dict[str, Analyzer] = {}


def register(name: str) -> Callable[[Analyzer], Analyzer]:
    """Decorator used by analyzer modules to register themselves."""
    def deco(fn: Analyzer) -> Analyzer:
        ANALYZER_REGISTRY[name] = fn
        return fn
    return deco
