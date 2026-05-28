"""
Auto-discovery: walks this directory at import time and imports every .py
file (except __init__.py). Each analyzer module's @register decorator fires
as a side effect, populating ANALYZER_REGISTRY.

To add a new analyzer: drop a .py file in this directory containing a
function decorated with @register("name"). Nothing else needs editing.
"""
from __future__ import annotations

import importlib
import pkgutil

for _, _name, _ispkg in pkgutil.iter_modules(__path__):
    if not _ispkg:
        importlib.import_module(f"{__name__}.{_name}")
