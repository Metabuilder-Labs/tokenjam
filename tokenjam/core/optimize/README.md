# `tokenjam.core.optimize` — adding a new analyzer

Drop a file under `analyzers/` containing a function decorated with `@register("<name>")`:

```python
# analyzers/my_finding.py
from tokenjam.core.optimize.registry import register
from tokenjam.core.optimize.types import AnalyzerContext

@register("my-finding")
def run(ctx: AnalyzerContext) -> None:
    # Read from ctx.conn / ctx.config / ctx.since / ctx.until / ctx.summary.
    # Mutate ctx.report to record findings.
    ...
```

Auto-discovery in `analyzers/__init__.py` imports every `.py` file under that
directory, so the `@register` side effect fires automatically. No edit to
`__init__.py` or `cmd_optimize.py` is required — the CLI reads valid
positional analyzer name choices from `ANALYZER_REGISTRY.keys()` at click decoration time.

## Ordering

Analyzers run in the order defined by `ANALYZER_ORDER` in `runner.py`. If
your analyzer depends on or is depended on by another (e.g. budget-projection
reads `ctx.report.downgrade`), add it to `ANALYZER_ORDER` in the right
position. Analyzers not in `ANALYZER_ORDER` still run, just last and in
arbitrary order.

## Honesty constraints

Every analyzer must:

- Declare a confidence level on each finding (`structural` / `replay_validated`
  / `user_validated`).
- Never claim quality equivalence; only that the structural shape matches a
  class of work worth reviewing.
- Carry a caveat string the renderer surfaces verbatim. Constants like
  `MODEL_DOWNGRADE_CAVEAT` live in `types.py`.

See `tokenjam-product-strategy.md` v3 §4.5 and §10 for the full discipline.
