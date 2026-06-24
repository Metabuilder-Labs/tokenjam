## Summary

<!-- What does this PR do? 1-3 sentences. -->

## Related issue

<!-- Closes #N  — so the issue auto-closes on merge. One "Closes" line per issue. Omit if there's no issue. -->

## Checklist

- [ ] Tests pass (`pytest tests/unit/ tests/synthetic/ tests/agents/ tests/integration/`)
- [ ] Lint clean (`ruff check tokenjam/`)
- [ ] Type check clean (`mypy tokenjam/`)
- [ ] CLAUDE.md updated (if architecture changed)
- [ ] Test spans use `tests/factories.py` (not raw `NormalizedSpan`)
- [ ] Requested `@anilmurty` as reviewer (or @-mentioned him above)
