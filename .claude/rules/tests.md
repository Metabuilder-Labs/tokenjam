---
description: Test-suite rules — span factories, OTel provider setup, isolation, FORCE_COLOR.
paths:
  - "tests/**"
  - "tests/factories.py"
---

# Test-suite rules

### Critical Rule 8 — All test spans via factory

Never construct `NormalizedSpan` directly in tests; use `tests/factories.py` (`make_llm_span`,
`make_session`, `make_tool_span`, `make_session_with_spans`). The factories carry safe defaults
(`billing_account="anthropic"`, `plan_tier="api"`) that preserve existing test behavior; tests
exercising subscription / local / unknown plan-tier rendering paths should pass the field explicitly.

### Critical Rule 11 — OTel TracerProvider is global and set-once

`trace.set_tracer_provider()` only works once per process. In tests, set the provider once at module
level (not per-test in a fixture) and clear spans between tests. Use a custom
`_CollectingExporter(SpanExporter)` since `InMemorySpanExporter` is not available in the installed
OTel version. See `tests/agents/test_mock_scenarios.py` for the SDK test pattern and
`tests/integration/test_full_pipeline.py` for the pipeline pattern.

## Isolation from the real `~/.tj` / `~/.config/tj`

`tests/conftest.py` installs two session-wide, autouse fixtures — `_tj_isolated_home` repoints
`HOME` (and `tokenjam.core.config.SEARCH_PATHS`, which bakes `Path.home()` into a module-level
constant at import time, so re-pointing `HOME` alone can't fix it retroactively) at a throwaway tmp
dir for the whole run, and `_tj_guard_real_home_db` wraps `DuckDBBackend.__init__` to raise loudly if
any test still resolves a db path under the developer's real `~/.tj` or `~/.config/tj`. This is a
backstop, not a replacement for a test's own explicit `tmp_path` fixtures — but it means the suite
can never again take the DuckDB lock on a real store and contend with a concurrently running
`tj serve` / CLI (or vice versa), even from a test that forgets to isolate itself.

Corollary for measurement work: because `_tj_isolated_home` repoints `HOME`, the `summarize` analyzer
measured from inside the suite reads a corpus that does not exist (Critical Rule 31).

## `FORCE_COLOR` in your shell breaks a large swathe of the CLI-render tests at once

None of them are real failures. Rich normally detects that pytest's captured stdout is not a tty and
emits plain text, which is what CI sees and what the render tests assert against
(`assert "rung 3" in out`). With `FORCE_COLOR` set — some agent harnesses and terminal multiplexers
export it — Rich interleaves ANSI escapes *inside* the asserted phrases, so substring matches fail
across `test_quickstart.py`, `test_ping.py`, `test_relearn.py`, `test_optimize_summarize.py` and
friends. Run the suite with `env -u FORCE_COLOR python -m pytest ...` before concluding you broke
something. `TERM=dumb` also silences the color but breaks the tests that assert on progress output,
so it is not the fix.
