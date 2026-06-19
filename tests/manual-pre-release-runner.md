# Pre-release Test Runner

Generic instructions for a sub-agent to execute a release-specific pre-release test pass and produce a structured log a human can review.

This file does not change between releases. Each release authors a focused `tests/manual-pre-release-vX.Y.Z.md` companion checklist; the runner consumes that file.

---

## Your job (as the sub-agent reading this)

1. **Find the release-specific checklist** at `tests/manual-pre-release-v<VERSION>.md`. `VERSION` will be in the parent's prompt (e.g. "v0.4.0"). If you can't determine the version, read `pyproject.toml` and assume the next minor version — but flag this uncertainty in the log.

2. **Verify environment before starting.** Run these checks; abort with a clear message if anything fails:

   ```bash
   git rev-parse --abbrev-ref HEAD            # should be a clean branch (main or release/*)
   git status --porcelain                     # should be empty (no uncommitted changes other than ignored files)
   tj --version                               # daemon must respond
   curl -s -f http://127.0.0.1:7391/health    # daemon must be running
   ```

   If `tj serve` isn't running, start it with `tj serve > /tmp/tj-prerelease.log 2>&1 &` and wait 3 seconds before continuing.

3. **Walk every step in the release-specific file, in order.** Each step has a structured shape:

   ```markdown
   ## Step N: Brief title

   **What:** one-line description of why this step exists

   **Setup:** (optional) commands that prepare state but aren't themselves tested

   **Test:** the commands whose output you must capture and evaluate

   **Expected:** plain-English pass criteria — compare captured output against these

   **Assertions:** (optional) commands that print `ok:` if they pass; treat absence of `ok:` as fail
   ```

   For each step:
   - Run **Setup** commands if present. Don't log their output unless they fail.
   - Run **Test** commands. Capture stdout + stderr.
   - Compare captured output to **Expected**. Decide PASS / FAIL / UNCLEAR.
   - Run **Assertions** commands if present. They print `ok:` on pass. If they don't print `ok:`, mark FAIL with the actual output.

4. **Write the result log** to `tests/results/manual-pre-release-<VERSION>-<TIMESTAMP>.md`. Create the `tests/results/` directory if it doesn't exist. Use this format:

   ````markdown
   # Pre-release test results — v<VERSION>

   - **Run at:** <ISO timestamp>
   - **Branch:** <branch name>
   - **HEAD:** <short SHA>
   - **Checklist:** tests/manual-pre-release-v<VERSION>.md
   - **Result:** N/M steps PASS, K FAIL, J UNCLEAR

   ---

   ## Step 1: <title>

   **Status:** ✅ PASS / ❌ FAIL / ⚠️ UNCLEAR

   **Test commands:**
   ```bash
   <commands run>
   ```

   **Output:**
   ```
   <captured stdout + stderr, trimmed if very long>
   ```

   **Notes:** <one-line observation; especially for UNCLEAR>

   ---

   ## Step 2: ...
   ````

5. **End with a final summary section** that lists the PASS/FAIL/UNCLEAR step numbers, the SHA + branch under test, and a recommendation: "Ready to release" / "Hold — N failures" / "Investigate — UNCLEAR steps need a human."

---

## Output trimming

If a step's output is more than ~50 lines, keep the first 30 and the last 10 with `... [N lines omitted] ...` in the middle. The point is enough context for a human reviewer to recognize the shape of the output; full transcripts go to `/tmp/tj-prerelease.log` automatically via the daemon.

## What to do when a step is UNCLEAR

Mark UNCLEAR rather than guessing if you can't tell whether output matches **Expected** — for example:
- The output mentions a number you can't verify without context (e.g., "spend total: $X")
- A subprocess timed out
- The CLI prints colored output that's hard to compare against plain-text expectations

A human reviewer will look at every UNCLEAR step, so it's not a failure — it's a "needs human eyes" flag.

## What to do when a step FAILS

Record the failure in the log and **continue** to the rest of the steps unless the failure clearly invalidates the environment (e.g., daemon crashed). The point is to surface every issue in one pass, not stop at the first.

If the daemon crashes mid-run, restart it and re-run the most recent step before continuing — note the crash in the log.

## Output file

After the run, print:
```
Results written to tests/results/manual-pre-release-<VERSION>-<TIMESTAMP>.md
Summary: <N>/<M> PASS, <K> FAIL, <J> UNCLEAR
```

The parent agent will read the log and decide release readiness.

## What you should NOT do

- Do not modify any source files (no test fixes, no playbook edits) — the runner is read-only on the codebase.
- Do not commit anything to git.
- Do not interact with external services (no web requests beyond localhost).
- Do not skip steps because they look slow.
- Do not infer pass/fail from CI status — this is a live test pass against the actual daemon.
