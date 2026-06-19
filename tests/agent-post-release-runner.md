# Post-release Test Runner

Generic instructions for a sub-agent to execute a release-specific post-release test pass against the **published artifact** (PyPI + npm) and produce a structured log a human can review.

This file does not change between releases. Each release authors a focused `tests/agent-post-release-vX.Y.Z.md` companion checklist; the runner consumes that file.

The comprehensive human-walkable playbook lives at `tests/manual-new-release-tests.md` — that's the reference for what's testable. The release-specific files for the runner pull only the v0.X.Y-critical subset.

---

## Your job (as the sub-agent reading this)

1. **Find the release-specific checklist** at `tests/agent-post-release-v<VERSION>.md`. `VERSION` will be in the parent's prompt (e.g. "v0.4.0"). If you can't determine the version, read `pyproject.toml` and use that — but flag the uncertainty in the log.

2. **Set up an isolated environment.** Post-release tests use the *published* PyPI artifact, so they install via `pipx`. To avoid touching the user's real install, set `HOME` to a fresh tmp directory for the entire run:

   ```bash
   export RUN_HOME="$(mktemp -d -t tj-postrelease-XXXXXX)"
   export ORIG_HOME="$HOME"
   export HOME="$RUN_HOME"
   echo "Running in isolated HOME: $RUN_HOME"
   ```

   Restore `HOME` at the end of the run (success or failure):

   ```bash
   export HOME="$ORIG_HOME"
   ```

   pipx will install into `$RUN_HOME/.local/pipx/venvs/tokenjam/`, completely separate from any existing install.

3. **Verify pipx is available** before starting:

   ```bash
   command -v pipx || { echo "pipx not found — abort"; exit 2; }
   ```

   If pipx isn't installed, abort with a clear message — post-release tests verify the recommended install path, which is pipx.

4. **Walk every step in the release-specific file, in order.** Same structured shape as the pre-release runner:

   ```markdown
   ## Step N: Brief title

   **What:** one-line description

   **Setup:** (optional) commands that prepare state but aren't themselves tested

   **Test:** the commands whose output you must capture and evaluate

   **Expected:** plain-English pass criteria

   **Assertions:** (optional) commands that print `ok:` on pass
   ```

   For each step:
   - Run **Setup** silently
   - Run **Test** and capture output
   - Compare to **Expected**, decide PASS / FAIL / UNCLEAR
   - Run **Assertions**; treat missing `ok:` as FAIL with actual output

5. **Write the result log** to `tests/results/agent-post-release-<VERSION>-<TIMESTAMP>.md`. Use the same format the pre-release runner uses:

   ````markdown
   # Post-release test results — v<VERSION>

   - **Run at:** <ISO timestamp>
   - **Isolated HOME:** <path>
   - **Tested artifact:** pipx-installed `tokenjam` from PyPI
   - **Checklist:** tests/agent-post-release-v<VERSION>.md
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
   <captured stdout + stderr, trimmed if >50 lines>
   ```

   **Notes:** <one-line observation>

   ---
   ````

6. **End with a final summary** listing PASS/FAIL/UNCLEAR steps, plus a recommendation: "Release is healthy" / "Hold — N failures" / "Investigate — UNCLEAR steps need a human."

7. **Clean up.** After the summary, remove the isolated HOME directory:

   ```bash
   export HOME="$ORIG_HOME"
   rm -rf "$RUN_HOME"
   ```

   Log this as the last action in the result file.

---

## Network and timing

The first step (`pipx install`) hits PyPI and takes 30–60 seconds depending on the connection. Don't time out. Subsequent commands run locally against the installed venv.

## Output trimming

Same rule as the pre-release runner: if a step's output is >50 lines, keep the first 30 and the last 10 with `... [N lines omitted] ...` in the middle.

## When a step is UNCLEAR

Same rules as pre-release: mark UNCLEAR rather than guessing. Continue past FAILs unless the install itself broke.

## What you should NOT do

- **Do not** unset or modify the user's real `HOME` — only `$RUN_HOME`.
- **Do not** modify any source files in the working tree.
- **Do not** commit anything.
- **Do not** call `tj uninstall --yes` or `pipx uninstall` against the real `HOME`. The isolation makes that unnecessary; doing it would wipe the user's real install.
- **Do not** skip the cleanup step.

## Output file

After the run, print:
```
Results written to tests/results/agent-post-release-<VERSION>-<TIMESTAMP>.md
Summary: <N>/<M> PASS, <K> FAIL, <J> UNCLEAR
HOME restored. Isolated dir removed: <path>
```
