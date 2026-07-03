"""Pure trim policy for the `tj hook cap-output` PostToolUse hook.

A large tool output (a 12k-token test log, a wide ``grep -r``, a fetched web
page) is not paid for once: it becomes part of the conversation prefix and is
re-read as cache on every subsequent turn. Trimming it *at ingestion* is the one
deterministic lever that compounds, and it needs zero agent cooperation.

This module is a **pure function** — no I/O, no config file reads, no network,
no clock. That keeps it fast (<1ms), trivially testable, and safe: the caller
(``cli/cmd_hook.py``) owns all side effects (reading stdin, writing the savings
log, persisting the raw output to disk) and stays fail-open. ``trim`` never
raises on well-typed input and returns ``None`` (pass-through) whenever it is not
sure it should act.

Token estimation everywhere is the cheap char/4 heuristic (no tokenizer in the
hot path). It is an *estimate* — surfaced strings say "estimated"/"reclaimed",
never "saved you" (honesty discipline, tokenjam CLAUDE.md Rule 14).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# --- constants -------------------------------------------------------------

# Cheap byte→token heuristic. 1 token ≈ 4 chars of English/code.
CHARS_PER_TOKEN = 4

# Tools whose output is skim-not-read and high-bloat. Read/Task are deliberately
# excluded (a Read is intentional and often needed whole). The active set is
# ultimately driven by config; this is the fallback default.
DEFAULT_ELIGIBLE_TOOLS = ("Bash", "Grep", "Glob", "WebFetch")

# Bash commands whose output is a test/build run — for these, error/fail lines
# are the signal and should survive trimming (smart-error mode).
_TEST_BUILD_RE = re.compile(
    r"""(?xi)
    \b(
        pytest | py\.test | jest | vitest | mocha
      | go\s+test | cargo\s+(test|build|check|clippy)
      | make\b | npm\s+run\s+build | npm\s+test | pnpm\s+(build|test)
      | yarn\s+(build|test) | tsc\b | webpack | eslint | ruff | mypy
      | gradle | mvn\b | rspec | phpunit | ctest | ninja | bazel
    )\b
    """
)

# Lines worth keeping from the middle of a test/build log — the failure signal.
# Substring (not word-bounded) so compound tokens survive: AssertionError,
# ValueError, failure, warning, errored. Over-inclusion is safe here — the error
# block is capped at _MAX_ERROR_LINES and the whole point is to protect signal.
_ERROR_LINE_RE = re.compile(r"(?i)(error|fail|warn|traceback|assert|exception|panic|[✗✖✘❌])")

# Never keep more than this many matched error lines — an all-errors log must
# still get trimmed, not passed through under the guise of "signal".
_MAX_ERROR_LINES = 200

# Sentinel the CLI substitutes with the real on-disk path once it has persisted
# the full output. Kept out of the pure function so `trim` needs no clock/fs.
PRESERVED_REF_PLACEHOLDER = "{tj_preserved_ref}"


@dataclass(frozen=True)
class TrimResult:
    """Outcome of a trim. All token counts are char/4 estimates."""
    kept_text:    str
    orig_bytes:   int
    kept_bytes:   int
    saved_bytes:  int
    orig_tokens:  int
    kept_tokens:  int
    saved_tokens: int
    trimmed_lines: int
    tool:         str


# --- helpers ---------------------------------------------------------------

def est_tokens(text: str) -> int:
    """Cheap char/4 token estimate (no tokenizer — hot path must stay fast)."""
    return len(text) // CHARS_PER_TOKEN


def is_test_build_command(command: str) -> bool:
    """True if a Bash command looks like a test/build run (smart-error mode)."""
    return bool(command) and bool(_TEST_BUILD_RE.search(command))


def _marker(trimmed_lines: int, saved_bytes: int, saved_tokens: int,
            preserved_ref: str | None) -> str:
    """The transparency marker inserted where the middle was removed.

    Every trim carries this — nothing is ever silent. When the caller has
    persisted the full output, ``preserved_ref`` points the agent at it so the
    trim is *recoverable* (the answer to "what if the trim dropped what I
    needed"): the agent can Read the file or re-run narrower. The wording is
    deliberately a legible tool annotation, not a bare token, so the agent reads
    it as tj metadata rather than a suspicious injection.
    """
    kb = saved_bytes / 1024
    tok_k = saved_tokens / 1000
    if preserved_ref:
        recover = f"full output saved to {preserved_ref}"
    else:
        recover = "re-run narrower to recover the rest"
    return (
        f"\n[tj cap-output: trimmed {trimmed_lines} lines / {kb:.1f} KB "
        f"(~{tok_k:.1f}k est. tokens reclaimed) — {recover}; "
        f"re-run narrower (grep/head) or with an offset to see more]\n"
    )


# --- the policy ------------------------------------------------------------

def trim(
    tool_name: str,
    tool_input: dict,
    output: str,
    config,
    preserved_ref: str | None = None,
) -> TrimResult | None:
    """Decide whether/how to trim ``output`` for ``tool_name``.

    Returns a ``TrimResult`` when it trimmed, or ``None`` to pass the original
    output through untouched. Pure and total on well-typed input.

    Pass-through (``None``) when: the hook is disabled or killswitched; the tool
    isn't eligible; the output is under budget; or trimming wouldn't save at
    least ``min_saving_tokens`` (the floor — don't churn a marginal output).

    ``config`` is a ``CapOutputConfig`` (duck-typed: any object with the same
    fields works, which keeps unit tests free of the full config tree).
    ``preserved_ref`` is embedded verbatim in the marker (the CLI passes the
    path where it saved the full output); ``None`` yields a re-run hint instead.
    """
    # Defensive gates (the CLI checks these too, but trim stays self-guarding).
    if not getattr(config, "enabled", True) or getattr(config, "killswitch", False):
        return None
    eligible = getattr(config, "tools", None) or list(DEFAULT_ELIGIBLE_TOOLS)
    if tool_name not in eligible:
        return None
    if not isinstance(output, str) or not output:
        return None

    budget_tokens = int(getattr(config, "budget_tokens", 8000))
    budget_bytes = budget_tokens * CHARS_PER_TOKEN
    orig_bytes = len(output)
    orig_tokens = est_tokens(output)

    # Under budget → pass through. This is the common, cheapest case.
    if orig_bytes <= budget_bytes:
        return None

    head_lines = max(0, int(getattr(config, "head_lines", 80)))
    tail_lines = max(0, int(getattr(config, "tail_lines", 80)))
    smart_errors = bool(getattr(config, "smart_errors", True))
    min_saving_tokens = int(getattr(config, "min_saving_tokens", 500))

    command = ""
    if isinstance(tool_input, dict):
        command = str(tool_input.get("command", "") or "")

    kept_text, trimmed_lines = _build_kept(
        output=output,
        tool_name=tool_name,
        command=command,
        head_lines=head_lines,
        tail_lines=tail_lines,
        smart_errors=smart_errors,
        budget_bytes=budget_bytes,
        preserved_ref=preserved_ref,
    )
    if kept_text is None:
        return None

    kept_bytes = len(kept_text)
    kept_tokens = est_tokens(kept_text)
    saved_bytes = orig_bytes - kept_bytes
    saved_tokens = orig_tokens - kept_tokens

    # Floor: never emit a "trim" that grew the output (marker overhead on a
    # barely-over output) or that doesn't clear the min-saving bar.
    if saved_bytes <= 0 or saved_tokens < min_saving_tokens:
        return None

    return TrimResult(
        kept_text=kept_text,
        orig_bytes=orig_bytes,
        kept_bytes=kept_bytes,
        saved_bytes=saved_bytes,
        orig_tokens=orig_tokens,
        kept_tokens=kept_tokens,
        saved_tokens=saved_tokens,
        trimmed_lines=trimmed_lines,
        tool=tool_name,
    )


def _build_kept(
    output: str,
    tool_name: str,
    command: str,
    head_lines: int,
    tail_lines: int,
    smart_errors: bool,
    budget_bytes: int,
    preserved_ref: str | None,
) -> tuple[str | None, int]:
    """Assemble the trimmed text. Returns ``(kept_text, trimmed_lines)`` or
    ``(None, 0)`` when no structural trim is possible.

    Two strategies: line-based head+tail (the normal case), and a char-based cap
    fallback for a few-but-enormous-lines output (a minified blob / one giant
    line) that line trimming can't shrink.
    """
    lines = output.split("\n")
    n = len(lines)

    # --- line-based head+tail (+ smart-error splice) -----------------------
    if n > head_lines + tail_lines:
        head = lines[:head_lines]
        tail = lines[n - tail_lines:] if tail_lines else []
        middle = lines[head_lines: n - tail_lines] if tail_lines else lines[head_lines:]
        trimmed_lines = len(middle)

        error_block: list[str] = []
        if (
            smart_errors
            and tool_name == "Bash"
            and is_test_build_command(command)
            and middle
        ):
            matched = [ln for ln in middle if _ERROR_LINE_RE.search(ln)]
            if matched:
                error_block = matched[:_MAX_ERROR_LINES]

        # Estimate saved on content only (marker is tiny) to size the marker.
        content_kept_len = len("\n".join(head + error_block + tail))
        est_saved_bytes = max(0, len(output) - content_kept_len)
        est_saved_tokens = est_tokens(output) - est_tokens("\n".join(head + error_block + tail))

        parts: list[str] = list(head)
        if error_block:
            parts.append(
                f"\n[tj cap-output: {trimmed_lines} middle lines trimmed; "
                f"{len(error_block)} error/warn lines preserved below]"
            )
            parts.extend(error_block)
        parts.append(_marker(trimmed_lines, est_saved_bytes, est_saved_tokens, preserved_ref))
        parts.extend(tail)
        kept_text = "\n".join(parts)
        # A valid line trim: we removed the whole middle. Even if the kept
        # head+tail still exceeds a very small budget, that's the floor of what
        # line trimming can do — the main `trim()` min-saving check decides
        # whether it was worth it. (Char-cap is only for the few-lines case.)
        return kept_text, trimmed_lines

    # --- char-based cap fallback (few lines, huge bytes) -------------------
    # Line trimming can't help (n <= head+tail): a handful of enormous lines, a
    # minified blob. Cap by characters instead.
    return _char_cap_kept(output, budget_bytes, preserved_ref)


def _char_cap_kept(
    output: str, budget_bytes: int, preserved_ref: str | None
) -> tuple[str | None, int]:
    """Keep head+tail *characters* when line trimming can't shrink the output.

    Keeps the first ~2/3 and last ~1/3 of the byte budget so the ends — where
    prompts/errors usually sit — survive.
    """
    head_bytes = (budget_bytes * 2) // 3
    tail_bytes = budget_bytes // 3
    if head_bytes + tail_bytes >= len(output):
        return None, 0
    head_txt = output[:head_bytes]
    tail_txt = output[-tail_bytes:] if tail_bytes else ""
    est_saved_bytes = len(output) - (len(head_txt) + len(tail_txt))
    est_saved_tokens = est_tokens(output) - est_tokens(head_txt + tail_txt)
    kept = head_txt + _marker(0, est_saved_bytes, est_saved_tokens, preserved_ref) + tail_txt
    return kept, 0
