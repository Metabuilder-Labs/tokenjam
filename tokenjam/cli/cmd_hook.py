"""`tj hook cap-output` — the PostToolUse output-trim hook entrypoint.

Claude Code invokes this on every eligible tool call, piping the PostToolUse
event JSON on stdin. We shrink a bloated tool output *before it enters context*
and print an `updatedToolOutput` that REPLACES what the model sees — the tool
itself already ran unchanged (true exit code, real behavior); we only trim the
presentation.

Contract (verified empirically against Claude Code 2.1.198):
  - stdin: `{tool_name, tool_input, tool_response, session_id, ...}`.
    For Bash, `tool_response` is `{stdout, stderr, interrupted, isImage,
    noOutputExpected}`; the model reads `stdout`.
  - stdout to replace: exit 0 + `{"hookSpecificOutput": {"hookEventName":
    "PostToolUse", "updatedToolOutput": <same shape as tool_response>}}`.
    `updatedToolOutput` MUST preserve the original shape — so we spread the
    original dict and replace only the one text field.
  - FAIL-OPEN: exit 0 with NO stdout → Claude Code keeps the ORIGINAL output.
    Every error path here emits nothing, so a tj bug can never lose data or
    break a session.

Fast: no DB (registered in `no_db_commands`), no network, char/4 estimation.
"""
from __future__ import annotations

import json
import sys

import click

from tokenjam.core.output_cap import TrimResult, trim
from tokenjam.core.savings_log import append_saving, persist_output

# Per-tool preference order for the STRING field carrying the text the model
# reads. Bash.stdout is empirically verified; the others are best-effort and
# fall back to the generic longest-string heuristic. Typed-list responses
# (native Grep matches / Glob paths) have no string payload → pass through,
# rather than risk reshaping an unverified structure (the trim-broke-it risk).
_STRING_FIELDS: dict[str, list[str]] = {
    "Bash": ["stdout"],
    "WebFetch": ["result", "body", "output", "text", "content"],
    "Grep": ["content", "output", "stdout", "text"],
    "Glob": ["output", "stdout", "text"],
}


def _payload_field(tool_name: str, resp: dict) -> str | None:
    """The dict key holding the trimmable text for this tool, or None."""
    for k in _STRING_FIELDS.get(tool_name, []):
        v = resp.get(k)
        if isinstance(v, str) and v:
            return k
    # Generic fallback: the single longest string field (shape stays preserved
    # since we only ever replace this one key).
    best, blen = None, 0
    for k, v in resp.items():
        if isinstance(v, str) and len(v) > blen:
            best, blen = k, len(v)
    return best


def _trim_text(tool_name: str, tool_input: dict, text: str, config,
               session_id: str) -> TrimResult | None:
    """Trim a text payload; persist the full original first so the marker can
    point at a recoverable copy. Two-phase so we only touch disk when we will
    actually trim (the common under-budget case does no I/O)."""
    cap = config.hooks.output_cap
    probe = trim(tool_name, tool_input, text, cap, preserved_ref=None)
    if probe is None:
        return None
    path = persist_output(config, tool_name, session_id, text)
    ref = str(path) if path else None
    return trim(tool_name, tool_input, text, cap, preserved_ref=ref)


def _extract_and_trim(tool_name, tool_input, tool_response, config, session_id):
    """Return `(updated_tool_output, TrimResult)` or None (pass through).

    Preserves the original response SHAPE: a string stays a string; a dict is
    spread and only its one text field is replaced.
    """
    if isinstance(tool_response, str):
        res = _trim_text(tool_name, tool_input, tool_response, config, session_id)
        return (res.kept_text, res) if res else None
    if isinstance(tool_response, dict):
        field = _payload_field(tool_name, tool_response)
        if field is None:
            return None
        res = _trim_text(tool_name, tool_input, tool_response[field], config, session_id)
        if res is None:
            return None
        updated = dict(tool_response)
        updated[field] = res.kept_text
        return (updated, res)
    return None


@click.group("hook")
def cmd_hook() -> None:
    """Claude Code hook entrypoints (installed out-of-band by `tj onboard`)."""


@cmd_hook.command("cap-output")
@click.pass_context
def cap_output(ctx: click.Context) -> None:
    """PostToolUse: trim a bloated tool output before it enters context.

    Reads the hook event JSON on stdin; prints an `updatedToolOutput` JSON when
    it trims, or nothing (pass-through). Fail-open on every path.
    """
    # 1) Read stdin — fail-open on any error / empty input.
    try:
        raw = sys.stdin.read()
    except Exception:
        return
    if not raw or not raw.strip():
        return
    try:
        payload = json.loads(raw)
    except Exception:
        return

    # 2) Apply the policy. A single broad guard keeps the WHOLE path fail-open:
    #    any exception → emit nothing → Claude Code keeps the original output.
    try:
        config = ctx.obj.get("config")
        cap = config.hooks.output_cap
        if not cap.enabled or cap.killswitch:
            return

        tool_name = payload.get("tool_name")
        if tool_name not in (cap.tools or []):
            return
        tool_input = payload.get("tool_input") or {}
        tool_response = payload.get("tool_response")
        if tool_response is None:
            tool_response = payload.get("tool_output")  # forward-compat alias
        session_id = payload.get("session_id") or ""

        result = _extract_and_trim(tool_name, tool_input, tool_response, config, session_id)
        if result is None:
            return  # under budget / not eligible / nothing worth trimming

        updated_output, res = result

        # 3) Record the saving (append-only JSONL, never a DB lock; fail-safe).
        append_saving(config, {
            "session_id": session_id,
            "tool": tool_name,
            "orig_tok_est": res.orig_tokens,
            "kept_tok_est": res.kept_tokens,
            "saved_tok_est": res.saved_tokens,
            "orig_bytes": res.orig_bytes,
            "saved_bytes": res.saved_bytes,
        })

        # 4) Emit the replacement. Build the full string, then one write — so a
        #    late error never emits a half-written JSON.
        out = {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "updatedToolOutput": updated_output,
            }
        }
        sys.stdout.write(json.dumps(out))
    except Exception:
        return
