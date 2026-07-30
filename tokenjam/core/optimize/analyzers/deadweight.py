"""
MCP dead-weight + always-injected context tax analyzer (self-improve loop,
cost quick-wins Component C).

Claude Code transcripts lane only (C1 + C2 of the cost quick-wins spec,
``.claude/self-improve-loop/COST-SPEC.md`` §2 Component C).

C1 — MCP dead weight: enumerate the MCP servers configured for a session
(project ``.mcp.json`` / ``.claude/settings*.json``, global ``~/.claude.json``
— all read-only, this module never writes to any of them) and count how
often each server's tools are actually INVOKED (``mcp__<server>__<tool>``
tool_use blocks) across the window's sessions. A server present in at least
``MIN_SESSIONS_DEADWEIGHT`` distinct sessions with ZERO invocations across all
of them is dead weight: its tool schemas are still injected into context for
no return.

Deferred-tools caveat (mandatory, spec hard rule). When a session's
transcript shows a deferred/ToolSearch-style listing naming a server's tools,
that server's full schemas were NOT loaded that turn — only a short
name+description line per tool appears in the listing, so the real per-turn
tax is much smaller. This module detects that marker per session and blends
``DEFERRED_SCHEMA_TAX_TOKENS`` into the estimate for those sessions; it never
claims the full ``FULL_SCHEMA_TAX_TOKENS`` tax for a deferred session.

C2 — always-injected context tax table (report-only, no proposals): a ranked,
per-source, per-session token-tax table for what actually shows up verbatim
in a session's first turn — session-start hook/environment output, rules
files, CLAUDE.md — plus the MCP schema-injection line per configured server.
Every figure is ``estimated``. The "never referenced" judgment (whether a
source's content ever gets used downstream) is explicitly OUT of scope for
this pass — see the spec's cut list.

Dedup. A server's MCP-schema tax-table row is purely informational and never
feeds ``DeadweightFinding.past_overspend_tokens`` — only the C1
dead-weight servers' own tax does, so a server's tax is never counted twice
(see ``compute_deadweight_finding``'s dedup note).

Never raises: an unreadable transcript, a malformed config file, or a missing
projects root is skipped, not fatal — mirrors relearn.py's unattended
robustness (this runs on the same schedule).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from json.decoder import scanstring
from pathlib import Path
from typing import Any

from tokenjam.core.optimize.registry import register
from tokenjam.core.optimize.span_pricing import blended_rates, price_span, span_instant
from tokenjam.core.optimize.types import AnalyzerContext
from tokenjam.core.transcript import _SYSTEM_REMINDER_RE, read_records, resolve_projects_root
from tokenjam.core.usage import AssistantUsage, assistant_message_key, parse_usage

# --- Tunables ------------------------------------------------------------

#: A server must be configured-present in at least this many DISTINCT
#: sessions, with zero invocations across all of them, before it's flagged
#: dead weight. Originally 10 (spec: "start N=10"); lowered to 5 after an
#: audit of all twelve analyzers found this the single biggest one-shot fix
#: for analyzers that rarely fire on a normal user's window — a server
#: configured-but-never-called is unlikely to be a fluke even at a much
#: lower bar. False-positive shape (modeling each session as an independent
#: Bernoulli trial with per-session use probability p): a server actually
#: needed 1-in-4 sessions has a (1-p)^N ~= 42% chance of a spurious
#: zero-invocation read at N=3, vs ~24% at N=5, vs ~6% at N=10 -- N=5 keeps
#: that chance in the same order of magnitude as the old default while
#: needing HALF the silent evidence to surface, materially increasing how
#: often this analyzer fires. N=3 was considered and rejected: it nearly
#: doubles the false-positive rate over N=5 for the same occasional-use
#: server, and this finding is apply-capable (see the removal machinery
#: below), so a wrongly-flagged server costs a real (user-approved, but
#: still avoidable) config edit, not just a noisy card. The module's own
#: DEADWEIGHT_HONESTY_CAVEAT and review-before-apply gate remain the
#: backstop for whatever residual false-positive risk N=5 still carries.
MIN_SESSIONS_DEADWEIGHT = 5

#: How many example session ids a dead server's card carries as evidence
#: (mirrors relearn.py's MAX_EXAMPLE_SESSIONS convention).
MAX_EXAMPLE_SESSIONS = 3

#: Full MCP-connector schema-injection tax, PER CALL, when its tool schemas
#: are loaded (not deferred). This is an ASSUMPTION carried into this module,
#: not a measurement taken here or anywhere in this repo: the on-disk
#: transcript has no per-schema token count to measure against (see
#: core/context_diagnostic.py's MCP_INJECTION_PARK_NOTE, the one real in-repo
#: source for the "~25K tokens/call" figure) — and that note itself describes
#: the tax for ALL of a session's attached MCP servers COMBINED, not for one
#: server. This module nonetheless charges the full 25K flat to EVERY
#: individual server, every call, regardless of how many tools that server
#: actually exposes. Do not treat this as measured or as a per-server figure
#: just because it is used that way below — it is neither. Do not change the
#: value here; a re-derived, era-correct constant is tracked separately.
#: The server's tool definitions ride in the `tools` array of EVERY call in
#: the session, not just the first — this is the FIRST call's token count;
#: subsequent calls in the same session re-send it too but are priced at the
#: cache-read rate, not this rate again (see the per-session multiplier in
#: the tax loop below).
FULL_SCHEMA_TAX_TOKENS = 25_000

#: When a session's transcript shows this server's tools in a DEFERRED
#: listing (ToolSearch-style), its schemas are NOT loaded that call — only a
#: short name+description line per tool appears in the listing.
#: Conservative estimate: ~10 tools x ~40 tokens/line for a typically-sized
#: server. Never used to claim the full tax for a deferred call. Like
#: FULL_SCHEMA_TAX_TOKENS, this is the first call's count; the same
#: per-session multiplier applies to later calls.
DEFERRED_SCHEMA_TAX_TOKENS = 400

#: Chars-per-token conversion for text measured directly off transcripts
#: (system-reminder blocks) — same convention as prompt_bloat.py's
#: CHARS_PER_TOKEN.
CHARS_PER_TOKEN = 4

DEADWEIGHT_HONESTY_CAVEAT = (
    "Structural detection off configured MCP servers and their measured "
    "tool-call counts, not a judgment about whether the server is useful. "
    "Review the window before removing a server; a low-traffic server can "
    "still be load-bearing for an occasional task."
)


# --- MCP config enumeration (read-only; never writes a config file) -------

_PROJECT_CONFIG_RELPATHS = (
    ".mcp.json", ".claude/settings.json", ".claude/settings.local.json",
)


def _read_json_safe(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _mcp_server_names(path: Path) -> set[str]:
    data = _read_json_safe(path)
    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        return set()
    return {str(name) for name in servers if str(name).strip()}


def _global_config_path(claude_home: Path | None = None) -> Path:
    # Resolved LAZILY (never at import time) so a test patching HOME sees the
    # fake home, never the developer's real ~/.claude.json. `claude_home`
    # scopes it further: under an explicit `--projects-root` the global MCP
    # config read must stay inside the root the caller drew, or a scoped run
    # still reports servers from the operator's real machine.
    if claude_home is not None:
        return claude_home / ".claude.json"
    return Path.home() / ".claude.json"


@dataclass
class ConfiguredServer:
    """One MCP server as read off config, and where it reaches."""
    name:   str
    scope:  str                      # "user" | "project"
    source: str                      # DETERMINISTICALLY chosen config file (for the card/apply)
    cwds:   set[str] = field(default_factory=set)  # project scope: every reachable cwd
    #: Every distinct source path that independently declares this server
    #: (project scope only), mapped to the cwds reachable through THAT file.
    #: The same server name can be declared in more than one physical
    #: ``.mcp.json``/``.claude/settings*.json`` (e.g. one committed into
    #: several worktrees of the same repo) -- `source` above is only ONE of
    #: these, chosen deterministically; the rest are named so a caller never
    #: silently treats the aggregate claim as fixed by editing just one.
    source_cwds: dict[str, set[str]] = field(default_factory=dict)


def enumerate_configured_servers(
    repo_cwds: set[str], *, claude_home: Path | None = None,
) -> dict[str, ConfiguredServer]:
    """Read-only enumeration of MCP servers across the three config
    locations: project ``.mcp.json`` / ``.claude/settings*.json`` under each
    given session cwd, plus the global ``~/.claude.json``. Never edits a
    config file (advise-only in v1 — see the module docstring).

    A user-scoped (global) server always wins scope over a same-named
    project entry: the global entry already reaches every session, so
    downgrading it to "project" would only narrow its true presence.

    ``repo_cwds`` is iterated in SORTED order and each project-scoped
    server's ``source`` is chosen AFTER the full scan (most-cwds-covered
    first, path string as the tie-break) — never "whichever cwd a raw
    ``set`` iteration happened to visit first". The same server name
    independently declared across several cwds (a duplicated ``.mcp.json``
    committed into multiple worktrees, say) used to hand its ONE apply
    target to whichever path a hash-seed-dependent set order visited first,
    so the exact same claim's fix could stop between a sliver and nearly
    all of the claimed tax depending on nothing but interpreter hash
    randomization. Deterministic now, and the deterministic choice is also
    the one that actually matters most (most sessions covered).
    """
    servers: dict[str, ConfiguredServer] = {}

    global_path = _global_config_path(claude_home)
    if global_path.is_file():
        for name in _mcp_server_names(global_path):
            servers[name] = ConfiguredServer(name=name, scope="user", source=str(global_path))

    for cwd in sorted(repo_cwds):
        if not cwd:
            continue
        base = Path(cwd)
        if not base.is_dir():
            continue
        for rel in _PROJECT_CONFIG_RELPATHS:
            path = base / rel
            if not path.is_file():
                continue
            for name in _mcp_server_names(path):
                existing = servers.get(name)
                if existing is not None and existing.scope == "user":
                    continue  # already global, broaden nothing
                entry = servers.setdefault(
                    name, ConfiguredServer(name=name, scope="project", source=str(path)),
                )
                entry.cwds.add(cwd)
                entry.source_cwds.setdefault(str(path), set()).add(cwd)

    for entry in servers.values():
        if entry.scope != "project" or len(entry.source_cwds) <= 1:
            continue
        # Most cwds covered wins; a path-string tie-break makes the choice
        # fully deterministic even when two sources cover the same count.
        entry.source = min(
            entry.source_cwds.items(), key=lambda kv: (-len(kv[1]), kv[0]),
        )[0]
    return servers


def _other_sources(server: ConfiguredServer) -> list[str]:
    """Every source path OTHER than the chosen canonical one that also
    independently declares this server -- sorted, so the disclosure text is
    stable across runs. Empty for a user-scoped or single-source server."""
    return sorted(p for p in server.source_cwds if p != server.source)


def server_still_configured(name: str, source: str) -> bool:
    """Read-only re-check: does ``name`` still appear in the ``mcpServers``
    block of its ORIGINAL detected config file (``source``)?

    Distinguishes "still configured" from "actually removed or
    project-scoped". A missing file and a present-but-empty-of-this-entry
    file both read as "no longer configured" — either way the tax stopped.
    Missing ``name``/``source`` can't be verified at all, so this
    conservatively reports "still configured" rather than falsely claiming a
    removal.
    """
    if not name or not source:
        return True
    path = Path(source)
    if not path.is_file():
        return False
    return name in _mcp_server_names(path)


# --- Deterministic apply: remove one server's entry from its config file --
#
# A dead server's fix is machine-editable — ``ConfiguredServer`` already
# resolved the exact config file, so there is no search step the way
# ``model_apply.model_swap`` needs one. The removal is a TARGETED TEXT SPLICE,
# never a ``json.loads`` -> mutate -> ``json.dumps`` round trip: re-serializing
# the whole document would reformat every byte (key order, indentation,
# spacing), turning a one-entry diff into a wholesale rewrite of the user's
# config. The functions below locate the exact character span of one server's
# entry inside its ``mcpServers`` block by walking the raw text — using
# ``json.decoder.scanstring`` only to skip string literals correctly
# (including escapes), never to reformat anything — and delete only that
# span. Every other byte in the file is untouched.

#: Apply-kind discriminator for this write, carried on the proposal and the
#: ledger record (mirrors ``model_apply.APPLY_KIND_*``).
APPLY_KIND_MCP_REMOVE = "mcp_remove"

_WS_CHARS = " \t\r\n"


def _skip_ws(text: str, i: int) -> int:
    n = len(text)
    while i < n and text[i] in _WS_CHARS:
        i += 1
    return i


def _skip_json_value(text: str, i: int) -> int:
    """Index just past the JSON value starting at ``text[i]`` (already past
    leading whitespace). Handles strings, objects, arrays and bare scalars
    (numbers / true / false / null) — enough to walk any value a ``.mcp.json``
    /``settings.json`` server entry can hold, without needing to know its
    shape ahead of time."""
    ch = text[i]
    if ch == '"':
        _, end = scanstring(text, i + 1)
        return end
    if ch in "{[":
        depth = 1
        i += 1
        n = len(text)
        while depth > 0 and i < n:
            c = text[i]
            if c == '"':
                _, i = scanstring(text, i + 1)
                continue
            if c in "{[":
                depth += 1
            elif c in "}]":
                depth -= 1
            i += 1
        return i
    j = i
    n = len(text)
    while j < n and text[j] not in ",}] \t\r\n":
        j += 1
    return j


def _object_entries(text: str, obj_open: int) -> list[tuple[str, int, int, int]]:
    """Every TOP-LEVEL entry of the object opening at ``text[obj_open] ==
    '{'``, as ``(key, key_start, value_start, value_end)``. Nested keys (a
    server's own ``env``/``args`` block, say) never surface here — depth
    tracking inside ``_skip_json_value`` is what keeps this to one level."""
    entries: list[tuple[str, int, int, int]] = []
    i = _skip_ws(text, obj_open + 1)
    while i < len(text) and text[i] != "}":
        key_start = i
        _key, i = scanstring(text, i + 1)
        i = _skip_ws(text, i)
        i += 1  # the colon
        i = _skip_ws(text, i)
        value_start = i
        value_end = _skip_json_value(text, i)
        entries.append((_key, key_start, value_start, value_end))
        i = _skip_ws(text, value_end)
        if i < len(text) and text[i] == ",":
            i = _skip_ws(text, i + 1)
    return entries


def _mcp_servers_object_open(text: str) -> int | None:
    """Index of the ``{`` opening the top-level ``mcpServers`` object, or
    ``None`` when the document doesn't open with an object or carries no such
    key — the caller falls back to a refusal rather than a guess."""
    root_open = _skip_ws(text, 0)
    if root_open >= len(text) or text[root_open] != "{":
        return None
    for key, _key_start, value_start, _value_end in _object_entries(text, root_open):
        if key == "mcpServers" and value_start < len(text) and text[value_start] == "{":
            return value_start
    return None


def _mcp_server_entry_span(text: str, server_name: str) -> tuple[int, int] | None:
    """The ``(start, end)`` character span to delete from ``text`` to remove
    ``server_name``'s entry from the ``mcpServers`` block, or ``None`` when it
    can't be located this way (no ``mcpServers`` object, or no such key).

    The span always reuses an ADJACENT separator rather than inventing new
    whitespace: removing a first/middle entry keeps the punctuation that sat
    between the PRECEDING entry and this one (which becomes the new
    connector to whatever follows); removing the last (or only) entry instead
    deletes the separator that sat between the entry before it and this one,
    so no dangling trailing comma is left before the closing ``}``.
    """
    obj_open = _mcp_servers_object_open(text)
    if obj_open is None:
        return None
    entries = _object_entries(text, obj_open)
    idx = next((i for i, e in enumerate(entries) if e[0] == server_name), None)
    if idx is None:
        return None
    _key, key_start, _value_start, value_end = entries[idx]
    is_last = idx == len(entries) - 1
    if is_last:
        prev_end = entries[idx - 1][3] if idx > 0 else obj_open + 1
        return prev_end, value_end
    next_key_start = entries[idx + 1][1]
    return key_start, next_key_start


def render_mcp_remove(pre_image: str | None, server_name: str) -> tuple[str | None, str]:
    """The config file's new content with ``server_name``'s entry removed
    from its ``mcpServers`` block.

    Returns ``(content, "")`` on success and ``(None, reason)`` when the
    removal cannot be made deterministically: no file, invalid JSON, no
    ``mcpServers`` block, or the server no longer named there (already
    removed by hand, or by a concurrent edit).
    """
    if not server_name:
        return None, "no server name given for the MCP removal."
    if pre_image is None:
        return None, "no config file at that path to edit."
    try:
        doc = json.loads(pre_image)
    except ValueError as exc:
        return None, f"that file is not valid JSON ({exc}) — refusing to edit it."
    servers = doc.get("mcpServers") if isinstance(doc, dict) else None
    if not isinstance(servers, dict) or server_name not in servers:
        return None, f"`{server_name}` is not in that file's mcpServers block any more."
    span = _mcp_server_entry_span(pre_image, server_name)
    if span is None:
        return None, (
            f"could not locate `{server_name}`'s entry precisely in the file "
            f"text — refusing a risky edit."
        )
    start, end = span
    return pre_image[:start] + pre_image[end:], ""


def mcp_remove_precheck(source_path: str, server_name: str) -> dict:
    """Whether ``server_name``'s entry may be removed from ``source_path``,
    re-checked at apply time — the repo can have moved, the file can have
    gone missing, or a human can have already hand-removed the entry between
    the moment the card was built and the moment it is approved.

    Every precondition must hold; any failure returns ``{"ok": False,
    "reason": ...}`` and the caller falls back to the one-paste ``claude mcp
    remove`` command, saying why on the card.
    """
    if not source_path or not server_name:
        return {"ok": False, "reason": (
            "no source config path or server name given for this MCP removal."
        )}
    path = Path(source_path).expanduser()
    if not path.is_file():
        return {"ok": False, "reason": f"{path} no longer exists on disk — nothing to edit."}
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"ok": False, "reason": f"{path} is not valid JSON ({exc}) — refusing to edit it."}
    if not server_still_configured(server_name, str(path)):
        return {"ok": False, "reason": (
            f"`{server_name}` is no longer in {path}'s mcpServers block — it may "
            f"already have been removed by hand."
        )}
    return {"ok": True, "reason": "", "target_path": str(path)}


def build_mcp_remove_plan(cluster: dict, target: Path, pre_image: str | None) -> str:
    """New content for an ``mcp_remove`` apply: ``cluster``'s server entry
    deleted from its ``mcpServers`` block, every other byte untouched.

    Raises ``RelearnApplyRefused`` with the refusal reason, which the API
    layer surfaces as a 409 and the card renders as the fallback explanation
    — mirrors ``model_apply.build_model_plan``'s contract so the two slot
    into the same ``relearn_apply._build_write_plan`` dispatch.
    """
    from tokenjam.core.optimize.relearn_apply import RelearnApplyRefused

    server_name = str(cluster.get("agent_name") or "")
    source_path = str(cluster.get("source_path") or "")
    check = mcp_remove_precheck(source_path, server_name)
    if not check["ok"]:
        raise RelearnApplyRefused(check["reason"])
    if Path(check["target_path"]) != target:
        raise RelearnApplyRefused(
            f"the MCP config now lives at {check['target_path']}, not {target}. "
            f"Refusing to write the stale target."
        )
    content, reason = render_mcp_remove(pre_image, server_name)
    if content is None:
        raise RelearnApplyRefused(reason)
    return content


# --- Transcript scanning ---------------------------------------------------

#: ``mcp__<server>__<tool>`` — server names may contain single underscores
#: (e.g. ``claude_ai_Apollo_io``) but never a literal ``__``, which is the
#: delimiter to the tool name; the non-greedy group stops at the FIRST ``__``.
_MCP_TOOL_NAME_RE = re.compile(r"^mcp__([A-Za-z0-9][\w.-]*?)__")
_MCP_TOOL_MENTION_RE = re.compile(r"\bmcp__([A-Za-z0-9][\w.-]*?)__")
_DEFERRED_MARKER_RE = re.compile(r"deferred tool", re.IGNORECASE)
_TOOLSEARCH_MARKER_RE = re.compile(r"toolsearch", re.IGNORECASE)
#: Claude Code emits a "Contents of <path> (...)" heading ahead of each doc it
#: injects verbatim into a system-reminder block (CLAUDE.md, rules files,
#: MEMORY.md, ...) — the C2 tax-table splitter keys off these.
_CONTENTS_OF_RE = re.compile(r"Contents of ([^\n(:]+)", re.IGNORECASE)


def _mcp_server_from_tool_name(tool_name: str) -> str | None:
    """``mcp__<server>__<tool>`` -> ``<server>``, else None for a non-MCP tool."""
    match = _MCP_TOOL_NAME_RE.match(tool_name or "")
    return match.group(1) if match else None


def _session_cwd(records: list[dict[str, Any]]) -> str:
    """Best-effort session cwd, from the first record that carries one.

    Delegates to ``core.transcript.first_recorded_cwd``, which is the one
    extractor this question has — relearn's repo-label map and rule placement
    read it too, and three copies of the same loop is three chances for them to
    disagree about which repo a session ran in.
    """
    from tokenjam.core.transcript import first_recorded_cwd

    return first_recorded_cwd(records)


def _bucket_for_doc(label: str) -> str:
    base = label.strip().rstrip("):").strip()
    name = base.rsplit("/", 1)[-1].lower()
    if name == "claude.md":
        return "CLAUDE.md"
    if name == "learnings.md":
        return "learnings.md"
    if "/rules/" in base.lower() or base.lower().startswith("rules/"):
        return "rules files"
    return "other referenced docs"


def _split_reminder_sources(blob: str) -> dict[str, int]:
    """Bucket ONE system-reminder blob's char count by source, using the
    ``Contents of <path> (...)`` headings ahead of each injected doc.
    Whatever isn't inside a doc segment (environment info, date, hook output)
    is lumped into "session-start hook output & environment" — finer
    attribution of that residual isn't attempted (heuristic, `estimated`).
    """
    matches = list(_CONTENTS_OF_RE.finditer(blob))
    buckets: dict[str, int] = {}
    doc_chars = 0
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(blob)
        segment_len = end - start
        doc_chars += segment_len
        bucket = _bucket_for_doc(m.group(1))
        buckets[bucket] = buckets.get(bucket, 0) + segment_len
    other = len(blob) - doc_chars
    if other > 0:
        buckets["session-start hook output & environment"] = (
            buckets.get("session-start hook output & environment", 0) + other
        )
    return buckets


@dataclass
class _SessionSignal:
    mcp_invocations: dict[str, int] = field(default_factory=dict)
    deferred_servers: set[str] = field(default_factory=set)
    #: server name -> assistant-turn ordinal (1-indexed, matches
    #: ``assistant_turns`` at the moment) of that server's FIRST tool_use in
    #: this session — the earliest point with POSITIVE evidence the schema was
    #: fully loaded. A deferred tool's own listing states calling it directly
    #: "will fail ... use ToolSearch to load their schema before calling
    #: them", so a successful invocation implies the schema was already loaded
    #: by that turn. A server never invoked in this session (every dead-weight
    #: candidate, by definition) has no entry here and is conservatively
    #: treated as deferred for its whole presence — matching the "never claim
    #: the full tax for a deferred call without evidence" discipline.
    first_invocation_turn: dict[str, int] = field(default_factory=dict)
    reminder_chars_by_source: dict[str, int] = field(default_factory=dict)
    #: assistant-turn model -> turn count, for pricing the token tax at a
    #: representative model's input rate (see ``_dominant_model``).
    models: dict[str, int] = field(default_factory=dict)
    #: Total assistant turns in the session — the session's ACTUAL call count
    #: (mirrors context_diagnostic.py's one-``TurnComposition``-per-turn
    #: convention). NOT one per "role == assistant" transcript record: Claude
    #: Code writes a SEPARATE record per content block (thinking / text /
    #: tool_use) of a single API response, all sharing one ``message.id``, so
    #: counting records one-for-one overcounts the real call count by
    #: however many blocks a response happened to split into. Deduped by
    #: message key in ``_analyze_session`` below — the same key
    #: ``_session_usage_from_records`` already dedupes on, for the identical
    #: reason. A configured MCP server's tool schemas ride in the ``tools``
    #: array of every one of these (deduped) calls, not just the first — see
    #: the per-call multiplier in the tax loop below.
    assistant_turns: int = 0


def _analyze_session(records: list[dict[str, Any]]) -> _SessionSignal:
    """One pass over a session's raw records: MCP invocation counts, which
    servers appeared in a deferred-tools listing, the C2 tax-table source
    buckets (measured off the FIRST system-reminder blob only — Claude Code
    injects it once at session start; later turns don't repeat it, so
    summing across turns would overcount), and the assistant model(s) used
    (for pricing the token tax).

    Claude Code writes a SEPARATE transcript record per content block
    (thinking / text / tool_use) of the same API response, all sharing one
    ``message.id`` — counting "role == assistant" records one-for-one with
    API calls overcounts by however many blocks a response happened to split
    into. ``assistant_turns`` dedupes by ``assistant_message_key`` (the same
    key ``_session_usage_from_records`` already uses for its own dedup, for
    the identical reason) so it stays the session's ACTUAL call count, not
    its record count.
    """
    signal = _SessionSignal()
    reminder_measured = False
    seen_message_keys: set[str] = set()

    for line_no, record in enumerate(records, start=1):
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        role = message.get("role") or record.get("type")

        text = content if isinstance(content, str) else ""
        blocks = content if isinstance(content, list) else []
        if not text and blocks:
            text = "\n".join(
                b.get("text", "") for b in blocks
                if isinstance(b, dict) and b.get("type") == "text"
            )

        if text:
            if _DEFERRED_MARKER_RE.search(text) and _TOOLSEARCH_MARKER_RE.search(text):
                signal.deferred_servers.update(_MCP_TOOL_MENTION_RE.findall(text))
            if not reminder_measured and role == "user":
                reminder_blobs = _SYSTEM_REMINDER_RE.findall(text)
                if reminder_blobs:
                    signal.reminder_chars_by_source = _split_reminder_sources(reminder_blobs[0])
                    reminder_measured = True

        if role == "assistant":
            # Dedupe by message key before counting a call: Claude Code can
            # write several records for ONE API response (one per content
            # block, e.g. a thinking block and a text block each get their
            # own record but share `message.id`). Counting every "assistant"
            # record here would overcount how many times the schema tax was
            # actually re-sent -- do not "simplify" this back to a bare
            # per-record increment.
            message_key = assistant_message_key(record, message, line_no)
            if message_key not in seen_message_keys:
                seen_message_keys.add(message_key)
                signal.assistant_turns += 1
                model = message.get("model")
                if isinstance(model, str) and model:
                    signal.models[model] = signal.models.get(model, 0) + 1

        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                server = _mcp_server_from_tool_name(str(block.get("name") or ""))
                if server:
                    signal.mcp_invocations[server] = signal.mcp_invocations.get(server, 0) + 1
                    # `assistant_turns` was already incremented above for THIS
                    # record's call (deduped), so this is the 1-indexed
                    # ordinal of the call that's doing the invoking.
                    signal.first_invocation_turn.setdefault(server, signal.assistant_turns)

    return signal


def _dominant_model(model_counts: dict[str, int]) -> str:
    """The most-frequent assistant model across the counted turns, or "" when
    none were observed. Ties broken by first-seen (dict insertion order)."""
    if not model_counts:
        return ""
    return max(model_counts.items(), key=lambda kv: kv[1])[0]


def _tax_construction_note(
    non_deferred: int, deferred_sessions: int, sessions_present: int,
    *, model: str = "", input_per_mtok: float | None = None,
    usd_per_session: float | None = None,
    avg_calls_per_session: float = 1.0,
    cache_read_ratio: float = 0.0,
) -> str:
    if sessions_present == 0:
        return ""
    if deferred_sessions == 0:
        note = (
            f"{FULL_SCHEMA_TAX_TOKENS:,} tok on the first call (full schema "
            f"injection), cited estimate, not a live per-call measurement."
        )
    elif non_deferred == 0:
        note = (
            f"{DEFERRED_SCHEMA_TAX_TOKENS:,} tok on the first call; ToolSearch "
            f"deferred this server's schemas in every observed session (name "
            f"and description line only, never the full schema tax)."
        )
    else:
        note = (
            f"{FULL_SCHEMA_TAX_TOKENS:,} tok on the first call when fully "
            f"loaded ({non_deferred} of {sessions_present} sessions) blended "
            f"with {DEFERRED_SCHEMA_TAX_TOKENS:,} tok when ToolSearch defers "
            f"this server's schemas ({deferred_sessions} of {sessions_present} "
            f"sessions); never claims the full tax for a deferred call."
        )
    if cache_read_ratio > 0:
        note += (
            f" The schema rides in the `tools` array of every call in the "
            f"session, not just the first: priced at the input rate once, "
            f"then at the cache-read rate (~{cache_read_ratio * 100:.0f}% of "
            f"input) on every later call (avg {avg_calls_per_session:.1f} "
            f"call(s)/session across these sessions, from each session's own "
            f"actual call count, never a global mean/median). Simplification: "
            f"assumes every call in a session lands inside Anthropic's "
            f"5-minute cache TTL; a call after a longer gap would instead "
            f"re-write the schema at the higher cache-write rate for that "
            f"call, so this understates sessions with long gaps between calls."
        )
    else:
        note += (
            " Charged once per session (no cache-read rate available for "
            "the priced model to compound later calls against)."
        )
    return note + " " + _pricing_note(model, input_per_mtok, usd_per_session)


def _pricing_note(model: str, input_per_mtok: float | None, usd_per_session: float | None) -> str:
    """The dollar-conversion clause appended to a server's construction
    footnote. Never fabricates a rate: when no priced model was observed
    across the server's sessions, states that plainly and stays tokens-only.
    """
    if not model or input_per_mtok is None or usd_per_session is None:
        return (
            "No dollar estimate: no priced model observed across these "
            "sessions (core/pricing.py has no rate for it); tokens only."
        )
    return (
        f"Priced at {model}'s input rate (${input_per_mtok:.2f}/MTok via "
        f"core/pricing.py) -> ${usd_per_session:,.4f}/session estimated."
    )


# --- Proposal + finding shapes ---------------------------------------------

@dataclass
class ServerDeadweight:
    """One configured MCP server's presence/invocation signal in the window."""
    name:                             str
    scope:                            str    # "user" | "project"
    source:                           str
    sessions_present:                 int
    invocations:                      int
    deferred_sessions:                int
    dead:                             bool
    #: The LITERAL (undiscounted) count of schema-injection tokens sent —
    #: the schema rides in the `tools` array of every call regardless of
    #: caching, so this is bytes actually transmitted, never the
    #: cache-discounted price-equivalent quantity `estimated_tax_usd_*` is
    #: derived from. Keeping these two axes on the same undiscounted-vs-
    #: price-equivalent basis would make one of them silently answer a
    #: different question than its name claims (Critical Rule 28); this
    #: field used to BE that price-equivalent quantity relabeled as tokens,
    #: understating the real count by roughly 1/cache_read_ratio.
    estimated_tax_tokens_per_session: int
    #: Window-scoped total: the SUM of each present session's own literal
    #: token count (first call's full schema + every later call's full
    #: re-send — see the per-session multiplier in
    #: `compute_deadweight_finding`), never
    #: `estimated_tax_tokens_per_session x sessions_present` (that would
    #: silently substitute the average call count for every session's actual
    #: one). NO projection folded in, and none may be applied downstream
    #: either: this is a past-tense window observation, and the product states
    #: no forward per-analyzer figure at all (see the field contract in the
    #: repo `CLAUDE.md`). This field used to fold a fixed 90-day projection in
    #: directly, which made it incomparable to every other analyzer's
    #: window-scoped figure and silently corrupted the rollup's basis when
    #: summed alongside them.
    estimated_tax_tokens_window:      int
    tax_construction:                 str
    fix:                              str
    example_sessions:                 list[str] = field(default_factory=list)
    #: Dollar conversion of the token tax, priced through core/pricing.py at
    #: the dominant model observed across this server's present sessions.
    #: ``None`` when no priced model was observed (never a fabricated rate).
    priced_model:                     str = ""
    estimated_tax_usd_per_session:    float | None = None
    #: Window-scoped dollar total — see `estimated_tax_tokens_window` above.
    estimated_tax_usd_window:         float | None = None
    #: Every OTHER source path that independently declares this same server
    #: name (project scope only) -- e.g. a duplicated `.mcp.json` committed
    #: into several worktrees. Empty when `source` is the only place this
    #: server is declared. NOT touched by the apply/one-paste fix, which
    #: targets `source` alone; see `primary_source_sessions` for how much of
    #: `sessions_present` that one edit actually reaches.
    other_sources:                    list[str] = field(default_factory=list)
    #: Of `sessions_present`, how many run from a cwd reachable through
    #: `source` specifically (as opposed to one of `other_sources`). Equals
    #: `sessions_present` whenever `other_sources` is empty. A claim
    #: aggregated across multiple source files must not let its one fix
    #: action silently imply full coverage.
    primary_source_sessions:         int = 0


@dataclass
class ContextTaxRow:
    """One always-injected content source's measured/estimated per-session tax."""
    source:                 str
    sessions:                int
    avg_tokens_per_session:  int
    total_tokens_window:      int
    tag:                       str = "estimated"
    construction:               str = ""


@dataclass
class DeadweightFinding:
    sessions_scanned:             int = 0
    configured_servers:           int = 0
    servers:                      list[ServerDeadweight] = field(default_factory=list)
    dead_servers:                 list[ServerDeadweight] = field(default_factory=list)
    tax_table:                    list[ContextTaxRow] = field(default_factory=list)
    past_overspend_tokens: int | None = None
    past_overspend_usd:    float | None = None
    estimate_basis:                str = ""
    estimate_confidence:            str = "estimated"
    caveat:                          str = DEADWEIGHT_HONESTY_CAVEAT
    notes:                            list[str] = field(default_factory=list)
    #: Distinct recorded session cwds that no longer exist on disk (a deleted
    #: worktree, typically) — `enumerate_configured_servers` silently
    #: `continue`s past these, so a vanished repo was previously
    #: indistinguishable, on this finding, from a live repo genuinely
    #: carrying no MCP config. Counted here so that blind spot is visible
    #: instead of silent (see `_unresolvable_coverage_note`).
    unresolvable_paths:               int = 0
    #: Sessions whose recorded cwd falls among `unresolvable_paths` above —
    #: this analyzer could not check project-scoped MCP config for any of
    #: them at all.
    unresolvable_sessions:            int = 0
    #: Total ACTUAL usage tokens (input+output+cache-read+cache-write, not
    #: the MCP schema tax) across `unresolvable_sessions`, measured directly
    #: off each session's transcript.
    unresolvable_tokens:              int = 0
    #: Dollar conversion of `unresolvable_tokens`, priced per session through
    #: core/pricing.py at that session's dominant model. `None` when no
    #: priced model was observed across ANY unresolvable session (never a
    #: fabricated rate) — see `unresolvable_unpriced_sessions` for how many
    #: of them are excluded from this sum.
    unresolvable_usd:                 float | None = None
    #: Of `unresolvable_sessions`, how many had no priced model observed and
    #: are therefore excluded from `unresolvable_usd` (the token figure
    #: above still includes them).
    unresolvable_unpriced_sessions:   int = 0
    #: Plain-language statement of the blind spot above, so a reader never
    #: has to infer "vanished repo" vs. "genuinely no config" from silence.
    #: Mirrors `context_resend._coverage_note` / `_relearn_coverage_note`.
    coverage_note:                    str = ""


# --- Unresolvable-path coverage (Defect 1: silence when a cwd is gone) ----

def _session_usage_from_records(records: list[dict[str, Any]]) -> AssistantUsage:
    """Total ACTUAL assistant usage over one already-parsed session,
    last-wins deduped by message key — same dedup policy as
    ``core.usage.session_usage``, just replayed over records this module
    already parsed once rather than re-reading the raw lines a second time.

    This is real usage (what the session actually cost), never to be
    confused with ``FULL_SCHEMA_TAX_TOKENS``/``DEFERRED_SCHEMA_TAX_TOKENS``
    above, which model the MCP schema-injection TAX, not actual spend.
    """
    by_key: dict[str, AssistantUsage] = {}
    for line_no, record in enumerate(records, start=1):
        if record.get("type") != "assistant":
            continue
        msg = record.get("message")
        if not isinstance(msg, dict) or not msg.get("usage"):
            continue
        usage = parse_usage(msg.get("usage"))
        if usage.total == 0:
            continue
        by_key[assistant_message_key(record, msg, line_no)] = usage
    total = AssistantUsage()
    for usage in by_key.values():
        total = AssistantUsage(
            total.input_tokens + usage.input_tokens,
            total.output_tokens + usage.output_tokens,
            total.cache_read_tokens + usage.cache_read_tokens,
            total.cache_write_tokens + usage.cache_write_tokens,
        )
    return total


def _session_actual_usd(
    usage: AssistantUsage, model: str, *, at: datetime,
) -> float | None:
    """Real dollar cost of one session's ACTUAL usage, priced through
    core/pricing.py at ``model`` — all four token buckets (input, output,
    cache-read, AND cache-write; see root CLAUDE.md's "cache token types in
    aggregates" note on why both cache buckets must be included or the total
    is silently short). ``None`` when ``model`` is empty or unpriced — never
    a fabricated rate.

    ``at`` is when the session ran, and is required: this is an OBSERVED cost,
    so it must use the rate that actually billed it rather than today's.
    """
    if not model:
        return None
    from tokenjam.core.pricing import provider_for_model

    usd = price_span(
        provider_for_model(model) or "unknown", model, at=at,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        cache_write_tokens=usage.cache_write_tokens,
    )
    # Kept at 6dp (price_span rounds to 8) so the per-session figures this
    # returns are bit-for-bit what they were before the pricer swap.
    return None if usd is None else round(usd, 6)


def _unresolvable_coverage_note(finding: DeadweightFinding) -> str:
    """State, in words, the blind spot a vanished recorded cwd leaves on
    this finding.

    The defect this closes: ``enumerate_configured_servers`` silently
    ``continue``s past a recorded session cwd that no longer exists on disk
    (a deleted worktree, typically) — indistinguishable, on the finding, from
    a live repo that genuinely carries no MCP config. Both used to read as
    "nothing to flag" to a reader; only one of them was actually checked.
    """
    if finding.unresolvable_sessions <= 0:
        return ""
    parts = [
        f"COVERAGE. {finding.unresolvable_sessions:,} of "
        f"{finding.sessions_scanned:,} session(s) in this window recorded a "
        f"working directory that no longer exists on disk "
        f"({finding.unresolvable_paths:,} distinct path(s), typically a "
        f"deleted worktree). This analyzer could not read that project's "
        f"`.mcp.json` / `.claude/settings*.json` for any of them. Its "
        f"silence there is NOT evidence the config was clean; it is "
        f"evidence the analyzer never got to look."
    ]
    if finding.unresolvable_usd is not None:
        parts.append(
            f"${finding.unresolvable_usd:,.2f} of spend sits behind those "
            f"sessions, priced per session through core/pricing.py at each "
            f"session's dominant model."
        )
        if finding.unresolvable_unpriced_sessions:
            parts.append(
                f"{finding.unresolvable_unpriced_sessions:,} of those "
                f"session(s) had no priced model observed and are excluded "
                f"from that dollar sum (the token figure still includes "
                f"them)."
            )
    elif finding.unresolvable_tokens:
        parts.append(
            f"~{finding.unresolvable_tokens:,} tok of usage sits behind "
            f"those sessions; no priced model was observed across them, so "
            f"no dollar figure is stated."
        )
    parts.append(
        "A user-scoped server (global ~/.claude.json) is unaffected by "
        "this: it resolves regardless of a session's cwd. Only "
        "project-scoped detection is blind here."
    )
    return " ".join(parts)


# --- Orchestration (pure, no ctx dependency — testable directly) ----------

def compute_deadweight_finding(
    since: datetime,
    until: datetime,
    *,
    projects_root: Path | str | None = None,
    claude_home: Path | None = None,
    min_sessions: int = MIN_SESSIONS_DEADWEIGHT,
    cache_dir: Path | None = None,
) -> DeadweightFinding:
    """Full pipeline over a window of Claude Code transcripts. Never raises —
    a missing projects root, an unreadable transcript, or a malformed config
    file is skipped, not fatal.

    Emits ``estimated_tax_tokens_window``/``estimated_tax_usd_window`` — the
    tax OBSERVED over ``since``..``until``, with no internal projection
    (#273). A forward-looking figure, when wanted, is the caller's job: the
    Review-inbox rollup applies one shared, centrally-computed 30-day-pace
    ratio on top of every cost analyzer's window figure alike, so a window
    parameter has nothing left to do here.

    ``min_sessions`` overrides ``MIN_SESSIONS_DEADWEIGHT`` (config-overridable
    via ``core.config.OptimizeConfig.min_sessions_deadweight``); the module
    constant remains the default so a caller that omits it sees today's
    behaviour unchanged.

    ``cache_dir``, when given, transparently caches each transcript's parsed
    records on disk (``core.transcript_cache``) so a re-run over an unchanged
    corpus skips the read + parse entirely. ``None`` (the default) preserves
    this function's original always-reparse behavior — only the registered
    ``run(ctx)`` entry point below opts in, so this function's existing
    "no I/O beyond the passed-in tmp_path root" test contract is unchanged
    for direct callers.
    """
    finding = DeadweightFinding()
    root = resolve_projects_root(projects_root)
    if not root.exists():
        return finding

    session_paths: list[tuple[str, Path]] = []
    # Transcript mtime per session — the only time signal this analyzer has (it
    # reads JSONL files, not spans) and the instant its dollar figures are
    # priced at. Without it every session would price at today's list rate; see
    # `tokenjam.core.optimize.span_pricing` for the convention.
    session_mtimes: dict[str, datetime] = {}
    for path in sorted(root.rglob("*.jsonl")):
        # Subagent/sidechain transcripts live at
        # `<parent-session-dir>/subagents/agent-<id>.jsonl` (core/transcript.py)
        # -- nested under `root`, so a plain rglob picks them up too. Counting
        # one as its own top-level "session" would double the call count fed
        # into the per-session schema-tax multiplier below (and, for a
        # user-scoped server, spuriously inflate `sessions_present` toward the
        # dead-weight threshold) purely because the parent session happened to
        # spawn a subagent -- never the parent session's own ACTUAL call count.
        if path.parent.name == "subagents":
            continue
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            continue
        if mtime < since or mtime >= until:
            continue
        session_paths.append((path.stem, path))
        session_mtimes[path.stem] = mtime

    finding.sessions_scanned = len(session_paths)
    if not session_paths:
        return finding

    per_session: dict[str, _SessionSignal] = {}
    session_cwds: dict[str, str] = {}
    session_usages: dict[str, AssistantUsage] = {}
    for session_id, path in session_paths:
        try:
            records = read_records(path, cache_dir=cache_dir)
        except Exception:
            continue
        session_cwds[session_id] = _session_cwd(records)
        per_session[session_id] = _analyze_session(records)
        session_usages[session_id] = _session_usage_from_records(records)

    repo_cwds = {c for c in session_cwds.values() if c}

    # Unresolvable-path coverage (Defect 1): computed BEFORE the
    # `if not configured: return` below, and before `configured` is even
    # enumerated — the worst case (every recorded path gone) is exactly the
    # one where `configured` comes back empty, so this must never sit behind
    # that early return or it would go silent in precisely the case it
    # exists to surface.
    unresolvable_cwds = {c for c in repo_cwds if not Path(c).is_dir()}
    if unresolvable_cwds:
        unresolvable_tokens = 0
        unresolvable_usd = 0.0
        priced_any = False
        unpriced_sessions = 0
        unresolvable_session_count = 0
        for session_id, cwd in session_cwds.items():
            if cwd not in unresolvable_cwds:
                continue
            unresolvable_session_count += 1
            usage = session_usages.get(session_id, AssistantUsage())
            unresolvable_tokens += usage.total
            model = _dominant_model(per_session[session_id].models)
            usd = _session_actual_usd(
                usage, model,
                at=span_instant(session_mtimes.get(session_id), window_start=since),
            )
            if usd is None:
                unpriced_sessions += 1
            else:
                unresolvable_usd += usd
                priced_any = True
        finding.unresolvable_paths = len(unresolvable_cwds)
        finding.unresolvable_sessions = unresolvable_session_count
        finding.unresolvable_tokens = unresolvable_tokens
        finding.unresolvable_usd = round(unresolvable_usd, 6) if priced_any else None
        finding.unresolvable_unpriced_sessions = unpriced_sessions
        finding.coverage_note = _unresolvable_coverage_note(finding)

    configured = enumerate_configured_servers(repo_cwds, claude_home=claude_home)
    finding.configured_servers = len(configured)
    if not configured:
        return finding

    from tokenjam.core.pricing import provider_for_model

    tax_rows: list[ContextTaxRow] = []
    reminder_bucket_totals: dict[str, list[int]] = {}

    for server in configured.values():
        sessions_present = 0
        invocations = 0
        deferred_sessions = 0
        primary_source_sessions = 0
        example_sessions: list[str] = []
        model_counts: dict[str, int] = {}
        # (when a present session ran, how many of this server's priced calls it
        # carried) — the weights `blended_rates` needs to give this server ONE
        # rate that equals pricing each session at its own instant and summing.
        model_volume_at: list[tuple[datetime | None, float]] = []
        # (deferred_calls, full_calls) per present session — each session's
        # OWN call count feeds its own cache-read multiplier below; never a
        # global mean/median (call-count distribution is severely
        # right-skewed, and the affected population is only sessions where
        # this server was present). Split per CALL, not once per session: a
        # server deferred early in a session and then actually invoked later
        # (schema fully loaded from that call on, per the deferred listing's
        # own "use ToolSearch to load their schema before calling them")
        # must not have every call in that session priced at the low
        # deferred base — only the calls before the load point.
        session_presence: list[tuple[int, int]] = []
        for session_id, signal in per_session.items():
            deferred_here = server.name in signal.deferred_servers
            present = deferred_here or server.scope == "user" or (
                session_cwds.get(session_id, "") in server.cwds
            )
            if not present:
                continue
            sessions_present += 1
            # How much of the claim below the ONE fix action (editing
            # `server.source`) actually reaches -- a server independently
            # declared in more than one config file must not let removing
            # just one silently read as removing the whole claimed tax.
            if server.scope == "user" or session_cwds.get(session_id, "") in (
                server.source_cwds.get(server.source) or set()
            ):
                primary_source_sessions += 1
            invocations += signal.mcp_invocations.get(server.name, 0)
            if deferred_here:
                deferred_sessions += 1
            if len(example_sessions) < MAX_EXAMPLE_SESSIONS:
                example_sessions.append(session_id)
            for model, count in signal.models.items():
                model_counts[model] = model_counts.get(model, 0) + count
                model_volume_at.append((session_mtimes.get(session_id), float(count)))

            total_calls = max(signal.assistant_turns, 1)
            if deferred_here:
                load_turn = signal.first_invocation_turn.get(server.name)
                # No invocation ever seen for this server in this session
                # (true for every dead-weight candidate, by definition): no
                # positive evidence the schema was ever fully loaded here, so
                # stay conservative and treat the whole session as deferred.
                deferred_calls = min(load_turn - 1, total_calls) if load_turn else total_calls
            else:
                deferred_calls = 0
            session_presence.append((max(deferred_calls, 0), total_calls - max(deferred_calls, 0)))

        dead = sessions_present >= min_sessions and invocations == 0
        non_deferred = max(sessions_present - deferred_sessions, 0)

        # Price the token tax through core/pricing.py at the dominant model
        # observed across this server's present sessions -- never a
        # hardcoded rate. usd/cache_read_ratio stay at their no-op defaults
        # when no priced model was seen.
        priced_model = _dominant_model(model_counts)
        input_per_mtok: float | None = None
        cache_read_ratio = 0.0
        if priced_model:
            provider = provider_for_model(priced_model) or "unknown"
            # One rate for the server's whole present population, blended across
            # the sessions' own dates and weighted by their call volume — the
            # aggregate-safe form of "price each session at its own rate, then
            # sum" (see span_pricing). Identical to a single lookup whenever no
            # rate moved inside the window, which is the usual case.
            rates = blended_rates(provider, priced_model, model_volume_at)
            if rates is not None and rates.input_per_mtok > 0:
                input_per_mtok = rates.input_per_mtok
                cache_read_ratio = rates.cache_read_per_mtok / rates.input_per_mtok
            else:
                priced_model = ""  # no rate available -- don't claim a model we can't price

        # The schema rides in the `tools` array of EVERY call in a session,
        # not just once — after the first call it's re-billed as a cache read
        # (~10% of input rate for every Anthropic model in pricing/
        # models.toml), not re-charged at full input rate again. Computed per
        # session from that session's own call count and summed for the
        # window total; tax_per_session below is the resulting average, for
        # display only — never the basis for tax_window itself. A session
        # whose deferred/full split changes partway through prices each
        # segment separately: the segment's own first call at the full
        # per-call base (the content just changed, so nothing is cached yet
        # for it), later calls in that SAME segment at the cache-read rate.
        #
        # `tax_window` is a PRICE-EQUIVALENT quantity (the cache discount
        # already folded in via `cache_read_ratio`) -- correct as the input
        # for the $ conversion below, but NOT a token count: a session's
        # schema is resent in full on every call regardless of caching, so
        # the actual bytes transmitted are larger than this by roughly
        # 1/cache_read_ratio. `tokens_window_real` tracks the literal
        # (undiscounted) count in parallel so `estimated_tax_tokens_*` can
        # answer "how many tokens were actually sent" rather than silently
        # reporting a $-shaped number through a field named `tokens`
        # (Critical Rule 28: both fields must answer the same question).
        tax_window = 0
        tokens_window_real = 0
        total_calls = 0
        for deferred_calls, full_calls in session_presence:
            if deferred_calls > 0:
                tax_window += round(
                    DEFERRED_SCHEMA_TAX_TOKENS * (1.0 + (deferred_calls - 1) * cache_read_ratio)
                )
                tokens_window_real += DEFERRED_SCHEMA_TAX_TOKENS * deferred_calls
            if full_calls > 0:
                tax_window += round(
                    FULL_SCHEMA_TAX_TOKENS * (1.0 + (full_calls - 1) * cache_read_ratio)
                )
                tokens_window_real += FULL_SCHEMA_TAX_TOKENS * full_calls
            total_calls += deferred_calls + full_calls
        tax_per_session = round(tax_window / sessions_present) if sessions_present else 0
        tokens_per_session_real = (
            round(tokens_window_real / sessions_present) if sessions_present else 0
        )
        avg_calls_per_session = (total_calls / sessions_present) if sessions_present else 1.0

        usd_per_session: float | None = None
        usd_window: float | None = None
        if input_per_mtok is not None:
            usd_per_session = round(tax_per_session / 1_000_000 * input_per_mtok, 6)
            usd_window = round(tax_window / 1_000_000 * input_per_mtok, 6)

        other_sources = _other_sources(server)
        # The CLAIM stays aggregated across every source that declares this
        # server (that is what actually happened this window), but the FIX
        # is deliberately scoped to the one canonical `source` rather than
        # made multi-target: extending the single-file MCP-remove apply path
        # to edit several files atomically is a real, separate feature, not
        # a one-line change, and disclosing the gap here is what keeps a
        # partial fix honest in the meantime. Do not "complete" this by
        # silently widening the claim to imply the one apply now covers
        # every source -- widen the apply first, or leave both as they are.
        # "Remove or project-scope" is only a real two-option choice for a
        # USER-scoped (global) server -- narrowing it to project scope is an
        # actual alternative. A server already at project scope has nothing
        # left to narrow: offering "project-scope it" there is a no-op that
        # would deliver $0 of the claim, not a genuine second option.
        action = "Remove or project-scope" if server.scope == "user" else "Remove"
        fix = (
            f"{action} the `{server.name}` MCP server ({server.source}); "
            f"zero tool calls across {sessions_present} session(s) in this "
            f"window."
        )
        if other_sources:
            plural = "" if len(other_sources) == 1 else "s"
            fix += (
                f" `{server.name}` is ALSO independently declared in "
                f"{len(other_sources)} other location{plural}, not touched "
                f"by this edit: {'; '.join(other_sources)}. Removing only "
                f"{server.source} stops the tax for {primary_source_sessions} "
                f"of the {sessions_present} session(s) counted here; the "
                f"remaining {sessions_present - primary_source_sessions} "
                f"session(s) need their own location edited too."
            )

        row = ServerDeadweight(
            name=server.name,
            scope=server.scope,
            source=server.source,
            sessions_present=sessions_present,
            invocations=invocations,
            deferred_sessions=deferred_sessions,
            dead=dead,
            estimated_tax_tokens_per_session=tokens_per_session_real,
            estimated_tax_tokens_window=tokens_window_real,
            tax_construction=_tax_construction_note(
                non_deferred, deferred_sessions, sessions_present,
                model=priced_model, input_per_mtok=input_per_mtok,
                usd_per_session=usd_per_session,
                avg_calls_per_session=avg_calls_per_session,
                cache_read_ratio=cache_read_ratio,
            ),
            fix=fix,
            example_sessions=example_sessions,
            priced_model=priced_model,
            estimated_tax_usd_per_session=usd_per_session,
            estimated_tax_usd_window=usd_window,
            other_sources=other_sources,
            primary_source_sessions=primary_source_sessions,
        )
        finding.servers.append(row)
        if sessions_present > 0:
            tax_rows.append(ContextTaxRow(
                source=f"MCP schema: {server.name}",
                sessions=sessions_present,
                avg_tokens_per_session=tokens_per_session_real,
                total_tokens_window=tokens_window_real,
                tag="estimated",
                construction=row.tax_construction,
            ))

    finding.servers.sort(key=lambda s: s.sessions_present, reverse=True)
    finding.dead_servers = sorted(
        (s for s in finding.servers if s.dead),
        key=lambda s: s.sessions_present, reverse=True,
    )

    # C2: fold the reminder-source buckets across sessions.
    for signal in per_session.values():
        for bucket, chars in signal.reminder_chars_by_source.items():
            reminder_bucket_totals.setdefault(bucket, []).append(chars)

    for bucket, char_counts in reminder_bucket_totals.items():
        sessions_with = len(char_counts)
        if sessions_with == 0:
            continue
        avg_tokens = round((sum(char_counts) / sessions_with) / CHARS_PER_TOKEN)
        tax_rows.append(ContextTaxRow(
            source=bucket,
            sessions=sessions_with,
            avg_tokens_per_session=avg_tokens,
            total_tokens_window=round(sum(char_counts) / CHARS_PER_TOKEN),
            tag="estimated",
            construction=(
                f"chars/{CHARS_PER_TOKEN} over the verbatim system-reminder "
                f"content Claude Code injects at session start, measured on "
                f"the first turn of each session."
            ),
        ))

    tax_rows.sort(key=lambda r: r.total_tokens_window, reverse=True)
    finding.tax_table = tax_rows

    # Dedup rule (spec, Component C): the recoverable total is ONLY the
    # dead-weight servers' own tax. The C2 tax table repeats a "MCP schema:
    # <name>" row for EVERY configured server (dead or alive) for visibility,
    # but that row never feeds this sum — so a server's tax is never counted
    # twice between the tax table and a dead-weight proposal.
    if finding.dead_servers:
        finding.past_overspend_tokens = sum(
            s.estimated_tax_tokens_window for s in finding.dead_servers
        )
        priced = [
            s.estimated_tax_usd_window for s in finding.dead_servers
            if s.estimated_tax_usd_window is not None
        ]
        basis = (
            f"sum of each dead server's schema-injection tax observed over "
            f"this window ({FULL_SCHEMA_TAX_TOKENS:,} tok/session full, "
            f"{DEFERRED_SCHEMA_TAX_TOKENS:,} tok/session when deferred); the "
            f"tax table's own MCP-schema rows are informational only and "
            f"never double-count into this total."
        )
        if priced:
            finding.past_overspend_usd = round(sum(priced), 6)
            basis += (
                " Dollar figure priced per server through core/pricing.py "
                "at the dominant model observed in that server's sessions "
                "(never a hardcoded rate)."
            )
            if len(priced) < len(finding.dead_servers):
                basis += (
                    f" {len(finding.dead_servers) - len(priced)} of "
                    f"{len(finding.dead_servers)} dead server(s) had no "
                    f"priced model observed and are excluded from the "
                    f"dollar sum (token figure still includes them)."
                )
        finding.estimate_basis = basis
    elif configured:
        finding.notes.append(
            f"No configured MCP server cleared the dead-weight bar "
            f"(>= {min_sessions} sessions present, 0 invocations). Lower "
            f"[optimize] min_sessions_deadweight in tj.toml to see servers "
            f"present in fewer sessions."
        )

    return finding


# --- Registry entry point ---------------------------------------------------

@register("deadweight")
def run(ctx: AnalyzerContext) -> None:
    """Registry entry point. Attaches a ``DeadweightFinding`` to
    ``ctx.report.findings["deadweight"]``. Claude Code transcripts lane only
    — reads on-disk JSONL directly, never ``ctx.conn`` (no DB spans needed).

    Passes the resolved persistent parse cache dir (``core.transcript_cache.
    default_cache_dir``) so a re-run over an unchanged corpus — including a
    repeat HTTP request against a live ``tj serve`` — skips re-parsing every
    session it already has a fresh cache entry for.
    """
    from tokenjam.core.optimize.scope import resolve_analyzer_scope
    from tokenjam.core.transcript_cache import default_cache_dir

    scope = ctx.scope if ctx.scope is not None else resolve_analyzer_scope(ctx.config)
    if not scope.enabled:
        # Scanned nothing, and says so on the report rather than leaving an
        # empty finding that reads like "no dead MCP servers here" (root
        # anti-pattern 22).
        ctx.report.filesystem_scan_skipped_reason = scope.reason
        ctx.report.findings["deadweight"] = DeadweightFinding()
        return

    optimize_cfg = getattr(ctx.config, "optimize", None)
    min_sessions = getattr(
        optimize_cfg, "min_sessions_deadweight", MIN_SESSIONS_DEADWEIGHT,
    )
    ctx.report.findings["deadweight"] = compute_deadweight_finding(
        ctx.since, ctx.until,
        projects_root=scope.projects_root,
        claude_home=scope.claude_home,
        min_sessions=min_sessions,
        cache_dir=default_cache_dir(ctx.config),
    )
