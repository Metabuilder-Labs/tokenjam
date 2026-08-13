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
one session, with nothing attributable to it firing anywhere in the trailing
``UNUSED_RECENCY_WINDOW_DAYS``-day recency window (and with enough corpus
history to trust that negative), is unused: its tool schemas are still
injected into context for no return. This is also the plugin lane's
liveness standard (Part B/C below) — one recency question, answered the same
way for every always-resident surface this module covers.

Both per-call sizes are MEASURED per server, not assumed:
``core/optimize/mcp_probe`` starts each configured server, asks it for its
tools, and counts the serialized schemas. A server that cannot be measured is
excluded from every priced figure rather than billed a default (see
``SchemaMeasurement`` and ``ServerDeadweight.schema_tokens_measured``).

Deferred-tools caveat (mandatory, spec hard rule). When a session's
transcript shows a deferred/ToolSearch-style listing naming a server's tools,
that server's full schemas were NOT loaded that turn — only a short
name+description line per tool appears in the listing, so the real per-turn
tax is much smaller. This module detects that marker per session and prices
those calls at ``SchemaMeasurement.deferred_tokens`` — the name+description
listing measured off the SAME ``tools/list`` response as the full schema, so
neither lane is an assumption — and never claims a deferred call cost the full
``SchemaMeasurement.tokens``.

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
from datetime import datetime, timedelta, timezone
from json.decoder import scanstring
from pathlib import Path
from typing import Any

from tokenjam.core.fixes import fix_text
from tokenjam.core.optimize.registry import register
from tokenjam.core.optimize.span_pricing import blended_rates, price_span, span_instant
from tokenjam.core.optimize.types import AnalyzerContext
from tokenjam.core.summarize.invocations import invoked_names_in_record
from tokenjam.core.transcript import _SYSTEM_REMINDER_RE, read_records, resolve_projects_root
from tokenjam.core.usage import AssistantUsage, assistant_message_key, parse_usage

# --- Tunables ------------------------------------------------------------

#: A configured MCP server or a resident plugin component (skill, agent, or
#: plugin-contributed MCP server) is UNUSED when nothing attributable to it
#: fired within this many trailing days. Founder direction: 20.
#:
#: Replaces two separate, disagreeing standards this analyzer used to run:
#: the MCP lane's ``MIN_SESSIONS_DEADWEIGHT`` (a session-COUNT floor, "present
#: in >= N sessions with 0 invocations") and the plugin lane's
#: ``usage_count == 0`` (a cumulative, non-timestamped counter out of
#: ``~/.claude.json``, which structurally cannot answer a recency question —
#: see ``core/agent_config.plugin_usage``'s own docstring on why it is kept
#: as informational context only, never as the liveness signal). A session
#: count is also opaque to a user: "not present in 5 sessions" says nothing
#: about how long ago that was. A trailing-day window answers the actual
#: question ("would removing this change anything going forward") the same
#: way for both lanes.
#:
#: This is DIFFERENT from the report's own ``since``..``until`` window
#: (``[optimize] scan_window_days``, default 30d): that knob decides how far
#: back the PRICED figures look, and a user can widen or narrow it. This
#: constant decides a separate question — how recent is "recent enough to
#: still count as used" — and stays fixed regardless of the report window, so
#: a `--since 7d` run cannot make everything look unused just because the
#: report itself only looked at a week. See ``_scan_recency_window`` for how
#: the two windows are reconciled: the recency check is its OWN independent
#: scan, anchored at the report's `until`, never narrowed or widened by
#: `since`.
UNUSED_RECENCY_WINDOW_DAYS = 20

#: How many example session ids a dead server's card carries as evidence
#: (mirrors relearn.py's MAX_EXAMPLE_SESSIONS convention).
MAX_EXAMPLE_SESSIONS = 3

# THE SCHEMA TAX IS MEASURED PER SERVER, AND THERE IS NO CONSTANT TO FALL BACK
# ON. This module used to carry `FULL_SCHEMA_TAX_TOKENS = 25_000` and charge it
# flat to EVERY individual server, every call, regardless of how many tools that
# server exposed. Its own docstring conceded the number was an assumption, and
# conceded worse: the only in-repo source for "~25K" (`core/context_diagnostic
# .py`'s MCP_INJECTION_PARK_NOTE) describes the tax for ALL of a session's
# attached servers COMBINED. So a per-server loop over an all-servers-combined
# figure made `past_overspend_usd` — the field every surface renders — linear in
# an unmeasured constant AND multiplied by however many servers the user had
# configured.
#
# `core/optimize/mcp_probe` measures each server instead: start it, ask for its
# tools, serialize those schemas as they are injected, count that. The
# deferred/ToolSearch lane comes off the same response (name + description per
# tool, no input schema), so both lanes are measured rather than one measured
# and one assumed.
#
# A server that could not be measured contributes NOTHING to any priced figure
# and says so on its own row and in the finding's coverage note. There is
# deliberately no default: a default is what made the old figure wrong, and an
# unmeasured server billed at a default is indistinguishable from a measured
# one. Both lanes are still FIRST-CALL counts — the schema rides in the `tools`
# array of every call in the session, and later calls are priced at the
# cache-read rate by the per-session multiplier in the tax loop below.

#: Chars-per-token conversion for text measured directly off transcripts
#: (system-reminder blocks) — same convention as prompt_bloat.py's
#: CHARS_PER_TOKEN.
CHARS_PER_TOKEN = 4

DEADWEIGHT_HONESTY_CAVEAT = (
    "Structural detection off configured MCP servers and their measured "
    "tool-call counts, not a judgment about whether the server is useful. "
    "Review the window before removing a server. A low-traffic server can "
    "still be load-bearing for an occasional task."
)


# --- MCP config enumeration (read-only; never writes a config file) -------
#
# WHICH config files declare a server, and where the global one lives, are
# stated once in `core/agent_config` (`PROJECT_MCP_RELPATHS` /
# `GLOBAL_MCP_RELPATH`) — this module used to hold a second copy of that list
# and its own lazy `~/.claude.json` resolver. Both moved with the walk itself
# when enumeration became an ingestion step; the lazy resolution rule did not
# change, and `core/agent_config._settings_paths` still resolves the global
# path against an explicit `claude_home` so a scoped run never reads the
# operator's real file.


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
    #: This server's own launch spec as declared in ``source`` (command/args/
    #: env, or a url + transport type). Carried because the schema measurement
    #: needs it and re-reading the config file to get it back would put the
    #: filesystem walk back in the middle of the analysis.
    spec: dict[str, Any] = field(default_factory=dict)
    #: Ingested-record id of the CANONICAL source, and of every source. The
    #: measurement is cached against these, so it survives between analysis
    #: runs — see ``core/agent_config`` and ``core/optimize/mcp_probe``.
    config_id: str = ""
    config_ids: dict[str, str] = field(default_factory=dict)


def enumerate_configured_servers(
    repo_cwds: set[str], *, claude_home: Path | None = None,
    store: "Any | None" = None, seen_at: datetime | None = None,
) -> dict[str, ConfiguredServer]:
    """MCP servers as INGESTED, not as re-read off disk at analysis time.

    The walk still happens — nothing else can discover a config file — but it is
    now the POPULATION step of ``core/agent_config``: it writes one record per
    (server name, declaring config file) into the agent-config store, and this
    function then builds its answer by reading those records back. That is what
    lets the schema measurement in ``core/optimize/mcp_probe`` be cached against
    a server's spec hash instead of re-taken (which means re-STARTING the
    server) on every analysis pass.

    ``store`` defaults to a fresh in-memory store, so a caller with no database
    gets exactly the enumeration the walk produced and no persistence. The
    registered ``run(ctx)`` passes a DuckDB-backed one.

    Behaviour is otherwise unchanged. A user-scoped (global) server still wins
    scope over a same-named project entry: the global entry already reaches
    every session, so downgrading it to "project" would only narrow its true
    presence. Roots are still visited in SORTED order and each project-scoped
    server's ``source`` is still chosen AFTER the full scan (most-cwds-covered
    first, path string as the tie-break) — never "whichever cwd a raw ``set``
    iteration happened to visit first", which used to hand the ONE apply target
    to a hash-seed-dependent winner.
    """
    from tokenjam.core import agent_config as ac

    store = store if store is not None else ac.InMemoryAgentConfigStore()
    at = ac.ingest_agent_config(
        store,
        roots=sorted(c for c in repo_cwds if c),
        claude_home=claude_home,
        kinds=(ac.KIND_MCP_SERVER,),
        seen_at=seen_at,
    )

    servers: dict[str, ConfiguredServer] = {}
    # (server name, declaring file) -> that file's own declaration of it. The
    # canonical `source` is only settled after the whole scan, so the spec and
    # the record id must both be resolvable per source until then — reading them
    # off "whichever record was seen last" would attach one file's launch
    # command to another file's chosen fix target.
    specs: dict[tuple[str, str], dict[str, Any]] = {}
    for record in store.select(kind=ac.KIND_MCP_SERVER, seen_at=at):
        spec = {k: v for k, v in record.detail.items() if k != "spec_hash"}
        specs[(record.name, record.path)] = spec
        if record.scope == ac.SCOPE_GLOBAL:
            servers[record.name] = ConfiguredServer(
                name=record.name, scope="user", source=record.path,
                spec=spec, config_id=record.config_id,
                config_ids={record.path: record.config_id},
            )
            continue
        existing = servers.get(record.name)
        if existing is not None and existing.scope == "user":
            continue  # already global, broaden nothing
        entry = servers.setdefault(
            record.name,
            ConfiguredServer(name=record.name, scope="project", source=record.path),
        )
        entry.cwds.add(record.root)
        entry.source_cwds.setdefault(record.path, set()).add(record.root)
        entry.config_ids[record.path] = record.config_id

    for entry in servers.values():
        if entry.scope == "project" and len(entry.source_cwds) > 1:
            # Most cwds covered wins; a path-string tie-break makes the choice
            # fully deterministic even when two sources cover the same count.
            entry.source = min(
                entry.source_cwds.items(), key=lambda kv: (-len(kv[1]), kv[0]),
            )[0]
        entry.config_id = entry.config_ids.get(entry.source, entry.config_id)
        entry.spec = specs.get((entry.name, entry.source), entry.spec)
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
        return None, f"that file is not valid JSON ({exc}). Refusing to edit it."
    servers = doc.get("mcpServers") if isinstance(doc, dict) else None
    if not isinstance(servers, dict) or server_name not in servers:
        return None, f"`{server_name}` is not in that file's mcpServers block any more."
    span = _mcp_server_entry_span(pre_image, server_name)
    if span is None:
        return None, (
            f"could not locate `{server_name}`'s entry precisely in the file "
            f"text. Refusing a risky edit."
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
        return {"ok": False, "reason": f"{path} no longer exists on disk. Nothing to edit."}
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"ok": False, "reason": f"{path} is not valid JSON ({exc}). Refusing to edit it."}
    if not server_still_configured(server_name, str(path)):
        return {"ok": False, "reason": (
            f"`{server_name}` is no longer in {path}'s mcpServers block. It may "
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


# --- Recency-window liveness (Part A: replaces session-count liveness) ----
#
# ONE independent scan answers "did anything attributable to this name fire
# in the last UNUSED_RECENCY_WINDOW_DAYS days" for BOTH lanes (MCP servers and
# plugin components alike), plus whether the corpus even HOLDS that much
# history to trust a negative answer. Deliberately its own pass rather than
# reusing `per_session` / `session_paths` from the report's own since..until
# scan: the report window is a user-facing knob (`--since`, default 30d) and
# can be narrower than the recency window, which would make "nothing fired in
# the sessions we happened to load" a false positive rather than a real
# absence. Anchored at `until` (the report's own end, "now" for a live run),
# never at `since`. The transcript parse cache (`cache_dir`) makes the
# resulting overlap with the report's own scan cheap rather than a second
# full read.


@dataclass(frozen=True)
class RecencyScan:
    """What fired, and whether the corpus is deep enough to trust a "no"."""

    #: Names (MCP server names, skill/agent slugs — whatever the caller feeds
    #: `_bump`-style bare-suffix matching) seen attributable to at least one
    #: call/invocation inside the trailing recency window.
    used_names: frozenset[str] = frozenset()
    #: True when the corpus holds a session at or before the window's own
    #: start (`until - UNUSED_RECENCY_WINDOW_DAYS`) — i.e. enough history was
    #: actually observed to make "nothing fired" a real negative rather than
    #: "we only started watching five minutes ago". False means every
    #: consumer must render "not enough history yet", never a finding.
    corpus_sufficient: bool = False


def _recency_matches(used_names: frozenset[str], name: str) -> bool:
    """Whether `name` (a slug/server name a caller is asking about) was seen
    in the recency scan — bare name OR, for a plugin-namespaced invocation
    (`plugin:slug`), its bare suffix, mirroring
    `core.summarize.invocations._bump`'s aliasing so a plugin's own skill
    directory name (never namespaced on disk) still matches an invocation
    Claude Code recorded with the plugin prefix."""
    if not name:
        return False
    if name in used_names:
        return True
    return any(u.rsplit(":", 1)[-1] == name for u in used_names if ":" in u)


def _scan_recency_window(
    root: Path, until: datetime, *, cache_dir: Path | None,
) -> RecencyScan:
    """Independent pass over the transcript tree answering the recency
    question for every name any caller might ask about at once — cheaper than
    one scan per server/plugin, and the corpus-depth answer is necessarily
    global anyway (see `RecencyScan.corpus_sufficient`).

    Never raises: mirrors this module's "unreadable transcript is skipped,
    not fatal" contract.
    """
    if not root.exists():
        return RecencyScan()
    recency_since = until - timedelta(days=UNUSED_RECENCY_WINDOW_DAYS)
    used: set[str] = set()
    earliest_mtime: datetime | None = None
    for path in sorted(root.rglob("*.jsonl")):
        if path.parent.name == "subagents":
            continue
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            continue
        if mtime >= until:
            continue
        if earliest_mtime is None or mtime < earliest_mtime:
            earliest_mtime = mtime
        if mtime < recency_since:
            continue
        try:
            records = read_records(path, cache_dir=cache_dir)
        except Exception:
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            message = record.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            blocks = content if isinstance(content, list) else []
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    server = _mcp_server_from_tool_name(str(block.get("name") or ""))
                    if server:
                        used.add(server)
            for name in invoked_names_in_record(record):
                if name:
                    used.add(name)
    corpus_sufficient = earliest_mtime is not None and earliest_mtime <= recency_since
    return RecencyScan(used_names=frozenset(used), corpus_sufficient=corpus_sufficient)


def _measurement_note(measurement: Any) -> str:
    """How this server's per-call size was arrived at, in the card's own words.

    Always states the PROVENANCE, never only the number: the whole defect this
    replaced was a figure that read as measured because it was rendered the same
    way a measured one would be.
    """
    if measurement is None:
        return "This server's schema size was not measured, so nothing is priced for it."
    if not getattr(measurement, "ok", False):
        detail = str(getattr(measurement, "detail", "") or "").strip()
        return (
            "This server's schema size could NOT be measured"
            + (f" ({detail})" if detail else "")
            + ". It is excluded from every priced figure here rather than "
            "billed a default."
        )
    tools = int(getattr(measurement, "tool_count", 0) or 0)
    cached = (
        " Re-used from the last measurement. The server's launch spec is "
        "unchanged since."
    ) if getattr(measurement, "from_cache", False) else ""
    return (
        f"Measured from this server's OWN tools/list response: {tools} tool "
        f"schema(s), serialized as they are injected.{cached}"
    )


def _tax_construction_note(
    non_deferred: int, deferred_sessions: int, sessions_present: int,
    *, model: str = "", input_per_mtok: float | None = None,
    usd_per_session: float | None = None,
    avg_calls_per_session: float = 1.0,
    cache_read_ratio: float = 0.0,
    full_tokens: int | None = None,
    deferred_tokens: int | None = None,
    measurement: Any = None,
) -> str:
    if sessions_present == 0:
        return ""
    if full_tokens is None:
        return _measurement_note(measurement)
    deferred_shown = deferred_tokens if deferred_tokens is not None else 0
    if deferred_sessions == 0:
        note = (
            f"{full_tokens:,} tok on the first call (full schema injection). "
            f"{_measurement_note(measurement)}"
        )
    elif non_deferred == 0:
        note = (
            f"{deferred_shown:,} tok on the first call. ToolSearch deferred "
            f"this server's schemas in every observed session (name and "
            f"description line only, never the full schema tax). "
            f"{_measurement_note(measurement)}"
        )
    else:
        note = (
            f"{full_tokens:,} tok on the first call when fully loaded "
            f"({non_deferred} of {sessions_present} sessions). This blends "
            f"with {deferred_shown:,} tok when ToolSearch defers this "
            f"server's schemas ({deferred_sessions} of {sessions_present} "
            f"sessions). Never claims the full tax for a deferred call. "
            f"{_measurement_note(measurement)}"
        )
    if cache_read_ratio > 0:
        note += (
            f" The schema rides in the `tools` array of every call in the "
            f"session, not just the first. It is priced at the input rate "
            f"once, then at the cache-read rate (~{cache_read_ratio * 100:.0f}% "
            f"of input) on every later call. The call count used is avg "
            f"{avg_calls_per_session:.1f} call(s)/session across these "
            f"sessions, from each session's own actual call count, never a "
            f"global mean/median. Simplification: this assumes every call in "
            f"a session lands inside Anthropic's 5-minute cache TTL. A call "
            f"after a longer gap would instead re-write the schema at the "
            f"higher cache-write rate for that call. This understates "
            f"sessions with long gaps between calls."
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
            "No dollar estimate: no priced model was observed across these "
            "sessions (core/pricing.py has no rate for it). Tokens only."
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
    #: True only when this server was PRESENT in at least one session AND
    #: nothing attributable to it fired anywhere in the trailing
    #: `UNUSED_RECENCY_WINDOW_DAYS`-day recency window AND the corpus holds
    #: enough history to trust that negative (`not insufficient_history`).
    #: Replaces the old session-count gate (`MIN_SESSIONS_DEADWEIGHT`) — see
    #: `_scan_recency_window`.
    unused:                           bool
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
    #: True when the corpus does not reach back far enough to answer the
    #: recency question at all — never both this AND `unused`. A caller must
    #: render "not enough history yet", never a finding, when this is set.
    insufficient_history:             bool = False
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
    #: Tokens THIS server's own tool schemas were measured to inject on a call
    #: (`core/optimize/mcp_probe`). ``None`` means the measurement did not
    #: happen — never a default, and never a zero, because a zero here would
    #: read as "this server costs nothing".
    schema_tokens_measured:          int | None = None
    #: The same measurement for the deferred/ToolSearch listing (name and
    #: description per tool, no input schema), off the same response.
    deferred_tokens_measured:        int | None = None
    #: How many tool schemas that measurement covered.
    measured_tool_count:             int = 0
    #: ``core/agent_config``'s MEASURE_* vocabulary: which of "measured",
    #: "unreachable", "unsupported" or "skipped" happened.
    measurement_status:              str = ""
    #: Why, in words, when it did not succeed.
    measurement_detail:              str = ""


@dataclass
class PluginComponent:
    """One always-resident piece a plugin contributes: a skill, an agent, or
    a plugin-contributed MCP server. The unit Part C's rollup reasons over —
    enable/disable is whole-plugin only (`claude plugin disable <name>`;
    `enabledPlugins` is a binary map, verified against the shipping CLI), so
    there is no per-component fix, only a per-component USED signal that
    decides whether the PLUGIN'S fix may be offered at all.
    """
    kind:                str    # "skill" | "agent" | "mcp_server"
    name:                str    # slug (skill/agent) or server name
    #: This component's own resident tokens. `None` for an `mcp_server` whose
    #: schema could not be measured — never a zero, which would read as "this
    #: component costs nothing" (identical convention to
    #: `ServerDeadweight.schema_tokens_measured`).
    resident_tokens:     int | None
    #: `None` = insufficient corpus history to say either way. `True` = fired
    #: within the recency window. `False` = did not.
    used:                bool | None
    #: Why `resident_tokens` is `None`, when it is. Empty otherwise.
    measurement_note:    str = ""


@dataclass
class PluginDeadweight:
    """One INSTALLED Claude Code plugin, and whether it costs anything.

    The MCP lane's shape applied to a different dependency: something installed
    once, paid for continuously, and possibly never used, whose fix is a
    one-line reversible edit.

    Every installed plugin gets a row, including the ones that cost nothing —
    "disabled, so free" and "we never looked" must not be the same absence, and
    a user deciding whether to ENABLE something large is exactly who needs the
    disabled rows.

    Three-state gating (root anti-pattern 22 in the meta-repo `CLAUDE.md`):
    a component's `used` can be unknown (`insufficient_history`), known-unused
    (`unused`, only when EVERY component agrees), or known-used-somewhere
    (`partial_use_no_fix` when some but not all components fired — the
    all-or-nothing toggle means there is genuinely no fix to offer there, so
    the row must say so plainly rather than dangle a number with no action
    behind it).
    """
    name:                     str
    enabled:                  bool
    install_scope:            str
    #: Both gates passed: enabled AND installed at a scope that loads globally.
    resident:                 bool
    #: Which gate it failed, in words. Empty when resident.
    not_resident_because:     str
    #: Every skill, agent and plugin-contributed MCP server this plugin owns.
    #: Empty for a non-resident plugin (nothing is resident to enumerate) —
    #: see `unused`'s vacuous-truth guard for why an empty list must never
    #: read as "every component agrees".
    components:               list[PluginComponent] = field(default_factory=list)
    skills:                   int = 0
    agents:                   int = 0
    #: Sum of every MEASURED component's own `resident_tokens` — the `name:
    #: description` listing surface (skills, agents) plus any
    #: plugin-contributed MCP server's measured schema size, and NOTHING
    #: else. A skill/agent BODY arrives on invocation, not at session start,
    #: so counting bodies would not be a conservative overestimate; it would
    #: be a different number about a different thing. Components that could
    #: not be measured (`PluginComponent.resident_tokens is None`) are
    #: excluded, so this is a FLOOR when `components_unmeasured` is nonzero.
    resident_tokens:          int = 0
    #: How many of `components` are an `mcp_server` whose schema could not be
    #: measured. The blind spot `resident_tokens` doesn't disclose on its own.
    components_unmeasured:    int = 0
    #: `pluginUsage` count out of `~/.claude.json`. `None` means Claude Code
    #: never recorded this plugin at all, which is not the same statement as
    #: "recorded and never used". INFORMATIONAL ONLY: this is a cumulative,
    #: non-timestamped counter (see `core/agent_config.plugin_usage`), so it
    #: never decides `unused` — only the recency scan does.
    usage_count:              int | None = None
    sessions_present:         int = 0
    #: True only when `components` is non-empty, at least one component was
    #: actually priced (`resident_tokens > 0`), and EVERY component's `used`
    #: resolved to `False` (never vacuously true over an empty or
    #: all-unknown set — see the module's recency-scan docs).
    unused:                   bool = False
    #: True when at least one component's `used` is `None` (insufficient
    #: corpus history) and no component is positively known used — renders as
    #: "not enough history yet", never a number, never a fix arrow.
    insufficient_history:     bool = False
    #: True when components disagree (some used, some not) — the toggle is
    #: whole-plugin, so there is genuinely no fix to offer; the row must name
    #: the unused components and say plainly that no action is available.
    partial_use_no_fix:       bool = False
    estimated_tax_tokens_window: int = 0
    estimated_tax_usd_window:    float | None = None
    priced_model:             str = ""
    tax_construction:         str = ""
    fix:                      str = ""


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
    #: Configured MCP servers nothing invoked within the recency window (see
    #: `UNUSED_RECENCY_WINDOW_DAYS`). Named `unused_servers`, not `dead_servers`
    #: — "dead" implied a judgment about the server; this is a structural
    #: recency observation about the WINDOW.
    unused_servers:               list[ServerDeadweight] = field(default_factory=list)
    #: Every INSTALLED plugin, resident or not (see `PluginDeadweight`).
    plugins:                      list[PluginDeadweight] = field(default_factory=list)
    #: Resident plugins whose every component is unused — the ones with a fix.
    #: A plugin with SOME unused components but at least one used one is
    #: never here (see `PluginDeadweight.partial_use_no_fix`): the toggle is
    #: whole-plugin, so partial disuse has no fix to offer.
    unused_plugins:               list[PluginDeadweight] = field(default_factory=list)
    #: How many installed plugins actually reach the model. The gap between
    #: this and `len(plugins)` is the whole point of the two gates: on a real
    #: machine most of what is installed is not loaded, and a figure built from
    #: installed-ness overstates by orders of magnitude.
    plugins_resident:             int = 0
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
    #: How many configured servers had their schema size actually measured, and
    #: how many did not. The second number is the size of this finding's blind
    #: spot: those servers are excluded from every priced figure, so a small
    #: total with a large `servers_unmeasured` is "we could not look", never
    #: "there is little here".
    servers_measured:                 int = 0
    servers_unmeasured:               int = 0
    #: Plain-language statement of that blind spot, empty when nothing was
    #: unmeasured. Rendered beside the figure, never instead of it.
    measurement_note:                 str = ""


# --- Unresolvable-path coverage (Defect 1: silence when a cwd is gone) ----

def _session_usage_from_records(records: list[dict[str, Any]]) -> AssistantUsage:
    """Total ACTUAL assistant usage over one already-parsed session,
    last-wins deduped by message key — same dedup policy as
    ``core.usage.session_usage``, just replayed over records this module
    already parsed once rather than re-reading the raw lines a second time.

    This is real usage (what the session actually cost), never to be confused
    with the per-server measurements in ``SchemaMeasurement`` (surfaced on the
    finding as ``ServerDeadweight.schema_tokens_measured`` /
    ``deferred_tokens_measured``), which size the MCP schema-injection TAX
    rather than actual spend.
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
            f"those sessions. No priced model was observed across them, so "
            f"no dollar figure is stated."
        )
    parts.append(
        "A user-scoped server (global ~/.claude.json) is unaffected by "
        "this: it resolves regardless of a session's cwd. Only "
        "project-scoped detection is blind here."
    )
    return " ".join(parts)


def _plugin_rows(
    store: Any,
    *,
    claude_dir: Path | None,
    claude_home: Path | None,
    seen_at: datetime | None,
    session_calls: list[int],
    input_per_mtok: float | None,
    cache_read_ratio: float,
    priced_model: str,
    recency: RecencyScan,
    measure_schemas: bool,
    schema_measurer: Any | None = None,
) -> list[PluginDeadweight]:
    """The plugin lane: what an ENABLED, IN-SCOPE plugin costs every session,
    rolled up to ONE verdict per plugin because enable/disable is whole-plugin
    only (verified against the shipping CLI: `enabledPlugins` is a binary map,
    `claude plugin disable <name>` the real command — there is no way to
    disable one skill inside a plugin).

    Two RESIDENCY gates, and neither is visible on the filesystem — a plugin
    being installed on disk says nothing about whether it reaches the model.
    ``core/agent_config.scan_plugins`` applies both and also enumerates each
    resident plugin's skills, agents, and its own contributed MCP servers
    (``detail["components"]`` plus sibling ``KIND_MCP_SERVER`` records tagged
    ``detail["plugin"]``); this function prices what survives, against the
    same per-session call multiplier the MCP lane uses (the listing rides in
    the system prompt of every call, so later calls are cache-read rather
    than re-charged at the input rate), and a SEPARATE liveness gate per
    component: whether anything attributable to it fired within
    ``recency`` (see ``_scan_recency_window``) — never the old
    session-count/usage-count gates.

    A resident plugin is present in EVERY session, which is what makes it a
    standing cost rather than a per-project one — and what makes the decision to
    enable one worth seeing priced before it is made.
    """
    from tokenjam.core import agent_config as ac
    from tokenjam.core.optimize.mcp_probe import resolve_schema_measurements

    at = ac.ingest_agent_config(
        store, kinds=(ac.KIND_PLUGIN,), claude_dir=claude_dir, seen_at=seen_at,
    )
    usage = ac.plugin_usage(claude_home)
    sessions_present = len(session_calls)

    # Plugin-contributed MCP servers landed in the SAME ingest pass as
    # KIND_MCP_SERVER records tagged with their owning plugin (`scan_plugins`
    # emits both kinds together). Grouped by plugin so each plugin's
    # `resolve_schema_measurements` call sees only ITS OWN servers, keyed by
    # their real bare name — the name that rides in `mcp__<name>__<tool>` and
    # must stay correct for both the schema measurement and the recency
    # attribution below, never a composite key that would corrupt it.
    servers_by_plugin: dict[str, dict[str, ConfiguredServer]] = {}
    for record in store.select(kind=ac.KIND_MCP_SERVER, seen_at=at):
        plugin = record.detail.get("plugin")
        if not plugin:
            continue
        spec = {k: v for k, v in record.detail.items() if k not in ("spec_hash", "plugin")}
        servers_by_plugin.setdefault(str(plugin), {})[record.name] = ConfiguredServer(
            name=record.name, scope="plugin", source=record.path,
            spec=spec, config_id=record.config_id,
        )

    rows: list[PluginDeadweight] = []
    for record in store.select(kind=ac.KIND_PLUGIN, seen_at=at):
        detail = record.detail
        resident = bool(detail.get("resident"))
        count = usage.get(record.name)

        components: list[PluginComponent] = []
        if resident:
            for c in detail.get("components") or []:
                slug = str(c.get("slug") or "")
                kind = str(c.get("kind") or "")
                tokens = ac.tokens_for_chars(int(c.get("chars") or 0))
                used = (
                    _recency_matches(recency.used_names, slug)
                    if recency.corpus_sufficient else None
                )
                components.append(PluginComponent(
                    kind=kind, name=slug, resident_tokens=tokens, used=used,
                ))
            plugin_servers = servers_by_plugin.get(record.name, {})
            measurements = (
                resolve_schema_measurements(
                    plugin_servers, store=store,
                    enabled=measure_schemas, measurer=schema_measurer,
                ) if plugin_servers else {}
            )
            for server_name, server in sorted(plugin_servers.items()):
                measurement = measurements.get(server_name)
                mtokens = measurement.tokens if measurement is not None else None
                used = (
                    _recency_matches(recency.used_names, server_name)
                    if recency.corpus_sufficient else None
                )
                components.append(PluginComponent(
                    kind="mcp_server", name=server_name, resident_tokens=mtokens,
                    used=used,
                    measurement_note="" if mtokens is not None else _measurement_note(measurement),
                ))

        resident_tokens = sum(c.resident_tokens or 0 for c in components)
        components_unmeasured = sum(
            1 for c in components if c.kind == "mcp_server" and c.resident_tokens is None
        )

        tax_window = 0
        if resident and resident_tokens:
            for calls in session_calls:
                tax_window += round(
                    resident_tokens * (1.0 + (max(calls, 1) - 1) * cache_read_ratio)
                )
        usd_window = (
            round(tax_window / 1_000_000 * input_per_mtok, 6)
            if input_per_mtok is not None and tax_window else None
        )

        # Three-state rollup. `recency.corpus_sufficient is False` means every
        # component's `used` above is `None` (insufficient history) — never
        # mix that with a positive "unused" claim. `all()`/`any()` over an
        # EMPTY `components` list would otherwise read as vacuously unused
        # (Critical Rule 42 in `.claude/rules/optimize-analyzers.md`);
        # `bool(components) and resident_tokens > 0` is the guard against
        # exactly that (a plugin with zero skills/agents/servers, or only
        # unmeasured MCP servers, has nothing priced and gets no verdict at
        # all).
        insufficient_history = resident and bool(components) and not recency.corpus_sufficient
        unused = False
        partial_use_no_fix = False
        if resident and components and recency.corpus_sufficient:
            all_used = all(c.used for c in components)
            none_used = not any(c.used for c in components)
            unused = none_used and resident_tokens > 0
            partial_use_no_fix = not all_used and not none_used

        skills = sum(1 for c in components if c.kind == "skill")
        agents = sum(1 for c in components if c.kind == "agent")
        mcp_count = sum(1 for c in components if c.kind == "mcp_server")

        parts: list[str] = []
        if resident:
            if skills or agents:
                parts.append(
                    f"{sum(c.resident_tokens or 0 for c in components if c.kind != 'mcp_server'):,} "
                    f"tok resident per call: the `name: description` line of "
                    f"each of this plugin's {skills} skill(s) and {agents} "
                    f"agent(s), measured off the installed files. BODIES are "
                    f"NOT counted. A body arrives when the skill/agent is "
                    f"invoked, not at session start."
                )
            if mcp_count:
                measured_mcp = mcp_count - components_unmeasured
                note = (
                    f"Plus {mcp_count} plugin-contributed MCP server(s), "
                    f"{measured_mcp} measured via core/optimize/mcp_probe "
                    f"(same live-probe path a user-declared server uses)."
                )
                if components_unmeasured:
                    note += (
                        f" {components_unmeasured} could not be measured and "
                        f"are excluded from the total."
                    )
                parts.append(note)
            if not parts:
                parts.append(
                    "No skill, agent or MCP server was found for this "
                    "plugin. Nothing to price."
                )
        else:
            parts.append(
                f"Not resident: {detail.get('not_resident_because') or 'gated off'}. "
                f"Nothing is priced for it."
            )
        construction = " ".join(parts)

        fix = ""
        if unused:
            names = ", ".join(f"{c.kind} `{c.name}`" for c in components)
            fix = (
                f"`{record.name}` unused: nothing fired for {names} in "
                f"{UNUSED_RECENCY_WINDOW_DAYS} days. "
                + fix_text("deadweight.disable_unused_plugin")
            )
        elif partial_use_no_fix:
            unused_names = ", ".join(
                f"{c.kind} `{c.name}`" for c in components if c.used is False
            )
            fix = (
                f"`{record.name}`: {unused_names} unused, another component "
                f"used. " + fix_text("deadweight.plugin_partial_use_no_fix")
            )

        rows.append(PluginDeadweight(
            name=record.name,
            enabled=bool(detail.get("enabled")),
            install_scope=str(detail.get("install_scope") or ""),
            resident=resident,
            not_resident_because=str(detail.get("not_resident_because") or ""),
            components=components,
            skills=skills,
            agents=agents,
            resident_tokens=resident_tokens,
            components_unmeasured=components_unmeasured,
            usage_count=count,
            sessions_present=sessions_present if resident else 0,
            unused=unused,
            insufficient_history=insufficient_history,
            partial_use_no_fix=partial_use_no_fix,
            estimated_tax_tokens_window=tax_window,
            estimated_tax_usd_window=usd_window,
            priced_model=priced_model if usd_window is not None else "",
            tax_construction=construction,
            fix=fix,
        ))
    rows.sort(key=lambda p: (not p.resident, -p.estimated_tax_tokens_window, p.name))
    return rows


def _measurement_coverage_note(finding: DeadweightFinding) -> str:
    """State, in words, how much of the configured surface was actually measured.

    The counterpart to ``_unresolvable_coverage_note`` for the other blind spot
    this analyzer has. A figure built from three of eleven servers is a FLOOR,
    and a reader who is shown only the figure has no way to know that — the same
    class of defect as the flat constant this measurement replaced, just one
    layer up.
    """
    if finding.servers_unmeasured <= 0:
        return ""
    total = finding.servers_measured + finding.servers_unmeasured
    return (
        f"MEASUREMENT COVERAGE. {finding.servers_measured:,} of {total:,} "
        f"configured MCP server(s) had their schema size measured from the "
        f"server's own tools/list response. The other "
        f"{finding.servers_unmeasured:,} could not be measured and contribute "
        f"NOTHING to any figure here. They are excluded rather than billed a "
        f"default, so every total on this finding is a floor. Each server's "
        f"own row says which of them it is and why."
    )


# --- Orchestration (pure, no ctx dependency — testable directly) ----------

def compute_deadweight_finding(
    since: datetime,
    until: datetime,
    *,
    projects_root: Path | str | None = None,
    claude_home: Path | None = None,
    claude_dir: Path | None = None,
    cache_dir: Path | None = None,
    store: Any | None = None,
    measure_schemas: bool = True,
    schema_measurer: Any | None = None,
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

    Liveness (which servers/plugin components are UNUSED) is answered by an
    independent recency scan anchored at ``until`` — see
    ``UNUSED_RECENCY_WINDOW_DAYS`` / ``_scan_recency_window`` — never by a
    session-count threshold against ``since``..``until``. That window is a
    fixed product constant, not a parameter: `since`/`until` still bound
    everything this function PRICES (the tax figures stay window-scoped,
    past-tense observations), they just no longer decide what counts as used.

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

    from tokenjam.core import agent_config as ac
    from tokenjam.core.optimize.mcp_probe import resolve_schema_measurements

    config_store = store if store is not None else ac.InMemoryAgentConfigStore()
    # ONE recency scan answers "unused?" for every MCP server AND every
    # plugin component below — see the module docstring above
    # `_scan_recency_window` for why this is its own independent pass rather
    # than a re-read of `per_session`/`since`.
    recency = _scan_recency_window(root, until, cache_dir=cache_dir)
    # The plugin lane. Runs BEFORE the `if not configured: return` below and is
    # independent of it: a user with no MCP servers at all can still be paying
    # for an enabled plugin every session, and hanging this off the MCP early
    # return would make it silent in exactly that case.
    window_models: dict[str, int] = {}
    window_volume_at: list[tuple[datetime | None, float]] = []
    for session_id, signal in per_session.items():
        for model, count in signal.models.items():
            window_models[model] = window_models.get(model, 0) + count
            window_volume_at.append((session_mtimes.get(session_id), float(count)))
    window_model = _dominant_model(window_models)
    window_rate: float | None = None
    window_cache_ratio = 0.0
    if window_model:
        from tokenjam.core.pricing import provider_for_model as _provider

        rates = blended_rates(
            _provider(window_model) or "unknown", window_model, window_volume_at,
        )
        if rates is not None and rates.input_per_mtok > 0:
            window_rate = rates.input_per_mtok
            window_cache_ratio = rates.cache_read_per_mtok / rates.input_per_mtok
        else:
            window_model = ""
    # A caller that scoped the MCP read away from the operator's real machine
    # meant to scope the PLUGIN read too. Deriving the plugin directory from
    # `claude_home` when it was not given explicitly is what stops an isolated
    # run from silently reading the operator's real `~/.claude/plugins` — the
    # same escape `_settings_paths` documents one level up.
    plugin_dir = claude_dir
    if plugin_dir is None:
        plugin_dir = (claude_home / ".claude") if claude_home is not None else None
    finding.plugins = _plugin_rows(
        config_store,
        claude_dir=plugin_dir,
        claude_home=claude_home,
        seen_at=None,
        session_calls=[max(s.assistant_turns, 1) for s in per_session.values()],
        input_per_mtok=window_rate,
        cache_read_ratio=window_cache_ratio,
        priced_model=window_model,
        recency=recency,
        measure_schemas=measure_schemas,
        schema_measurer=schema_measurer,
    )
    finding.plugins_resident = sum(1 for p in finding.plugins if p.resident)
    finding.unused_plugins = [p for p in finding.plugins if p.unused]

    configured = enumerate_configured_servers(
        repo_cwds, claude_home=claude_home, store=config_store,
    )
    finding.configured_servers = len(configured)
    if not configured:
        _finish_totals(finding, configured=configured)
        return finding

    # The per-server schema measurement (see `core/optimize/mcp_probe`). Cached
    # in the same store the enumeration just populated, keyed on the server's
    # spec hash, so an unchanged server is measured ONCE rather than started
    # again on every analysis pass.
    measurements = resolve_schema_measurements(
        configured, store=config_store,
        enabled=measure_schemas, measurer=schema_measurer,
    )
    finding.servers_measured = sum(
        1 for m in measurements.values() if m.tokens is not None
    )
    finding.servers_unmeasured = len(measurements) - finding.servers_measured

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

        # Liveness is the recency scan's answer, not a session-count floor
        # (Critical Rule / Part A: retires MIN_SESSIONS_DEADWEIGHT). A server
        # must have been PRESENT at least once this window (there is a real
        # standing tax to report) AND unseen anywhere in the trailing
        # `UNUSED_RECENCY_WINDOW_DAYS`-day window AND the corpus must hold
        # enough history to trust that negative — never vacuously true for a
        # server present zero times, which is what keeps a $0 row from ever
        # carrying a disable arrow (Part C).
        insufficient_history = sessions_present > 0 and not recency.corpus_sufficient
        unused = (
            sessions_present > 0
            and recency.corpus_sufficient
            and not _recency_matches(recency.used_names, server.name)
        )
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
        #
        # BOTH per-call bases come from this server's own measurement. When it
        # has none, every quantity below stays at zero and the row is excluded
        # from the finding's priced totals — never billed a stand-in, which is
        # the exact defect the flat 25K constant was.
        measurement = measurements.get(server.name)
        measured_full = measurement.tokens if measurement is not None else None
        measured_deferred = (
            measurement.deferred_tokens if measurement is not None else None
        )
        if measured_full is not None and measured_deferred is None:
            # A cached measurement predating the deferred lane. The deferred
            # size is genuinely unknown, so deferred calls are charged nothing
            # rather than charged the full schema they demonstrably did not
            # carry — understating a deferred session beats overstating it.
            measured_deferred = 0

        tax_window = 0
        tokens_window_real = 0
        total_calls = 0
        full_base = measured_full or 0
        deferred_base = measured_deferred or 0
        for deferred_calls, full_calls in session_presence:
            if measured_full is not None:
                if deferred_calls > 0:
                    tax_window += round(
                        deferred_base * (1.0 + (deferred_calls - 1) * cache_read_ratio)
                    )
                    tokens_window_real += deferred_base * deferred_calls
                if full_calls > 0:
                    tax_window += round(
                        full_base * (1.0 + (full_calls - 1) * cache_read_ratio)
                    )
                    tokens_window_real += full_base * full_calls
            total_calls += deferred_calls + full_calls
        tax_per_session = round(tax_window / sessions_present) if sessions_present else 0
        tokens_per_session_real = (
            round(tokens_window_real / sessions_present) if sessions_present else 0
        )
        avg_calls_per_session = (total_calls / sessions_present) if sessions_present else 1.0

        usd_per_session: float | None = None
        usd_window: float | None = None
        if input_per_mtok is not None and measured_full is not None:
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
        # GROUNDING here, POLICY in the catalog. The rule about a server
        # declared in several files — that editing one stops the tax only for
        # the sessions reaching it through that file — used to be APPENDED to
        # this string with `+=`, which the prose guard did not inspect at all.
        # So a durable instruction lived here, unlinted, hidden behind a
        # grounded sentence. What is left below is only this row's evidence.
        fix = (
            f"{action} the `{server.name}` MCP server ({server.source}). "
            f"Zero tool calls across {sessions_present} session(s) in this "
            f"window. " + fix_text("deadweight.remove_unused_server")
        )
        if other_sources:
            plural = "" if len(other_sources) == 1 else "s"
            fix += (
                f" Also declared in {len(other_sources)} other "
                f"location{plural}: {'; '.join(other_sources)}. Editing "
                f"{server.source} covers {primary_source_sessions} of the "
                f"{sessions_present} session(s) counted here."
            )

        row = ServerDeadweight(
            name=server.name,
            scope=server.scope,
            source=server.source,
            sessions_present=sessions_present,
            invocations=invocations,
            deferred_sessions=deferred_sessions,
            unused=unused,
            estimated_tax_tokens_per_session=tokens_per_session_real,
            estimated_tax_tokens_window=tokens_window_real,
            tax_construction=_tax_construction_note(
                non_deferred, deferred_sessions, sessions_present,
                model=priced_model, input_per_mtok=input_per_mtok,
                usd_per_session=usd_per_session,
                avg_calls_per_session=avg_calls_per_session,
                cache_read_ratio=cache_read_ratio,
                full_tokens=measured_full,
                deferred_tokens=measured_deferred,
                measurement=measurement,
            ),
            fix=fix,
            insufficient_history=insufficient_history,
            example_sessions=example_sessions,
            priced_model=priced_model,
            estimated_tax_usd_per_session=usd_per_session,
            estimated_tax_usd_window=usd_window,
            other_sources=other_sources,
            primary_source_sessions=primary_source_sessions,
            schema_tokens_measured=measured_full,
            deferred_tokens_measured=measured_deferred if measured_full is not None else None,
            measured_tool_count=(
                int(getattr(measurement, "tool_count", 0) or 0)
                if measurement is not None else 0
            ),
            measurement_status=(
                str(getattr(measurement, "status", "")) if measurement is not None else ""
            ),
            measurement_detail=(
                str(getattr(measurement, "detail", "")) if measurement is not None else ""
            ),
        )
        finding.servers.append(row)
        if sessions_present > 0 and measured_full is not None:
            tax_rows.append(ContextTaxRow(
                source=f"MCP schema: {server.name}",
                sessions=sessions_present,
                avg_tokens_per_session=tokens_per_session_real,
                total_tokens_window=tokens_window_real,
                tag="estimated",
                construction=row.tax_construction,
            ))

    finding.servers.sort(key=lambda s: s.sessions_present, reverse=True)
    finding.unused_servers = sorted(
        (s for s in finding.servers if s.unused),
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
    # unused servers' own tax. The C2 tax table repeats a "MCP schema:
    # <name>" row for EVERY configured server (unused or alive) for
    # visibility, but that row never feeds this sum — so a server's tax is
    # never counted twice between the tax table and an unused-server proposal.
    _finish_totals(finding, configured=configured)
    return finding


def _finish_totals(finding: DeadweightFinding, *, configured: dict) -> DeadweightFinding:
    """Roll the recoverable totals up and state what is missing from them.

    Split out because it has to run on BOTH exits from the pipeline: a window
    with no configured MCP server can still carry an unused plugin, and leaving
    this behind the MCP early return would make the plugin lane silent in
    exactly the case where it is the only thing there is to say.
    """
    measured_unused = [
        s for s in finding.unused_servers if s.schema_tokens_measured is not None
    ]
    unmeasured_unused = len(finding.unused_servers) - len(measured_unused)
    if measured_unused:
        finding.past_overspend_tokens = sum(
            s.estimated_tax_tokens_window for s in measured_unused
        )
        priced = [
            s.estimated_tax_usd_window for s in measured_unused
            if s.estimated_tax_usd_window is not None
        ]
        sizes = sorted(
            s.schema_tokens_measured for s in measured_unused
            if s.schema_tokens_measured is not None
        )
        # The RANGE, not one number: each server is measured separately now, so
        # a single figure here would recreate exactly the "one constant stands
        # for every server" claim this measurement replaced.
        span = (
            f"{sizes[0]:,} tok/session" if sizes[0] == sizes[-1]
            else f"{sizes[0]:,}-{sizes[-1]:,} tok/session"
        )
        basis = (
            f"sum of each unused server's schema-injection tax observed over "
            f"this window. Each server's per-call size is MEASURED from "
            f"its own tools/list response rather than assumed ({span} across "
            f"the {len(measured_unused)} of {len(finding.unused_servers)} unused "
            f"server(s) that could be measured). The tax table's own "
            f"MCP-schema rows are informational only and never double-count "
            f"into this total."
        )
        if unmeasured_unused:
            basis += (
                f" {unmeasured_unused} unused server(s) could not be measured "
                f"and contribute nothing to this figure. It is a floor, not a "
                f"total."
            )
        if priced:
            finding.past_overspend_usd = round(sum(priced), 6)
            basis += (
                " Dollar figure priced per server through core/pricing.py "
                "at the dominant model observed in that server's sessions "
                "(never a hardcoded rate)."
            )
            if len(priced) < len(measured_unused):
                basis += (
                    f" {len(measured_unused) - len(priced)} of "
                    f"{len(measured_unused)} measured unused server(s) had no "
                    f"priced model observed and are excluded from the "
                    f"dollar sum (token figure still includes them)."
                )
        finding.estimate_basis = basis
    elif finding.unused_servers:
        # Unused servers exist but not one of them could be measured. Saying
        # "nothing cleared the bar" here would be the report asserting more
        # than its data supports in the most misleading direction available:
        # the servers ARE unused, the analyzer just cannot say what they cost.
        finding.notes.append(
            f"{len(finding.unused_servers)} configured MCP server(s) had "
            f"nothing attributable fire in the last "
            f"{UNUSED_RECENCY_WINDOW_DAYS} days. None of them could be "
            f"measured, so no token or dollar figure is stated for them. See "
            f"each row's own construction note for why."
        )
    elif configured:
        finding.notes.append(
            f"No configured MCP server went {UNUSED_RECENCY_WINDOW_DAYS} days "
            f"with nothing attributable firing (or the corpus doesn't hold "
            f"that much history yet for some of them; see each server's own "
            f"row)."
        )

    # The plugin lane adds to the same totals — it is the same shape of waste
    # (an installed dependency paid for continuously and never used) and the
    # populations do not overlap, so there is nothing to double-count between
    # them. Kept as a separate summand rather than folded into the server loop
    # so a reader can always see which lane a figure came from.
    plugin_tokens = sum(p.estimated_tax_tokens_window for p in finding.unused_plugins)
    plugin_usd = [
        p.estimated_tax_usd_window for p in finding.unused_plugins
        if p.estimated_tax_usd_window is not None
    ]
    if plugin_tokens:
        finding.past_overspend_tokens = (finding.past_overspend_tokens or 0) + plugin_tokens
        finding.estimate_basis = (finding.estimate_basis or "") + (
            f" Plus {len(finding.unused_plugins)} enabled plugin(s) with "
            f"nothing attributable firing in the last "
            f"{UNUSED_RECENCY_WINDOW_DAYS} days. Their skill/agent `name: "
            f"description` listing (plus any plugin-contributed MCP server's "
            f"measured schema) is resident in every session, priced per call "
            f"at the window's dominant model. BODIES are never counted. A "
            f"body arrives on invocation, not at session start."
        ).lstrip()
    if plugin_usd:
        finding.past_overspend_usd = round(
            (finding.past_overspend_usd or 0.0) + sum(plugin_usd), 6,
        )
    resident_plugins = [p for p in finding.plugins if p.resident]
    insufficient_plugins = [p for p in resident_plugins if p.insufficient_history]
    partial_plugins = [p for p in resident_plugins if p.partial_use_no_fix]
    if insufficient_plugins:
        finding.notes.append(
            f"{len(insufficient_plugins)} of {finding.plugins_resident} "
            f"resident plugin(s) don't have {UNUSED_RECENCY_WINDOW_DAYS} "
            f"days of session history yet, so no unused claim can be made "
            f"for them. Not a finding, just not enough evidence."
        )
    if partial_plugins:
        finding.notes.append(
            f"{len(partial_plugins)} resident plugin(s) have some "
            f"components that fired in the last "
            f"{UNUSED_RECENCY_WINDOW_DAYS} days and some that did not. "
            f"Enable/disable is whole-plugin only, so no fix is offered "
            f"for a partially-used plugin. See each one's own row."
        )
    if resident_plugins and not finding.unused_plugins and not insufficient_plugins and not partial_plugins:
        finding.notes.append(
            f"{finding.plugins_resident} of {len(finding.plugins)} installed "
            f"plugin(s) are actually resident (the rest are disabled or scoped "
            f"to one project, and cost nothing). Every resident one had "
            f"something attributable fire in the last "
            f"{UNUSED_RECENCY_WINDOW_DAYS} days, so no plugin is flagged."
        )

    finding.measurement_note = _measurement_coverage_note(finding)
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
    from tokenjam.core.agent_config import store_for
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
    # Measuring means STARTING each configured MCP server, so it is gated on a
    # config flag and only ever happens on this out-of-band analyzer pass —
    # never inline on a request. A scoped run (`--projects-root` / an explicit
    # `--db`) never gets here at all, because the scope gate above already
    # returned: measuring would start the operator's real servers on behalf of a
    # run that was explicitly asked to stay out of their environment.
    measure_schemas = bool(getattr(optimize_cfg, "measure_mcp_schemas", True))
    ctx.report.findings["deadweight"] = compute_deadweight_finding(
        ctx.since, ctx.until,
        projects_root=scope.projects_root,
        claude_home=scope.claude_home,
        claude_dir=scope.claude_home,
        cache_dir=default_cache_dir(ctx.config),
        store=store_for(getattr(ctx, "conn", None), getattr(ctx, "write_lock", None)),
        measure_schemas=measure_schemas,
    )
