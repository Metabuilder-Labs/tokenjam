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
feeds ``DeadweightFinding.estimated_recoverable_tokens`` — only the C1
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
from pathlib import Path
from typing import Any

from tokenjam.core.optimize.registry import register
from tokenjam.core.optimize.types import AnalyzerContext
from tokenjam.core.transcript import _SYSTEM_REMINDER_RE, read_records, resolve_projects_root

# --- Tunables ------------------------------------------------------------

#: A server must be configured-present in at least this many DISTINCT
#: sessions, with zero invocations across all of them, before it's flagged
#: dead weight (spec: "start N=10").
MIN_SESSIONS_DEADWEIGHT = 10

#: How many example session ids a dead server's card carries as evidence
#: (mirrors relearn.py's MAX_EXAMPLE_SESSIONS convention).
MAX_EXAMPLE_SESSIONS = 3

#: Full MCP-connector schema-injection tax, per server, when its tool schemas
#: are loaded (not deferred) — a documented community/founder-research figure
#: (~25K tokens/call for an attached MCP connector's injected tool
#: definitions; see .claude/context/research/evidence/
#: subscription-vs-cost-framing.md and feature-context-diagnostic.md), NOT a
#: live per-call measurement — the on-disk transcript carries no per-schema
#: token count (see core/context_diagnostic.py's MCP_INJECTION_PARK_NOTE).
#: `estimated`, conservative, cited in the card footnote.
FULL_SCHEMA_TAX_TOKENS = 25_000

#: When a session's transcript shows this server's tools in a DEFERRED
#: listing (ToolSearch-style), its schemas are NOT loaded that turn — only a
#: short name+description line per tool appears in the listing.
#: Conservative estimate: ~10 tools x ~40 tokens/line for a typically-sized
#: server. Never used to claim the full tax for a deferred session.
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


def _global_config_path() -> Path:
    # Resolved LAZILY (never at import time) so a test patching HOME sees the
    # fake home, never the developer's real ~/.claude.json.
    return Path.home() / ".claude.json"


@dataclass
class ConfiguredServer:
    """One MCP server as read off config, and where it reaches."""
    name:   str
    scope:  str                      # "user" | "project"
    source: str                      # config file path (str, for the card)
    cwds:   set[str] = field(default_factory=set)  # project scope: reachable cwds


def enumerate_configured_servers(repo_cwds: set[str]) -> dict[str, ConfiguredServer]:
    """Read-only enumeration of MCP servers across the three config
    locations: project ``.mcp.json`` / ``.claude/settings*.json`` under each
    given session cwd, plus the global ``~/.claude.json``. Never edits a
    config file (advise-only in v1 — see the module docstring).

    A user-scoped (global) server always wins scope over a same-named
    project entry: the global entry already reaches every session, so
    downgrading it to "project" would only narrow its true presence.
    """
    servers: dict[str, ConfiguredServer] = {}

    global_path = _global_config_path()
    if global_path.is_file():
        for name in _mcp_server_names(global_path):
            servers[name] = ConfiguredServer(name=name, scope="user", source=str(global_path))

    for cwd in repo_cwds:
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
    return servers


def server_still_configured(name: str, source: str) -> bool:
    """Read-only re-check: does ``name`` still appear in the ``mcpServers``
    block of its ORIGINAL detected config file (``source``)?

    Used by ``cost_verify``'s deadweight branch after a mark-applied to tell
    "still configured, nothing to measure yet" apart from "actually removed
    or project-scoped, measure the token drop". A missing file and a
    present-but-empty-of-this-entry file both read as "no longer
    configured" — either way the tax stopped. Missing ``name``/``source``
    can't be verified at all, so this conservatively reports "still
    configured" rather than falsely claiming a removal.
    """
    if not name or not source:
        return True
    path = Path(source)
    if not path.is_file():
        return False
    return name in _mcp_server_names(path)


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
    """Best-effort session cwd, from the first record that carries one
    (mirrors relearn.py's ``_repo_cwd_map_for``)."""
    for record in records[:5]:
        cwd = record.get("cwd")
        if isinstance(cwd, str) and cwd:
            return cwd
    return ""


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
    reminder_chars_by_source: dict[str, int] = field(default_factory=dict)


def _analyze_session(records: list[dict[str, Any]]) -> _SessionSignal:
    """One pass over a session's raw records: MCP invocation counts, which
    servers appeared in a deferred-tools listing, and the C2 tax-table
    source buckets (measured off the FIRST system-reminder blob only —
    Claude Code injects it once at session start; later turns don't repeat
    it, so summing across turns would overcount)."""
    signal = _SessionSignal()
    reminder_measured = False

    for record in records:
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

        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                server = _mcp_server_from_tool_name(str(block.get("name") or ""))
                if server:
                    signal.mcp_invocations[server] = signal.mcp_invocations.get(server, 0) + 1

    return signal


def _tax_construction_note(non_deferred: int, deferred_sessions: int, sessions_present: int) -> str:
    if sessions_present == 0:
        return ""
    if deferred_sessions == 0:
        return (
            f"{FULL_SCHEMA_TAX_TOKENS:,} tok/session (full schema injection), "
            f"cited estimate, not a live per-call measurement."
        )
    if non_deferred == 0:
        return (
            f"{DEFERRED_SCHEMA_TAX_TOKENS:,} tok/session; ToolSearch deferred "
            f"this server's schemas in every observed session (name and "
            f"description line only, never the full schema tax)."
        )
    return (
        f"{FULL_SCHEMA_TAX_TOKENS:,} tok/session when fully loaded "
        f"({non_deferred} of {sessions_present} sessions) blended with "
        f"{DEFERRED_SCHEMA_TAX_TOKENS:,} tok/session when ToolSearch defers "
        f"this server's schemas ({deferred_sessions} of {sessions_present} "
        f"sessions); never claims the full tax for a deferred session."
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
    estimated_tax_tokens_per_session: int
    estimated_tax_tokens_90d:         int
    tax_construction:                 str
    fix:                              str
    example_sessions:                 list[str] = field(default_factory=list)


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
    estimated_recoverable_tokens: int | None = None
    estimate_basis:                str = ""
    estimate_confidence:            str = "estimated"
    caveat:                          str = DEADWEIGHT_HONESTY_CAVEAT
    notes:                            list[str] = field(default_factory=list)


# --- Orchestration (pure, no ctx dependency — testable directly) ----------

def compute_deadweight_finding(
    since: datetime,
    until: datetime,
    *,
    projects_root: Path | str | None = None,
    window_days: float | None = None,
) -> DeadweightFinding:
    """Full pipeline over a window of Claude Code transcripts. Never raises —
    a missing projects root, an unreadable transcript, or a malformed config
    file is skipped, not fatal.
    """
    finding = DeadweightFinding()
    root = resolve_projects_root(projects_root)
    if not root.exists():
        return finding

    session_paths: list[tuple[str, Path]] = []
    for path in sorted(root.rglob("*.jsonl")):
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            continue
        if mtime < since or mtime >= until:
            continue
        session_paths.append((path.stem, path))

    finding.sessions_scanned = len(session_paths)
    if not session_paths:
        return finding

    per_session: dict[str, _SessionSignal] = {}
    session_cwds: dict[str, str] = {}
    for session_id, path in session_paths:
        try:
            records = read_records(path)
        except Exception:
            continue
        session_cwds[session_id] = _session_cwd(records)
        per_session[session_id] = _analyze_session(records)

    repo_cwds = {c for c in session_cwds.values() if c}
    configured = enumerate_configured_servers(repo_cwds)
    finding.configured_servers = len(configured)
    if not configured:
        return finding

    days = window_days if window_days and window_days > 0 else max(
        (until - since).total_seconds() / 86400.0, 1.0,
    )
    projection_factor = 90.0 / days

    tax_rows: list[ContextTaxRow] = []
    reminder_bucket_totals: dict[str, list[int]] = {}

    for server in configured.values():
        sessions_present = 0
        invocations = 0
        deferred_sessions = 0
        example_sessions: list[str] = []
        for session_id, signal in per_session.items():
            deferred_here = server.name in signal.deferred_servers
            present = deferred_here or server.scope == "user" or (
                session_cwds.get(session_id, "") in server.cwds
            )
            if not present:
                continue
            sessions_present += 1
            invocations += signal.mcp_invocations.get(server.name, 0)
            if deferred_here:
                deferred_sessions += 1
            if len(example_sessions) < MAX_EXAMPLE_SESSIONS:
                example_sessions.append(session_id)

        dead = sessions_present >= MIN_SESSIONS_DEADWEIGHT and invocations == 0
        non_deferred = max(sessions_present - deferred_sessions, 0)
        tax_per_session = (
            round(
                (non_deferred * FULL_SCHEMA_TAX_TOKENS + deferred_sessions * DEFERRED_SCHEMA_TAX_TOKENS)
                / sessions_present
            )
            if sessions_present else 0
        )
        tax_90d = round(tax_per_session * sessions_present * projection_factor)

        row = ServerDeadweight(
            name=server.name,
            scope=server.scope,
            source=server.source,
            sessions_present=sessions_present,
            invocations=invocations,
            deferred_sessions=deferred_sessions,
            dead=dead,
            estimated_tax_tokens_per_session=tax_per_session,
            estimated_tax_tokens_90d=tax_90d,
            tax_construction=_tax_construction_note(non_deferred, deferred_sessions, sessions_present),
            fix=(
                f"Remove or project-scope the `{server.name}` MCP server "
                f"({server.source}); zero tool calls across {sessions_present} "
                f"session(s) in this window."
            ),
            example_sessions=example_sessions,
        )
        finding.servers.append(row)
        if sessions_present > 0:
            tax_rows.append(ContextTaxRow(
                source=f"MCP schema: {server.name}",
                sessions=sessions_present,
                avg_tokens_per_session=tax_per_session,
                total_tokens_window=tax_per_session * sessions_present,
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
        finding.estimated_recoverable_tokens = sum(
            s.estimated_tax_tokens_90d for s in finding.dead_servers
        )
        finding.estimate_basis = (
            f"sum of each dead server's projected 90-day schema-injection tax "
            f"({FULL_SCHEMA_TAX_TOKENS:,} tok/session full, "
            f"{DEFERRED_SCHEMA_TAX_TOKENS:,} tok/session when deferred); the "
            f"tax table's own MCP-schema rows are informational only and "
            f"never double-count into this total."
        )
    elif configured:
        finding.notes.append(
            f"No configured MCP server cleared the dead-weight bar "
            f"(>= {MIN_SESSIONS_DEADWEIGHT} sessions present, 0 invocations)."
        )

    return finding


# --- Registry entry point ---------------------------------------------------

@register("deadweight")
def run(ctx: AnalyzerContext) -> None:
    """Registry entry point. Attaches a ``DeadweightFinding`` to
    ``ctx.report.findings["deadweight"]``. Claude Code transcripts lane only
    — reads on-disk JSONL directly, never ``ctx.conn`` (no DB spans needed).
    """
    ctx.report.findings["deadweight"] = compute_deadweight_finding(
        ctx.since, ctx.until, window_days=ctx.window_days,
    )
