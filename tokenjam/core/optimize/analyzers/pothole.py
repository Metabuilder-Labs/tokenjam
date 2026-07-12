"""
Cross-session pothole aggregator (self-improve loop, Phase 1: detect + surface).

A "pothole" is a blocker a Claude Code agent silently re-hits across many
unwatched sessions — a wrong-cwd Read, an Edit before a Read, a blocked
sleep-chain, a stale-read race, a domain-blocked WebFetch, and so on. shiploop
only codifies what a human noticed; this module catches what nobody watched.

Pipeline (validated 2026-07-12 against the full local corpus, see
``.claude/self-improve-loop/SPEC.md`` §2/§9):

  1. EXTRACT  — for each session, build the Story (``core.transcript.
     build_session_story``) and fold it through ``core.method_spine.
     build_method_spine`` for the ``delegate``/``dead_end``/``verify``/``act``
     tags, walking subagents recursively. Every step whose tool errored is a
     raw failure episode; its RAW error text comes straight from the Story's
     tool dict (``step["tools"][i]["error"]``) — method_spine's own
     ``_evidence()`` strips that field for privacy, so this module reads the
     Story directly rather than trusting the spine's evidence.
  2. CLUSTER  — normalize each failure into a signature. A handful of known,
     validated families (cwd confusion, edit-before-read, blocked sleep-chain,
     stale-read race, edit string-not-found, deferred-tool-cold, command not
     found, malformed Read offset, WebFetch domain-block) match via regex
     against the raw error text. Regex alone only recovers about half the
     recurring signal (validated) — everything else falls into a generic
     bucket normalized by stripping paths/ids/numbers/timestamps.
  3. DISTILL  — a bounded, cached pass over the residual generic clusters via
     the local ``claude`` CLI (``core.distill``) recovers a human title +
     root cause + proposed fix, and a ``family_key`` so distill can merge
     multiple generic signatures that share one root cause.
  4. NOVELTY  — clusters already codified in a reachable CLAUDE.md/
     learnings.md (walking up from each contributing session's cwd) are
     dropped — the shiploop-miss check.
  5. PROPOSE  — surviving, recurring (>=3 distinct sessions) clusters get a
     conservative token estimate (occurrences x grounded per-turn cost, never
     the inflated afflicted-session footprint), a target rung (§6 of the
     intervention ladder) and a scope (project vs user-global, by how many
     distinct repos the cluster's sessions span).

Never raises: a single unreadable transcript, a distill failure, or a missing
CLAUDE.md is skipped, not fatal — this runs unattended on a schedule.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from tokenjam.core import distill as distill_mod
from tokenjam.core.method_spine import build_method_spine
from tokenjam.core.optimize.registry import register
from tokenjam.core.optimize.types import AnalyzerContext
from tokenjam.core.transcript import build_session_story, resolve_projects_root

# --- Tunables ----------------------------------------------------------------

#: Recurrence threshold (§7 of the spec: K approx 3 distinct sessions).
MIN_RECURRING_SESSIONS = 3
#: How many example sessions to carry per cluster (repro links).
MAX_EXAMPLE_SESSIONS = 3
#: Conservative per-occurrence token cost. A pothole occurrence costs roughly
#: one extra assistant turn's overhead (re-issue the tool call, re-parse the
#: harness context, re-narrate) — NOT the inflated whole-afflicted-session
#: footprint the spec explicitly warns against. This is a heuristic magnitude
#: signal, never a causal claim; surfaced with ``estimate_basis`` below.
GROUNDED_TOKENS_PER_OCCURRENCE = 1_500
#: Cap on how many residual (non-family-matched) clusters get a distill call,
#: bounding both latency and $ spend on a full-corpus run.
MAX_DISTILL_CLUSTERS = 20
#: Minimum cluster size before it's even worth a distill call.
MIN_DISTILL_CLUSTER_SESSIONS = MIN_RECURRING_SESSIONS
DISTILL_MODEL = "haiku"

ESTIMATE_BASIS = (
    "occurrences x a conservative per-turn token cost (one re-issued tool "
    "call + re-narration) — never the inflated whole-session footprint; "
    "review the example sessions before applying a fix"
)
HONESTY_CAVEAT = (
    "Structural failure-signature clustering, not a quality judgment. "
    "Review the example sessions and the proposed fix before applying it."
)

# --- Known, validated pothole families ----------------------------------------
# Each entry: (family key, human title, tool-name filter (None = any),
# regex over the raw error text, default rung, default proposed fix).
# Rungs follow the intervention ladder (SPEC §6): 1 CLAUDE.md note,
# 2 skill/scoped doc, 3 hook, 4 wrapper/script, 5 config/env.
_KNOWN_FAMILIES: list[dict[str, Any]] = [
    {
        "key": "cwd_confusion",
        "title": "cwd / relative-path confusion",
        "tools": None,
        "pattern": re.compile(
            r"no such file or directory|"
            r"file does not exist\.\s*note:\s*your current working directory",
            re.IGNORECASE,
        ),
        "rung": 3,
        "fix": (
            "PreToolUse hook: verify (or inject) an absolute cwd before a Bash "
            "`cd`/relative-path Read so the agent never guesses its working directory."
        ),
    },
    {
        "key": "edit_before_read",
        "title": "Edit/Write before Read",
        "tools": {"Edit", "Write", "MultiEdit", "NotebookEdit"},
        "pattern": re.compile(r"has not been read yet", re.IGNORECASE),
        "rung": 3,
        "fix": (
            "PreToolUse hook: auto-Read the target file first when an Edit/Write "
            "targets a file the session hasn't read yet."
        ),
    },
    {
        "key": "sleep_chain",
        "title": "blocked sleep-chain",
        "tools": {"Bash"},
        # The block usually reads generically ("blocked"/"disallowed"/timed out)
        # with nothing pothole-specific in the wording — the ONE reliable tell is
        # the command itself leading with `sleep`. So this family also matches on
        # the tool's LABEL (its Bash command), not just the error text; see
        # `classify_known_family`.
        "pattern": re.compile(
            r"sleep.{0,40}(block|disallow|not permit)|"
            r"(block|disallow|not permit).{0,40}sleep|"
            r"long.{0,10}leading sleep",
            re.IGNORECASE,
        ),
        "label_pattern": re.compile(r"^\s*sleep\b", re.IGNORECASE),
        "rung": 3,
        "fix": (
            "PreToolUse hook: block a `sleep N && <check>` Bash chain and point the "
            "agent at the Monitor tool instead of a busy-wait."
        ),
    },
    {
        "key": "stale_read_race",
        "title": "file modified since read (linter/hook race)",
        "tools": {"Edit", "Write", "MultiEdit"},
        "pattern": re.compile(r"modified since (it was last read|read)", re.IGNORECASE),
        "rung": 3,
        "fix": (
            "PostToolUse hook: re-Read a file after a formatter/linter hook rewrites "
            "it so the next Edit targets the current bytes."
        ),
    },
    {
        "key": "edit_string_not_found",
        "title": "Edit string-not-found (stale/whitespace/conflict)",
        "tools": {"Edit", "MultiEdit"},
        "pattern": re.compile(
            r"string to replace not found|old_string not found|not found in file",
            re.IGNORECASE,
        ),
        "rung": 3,
        "fix": "PreToolUse hook: re-Read the target file before a follow-up Edit on it.",
    },
    {
        "key": "deferred_tool_cold",
        "title": "deferred tool called cold (no ToolSearch first)",
        "tools": None,
        "pattern": re.compile(
            r"inputvalidationerror|the following issues|required parameter.{0,20}is missing",
            re.IGNORECASE,
        ),
        "rung": 2,
        "fix": (
            "Skill/scoped note: deferred tools need a ToolSearch lookup for their "
            "schema before the first call; optionally a PreToolUse intercept hook."
        ),
    },
    {
        "key": "command_not_found",
        "title": "command not found (bashisms under zsh, bare interpreter)",
        "tools": {"Bash"},
        "pattern": re.compile(r"command not found", re.IGNORECASE),
        "rung": 5,
        "fix": "Config/env fix: alias or install the missing binary in the harness's shell profile.",
    },
    {
        "key": "read_offset_malformed",
        "title": "Read malformed offset (array, not scalar)",
        "tools": {"Read"},
        "pattern": re.compile(r"offset.{0,20}(must be|invalid|expected)|invalid.{0,20}offset", re.IGNORECASE),
        "rung": 1,
        "fix": "CLAUDE.md/skill note: Read's `offset`/`limit` are scalars, not arrays.",
    },
    {
        "key": "webfetch_domain_blocked",
        "title": "WebFetch domain-blocked",
        "tools": {"WebFetch"},
        # Real wording (validated against the local corpus): "Claude Code is
        # unable to fetch from <domain>" — not "not allowed"/"blocked" as the
        # phrasing might suggest.
        "pattern": re.compile(
            r"unable to fetch from|domain.{0,30}(not allowed|block)|"
            r"not allowed to fetch|blocked domain",
            re.IGNORECASE,
        ),
        "rung": 1,
        "fix": "CLAUDE.md/skill note: this domain is blocked — use a search tool or a different source instead.",
    },
]

_FAMILY_BY_KEY = {fam["key"]: fam for fam in _KNOWN_FAMILIES}

# --- Generic (residual-bucket) normalization ----------------------------------

_PATH_RE = re.compile(r"(/[\w.\-]+){2,}")
_UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_HEX_ID_RE = re.compile(r"\b[0-9a-fA-F]{12,}\b")
_NUMBER_RE = re.compile(r"\b\d+\b")
_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:?\d{2})?")
_WS_RE = re.compile(r"\s+")


def _normalize_generic(text: str) -> str:
    """Strip paths/uuids/hex-ids/numbers/timestamps so unrelated values collapse
    into the same signature. Deterministic, no LLM — the fast path before the
    distill pass on whatever this misses."""
    out = _TIMESTAMP_RE.sub("<TS>", text)
    out = _UUID_RE.sub("<UUID>", out)
    out = _PATH_RE.sub("<PATH>", out)
    out = _HEX_ID_RE.sub("<ID>", out)
    out = _NUMBER_RE.sub("<N>", out)
    out = _WS_RE.sub(" ", out).strip().lower()
    return out


def _generic_signature(tool_name: str, error_text: str) -> str:
    normalized = _normalize_generic(error_text)[:160]
    return f"{tool_name}:{normalized}"


#: Not potholes: the tool_result carries ``is_error`` because a HUMAN declined
#: the action (a permission prompt, an AskUserQuestion decline, "Exit plan
#: mode?" answered no) — expected interactive UI, not a blocker an agent
#: silently re-hits. Validated against the local corpus (2026-07-12): these
#: were the single biggest source of noise in the raw failure count, easily
#: mistaken for a recurring "gotcha" by naive clustering. Excluded at
#: extraction time so they never enter a cluster at all.
_USER_DECLINE_RE = re.compile(
    r"doesn.t want to proceed with this tool use|"
    r"^exit plan mode\?\s*$",
    re.IGNORECASE,
)


def is_user_decline(error_text: str) -> bool:
    """True if this 'failure' is really a human declining an action, not a
    pothole — see ``_USER_DECLINE_RE``."""
    return bool(error_text) and bool(_USER_DECLINE_RE.search(error_text.strip()))


def classify_known_family(tool_name: str, error_text: str, label: str = "") -> str | None:
    """Return the matching known family key, or None. Tried in declaration order
    (first match wins; the families are mutually near-exclusive by wording).

    Most families match on the raw error text alone. A few (e.g. the blocked
    sleep-chain, whose block message is often generic — "timed out"/"blocked"
    with nothing pothole-specific in the wording) additionally require the
    tool's ``label`` (its command/arg) to match a ``label_pattern`` — the
    reliable tell is the command itself, not the error text.
    """
    if not error_text:
        return None
    for fam in _KNOWN_FAMILIES:
        tools = fam["tools"]
        if tools is not None and tool_name not in tools:
            continue
        if not fam["pattern"].search(error_text):
            continue
        label_pattern = fam.get("label_pattern")
        if label_pattern is not None and not label_pattern.search(label or ""):
            continue
        return fam["key"]
    return None


# --- Extraction ----------------------------------------------------------------

@dataclass
class FailureEpisode:
    """One erroring tool call, structurally tagged, with its raw error text."""
    session_id:  str
    repo:        str            # human-ish repo label (agent_id sans provider prefix, or "unknown")
    ts:          str | None
    tool_name:   str
    label:       str            # the tool's short arg label (never full input)
    error_text:  str            # RAW, already length-capped by transcript.py
    kind:        str            # method_spine move kind: delegate/dead_end/verify/act
    is_retry:    bool
    depth:       int            # 0 = main thread, >0 = nested subagent


def _walk_moves(
    steps: list[dict[str, Any]], moves: list[dict[str, Any]], depth: int,
) -> Iterable[tuple[dict[str, Any], dict[str, Any], int]]:
    """Zip a Story's raw steps with method_spine's moves (1:1, same order),
    recursing into delegate moves' expanded subagent stories. Mirrors
    method_spine's own recursion so kinds line up exactly; never re-derives
    them independently."""
    for step, move in zip(steps, moves):
        yield step, move, depth
        if move.get("kind") != "delegate":
            continue
        subs = ([step["subagent"]] if step.get("subagent") else []) + list(
            step.get("subagents") or []
        )
        delegations = move.get("delegations") or []
        for sub_dict, delegation in zip(subs, delegations):
            if delegation.get("capped") is not None:
                continue  # not expanded — nothing to walk
            sub_steps = [s for s in (sub_dict.get("steps") or []) if "omitted" not in s]
            sub_spine = delegation.get("spine") or []
            yield from _walk_moves(sub_steps, sub_spine, depth + 1)


def extract_failures_for_session(
    session_id: str,
    repo: str,
    projects_root: Path | str | None = None,
) -> list[FailureEpisode]:
    """Every erroring tool call in one session (main thread + subagents).

    Returns ``[]`` when the session has no on-disk transcript (SDK session,
    pruned). Never raises — a malformed transcript yields whatever could be
    parsed (``build_session_story`` already tolerates bad lines).
    """
    story = build_session_story(session_id, projects_root=projects_root, include_subagents=True)
    if story is None:
        return []

    real_steps = [s for s in (story.get("steps") or []) if "omitted" not in s]
    spine = build_method_spine(story)

    failures: list[FailureEpisode] = []
    for step, move, depth in _walk_moves(real_steps, spine, 0):
        for tool in step.get("tools") or []:
            if tool.get("status") != "error":
                continue
            error_text = tool.get("error") or ""
            if is_user_decline(error_text):
                continue  # a human's own choice, not a pothole — see is_user_decline
            failures.append(FailureEpisode(
                session_id=session_id,
                repo=repo,
                ts=step.get("ts"),
                tool_name=tool.get("name") or "unknown",
                label=tool.get("label") or "",
                error_text=error_text,
                kind=move.get("kind", "act"),
                is_retry=bool(move.get("is_retry")),
                depth=depth,
            ))
    return failures


# --- Clustering ------------------------------------------------------------

@dataclass
class _RawCluster:
    signature:      str
    family_key:     str | None      # None until a known family or distill assigns one
    title:          str
    failures:       list[FailureEpisode] = field(default_factory=list)

    @property
    def session_ids(self) -> set[str]:
        return {f.session_id for f in self.failures}

    @property
    def repos(self) -> set[str]:
        return {f.repo for f in self.failures}


def cluster_failures(failures: list[FailureEpisode]) -> dict[str, _RawCluster]:
    """Bucket failures by known family first, else a generic normalized signature."""
    clusters: dict[str, _RawCluster] = {}
    for failure in failures:
        family_key = classify_known_family(failure.tool_name, failure.error_text, failure.label)
        if family_key is not None:
            sig = family_key
            title = _FAMILY_BY_KEY[family_key]["title"]
        else:
            sig = _generic_signature(failure.tool_name, failure.error_text)
            title = f"{failure.tool_name}: {failure.error_text[:60] or failure.label}"
        bucket = clusters.get(sig)
        if bucket is None:
            bucket = _RawCluster(signature=sig, family_key=family_key, title=title)
            clusters[sig] = bucket
        bucket.failures.append(failure)
    return clusters


def _recurring(clusters: dict[str, _RawCluster], min_sessions: int) -> list[_RawCluster]:
    return [c for c in clusters.values() if len(c.session_ids) >= min_sessions]


# --- Distill pass over the residual (non-family) bucket -----------------------

def _distill_cache_dir() -> Path:
    return Path.home() / ".tj" / "distill_cache" / "pothole"


def _cluster_hash(cluster: _RawCluster) -> str:
    payload = cluster.signature + "|" + "|".join(
        sorted(f.error_text for f in cluster.failures[:10])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def distill_pothole_cluster(
    tool_name: str, samples: list[str], *, model: str = DISTILL_MODEL, timeout: int = 60,
) -> dict[str, str]:
    """Ask the local ``claude`` CLI to name a residual failure cluster.

    Returns ``{"title", "family_key", "fix"}`` — ``family_key`` is a short
    slug distill invents so several generic signatures sharing one root cause
    merge under it. Returns ``{}`` on any failure (missing CLI, bad JSON,
    timeout) — never raises. Shells out via ``core.distill._invoke_claude``,
    the same pinned invocation ``distill_titles`` uses.
    """
    if not samples:
        return {}

    numbered = "\n".join(f"{i + 1}. {s[:300]}" for i, s in enumerate(samples[:8]))
    prompt = (
        "Below are raw error messages a coding agent hit repeatedly while using the "
        f"`{tool_name}` tool.\n"
        "Decide whether they share ONE root cause (an environmental/procedural/tooling "
        "gotcha, NOT a one-off bug in the task's own code). "
        'Return ONLY a JSON object: {"title": "<=8 word name>", "family_key": '
        '"<short_snake_case_slug>", "fix": "<one sentence proposed fix>"}. '
        "No prose, no code fence required.\n\n"
        f"{numbered}"
    )

    result = distill_mod._invoke_claude(prompt, model=model, timeout=timeout)
    if result is None:
        return {}

    match = distill_mod._JSON_OBJECT_RE.search(result)
    if not match:
        return {}
    import json as _json

    try:
        raw = _json.loads(match.group(0))
    except (_json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    title = str(raw.get("title") or "").strip()
    family_key = str(raw.get("family_key") or "").strip().lower().replace(" ", "_")
    fix = str(raw.get("fix") or "").strip()
    if not title or not family_key:
        return {}
    return {"title": title, "family_key": family_key, "fix": fix}


def _distill_cached(tool_name: str, cluster: _RawCluster, cache_dir: Path) -> dict[str, str]:
    """Cached wrapper (keyed by cluster content hash) — never re-spends on an
    unchanged cluster. Best-effort: any I/O error degrades to a cache miss."""
    import json as _json

    cache_file = cache_dir / f"{_cluster_hash(cluster)}.json"
    try:
        cached = _json.loads(cache_file.read_text(encoding="utf-8"))
        if isinstance(cached, dict) and cached.get("title"):
            return cached
    except (OSError, ValueError):
        pass

    samples = [f.error_text for f in cluster.failures if f.error_text][:8]
    result = distill_pothole_cluster(tool_name, samples)
    if not result:
        return {}
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(_json.dumps(result), encoding="utf-8")
    except OSError:
        pass
    return result


def apply_distill_to_residual(
    clusters: list[_RawCluster], *, cache_dir: Path | None = None, enabled: bool = True,
) -> list[_RawCluster]:
    """Distill the top (by session count) residual clusters and merge any that
    distill assigns the same ``family_key``. Bounded by ``MAX_DISTILL_CLUSTERS``
    so a huge residual bucket never triggers unbounded $ spend.

    Clusters already matched to a known family are left untouched. When
    ``enabled`` is False (no ``claude`` CLI / caller opt-out) the residual
    clusters pass through with their generic titles, unmerged.
    """
    if cache_dir is None:
        cache_dir = _distill_cache_dir()

    known = [c for c in clusters if c.family_key is not None]
    residual = [c for c in clusters if c.family_key is None]
    if not enabled or not residual:
        return clusters

    residual.sort(key=lambda c: len(c.session_ids), reverse=True)
    to_distill = [c for c in residual if len(c.session_ids) >= MIN_DISTILL_CLUSTER_SESSIONS]
    to_distill = to_distill[:MAX_DISTILL_CLUSTERS]
    untouched = [c for c in residual if c not in to_distill]

    merged: dict[str, _RawCluster] = {}
    for cluster in to_distill:
        tool_name = cluster.failures[0].tool_name if cluster.failures else "unknown"
        result = _distill_cached(tool_name, cluster, cache_dir)
        if not result:
            untouched.append(cluster)
            continue
        family_key = f"distilled:{result['family_key']}"
        target = merged.get(family_key)
        if target is None:
            target = _RawCluster(signature=family_key, family_key=family_key, title=result["title"])
            merged[family_key] = target
            # Stash the distilled fix on the family table so proposal-building
            # can look it up like a known family (keeps one code path).
            _FAMILY_BY_KEY.setdefault(family_key, {
                "key": family_key, "title": result["title"], "tools": None,
                "pattern": None, "rung": 1, "fix": result.get("fix") or "",
            })
        target.failures.extend(cluster.failures)

    return known + list(merged.values()) + untouched


# --- Novelty filter (cross-ref codified knowledge) -----------------------------

def _candidate_doc_paths(repo_cwds: set[str]) -> list[Path]:
    """CLAUDE.md/learnings.md reachable from any contributing session's cwd,
    walking up a few parent levels so a workspace-root doc (a meta-repo's
    CLAUDE.md above the sub-repo) is found too, not just the sub-repo's own."""
    names = ("CLAUDE.md", "learnings.md")
    seen: set[Path] = set()
    paths: list[Path] = []
    for cwd in repo_cwds:
        base = Path(cwd) if cwd else None
        if base is None or not base.exists():
            continue
        for ancestor in [base, *base.parents[:3]]:
            for name in names:
                candidate = ancestor / name
                if candidate in seen:
                    continue
                seen.add(candidate)
                if candidate.is_file():
                    paths.append(candidate)
    return paths


def _doc_text(paths: list[Path], max_chars: int = 200_000) -> str:
    parts: list[str] = []
    total = 0
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        parts.append(text.lower())
        total += len(text)
        if total >= max_chars:
            break
    return "\n".join(parts)


#: A few representative keywords per known family — a cheap, inspectable
#: novelty heuristic (NOT an LLM call): "already codified" iff every keyword
#: co-occurs somewhere in the reachable docs. Deliberately conservative (few,
#: specific terms) so a coincidental single-word match doesn't wrongly drop a
#: real, still-uncodified pothole.
#: One DISTINCTIVE multi-word phrase per known family — deliberately full
#: phrases (not independent short words ANDed together). Validated the hard
#: way (2026-07-12): an earlier version used tuples like ("read", "before
#: editing") ANDed independently, and "read" alone is common enough that it
#: co-occurred with unrelated text in reachable docs, silently dropping the
#: single BIGGEST validated pothole (edit-before-read, 178 sessions) as
#: "already codified" when it demonstrably wasn't. A short generic word is
#: not a novelty signal; a verbatim-ish multi-word phrase close to the
#: harness's actual wording is.
_FAMILY_NOVELTY_PHRASES: dict[str, str] = {
    "cwd_confusion": "no such file or directory",
    "edit_before_read": "has not been read yet",
    "sleep_chain": "foreground sleep",
    "stale_read_race": "modified since read",
    "edit_string_not_found": "string to replace not found",
    "deferred_tool_cold": "toolsearch",
    "command_not_found": "command not found",
    "read_offset_malformed": "offset must be a scalar",
    "webfetch_domain_blocked": "unable to fetch from",
}


def is_already_codified(cluster: _RawCluster, doc_text: str) -> bool:
    """Heuristic novelty check (the shiploop-miss check): does a reachable
    CLAUDE.md/learnings.md already name this exact gotcha?

    Deliberately narrow — a single distinctive phrase per KNOWN family (see
    ``_FAMILY_NOVELTY_PHRASES``). Residual/distilled clusters (no known
    family) have no safe phrase to check against, so they're always treated
    as novel: a missed "already codified" drop costs a human one glance at
    the review inbox; a wrongful drop silently hides the exact signal this
    detector exists to surface. Never guess on a generic word.
    """
    if not doc_text or not cluster.family_key:
        return False
    phrase = _FAMILY_NOVELTY_PHRASES.get(cluster.family_key)
    if not phrase:
        return False
    return phrase in doc_text


# --- Proposal building -----------------------------------------------------

@dataclass
class PotholeExample:
    session_id: str
    repo:       str
    ts:         str | None
    snippet:    str            # short excerpt of the raw error (evidence)


@dataclass
class PotholeCluster:
    signature:                 str
    family_key:                str | None
    title:                     str
    sessions:                  int
    occurrences:                int
    repos:                      list[str]
    rung:                       int             # 1-5, SPEC §6 intervention ladder
    scope:                      str              # "project" | "user-global"
    proposed_fix:                str
    examples:                    list[PotholeExample] = field(default_factory=list)
    estimated_recoverable_tokens: int = 0
    confidence:                   str = "heuristic"
    novel:                        bool = True
    # Phase 2 (apply) — best-effort cwd of the cluster's (sole, if project-
    # scoped) repo, and a suggested rung-1 write target derived from it. Both
    # are just a DEFAULT for the Review inbox card's scope/target override
    # (§7's "repo-identity is noisy" — never applied blindly); "" when
    # unknown (multi-repo / user-global / no cwd could be resolved).
    repo_cwd:                     str = ""
    suggested_target:             str = ""


@dataclass
class PotholeFinding:
    clusters:            list[PotholeCluster] = field(default_factory=list)
    sessions_scanned:     int = 0
    failures_examined:    int = 0
    distilled_clusters:   int = 0
    dropped_codified:     int = 0
    estimated_recoverable_tokens: int | None = None
    estimate_basis:        str = ESTIMATE_BASIS
    estimate_confidence:   str = "heuristic"
    caveat:                 str = HONESTY_CAVEAT


def _snippet(failure: FailureEpisode) -> str:
    text = failure.error_text or failure.label
    return text[:200]


def _scope_for(repos: set[str]) -> str:
    """§7: concentrated in one repo -> project; spread across many -> user-global."""
    return "project" if len(repos) <= 1 else "user-global"


def build_proposals(
    clusters: list[_RawCluster],
    *,
    min_sessions: int = MIN_RECURRING_SESSIONS,
    doc_text: str = "",
    repo_cwd_map: dict[str, str] | None = None,
) -> tuple[list[PotholeCluster], int]:
    """Turn surviving raw clusters into ranked proposals. Returns
    ``(proposals, dropped_codified_count)``.

    ``repo_cwd_map`` (repo label -> a representative cwd) is optional,
    best-effort enrichment used only to pre-fill the Apply stage's suggested
    target path (Phase 2) — clustering itself needs none of it.
    """
    from tokenjam.core.optimize.pothole_apply import default_target_path, slugify

    repo_cwd_map = repo_cwd_map or {}
    proposals: list[PotholeCluster] = []
    dropped = 0
    for cluster in clusters:
        sessions = cluster.session_ids
        if len(sessions) < min_sessions:
            continue
        if is_already_codified(cluster, doc_text):
            dropped += 1
            continue

        family = _FAMILY_BY_KEY.get(cluster.family_key or "")
        rung = family["rung"] if family else 1
        fix = family["fix"] if family else "Review examples — no known fix template matched."

        repos = sorted(cluster.repos)
        occurrences = len(cluster.failures)
        examples = [
            PotholeExample(
                session_id=f.session_id, repo=f.repo, ts=f.ts, snippet=_snippet(f),
            )
            for f in sorted(cluster.failures, key=lambda f: f.ts or "", reverse=True)[:MAX_EXAMPLE_SESSIONS]
        ]

        scope = _scope_for(cluster.repos)
        repo_cwd = repo_cwd_map.get(repos[0], "") if len(repos) == 1 else ""
        try:
            suggested_target = default_target_path(
                rung, scope, repo_cwd, slugify(cluster.title),
            )
        except Exception:
            suggested_target = ""   # never let a bad path computation sink the proposal

        proposals.append(PotholeCluster(
            signature=cluster.signature,
            family_key=cluster.family_key,
            title=cluster.title,
            sessions=len(sessions),
            occurrences=occurrences,
            repos=repos,
            rung=rung,
            scope=scope,
            proposed_fix=fix,
            examples=examples,
            estimated_recoverable_tokens=occurrences * GROUNDED_TOKENS_PER_OCCURRENCE,
            novel=True,
            repo_cwd=repo_cwd,
            suggested_target=suggested_target,
        ))

    proposals.sort(key=lambda p: p.sessions, reverse=True)
    return proposals, dropped


# --- Orchestration (pure, no ctx dependency — testable directly) --------------

def analyze_potholes(
    sessions: list[tuple[str, str]],     # [(session_id, repo), ...]
    *,
    projects_root: Path | str | None = None,
    min_sessions: int = MIN_RECURRING_SESSIONS,
    distill_enabled: bool = True,
    distill_cache_dir: Path | None = None,
    codified_doc_text: str = "",
    repo_cwd_map: dict[str, str] | None = None,
) -> PotholeFinding:
    """Full pipeline over an explicit session list — the pure core the
    registry entry point and the on-disk cache job both call. Never raises."""
    all_failures: list[FailureEpisode] = []
    scanned = 0
    for session_id, repo in sessions:
        try:
            failures = extract_failures_for_session(session_id, repo, projects_root)
        except Exception:
            continue
        scanned += 1
        all_failures.extend(failures)

    raw_clusters = cluster_failures(all_failures)
    recurring = _recurring(raw_clusters, min_sessions)
    distilled = apply_distill_to_residual(
        recurring, cache_dir=distill_cache_dir, enabled=distill_enabled,
    )
    distilled_count = sum(1 for c in distilled if (c.family_key or "").startswith("distilled:"))

    proposals, dropped = build_proposals(
        distilled, min_sessions=min_sessions, doc_text=codified_doc_text,
        repo_cwd_map=repo_cwd_map,
    )
    total_tokens = sum(p.estimated_recoverable_tokens for p in proposals)

    return PotholeFinding(
        clusters=proposals,
        sessions_scanned=scanned,
        failures_examined=len(all_failures),
        distilled_clusters=distilled_count,
        dropped_codified=dropped,
        estimated_recoverable_tokens=total_tokens if proposals else None,
    )


# --- Registry entry point -----------------------------------------------------

def _repo_map_from_db(conn) -> dict[str, str]:
    """``session_id -> repo`` from the ``sessions`` table's ``agent_id``
    (``claude-code-<basename(cwd)>`` — see ``core.backfill._agent_id_from_cwd``).
    Best-effort: an empty/failed query just means every session falls back to
    "unknown" and clusters still work, just with weaker repo/scope info."""
    try:
        rows = conn.execute("SELECT session_id, agent_id FROM sessions").fetchall()
    except Exception:
        return {}
    out: dict[str, str] = {}
    for session_id, agent_id in rows:
        repo = str(agent_id or "unknown")
        if repo.startswith("claude-code-"):
            repo = repo[len("claude-code-"):]
        out[str(session_id)] = repo
    return out


def _repo_cwd_map_for(sessions: list[tuple[str, str]], projects_root: Path) -> dict[str, str]:
    """Best-effort repo-label -> cwd, for the novelty doc search AND (Phase 2)
    the Apply stage's suggested target path. Derived from the encoded project
    directory name is unreliable, so this reads each session's transcript's
    first ``cwd`` field directly (cheap: short-circuits after the first hit)
    for one representative session per repo."""
    from tokenjam.core.transcript import _locate_transcript, read_records

    out: dict[str, str] = {}
    for session_id, repo in sessions:
        if repo in out:
            continue
        path = _locate_transcript(session_id, projects_root)
        if path is None:
            continue
        for record in read_records(path)[:5]:
            cwd = record.get("cwd")
            if isinstance(cwd, str) and cwd:
                out[repo] = cwd
                break
    return out


def compute_pothole_finding(
    conn: Any | None = None,
    since: Any | None = None,
    *,
    projects_root: Path | str | None = None,
    distill_enabled: bool = True,
) -> PotholeFinding:
    """Standalone entry point that doesn't need a full ``AnalyzerContext`` —
    used by the serve-time background cache job (``api/routes/pothole.py``)
    and by tests. ``conn`` is an OPTIONAL DuckDB connection used only for
    repo-name enrichment (``sessions.agent_id``); pass ``None`` and every
    session falls back to a ``"unknown"`` repo label — clustering itself needs
    no DB at all, it's a pure filesystem scan.

    Full-corpus by design: enumerates every on-disk Claude Code transcript
    (not window-scoped like the other analyzers — a pothole recurring across
    months of history is exactly the signal this detector exists to find),
    optionally pre-filtered to ``since`` when the caller wants an incremental
    scan. Heavy (tens of seconds over a full local corpus) — callers that
    serve this over HTTP MUST cache the result, not compute it per-request.
    """
    root = resolve_projects_root(projects_root)
    repo_map = _repo_map_from_db(conn) if conn is not None else {}

    paths = sorted(root.rglob("*.jsonl")) if root.exists() else []
    sessions: list[tuple[str, str]] = []
    for path in paths:
        if since is not None:
            try:
                from datetime import datetime, timezone
                mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                if mtime < since:
                    continue
            except OSError:
                continue
        session_id = path.stem
        sessions.append((session_id, repo_map.get(session_id, "unknown")))

    doc_text = ""
    repo_cwd_map: dict[str, str] = {}
    try:
        repo_cwd_map = _repo_cwd_map_for(sessions, root)
        doc_text = _doc_text(_candidate_doc_paths(set(repo_cwd_map.values())))
    except Exception:
        doc_text = ""

    return analyze_potholes(
        sessions, projects_root=root, codified_doc_text=doc_text,
        distill_enabled=distill_enabled, repo_cwd_map=repo_cwd_map,
    )


@register("pothole")
def run(ctx: AnalyzerContext) -> None:
    """Registry entry point. Attaches a ``PotholeFinding`` to
    ``ctx.report.findings["pothole"]`` — see ``compute_pothole_finding`` for
    the full-corpus behaviour and performance note.
    """
    ctx.report.findings["pothole"] = compute_pothole_finding(ctx.conn, ctx.since)
