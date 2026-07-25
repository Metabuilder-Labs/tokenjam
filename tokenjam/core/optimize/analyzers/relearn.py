"""
Cross-session relearn aggregator (self-improve loop, Phase 1: detect + surface).

A "relearn" is a blocker a Claude Code agent silently re-hits across many
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
from tokenjam.core.optimize.clustering import group_by_key, mask_variables, recurring
from tokenjam.core.optimize.projection import build_projection_basis
from tokenjam.core.optimize.rate_profile import RateProfile, blended_rate_profile
from tokenjam.core.optimize.registry import register
from tokenjam.core.optimize.types import AnalyzerContext
from tokenjam.core.transcript import build_session_story, resolve_projects_root

# --- Tunables ----------------------------------------------------------------

#: Recurrence threshold (§7 of the spec: K approx 3 distinct sessions).
MIN_RECURRING_SESSIONS = 3
#: How many example sessions to carry per cluster (repro links).
MAX_EXAMPLE_SESSIONS = 3
#: Conservative per-occurrence token cost. A relearn occurrence costs roughly
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

# --- Per-occurrence re-read tail ---------------------------------------------
# A failure does not cost one turn's tokens once. Its text lands in the
# conversation and is re-sent on every LATER call that still carries it, so an
# occurrence at call index k of a session whose context survives `tail` more
# calls actually bills
#
#     GROUNDED_TOKENS_PER_OCCURRENCE x input_rate
#   + GROUNDED_TOKENS_PER_OCCURRENCE x tail x cache_read_rate
#
# Everything below is expressed in INPUT-TOKEN EQUIVALENTS so the token figure
# and the dollar figure stay proportional to one another (the existing
# recoverable contract prices one against the other, and the write budget nets
# a rule's standing cost against the token side). A re-read token bills at the
# cache-read rate — exactly 0.100 x the input rate for every Anthropic model in
# pricing/models.toml — so one occurrence is worth
#
#     1 + cache_read_ratio x tail
#
# input-token equivalents rather than 1. That ratio is measured from the
# cluster's own models, never assumed.

#: A prompt whose size falls to at most this share of the previous turn's has
#: had its context window reset — a `/compact`, a resume, or a fresh window.
#: The tail STOPS there: the failure text is no longer in what gets re-sent.
#: Without this the multiplier assumes a failure is re-read for the rest of the
#: session, which is measurably wrong on long automated-harness sessions (the
#: ones that dominate a corpus by call count) and inflates the aggregate.
COMPACTION_PROMPT_DROP_RATIO = 0.5

ESTIMATE_BASIS = (
    "occurrences x a conservative per-turn token cost (one re-issued tool "
    "call + re-narration) — never the inflated whole-session footprint — "
    "then x the occurrence's own re-read tail: a failure's text is re-sent on "
    "every later call that still carries it, billed at the cache-read rate, so "
    "one occurrence is worth 1 + cache_read_ratio x tail input-token "
    "equivalents. The tail is truncated at the first compaction (a prompt-size "
    "collapse), not run to end-of-session, and the cluster takes the MEDIAN of "
    "its occurrences' multipliers rather than the mean — a handful of very long "
    "uncompacted sessions otherwise dominate. Reported NET of what the proposed "
    "fix costs to keep (a CLAUDE.md rule is re-sent on every future session, a "
    "hook is not); a cluster with no derived fix claims nothing at all. Review "
    "the example sessions before applying a fix"
)
HONESTY_CAVEAT = (
    "Structural failure-signature clustering, not a quality judgment. "
    "Review the example sessions and the proposed fix before applying it."
)

# --- Known, validated relearn families ----------------------------------------
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
            "PostToolUseFailure hook (Bash/Read): react only after a "
            "'no such file or directory' failure by injecting the real cwd + "
            "a short directory listing as additionalContext, so the agent "
            "recovers in one shot instead of a PreToolUse guess-and-block on "
            "every relative path (which would misfire on normal usage)."
        ),
    },
    {
        "key": "edit_before_read",
        "title": "Edit/Write before Read",
        "tools": {"Edit", "Write", "MultiEdit", "NotebookEdit"},
        "pattern": re.compile(r"has not been read yet", re.IGNORECASE),
        # Downgraded from rung 3 (Phase 2.5): the harness already errors
        # clearly on this ("has not been read yet") and the agent virtually
        # always self-corrects on the very next turn by reading the file —
        # there's no failure-recovery gap for a reactive hook to close. A
        # PreToolUse guard would need to track per-session read-state itself
        # (which files has THIS session read, reset per session/compaction) —
        # exactly the kind of fragile, easy-to-get-wrong state the harness
        # already maintains authoritatively. Duplicating it in a hook risks a
        # false block on a file the harness knows was read but our own
        # tracking missed (a session resume, a compaction, a subagent read).
        # Safer to note the pattern than to guess at its state.
        "rung": 1,
        "fix": (
            "CLAUDE.md/skill note: the harness already blocks an Edit/Write "
            "before a Read with a clear error ('has not been read yet') and "
            "agents reliably self-correct by reading next turn — no hook "
            "needed, this is advisory awareness only."
        ),
    },
    {
        "key": "sleep_chain",
        "title": "blocked sleep-chain",
        "tools": {"Bash"},
        # The block usually reads generically ("blocked"/"disallowed"/timed out)
        # with nothing relearn-specific in the wording — the ONE reliable tell is
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
            "PostToolUseFailure hook (Edit/Write/MultiEdit): react only after "
            "a 'modified since read' failure by injecting a re-Read reminder "
            "as additionalContext — never touches a successful edit."
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
        "fix": (
            "PostToolUseFailure hook (Edit/MultiEdit): react only after a "
            "string-not-found failure by injecting a re-Read reminder as "
            "additionalContext — never touches a successful edit."
        ),
    },
    {
        # MUST stay ordered before "deferred_tool_cold" below: that family's
        # pattern (`inputvalidationerror`, tools=None -> matches ANY tool)
        # also fires on the real wording of THIS family's evidence --
        # "InputValidationError: Read failed due to the following issue:\n
        # The parameter `offset` type is expected as `number` but provided
        # as `array`" matches both patterns. classify_known_family is
        # first-match-wins over declaration order, so the more-specific
        # family (Read-only, offset-specific) has to be checked first or its
        # evidence is silently absorbed by the generic one and mislabeled
        # with the wrong fix. Validated against the real corpus (2026-07-14):
        # with the old order, 100% of read_offset_malformed's evidence
        # (~35% of deferred_tool_cold's Read-tool occurrences) was shadowed
        # this way -- the family never once surfaced a proposal despite
        # matching real, recurring evidence.
        "key": "read_offset_malformed",
        "title": "Read malformed offset (array, not scalar)",
        "tools": {"Read"},
        "pattern": re.compile(r"offset.{0,20}(must be|invalid|expected)|invalid.{0,20}offset", re.IGNORECASE),
        "rung": 1,
        "fix": "CLAUDE.md/skill note: Read's `offset`/`limit` are scalars, not arrays.",
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
        # Downgraded from rung 5 (Phase 2.5, 2026-07-14): rung 5 promises a
        # "config/env fix", but there is no safe automatic config/env writer
        # in this codebase -- Apply used to render an inert stub hook for
        # this family (`_render_stub_hook`, never wired to block/inject
        # anything), advertising a fix that did nothing. A rung-1 CLAUDE.md
        # note is honest about what's actually deliverable and still useful.
        "key": "command_not_found",
        "title": "command not found (bashisms under zsh, bare interpreter)",
        "tools": {"Bash"},
        "pattern": re.compile(r"command not found", re.IGNORECASE),
        "rung": 1,
        "fix": (
            "CLAUDE.md/skill note: this shell doesn't have that binary/builtin on "
            "PATH. Common causes here: using bare `python` instead of `python3`, "
            "or a bash-only builtin (`mapfile`, `shopt`, `[[ ... ]]` extensions) "
            "that doesn't exist under this shell (e.g. zsh, sh) or POSIX mode. "
            "Prefer the portable/explicit form."
        ),
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

#: Ordered substitutions for the generic-signature normalizer. Order matters:
#: timestamps and uuids are masked before bare hex/number runs so their internal
#: digits aren't partially replaced first (see mask_variables).
_GENERIC_SUBS = [
    (_TIMESTAMP_RE, "<TS>"),
    (_UUID_RE, "<UUID>"),
    (_PATH_RE, "<PATH>"),
    (_HEX_ID_RE, "<ID>"),
    (_NUMBER_RE, "<N>"),
]


def _normalize_generic(text: str) -> str:
    """Strip paths/uuids/hex-ids/numbers/timestamps so unrelated values collapse
    into the same signature. Deterministic, no LLM — the fast path before the
    distill pass on whatever this misses."""
    return mask_variables(text, _GENERIC_SUBS, collapse_ws=True, lowercase=True)


def _generic_signature(tool_name: str, error_text: str) -> str:
    normalized = _normalize_generic(error_text)[:160]
    return f"{tool_name}:{normalized}"


#: Not relearns: the tool_result carries ``is_error`` because a HUMAN declined
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
    relearn — see ``_USER_DECLINE_RE``."""
    return bool(error_text) and bool(_USER_DECLINE_RE.search(error_text.strip()))


def classify_known_family(tool_name: str, error_text: str, label: str = "") -> str | None:
    """Return the matching known family key, or None. Tried in declaration order
    (first match wins; the families are mutually near-exclusive by wording).

    Most families match on the raw error text alone. A few (e.g. the blocked
    sleep-chain, whose block message is often generic — "timed out"/"blocked"
    with nothing relearn-specific in the wording) additionally require the
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
    *,
    transcript_cache_dir: Path | None = None,
) -> list[FailureEpisode]:
    """Every erroring tool call in one session (main thread + subagents).

    Returns ``[]`` when the session has no on-disk transcript (SDK session,
    pruned). Never raises — a malformed transcript yields whatever could be
    parsed (``build_session_story`` already tolerates bad lines).

    ``transcript_cache_dir`` is forwarded straight to ``build_session_story``'s
    ``cache_dir`` — see that function's docstring and ``core.transcript_cache``.
    Named distinctly from this module's own ``distill_cache_dir`` (a different,
    unrelated cache) to avoid the two being confused at a call site.
    """
    story = build_session_story(
        session_id, projects_root=projects_root, include_subagents=True,
        cache_dir=transcript_cache_dir,
    )
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
                continue  # a human's own choice, not a relearn — see is_user_decline
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


def _failure_signature(failure: FailureEpisode) -> tuple[str, str | None, str]:
    """``(signature, family_key, title)`` for one failure: a known family (sig ==
    family_key) or a generic normalized signature. The single classify point
    ``cluster_failures`` keys on."""
    family_key = classify_known_family(failure.tool_name, failure.error_text, failure.label)
    if family_key is not None:
        return family_key, family_key, _FAMILY_BY_KEY[family_key]["title"]
    sig = _generic_signature(failure.tool_name, failure.error_text)
    return sig, None, f"{failure.tool_name}: {failure.error_text[:60] or failure.label}"


def cluster_failures(failures: list[FailureEpisode]) -> dict[str, _RawCluster]:
    """Bucket failures by known family first, else a generic normalized signature.

    Groups via the shared ``group_by_key`` (order-preserving), then builds one
    ``_RawCluster`` per group with its title/family taken from the group's FIRST
    failure — the same failure that used to create the bucket inline, so titles
    stay byte-identical."""
    buckets = group_by_key(failures, lambda f: _failure_signature(f)[0])
    clusters: dict[str, _RawCluster] = {}
    for sig, group in buckets.items():
        _, family_key, title = _failure_signature(group[0])
        clusters[sig] = _RawCluster(
            signature=sig, family_key=family_key, title=title, failures=group,
        )
    return clusters


def _recurring(clusters: dict[str, _RawCluster], min_sessions: int) -> list[_RawCluster]:
    """Clusters seen across at least ``min_sessions`` DISTINCT sessions — the
    recurrence gate, on distinct-session count (not raw occurrences)."""
    kept = recurring(
        clusters, min_members=min_sessions, size_fn=lambda c: len(c.session_ids),
    )
    return list(kept.values())


# --- Distill pass over the residual (non-family) bucket -----------------------

def _distill_cache_dir() -> Path:
    return Path.home() / ".tj" / "distill_cache" / "relearn"


def _cluster_hash(cluster: _RawCluster) -> str:
    payload = cluster.signature + "|" + "|".join(
        sorted(f.error_text for f in cluster.failures[:10])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def distill_relearn_cluster(
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


# --- Distill confidence gate (SPEC honesty requirement) ------------------------
#
# Validated against the real corpus (2026-07-14): fed only bare/near-empty
# evidence, the distill model reliably CONFABULATES a specific-sounding but
# ungrounded fix rather than declining. The single biggest real example: a
# multi-command `&&` Bash chain whose LAST command exits nonzero with no
# error text of its own (a trailing `grep`/`find` "no match", commonly) —
# the captured "error" is either empty, a bare digit/punctuation residue
# (`0`, `000`, `---`), or literally just leftover STDOUT from an EARLIER,
# successful command in the chain (an `ls -la` dump) that has nothing to do
# with why the chain's exit code was nonzero. Distill invented FIVE different
# titled "fixes" for that one benign phenomenon (bash_stderr_missing,
# bash_error_reporting, bash_env_setup, bash_output_buffer_limit,
# bash_output_truncation) — each confident, each wrong, none traceable to any
# actual quoted error text. A fix can only be grounded in evidence that
# itself says something; this gate rejects a cluster BEFORE distillation when
# none of its samples clear that bar, rather than trusting the model to
# decline on its own.

_EXIT_CODE_PREFIX_RE = re.compile(r"^\s*exit code\s+\d+\s*\n?", re.IGNORECASE)
#: Body is "noise" if, after stripping a leading exit-code line, nothing but
#: digits/whitespace/punctuation is left (catches "", "0", "000", "---").
_ONLY_NOISE_RE = re.compile(r"^[\s\d\W]*$")
#: A body that's actually just a raw `ls -la`-style directory dump — real
#: text, but leftover stdout from an earlier chain step, not an error
#: description of why the LAST command in the chain failed.
_LS_LISTING_RE = re.compile(r"^\s*total\s+\d+\s*\n\s*[dlpscb\-][rwxst\-]{9}[@+.]?\s", re.IGNORECASE)


def _is_substantive_error_text(text: str) -> bool:
    """False for evidence too thin to ground a specific distilled fix in —
    see the confidence-gate note above. Never raises."""
    if not text:
        return False
    body = _EXIT_CODE_PREFIX_RE.sub("", text, count=1).strip()
    if not body:
        return False
    if _ONLY_NOISE_RE.match(body):
        return False
    if _LS_LISTING_RE.match(body):
        return False
    return True


def _evidence_too_thin_for_distill(cluster: _RawCluster, *, sample_cap: int = 8) -> bool:
    """True when NONE of a cluster's (capped) raw samples carry substantive
    error text — the distill confidence gate. A cluster failing this check is
    suppressed entirely rather than distilled (see ``apply_distill_to_residual``):
    showing a human a confident title + fix that traces to nothing but bare
    exit codes / leftover stdout is worse than surfacing nothing."""
    samples = [f.error_text for f in cluster.failures if f.error_text][:sample_cap]
    if not samples:
        return True
    return not any(_is_substantive_error_text(s) for s in samples)


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
    result = distill_relearn_cluster(tool_name, samples)
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
        if _evidence_too_thin_for_distill(cluster):
            continue  # confidence gate: suppressed, not distilled — see note above
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
#: real, still-uncodified relearn.
#: One DISTINCTIVE multi-word phrase per known family — deliberately full
#: phrases (not independent short words ANDed together). Validated the hard
#: way (2026-07-12): an earlier version used tuples like ("read", "before
#: editing") ANDed independently, and "read" alone is common enough that it
#: co-occurred with unrelated text in reachable docs, silently dropping the
#: single BIGGEST validated relearn (edit-before-read, 178 sessions) as
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
class RelearnExample:
    session_id: str
    repo:       str
    ts:         str | None
    snippet:    str            # short excerpt of the raw error (evidence)


@dataclass
class RelearnCluster:
    signature:                 str
    family_key:                str | None
    title:                     str
    sessions:                  int
    occurrences:                int
    repos:                      list[str]
    rung:                       int             # 1-5, SPEC §6 intervention ladder
    scope:                      str              # "project" | "user-global"
    proposed_fix:                str
    examples:                    list[RelearnExample] = field(default_factory=list)
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
    # ADVISE lane (workspace-less agents). True when every contributing repo is
    # an agent tokenjam has no workspace to write into (an SDK/OTel service, not
    # a checkout) — so there is no apply path at all: the card carries a
    # recommendation the user applies themselves, `suggested_target` stays "",
    # and Verify runs off spans instead of transcripts. See
    # `core/optimize/relearn_otel.py`.
    advise_only:                  bool = False
    # Monthly-basis fields (Review inbox stat tiles). The window-basis field
    # above (`estimated_recoverable_tokens`) is the raw full-corpus-scan
    # total, not a monthly rate — relearn scans unbounded history, unlike a
    # window-scoped cost analyzer, so there is no natural "the window IS a
    # month" shortcut. These extrapolate occurrences-per-day (over the run's
    # own observed timespan, `RelearnFinding.window_days`) to 30 days —
    # mirrors `monthly_savings_usd` in model_downgrade.py, a NEW explicitly-
    # named basis alongside the existing one (the Recoverable-savings
    # contract there is unchanged; Overview/Optimize keep reading the window
    # field). `estimated_monthly_usd` is populated only when a blended $/token
    # rate could be derived from the cluster's own sessions' spans (see
    # `_blended_dollar_rate`); `monthly_rate_basis` names the models that rate
    # came from so it's never a silent number.
    estimated_monthly_tokens:     int = 0
    estimated_monthly_usd:        float | None = None
    monthly_rate_basis:           str = ""
    #: The measured re-read tail behind the figures above: the MEDIAN number of
    #: later calls that still carried one of this cluster's occurrences before
    #: the context was compacted away, and the input-token-equivalent
    #: multiplier that follows from it (`1 + cache_read_ratio x tail`). Carried
    #: so a reader can see WHY an occurrence is worth more than its 1,500
    #: tokens; 0 / 1.0 when no tail could be measured.
    tail_calls_median:            int = 0
    tail_multiplier:              float = 1.0
    # Net-of-standing-cost accounting (`core/optimize/write_budget.py`). The
    # four `estimated_*` fields above are reported NET of what the proposed
    # artifact costs to KEEP: a rung-1 CLAUDE.md rule is re-sent on every
    # future session forever, so its block is priced against the same session
    # pace the saving is projected on and subtracted. The pre-net figures stay
    # here, inspectable, so the derivation is never hidden — but no surface may
    # claim a saving larger than the net one. Rung 3+ (hook / wrapper / config)
    # is never sent to the model as prompt text, so its standing cost is a
    # genuine zero and net == gross.
    gross_recoverable_tokens:         int = 0
    gross_monthly_tokens:             int = 0
    gross_monthly_usd:                float | None = None
    standing_cost_tokens_per_session: int = 0
    standing_cost_tokens:             int = 0
    standing_cost_basis:              str = ""
    #: gross / standing. Below 1.0 the rule costs more to keep than it saves.
    #: A ratio, not a break-even call count: unlike summarize's one-time
    #: rewrite cost, this cost recurs, so the session count cancels out.
    payback_ratio:                    float | None = None
    net_negative:                     bool = False
    # Whether a PERMANENT artifact is actually on offer for this cluster, and
    # why not when it isn't (placeholder fix, net-negative payback, budget
    # exhausted, or merged into the family's single block). A suppressed write
    # also sets `advise_only`, so the Review inbox's existing no-apply-path
    # lane renders it with this reason in place of the generic OTel one.
    write_offered:                    bool = True
    write_blocked_reason:             str = ""


@dataclass
class RelearnFinding:
    clusters:            list[RelearnCluster] = field(default_factory=list)
    sessions_scanned:     int = 0
    failures_examined:    int = 0
    distilled_clusters:   int = 0
    dropped_codified:     int = 0
    estimated_recoverable_tokens: int | None = None
    estimate_basis:        str = ESTIMATE_BASIS
    estimate_confidence:   str = "heuristic"
    caveat:                 str = HONESTY_CAVEAT
    # The effective recurrence bar this run applied (config-overridable, see
    # core.config.OptimizeConfig.min_recurring_sessions) — carried on the
    # finding so a renderer's empty-state message never hardcodes a number
    # that could be stale against the user's own config.
    min_sessions:           int = MIN_RECURRING_SESSIONS
    # The observed span (days, earliest to latest timestamped occurrence
    # across every failure this run examined) every cluster's monthly figure
    # was extrapolated from. ``None`` when nothing in the run carried a
    # parseable timestamp — callers then treat the scale as 1 (no
    # extrapolation) rather than inventing a window. See `_corpus_window_days`.
    window_days:             float | None = None
    # Sum of every cluster's `estimated_monthly_tokens` — the Review inbox
    # headline's token-basis total when it can't lead with dollars.
    estimated_monthly_tokens: int | None = None
    # Sum of every cluster's `estimated_monthly_usd`. ``None`` when no cluster
    # could be priced at all (no DB connection, no priced spans, no model with
    # pricing data) — a zero would read as "this waste is free", which it is
    # not (CLAUDE.md anti-pattern #22). Clusters with no derived fix contribute
    # nothing here by construction: they claim zero on every basis.
    estimated_monthly_usd: float | None = None


def _snippet(failure: FailureEpisode) -> str:
    text = failure.error_text or failure.label
    return text[:200]


def _scope_for(repos: set[str]) -> str:
    """§7: concentrated in one repo -> project; spread across many -> user-global."""
    return "project" if len(repos) <= 1 else "user-global"


def _parse_failure_ts(ts: str | None) -> Any:
    """Best-effort ISO-8601 parse of a failure's timestamp. Returns ``None``
    on anything unparseable — never raises; a bad/missing timestamp just
    doesn't contribute to the window span."""
    if not ts:
        return None
    from datetime import datetime as _dt

    try:
        return _dt.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _corpus_window_days(failures: list[FailureEpisode]) -> float | None:
    """Span, in days, between the earliest and latest timestamped occurrence
    across every failure this run examined — the shared basis every cluster's
    monthly extrapolation scales against.

    Relearn scans unbounded on-disk history (not a fixed window like a cost
    analyzer's `since`/`until`), so there is no ready-made "the window is a
    month" shortcut the way `model_downgrade.monthly_savings_usd` has. This
    derives an equivalent window from the data itself: how far back the
    observed occurrences actually span. Returns ``None`` (not a number) when
    fewer than two occurrences carry a parseable timestamp — the caller then
    applies a scale of 1 (no extrapolation) rather than inventing a window
    from missing data. Clamped to a 1-day floor so a same-day burst of
    occurrences doesn't get divided by a near-zero span into an absurd rate.
    """
    stamps = [t for t in (_parse_failure_ts(f.ts) for f in failures) if t is not None]
    if len(stamps) < 2:
        return None
    span_days = (max(stamps) - min(stamps)).total_seconds() / 86400.0
    return span_days if span_days >= 1.0 else 1.0


def _corpus_active_days(failures: list[FailureEpisode]) -> int:
    """Distinct calendar days on which this run observed any occurrence.

    ``D_active`` for the shared projection basis (``core/optimize/
    projection.py``). Relearn has no session table to count active days off,
    so it counts the days its own evidence actually landed on, which is the
    same quantity measured from the data relearn does hold. Zero when nothing
    carried a parseable timestamp; the basis then suppresses the projection
    rather than inventing one.
    """
    stamps = [t for t in (_parse_failure_ts(f.ts) for f in failures) if t is not None]
    return len({t.date() for t in stamps})


def _monthly_scale(window_days: float | None) -> float:
    """The occurrences-per-day -> per-30-days multiplier. 1.0 (no
    extrapolation) when the window is unknown or degenerate — never invent a
    multiplier from missing data (behavioral requirement #1)."""
    if not window_days or window_days <= 0:
        return 1.0
    return 30.0 / window_days


def _prompt_timelines(conn: Any, session_ids: set[str]) -> dict[str, list[tuple[Any, int]]]:
    """``session_id -> [(start_time, prompt_size), ...]`` in wall-clock order.

    ``prompt_size`` is ``input_tokens + cache_tokens``, the same per-turn
    quantity ``analyzers/context_resend.py`` measures repeat share on. It is
    what a compaction collapses, which is how the tail knows where to stop.
    Best-effort: an unavailable DB just means no timeline and no tail.
    """
    if conn is None or not session_ids:
        return {}
    ids = sorted(session_ids)
    placeholders = ", ".join(f"${i + 1}" for i in range(len(ids)))
    try:
        rows = conn.execute(
            f"SELECT session_id, start_time, "
            f"COALESCE(input_tokens, 0) + COALESCE(cache_tokens, 0) "
            f"FROM spans WHERE session_id IN ({placeholders}) "
            f"AND name = 'gen_ai.llm.call' AND start_time IS NOT NULL "
            f"ORDER BY session_id, start_time",
            ids,
        ).fetchall()
    except Exception:
        return {}
    out: dict[str, list[tuple[Any, int]]] = {}
    for session_id, start_time, prompt_size in rows:
        out.setdefault(str(session_id), []).append((start_time, int(prompt_size or 0)))
    return out


def _tail_calls(timeline: list[tuple[Any, int]], failure_ts: Any) -> int:
    """How many later calls still re-read a failure that landed at ``failure_ts``.

    Walks forward from the first call after the failure and stops at the first
    COMPACTION boundary — a turn whose prompt collapses to at most
    ``COMPACTION_PROMPT_DROP_RATIO`` of the previous turn's. Past that point the
    failure text is no longer in the window being re-sent, so counting those
    calls would claim a cost the user never paid.
    """
    if failure_ts is None or not timeline:
        return 0
    tail = 0
    previous_size: int | None = None
    for start_time, prompt_size in timeline:
        if start_time is None or start_time <= failure_ts:
            previous_size = prompt_size
            continue
        if (
            previous_size is not None
            and previous_size > 0
            and prompt_size <= previous_size * COMPACTION_PROMPT_DROP_RATIO
        ):
            break
        tail += 1
        previous_size = prompt_size
    return tail


def _tail_multiplier(
    failures: list[FailureEpisode],
    timelines: dict[str, list[tuple[Any, int]]],
    profile: RateProfile | None,
) -> tuple[float, int]:
    """``(multiplier, median_tail_calls)`` in input-token equivalents.

    The MEDIAN of the per-occurrence multipliers, not the mean: a corpus's call
    volume is dominated by long automated-harness sessions, whose uncompacted
    tails drag a mean far above what a typical occurrence actually costs.
    Returns ``(1.0, 0)`` — one occurrence, no tail — whenever the tail cannot be
    measured, which is the conservative direction (never invent a multiplier).
    """
    if profile is None or not timelines:
        return 1.0, 0
    tails = [
        _tail_calls(timelines.get(f.session_id, []), _parse_failure_ts(f.ts))
        for f in failures
    ]
    tails = [t for t in tails if t > 0]
    if not tails:
        return 1.0, 0
    import statistics

    median_tail = int(statistics.median(tails))
    return 1.0 + profile.cache_read_ratio * median_tail, median_tail


def build_proposals(
    clusters: list[_RawCluster],
    *,
    min_sessions: int = MIN_RECURRING_SESSIONS,
    doc_text: str = "",
    repo_cwd_map: dict[str, str] | None = None,
    advise_only_repos: set[str] | None = None,
    conn: Any | None = None,
    window_days: float | None = None,
    persona: str = "unknown",
    projection: Any | None = None,
    existing_agent_file_tokens: int | None = None,
    sessions_by_repo: dict[str, int] | None = None,
) -> tuple[list[RelearnCluster], int]:
    """Turn surviving raw clusters into ranked proposals. Returns
    ``(proposals, dropped_codified_count)``.

    ``projection`` (a ``core.optimize.projection.ProjectionBasis``) and
    ``existing_agent_file_tokens`` drive the write budget: how many permanent
    rules may be offered at all, and what each one costs to keep. Omitting
    ``projection`` leaves the netting inert (a zero session count charges a
    rule nothing) while the quality floor and the write count cap still apply,
    so a caller that only wants clustering is never silently given a budget it
    did not ask for.

    ``repo_cwd_map`` (repo label -> a representative cwd) is optional,
    best-effort enrichment used only to pre-fill the Apply stage's suggested
    target path (Phase 2) — clustering itself needs none of it.

    ``advise_only_repos`` names the agents tokenjam has NO workspace for (the
    OTel lane — see ``core/optimize/relearn_otel.py``). A cluster whose repos are
    all in that set is marked ``advise_only`` and gets NO suggested target: there
    is nothing to apply into, so the card must not imply an apply path exists.

    ``persona`` gates the rung-1/rung-2 CLAUDE.md/skill write exactly like
    ``cost_proposals._persona_gated_write_fields`` gates the script/reuse/
    resend cards it shares that same write surface with (``verbosity`` is NOT
    a peer here — it no longer routes through that helper and is
    unconditionally advise-only for every persona): only a
    ``"claude-code"``/``"mixed"`` window is offered the write. An
    ``"sdk"``/``"unknown"`` window (the conservative default) never gets one
    — nothing in an SDK-only service's request path reads a CLAUDE.md or a
    ``.claude/skills/`` note, so offering it there is a write that visibly
    succeeds and changes nothing. ``proposed_fix`` (the recommendation text)
    is unaffected either way — only the apply path is.
    """
    from tokenjam.core.optimize.relearn_apply import default_target_path, slugify
    from tokenjam.core.optimize.write_budget import REASON_PLACEHOLDER, is_placeholder_fix

    repo_cwd_map = repo_cwd_map or {}
    write_offered = persona in {"claude-code", "mixed"}
    proposals: list[RelearnCluster] = []
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
            RelearnExample(
                session_id=f.session_id, repo=f.repo, ts=f.ts, snippet=_snippet(f),
            )
            for f in sorted(cluster.failures, key=lambda f: f.ts or "", reverse=True)[:MAX_EXAMPLE_SESSIONS]
        ]

        scope = _scope_for(cluster.repos)
        # Workspace-less (OTel) clusters have nowhere to write: no cwd, no
        # target, and the card must not offer an apply path it can't honor.
        # `bool(advise_only_repos) and ...` defeats mypy's None-narrowing since
        # it's wrapped in a call; guarding on the name itself narrows it to
        # `set[str]` inside the genexpr while keeping identical truthiness.
        # `not write_offered` folds in the persona gate above: even a
        # workspace-having cluster gets no apply path for an sdk/unknown
        # window.
        advise_only = bool(
            advise_only_repos and all(r in advise_only_repos for r in repos)
        ) or not write_offered
        repo_cwd = "" if advise_only else (
            repo_cwd_map.get(repos[0], "") if len(repos) == 1 else ""
        )
        if advise_only:
            suggested_target = ""
        else:
            try:
                suggested_target = default_target_path(
                    rung, scope, repo_cwd, slugify(cluster.title),
                )
            except Exception:
                suggested_target = ""   # never let a bad path computation sink the proposal

        # A cluster with no derived fix claims NOTHING, on any basis. 1387 of
        # 2311 occurrences in a real corpus match no known family and fall back
        # to the generic "review examples" text — attaching a dollar figure to
        # those would put a number on a card whose fix the user cannot apply,
        # which is the "would this be a quiet lie in the user's favour?" test
        # (CLAUDE.md anti-pattern #22) failing outright. The write budget
        # already refuses to WRITE a placeholder; this is the same verdict
        # applied to the CLAIM, so it holds for every persona rather than only
        # for the windows that reach the write path at all.
        has_real_fix = not is_placeholder_fix(fix)

        # A placeholder cluster is priced at the bare per-occurrence cost with
        # no tail — the raw observation stays inspectable on the `gross_*`
        # fields, and the two round-trip queries the tail needs are skipped for
        # a cluster that will claim nothing anyway.
        profile = blended_rate_profile(conn, session_ids=sessions) if has_real_fix else None
        timelines = _prompt_timelines(conn, sessions) if profile is not None else {}
        multiplier, median_tail = _tail_multiplier(cluster.failures, timelines, profile)

        # Input-token EQUIVALENTS: the head token at the input rate plus each
        # re-read at the cache-read rate, expressed on the head's basis so the
        # token and dollar figures stay proportional. See the constants above.
        scale = _monthly_scale(window_days)
        gross_tokens = round(occurrences * GROUNDED_TOKENS_PER_OCCURRENCE * multiplier)
        gross_monthly_tokens = round(gross_tokens * scale)
        gross_monthly_usd = (
            round(gross_monthly_tokens * profile.input_rate_per_token, 6)
            if profile is not None else None
        )
        # The CLAIM, as distinct from the observation above.
        recoverable_tokens = gross_tokens if has_real_fix else 0
        monthly_tokens = gross_monthly_tokens if has_real_fix else 0
        monthly_usd = gross_monthly_usd if has_real_fix else None
        rate_basis = "" if profile is None else (
            f"{profile.basis}; x{multiplier:.2f} for a median re-read tail of "
            f"{median_tail} later call(s) per occurrence"
        )

        proposals.append(RelearnCluster(
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
            estimated_recoverable_tokens=recoverable_tokens,
            novel=True,
            repo_cwd=repo_cwd,
            suggested_target=suggested_target,
            advise_only=advise_only,
            estimated_monthly_tokens=monthly_tokens,
            estimated_monthly_usd=monthly_usd,
            monthly_rate_basis=rate_basis,
            tail_calls_median=median_tail,
            tail_multiplier=round(multiplier, 4),
            # Pre-net figures, kept inspectable. `_apply_write_budget` nets the
            # four `estimated_*` fields above in place, so nothing downstream
            # can read a gross figure by accident.
            gross_recoverable_tokens=gross_tokens,
            gross_monthly_tokens=gross_monthly_tokens,
            gross_monthly_usd=gross_monthly_usd,
            write_offered=has_real_fix,
            write_blocked_reason="" if has_real_fix else REASON_PLACEHOLDER,
        ))

    proposals.sort(key=lambda p: p.sessions, reverse=True)
    proposals = _apply_write_budget(
        proposals, projection=projection,
        existing_agent_file_tokens=existing_agent_file_tokens,
        sessions_by_repo=sessions_by_repo,
    )
    return proposals, dropped


def _write_exposure_sessions(
    proposal: RelearnCluster, sessions_by_repo: dict[str, int] | None, total: int,
) -> int | None:
    """How many of the run's sessions would actually re-send this cluster's
    artifact.

    A ``user-global`` rule lands in ``~/.claude/CLAUDE.md`` and is paid on
    every session. A ``project`` rule lands in one repo's file and is paid only
    on that repo's sessions. Charging a project rule against the whole corpus
    would net a cluster-scoped saving against a corpus-scoped cost, which is
    the same time-basis mistake this accounting exists to remove, just wearing
    a different hat. ``None`` (no per-repo counts available) falls back to the
    projection basis's own session count.
    """
    if proposal.scope != "project" or sessions_by_repo is None:
        return None
    scoped = sum(sessions_by_repo.get(repo, 0) for repo in proposal.repos)
    # A repo we have no count for must not silently price the rule at zero.
    return min(scoped, total) if scoped > 0 else None


def _apply_write_budget(
    proposals: list[RelearnCluster],
    *,
    projection: Any | None,
    existing_agent_file_tokens: int | None,
    sessions_by_repo: dict[str, int] | None = None,
) -> list[RelearnCluster]:
    """Net every proposal's saving against what its fix costs to KEEP, and cap
    how many permanent rules are offered at all.

    Three things happen here and nowhere else:

    * A cluster whose fix is the generic "Review examples" placeholder never
      becomes a permanent rule, and claims nothing. There is no fix to claim.
    * Same-family clusters collapse onto ONE block. They share a single fix
      template, so N clusters used to mean N identical CLAUDE.md blocks; now
      the family's largest cluster carries the write and its siblings say so.
    * What survives is ranked by net value and offered until the budget runs
      out. Anything past that is deferred, not deleted: its recommendation is
      still on the card, so its net claim stands.

    A cluster with no apply path at all (the workspace-less OTel lane) is
    skipped entirely: nothing is written for it, so it has no standing cost and
    its figures pass through untouched.
    """
    from dataclasses import asdict, replace

    from tokenjam.core.optimize import write_budget as wb
    from tokenjam.core.optimize.projection import build_projection_basis
    from tokenjam.core.optimize.relearn_apply import artifact_for_rung, slugify

    basis = projection or build_projection_basis(0.0, 0, 0)
    candidates: list[wb.WriteCandidate] = []
    for p in proposals:
        if p.advise_only or not p.suggested_target:
            continue
        try:
            artifact = artifact_for_rung(asdict(p), p.signature, p.rung, slugify(p.title))
        except Exception:
            artifact = p.proposed_fix     # never let a render hiccup sink a proposal
        candidates.append(wb.WriteCandidate(
            key=p.signature,
            # Family-unmatched clusters have no family_key; keying them on
            # their own signature keeps each a family of one rather than
            # collapsing every unrelated residual into a single bucket.
            family=p.family_key or f"signature:{p.signature}",
            rung=p.rung,
            artifact_text=artifact or p.proposed_fix,
            gross_tokens=p.gross_recoverable_tokens,
            exposure_sessions=_write_exposure_sessions(
                p, sessions_by_repo, basis.sessions,
            ),
        ))

    budget = wb.build_write_budget(
        lane_budget_tokens=wb.RELEARN_WRITE_BUDGET_TOKENS,
        lane_max_writes=wb.RELEARN_MAX_OFFERED_WRITES,
        existing_agent_file_tokens=existing_agent_file_tokens,
    )
    decisions = wb.allocate_writes(candidates, budget, basis)

    out: list[RelearnCluster] = []
    for p in proposals:
        decision = decisions.get(p.signature)
        if decision is None:
            # Never entered the budget at all — advise-only, or no target to
            # write into. Nothing is on offer, so say so rather than letting
            # the dataclass default (`write_offered=True`) stand and claim a
            # write that does not exist.
            out.append(replace(p, write_offered=False))
            continue
        # The monthly field lives on the 30-day basis, so it is netted against
        # the PROJECTED session count while the window field is netted against
        # the observed one. Both use the same per-session standing cost, which
        # is what keeps the two bases from drifting (the exact error a
        # window-scoped saving minus a 30-day cost would introduce).
        monthly_standing = round(
            decision.standing_tokens_per_session * decision.exposure_sessions * basis.ratio
        )
        net_monthly_tokens = (
            0 if decision.claim_suppressed
            else max(p.gross_monthly_tokens - monthly_standing, 0)
        )
        rate = (
            p.gross_monthly_usd / p.gross_monthly_tokens
            if p.gross_monthly_usd is not None and p.gross_monthly_tokens > 0
            else None
        )
        net_monthly_usd = (
            round(net_monthly_tokens * rate, 6) if rate is not None
            else (None if p.gross_monthly_usd is None else 0.0)
        )
        out.append(replace(
            p,
            estimated_recoverable_tokens=decision.claimed_tokens,
            estimated_monthly_tokens=net_monthly_tokens,
            estimated_monthly_usd=net_monthly_usd,
            standing_cost_tokens_per_session=decision.standing_tokens_per_session,
            standing_cost_tokens=decision.standing_tokens,
            standing_cost_basis=decision.basis,
            payback_ratio=decision.payback_ratio,
            net_negative=decision.net_negative,
            write_offered=decision.offered,
            write_blocked_reason=decision.reason,
            # A suppressed write has no apply path, which is exactly what
            # `advise_only` already means to every surface. Reusing that flag
            # (rather than teaching each renderer a second one) makes the
            # Review inbox show this decision's own reason in place of the
            # generic workspace-less one.
            advise_only=p.advise_only or not decision.offered,
            suggested_target=p.suggested_target if decision.offered else "",
        ))
    return out


# --- Orchestration (pure, no ctx dependency — testable directly) --------------

def analyze_relearns(
    sessions: list[tuple[str, str]],     # [(session_id, repo), ...]
    *,
    projects_root: Path | str | None = None,
    min_sessions: int = MIN_RECURRING_SESSIONS,
    distill_enabled: bool = True,
    distill_cache_dir: Path | None = None,
    transcript_cache_dir: Path | None = None,
    codified_doc_text: str = "",
    repo_cwd_map: dict[str, str] | None = None,
    extra_failures: list[FailureEpisode] | None = None,
    advise_only_repos: set[str] | None = None,
    conn: Any | None = None,
    persona: str = "unknown",
    existing_agent_file_tokens: int | None = None,
) -> RelearnFinding:
    """Full pipeline over an explicit session list — the pure core the
    registry entry point and the on-disk cache job both call. Never raises.

    ``extra_failures`` are episodes extracted somewhere other than an on-disk
    transcript — today the OTel lane's failing spans (see
    ``core/optimize/relearn_otel.py``). They join the SAME clustering pass, so a
    signature that recurs across both lanes clusters as one relearn.
    ``advise_only_repos`` is forwarded to ``build_proposals`` to mark the
    workspace-less clusters. ``transcript_cache_dir`` (distinct from
    ``distill_cache_dir``, the LLM-distill cache) is forwarded to every
    per-session transcript parse — see ``core.transcript_cache``.
    ``conn`` (optional DuckDB connection) is forwarded to ``build_proposals``
    for the per-cluster blended-dollar-rate lookup (Review inbox monthly-$
    basis) — ``None`` keeps every cluster tokens-only, same as today.
    ``persona`` is forwarded to ``build_proposals`` to gate the rung-1/rung-2
    write — see its docstring.
    """
    all_failures: list[FailureEpisode] = []
    scanned = 0
    for session_id, repo in sessions:
        try:
            failures = extract_failures_for_session(
                session_id, repo, projects_root, transcript_cache_dir=transcript_cache_dir,
            )
        except Exception:
            continue
        scanned += 1
        all_failures.extend(failures)

    if extra_failures:
        all_failures.extend(extra_failures)
        # Span-sourced sessions are real scanned exposure too; counting them
        # keeps sessions_scanned honest against the recurrence denominator.
        scanned += len({f.session_id for f in extra_failures})

    raw_clusters = cluster_failures(all_failures)
    recurring = _recurring(raw_clusters, min_sessions)
    distilled = apply_distill_to_residual(
        recurring, cache_dir=distill_cache_dir, enabled=distill_enabled,
    )
    distilled_count = sum(1 for c in distilled if (c.family_key or "").startswith("distilled:"))

    # The shared monthly-extrapolation basis (behavioral requirement #1): the
    # observed span across every failure this run examined, not a fixed
    # window — relearn scans unbounded history. See `_corpus_window_days`.
    window_days = _corpus_window_days(all_failures)

    # The SAME basis the monthly extrapolation above uses, expressed once as a
    # ProjectionBasis so the write budget can price a permanent rule against
    # the identical session pace the saving is projected on. Mixing the two
    # would reintroduce exactly the time-basis error this accounting exists to
    # remove. See `core/optimize/projection.py`.
    projection = build_projection_basis(
        window_days or 0.0, _corpus_active_days(all_failures), scanned,
    )

    # Per-repo session counts: what a PROJECT-scoped rule's standing cost is
    # actually charged against (a user-global one is charged against them all).
    sessions_by_repo: dict[str, int] = {}
    for _session_id, repo in sessions:
        sessions_by_repo[repo] = sessions_by_repo.get(repo, 0) + 1

    proposals, dropped = build_proposals(
        distilled, min_sessions=min_sessions, doc_text=codified_doc_text,
        repo_cwd_map=repo_cwd_map, advise_only_repos=advise_only_repos,
        conn=conn, window_days=window_days, persona=persona,
        projection=projection,
        existing_agent_file_tokens=existing_agent_file_tokens,
        sessions_by_repo=sessions_by_repo,
    )
    total_tokens = sum(p.estimated_recoverable_tokens for p in proposals)
    total_monthly_tokens = sum(p.estimated_monthly_tokens for p in proposals) if proposals else None
    priced = [p.estimated_monthly_usd for p in proposals if p.estimated_monthly_usd is not None]
    total_monthly_usd = round(sum(priced), 6) if priced else None

    return RelearnFinding(
        clusters=proposals,
        sessions_scanned=scanned,
        failures_examined=len(all_failures),
        distilled_clusters=distilled_count,
        dropped_codified=dropped,
        estimated_recoverable_tokens=total_tokens if proposals else None,
        min_sessions=min_sessions,
        window_days=window_days,
        estimated_monthly_tokens=total_monthly_tokens,
        estimated_monthly_usd=total_monthly_usd,
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


def _repo_cwd_map_for(
    sessions: list[tuple[str, str]],
    projects_root: Path,
    *,
    transcript_cache_dir: Path | None = None,
) -> dict[str, str]:
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
        for record in read_records(path, cache_dir=transcript_cache_dir)[:5]:
            cwd = record.get("cwd")
            if isinstance(cwd, str) and cwd:
                out[repo] = cwd
                break
    return out


def compute_relearn_finding(
    conn: Any | None = None,
    since: Any | None = None,
    *,
    projects_root: Path | str | None = None,
    distill_enabled: bool = True,
    min_sessions: int = MIN_RECURRING_SESSIONS,
    transcript_cache_dir: Path | None = None,
    persona: str = "unknown",
    existing_agent_file_tokens: int | None = None,
) -> RelearnFinding:
    """Standalone entry point that doesn't need a full ``AnalyzerContext`` —
    used by the serve-time background cache job (``api/routes/relearn.py``)
    and by tests. ``conn`` is an OPTIONAL DuckDB connection used only for
    repo-name enrichment (``sessions.agent_id``); pass ``None`` and every
    session falls back to a ``"unknown"`` repo label — clustering itself needs
    no DB at all, it's a pure filesystem scan.

    ``persona`` (default ``"unknown"``, the conservative no-write default —
    see ``build_proposals``) is forwarded to gate the rung-1/rung-2 write.
    ``run(ctx)`` below passes the report's own ``ctx.persona`` rather than
    re-deriving it here, so a report never carries two different persona
    classifications for the same window.

    Full-corpus by design: enumerates every on-disk Claude Code transcript
    (not window-scoped like the other analyzers — a relearn recurring across
    months of history is exactly the signal this detector exists to find),
    optionally pre-filtered to ``since`` when the caller wants an incremental
    scan. Heavy (tens of seconds over a full local corpus) — callers that
    serve this over HTTP MUST cache the result, not compute it per-request.

    TWO LANES. On-disk transcripts cover the workspace agents; when ``conn`` is
    given, failing spans from NON-coding agents are folded into the same
    clustering pass so SDK/OTel services are no longer invisible to the
    detector (``core/optimize/relearn_otel.py``). Their clusters come back
    ``advise_only``: detect and advise, never apply.

    ``transcript_cache_dir``, when given, transparently caches each session's
    parsed transcript on disk (``core.transcript_cache``) so a re-run over an
    unchanged corpus skips re-parsing every session it already has a fresh
    cache entry for. ``None`` (the default) preserves this function's
    original always-reparse behavior — only the registered ``run(ctx)`` entry
    point and the serve-time background job opt in.
    """
    root = resolve_projects_root(projects_root)
    repo_map = _repo_map_from_db(conn) if conn is not None else {}

    # The OTel lane. Best-effort: a failure here must never sink the (already
    # working) transcript scan.
    span_failures: list[FailureEpisode] = []
    advise_only_repos: set[str] = set()
    if conn is not None:
        try:
            from tokenjam.core.optimize.relearn_otel import (
                extract_span_failures,
                non_coding_agent_ids,
            )

            span_failures = extract_span_failures(conn, since)
            advise_only_repos = non_coding_agent_ids(conn)
        except Exception:
            span_failures = []
            advise_only_repos = set()

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
        repo_cwd_map = _repo_cwd_map_for(
            sessions, root, transcript_cache_dir=transcript_cache_dir,
        )
        doc_text = _doc_text(_candidate_doc_paths(set(repo_cwd_map.values())))
    except Exception:
        doc_text = ""

    return analyze_relearns(
        sessions, projects_root=root, codified_doc_text=doc_text,
        distill_enabled=distill_enabled, repo_cwd_map=repo_cwd_map,
        extra_failures=span_failures, advise_only_repos=advise_only_repos,
        min_sessions=min_sessions, transcript_cache_dir=transcript_cache_dir,
        conn=conn, persona=persona,
        existing_agent_file_tokens=existing_agent_file_tokens,
    )


@register("relearn")
def run(ctx: AnalyzerContext) -> None:
    """Registry entry point. Attaches a ``RelearnFinding`` to
    ``ctx.report.findings["relearn"]`` — see ``compute_relearn_finding`` for
    the full-corpus behaviour and performance note.

    Passes the resolved persistent parse cache dir (``core.transcript_cache.
    default_cache_dir``) so a re-run over an unchanged corpus skips
    re-parsing every session it already has a fresh cache entry for.
    """
    from tokenjam.core.transcript_cache import default_cache_dir

    optimize_cfg = getattr(ctx.config, "optimize", None)
    min_sessions = getattr(
        optimize_cfg, "min_recurring_sessions", MIN_RECURRING_SESSIONS,
    )
    # The write budget's headroom comes from the `summarize` analyzer's own
    # measurement of the agent files these proposals would append to. It runs
    # ahead of relearn in ANALYZER_ORDER, so its finding is already on the
    # report; when it wasn't selected this is None and the lane cap stands
    # alone. This is the cross-reference the two halves of the loop were
    # missing: relearn can no longer offer rules for a file the same report
    # is recommending the user compress.
    from tokenjam.core.optimize.write_budget import measured_agent_file_tokens

    ctx.report.findings["relearn"] = compute_relearn_finding(
        ctx.conn, ctx.since, min_sessions=min_sessions,
        transcript_cache_dir=default_cache_dir(ctx.config),
        persona=ctx.persona,
        existing_agent_file_tokens=measured_agent_file_tokens(
            ctx.report.findings.get("summarize"),
        ),
    )
