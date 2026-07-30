"""
Cross-session relearn aggregator (self-improve loop, Phase 1: detect + surface).

A "relearn" is a blocker a Claude Code agent silently re-hits across many
unwatched sessions — a wrong-cwd Read, an Edit before a Read, a blocked
sleep-chain, a stale-read race, a domain-blocked WebFetch, and so on. Writing
a durable fix (a CLAUDE.md rule, a hook) into the project only codifies what
a human happened to notice; this module catches what nobody watched.

Pipeline (validated 2026-07-12 against the full local corpus):

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
     dropped — the already-documented check.
  5. PROPOSE  — surviving, recurring (>=3 distinct sessions) clusters get a
     conservative token estimate (occurrences x grounded per-turn cost, never
     the inflated afflicted-session footprint), a delivery mechanism (see
     ``core/rulewrite/delivery``,
     intervention ladder) and a scope (project vs user-global, by how many
     distinct repos the cluster's sessions span).

THE HORIZON IS TOKENJAM'S ARCHIVE, NOT CLAUDE CODE'S ROTATION. Step 1 above
reads on-disk transcripts, which Claude Code rotates on its own schedule
(``cleanupPeriodDays``), so a transcript-only detector can never accumulate the
long-horizon recurrence that is this module's entire premise. Sessions whose
transcript has been rotated away are recovered from tokenjam's retained spans
instead — see ``compute_relearn_finding``'s THREE LANES note. The two lanes are
disjoint by session id, so no failure is extracted twice.

ONE KIND OF NUMBER. A relearn cluster reports only ``past_overspend_*`` — a
BACKWARD observation of what the recurrence already cost, for every cluster
including the ones no fix template matches and the ones whose rule is
uneconomic. There is no forward "you could recover $X" claim anywhere on the
cluster or the finding: a relearn card shows its past figure only, same as
every other analyzer's card, per the repo `CLAUDE.md`'s per-analyzer
dollar-field contract. "We have no action for this" is not "this was
unavoidable", and no future maintenance cost is ever netted out of money
already spent. The net-of-standing-cost arithmetic in
``core/optimize/write_budget.py`` still runs — it decides whether a
PERMANENT artifact is worth OFFERING at all (``write_offered`` /
``advise_only`` / ``write_blocked_reason`` / ``payback_ratio``) — but its
output is never rendered as a savings figure.

Never raises: a single unreadable transcript, a distill failure, or a missing
CLAUDE.md is skipped, not fatal — this runs unattended on a schedule.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from tokenjam.core.analysis_span import retention_days_for
from tokenjam.core import distill as distill_mod
from tokenjam.core.method_spine import build_method_spine
from tokenjam.core.optimize.clustering import group_by_key, mask_variables, recurring
from tokenjam.core.optimize.projection import build_projection_basis
from tokenjam.core.optimize.analyzers.resend_tail import RELEARN_RESEND_BOUNDARY
from tokenjam.core.optimize.rate_profile import RateProfile, blended_rate_profile
from tokenjam.core import fixes as _fixes
from tokenjam.core.optimize.registry import register
from tokenjam.core.optimize.relearn_window import (
    RELEARN_WINDOW_LABELS,
    WINDOWED_BASIS,
    RelearnWindowedObservation,
    RelearnWindowTotal,
    sum_windowed,
    window_days,
    window_labels_including,
)
from tokenjam.core.optimize.types import AnalyzerContext
from tokenjam.core.rulewrite.kinds import (
    DELIVERY_CLAUDE_MD_RULE,
    DELIVERY_EXECUTING_HOOK,
    DELIVERY_INJECTING_HOOK,
    DELIVERY_SKILL,
)
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
#: signal, never a causal claim; surfaced with ``past_overspend_basis`` below.
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

#: The basis behind relearn's one figure — a backward, ungated OBSERVATION,
#: never a forward claim. Quotes `RELEARN_RESEND_BOUNDARY` verbatim (the same
#: constant `resend`'s own basis quotes) so the two analyzers' cards cannot
#: drift into two different accounts of where one prices a call's existence
#: and the other prices a call's size — see
#: `test_the_boundary_is_stated_once_and_quoted_by_both_analyzers`.
PAST_OVERSPEND_BASIS = (
    "each observed failure's MEASURED recovery arc (the assistant turns between "
    "hitting the pothole and the same tool succeeding again, median 2 on a real "
    "corpus rather than the 1 a flat charge assumes, with turns shared by "
    "overlapping failures split between them) x the MEASURED cost of one "
    "assistant turn in the sessions the cluster occurred in — "
    + RELEARN_RESEND_BOUNDARY + " — PLUS the error "
    "text's own measured "
    "re-read tail, priced at the rate the contributing sessions actually "
    "billed at. "
    "Accumulated over the scanned corpus and NEVER paced, projected or "
    "extrapolated to a month. A companion figure bounded to a trailing window "
    "may sit beside this one (past_overspend_windows); that is the same "
    "occurrences FILTERED by date and capped at this figure, so it is always "
    "the smaller of the two and is never this figure rescaled. Deliberately "
    "ungated: a cluster with no fix template in our library and a cluster "
    "whose rule is uneconomic to keep both still cost this, and no future "
    "maintenance cost is netted out of it. The re-read share is broken out "
    "as past_reread_* — a COMPONENT of this figure, not an addend — because "
    "that share is re-sent context, which the context re-send analyzer prices "
    "in full; the two overlap there and must never be added together"
)

#: Why the recurrence gate's residue is counted but never claimed.
BELOW_THRESHOLD_BASIS = (
    "failures observed but seen in fewer than the recurrence threshold's "
    "distinct sessions, so they never became a cluster. Priced on the same "
    "per-occurrence basis and reported so that 'no action for this' is never "
    "shown as 'this was free'. Not claimable and not rolled up: a one-off "
    "failure is not a relearn"
)

HONESTY_CAVEAT = (
    "Structural failure-signature clustering, not a quality judgment. "
    "Review the example sessions and the proposed fix before applying it."
)

# --- Known, validated relearn families ----------------------------------------
# Each entry: (family key, human title, tool-name filter (None = any),
# regex over the raw error text, DELIVERY MECHANISM, default proposed fix).
#
# The delivery is declared per family, in words, because it is the family that
# knows: `sleep_chain` blocks a command and injects nothing, while the three
# PostToolUseFailure families exist precisely to inject text. Those two cost
# opposite amounts, and no property of the artifact tells them apart — only the
# family's own matcher does. See `core/rulewrite/delivery`.
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
        "delivery": DELIVERY_INJECTING_HOOK,
        "fix": _fixes.fix_text("relearn.cwd_confusion"),
    },
    {
        # BY FAR the largest family on a real coding corpus (measured
        # 2026-07-26: 916 distinct sessions, 964 occurrences — more sessions
        # than every other family combined), and until it was named here it
        # fell into the generic bucket, got the "Review examples" placeholder,
        # and therefore claimed exactly $0 despite being the single most
        # recurrent blocker in the corpus.
        #
        # It is also the family that sits closest to `context_resend`, so the
        # boundary is worth stating at the definition: `resend` prices the
        # re-sent context inside calls that HAPPENED; this prices the call that
        # got REJECTED outright — an API-level 400 that returned no completion
        # at all, forcing the session to compact and re-issue. That rejected
        # call is a call that should not have happened, which is this
        # analyzer's whole population (see THE LINE BETWEEN THE TWO ANALYZERS
        # in `build_proposals`).
        "key": "context_overflow",
        "title": "context window overflowed (prompt rejected)",
        "tools": None,
        "pattern": re.compile(
            r"prompt is too long|"
            r"input length and .max_tokens. exceed context limit|"
            r"exceeds? the (?:maximum )?context (?:window|length|limit)",
            re.IGNORECASE,
        ),
        "delivery": DELIVERY_CLAUDE_MD_RULE,
        # Lead-in names what THIS family observed; the durable instruction is
        # the shared catalog record, not a third wording of it (see that
        # record's note on why three copies is worse than one). BOTH halves are
        # catalogued — the lead-in lives on the record as this analyzer's
        # framing, because a sentence written here is prose the lint cannot see,
        # and prose the lint cannot see is how one instruction comes to be
        # stated twice inside one written block.
        "fix": _fixes.fix_text_for("resend.offload_to_subagent", "relearn"),
    },
    {
        "key": "edit_before_read",
        "title": "Edit/Write before Read",
        "tools": {"Edit", "Write", "MultiEdit", "NotebookEdit"},
        "pattern": re.compile(r"has not been read yet", re.IGNORECASE),
        # Downgraded from a hook (Phase 2.5): the harness already errors
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
        "delivery": DELIVERY_CLAUDE_MD_RULE,
        # ADVISORY ONLY, and the flag is what makes that mechanical. This
        # family's own fix text says there is nothing to do — the harness
        # already errors clearly and agents self-correct next turn — so the
        # card must not occupy an apply slot offering it. The recurrence still
        # COST something and that figure stands untouched (Critical Rule 32):
        # a gate on whether we have an action available never reaches back and
        # edits what a behaviour already cost.
        "advisory_only": True,
        "fix": _fixes.fix_text("relearn.edit_before_read"),
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
        "delivery": DELIVERY_EXECUTING_HOOK,
        "fix": _fixes.fix_text("relearn.sleep_chain"),
    },
    {
        "key": "stale_read_race",
        "title": "file modified since read (linter/hook race)",
        "tools": {"Edit", "Write", "MultiEdit"},
        "pattern": re.compile(r"modified since (it was last read|read)", re.IGNORECASE),
        "delivery": DELIVERY_INJECTING_HOOK,
        "fix": _fixes.fix_text("relearn.reread_before_retrying_edit"),
    },
    {
        "key": "edit_string_not_found",
        "title": "Edit string-not-found (stale/whitespace/conflict)",
        "tools": {"Edit", "MultiEdit"},
        "pattern": re.compile(
            r"string to replace not found|old_string not found|not found in file",
            re.IGNORECASE,
        ),
        "delivery": DELIVERY_INJECTING_HOOK,
        "fix": _fixes.fix_text("relearn.reread_before_retrying_edit"),
    },
    {
        # MUST stay ordered before "edit_string_not_found" above would have
        # been a hazard had that family's pattern been any looser: this is the
        # OPPOSITE failure (too many matches, not zero) and takes the opposite
        # fix, so the two must never share a bucket.
        "key": "edit_ambiguous_match",
        "title": "Edit matched multiple times (replace_all not set)",
        "tools": {"Edit", "MultiEdit"},
        "pattern": re.compile(
            r"found \d+ matches of the string to replace|"
            r"replace_all is false",
            re.IGNORECASE,
        ),
        "delivery": DELIVERY_CLAUDE_MD_RULE,
        "fix": _fixes.fix_text("relearn.edit_ambiguous_match"),
    },
    {
        "key": "read_too_large",
        "title": "Read exceeded the max-tokens ceiling",
        "tools": {"Read"},
        "pattern": re.compile(
            r"exceeds maximum allowed tokens|"
            r"file content \(\d+ tokens\) exceeds",
            re.IGNORECASE,
        ),
        "delivery": DELIVERY_CLAUDE_MD_RULE,
        "fix": _fixes.fix_text("relearn.read_too_large"),
    },
    {
        "key": "read_directory",
        "title": "Read pointed at a directory, not a file",
        "tools": {"Read"},
        "pattern": re.compile(
            r"eisdir|illegal operation on a directory", re.IGNORECASE,
        ),
        "delivery": DELIVERY_CLAUDE_MD_RULE,
        "fix": _fixes.fix_text("relearn.read_directory"),
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
        "delivery": DELIVERY_CLAUDE_MD_RULE,
        "fix": _fixes.fix_text("relearn.read_offset_malformed"),
    },
    {
        "key": "deferred_tool_cold",
        "title": "deferred tool called cold (no ToolSearch first)",
        "tools": None,
        "pattern": re.compile(
            r"inputvalidationerror|the following issues|"
            r"required parameter.{0,20}is missing|"
            # Same root cause, different harness wording: the tool exists but
            # was never brought into the context, so the call cannot resolve.
            r"no such tool available|is not enabled in this context",
            re.IGNORECASE,
        ),
        "delivery": DELIVERY_SKILL,
        "fix": _fixes.fix_text("relearn.deferred_tool_cold"),
    },
    {
        # Downgraded from a config/env fix (Phase 2.5, 2026-07-14): there is
        # no safe automatic config/env writer in this codebase -- Apply used to
        # render an inert stub hook for this family (`_render_stub_hook`, never
        # wired to block/inject anything), advertising a fix that did nothing.
        # A CLAUDE.md rule is honest about what's actually deliverable and
        # still useful.
        "key": "command_not_found",
        "title": "command not found (bashisms under zsh, bare interpreter)",
        "tools": {"Bash"},
        # The second alternative catches the shell's OTHER phrasing for the
        # same fault — `uv not found` / `pnpm not found`, emitted by a wrapper
        # or a version manager rather than by the shell's own `command not
        # found` handler. Measured 2026-07-26: 39 sessions across two generic
        # clusters that never reached this family. Anchored to a line start
        # and a bare word so it cannot swallow prose like "string to replace
        # not found" (a different family, and Edit-only anyway).
        "pattern": re.compile(
            r"command not found|^\s*[\w.\-/]+:? not found\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        "delivery": DELIVERY_CLAUDE_MD_RULE,
        "fix": _fixes.fix_text("relearn.command_not_found"),
    },
    {
        "key": "bash_timeout",
        "title": "Bash command timed out (blocking wait in the foreground)",
        "tools": {"Bash"},
        # Ordered AFTER sleep_chain deliberately: a `sleep N && check` chain
        # that times out is the sleep-chain pothole and keeps that family's
        # more specific fix. What lands here is the general case — a build, a
        # test run, a dev server — held in the foreground until the harness
        # killed it (exit 143 is SIGTERM).
        "pattern": re.compile(
            r"command timed out after|"
            r"exit code 143\b.{0,80}tim(?:ed )?out",
            re.IGNORECASE | re.DOTALL,
        ),
        "delivery": DELIVERY_CLAUDE_MD_RULE,
        "fix": _fixes.fix_text("relearn.bash_timeout"),
    },
    {
        "key": "bash_chained_approval",
        "title": "chained Bash command tripped the approval prompt",
        "tools": {"Bash"},
        # Distinct from the bare "requires approval" case, which is the user's
        # own allowlist and is filtered out as a non-relearn at extraction
        # (see `_USER_DECLINE_RE`). THIS one is agent-avoidable: the chaining
        # is what forced the prompt, and un-chaining removes it.
        "pattern": re.compile(
            r"bash command contains multiple operations", re.IGNORECASE,
        ),
        "delivery": DELIVERY_CLAUDE_MD_RULE,
        "fix": _fixes.fix_text("relearn.bash_chained_approval"),
    },
    {
        "key": "git_branch_exists",
        "title": "git branch already exists",
        "tools": {"Bash"},
        "pattern": re.compile(
            r"a branch named .+ already exists|"
            r"already exists and is not a valid branch name",
            re.IGNORECASE,
        ),
        "delivery": DELIVERY_CLAUDE_MD_RULE,
        "fix": _fixes.fix_text("relearn.git_branch_exists"),
    },
    {
        "key": "webfetch_domain_blocked",
        "title": "WebFetch domain-blocked",
        # NOT WebFetch-only: the same block surfaces on the model call itself
        # ("The following domains are not accessible to our user agent"), under
        # the `gen_ai.llm.call` name, when the fetch is attempted server-side.
        # Gating on the tool name hid that variant in the generic bucket; the
        # pattern is specific enough to stand without the tool filter.
        "tools": None,
        # Real wording (validated against the local corpus): "Claude Code is
        # unable to fetch from <domain>" — not "not allowed"/"blocked" as the
        # phrasing might suggest.
        "pattern": re.compile(
            r"unable to fetch from|domain.{0,30}(not allowed|block)|"
            r"not allowed to fetch|blocked domain|"
            r"following domains are not accessible",
            re.IGNORECASE,
        ),
        "delivery": DELIVERY_CLAUDE_MD_RULE,
        "fix": _fixes.fix_text("relearn.webfetch_domain_blocked"),
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
    r"the user (?:rejected|canceled|cancelled|interrupted)|"
    r"tool use was rejected|"
    # A bare permission prompt is the user's own allowlist configuration, not
    # a pothole the agent can route around: the SAME command succeeds once the
    # user approves it, and no rule written into any of the seven surfaces
    # changes that. NOT to be confused with the "multiple operations" variant,
    # which IS agent-avoidable (don't chain `cd X && cmd`) and has its own
    # family below — hence the negative lookahead rather than a bare match.
    r"^(?!.*multiple operations).*this command requires approval|"
    r"^exit plan mode\?\s*$",
    re.IGNORECASE | re.DOTALL,
)

#: NOT an independent failure: Claude Code cancels the SIBLINGS of a parallel
#: tool block when one member of the block errors, and stamps each cancelled
#: sibling with its own ``is_error`` result. The sibling never ran, so it
#: taught the agent nothing and forced no recovery turn of its own — the ONE
#: recovery turn belongs to the member that actually failed, which is already
#: counted. Clustering these as failures therefore both invents a pothole that
#: does not exist ("Bash: Cancelled: parallel tool call Bash(cd …)" was the
#: 4th-largest cluster on the local corpus, 127 sessions) and double-counts the
#: real one. Measured 2026-07-26: 385 occurrences, ~11% of the whole clustered
#: figure. Excluded at extraction time, alongside the human-decline case, so
#: they never enter a cluster at all.
_CASCADE_RE = re.compile(
    r"cancelled:\s*parallel tool call|canceled:\s*parallel tool call",
    re.IGNORECASE,
)


def is_user_decline(error_text: str) -> bool:
    """True if this 'failure' is not a relearn: a human declining an action
    (``_USER_DECLINE_RE``), or a sibling cancelled by another call's failure
    (``_CASCADE_RE``). Named for its original case; both are the same
    "observed as an error, but nothing an agent could have learned" class, and
    both are excluded at the same single extraction-time gate."""
    if not error_text:
        return False
    stripped = error_text.strip()
    return bool(_USER_DECLINE_RE.search(stripped)) or bool(_CASCADE_RE.search(stripped))


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
    #: How many assistant turns this failure actually cost, MEASURED by walking
    #: forward to the turn where the same tool finally succeeded (see
    #: `_stamp_detour_turns`). ``None`` when it could not be measured — the
    #: archive and OTel lanes have spans, not an ordered step list, so they
    #: carry no detour and fall back to 1.0 (the conservative floor, and the
    #: value the whole analyzer assumed before this was measured).
    detour_turns: float | None = None


#: Stop looking for the recovery turn after this many steps. Past it the agent
#: has abandoned the approach rather than retried it, and whatever it went on to
#: do is no longer attributable to this failure. Measured on the local corpus,
#: 95.4% of failures recover well inside this bound.
MAX_RECOVERY_SCAN_STEPS = 40


def _stamp_detour_turns(ordered: list[tuple[int, str, Any]], total_steps: int) -> None:
    """Measure what each failure actually cost, in assistant turns, and stamp it.

    THE ASSUMPTION THIS REPLACES. Every figure in this module used to price one
    occurrence as exactly ONE forced turn: a failed call makes the model emit a
    recovery turn a successful call would not have needed. That is the right
    SHAPE and the wrong SIZE. A failure is rarely one turn — the agent reads the
    error, tries something, often fails again, and only then recovers. Measured
    over 914 failure-bearing sessions on the local corpus (2026-07-26): the
    median failure costs 2 turns and the mean 2.47, so the old basis understated
    every relearn figure by about half. Per-family it ranges from 0.97x
    (a git branch that already exists — one clean retry) to 4.62x (a stale-read
    race, where the agent re-reads, re-edits and races the linter again).

    WHY THE UNION, AND NOT THE SUM. Two failures a step apart have OVERLAPPING
    recovery windows, and billing each one its own full detour charges the same
    assistant turn twice. That is the CLAUDE.md rule 27 double-count, just
    inside one analyzer instead of between two, and it is not small: on the
    corpus the naive sum is 2.45x per occurrence against a true union of 2.07x,
    so 15.3% of it is the same turns counted repeatedly. A turn claimed by k
    overlapping failures is therefore split 1/k between them — every detour turn
    in a session is billed exactly once, however many potholes were in flight.

    ``ordered`` is ``[(step_index, tool_name, episode), ...]`` in walk order for
    ONE session, already filtered to real failures. Mutates the episodes in
    place. Never raises: an unmeasurable session just leaves ``detour_turns``
    at ``None``, which prices at the old 1.0.
    """
    if not ordered:
        return

    # Where does each tool next SUCCEED? Built once per session rather than
    # rescanned per failure, so a long session stays linear.
    succeeded_at: dict[str, list[int]] = {}
    for index, tool_name, episode in ordered:
        if episode is None:                       # a success marker, not a failure
            succeeded_at.setdefault(tool_name, []).append(index)

    failures = [(i, t, e) for i, t, e in ordered if e is not None]
    windows: list[tuple[Any, set[int]]] = []
    measured: list[int] = []
    for index, tool_name, episode in failures:
        recovery = next(
            (
                s for s in succeeded_at.get(tool_name, [])
                if index < s <= index + MAX_RECOVERY_SCAN_STEPS
            ),
            None,
        )
        if recovery is None:
            windows.append((episode, set()))      # resolved below, once a median exists
            continue
        span = recovery - index
        measured.append(span)
        windows.append((episode, set(range(index + 1, index + 1 + span))))

    # A failure that never recovered inside the scan window DID cost something —
    # the agent abandoned the approach, which is not cheaper than retrying it.
    # Its length is genuinely unmeasurable though, so it is charged this
    # session's OWN median detour rather than the scan cap (which would let the
    # unknown case dominate) or a global constant (which would not be measured
    # from this user's data at all). No median to borrow => the 1.0 floor.
    import statistics

    fallback = max(int(statistics.median(measured)) if measured else 1, 1)
    resolved: list[tuple[Any, set[int]]] = []
    for (index, _tool, episode), (_e, window) in zip(failures, windows):
        if window:
            resolved.append((episode, window))
            continue
        end = min(index + 1 + fallback, total_steps + 1)
        resolved.append((episode, set(range(index + 1, end))))

    # Split every shared turn evenly across the failures claiming it.
    claims: dict[int, int] = {}
    for _episode, window in resolved:
        for step in window:
            claims[step] = claims.get(step, 0) + 1
    for episode, window in resolved:
        episode.detour_turns = round(
            sum(1.0 / claims[step] for step in window), 4,
        ) if window else 1.0


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
    # `(step_index, tool_name, episode_or_None)` in walk order. SUCCESSES are
    # recorded too, with a None episode, because the recovery measurement needs
    # to know where each tool started working again — see `_stamp_detour_turns`.
    # The index is the position in the flattened walk, subagent steps included,
    # which is the sequence the model actually emitted turns for.
    ordered: list[tuple[int, str, Any]] = []
    for step_index, (step, move, depth) in enumerate(_walk_moves(real_steps, spine, 0)):
        for tool in step.get("tools") or []:
            tool_name = tool.get("name") or "unknown"
            if tool.get("status") != "error":
                ordered.append((step_index, tool_name, None))
                continue
            error_text = tool.get("error") or ""
            if is_user_decline(error_text):
                # A human's own choice, not a relearn (see is_user_decline). It
                # is not a RECOVERY either, so it is dropped from `ordered`
                # entirely rather than recorded as a success — counting a
                # declined call as the moment the tool started working would
                # end a detour that never actually ended.
                continue
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
            ordered.append((step_index, tool_name, failures[-1]))
    _stamp_detour_turns(ordered, len(real_steps))
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


def _below_threshold_residue(
    clusters: dict[str, _RawCluster],
    kept: list[_RawCluster],
    conn: Any | None,
) -> dict[str, Any]:
    """What the recurrence gate threw away, counted and priced.

    The gate is right to exist — a failure seen once is not a *re*-learn, so it
    gets no cluster, no fix and no claim. It is NOT right to let that read as
    "this was free": those occurrences burned real tokens. So the residue is
    reported as its own explicitly-named quantity rather than silently dropped,
    on the same head-term basis as ``past_overspend_tokens`` (see
    ``BELOW_THRESHOLD_BASIS``). One rate lookup for the whole residue, not one
    per cluster: it is a single aggregate figure, never a per-card one.
    """
    kept_signatures = {c.signature for c in kept}
    dropped = [c for sig, c in clusters.items() if sig not in kept_signatures]
    if not dropped:
        return {"clusters": 0, "occurrences": 0, "tokens": 0, "usd": None}
    occurrences = sum(len(c.failures) for c in dropped)
    sessions = {f.session_id for c in dropped for f in c.failures}
    profile = blended_rate_profile(conn, session_ids=sessions)
    # The SAME head basis the kept clusters use -- the measured cost of the
    # recovery turn these occurrences forced, not the error text's size. This
    # has to move whenever the head term moves or the docstring's "same
    # head-term basis" promise silently becomes false, and the residue starts
    # understating itself by the same ~15-20x the head term used to.
    turn_tokens = (
        _measured_turn_tokens(conn, sessions, profile)
        or GROUNDED_TOKENS_PER_OCCURRENCE
    )
    # Same recovery-arc basis the kept clusters use (see `build_proposals`) —
    # this has to move whenever the head term moves, or the docstring's "same
    # head-term basis" promise silently becomes false.
    tokens = round(sum(
        f.detour_turns or 1.0 for c in dropped for f in c.failures
    ) * turn_tokens)
    return {
        "clusters": len(dropped),
        "occurrences": occurrences,
        "tokens": tokens,
        "usd": (
            round(tokens * profile.input_rate_per_token, 6)
            if profile is not None else None
        ),
    }


# --- Distill pass over the residual (non-family) bucket -----------------------

def _distill_cache_dir(config: Any | None = None) -> Path:
    """Where distilled cluster titles are cached between runs.

    This is a SECOND cache, structurally separate from
    ``relearn_store.default_cache_path``. That one was already threaded through
    ``relearn_apply._storage_base_dir`` so an isolated config (``:memory:``, a
    throwaway ``--db``) writes into a temp root instead of the operator's real
    ``~/.tj``; this one was not, and wrote real files under the real home
    regardless — silently, since it fires only after a successful distill LLM
    call. Same helper, same guarantee, so neither cache can leak on its own.

    ``config`` is None only for direct callers that never had a config to
    thread (the standalone helpers and their tests); those keep the historical
    path.
    """
    if config is not None:
        from tokenjam.core.optimize.relearn_apply import _storage_base_dir
        return _storage_base_dir(config) / "distill_cache" / "relearn"
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
                "pattern": None, "delivery": DELIVERY_CLAUDE_MD_RULE,
                "fix": result.get("fix") or "",
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
    """Heuristic novelty check (the already-documented check): does a
    reachable CLAUDE.md/learnings.md already name this exact gotcha?

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
    #: HOW this fix reaches the agent, and therefore what gets written and
    #: where (``core/rulewrite/kinds``). Declared by the family, never derived
    #: from the artifact's shape.
    delivery:                   str
    scope:                      str              # "project" | "user-global"
    proposed_fix:                str
    examples:                    list[RelearnExample] = field(default_factory=list)
    confidence:                   str = "heuristic"
    novel:                        bool = True
    # Phase 2 (apply) — best-effort cwd of the cluster's (sole, if project-
    # scoped) repo, and a suggested write target derived from it. Both
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
    #: The measured re-read tail behind the figures below: the MEDIAN number of
    #: later calls that still carried one of this cluster's occurrences before
    #: the context was compacted away, and the input-token-equivalent
    #: multiplier that follows from it (`1 + cache_read_ratio x tail`). Carried
    #: so a reader can see WHY an occurrence is worth more than its 1,500
    #: tokens; 0 / 1.0 when no tail could be measured.
    tail_calls_median:            int = 0
    tail_multiplier:              float = 1.0
    # Net-of-standing-cost accounting (`core/optimize/write_budget.py`). This
    # decides whether a PERMANENT artifact is worth writing at all: a CLAUDE.md
    # rule is re-sent on every future session forever, so its standing cost is
    # priced against the same session pace and compared against what the
    # cluster cost (`past_overspend_tokens` below — there is no separate pre-net
    # figure any more; the observation IS the netting input). An EXECUTING hook
    # is never sent to the model as prompt text, so its standing cost is a
    # genuine zero; an INJECTING one is prompt text and is charged for it.
    standing_cost_tokens_per_session: int = 0
    standing_cost_tokens:             int = 0
    standing_cost_basis:              str = ""
    #: gross / standing. Below 1.0 the rule costs more to keep than it saves.
    #: A ratio, not a break-even call count: unlike summarize's one-time
    #: rewrite cost, this cost recurs, so the session count cancels out.
    payback_ratio:                    float | None = None
    net_negative:                     bool = False
    # PAST OVERSPEND — what this recurrence ALREADY COST, observed, before any
    # gate touches it. Structurally separate from every `estimated_*` field
    # above and NEVER netted, suppressed or zeroed:
    #
    #   * a cluster with no fix template still cost real money. "We have no
    #     fix for this" is a gap in OUR library, not a property of the waste;
    #   * a rule that is uneconomic to keep does not retroactively make the
    #     failures free. "Is codifying this worth it?" and "did this cost
    #     anything?" are different questions and get different fields;
    #   * a fix's FUTURE standing cost is never subtracted from money ALREADY
    #     spent. Netting is legitimate for a forward "should I do this"
    #     decision (the `estimated_*` fields) and illegitimate here.
    #
    # The FULL observed cost: the re-issued turns plus their measured re-read
    # tail, in input-token equivalents at the cluster's own blended rate. It is
    # never smaller than what any forward claim off the same cluster can be,
    # which is what keeps a card from ever reading "cost you $14, of which $29
    # was avoidable".
    past_overspend_tokens:            int = 0
    past_overspend_usd:               float | None = None
    past_overspend_basis:             str = ""
    #: The re-read tail's share OF the figure above (a component, not an
    #: addend): what the failure text cost on every later call that still
    #: carried it. Broken out because it is re-sent CONTEXT, the quantity
    #: `analyzers/context_resend.py` prices in full — so a caller comparing the
    #: two analyzers can see exactly which part overlaps rather than guessing.
    past_reread_tokens:               int = 0
    past_reread_usd:                  float | None = None
    #: The SAME observation as `past_overspend_*`, filtered to each of a small
    #: vocabulary of trailing windows, keyed by window label ("30d"). Parallel
    #: to the unbounded fields above and never a replacement for them: those are
    #: the write budget's pre-net gross, so shrinking them in place would
    #: silently flip clusters between "worth a permanent rule" and net-negative.
    #: A FILTER, not a rescale, so each figure is capped at the unbounded one.
    #: ``None`` means UNKNOWN, never zero: either the run computed no windows,
    #: or not one of this cluster's occurrences carries a parseable timestamp,
    #: or the cache predates this field. See `core/optimize/relearn_window.py`.
    past_overspend_windows: dict[str, RelearnWindowedObservation] | None = None
    # Whether a PERMANENT artifact is actually on offer for this cluster, and
    # why not when it isn't (placeholder fix, net-negative payback, budget
    # exhausted, or merged into the family's single block). A suppressed write
    # also sets `advise_only`, so the Review inbox's existing no-apply-path
    # lane renders it with this reason in place of the generic OTel one.
    write_offered:                    bool = True
    write_blocked_reason:             str = ""
    #: The same verdict as a short label, for a dense list where the sentence
    #: above would be a paragraph per row. Derived from the reason by
    #: `write_budget.short_reason`, never phrased locally, so the CLI list and
    #: the Review inbox row cannot name one flag two ways.
    write_blocked_short:              str = ""


@dataclass
class RelearnFinding:
    clusters:            list[RelearnCluster] = field(default_factory=list)
    sessions_scanned:     int = 0
    failures_examined:    int = 0
    distilled_clusters:   int = 0
    dropped_codified:     int = 0
    caveat:                 str = HONESTY_CAVEAT
    # The effective recurrence bar this run applied (config-overridable, see
    # core.config.OptimizeConfig.min_recurring_sessions) — carried on the
    # finding so a renderer's empty-state message never hardcodes a number
    # that could be stale against the user's own config.
    min_sessions:           int = MIN_RECURRING_SESSIONS
    # The observed span (days, earliest to latest timestamped occurrence
    # across every failure this run examined). ``None`` when nothing in the
    # run carried a parseable timestamp. See `_corpus_window_days`.
    window_days:             float | None = None
    # PAST OVERSPEND, summed across EVERY cluster — including the ones with no
    # fix template and the ones whose write is uneconomic. See
    # `RelearnCluster.past_overspend_tokens` for why none of those gates may
    # zero an observed cost. This is the figure the Review inbox card leads
    # with — the only figure a relearn cluster displays.
    past_overspend_tokens: int = 0
    past_overspend_usd:    float | None = None
    past_overspend_basis:  str = ""
    #: The re-read tail across every cluster, reported alongside and never
    #: summed into a rollup (`analyzers/context_resend.py` prices re-sent
    #: context in full).
    past_reread_tokens:    int = 0
    past_reread_usd:       float | None = None
    #: The windowed totals, keyed by the same window labels the clusters carry.
    #: Each is the sum of exactly the per-cluster figures for that window and of
    #: nothing else, so a headline and a per-row floor note can read ONE
    #: quantity over ONE population. ``None`` when the run computed no windows
    #: or the cache predates the field: unknown, never zero.
    past_overspend_windows: dict[str, RelearnWindowTotal] | None = None
    # The recurrence gate's OWN residue: failures real enough to observe but
    # not spread across `min_sessions` distinct sessions, so they never become
    # a cluster. They still cost money. Counted and priced here — on the same
    # head-term basis as `past_overspend_tokens` — rather than silently
    # dropped, so "this analyzer has no action for it" never reads as "this
    # was free". Never claimable and never rolled up: a one-off failure is not
    # a relearn, which is precisely why it gets a count and not a card.
    below_threshold_clusters:      int = 0
    below_threshold_occurrences:   int = 0
    below_threshold_past_overspend_tokens: int = 0
    below_threshold_past_overspend_usd:    float | None = None
    # Which lanes fed this run, and how many sessions each contributed. The
    # archive lane (sessions tokenjam retains telemetry for but Claude Code has
    # already rotated the transcript away) is the whole reason relearn can
    # accumulate history at all — see `compute_relearn_finding`.
    transcript_sessions_scanned: int = 0
    archived_sessions_scanned:   int = 0
    corpus_basis:                str = ""


def _snippet(failure: FailureEpisode) -> str:
    text = failure.error_text or failure.label
    return text[:200]


def _scope_for(repos: set[str]) -> str:
    """§7: concentrated in one repo -> project; spread across many -> user-global."""
    return "project" if len(repos) <= 1 else "user-global"


#: Anything stamped before this year is a SENTINEL, not an observation. Ingest
#: writes `1970-01-01` (a zero epoch) when a span/session carries no usable
#: timestamp, and the DB holds thousands of them. A single sentinel reaching
#: `_corpus_window_days`'s MIN would stretch the observed span to ~56 years and
#: crush every monthly figure to zero, which is exactly how a horizon widened
#: from Claude's transcripts to tokenjam's archive would have broken. Treated as
#: unparseable — the same "don't invent a window from missing data" rule the
#: rest of this module already applies.
MIN_PLAUSIBLE_TS_YEAR = 2000


def _parse_failure_ts(ts: str | None) -> Any:
    """Best-effort ISO-8601 parse of a failure's timestamp. Returns ``None``
    on anything unparseable OR on a pre-``MIN_PLAUSIBLE_TS_YEAR`` sentinel —
    never raises; a bad/missing/sentinel timestamp just doesn't contribute to
    the window span."""
    if not ts:
        return None
    from datetime import datetime as _dt

    try:
        parsed = _dt.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return None if parsed.year < MIN_PLAUSIBLE_TS_YEAR else parsed


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


def _measured_turn_tokens(
    conn: Any, session_ids: set[str], profile: RateProfile | None,
) -> int | None:
    """What ONE extra assistant turn actually cost in these sessions, in
    input-token equivalents. ``None`` when it cannot be measured.

    This is the head term's basis, and it replaces a guess. A failed tool call
    forces a retry: the harness hands the error back to the model, the model
    has to emit another turn to recover. A SUCCESSFUL call would not have
    needed that turn, so it is the marginal cost of the failure, and it is one
    whole round trip -- not the ~1,500 tokens of error text that ride along in
    it.

    Measured from ``cost_usd``, the rate the contributing sessions were
    actually billed at, then divided back through the cluster's own input rate
    so the result lands in the same input-token-equivalent unit every other
    figure here uses. Going through the billed cost rather than summing token
    columns means the cache-read / cache-write / output rate mix is whatever
    those calls really paid, with no ratio assumed locally.

    The MEDIAN, never the mean: per-call cost in a coding corpus is heavily
    right-skewed (a handful of near-context-limit calls dwarf the rest), and a
    mean would let those set the price of every occurrence.
    """
    if conn is None or not session_ids or profile is None:
        return None
    if not profile.input_rate_per_token:
        return None
    ids = sorted(session_ids)
    placeholders = ", ".join(f"${i + 1}" for i in range(len(ids)))
    try:
        row = conn.execute(
            f"SELECT median(cost_usd), "
            f"median(COALESCE(input_tokens, 0) "
            f"       + COALESCE(cache_tokens, 0) * {profile.cache_read_ratio}) "
            f"FROM spans WHERE session_id IN ({placeholders}) "
            f"AND name = 'gen_ai.llm.call'",
            ids,
        ).fetchone()
    except Exception:
        return None
    if not row:
        return None
    billed, from_tokens = row[0], row[1]
    if billed:
        # Preferred: the rate these calls were ACTUALLY billed at, divided back
        # through the cluster's input rate so the result is an input-token
        # equivalent. Assumes no rate mix locally -- the bill already knows it.
        tokens = float(billed) / profile.input_rate_per_token
    elif from_tokens:
        # Fallback for a corpus with no cost recorded (a partial ingest): the
        # prompt's own size in input-token equivalents, on the same
        # `input + cache_read x ratio` convention `_prompt_timelines` uses.
        # Conservative -- it counts the re-sent prompt and not the output.
        tokens = float(from_tokens)
    else:
        return None
    if tokens <= 0:
        return None
    # Floor at the text constant: a measured turn is always the larger of the
    # two, and a pathologically cheap corpus must never price a forced retry
    # BELOW the error text it carries.
    return max(int(round(tokens)), GROUNDED_TOKENS_PER_OCCURRENCE)


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


def _windowed_observations(
    failures: list[FailureEpisode],
    *,
    labels: Sequence[str],
    anchor: Any,
    turn_tokens: int | None,
    profile: RateProfile | None,
    timelines: dict[str, list[tuple[Any, int]]],
    unbounded_tokens: int,
    unbounded_head_tokens: int,
) -> dict[str, RelearnWindowedObservation] | None:
    """This cluster's observed cost, bounded to each trailing window in
    ``labels``. ``None`` when no window can be asserted at all.

    Read ``core/optimize/relearn_window.py``'s module docstring first: it owns
    WHAT this quantity is (a filter, never a rescale), WHY the bound is computed
    here rather than at the API (the cache keeps no per-occurrence dates), and
    the rate/session-set decision (re-derive the occurrence set and the tail,
    reuse the full cluster's price). This function is just that decision applied.

    Returns ``None``, never a bucket reading zero, when not one occurrence
    carries a parseable timestamp: an unplaceable observation is UNKNOWN over a
    window, and a zero there would read as "this cost nothing in that window".
    """
    if not labels:
        return None
    from datetime import timedelta

    anchor_dt = _as_utc(anchor)
    if anchor_dt is None:
        return None
    placeable = [
        (f, stamp) for f, stamp in ((f, _as_utc(_parse_failure_ts(f.ts))) for f in failures)
        if stamp is not None
    ]
    if not placeable:
        return None
    undated = len(failures) - len(placeable)
    # Reused from the full cluster, never recomputed per window. See the module
    # docstring's rate/session-set decision.
    per_turn_tokens = turn_tokens or GROUNDED_TOKENS_PER_OCCURRENCE

    out: dict[str, RelearnWindowedObservation] = {}
    for label in labels:
        span_days = window_days(label)
        start = anchor_dt - timedelta(days=span_days)
        # `>= start`, with no upper bound, is exactly what `since` means on every
        # other route: a trailing window, not a closed interval. A clock-skewed
        # stamp a few seconds past the anchor stays counted rather than falling
        # out of every window at once.
        inside = [f for f, stamp in placeable if stamp >= start]
        occurrences = len(inside)
        detour_turns = sum(f.detour_turns or 1.0 for f in inside)
        multiplier, median_tail = _tail_multiplier(inside, timelines, profile)
        head_tokens = round(detour_turns * per_turn_tokens)
        # Monotone by construction (a subset's detour turns cannot exceed the
        # whole's, and the per-turn price is the same figure), so this `min` is
        # an invariant written down rather than a correction that ever fires.
        if unbounded_head_tokens > 0:
            head_tokens = min(head_tokens, unbounded_head_tokens)
        text_tokens = occurrences * GROUNDED_TOKENS_PER_OCCURRENCE
        reread_raw = round(text_tokens * max(multiplier - 1.0, 0.0))
        # THE ONE PLACE RE-DERIVATION CAN OVERSHOOT. A filtered sample's median
        # tail may exceed the whole cluster's when the window happens to hold
        # the occurrence that sat in context longest. That is noise in a
        # multiplier, not money that was spent twice, so the subset is capped at
        # the whole and the bucket declares it instead of publishing a part
        # larger than its whole.
        reread_tokens = min(reread_raw, max(unbounded_tokens - head_tokens, 0))
        out[label] = RelearnWindowedObservation(
            label=label,
            window_days=span_days,
            window_start=start.isoformat(),
            window_end=anchor_dt.isoformat(),
            occurrences=occurrences,
            sessions=len({f.session_id for f in inside}),
            detour_turns=round(detour_turns, 4),
            undated_occurrences=undated,
            tail_calls_median=median_tail,
            tail_multiplier=round(multiplier, 4),
            past_overspend_tokens=head_tokens + reread_tokens,
            past_overspend_usd=(
                round((head_tokens + reread_tokens) * profile.input_rate_per_token, 6)
                if profile is not None else None
            ),
            past_reread_tokens=reread_tokens,
            past_reread_usd=(
                round(reread_tokens * profile.input_rate_per_token, 6)
                if profile is not None else None
            ),
            capped_at_unbounded=reread_tokens < reread_raw,
            basis=WINDOWED_BASIS,
        )
    return out


def _as_utc(value: Any) -> Any:
    """A datetime made timezone-aware (UTC assumed when naive), or ``None``.

    Failure timestamps come from transcripts and spans and are inconsistently
    stamped; comparing a naive one against an aware anchor raises, and a raised
    exception here would sink a whole cluster's figures.
    """
    if value is None:
        return None
    from datetime import timezone

    try:
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    except AttributeError:
        return None


def build_proposals(
    clusters: list[_RawCluster],
    *,
    min_sessions: int = MIN_RECURRING_SESSIONS,
    doc_text: str = "",
    # The scope's Claude home, used only to suggest a user-global write target
    # (see `relearn_apply.default_target_path`). `None` keeps `~/.claude`.
    claude_home: Path | None = None,
    repo_cwd_map: dict[str, str] | None = None,
    advise_only_repos: set[str] | None = None,
    conn: Any | None = None,
    window_days: float | None = None,
    persona: str = "unknown",
    projection: Any | None = None,
    existing_agent_file_tokens: int | None = None,
    sessions_by_repo: dict[str, int] | None = None,
    window_labels: Sequence[str] | None = None,
    window_anchor: Any | None = None,
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

    ``persona`` gates the CLAUDE.md/skill write exactly like
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

    ``window_labels`` (with ``window_anchor``, the moment the windows trail back
    from) additionally computes each cluster's BOUNDED figure for each named
    window, on new parallel fields. Omitting it is today's behaviour exactly:
    the unbounded fields are byte-identical either way, which they must be,
    since the write budget nets against them as its pre-net gross.
    """
    from tokenjam.core.optimize.relearn_apply import default_target_path, slugify
    from tokenjam.core.optimize.write_budget import (
        REASON_ADVISORY_ONLY,
        REASON_PLACEHOLDER,
        is_placeholder_fix,
        short_reason,
    )

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
        # A cluster that matched no known family has no family to declare a
        # mechanism, so it gets the default one. That is a real default, not a
        # guess about the artifact: with no matcher there is no hook to write.
        delivery = family["delivery"] if family else DELIVERY_CLAUDE_MD_RULE
        fix = family["fix"] if family else _fixes.fix_text(
            "relearn.no_template_matched",
        )

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
                    delivery, scope, repo_cwd, slugify(cluster.title),
                    claude_home=claude_home,
                )
            except Exception:
                suggested_target = ""   # never let a bad path computation sink the proposal

        # A cluster with no derived fix CLAIMS nothing — there is no fix to
        # claim, and a forward "you could recover $X" off a fix the user cannot
        # apply is the "quiet lie in the user's favour" test (CLAUDE.md
        # anti-pattern #22) failing outright.
        #
        # It still COST something, and that is a different field. The absence of
        # a fix template is a gap in OUR library, not evidence the waste was
        # unavoidable; the observed cost is computed for every cluster below,
        # placeholder or not, and reported on the `past_overspend_*` fields.
        # A family whose own fix text says no action is needed is never
        # OFFERED, however well-formed that text is. `is_placeholder_fix` can't
        # see this: the text is a real, specific, non-placeholder sentence — it
        # just happens to say "there is nothing to do here", which is the one
        # thing an offered write must not say.
        advisory_only = bool(family.get("advisory_only")) if family else False
        has_real_fix = not is_placeholder_fix(fix) and not advisory_only

        # Priced for EVERY cluster now, not only the ones with a fix: the tail
        # is part of what the recurrence actually cost, and a cluster that will
        # claim nothing still has to be able to say what it cost.
        profile = blended_rate_profile(conn, session_ids=sessions)
        timelines = _prompt_timelines(conn, sessions) if profile is not None else {}
        multiplier, median_tail = _tail_multiplier(cluster.failures, timelines, profile)

        # Input-token EQUIVALENTS: the head token at the input rate plus each
        # re-read at the cache-read rate, expressed on the head's basis so the
        # token and dollar figures stay proportional. See the constants above.
        # TWO DIFFERENT QUANTITIES, two different bases. They used to share the
        # `GROUNDED_TOKENS_PER_OCCURRENCE` constant, which is right for one and
        # wrong for the other by a measured ~15-20x:
        #
        #   HEAD -- the forced retry. A failed call makes the model emit a turn
        #     it would not have emitted had the call succeeded. That turn costs
        #     a whole round trip, and in a coding session a round trip re-sends
        #     the entire context (this product's own central measurement). On
        #     this corpus a real turn measured ~24k input-token equivalents
        #     against the 1,500 that were being charged.
        #   TAIL -- the error TEXT, re-sent on every later call that still
        #     carries it. ~1,500 tokens IS the right size for a block of error
        #     text, so the constant stays exactly where it was earned.
        #
        # Conflating them priced the retry as though it were the text.
        turn_tokens = _measured_turn_tokens(conn, sessions, profile)
        # NOT `occurrences x one turn`. A failure costs the whole recovery ARC —
        # the turns between hitting the pothole and getting the same tool to
        # work again — which is measured per occurrence at extraction time and
        # medians 2 turns on a real corpus, not 1. Overlapping arcs are already
        # de-duplicated there (a turn shared by two failures is split between
        # them), so summing per-occurrence detours here cannot double-count.
        # An occurrence with no measurable arc (the archive and OTel lanes have
        # spans, not ordered steps) falls back to 1.0: the old assumption, kept
        # as the conservative floor rather than back-filled with a corpus
        # average measured on a different lane.
        detour_turns = sum(f.detour_turns or 1.0 for f in cluster.failures)
        head_tokens = round(
            detour_turns * (turn_tokens or GROUNDED_TOKENS_PER_OCCURRENCE)
        )
        # `multiplier - 1` is the tail's own share (`cache_read_ratio x tail`),
        # kept on the TEXT basis rather than rescaled by the head.
        text_tokens = occurrences * GROUNDED_TOKENS_PER_OCCURRENCE
        gross_tokens = head_tokens + round(text_tokens * max(multiplier - 1.0, 0.0))
        # THE OBSERVATION. Accumulated over the whole scanned corpus, never
        # paced to 30 days, never netted against a fix's standing cost, never
        # zeroed by a gate. Bounded trailing-window views of these SAME
        # occurrences are computed further down onto separate parallel fields —
        # a date filter capped at this figure, never a rescale of it, and never
        # written back over these two lines, which the write budget nets
        # against as its pre-net gross. The re-read tail is broken out as a COMPONENT (not
        # an addend) so a reader can see which part of the figure overlaps the
        # re-sent context `analyzers/context_resend.py` prices in full.
        reread_tokens = max(gross_tokens - head_tokens, 0)
        past_overspend_usd = (
            round(gross_tokens * profile.input_rate_per_token, 6)
            if profile is not None else None
        )
        past_reread_usd = (
            round(reread_tokens * profile.input_rate_per_token, 6)
            if profile is not None else None
        )
        # The CLAIM, as distinct from the observation above -- and deliberately
        # the HEAD ONLY, which is what makes it disjoint from `resend`.
        #
        # THE LINE BETWEEN THE TWO ANALYZERS is defined once, in
        # `resend_tail.RELEARN_RESEND_BOUNDARY`, and quoted verbatim by both
        # sides' basis strings. Read it there rather than restating it here:
        # the short form is that relearn owns whether a call EXISTS (its fix
        # deletes the turn, so the turn's whole cost goes, cache reads
        # included) and `resend` owns how BIG a call is. The head is that
        # deleted turn. The tail -- the error text re-read by LATER calls that
        # happen regardless -- is `resend`'s population by definition, and
        # claiming it here would price the same tokens on two cards
        # (CLAUDE.md rule 27). So the tail stays in the OBSERVED figure, broken
        # out as `past_reread_*`, and is claimed by `resend` alone.

        # The same observation, bounded to each requested trailing window. New
        # parallel fields only: nothing above is touched, because the figures
        # above are the write budget's netting input.
        windows = _windowed_observations(
            cluster.failures,
            labels=window_labels or (),
            anchor=window_anchor,
            turn_tokens=turn_tokens,
            profile=profile,
            timelines=timelines,
            unbounded_tokens=gross_tokens,
            unbounded_head_tokens=head_tokens,
        ) if window_labels else None

        proposals.append(RelearnCluster(
            signature=cluster.signature,
            family_key=cluster.family_key,
            title=cluster.title,
            sessions=len(sessions),
            occurrences=occurrences,
            repos=repos,
            delivery=delivery,
            scope=scope,
            proposed_fix=fix,
            examples=examples,
            novel=True,
            repo_cwd=repo_cwd,
            suggested_target=suggested_target,
            advise_only=advise_only,
            tail_calls_median=median_tail,
            tail_multiplier=round(multiplier, 4),
            # The one observation. `_apply_write_budget` consults it (via
            # `WriteCandidate.gross_tokens`) to decide whether a permanent
            # write is offered at all — the netting disclosure only, never a
            # headline.
            past_overspend_tokens=gross_tokens,
            past_overspend_usd=past_overspend_usd,
            past_overspend_basis=PAST_OVERSPEND_BASIS,
            past_reread_tokens=reread_tokens,
            past_reread_usd=past_reread_usd,
            past_overspend_windows=windows,
            write_offered=has_real_fix,
            write_blocked_reason=(
                "" if has_real_fix
                else (REASON_ADVISORY_ONLY if advisory_only else REASON_PLACEHOLDER)
            ),
            write_blocked_short=(
                "" if has_real_fix
                else short_reason(
                    REASON_ADVISORY_ONLY if advisory_only else REASON_PLACEHOLDER
                )
            ),
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
    from tokenjam.core.optimize.relearn_apply import artifact_for_delivery, slugify

    basis = projection or build_projection_basis(0.0, 0, 0)
    candidates: list[wb.WriteCandidate] = []
    for p in proposals:
        # An ADVISORY family never enters the budget. Its `write_offered` was
        # already set False at construction, and letting it become a candidate
        # here would have `decision.offered` overwrite that a few lines below —
        # which is exactly how the withdrawal was reaching the unit test and
        # NOT the live report. A flag set upstream of a pass that rewrites the
        # same field is not a flag, it is a suggestion.
        family = _FAMILY_BY_KEY.get(p.family_key or "")
        if p.advise_only or not p.suggested_target or (
            family is not None and family.get("advisory_only")
        ):
            continue
        try:
            artifact = artifact_for_delivery(
                asdict(p), p.signature, p.delivery, slugify(p.title),
            )
        except Exception:
            artifact = p.proposed_fix     # never let a render hiccup sink a proposal
        candidates.append(wb.WriteCandidate(
            key=p.signature,
            # Family-unmatched clusters have no family_key; keying them on
            # their own signature keeps each a family of one rather than
            # collapsing every unrelated residual into a single bucket.
            family=p.family_key or f"signature:{p.signature}",
            delivery=p.delivery,
            artifact_text=artifact or p.proposed_fix,
            # `past_overspend_tokens` IS the pre-net observation — there is no
            # separate gross field any more; the past-tense figure doubles as
            # the netting input.
            gross_tokens=p.past_overspend_tokens,
            # Same quantity as `gross_tokens`, priced at the cluster's own
            # rate (`past_overspend_usd` IS `gross_tokens x rate`), so the two
            # divide back to a real price band (repo CLAUDE.md rule 28) and
            # `write_budget`'s value floor has a dollar figure to compare
            # against. Without this the budget netted tokens-only and the
            # floor could never fire.
            gross_usd=p.past_overspend_usd,
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
        out.append(replace(
            p,
            standing_cost_tokens_per_session=decision.standing_tokens_per_session,
            standing_cost_tokens=decision.standing_tokens,
            standing_cost_basis=decision.basis,
            payback_ratio=decision.payback_ratio,
            net_negative=decision.net_negative,
            write_offered=decision.offered,
            write_blocked_reason=decision.reason,
            write_blocked_short=wb.short_reason(decision.reason),
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
    # The scope's Claude home, used ONLY to suggest a user-global write target.
    # `None` keeps the historical `~/.claude`; see `default_target_path`.
    claude_home: Path | None = None,
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
    window_labels: Sequence[str] | None = None,
    window_anchor: Any | None = None,
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
    ``persona`` is forwarded to ``build_proposals`` to gate the
    write — see its docstring.

    ``window_labels`` additionally computes each cluster's observed cost BOUNDED
    to each named trailing window, plus the matching totals on the finding, on
    new parallel fields. It does not scope the SCAN: relearn's horizon is still
    everything tokenjam kept, which is the premise of the analyzer (see
    ``compute_relearn_finding``'s THREE LANES note). What it scopes is which
    occurrences a given figure counts, so a caller with a window selector can
    show relearn money on the same basis as the rest of its page. Default
    ``None`` computes no windows at all and leaves this function's output
    exactly as it was. ``window_anchor`` defaults to now, the only honest
    trailing anchor for a run happening now.
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
    residue = _below_threshold_residue(raw_clusters, recurring, conn)
    distilled = apply_distill_to_residual(
        recurring, cache_dir=distill_cache_dir, enabled=distill_enabled,
    )
    distilled_count = sum(1 for c in distilled if (c.family_key or "").startswith("distilled:"))

    # The shared monthly-extrapolation basis (behavioral requirement #1): the
    # observed span across every failure this run examined, not a fixed
    # window — relearn scans unbounded history. See `_corpus_window_days`.
    # Named `corpus_window_days`, not `window_days`: `window_days` is now also a
    # module-level function (a WINDOW LABEL's span, from `relearn_window`), and
    # the two are different quantities — one is what the corpus happened to
    # span, the other is what a caller asked to bound a figure to.
    corpus_window_days = _corpus_window_days(all_failures)

    # The SAME basis the monthly extrapolation above uses, expressed once as a
    # ProjectionBasis so the write budget can price a permanent rule against
    # the identical session pace the saving is projected on. Mixing the two
    # would reintroduce exactly the time-basis error this accounting exists to
    # remove. See `core/optimize/projection.py`.
    projection = build_projection_basis(
        corpus_window_days or 0.0, _corpus_active_days(all_failures), scanned,
    )

    # Per-repo session counts: what a PROJECT-scoped rule's standing cost is
    # actually charged against (a user-global one is charged against them all).
    sessions_by_repo: dict[str, int] = {}
    for _session_id, repo in sessions:
        sessions_by_repo[repo] = sessions_by_repo.get(repo, 0) + 1

    # The moment every bounded window trails back from. Fixed once for the whole
    # run so every cluster's "last 30 days" means the same 30 days, and carried
    # onto each bucket so a reader of a CACHED finding can see that the window
    # ended when the detector ran rather than when they opened the page.
    anchor = window_anchor
    if window_labels and anchor is None:
        from tokenjam.utils.time_parse import utcnow as _utcnow

        anchor = _utcnow()

    proposals, dropped = build_proposals(
        distilled, min_sessions=min_sessions, doc_text=codified_doc_text,
        claude_home=claude_home,
        repo_cwd_map=repo_cwd_map, advise_only_repos=advise_only_repos,
        conn=conn, window_days=corpus_window_days, persona=persona,
        projection=projection,
        existing_agent_file_tokens=existing_agent_file_tokens,
        sessions_by_repo=sessions_by_repo,
        window_labels=window_labels, window_anchor=anchor,
    )
    # The past-tense totals sum EVERY cluster, gated or not — that is the whole
    # point of the field, and the only figure a relearn cluster displays.
    past_tokens = sum(p.past_overspend_tokens for p in proposals)
    past_priced = [p.past_overspend_usd for p in proposals if p.past_overspend_usd is not None]
    reread_tokens = sum(p.past_reread_tokens for p in proposals)
    reread_priced = [p.past_reread_usd for p in proposals if p.past_reread_usd is not None]

    # The windowed totals: the sum of the CLUSTERS' OWN windowed figures and
    # nothing else, so a headline and a per-row floor note read one quantity over
    # one population. Clusters that could not be placed in a window are counted
    # as unknown inside each total rather than summed in as zero.
    windowed_totals: dict[str, RelearnWindowTotal] | None = None
    if window_labels and proposals:
        per_cluster = [p.past_overspend_windows for p in proposals]
        windowed_totals = {}
        for label in window_labels:
            total = sum_windowed(
                per_cluster, label,
                anchor_start="", anchor_end=anchor.isoformat() if anchor else "",
            )
            # The start is the label's own span back from the shared anchor; take
            # it off a cluster that actually computed it rather than recomputing
            # the arithmetic in a second place.
            for windows in per_cluster:
                if windows and label in windows:
                    total.window_start = windows[label].window_start
                    break
            windowed_totals[label] = total

    return RelearnFinding(
        clusters=proposals,
        sessions_scanned=scanned,
        failures_examined=len(all_failures),
        distilled_clusters=distilled_count,
        dropped_codified=dropped,
        min_sessions=min_sessions,
        window_days=corpus_window_days,
        past_overspend_tokens=past_tokens,
        past_overspend_usd=round(sum(past_priced), 6) if past_priced else None,
        past_overspend_basis=PAST_OVERSPEND_BASIS if proposals else "",
        past_reread_tokens=reread_tokens,
        past_reread_usd=round(sum(reread_priced), 6) if reread_priced else None,
        past_overspend_windows=windowed_totals,
        below_threshold_clusters=residue["clusters"],
        below_threshold_occurrences=residue["occurrences"],
        below_threshold_past_overspend_tokens=residue["tokens"],
        below_threshold_past_overspend_usd=residue["usd"],
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


def _retention_cutoff(retention_days: int | None) -> Any:
    """The oldest timestamp tokenjam still keeps, or ``None`` for unbounded.

    The archive lane's horizon. Deliberately derived from the STORE's retention
    rather than the report window: relearn's whole premise is that it sees
    further back than one window, and re-imposing a 30-day filter here is the
    exact defect the archive lane exists to remove.
    """
    if not retention_days or retention_days <= 0:
        return None
    from datetime import timedelta

    from tokenjam.utils.time_parse import utcnow

    return utcnow() - timedelta(days=int(retention_days))


def _archived_session_ids(
    conn: Any, on_disk: set[str], since: Any = None,
) -> set[str]:
    """Coding sessions tokenjam retains but Claude Code has already rotated.

    The archive lane's population: every session id the ``sessions`` table knows
    that has NO ``.jsonl`` left on disk. Subtracting ``on_disk`` is what keeps
    the two lanes disjoint, so a failure is never extracted twice.

    Best-effort — an unreadable table yields an empty set, which degrades the
    run to exactly today's transcript-only behaviour rather than failing it.
    """
    sql = "SELECT DISTINCT session_id FROM sessions"
    params: list[Any] = []
    if since is not None:
        sql += " WHERE started_at >= $1"
        params.append(since)
    try:
        rows = conn.execute(sql, params).fetchall()
    except Exception:
        return set()
    return {str(r[0]) for r in rows if r[0] and str(r[0]) not in on_disk}


def _corpus_basis(
    transcript_sessions: int,
    archived_sessions: int,
    window_days: float | None,
    retention_days: int | None,
) -> str:
    """One sentence naming what the run actually scanned, so a reader can tell
    an empty result from an un-scanned one — and can see that the horizon is
    tokenjam's retention, not Claude Code's transcript rotation."""
    span = f"{window_days:.1f} days" if window_days else "an undetermined span"
    kept = (
        f"bounded by tokenjam's {retention_days}-day retention"
        if retention_days else "unbounded by tokenjam's retention settings"
    )
    return (
        f"{transcript_sessions} session(s) read from on-disk transcripts plus "
        f"{archived_sessions} whose transcript Claude Code has already rotated "
        f"away, recovered from retained telemetry ({kept}); occurrences span "
        f"{span}."
    )


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
    from tokenjam.core.transcript import (
        _locate_transcript,
        first_recorded_cwd,
        read_records,
    )

    out: dict[str, str] = {}
    for session_id, repo in sessions:
        if repo in out:
            continue
        path = _locate_transcript(session_id, projects_root)
        if path is None:
            continue
        # `first_recorded_cwd` is the shared extractor (deadweight and rule
        # placement read it too) — see its docstring for why this is not three
        # copies of the same five-record loop any more.
        cwd = first_recorded_cwd(read_records(path, cache_dir=transcript_cache_dir))
        if cwd:
            out[repo] = cwd
    return out


def compute_relearn_finding(
    conn: Any | None = None,
    since: Any | None = None,
    *,
    projects_root: Path | str | None = None,
    claude_home: Path | None = None,
    distill_cache_dir: Path | None = None,
    distill_enabled: bool = True,
    min_sessions: int = MIN_RECURRING_SESSIONS,
    transcript_cache_dir: Path | None = None,
    persona: str = "unknown",
    existing_agent_file_tokens: int | None = None,
    retention_days: int | None = None,
    window_labels: Sequence[str] | None = RELEARN_WINDOW_LABELS,
) -> RelearnFinding:
    """Standalone entry point that doesn't need a full ``AnalyzerContext`` —
    used by the serve-time background cache job (``api/routes/relearn.py``)
    and by tests. ``conn`` is an OPTIONAL DuckDB connection; without it the run
    degrades to the transcript lane alone (weaker repo labels, and NO archive —
    see below), which is the only thing a filesystem-only scan can offer.

    ``persona`` (default ``"unknown"``, the conservative no-write default —
    see ``build_proposals``) is forwarded to gate the CLAUDE.md/skill write.
    ``run(ctx)`` below passes the report's own ``ctx.persona`` rather than
    re-deriving it here, so a report never carries two different persona
    classifications for the same window.

    THE HORIZON IS TOKENJAM'S ARCHIVE, NOT CLAUDE'S ROTATION. This is the whole
    premise of the analyzer and it used to be false. relearn is the one detector
    whose signal is long-horizon recurrence, and it read Claude Code's on-disk
    ``.jsonl`` transcripts — which Claude Code itself rotates
    (``cleanupPeriodDays``, default 30). Measured on a real corpus the scanned
    span was 37.4 days against a DuckDB holding 120, so relearn was structurally
    incapable of accumulating history however long tokenjam ran: the one
    analyzer that needed the archive was wired to the source the archive exists
    to outlive.

    THREE LANES, disjoint by construction so nothing is counted twice:

    1. **Transcript** — every session Claude Code still has a ``.jsonl`` for.
       Richest signatures (raw tool error text + the method-spine move), so it
       stays the fast path for recent history.
    2. **Archive** — coding sessions tokenjam still holds telemetry for whose
       transcript has already been rotated away. Their failures come from the
       ``spans`` table (``relearn_otel.extract_archived_coding_failures``);
       coarser signatures, but the alternative is that they vanish. This lane is
       what makes the horizon tokenjam's retention rather than Claude's.
    3. **OTel** — failing spans from NON-coding agents, which never had a
       transcript at all. Marked ``advise_only``: detect and advise, never
       apply.

    ``retention_days`` bounds lanes 2 and 3 to what tokenjam actually keeps
    (``storage.retention_days``); ``None`` means unbounded, which is correct for
    a store the caller has already scoped. ``since`` still pre-filters the
    transcript lane by mtime for an explicitly incremental scan — but note that
    passing a 30-day window here re-imposes exactly the horizon this design
    removes, so ``run(ctx)`` deliberately does NOT pass the report window.
    Heavy (tens of seconds over a full local corpus) — callers that serve this
    over HTTP MUST cache the result, not compute it per-request.

    ``transcript_cache_dir``, when given, transparently caches each session's
    parsed transcript on disk (``core.transcript_cache``) so a re-run over an
    unchanged corpus skips re-parsing every session it already has a fresh
    cache entry for. ``None`` (the default) preserves this function's
    original always-reparse behavior — only the registered ``run(ctx)`` entry
    point and the serve-time background job opt in.

    ``window_labels`` defaults ON here, unlike on ``analyze_relearns``: this is
    the entry point whose result gets CACHED and served over HTTP, and the cache
    is the one place a bounded figure can come from at all (the cached cluster
    keeps no per-occurrence dates, so no route can derive one later). Computing
    them costs no extra database round trips — every window reuses the rate
    profile, per-turn cost and prompt timelines the unbounded figure already
    built. Pass ``None`` to opt out. Nothing here SCOPES the scan to a window:
    the horizon stays tokenjam's retention, per the note above.
    """
    root = resolve_projects_root(projects_root)
    repo_map = _repo_map_from_db(conn) if conn is not None else {}
    archive_since = _retention_cutoff(retention_days)

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

            span_failures = extract_span_failures(conn, archive_since or since)
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

    # THE ARCHIVE LANE. Every coding session tokenjam still holds telemetry for
    # whose transcript Claude Code has already rotated away. Without this the
    # detector's horizon is Claude's `cleanupPeriodDays`, not tokenjam's
    # retention, and no amount of running longer accumulates any history at all.
    # Best-effort, exactly like the OTel lane above.
    on_disk = {session_id for session_id, _repo in sessions}
    archived_failures: list[FailureEpisode] = []
    if conn is not None:
        try:
            from tokenjam.core.optimize.relearn_otel import (
                extract_archived_coding_failures,
            )

            archived_failures = extract_archived_coding_failures(
                conn, _archived_session_ids(conn, on_disk, archive_since), archive_since,
            )
        except Exception:
            archived_failures = []

    doc_text = ""
    repo_cwd_map: dict[str, str] = {}
    try:
        repo_cwd_map = _repo_cwd_map_for(
            sessions, root, transcript_cache_dir=transcript_cache_dir,
        )
        doc_text = _doc_text(_candidate_doc_paths(set(repo_cwd_map.values())))
    except Exception:
        doc_text = ""

    finding = analyze_relearns(
        sessions, projects_root=root, claude_home=claude_home,
        codified_doc_text=doc_text,
        distill_enabled=distill_enabled, distill_cache_dir=distill_cache_dir,
        repo_cwd_map=repo_cwd_map,
        extra_failures=span_failures + archived_failures,
        advise_only_repos=advise_only_repos,
        min_sessions=min_sessions, transcript_cache_dir=transcript_cache_dir,
        conn=conn, persona=persona,
        existing_agent_file_tokens=existing_agent_file_tokens,
        window_labels=window_labels,
    )
    archived_sessions = len({f.session_id for f in archived_failures})
    from dataclasses import replace as _replace

    return _replace(
        finding,
        transcript_sessions_scanned=len(sessions),
        archived_sessions_scanned=archived_sessions,
        corpus_basis=_corpus_basis(
            len(sessions), archived_sessions, finding.window_days, retention_days,
        ),
    )


@register("relearn")
def run(ctx: AnalyzerContext) -> None:
    """Registry entry point. Attaches a ``RelearnFinding`` to
    ``ctx.report.findings["relearn"]`` — see ``compute_relearn_finding`` for
    the full-corpus behaviour and performance note.

    Passes the resolved persistent parse cache dir (``core.transcript_cache.
    default_cache_dir``) so a re-run over an unchanged corpus skips
    re-parsing every session it already has a fresh cache entry for.

    Deliberately does NOT forward ``ctx.since``. Every OTHER analyzer is
    window-scoped; relearn is the one whose entire signal is recurrence across
    history, so scoping it to the report window would cap its horizon at the
    window the user happened to ask about. Its horizon is
    ``storage.retention_days`` — what tokenjam actually kept — which is the
    point of keeping it.
    """
    from tokenjam.core.optimize.report_window import report_window_label
    from tokenjam.core.optimize.scope import resolve_analyzer_scope, resolve_write_scope
    from tokenjam.core.transcript_cache import default_cache_dir

    scope = ctx.scope if ctx.scope is not None else resolve_analyzer_scope(ctx.config)
    if not scope.enabled:
        # The leak this guards is not hypothetical: served against an empty
        # throwaway `--db`, this analyzer surfaced recurring-mistake entries
        # carrying real file paths from an unrelated project, because its
        # evidence came off the machine's global transcript tree rather than
        # the database being served.
        ctx.report.filesystem_scan_skipped_reason = scope.reason
        ctx.report.findings["relearn"] = RelearnFinding()
        return

    optimize_cfg = getattr(ctx.config, "optimize", None)
    min_sessions = getattr(
        optimize_cfg, "min_recurring_sessions", MIN_RECURRING_SESSIONS,
    )
    storage_cfg = getattr(ctx.config, "storage", None)
    # Resolved, never read off the field: `storage.retention_days` is now
    # derived from the chosen analysis span and is None on a default config,
    # which a raw read would take to mean "unbounded" — the opposite of what a
    # 90-day span promises. See core/analysis_span.py.
    retention_days = (
        retention_days_for(storage_cfg) if storage_cfg is not None else None
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
        ctx.conn, min_sessions=min_sessions,
        retention_days=retention_days,
        # So the inbox's one window label always has a bucket on this side
        # too. Resolved through `core/optimize/report_window`, the seam the
        # cost side and the stored report both take their window from — the
        # inbox matches this vocabulary EXACTLY, so a label derived any other
        # way here drops every cluster out of the headline.
        window_labels=window_labels_including(
            report_window_label(ctx.config, ctx.conn)
        ),
        projects_root=scope.projects_root,
        # THE APPLY TARGET AND THE WRITE GUARD MUST COME FROM ONE PLACE. This
        # passed `scope.claude_home` directly while `relearn_store` passed
        # `resolve_write_scope(scope=scope).suggest_root` for the same purpose,
        # and that store carries a comment recording what independent
        # derivation cost last time: the API's write guard authorizes against
        # the OTHER half of this same type, so a card whose evidence is scoped
        # one way and whose write target is scoped another describes two
        # different machines. Both callers now resolve it here.
        claude_home=resolve_write_scope(scope=scope).suggest_root,
        distill_cache_dir=_distill_cache_dir(ctx.config),
        transcript_cache_dir=default_cache_dir(ctx.config),
        persona=ctx.persona,
        existing_agent_file_tokens=measured_agent_file_tokens(
            ctx.report.findings.get("summarize"),
        ),
    )
