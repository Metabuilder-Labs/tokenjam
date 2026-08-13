"""The catalogued fixes themselves.

Split from ``catalog.py`` so the data sits apart from the machinery, and
imported for its side effect the way ``core/optimize/analyzers`` is. Every
entry names the observation it answers and the shapes it may not excuse, which
is what lets ``lint.py`` check the property that matters: applying the fix must
be able to ERASE the number its analyzer found.
"""
from __future__ import annotations

from tokenjam.core.fixes.catalog import (
    Substitution,
    LEVER_AWARENESS,
    LEVER_EFFORT,
    LEVER_MODEL,
    LEVER_OFFLOAD,
    LEVER_ROUTING,
    PERSONA_ANY,
    PERSONA_CLAUDE_CODE,
    PERSONA_SDK,
    FixRecord,
    register,
)

#: The shapes the subagent/resend family must never give a pass to. These are
#: exactly what the analyzer bills for: the `output_tokens < 2000` /
#: `tool_calls <= 5` gate was DELETED from `subagent_rightsizing` because on a
#: real Claude Code corpus it excluded the expensive dispatches, so a fix that
#: re-imposes it in prose undoes the gate fix in the one place the user reads.
_SIZING_RELICENSE = frozenset({
    "little tool work",
    "few tool calls",
    "short result rarely needs",
    "short conclusion rarely needs",
    "rarely needs the premium tier",
})

SUBAGENT_RUBRIC = register(FixRecord(
    key="subagent.sizing_rubric",
    text=(
        "Right-size Task-dispatched subagents: default every subagent to the "
        "cheapest same-family model that fits its shape, and treat that "
        "default as the answer unless the dispatch itself states one of these "
        "conditions: the subtask IS the architecture or design decision this "
        "session exists to make; it must reconcile sources that disagree into "
        "one judgement a later step cannot re-derive; or a cheaper model "
        "already attempted it in this session and its output was rejected. "
        "How much tool work a subagent does and how long its result runs are "
        "not on that list: a broad, tool-heavy, long-output dispatch is the "
        "expensive one, not the hard one, and it is the one this rule exists "
        "to route down."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"subagent"}),
    answers="premium-tier models running Task dispatches that a cheaper same-family model fits",
    lever=LEVER_MODEL,
    must_not_relicense=_SIZING_RELICENSE,
    grounding=(
        Substitution(
            find="Right-size Task-dispatched subagents",
            template="Right-size the {agents} dispatches",
        ),
    ),
))

RIGHTSIZE_TEMPLATE = register(FixRecord(
    key="resend.rightsize_worker",
    text=(
        "Then right-size what you offload to. Default the worker to the "
        "cheapest same-family model that fits its shape, and pin both that "
        "model and its reasoning effort in its own definition file so every "
        "future dispatch inherits them instead of defaulting to whatever the "
        "parent runs on. How much it reads and how long its conclusion runs "
        "are not reasons to keep the premium tier; they are what makes the "
        "dispatch expensive in the first place."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"resend"}),
    answers="context-heavy work run inline on a premium model instead of in a right-sized worker",
    lever=LEVER_MODEL,
    must_not_relicense=_SIZING_RELICENSE,
    grounding=(
        Substitution(find="the worker", template="the {agents} workers"),
    ),
))

#: The CREATE fix. A Claude Code dispatch is almost always a BUILT-IN
#: (`general-purpose`, `Explore`, `Plan`, `fork`), which has no definition file
#: — and the product used to read "no file" as "no fix", falling through to
#: prose. The docs are explicit that a user or project subagent of the same
#: name OVERRIDES the built-in and keeps its own `model` field, so the file can
#: be created. The waste's origin belongs in the same breath: `model` defaults
#: to `inherit`, so an opus-driven session dispatches opus workers unless
#: pinned. Nobody decided that; it is an unset default.
SUBAGENT_DEFINE_BUILTIN = register(FixRecord(
    key="subagent.define_builtin_override",
    text=(
        "Pin the model for the built-in subagents this session dispatches. A "
        "subagent's `model` defaults to `inherit`, so an Opus-driven session "
        "hands every Task dispatch an Opus worker unless something says "
        "otherwise — that is an unset default, not a decision. A user or "
        "project subagent defined with the same name as a built-in overrides "
        "it and keeps its own `model`, so creating the definition file is the "
        "fix: give it the cheapest same-family model that fits the work it "
        "actually does."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"subagent"}),
    answers="built-in subagents inheriting the driver's premium model because no definition file pins one",
    lever=LEVER_MODEL,
    must_not_relicense=_SIZING_RELICENSE,
))

SUBAGENT_EFFORT = register(FixRecord(
    key="subagent.pin_effort",
    text=(
        "Pin a subagent's effort alongside its model. They answer different "
        "diagnoses and neither substitutes for the other: a worker that did "
        "not know enough needs a larger model, a worker that did not try hard "
        "enough needs more effort. Both are frontmatter fields on the "
        "subagent's own definition file, so a dispatch inherits them instead "
        "of taking the parent's."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"subagent"}),
    answers="subagents inheriting the parent's effort setting rather than one sized to their task",
    lever=LEVER_EFFORT,
))

#: THE offload instruction. Three analyzers used to write this into a
#: CLAUDE.md in three separately-authored wordings — `resend`'s
#: `SUBAGENT_OFFLOAD_FIX`, `downsize`'s driver-role advice, and `relearn`'s
#: `context_overflow` family. They price genuinely different span populations
#: (Critical Rule 27 is untouched), but they were all telling the agent the
#: same thing, so three near-identical blocks could land in one file.
#:
#: That is not merely untidy: length and redundancy REDUCE adherence, so
#: writing the rule three times makes it less likely to be followed than
#: writing it once — the opposite of what each analyzer intended. It also
#: reads as broken to anyone who opens the file.
#:
#: The text below is what the three wordings shared, stated once. Per-analyzer
#: framing (why THIS card is showing it) belongs in the card's advise text,
#: never in a second copy of the rule.
#:
#: **This record is about WHERE the work runs and nothing else.** Model pinning
#: belongs to ``resend.rightsize_worker``, which owns "what it runs on", and
#: keeping the two separate is not pedantry: the compound artifact renders both
#: records back to back, so a pinning sentence here lands immediately before
#: the one that owns it and the user reads the same instruction twice. Caught
#: by rendering the composed artifact and reading it — the pairwise catalog
#: lint scored the pair at 42%, under threshold, because each record is only
#: partly redundant. One record, one instruction, is what makes composition
#: safe.
OFFLOAD_RULE = register(FixRecord(
    key="resend.offload_to_subagent",
    text=(
        "Offload context-heavy sub-tasks (broad file reads, log sweeps, "
        "multi-file search, long tool-output loops, exploratory "
        "investigation) to a subagent instead of running them inline in the "
        "main thread, and prefer a targeted search plus a bounded read over "
        "reading a large file end to end. A subagent's own tool logs and "
        "intermediate output stay in its own context; only its short "
        "conclusion returns to the caller, so the material that would "
        "otherwise be re-sent on every later turn never accumulates on the "
        "main thread to begin with. This is not a request to downgrade the "
        "session you are driving: the driver keeps the premium model and "
        "keeps making the decisions. What changes is where the bulk context "
        "lives."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    # One record, three analyzers. Each prices a different population and each
    # references THIS text rather than authoring its own.
    analyzers=frozenset({"resend", "downsize", "relearn"}),
    answers="context-heavy work run inline on the main thread, re-sent on every later turn",
    lever=LEVER_OFFLOAD,
    # `relearn`'s `context_overflow` family shows this record for a reason the
    # other two do not have — the request was REJECTED, not merely expensive —
    # and that sentence used to be written beside the family table. It names
    # what was observed and stops; restating any part of the rule below is the
    # defect this record exists to have already fixed.
    lead_ins=((
        "relearn",
        "This session hit the model's context ceiling and the request was "
        "rejected outright: the tokens were spent and no completion came back.",
    ),),
    # REPLACES the vague enumeration with the observed one; it does not append
    # to it. "the `Read` and `Grep` sweeps you run in `optimize/`" is shorter
    # than the parenthesised list it stands in for, which is the point — the
    # guidance is specific AND concise, and a longer rule is a less-followed
    # rule.
    grounding=(
        Substitution(
            find=(
                "context-heavy sub-tasks (broad file reads, log sweeps, "
                "multi-file search, long tool-output loops, exploratory "
                "investigation)"
            ),
            template="the {tools} sweeps you run in {repos}",
        ),
        # Falls back to naming just the directories when the tool mix was not
        # observed — still concrete, still shorter than the generic span.
        Substitution(
            find=(
                "context-heavy sub-tasks (broad file reads, log sweeps, "
                "multi-file search, long tool-output loops, exploratory "
                "investigation)"
            ),
            template="the context-heavy work you run in {repos}",
        ),
    ),
))

#: The SDK counterpart of the same observation. A Claude Code user cannot
#: paste this and an SDK caller cannot use the offload rule above; handing
#: either the other's fix names an action they cannot take.
#:
#: FOUR analyzers' worth of call sites used to state this instruction in four
#: separately-authored wordings — "Add a stable cache prefix / enable prompt
#: caching", "Add a cache_control breakpoint on this agent's stable prefix",
#: "Add a cache_control breakpoint right after this prefix", and this record.
#: They are one instruction, so they are now one record, and the observed
#: specifics (which model, which prefix) are grounding rather than a fifth
#: wording.
#:
#: The trim half moved out to ``resend.sdk_trim_context``. It was a SECOND
#: instruction living in this record's text, which is the shape that makes
#: composition unsafe: the resend SDK card renders both, so a user was told to
#: trim the request in the same block twice.
SDK_CACHE_CONTROL = register(FixRecord(
    key="resend.sdk_cache_breakpoint",
    text=(
        "Adopt a cache_control breakpoint at the call site, on the stable "
        "prefix the calls share (system prompt and tool definitions), so that "
        "prefix bills at the cache rate instead of full input price on every "
        "turn."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_SDK}),
    # One record, every analyzer that hands out a caching instruction. The
    # per-analyzer runtime gate is unchanged and still lives in
    # ``cost_proposals._persona_gated_cache_fields``.
    analyzers=frozenset({"resend", "cache"}),
    answers="repeated prompt prefix billed at full input rate on every turn",
    lever=LEVER_ROUTING,
    grounding=(
        Substitution(
            find="on every turn",
            template="on every {models} call",
        ),
    ),
))

#: The trim instruction, once. It was authored twice: as a clause inside
#: ``resend.sdk_cache_breakpoint`` and as ``context_resend.RESEND_SDK_TRIM_FIX``,
#: and both can be shown on the same card.
#:
#: Deliberately scoped to what an SDK caller controls DIRECTLY — the content it
#: assembles into the next request. `/compact` is a Claude Code interactive
#: command an SDK caller has no access to, which is why the compaction record
#: below is a different record for a different persona rather than a branch in
#: this one.
SDK_TRIM_CONTEXT = register(FixRecord(
    key="resend.sdk_trim_context",
    text=(
        "Trim or summarize the accumulated context before including it in the "
        "next request instead of resending it unchanged turn over turn. An SDK "
        "caller constructs the whole prompt, so the repeated volume this "
        "finding measures is yours to bound; cutting it at the call site "
        "reduces future prompt size independent of whether caching is enabled."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_SDK}),
    analyzers=frozenset({"resend"}),
    answers="accumulated context reassembled into every request unchanged",
    lever=LEVER_ROUTING,
    must_not_relicense=frozenset({
        # The billed behaviour is resending the accumulated context unchanged.
        "resend it unchanged",
        "the harness will trim",
    }),
))

#: IMMEDIATE RELIEF, and labelled as such rather than as a fix. `/compact` is
#: transient, persists nothing past the session, and never changes the pattern
#: going forward — the record's own text says so, which is why it is not
#: advisory_only: there IS an action, it is simply bounded.
COMPACTION_RELIEF = register(FixRecord(
    key="resend.compaction_relief",
    text=(
        "Run /compact (or start a fresh session) once accumulated context "
        "crosses your working set. The repeated volume this finding measures "
        "is the same content being re-sent turn over turn: trimming it "
        "directly cuts future prompt size, regardless of whether caching is "
        "on. This is a manual, per-session action, so it never fixes the "
        "pattern going forward — treat it as immediate relief for an "
        "already-full session, not the durable fix."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"resend"}),
    answers="a single session already carrying more accumulated context than its working set",
    lever=LEVER_AWARENESS,
    must_not_relicense=frozenset({
        # This record is allowed to be transient — it says so — but it must not
        # present itself as the durable fix, because the analyzer bills for the
        # pattern continuing, not for one full session.
        "this fixes the pattern",
        "no further change is needed",
    }),
))

#: An OBSERVATION for the Claude Code persona, not an offer. Prompt caching is
#: controlled by fields on the raw API request, and a Claude Code session does
#: not construct that request — the harness does. There is no code for this user
#: to edit, so there is no action to sell, and the record says so.
#:
#: ``advisory_only`` is what makes that mechanical rather than a property of
#: whoever last read the prose. What the behaviour already cost is untouched
#: (Critical Rule 32); only the offer is withheld.
CACHE_NO_CC_LEVER = register(FixRecord(
    key="cache.no_claude_code_lever",
    text=(
        "Prompt caching is controlled by cache_control fields on the raw "
        "Anthropic API request. A Claude Code session doesn't construct that "
        "request itself; the harness does. There's no code here for you to "
        "edit, so no fix is needed or shown for it."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"cache"}),
    answers="repeated context re-billed as fresh input in a harness the user does not construct requests for",
    lever=LEVER_AWARENESS,
    advisory_only=True,
))

#: The Claude Code levers for a model-routing finding. A CC session cannot swap
#: its own interactive model mid-turn, so the generic "route to a cheaper model"
#: text an SDK caller gets names an action this persona cannot take.
#:
#: `/compact` IS NOT ON THIS LIST, deliberately: it is transient, persists
#: nothing past the session, and users who feel a session filling already do it
#: unprompted. Listing it beside three durable levers implies it is the same
#: kind of thing, and pads a list whose length is itself a cost to adherence.
#:
#: The "mixed window" wording was a SECOND, shorter copy of this same list
#: (``_DOWNSIZE_MIXED_CC_NOTE``). It is now this record plus a lead-in naming
#: which share of the window it applies to — the framing differs, the
#: instruction does not.
DOWNSIZE_CC_LEVERS = register(FixRecord(
    key="downsize.claude_code_levers",
    text=(
        "You can't switch your own interactive model mid-session, so this is "
        "not a fix to paste into your own request the way an SDK caller would. "
        "The actionable levers instead: `tj route export --target ccr` (or "
        "--target litellm) to route future calls through a cheaper model, `tj "
        "optimize subagent` to right-size subagent models and context, or a "
        "CLAUDE.md/subagent directive telling this agent to dispatch cheaper "
        "subagents for this shape of work."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"downsize"}),
    answers="a premium model running work a cheaper same-family model fits, in a session that cannot swap its own model",
    lever=LEVER_ROUTING,
    lead_ins=((
        "downsize.mixed_window",
        "For the Claude Code sessions in this window:",
    ),),
))

#: The `script` proposal's note. Behavioural, same class of surface as the
#: subagent rubric: an orchestrator or the model itself reads it.
SCRIPT_DETERMINISTIC = register(FixRecord(
    key="script.replace_agent_turn_with_script",
    text=(
        "This tool-call pattern repeated across many sessions with the same "
        "structural shape (same tools, same argument types, different values). "
        "Consider replacing it with a deterministic script that runs these "
        "calls directly, and reserve the agent turn for the parts that "
        "actually need a model's judgment."
    ),
    delivery="skill",
    personas=frozenset({PERSONA_ANY}),
    analyzers=frozenset({"script"}),
    answers="a deterministic tool-call sequence re-derived by a model turn every time it runs",
    lever=LEVER_OFFLOAD,
))

#: The `reuse` proposal's note. Carries its own review caveat because a
#: skeleton match is a candidate, not proof the plan is identical — that
#: sentence is honesty about the SIGNAL, not an escape hatch on the
#: instruction, and the two read alike but are not the same thing.
REUSE_PLAN_SKELETON = register(FixRecord(
    key="reuse.template_the_plan_skeleton",
    text=(
        "This class of task shares a planning skeleton: the same tool sequence "
        "follows the first planning call, session after session, with only the "
        "argument values differing (dates, versions, paths). Consider "
        "templating the plan for this shape instead of re-planning it from "
        "scratch each time. Review before reusing: a skeleton match is a "
        "candidate, not proof the plan is identical."
    ),
    delivery="skill",
    personas=frozenset({PERSONA_ANY}),
    analyzers=frozenset({"reuse"}),
    answers="the same planning skeleton re-derived from scratch across sessions",
    lever=LEVER_OFFLOAD,
))

#: Framed more conservatively than its peers on purpose: a long answer is often
#: the right one, so this is a candidate to review rather than a rule to
#: enforce. Note what the hedge is and is not — it conditions the instruction on
#: whether brevity COSTS something (completeness, correctness, clarity), which a
#: reader of the output can check, not on the agent's own rating of how hard its
#: task was. That distinction is the whole of the self-graded-hatch property.
VERBOSITY_PREFER_SHORTER = register(FixRecord(
    key="verbosity.prefer_shorter_when_it_serves",
    text=(
        "This shape of task ran noticeably longer, output-wise, than similar "
        "sessions recently. Worth a look: if a shorter answer would serve this "
        "shape of task just as well, prefer it — but only when brevity doesn't "
        "cost completeness, correctness, or clarity. A longer answer is often "
        "the right one; this is a candidate to review, not a rule to enforce."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_ANY}),
    analyzers=frozenset({"verbosity"}),
    answers="output tokens running well above the median for the same task shape",
    lever=LEVER_AWARENESS,
    must_not_relicense=frozenset({
        # The analyzer bills for output well above the cohort median. A fix that
        # says long output is fine gives a pass to its entire population — and
        # this record's honest hedge sits one clause away from exactly that, so
        # it is the record most worth guarding mechanically.
        "length is not a problem",
        "longer output is always",
        "ignore the length",
    }),
))

#: The batch-placement instruction. Advise-only in practice — adoption is an
#: architectural change rather than a configuration flip, and the card says so
#: next to the number — but the instruction itself is one sentence and belongs
#: here rather than inline in the one-paste block, which is precisely the slot
#: whose text a user pastes without re-reading.
BATCH_SUBMIT_ASYNC = register(FixRecord(
    key="batch.submit_through_batch_api",
    text=(
        "Submit these workloads through the Batch API instead of the "
        "synchronous endpoint: they are billed at the batch rate, and nothing "
        "downstream of them is waiting on the response."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_SDK}),
    analyzers=frozenset({"batch"}),
    answers="non-interactive workloads billed at the synchronous rate",
    lever=LEVER_ROUTING,
))

#: The residual bucket's placeholder. Not an instruction and not pretending to
#: be one: a cluster that matched no known family has no fix to hand out, and
#: this says so rather than inventing something plausible.
#:
#: It is catalogued anyway, for two reasons. It is text a user reads, so it
#: belongs where the other text a user reads lives; and ``is_placeholder_fix``
#: matches on its wording to block the write, so a wording drifting here while
#: that matcher stays put would silently start OFFERING a placeholder as a rule.
#: One home means the string and the matcher cannot drift apart.
#:
#: Distinct from ``relearn.edit_before_read``, which is marked advisory_only:
#: "no template matched" and "none is needed" are different statements, and
#: collapsing them would let a real do-nothing fix hide as an unmatched one.
RELEARN_NO_TEMPLATE = register(FixRecord(
    key="relearn.no_template_matched",
    text="Review examples — no known fix template matched.",
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_ANY}),
    analyzers=frozenset({"relearn"}),
    answers="a recurring failure cluster that matched no known family",
    lever=LEVER_AWARENESS,
))

#: The prompt-template trim instruction. Distinct from
#: ``resend.sdk_trim_context``, which is about context ACCUMULATED across turns:
#: this one is about a static template that carries dead weight on every single
#: call, and the two are fixed in different places.
TRIM_PROMPT_TEMPLATE = register(FixRecord(
    key="trim.drop_low_significance_regions",
    text=(
        "Trim the low-significance regions from this step's prompt template "
        "(boilerplate, repeated instructions, dead context) so every call "
        "carries fewer input tokens."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_SDK}),
    analyzers=frozenset({"trim"}),
    answers="a prompt template carrying regions that contribute nothing on every call",
    lever=LEVER_ROUTING,
))

#: An OBSERVATION, not an offer. The harness already errors clearly and agents
#: self-correct next turn, so there is no action to sell — but the recurrence
#: genuinely cost something, and that figure stands (Critical Rule 32).
EDIT_BEFORE_READ = register(FixRecord(
    key="relearn.edit_before_read",
    text=(
        "The harness already blocks an Edit/Write before a Read with a clear "
        "error and agents reliably self-correct by reading on the next turn, "
        "so no rule or hook is needed here. This is reported as awareness "
        "only: what the retries already cost is real and is shown, but there "
        "is no change to make."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"relearn"}),
    answers="Edit/Write attempted before Read, retried after the harness error",
    lever=LEVER_AWARENESS,
    advisory_only=True,
))

#: An awareness fix that IS confined to identifiable files, so it can carry a
#: glob and cost almost nothing. The observation names the file class directly:
#: the mistake happens when editing a migration without reading it, which is a
#: statement about a kind of file rather than about the shape of the next
#: action. Contrast every model/effort/offload record above, which decide
#: something BEFORE any file is read and so must stay always-resident.
MIGRATION_READ_FIRST = register(FixRecord(
    key="relearn.migration_read_before_edit",
    text=(
        "Read a migration file in full before editing it. Migrations are "
        "append-only and ordered, so an edit written from a remembered shape "
        "rather than the current contents lands in the wrong place or "
        "duplicates an existing step, and the failure surfaces later as a "
        "schema that does not match its own history."
    ),
    delivery="path_scoped_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"relearn"}),
    answers="edits to migration files written without reading the file first",
    lever=LEVER_AWARENESS,
    path_globs=("**/migrations/**", "**/migrate/**"),
))


# --- relearn: the known failure families -------------------------------------#
#
# One record per family, keyed ``relearn.<family_key>`` so the family table and
# the catalog cannot drift apart on which text belongs to which matcher. These
# were inline strings in ``analyzers/relearn.py``; nothing checked them, so the
# properties the lint enforces held only by whoever last read the file.
#
# Their delivery is declared per family because the FAMILY is what knows:
# `sleep_chain` blocks a command and injects nothing, while the PostToolUseFailure
# families exist precisely to inject text. Those cost opposite amounts and no
# property of the artifact tells them apart.

RELEARN_CWD_CONFUSION = register(FixRecord(
    key="relearn.cwd_confusion",
    text=(
        "PostToolUseFailure hook (Bash/Read): react only after a "
        "'no such file or directory' failure by injecting the real cwd + "
        "a short directory listing as additionalContext, so the agent "
        "recovers in one shot instead of a PreToolUse guess-and-block on "
        "every relative path (which would misfire on normal usage)."
    ),
    delivery="injecting_hook",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"relearn"}),
    answers="tool calls failing on a relative path resolved against the wrong working directory",
    lever=LEVER_AWARENESS,
))

RELEARN_SLEEP_CHAIN = register(FixRecord(
    key="relearn.sleep_chain",
    text=(
        "PreToolUse hook: block a `sleep N && <check>` Bash chain and point the "
        "agent at the Monitor tool instead of a busy-wait."
    ),
    delivery="executing_hook",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"relearn"}),
    answers="blocked sleep-chain Bash commands retried after the harness refused them",
    lever=LEVER_AWARENESS,
    must_not_relicense=frozenset({
        # A busy-wait is the whole population. Suggesting a shorter sleep keeps
        # the sleep.
        "use a shorter sleep",
        "sleep for less",
    }),
))

#: ONE record for TWO families. `stale_read_race` ("modified since read") and
#: `edit_string_not_found` ("string to replace not found") are different
#: matchers over different failures, and they stay different families — the
#: hook spec, the delivery and the evidence are all per-family. But the
#: INSTRUCTION is identical in both: re-read the file before retrying the edit.
#: They shipped as two separately-authored wordings and scored 94% against each
#: other, which is the three-analyzers-one-rule case in miniature: both hooks
#: can be applied by the same user, so the same sentence lands twice.
#:
#: Which failure prompted it is the FAMILY's job to say (each has its own title
#: and evidence), not this text's. Naming the trigger here is what forked one
#: instruction into two in the first place.
RELEARN_REREAD_BEFORE_RETRYING_EDIT = register(FixRecord(
    key="relearn.reread_before_retrying_edit",
    text=(
        "PostToolUseFailure hook (Edit/Write/MultiEdit): react only after the "
        "edit has already failed, by injecting a re-Read reminder as "
        "additionalContext so the retry is written against the file's current "
        "contents — never touches a successful edit."
    ),
    delivery="injecting_hook",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"relearn"}),
    answers="edits retried against remembered file contents after the file moved underneath them",
    lever=LEVER_AWARENESS,
    must_not_relicense=frozenset({
        # The billed behaviour IS the retry against remembered contents. A fix
        # that says to retry, or to try a shorter string, re-licenses it.
        "retry the same edit",
        "try again with",
        "retry without reading",
    }),
))

#: The OPPOSITE failure to ``edit_string_not_found`` — too many matches, not
#: zero — and it takes the opposite fix, which is why the two families must
#: never share a bucket and why their records are separate here.
RELEARN_EDIT_AMBIGUOUS_MATCH = register(FixRecord(
    key="relearn.edit_ambiguous_match",
    text=(
        "CLAUDE.md/skill note: when an Edit's `old_string` appears more "
        "than once, include enough surrounding lines to make it unique "
        "rather than retrying the same short string — or pass "
        "`replace_all: true` when every occurrence really should change."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"relearn"}),
    answers="Edit calls rejected because old_string matched more than once",
    lever=LEVER_AWARENESS,
))

RELEARN_READ_TOO_LARGE = register(FixRecord(
    key="relearn.read_too_large",
    text=(
        "CLAUDE.md/skill note: this file is too large to read whole. Grep "
        "for the symbol first and Read only the region around the hit "
        "(`offset`/`limit`), or delegate the sweep to a subagent so the "
        "bulk never lands in this thread's context."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"relearn"}),
    answers="Read calls rejected for exceeding the tool's max-tokens ceiling",
    lever=LEVER_AWARENESS,
))

RELEARN_READ_DIRECTORY = register(FixRecord(
    key="relearn.read_directory",
    text=(
        "CLAUDE.md/skill note: Read takes a file path. To see what is in a "
        "directory use Glob (or `ls` via Bash), then Read the file you "
        "actually want."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"relearn"}),
    answers="Read calls pointed at a directory rather than a file",
    lever=LEVER_AWARENESS,
))

RELEARN_READ_OFFSET_MALFORMED = register(FixRecord(
    key="relearn.read_offset_malformed",
    text=(
        "CLAUDE.md/skill note: Read's `offset`/`limit` are scalars, not "
        "arrays — pass a single number for each."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"relearn"}),
    answers="Read calls rejected for passing offset or limit as an array",
    lever=LEVER_AWARENESS,
))

RELEARN_DEFERRED_TOOL_COLD = register(FixRecord(
    key="relearn.deferred_tool_cold",
    text=(
        "Skill/scoped note: deferred tools need a ToolSearch lookup for their "
        "schema before the first call; optionally a PreToolUse intercept hook."
    ),
    delivery="skill",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"relearn"}),
    answers="deferred tools called before their schema was fetched, so the call could not resolve",
    lever=LEVER_AWARENESS,
))

RELEARN_COMMAND_NOT_FOUND = register(FixRecord(
    key="relearn.command_not_found",
    text=(
        "CLAUDE.md/skill note: this shell doesn't have that binary/builtin on "
        "PATH. Common causes here: using bare `python` instead of `python3`, "
        "or a bash-only builtin (`mapfile`, `shopt`, `[[ ... ]]` extensions) "
        "that doesn't exist under this shell (e.g. zsh, sh) or POSIX mode. "
        "Prefer the portable/explicit form."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"relearn"}),
    answers="Bash calls failing on a binary or builtin this shell does not have",
    lever=LEVER_AWARENESS,
))

RELEARN_BASH_TIMEOUT = register(FixRecord(
    key="relearn.bash_timeout",
    text=(
        "CLAUDE.md/skill note: this command outlived the tool's timeout "
        "and was killed, so its work was lost and the tokens spent "
        "waiting bought nothing. Run long jobs in the background "
        "(`run_in_background`) and poll for completion, or raise the "
        "call's own timeout when the wait is genuinely expected."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"relearn"}),
    answers="Bash commands held in the foreground until the harness killed them",
    lever=LEVER_AWARENESS,
    must_not_relicense=frozenset({
        # The analyzer bills for the tokens spent WAITING on a call the harness
        # then killed. A fix that says to wait longer, or to keep it in the
        # foreground, leaves that number exactly where it found it.
        "wait for it to finish",
        "just wait",
        "run it in the foreground",
        "block until",
    }),
))

RELEARN_BASH_CHAINED_APPROVAL = register(FixRecord(
    key="relearn.bash_chained_approval",
    text=(
        "CLAUDE.md/skill note: a chained Bash command (`cd X && cmd`, "
        "`a; b`) is approved as a whole, so one un-allowlisted part blocks "
        "the entire chain. Issue the parts as separate Bash calls, and "
        "prefer an absolute path over a leading `cd`."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"relearn"}),
    answers="chained Bash commands tripping the approval prompt the parts alone would not",
    lever=LEVER_AWARENESS,
))

RELEARN_GIT_BRANCH_EXISTS = register(FixRecord(
    key="relearn.git_branch_exists",
    text=(
        "CLAUDE.md/skill note: check out the existing branch "
        "(`git checkout <name>`) instead of re-creating it, or pick a "
        "fresh name — `git checkout -b` on an existing branch always "
        "fails."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"relearn"}),
    answers="`git checkout -b` retried against a branch name that already exists",
    lever=LEVER_AWARENESS,
))

RELEARN_WEBFETCH_DOMAIN_BLOCKED = register(FixRecord(
    key="relearn.webfetch_domain_blocked",
    text=(
        "CLAUDE.md/skill note: this domain is blocked — use a search tool "
        "or a different source instead of retrying the fetch."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"relearn"}),
    answers="fetches retried against a domain the harness will not reach",
    lever=LEVER_AWARENESS,
))


#: `deadweight` and `summarize` had NO catalog record at all, which is a
#: sharper problem than it sounds: `lint_catalog` can only report on what is
#: catalogued, so the lint was structurally incapable of saying anything about
#: two analyzers' user-facing prose while reporting a clean bill of health for
#: everything. An analyzer with no record is not a clean analyzer; it is an
#: unexamined one.

#: The DURABLE half of a dead-server card. The grounded half — this server's
#: name, its config path, how many sessions it sat in — stays at the render
#: site, which is exactly the split the guard's docstring draws.
#:
#: What must be stated once, not per card, is the multi-declaration rule: the
#: analyzer aggregates a server's tax across every file that declares it, while
#: the apply edits ONE. A card that leaves that implicit lets a partial fix read
#: as a whole one.
DEADWEIGHT_REMOVE_SERVER = register(FixRecord(
    key="deadweight.remove_unused_server",
    text=(
        "Remove the MCP server, or narrow a user-scoped one to the projects "
        "that use it: its tool schemas are injected into every call of every "
        "session it is configured for, and nothing in this window called any "
        "of them. Editing one declaration stops the tax only for the sessions "
        "that reach the server through that file, so a server declared in "
        "several locations needs each one edited before the whole figure goes "
        "away."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"deadweight"}),
    answers="MCP servers whose schemas are injected every session and never called",
    lever=LEVER_AWARENESS,
    #: A low-traffic server is still a candidate for removal; the card's own
    #: honesty caveat is what asks for review, and the fix text must not
    #: pre-emptively excuse the exact shape the analyzer bills for.
    must_not_relicense=frozenset({
        "occasional use is fine",
        "a rarely-called server is harmless",
    }),
))

#: The plugin lane's fix. Same shape as the MCP one — an installed dependency
#: paid for continuously — and the same one-line reversible edit, which is why
#: it is a separate record rather than a widening of the server text: the
#: command to run is different, and a fix that names the wrong mechanism is
#: worse than one that says nothing. Verified against the shipping CLI:
#: enable/disable is whole-plugin only (`enabledPlugins` is a binary map) and
#: the real command is `claude plugin disable <name>` — NOT a hand-edit of
#: `~/.claude/settings.json`, which this record used to say (and was tagged
#: `delivery="claude_md_rule"`, which it never was either: this is a CLI
#: command the user runs themselves, not a rule written into an
#: instruction file).
DEADWEIGHT_DISABLE_PLUGIN = register(FixRecord(
    key="deadweight.disable_unused_plugin",
    text=(
        "Run `claude plugin disable <name>`. Every skill and agent an enabled "
        "plugin ships lists its name and description into the context of "
        "every session before anything is invoked, and any MCP server it "
        "contributes injects its own tool schemas the same way, so a plugin "
        "that is never used is paying that listing for nothing. Re-enabling "
        "with `claude plugin enable <name>` costs nothing and keeps the "
        "plugin installed, so this is reversible in one command."
    ),
    delivery="cli_command",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"deadweight"}),
    answers=(
        "enabled plugins whose skill/agent listing (and any contributed MCP "
        "server) is resident every session and never invoked"
    ),
    lever=LEVER_AWARENESS,
))

#: The plugin lane's PARTIAL-use case. Enable/disable is whole-plugin only
#: (verified against the shipping CLI, same fact `DEADWEIGHT_DISABLE_PLUGIN`
#: rests on), so a plugin with some components used and some not has no clean
#: fix: disabling it would remove a component something still relies on. A
#: card that named unused components here but stayed silent about WHY no
#: action follows would read as an oversight rather than a deliberate
#: statement of the toggle's own shape.
DEADWEIGHT_PLUGIN_PARTIAL_USE = register(FixRecord(
    key="deadweight.plugin_partial_use_no_fix",
    text=(
        "Enable/disable is whole-plugin only, so a plugin with some "
        "components used and some not has no fix on offer here: turning it "
        "off would remove a component something else still relies on."
    ),
    delivery="cli_command",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"deadweight"}),
    answers=(
        "plugins where some components fired in the recency window and some "
        "did not, so the whole-plugin toggle has no fix that avoids breaking "
        "the used component"
    ),
    lever=LEVER_AWARENESS,
))

#: The DURABLE half of the oversized-instruction-file card. This is the text
#: that used to live as a ~330-character paragraph hardcoded in
#: `cost_proposals._summarize_to_proposals`, invisible to the guard because the
#: slot it was assigned to was not listed, the constant check looked for a name
#: one letter off, and the length floor answered a different question.
SUMMARIZE_REVIEW_OVERSIZED = register(FixRecord(
    key="summarize.review_oversized_files",
    text=(
        "Reduce an oversized always-loaded instruction file through the "
        "summarize review surface rather than editing it blind: `tj summarize "
        "list`, then `tj summarize check`, then `tj summarize apply`. Every "
        "code block, table and tag comes back verbatim and only prose is "
        "rewritten, one file at a time, and nothing is written until you have "
        "read the diff and passed --go."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"summarize"}),
    answers="always-loaded instruction files carrying compressible prose on every session",
    lever=LEVER_AWARENESS,
))


__all__ = [
    "CACHE_NO_CC_LEVER",
    "DEADWEIGHT_DISABLE_PLUGIN",
    "DEADWEIGHT_REMOVE_SERVER",
    "SUMMARIZE_REVIEW_OVERSIZED",
    "COMPACTION_RELIEF",
    "DOWNSIZE_CC_LEVERS",
    "EDIT_BEFORE_READ",
    "OFFLOAD_RULE",
    "RIGHTSIZE_TEMPLATE",
    "REUSE_PLAN_SKELETON",
    "SCRIPT_DETERMINISTIC",
    "SDK_CACHE_CONTROL",
    "SDK_TRIM_CONTEXT",
    "SUBAGENT_DEFINE_BUILTIN",
    "SUBAGENT_EFFORT",
    "SUBAGENT_RUBRIC",
    "TRIM_PROMPT_TEMPLATE",
    "VERBOSITY_PREFER_SHORTER",
]
