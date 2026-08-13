---
description: Dollar/token figure discipline — actionability, disjointness, basis, horizon, round-trip typing.
paths:
  - "tokenjam/core/optimize/**"
  - "tokenjam/core/framing.py"
  - "tokenjam/core/cost.py"
  - "tokenjam/core/fixes/**"
  - "tokenjam/api/routes/optimize.py"
  - "tokenjam/api/routes/cost.py"
  - "tokenjam/cli/cmd_optimize.py"
  - "tokenjam/cli/cmd_cost.py"
---

# Cost-figure rules

The canonical per-analyzer dollar-field contract lives in `.claude/rules/optimize-architecture.md`.
Analyzer authoring / gating rules are in `.claude/rules/optimize-analyzers.md`.

### Critical Rule 22 — Never show a figure the user cannot act on; suppress it and say why

A dollar amount is only honest when the reader can realise it on THEIR plan and with the data
actually observed. Three instances now: `deadweight` prints no dollars for a server where no priced
model was observed (a `$0.00` would read as "this server is free"); `placement` prints no dollars off
`api` plans, because the Batch API's flat discount is an api-billed price lever a subscription user
cannot pull; and the long-standing subscription/local framing in `core/framing.py` suppresses spend
for plans that don't pay per token. The rule for the NEXT analyzer, not a log of these three: **decide
what the number MEANS on each pricing mode before you render it, and where it cannot be realised,
print the token figure plus one sentence saying why there is no dollar figure — never a zero, never a
blank, never a number borrowed from a mode that does happen to bill.** Two traps that look like
helpfulness: (a) reusing the peer analyzers' "estimated recoverable" wording for a figure that is a
PRICE difference on the same tokens rather than tokens freed (`placement` is exactly this — batch
bills the same work at half rate, it frees nothing, so its card says "estimated price difference" and
labels the token count as the size of the affected workload); (b) falling back to a plausible default
rate when none was observed. The test to apply to any number you are about to render: **would this be
a quiet lie in the user's favour?** If yes, it does not ship, however good it looks on the card.

### Critical Rule 27 — Two analyzers that both claim `past_overspend_*` must draw from DISJOINT spans

The rollup cannot save you. `past_overspend_rollup` (`core/optimize/cost_proposals.py`) dedupes by
EXACT signature string, so two analyzers pricing identical tokens both survive (Claude Code files a
subagent's turns under the PARENT session id, so an unfiltered `GROUP BY session_id` absorbs them —
how `downsize` (per `session_id`) and `subagent` (per `(session_id, sub_agent_id)`) double-counted a
Task dispatch — and `MODE(model)` can even name a subagent's model as the session's). **Before adding
a recoverable figure, name the exact span population it claims and prove no shipped analyzer already
claims those rows.** Three remedies, prefer the first. **(a) Disjoint at the source** —
`analyze_model_downgrade` filters `sub_agent_id IS NULL`; use it whenever a column filter separates
them, since it fixes the derived shapes (`MODE(model)`, per-turn tool counts) too. **(b) Subtract
what the more specific card claimed** — `_per_agent_cache_recoverable_by_model`, for a generic row
that must stay window-wide. **(c) Partition the POPULATION behind one shared predicate** — `resend`
and `downsize` price the identical mechanism, so only *which sessions* each may claim separates them:
`analyzers/resend_tail.premium_driver_role`, imported by both, with the shared tail arithmetic beside
it. One predicate cannot drift; two tuned threshold sets will. Two traps: **only the CLAIM narrows,
never the DENOMINATOR** (`percent_of_sessions`/`percent_of_tokens` are whole-window shares — hence
the separate unfiltered totals query), and **a share computed against a narrowed figure must narrow
too** (`thinking_tokens_by_session(..., main_thread_only=True)`, else the ratio can exceed 1.0). Pin
end-to-end `build_report` → `cost_proposals_from_report` → `past_overspend_rollup` against the seeded
session's real token total (`tests/unit/test_rollup_subagent_downsize_dedup.py`); a per-analyzer test
cannot see an overlap between two analyzers.

### Critical Rule 28 — A finding's `past_overspend_tokens` and `past_overspend_usd` must count the SAME events

Divide one by the other and the implied per-token rate has to land inside a real price band. Rule 27
covers two analyzers claiming the same spans; this is the same failure INSIDE one finding, harder to
see because both fields are individually defensible (caught when `summarize` priced dollars
per-call-multiplied while its token field stayed on the one-time file reduction; `deadweight`'s
per-session-vs-per-call bug was the same class). The cross-analyzer rollup sums the token fields, so
one basis leak corrupts the product's token floor. **Write the check as a test:** assert
`usd / tokens` equals the blend the basis string advertises and stays between the cache-read rate and
the input rate. A hardcoded-number assertion drifts along with the fields; a rate assertion cannot,
since a basis mismatch throws the implied rate out of band. Two corollaries. **(a) The "no evidence"
degrade is symmetric** — if the load count was not observed BOTH fields go `None`, never a zero on
one side and a number on the other; a candidate contributes to both sums or neither. **(b) A second
basis gets its OWN named field** — `file_reduction_tokens`, declared in `estimate_basis` — never
overloading the aggregate field cross-analyzer rollups read. Re-check derived percentages after a
basis change: `reduction_pct` is saved ÷ source tokens and must keep the one-time numerator.

### Critical Rule 30 — A figure may wear the "waste"/"overspend" label only to the extent it was AVOIDABLE

And two figures computed over different POPULATIONS must never be shown as two views of one quantity.
`resend` computed `cost_of_waste_usd` over every session with repeat volume and `past_overspend_usd`
over a much smaller filtered subset, then rendered the first as the past-tense hero. Nobody wrote the
claim down, but the pairing made it: that the large majority of the money had been shown to be
unavoidable. It had not. When the gap was actually decomposed it was entirely made of sessions **not
analysed**, in three distinct ways: sessions **ceded whole to `downsize`'s driver-role case**
(analysed on ANOTHER card, per Rule 27), sessions **dropped by `MIN_SESSION_CONTEXT_TOKENS`** before
the calculation ran, and sessions outside the compaction-bounded tail definition or below the measured
offloadable share. Three ways of not being analysed; zero findings of necessity. The ratio of two
figures over two populations means nothing, and a reader will compute it anyway. **The rules this
generalises to any analyzer pairing an observed cost with an avoidable share.** *(a) The label follows
the arithmetic, not the magnitude.* Wanting a bigger headline is not a reason; `reuse` already made
this trade correctly by pricing `reps - 1` rather than `reps`, because the necessary first planning
instance is not waste. The central stamper (`cost_proposals._with_past_overspend`) now puts
`past_overspend_usd` on `past_overspend_*` for EVERY analyzer, and there is no second field for a
total cost to land on: the `observed_cost_*` pair that once carried one is deleted, so a coverage
statement in words is the only way this gap gets reported. *(b) If you ship both, partition the cost
by the SAME predicate the avoidable figure uses and state the split on the card.*
`context_resend._coverage_class` is the one classification deciding both which sessions enter the
avoidable figure and which bucket their cost lands in, so the coverage the card states cannot
disagree with the coverage the code applied; a test asserts the three buckets re-sum to the total. The
prose (`coverage_note`) must end by saying the difference is what was NOT ANALYSED, not what was
unavoidable — that sentence is the whole point. *(c) A behavioural sample generalised onto a
different population has to disclose its sample size and spread every time it is shown.*
`offloadable_share` is measured across only the small minority of sessions that delegate at all, then
applied to the large majority that never do — which are exactly the ones being advised — and its
per-session spread runs nearly the whole 0-1 range, so the mean is a weak summary. Those figures
belong in the basis string the card prints (recomputed per window), never restated as a constant in
prose. A structural measure would be better, but the per-tool-call delegability it needs is computed
nowhere in this tree, so the honest move is to say so IN the basis string rather than to keep a bare
scalar that reads like a corpus property. *(d) Both sides of a paired display read the same raw
window, never a paced one.* A pace ratio applied to only one side makes a pure time-basis artifact
look like avoidability. This was fixed STRUCTURALLY rather than by convention: `compute_projection_ratio`
and the `estimated_monthly_*` fields it fed are deleted, so there is no paced figure left to render by
mistake. **The test that catches the whole class: build a report, adapt it, and assert no proposal
carries a second per-analyzer dollar field beside `past_overspend_usd`**
(`tests/unit/test_past_overspend.py::test_the_waste_labelled_figure_never_exceeds_the_avoidable_figure`
and `::test_there_is_exactly_one_rollup_and_one_per_analyzer_dollar_field`) — a per-card assertion on
hand-built numbers cannot see a stamper that routes the wrong field into the headline slot.

### Critical Rule 32 — An analyzer's horizon is a property of its DATA SOURCE, not of its docstring

And "we have no action for this" is never "this was unavoidable". Two independent failures, both
found in `relearn`, both worth checking for in any analyzer. **(a) Horizon.** `relearn` was documented
and treated as the unbounded-history analyzer, and read Claude Code's on-disk `.jsonl` transcripts —
which Claude Code itself rotates and prunes on its own retention setting (`cleanupPeriodDays`).
Measured once on a real machine, the scanned span was capped near that retention setting while
tokenjam's own DuckDB held substantially more history — so the one analyzer whose entire signal is
long-horizon recurrence was structurally incapable of accumulating any history at all, however long
tokenjam ran. **The product's value proposition is that it retains what the agent discards; any
analyzer reading the discarded source cannot deliver it.** The fix shape is an ARCHIVE LANE, not a
rewrite: keep the rich source as a fast path for what it still holds, and recover everything it has
dropped from the DB, made disjoint by an explicit id set so nothing is counted twice
(`relearn_otel.extract_archived_coding_failures`). Widening a horizon to the DB also brings the
`1970-01-01` sentinel timestamps into range — guard any MIN-based span derivation with a
plausible-year floor or a single sentinel reports a decades-long window and crushes every rate to
zero. **(b) Action-availability gates.** Three separate gates in `relearn` turned a real, incurred
cost into `$0`, and on a real corpus they disqualified most of the candidates between them: no fix
template matched, the write-budget judged a rule net-negative, and the recurrence threshold. None of
the three establishes that the waste was avoidable — they establish that *we* have no remedy, which is
a gap in the PRODUCT. Rule 22 ("never show a figure the user cannot act on") governs a FORWARD claim —
"you could recover $X" off a fix that does not exist is the quiet lie it exists to stop. It does NOT
govern a PAST observation: what a behaviour already cost is true whether or not our library has a
remedy. So carry the two as separate fields with separate names, and never let a gate on one touch the
other. Corollary: **a fix's standing/maintenance cost may be netted out of a forward "is this worth
doing" figure and never out of a backward one** — netting a future cost against past spend materially
understated relearn's past figure when it was measured. Corollary 2: when adding a past-tense figure
to a cross-analyzer rollup, check rule 27's disjointness for the CLAIM specifically — relearn's
re-read tail is the same re-sent context `resend` already claims, so its card carries the observation
and deliberately no `past_overspend_*`.

### Critical Rule 41 — A `list[Any]` field loses its TYPE across a round trip while its VALUE survives intact

And a broad `except` downstream turns that into a whole analyzer silently missing from a published
total. `hydrate_dataclass` dispatches on the annotation, so a field declared `list[Any]` to keep
`core/optimize/types.py` free of an analyzer import gives it nothing to work with: the rows come back
as plain dicts, correctly shaped and correctly numbered, with the class gone. Nothing fails at the
seam. The failure surfaces far away, as `'dict' object has no attribute ...` inside whichever consumer
reads a row attribute — and if that consumer sits under a per-item `except Exception: continue`, the
item is dropped and the total it fed just gets smaller. That is how one analyzer's entire contribution
disappeared from the Review inbox while the Dashboard tile, reading the finding directly and never
round-tripping, went on publishing it: two surfaces, one analyzer, no error anywhere. Three rules.
**(a) Declare the element type as DATA when you cannot import it**:
`field(metadata={"hydrate": "package.module:ClassName"})`, resolved lazily at hydration time
(`runner._hydrate_target`), so the layering stays intact and the round trip is lossless in type as
well as value. A test asserts every `list[Any]` in `types.py` carries it. **(b) A round-trip test that
compares VALUES cannot see this** — it passes on dicts, because the numbers really did survive.
Assert `isinstance` of the rehydrated row, not just its fields. **(c) A broad `except` around a
per-item transform must record WHAT it skipped**, and the caller must publish that as a
known-unknown rather than absorbing it: a skipped item is a hole in every figure summed from the
result, and a null-figure disclosure through the `excluded` channel is the difference between a total
that says "this analyzer is missing" and one that quietly reads as complete. Continuing on error is
right; continuing without a trace is what makes the arithmetic lie.
