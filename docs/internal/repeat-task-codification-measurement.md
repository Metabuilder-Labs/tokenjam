# Does writing a lesson down make the same work cheaper? — a measurement

**Status: measured, negative. Do not build the session-level relearn rework on this corpus.**
Run 2026-07-26 against the full local corpus (6,176 sessions / 527,089 spans / 2026-05-22 → 2026-07-24).
Reproduce with `tokenjam/core/optimize/repeat_task.py` (unit tests: `tests/unit/test_repeat_task.py`).

## The question

`relearn` prices **failure episodes** — individual erroring tool calls, clustered by failure
signature. Measured total on this corpus: **$46.26 across 55 clusters**. The hypothesis was that this
is implausibly small because the episode is the wrong unit: the waste is not the failed call, it is
the **inflated session** around it. An agent that does not know a project's constraints explores,
reads the wrong files, backtracks, and re-sends a growing context on every turn of that flailing.

The proposed replacement: cluster sessions doing the same repeated work, then compare cost across the
point at which the relevant lesson was written down. The field observation motivating it: setting up
one project repeatedly took 40+ minutes, and dropped to under 20 minutes after its faults were
codified.

Four things had to be true. Three of them are not.

## 1. Are repeat-task clusters identifiable? — **Yes, and easily**

Using the session's first user prompt as a task statement, with ids / paths / numbers masked and
scoped to the project (`repeat_task.task_cluster_key`):

| | |
|---|---|
| sessions with a recoverable task statement | 2,711 |
| distinct clusters | 346 |
| clusters with ≥ 24 sessions | 19, covering 1,668 sessions |
| largest cluster | 496 sessions |

31 clusters at n ≥ 5 cover **89%** of all sessions with a task statement. This is much better than
expected, for a reason worth stating plainly: most of these prompts are **machine-issued templates**,
so matching on them is closer to an identity test than a similarity test. False-positive risk is
near zero *for templated work*.

That caveat is load-bearing. This corpus belongs to a heavy agent-harness user, and its repeat
structure is harness-generated. A corpus of hand-typed human prompts would cluster far worse, and
nothing here measures how much worse.

## 2. Does the method reach back far enough? — **No. ~30 days.**

The task statement lives in the agent harness's on-disk transcript, which the harness rotates on its
own schedule. Availability, by month, over sessions tokenjam retained spans for:

| month | sessions in DB | with a recoverable task statement |
|---|---|---|
| 2026-05 | 42 | 0 (**0.0%**) |
| 2026-06 | 3,906 | 142 (**3.6%**) |
| 2026-07 | 2,188 | 2,186 (**99.9%**) |

This is fatal in a specific way. A before/after needs the **uninformed** runs, and the uninformed runs
are by definition the *old* ones — the ones that happened before anyone had written the lesson down.
Those are exactly the sessions whose task statement no longer exists. The high-confidence clusterer
can only see the informed side.

For the motivating project specifically: 22 sessions in May, **0 with a recoverable task statement**.

## 3. Can the span-only fallback substitute? — **No, and this is the load-bearing negative**

Spans are retained past transcript rotation, so a span-only "same work" similarity is the only thing
that reaches the uninformed era. The ticket proposed `cwd` + tool-sequence shape. That was scored
against the task-statement ground truth on the window where both exist (972 July sessions with ≥ 8
tool calls, 200,000 sampled session pairs). Baseline same-work pair rate: **0.147**.

| predicate | precision | recall | lift |
|---|---|---|---|
| identical opening-3 tools | 0.260 | 0.501 | 1.8× |
| identical opening-8 tools | 0.247 | 0.183 | 1.7× |
| tool-mix cosine ≥ 0.98 | 0.236 | 0.356 | 1.6× |
| tool-mix cosine ≥ 0.99 | 0.247 | 0.202 | 1.7× |
| **same project** (alone) | 0.383 | 0.244 | 2.6× |
| same project + opening-8 | 0.481 | 0.051 | 3.3× |
| same project + cosine ≥ 0.98 | 0.499 | 0.096 | 3.4× |
| **same project + opening-8 + cosine ≥ 0.95** | **0.538** | **0.043** | **3.7×** |
| same project + opening-12 + cosine ≥ 0.99 | 0.535 | 0.015 | 3.6× |

The best achievable operating point is **precision 0.54 at recall 0.04**: roughly **half** of the
pairs it calls "the same work" are not, while it captures 4% of the real repeats. Tightening the
predicate does not help — precision plateaus at ~0.54 and recall collapses.

The reason is mundane. Tool-call shape is dominated by *which tool the agent is holding*, not by
*what task it is doing*. Almost every coding-agent session looks like Bash, Read, Edit, Bash, Bash.

`repeat_task.TOOL_SHAPE_MATCH` carries this measured 0.54 as data, and
`measure_codification_delta` **refuses** to price any cluster derived from it.

### This also settles the `script` / `relearn` boundary

`analyzers/workflow_restructure.py` (the `script` analyzer) clusters sessions by ordered tool-call
signature — the same primitive measured above. So the table is a direct audit of `script`'s
clustering, and it says: **that primitive identifies a shape, not a task.**

That is not a bug in `script`, because a script replaces a *shape*. It is exactly the right unit for
"could a deterministic script do this". It is the wrong unit for "was this the same work", which is
why `relearn` cannot borrow it.

The two analyzers are **disjoint and should stay separate**:

| | `script` | `relearn` |
|---|---|---|
| recurrence detected | same tool-call **shape** | same **failure** signature |
| question | is this deterministic enough to replace with code? | is this recurring because a lesson isn't written down? |
| fix emitted | a script | a rule (CLAUDE.md / skill / hook) |
| unit | the session's shape | the failure episode |

They overlap only in that both are triggered by repetition. Neither subsumes the other, and
converging them would merge a shape-clusterer with a failure-clusterer that were measured here to be
answering different questions. **Recorded boundary: keep both; `relearn` must not adopt `script`'s
signature as a same-work test.**

## 4. Is the effect detectable above the noise? — **No, not at this effect size**

Even inside the tightest cluster obtainable — one machine-generated prompt template, one project —
per-session cost is extremely noisy:

| cluster | n | median $ | CV | p10 | p90 |
|---|---|---|---|---|---|
| ticket-worker × project A | 87 | 4.41 | 0.79 | 1.48 | 12.07 |
| ticket-worker × project B | 78 | 2.23 | 1.15 | 0.26 | 8.79 |
| ticket-worker × project C | 95 | 1.34 | 1.21 | 0.51 | 7.32 |
| ticket-worker × project D | 63 | 2.77 | 0.85 | 1.13 | 8.91 |
| interactive sessions × project B | 29 | 20.32 | 1.36 | 1.71 | 296.48 |

Coefficient of variation **0.7–1.2**, with p10→p90 spanning an order of magnitude. The 95% bootstrap
interval on the median at n ≈ 45 per side is **±46–49%**.

The effect being hunted is ~50% (40 min → under 20 min). **It sits at the noise floor of the best
cluster available.** Detecting it reliably needs either far larger n per side or a paired design that
holds task difficulty constant — and the residual variance here is task difficulty, since each run of
the "same" template does a different ticket.

## 5. The before/after test itself

Codification events = commits touching `CLAUDE.md` / `learnings.md` in each project (135 events in
the motivating project, spread over ~30 distinct days from 2026-05-28 to 2026-07-17 — codification is
**continuous and drip-fed**, not a single step change, which is itself a problem for a before/after
design).

Across the entire corpus there are **13** testable (cluster × codification-event) pairs with ≥ 12
sessions on both sides. Results:

| project | event | n before / after | median $ before → after | ratio | 95% CI | verdict |
|---|---|---|---|---|---|---|
| C | 2026-07-18 | 56 / 39 | 0.92 → 3.09 | 3.35 | [1.92, 4.83] | dearer |
| C | 2026-07-18 | 70 / 25 | 1.21 → 4.05 | 3.36 | [2.32, 6.28] | dearer |
| C | 2026-07-19 | 76 / 19 | 1.27 → 4.11 | 3.23 | [2.31, 6.55] | dearer |
| A | 2026-06-28 | 13 / 74 | 3.26 → 4.96 | 1.52 | [0.93, 2.10] | null |
| A | 2026-06-29 | 30 / 57 | 3.52 → 5.14 | 1.46 | [0.97, 2.34] | null |
| A | 2026-07-02 | 60 / 27 | 3.42 → 7.19 | 2.10 | [1.25, 2.84] | dearer |
| B | 2026-07-03 | 24 / 54 | 5.33 → 1.45 | 0.27 | [0.17, 0.48] | cheaper |
| B | 2026-07-10 | 26 / 52 | 4.87 → 1.45 | 0.30 | [0.17, 0.51] | cheaper |
| B | 2026-07-16 | 50 / 28 | 3.36 → 0.52 | 0.16 | [0.09, 0.44] | cheaper |
| A | 2026-07-16 | 24 / 39 | 2.83 → 2.77 | 0.98 | [0.54, 1.60] | null |
| B (interactive) | 3 events | ~13-17 / ~12-16 | — | 0.56–1.25 | e.g. [0.06, 5.88] | null |

**The same treatment produces ratios from 0.16× to 3.35×, with intervals excluding 1.0 in both
directions.** That is the signature of a confound, not an effect.

### The confound, identified

Model routing. The harness right-sizes the model per ticket, and that mix moved *hard* under every
comparison:

| project | week | median $ | model mix |
|---|---|---|---|
| B | 2026-W27 | 6.48 | opus 94% |
| B | 2026-W28 | 2.35 | sonnet 71%, opus 22%, haiku 6% |
| B | 2026-W29 | 1.40 | sonnet 36%, opus 35%, haiku 27% |
| C | 2026-W29 | 1.25 | opus 57%, sonnet 39% |
| C | 2026-W30 | 7.00 | opus 99% |

Project B's "codification saved 84%" is the model right-sizing rollout. Project C's "codification
cost 235% more" is the same mechanism running backwards. **5 of the 7 apparently-significant results
had the model mix change underneath them.**

### Controlling for it

Restricting to sessions ≥ 90% served by a single model (`repeat_task.model_mix_is_stable`):

| project | event | n before / after | ratio | 95% CI | verdict |
|---|---|---|---|---|---|
| C | 2026-07-18 | 25 / 33 | 2.26 | [1.51, 3.12] | dearer |
| C | 2026-07-18 | 34 / 24 | 2.36 | [1.50, 4.00] | dearer |
| C | 2026-07-19 | 40 / 18 | 2.19 | [1.12, 4.26] | dearer |
| **A** | 2026-06-28 | 13 / 73 | 1.48 | [0.91, 2.14] | **null** |
| **A** | 2026-06-29 | 30 / 56 | 1.52 | [0.96, 2.37] | **null** |
| **A** | 2026-07-02 | 59 / 27 | 2.17 | [1.26, 2.87] | **dearer** |
| **A** | 2026-07-16 | 21 / 37 | 0.89 | [0.51, 1.55] | **null** |
| B | 2026-07-03 | 22 / 22 | 0.40 | [0.24, 0.75] | cheaper |
| B | 2026-07-10 | 24 / 20 | 0.45 | [0.25, 0.88] | cheaper |
| B | 2026-07-16 | 30 / 14 | 0.37 | [0.21, 0.78] | cheaper |

**Project A is the motivating project — the known-positive the method exists to reproduce. It comes
back `null` or `dearer` at every testable codification event**, with up to 144 model-matched sessions.
The gate prices it at **$0.00**.

The signs still disagree across projects for the same treatment, so the surviving B/C results are
residual confounding (ticket-difficulty drift, harness version), not a demonstrated effect.

## 6. What about the original field observation?

It is real as a *project-level trend*, and it is not attributable to codification. Median active
session minutes for the motivating project:

| week | n | median active min | p25 | p75 | median $ |
|---|---|---|---|---|---|
| 2026-W22 | 16 | 111.4 | 9.6 | 812.4 | 21.03 |
| 2026-W23 | 71 | **37.5** | 9.5 | 315.4 | 22.73 |
| 2026-W24 | 76 | **45.1** | 11.8 | 255.5 | 24.29 |
| 2026-W26 | 13 | 30.2 | 1.6 | 122.3 | 2.39 |
| 2026-W27 | 12 | **16.7** | 3.0 | 454.6 | 3.20 |
| 2026-W29 | 15 | 29.5 | 4.0 | 141.4 | 1.64 |

37.5–45.1 min falling to 16.7–29.5 min matches the recollection closely. But:

- The interquartile range is enormous (9.5 → 315 min in W23). The median is drifting inside noise an
  order of magnitude wider than the drift.
- Median cost fell **10×** ($22.73 → $1.64) over the same span. No plausible codification effect is
  10×.
- What actually changed is **what a "project A session" is**: long human-driven interactive sessions
  in W22–W24 gave way to short headless ticket-worker sessions from W26 on. That is a change of task
  mix, not a change in the informedness of a fixed task.
- There is no repeat-task cluster spanning the boundary, because W22–W24 transcripts are gone.

The observation is consistent with codification helping. It is equally consistent with three other
explanations, and this corpus cannot separate them.

## Verdict

**The uninformed-vs-informed signal cannot be measured on this corpus at a confidence that would
support pricing.** Specifically:

1. Repeat-task clusters are identifiable — but only for ~30 days, and the uninformed side is older.
2. The only similarity method that reaches further tops out at precision 0.54. Half its clusters
   would be wrong, in an unknown direction, behind a causal claim.
3. Within-cluster cost noise (CV 0.7–1.2) puts the ±47% detection floor right at the ~50% effect size.
4. Every apparently-significant delta is explained by model routing; controlling for it leaves the
   known-positive project at null.

Shipping a session-level relearn rework on this basis would produce **confident nonsense**, which is
worse than $46.26. The session-inflation hypothesis is not refuted — it is **unfalsifiable with
current instrumentation**.

## What this does to the write-budget verdict

The write budget suppresses a proposed rule when its modelled standing cost exceeds its modelled
gross saving — recorded as net-negative for 18 of 25 clusters, 6–24× underwater. The hope was that a
measured before/after would either validate or overturn that model.

**It does neither, and that is the honest status.** The measurement that could adjudicate it is not
obtainable here. Concretely:

- **Not overturned.** There is no measured evidence that codification saves more than the model
  credits it for. In the one project where a saving was expected, the measured result is null.
- **Not validated.** The model's blind spot is real and unaddressed: it counts what a rule costs to
  carry and never counts what it saves by preventing a flail. This measurement does not fill that
  gap; it shows the gap cannot be filled by observation alone at present.
- **Practical consequence:** keep the current suppression rule, and stop describing it as
  empirically settled in either direction. It is a model, and it remains one.

## What would make this measurable

Not more analysis of this corpus — different instrumentation. In rough order of value:

1. **Persist the task statement at ingest.** A normalized, masked task-statement hash written to the
   session row at ingest survives transcript rotation and would extend the high-confidence clusterer
   backwards indefinitely. This is the single highest-value change and it is cheap.
2. **Stamp the codification event on the session.** Sessions already resolve a project `cwd`;
   recording which CLAUDE.md/learnings.md revision was in effect turns the before/after from a
   date-join into an exact attribute, and handles drip-fed codification correctly.
3. **Record the routed model on the session row.** The confound that dominated every result here has
   to be a first-class control, not something reconstructed from span aggregation.
4. **Only then** revisit the session-level unit, on repeat clusters that straddle a *specific*
   lesson rather than a project-wide commit stream.

Until at least (1) and (3) exist, `relearn` should keep pricing failure episodes. The $46.26 is
small, but it is *measured*, and the alternative on offer is not.
