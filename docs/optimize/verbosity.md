# Verbosity

Product name: **Verbosity**. Internal/CLI name: `verbosity`.

```bash
tj optimize verbosity
```

The other analyzers target **input** and model choice. Verbosity is the one
that looks at **output** — sessions burning tokens on answers that run long for
the work they did. It flags a session only when its output is a clear outlier
against sessions doing the *same shape* of work, then surfaces a brevity remedy
for you to measure.

## The weakest-grounded analyzer — treated that way

Output length is not waste. A terse answer can drop information the task needed,
and "too verbose" is task-dependent — so verbosity's recoverable figure is the
least defensible of any analyzer. Two consequences are baked in:

- The baseline is the **per-`(tool, arg-shape)` median** — the only signal
  grounded in like-for-like tasks. A high output:input ratio is carried as a
  descriptive field only; it never flags a session on its own (a legitimately
  long answer to a short prompt is not waste).
- The recoverable figure is a **soft upper bound**, and `--validate` (measuring
  a brevity constraint on your own calls) is load-bearing, not optional, before
  anyone claims a dollar saving.

## What it flags

Sessions are grouped into cohorts by their task-shape signature — the
`(tool_name, arg_shape)` sequence, the same signature the [Script](script.md)
analyzer uses. For each cohort:

| Step | Rule |
|---|---|
| Cohort must be real | At least `MIN_COHORT_SESSIONS` (5) sessions share the signature, else its median is noise and the cohort is skipped. |
| Baseline | The cohort's **median** output tokens. |
| Flag | A session whose output exceeds `HIGH_VERBOSITY_MULTIPLE` (2.0×) the cohort median. |

Thresholds live as module constants in
`tokenjam/core/optimize/analyzers/output_verbosity.py`
(`MIN_COHORT_SESSIONS`, `HIGH_VERBOSITY_MULTIPLE`, `MAX_EXAMPLES`).

It reads aggregate token counts and the tool-call shape — no prompt/completion
content capture required.

## Output

The finding reports, for the window:

- `total_candidates` — the true number of flagged sessions
- `candidates` — the top `MAX_EXAMPLES` (5) by over-baseline output; when more
  were flagged the CLI shows "showing top 5 of N" so the headline never
  under-reports
- per candidate: `output_tokens`, `baseline_output_tokens` (the cohort median),
  `over_baseline_tokens`, `over_baseline_multiple`, and the descriptive-only
  `output_input_ratio`
- `suggested_max_tokens` — the cohort baseline, an advisory `max_tokens` cap to
  review (never applied)
- `sessions_examined` / `cohorts_examined`

Rendering follows the same plan-tier-aware convention as the rest of
`tj optimize`: `api` plans see the dollar figure, subscription/local/unknown
plans see the over-baseline token figure instead.

## Estimate basis / confidence

`estimated_recoverable_tokens` is the over-baseline output summed across flagged
sessions; `estimated_recoverable_usd` prices it at **output** rates.
`estimate_confidence` is `"heuristic"` and `estimate_basis` reads:

> output tokens above the per-task-shape median, priced at output rates — a
> SOFT upper bound, not a measured saving: a brevity constraint can be
> net-negative once its own overhead is counted, so measure before claiming

`confidence` on the finding is `structural` (Rule 14). The mandatory caveat,
surfaced in every render mode:

> Predicted high-verbosity output — review before constraining a response.
> Output length is not waste: a terse answer can drop information the task
> needed. This is a candidate to look at, never a claim you are wasting tokens.
> Measure a brevity constraint before applying it.

## Remedy — surfaced, not applied

The finding carries a terse system-prompt snippet and a suggested `max_tokens`
cap (the cohort baseline). Both are advisory strings — you apply them and
measure the result. Enforcing a brevity policy fleet-wide is a Cloud concern,
not something the OSS analyzer does.

## See also

- [Trim](trim.md) — the input-side counterpart: low-significance tokens in captured prompts
- [Downsize](downsize.md) — cheaper-model candidates by session shape
- [Script](script.md) — the same `(tool_name, arg_shape)` signature, used to find deterministic workflows
