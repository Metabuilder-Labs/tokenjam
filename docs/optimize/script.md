# Script

Product name: **Script**. Internal/CLI name: `script`.

```bash
tj optimize script
```

Flags sessions whose tool-call sequence is structurally identical
across many runs — strong signal that a deterministic shell script
could replace the agent for that workload, saving 100% of the cost on
that pattern.

## Signature definition

A session's signature is the ordered tuple of `(tool_name, arg_shape)`
pairs across its tool spans.

`arg_shape` is the **type** of each argument, not the value:

| Category | Matches |
|---|---|
| `file_path` | strings starting with `/`, `~`, `.`, or matching `[A-Z]:\` |
| `command_string` | strings starting with a known shell command (`git`, `npm`, `pytest`, etc.) |
| `json_object` | dict values |
| `array` | list values |
| `number`, `boolean` | scalar primitives |
| `string` | generic string |

Argument **keys** are sorted before signature construction, so dict
iteration order doesn't change the result.

## Why structural shape, not values

Two sessions can have identical structural shape but different argument
values. Consider a "deploy staging" pattern:

```
session-A: bash("git pull origin staging") → bash("npm install") → bash("pm2 restart")
session-B: bash("git pull origin prod")    → bash("npm install") → bash("pm2 restart")
```

Both sessions have the signature `[(bash, command_string), (bash, command_string), (bash, command_string)]`.
The values differ but the *pattern* is the same — and the pattern is
what would map cleanly to a parametrised shell script.

If we clustered by values, every session that touches a different file
would land in its own cluster and the analyzer would find nothing. If
we ignored args entirely, sessions that happen to share tool names but
differ structurally (e.g. one uses paths, another uses JSON blobs)
would get merged incorrectly.

The arg_shape compromise captures structural shape without false-merging
on value variation. It's the right granularity for "is this a script?"

## Thresholds

A cluster must have **≥20 sessions** with identical signature before it's
flagged. v1 errs hard on the side of false negatives — surfacing one
false-positive recommendation that the user investigates and rejects
erodes trust faster than missing a real opportunity.

Across-cluster branching (sessions that look "almost like" the cluster
but deviate) isn't surfaced in v1 — those sessions form their own
clusters and contribute to neither the flagged cluster nor any signal
about variation. Fuzzy clustering is a future research project.

## Degraded mode

Without `[capture] tool_inputs = true` the analyzer can't extract
`arg_shape` from the captured tool input. In that case it degrades to
clustering by tool-name sequence only and marks the finding as
`degraded: true` so the renderer can disclose the weaker signal.

Even degraded, the recommendation can be useful — a session that calls
`(bash, bash, bash, pm2)` 23 times is probably a deployment pattern
regardless of the exact argument shapes. The user should still review.

## Confidence

`structural` with an explicit caveat about reviewing each cluster before
replacing with a script. The analyzer doesn't claim quality equivalence;
it claims structural identity, which is a different (and weaker) thing.

## See also

- [Downsize](downsize.md) — flag sessions whose shape matches a cheaper-model candidate
- [Cache](cache.md) — measure and improve prompt-cache usage
- [Trim](trim.md) — identify low-significance tokens in captured prompts
- [Subagent](subagent.md) — per-subagent cost breakdown and right-sizing candidates (Claude Code only)
