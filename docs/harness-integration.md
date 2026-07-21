# Harness integration — grouping a fan-out run

A *harness* (a governor / launcher / orchestrator) spawns many agent sessions to
do one job: a backlog of tickets, a fleet of workers, a research sweep. tj can
show that whole **run** as one linked thing on the dashboard — its workers, their
cost, how far each got — instead of a scatter of disconnected sessions.

This page explains how that linkage works, the one thing a harness does to opt
in, and the easiest way to set it up (an MCP command). It also states tj's honest
boundary: tj links only work that is **recorded and correlated** — it never
reverse-engineers a harness or guesses silently.

## The one idea: a shared `tokenjam.run_id`

tj groups sessions into a run by a single OTel **resource attribute**,
`tokenjam.run_id`, stamped on the launcher **and** every worker it spawns. Same
id → same run. That's the whole contract. It is declared by the spawner (exactly
like a trace id in distributed tracing), never inferred from behavior.

Once present, the run shows up automatically:
- the **Runs** view (`#/runs/<run_id>`) rolls up every member session, and
- a worker's or launcher's **Map** shows a *run card* linking to the others.

## Three tiers (most harnesses do nothing)

You do **not** route every harness through this doc. Pick the lowest tier that
already covers you:

| Tier | Who it covers | What you do |
|---|---|---|
| **1 — automatic** | Harnesses that spawn via the Claude Code **Task tool** | Nothing — tj already maps the subagent tree |
| **2 — convention** | Harnesses that spawn **separate `claude` processes / containers / API sessions** | Stamp `tokenjam.run_id` on the launcher + every worker (below) |
| **3 — inference** | Harnesses that **announce a run id** in their output (e.g. a governor printing `TokenJam run id: gov-…`) | Nothing — tj scrapes the id from the launcher's transcript and confirms it against real run data (a labeled best-effort guess) |

Tier 2 is the scalable answer: tj stays agnostic to *how* you spawn work; your
harness conforms to one thin correlation id.

## Easiest setup: the `setup_harness` MCP command

`tj onboard --claude-code` does **not** wire the MCP server (an in-loop MCP is a
per-turn token tax on subscription users, +36% measured) — but this one-time setup
helper is a legitimate reason to wire it temporarily. Run
`claude mcp add tj --scope user -- tj mcp`, then from **inside your harness
repo** in Claude Code:

- **`setup_harness(mode="instrument")`** — writes a drop-in helper
  (`.tj/run-env.sh`) that mints one run id per launch and exports
  `tokenjam.run_id`, then reports the spawn points it found and the exact one-line
  wiring to add to your launcher. It describes the change and writes only its own
  helper file — it never edits your harness's code. Claude then wires
  `source .tj/run-env.sh` into the launcher for you.
- **`setup_harness(mode="map")`** — makes no changes; scans your harness structure
  and reports how tj already groups its sessions (existing runs) plus a
  recommendation.

After wiring, launch a run and confirm the workers grouped in the Runs view (or
re-run `mode="map"`). Once `setup_harness` has done its job, deregister the MCP
server (`claude mcp remove tj --scope user`) to avoid paying the per-turn tax
on unrelated sessions — the `.tj/run-env.sh` helper it wrote keeps working
without it.

## Wiring it by hand

The helper does this; you can also do it directly. In the **launcher**, before it
spawns any worker, export a per-launch run id so every spawned process inherits
it:

```bash
# once, at the top of the launcher — workers inherit this via the environment
if [ -z "${TJ_RUN_ID:-}" ]; then
  TJ_RUN_ID="run-$(date -u +%Y%m%dT%H%M%SZ)-$$"
fi
export TJ_RUN_ID
export OTEL_RESOURCE_ATTRIBUTES="${OTEL_RESOURCE_ATTRIBUTES:+${OTEL_RESOURCE_ATTRIBUTES},}tokenjam.run_id=${TJ_RUN_ID}"
```

Python launcher equivalent:

```python
import os, time
run_id = os.environ.setdefault(
    "TJ_RUN_ID", f"run-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}")
attrs = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")
if "tokenjam.run_id=" not in attrs:
    os.environ["OTEL_RESOURCE_ATTRIBUTES"] = (attrs + "," if attrs else "") + f"tokenjam.run_id={run_id}"
```

Notes:
- Make the id **per launch**, not a constant baked into `.claude/settings.json` —
  otherwise every run reuses one id and they all collapse together.
- If a worker's `.claude/settings.json` hard-sets `OTEL_RESOURCE_ATTRIBUTES`, add
  `tokenjam.run_id=${TJ_RUN_ID}` there too so it isn't overwritten.
- Optionally also stamp `tokenjam.parent_session_id` (the spawning session's id)
  for nested spawns; the Runs view uses it to render a parent tree.

## The honest boundary

tj maps structure only as far as work is **instrumented and correlated**:
- Task-tool subagents (tier 1) are recorded by Claude Code itself.
- A shared `tokenjam.run_id` (tier 2) groups anything, regardless of spawn
  mechanism.
- Inference (tier 3) is a labeled guess from an announced id.

Untagged, uninstrumented workers are **invisible** — no method recovers them. tj
reports what it can see and says so; it never fabricates a run.
