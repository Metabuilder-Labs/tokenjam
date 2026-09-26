# The first hour

Once spans are flowing (see [docs/getting-started.md](getting-started.md) for how to get there),
the question changes from "does this work" to "what do I do with it." Here's the path, in order.

## 1. `tj status` — confirm what's arriving

```bash
tj status
tj status --agent my-agent
```

Current session info, cost, token counts, and any active alerts for the agent(s) you've wired up.
This is the fastest way to see that real usage is landing, before you go looking at anything deeper.

## 2. Lens — the dashboard

```bash
tj serve
```

Opens Lens at `http://127.0.0.1:7391/` — a **Dashboard** view that lands you on recoverable waste and
health at a glance, with an embedded explorer to slice usage any way (metric × dimension × chart),
plus dedicated Status, Traces, Cost, Analytics, Alerts, Drift, Optimize, and Budget screens. Fully
offline, plan-tier-aware, no signup. Spend a few minutes here before running the analyzers below —
Lens will often point you at which analyzer is worth running first.

## 3. `tj optimize` — find the savings

```bash
tj optimize                          # every analyzer your setup can act on
tj optimize downsize cache reuse     # just the ones you care about
```

Runs the cost-optimization analyzers (Downsize, Cache, Cache-recommend, Resend, Script, Trim,
Reuse, Subagent right-sizing, Summarize, Verbosity, Deadweight, Relearn, Stream-usage,
Budget-projection) against your real usage and surfaces candidates — never a guaranteed saving,
always a "worth a look." See [docs/optimize/](optimize/) for what each analyzer looks for and how to
read its output.

## 4. `tj optimize shipped` — what the spend produced

```bash
tj optimize shipped
```

The one analyzer that is not about spending less. It joins your sessions to the commits they
produced and reports how many shipped to the default branch, what the sessions that shipped nothing
cost, and how much of that work was rewritten within two weeks. Every dollar in it is measured
spend, not a saving, and a session can ship value without a commit. Read it as a question to ask.
[docs/optimize/shipped.md](optimize/shipped.md) covers the finding;
[docs/ledger/overview.md](ledger/overview.md) covers the join behind it.

## 5. `tj tokenmaxx` — the shareable summary

```bash
tj tokenmaxx
tj tokenmaxx --since 7d
```

A shareable spend-tier callout paired with the downsize savings figure — the one-line version of
"here's roughly where I stand" for a standup or a PR description.

---

That's the loop: check status, look at Lens, run the analyzers, share the summary. From here,
[docs/alerts.md](alerts.md) covers setting up ongoing notifications instead of checking manually, and
the [Documentation table in the README](../README.md#documentation) links every deeper reference.
