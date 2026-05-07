---
title: The retry loop is the most common agent failure I see, and `print()` will never catch it
published: false
tags: ai, python, devops, productivity
series: Agent Incident Library
---

I work on [TokenJam](https://github.com/Metabuilder-Labs/TokenJam), an open-source observability tool for AI agents. A lot of what I do is stare at other people's agent traces — the ones their print logs say are fine and their users say are slow.

The single most common pattern I see is the silent retry loop. It looks like this:

```
[tool] search_knowledge_base called
[tool] search_knowledge_base returned: null
[tool] search_knowledge_base called
[tool] search_knowledge_base returned: null
[tool] search_knowledge_base called
[tool] search_knowledge_base returned: null
[tool] search_knowledge_base called
[tool] search_knowledge_base returned: null
```

Same call, same input, same null. Four times in a row.

Nothing here is technically an error. The HTTP status is 200. The tool ran. The model decided to call it again. From the log's perspective, this is four successful operations. From the user's perspective, the agent is hung.

This is why people say agents are "flaky" — there's no error to grep for, just behavior that doesn't terminate. And `print()` will never tell you, because each line in isolation is correct. The pathology is in the *sequence*, and a flat log file has no concept of sequence beyond timestamps.

---

When I designed the `retry_loop` detector for OCW, the rule I landed on was deliberately boring: fire when the same tool name shows up 4+ times in the last 6 spans. No ML, no per-agent tuning. Most real loops are tighter than that — they're 6+ identical calls in a row — so 4-of-6 catches them early without false positives on legitimate retries.

It runs alongside `failure_rate`, which trips when more than 20% of recent spans error out. Both default-on. Together they cover the two flavors of "stuck": looping on success and looping on failure.

```
Alerts fired:
  ALERT retry_loop
  ALERT failure_rate
```

Visible from span 4. No threshold tuning. No dashboard.

---

I'm not arguing the agent is doing something wrong here. Tools return `null`. APIs go down. An agent retrying when it gets nothing back is reasonable behavior in isolation — the bug is that it has no termination condition for *silence*, only for errors. Fixing that is a prompt-engineering problem.

But you can't fix what you can't see, and the reason I built this detector is that the typical observability path for agents is: ship with `print()`, get a vague "it's slow" report, restart the process, blame the upstream, ship again. The loop never gets diagnosed because nothing in the workflow surfaces it.

---

The demo reproduces the failure end-to-end with no API keys and no setup:

```bash
pip install tokenjam
tj demo retry-loop
```

It synthesizes the span sequence above, runs both detectors against it, and shows you the `print()` view next to the OCW view. About 30 seconds.

---

To wire it into a real agent, the SDK is three lines:

```python
from ocw.sdk import patch_anthropic, watch

patch_anthropic()

@watch(agent_id="my-agent")
def run():
    ...  # your existing code, unchanged
```

Run `tj serve` in the background. `tj alerts` shows what fired. `tj traces` shows the full span waterfall. Local DuckDB, no cloud, no signup.

---

The framing I keep pushing back on is "you can't trust agents in production." That's two different statements collapsed into one. There's a real difference between an agent that retried four times because a tool returned null, and an agent that retried four times for no reason anyone can reconstruct. The first is a fixable infrastructure problem. The second is a monitoring gap masquerading as a reliability problem.

Most of the agents I see have the second problem. Once you can replay the span sequence, the first problem becomes a normal engineering ticket.

`tj demo retry-loop` — give it 30 seconds, see the alert fire.

---

*Part of the [Agent Incident Library](https://github.com/Metabuilder-Labs/TokenJam) — reproducible scenarios for the failures that don't show up in your logs.*
