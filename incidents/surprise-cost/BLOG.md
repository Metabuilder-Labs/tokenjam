---
title: Three of my agent's API calls were Opus. My logs said "200 OK" eight times.
published: false
tags: ai, python, devops, productivity
series: Agent Incident Library
---

If you run a multi-agent workflow — LangChain with fallbacks, CrewAI with different models per agent, AutoGen, or anything where someone (maybe past-you) configured model routing — this post is for you.

Here's what the logs showed:

```
[agent] Starting document analysis...
[llm] Response received (200 OK)
[llm] Response received (200 OK)
[llm] Response received (200 OK)
[llm] Response received (200 OK)
[llm] Response received (200 OK)
[llm] Response received (200 OK)
[llm] Response received (200 OK)
[llm] Response received (200 OK)
[agent] Task complete.
```

Eight successes. Nothing to investigate.

Here's what actually happened:

```
Model               Calls    Cost (USD)
─────────────────────────────────────
claude-opus-4-6       3       $3.2325
claude-sonnet-4-6     3       $0.2775
claude-haiku-4-5      2       $0.0092
─────────────────────────────────────
Total                          $3.5192
```

Three calls to Opus. 92% of the bill. The `model=` config said Haiku. A fallback router in the chain was escalating harder subtasks — exactly as configured, two weeks ago, by someone who then forgot.

`print()` has no way to tell you which model handled which call. HTTP responses don't include "by the way, this one cost $1.20." OCW does.

---

This happens whenever:

- A LangChain fallback escalates to a stronger model on error or complexity
- A CrewAI crew has different models per agent and you've lost track
- A config override somewhere in your stack that past-you set

The per-session cost looks fine until it compounds. $3.52 per session × 3 sessions/day × 20 working days = **$211/month** on a workflow you thought cost $20.

---

See it in 30 seconds, no API keys:

```bash
pip install tokenjam
tj demo surprise-cost
```

8 synthetic LLM spans with real pricing math — same model mix, same token counts as the real scenario. Side-by-side: what `print()` shows vs. what OCW reveals.

---

Wire up your real agent:

```python
from tokenjam.sdk import patch_anthropic, watch

patch_anthropic()

@watch(agent_id="my-agent")
def run():
    ...  # your existing code unchanged
```

Set a budget cap:

```toml
# tj.toml
[agents.my-agent.budget]
session_usd = 5.00
```

OCW fires an alert when you cross it. Not on the bill. When the call happens.

---

The cost isn't the problem. Invisibility is the problem. Once you can see which model ran which call, the budget conversation becomes a technical decision instead of a 2am surprise.

`tj demo surprise-cost` — run it, see what was hiding.

---

*Part of the [Agent Incident Library](https://github.com/Metabuilder-Labs/TokenJam)*
