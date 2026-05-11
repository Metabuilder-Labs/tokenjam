---
title: My agent worked yesterday. Today it's possessed.
published: false
tags: ai, python, devops, productivity
series: Agent Incident Library
---

Two weeks of clean runs. Same prompts, same repo, same results.

Then Tuesday happened.

The outputs were longer. Different variable names. Tool calls you'd never seen before. You asked the agent about it. It explained confidently. The explanation sounded plausible.

No stack trace. No error. No crash. Just behavior that used to be one thing and is now quietly something else.

This is the hardest failure to diagnose because you have nothing to point at. You have a feeling. A feeling is not a measurement.

---

Here's what five baseline sessions looked like:

```
Session 1: ~1,000 tokens | tools: [search, summarize]
Session 2: ~1,000 tokens | tools: [search, summarize]
Session 3: ~1,100 tokens | tools: [search, summarize]
Session 4:   ~950 tokens | tools: [search, summarize]
Session 5: ~1,050 tokens | tools: [search, summarize]
```

Here's session 6:

```
Session 6: 50,000 tokens | tools: [fetch_url, parse_html, extract_entities, classify, store_results]
```

Five new tools. 50x the tokens. Every metric off the chart.

Your `print()` logs said: *output looks reasonable. Moving on.*

TokenJam fired `drift_detected` the moment the session closed.

---

The `DriftDetector` builds a rolling baseline from prior sessions. When a new session's token counts exceed a Z-score of 2.0, or the tool sequence diverges past a Jaccard distance of 0.4 — it fires. No manual baseline to set up. No dashboard to configure. It learns from your agent's own history.

You find out in seconds. Not after a week of "huh, that seemed weird."

---

```bash
pip install tokenjam
tj demo hallucination-drift
```

No API keys. Runs entirely in-process. Watch 5 normal sessions, then 1 anomalous one, then the alert.

Enable it for your real agent:

```toml
# tj.toml
[agents.my-agent.drift]
enabled            = true
baseline_sessions  = 10
token_threshold    = 2.0
tool_sequence_diff = 0.4
```

Then `tj drift` shows Z-scores per session. `tj alerts` shows when the threshold was crossed.

---

The take that makes people mad: *"LLMs are non-deterministic — you can't test them."*

You're right. You can't test them the way you test functions. But you can measure them. You can build a baseline and alert when behavior leaves it.

Testing asks "is this correct?" Drift detection asks "is this different from how it's always behaved?" The second question is answerable. It just requires keeping score.

`tj demo hallucination-drift` — run it, see what keeping score looks like.

---

*Part of the [Agent Incident Library](https://github.com/Metabuilder-Labs/TokenJam)*
