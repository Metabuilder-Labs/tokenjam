# Your agent isn't flaky. You're blind.

**Run it:** `pipx install tokenjam && tj demo retry-loop`

---

## The bug that isn't a bug

Wednesday, 2pm. Users are saying the agent is "slow." You check the logs.

```
[tool] search_knowledge_base called
[tool] search_knowledge_base returned: null
[tool] search_knowledge_base called
[tool] search_knowledge_base returned: null
```

Four times. Five. You restart. It goes away. You blame the API provider, file a mental note, move on.

Here's what actually happened: the tool was returning `null` — not an error, just nothing. Your agent didn't know how to interpret silence, so it asked again. And again. Five identical calls in a row. Each one burned tokens and latency on a question that was never going to get answered.

The logs said "tool called, tool returned." They were right. They just didn't tell you anything useful.

## What you see with print()

```
[agent] Starting task...
[tool] search_knowledge_base called
[tool] search_knowledge_base returned: null
[tool] search_knowledge_base called
[tool] search_knowledge_base returned: null
[tool] search_knowledge_base called
[tool] search_knowledge_base returned: null
[tool] search_knowledge_base called
[tool] search_knowledge_base returned: null
[agent] Retrying...
```

Technically correct. Completely useless.

## What you see with TokenJam

```
Spans ingested: 6
Traces: 1

Alerts fired:
  ALERT failure_rate
  ALERT retry_loop
  ALERT retry_loop
```

Two rules tripped automatically. `retry_loop` fires when the same tool appears 4+ times in the last 6 spans. `failure_rate` fires when more than 20% of recent spans error out. No configuration. No threshold tuning. On by default.

The loop was visible from span #4. Your logs didn't surface it until a user complained.

## Try it yourself

```bash
pipx install tokenjam
tj demo retry-loop
```

30 seconds, no API keys, no config file. The demo runs against an in-memory backend so nothing persists to disk.

To catch this in your real agent, wire up the TokenJam SDK (`@watch()` + `patch_anthropic()` or `patch_openai()`) and run `tj serve` in the background. After that, `tj alerts` and `tj traces` work against your live data.

## Next in the incident library

- [Why did my agent just spend $47 on a hello world?](../surprise-cost/README.md)
- [My agent worked yesterday. Today it's possessed.](../hallucination-drift/README.md)

---

[TokenJam](https://github.com/Metabuilder-Labs/TokenJam) is a local-first, zero-signup observability CLI for AI agents. No cloud. No account. Just `pipx install tokenjam` and start seeing what your agent actually does.
