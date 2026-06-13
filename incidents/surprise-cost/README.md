# Why did my agent just spend $47 on a hello world?

**Run it:** `pipx install tokenjam && tj demo surprise-cost`

---

## The invisible upgrade

You built a document analysis agent on Claude Haiku. Cheap, fast, good enough. You tested it. $0.003 per run. Shipped it.

A week later, your billing dashboard says $47 for a single session.

You check the code. The model parameter says `claude-haiku-4-5`. You check again. Still Haiku. You add a print statement:

```
[llm] Response received (200 OK)
[llm] Response received (200 OK)
[llm] Response received (200 OK)
```

Eight responses. All 200. Nothing wrong. Except three of those calls weren't Haiku. Somewhere in your chain — a fallback handler, a LangChain router, a config override you forgot about — the model silently escalated to Opus. Haiku costs fractions of a cent. Opus costs dollars. And nobody told you.

## What you see with print()

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

Eight successes. No errors. Nothing to investigate.

## What you see with TokenJam

```
 Model               Calls   Cost (USD)
────────────────────────────────────────
 claude-opus-4-6         3      $3.2325
 claude-sonnet-4-6       3      $0.2775
 claude-haiku-4-5        2      $0.0092

Total session cost: $3.5192
```

Two Haiku calls: $0.009. Three Opus calls: $3.23. You paid 350x more for Opus than Haiku, in the same session, and `print()` gave you eight identical lines.

TokenJam records `model`, `input_tokens`, and `output_tokens` on every LLM span. The `CostEngine` prices each call using per-model rates from `pricing/models.toml`. The escalation is visible the moment it happens — not at the end of the month on a bill.

## Set a budget before it happens

Add to `tj.toml`:

```toml
[agents.my-agent.budget]
session_usd = 1.00   # alert if a single session exceeds this
daily_usd   = 5.00   # alert if daily spend exceeds this
```

Or set a global default for all agents:

```toml
[defaults.budget]
daily_usd = 10.00
```

TokenJam fires `cost_budget_session` and `cost_budget_daily` alerts when limits are crossed.

## Try it yourself

```bash
pipx install tokenjam
tj demo surprise-cost
```

Emits 8 synthetic LLM spans with real pricing math. No API keys, no model calls, no network traffic.

To track real spend, instrument your agent with the tokenjam SDK and run `tj serve`. Then `tj cost --by model` shows live per-model attribution.

## Next in the incident library

- [Your agent isn't flaky. You're blind.](../retry-loop/README.md)
- [My agent worked yesterday. Today it's possessed.](../hallucination-drift/README.md)

---

[TokenJam](https://github.com/Metabuilder-Labs/TokenJam) is a local-first, zero-signup observability CLI for AI agents. No cloud. No account. Just `pipx install tokenjam` and start seeing what your agent actually does.
