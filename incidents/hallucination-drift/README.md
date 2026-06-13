# My agent worked yesterday. Today it's possessed.

**Run it:** `pipx install tokenjam && tj demo hallucination-drift`

---

## The failure with no error message

Your coding agent has been solid for two weeks. Same prompts, same repo, same everything. Then on Tuesday the outputs look... off. Longer. Different variable names. Tool calls you've never seen before.

You ask the agent why. It explains confidently. The explanation sounds plausible. You can't prove anything is wrong.

This is the worst kind of bug. No stack trace. No error. No crash. Just behavior that used to be one thing and is now quietly something else. You don't have a baseline. You don't have numbers. You have a feeling, and a feeling is not a measurement.

## What you see with print()

```
[agent] Session 1... output looks reasonable
[agent] Session 2... output looks reasonable
[agent] Session 3... output looks reasonable
[agent] Session 4... output looks reasonable
[agent] Session 5... output looks reasonable
[agent] Session 6... output looks... different?
[agent] Hmm, that response was longer than usual.
[agent] But hey, it completed successfully. Moving on.
```

## What you see with TokenJam

```
Sessions: 5 baseline + 1 anomalous
Spans ingested: 33

Alerts fired:
  ALERT drift_detected

The anomalous session had:
  Input tokens: 50,000 vs baseline ~1,000 (Z-score: inf)
  Tool sequence: 5 new tools never seen in baseline
```

Five sessions averaged 1,000 input tokens with tools `[search, summarize]`. Session 6 came in with 50,000 tokens and tools `[fetch_url, parse_html, extract_entities, classify, store_results]`. Every metric was off the chart. TokenJam fired `drift_detected` the moment the session closed.

The `DriftDetector` builds a rolling baseline from prior sessions. When a new session's token counts exceed a Z-score of 2.0, or the tool sequence Jaccard distance exceeds 0.4 — it fires. You find out in seconds, not after a week of "huh, that output seemed weird."

## Enable drift detection

In `tj.toml`:

```toml
[agents.my-agent.drift]
enabled            = true
baseline_sessions  = 10    # how many sessions to learn from
token_threshold    = 2.0   # Z-score to trigger on
tool_sequence_diff = 0.4   # Jaccard distance to trigger on
```

The demo uses `baseline_sessions = 5` for speed. In production, 10–50 sessions gives a more stable baseline so normal variance doesn't get flagged.

## Try it yourself

```bash
pipx install tokenjam
tj demo hallucination-drift
```

Runs entirely in-process. No API keys, no real model calls, no network traffic.

To track drift on your real agent, wire up the TokenJam SDK, enable drift in `tj.toml`, and run `tj serve`. Then `tj drift` shows Z-scores; `tj alerts` shows the events.

## Next in the incident library

- [Your agent isn't flaky. You're blind.](../retry-loop/README.md)
- [Why did my agent just spend $47 on a hello world?](../surprise-cost/README.md)

---

[TokenJam](https://github.com/Metabuilder-Labs/TokenJam) is a local-first, zero-signup observability CLI for AI agents. No cloud. No account. Just `pipx install tokenjam` and start seeing what your agent actually does.
