# `tj optimize --finding prompt-bloat` (Trim)

User-facing product name: **Trim**. Internal/CLI name: `prompt-bloat`.

Scores token-by-token significance in captured prompts using
LLMLingua-2 (BERT-class classifier, MIT-licensed, runs on CPU).
Identifies long, low-significance regions the model likely doesn't use,
then surfaces them for review and manual editing.

The Trim analyzer never auto-rewrites prompts. The honesty constraint
is strict: aggressive compression breaks tasks in surprising ways. The
report says "this region looks unimportant" — the user investigates and
decides.

## Installation

LLMLingua-2 pulls in PyTorch and transformers (~2GB). Kept out of the
base install:

```bash
pip install "tokenjam[bloat]"
```

The base `pip install tokenjam` does NOT pull torch. Trim shows up in
`tj optimize --finding` choices regardless, but running it without the
extra prints a clear install hint and exits.

## Requirements

- `[capture] prompts = true` in your config. Trim needs captured prompt
  text to score. Without it, the analyzer prints a hint pointing at
  the config flag.
- LLMLingua-2's model (~110MB) downloads on first run and caches under
  `~/.cache/tokenjam/models/`. Subsequent runs are offline-capable.
  Override the cache directory with `TOKENJAM_MODEL_CACHE=/path`.

## What the analyzer reports

For each scored prompt (up to 50 per run, biggest first):

- `prompt_chars` — total length of the captured prompt
- `significant_chars` — chars above the 0.40 significance threshold
- `bloat_chars` — chars in flagged regions
- `estimated_token_reduction` — rough token-savings estimate (4 chars/token)
- `regions` — list of contiguous low-significance spans (≥20 chars each)
  with start/end positions, average score, and a sample preview

The renderer dimmed-and-struck-through bloat regions in the HTML output;
the user can read the original prompt with the flagged regions clearly
marked.

## HTML report

```bash
tj report --bloat                # all agents, 30d window
tj report --bloat my-agent       # scope to one agent
tj report --bloat --since 7d     # custom window
tj report --bloat --no-open      # write file without opening browser
```

Output goes to `~/.cache/tokenjam/reports/trim-<timestamp>.html` and
opens in the user's default browser.

## Performance

On an M1 MacBook, scoring a 4K-token prompt takes ~200–400ms after the
model is loaded (first run includes ~1.5s model load). 50 prompts × 300ms
≈ 15s per run. CPU-only; no GPU required. On Linux servers without
heavy parallelism, expect similar latency.

## Confidence

`structural`. The classifier is trained on data outside the user's
domain, so its low-significance predictions are heuristic — not a quality
claim about whether the model would have produced the same output with
the region removed. The caveat surfaces this in every report.
