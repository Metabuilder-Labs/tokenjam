---
description: Alert dispatch and captured-content stripping rules.
paths:
  - "tokenjam/core/alerts.py"
  - "tokenjam/core/ingest.py"
  - "tokenjam/core/drift.py"
  - "tokenjam/core/schema_validator.py"
  - "tokenjam/api/routes/alerts.py"
---

# Alert rules

### Critical Rule 5 — Alert content stripping

Remove `gen_ai.prompt.content`, `gen_ai.completion.content`, `gen_ai.tool.input`,
`gen_ai.tool.output` from alert payloads sent to external channels unless
`alerts.include_captured_content = true`. Stdout and file channels always get the full payload.

Note: content is also stripped at *ingest* (before DB write) by `strip_captured_content()` in
`core/ingest.py` per the four `[capture]` toggles (`prompts` / `completions` / `tool_inputs` /
`tool_outputs`) — so the alert flag is moot when the corresponding capture flag is off.
