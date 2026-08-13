---
description: Layout convention for examples/ and the Agent Incident Library.
paths:
  - "examples/**"
  - "incidents/**"
  - "tokenjam/demo/**"
  - "tokenjam/cli/cmd_demo.py"
---

# Examples convention

Each provider integration in `examples/single_provider/` and each framework in
`examples/single_framework/` lives in **its own file** — when adding a new SDK integration, mirror
this layout (one demo file per integration) so the examples directory stays a 1:1 map of supported
integrations. Multi-provider/framework demos go in `examples/multi/`; alert and drift demos that need
no API keys go in `examples/alerts_and_drift/`.

The Agent Incident Library at `incidents/` is separate: each scenario is a `scenario.py` + `README.md`
pair, invoked via `tj demo <scenario>`. Scenarios inject synthetic spans through `tokenjam/demo/env.py`
to simulate real failures (retry-loop, surprise-cost, hallucination-drift) without API keys or a live
server.
