---
description: TOML read/write discipline for tj config and pricing files.
paths:
  - "tokenjam/core/config.py"
  - "tokenjam/core/pricing.py"
  - "tokenjam/cli/cmd_onboard.py"
  - "tokenjam/cli/cmd_policy.py"
  - "tokenjam/cli/cmd_pricing.py"
  - "**/*.toml"
---

# Config / TOML rules

### Critical Rule 2 — TOML binary mode

`tomllib.load()` requires `open(path, "rb")` not `"r"`. Text mode raises `TypeError` at runtime. Use
the conditional import: `tomllib` (3.11+) or `tomli` (3.10). Writing config uses `tomli_w`.
