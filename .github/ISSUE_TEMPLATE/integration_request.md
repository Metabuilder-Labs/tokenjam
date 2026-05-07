---
name: Integration request
about: Request a new framework or provider integration
title: 'Integration: '
labels: integration
assignees: ''
---

**Framework / Provider**
Name and link to the project.

**How it handles telemetry today**
Does it have built-in OTel support? What spans/events does it emit?

**Proposed approach**
- [ ] Provider patch (monkey-patch API calls)
- [ ] Framework patch (wrap LLM/tool abstractions)
- [ ] OTLP bridge (thin wrapper around built-in OTel support)

**Are you willing to implement this?**
Yes / No — if yes, please read `tj/sdk/integrations/anthropic.py` as the reference implementation before starting.
