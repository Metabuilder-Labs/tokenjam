# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in TokenJam, please report it responsibly.

**Email:** [security@metabuilder.dev](mailto:security@metabuilder.dev)

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We will acknowledge receipt within 48 hours and aim to provide a fix within 7 days for critical issues.

## Scope

TokenJam handles sensitive data including:
- API keys (ingest secrets, provider API keys in config)
- Agent telemetry (prompts, completions, tool inputs/outputs when capture is enabled)
- Cost and usage data

Security issues in these areas are especially important to report.

## Not in Scope

- Vulnerabilities in upstream dependencies (report to the upstream project)
- Issues that require physical access to the machine running `ocw`
