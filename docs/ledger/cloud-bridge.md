# The TokenJam Cloud bridge

TokenJam runs entirely on your machine and always will. The bridge is the one opt-in that changes
that: it forwards what your local store already holds to a TokenJam Cloud organization, so a team
can see cost per merged PR built from real sessions rather than from a vendor's monthly total.

It is off unless you turn it on, it tells you exactly what will leave before the first byte does,
and it can be turned off in place.

```bash
tj init --cloud tj_live_... --org org_...   # connect this machine, after a yes
tj init --cloud off                          # stop forwarding, keep the key
```

Both values are on Cloud's Connect screen. The key alone does not identify the organization, which
is why `--org` is not optional. `--cloud-endpoint <url>` points at a different API; `--yes` skips
the confirmation for a scripted install.

## What leaves this machine

The command prints this list and waits for a yes before anything is sent.

**Leaves the machine:** token counts, model names, cost, timestamps, tool names, the file *paths*
touched, session / repo / branch / commit identifiers, the hashed developer id, the git author
email.

**Never leaves by default:** prompt text, completions, tool outputs, file contents, diffs, secrets.

**Never, under any setting:** your subscription OAuth credentials are never proxied or forwarded,
and subscription-plan traffic is never intercepted.

Content crosses only when two independent switches are both on: your local `[capture]` toggles must
be keeping it, and `[cloud] forward_content` must be `true`. The strip runs on the sending side, so
the promise does not depend on the receiver, and it sweeps for content-shaped keys beyond the named
list.

On the Cloud side the defaults match: developers are pseudonymous, per-developer views are
admin-only and suppressed below a cohort floor, there is no rank column anywhere, and the
organization kill switch can only tighten what is collected.

## What is forwarded, and how

The `tj serve` daemon runs a pass every five minutes, plus one immediately when you connect. Three
streams:

| Stream | Endpoint | Shape |
|---|---|---|
| spans | `POST /api/v1/spans` | the existing OTLP JSON. The ledger attributes ride as resource and span attributes; nothing new is on the wire |
| sessions | `POST /api/v1/ledger/sessions` | session rows with their repo, branch, developer, plan tier and pricing mode |
| session commits | the same endpoint | the [join rows](overview.md#the-join), with confidence and source |

Sessions and commits go in batches of at most 500 and are idempotent on their primary key, so a
re-send is free.

When you connect, `tj init --cloud` does three things before its first push: it refills repo context
on sessions that were ingested without it, runs the session-to-commit matcher, then forwards. Cloud
fills with joinable history rather than waiting for the daemon's next tick.

## Resuming

`~/.tj/cloud_sync.json` holds one high-water mark per stream, in **arrival** order rather than event
order: spans by their insert time, sessions by `updated_at`, commit rows by `matched_at`. Event time
would lose rows, because a backfill inserts spans that are older than everything already forwarded.
A session whose totals grew, that was closed, or whose plan tier was stamped later is re-sent and
deduped on arrival.

State is written after a batch is acknowledged, never before, so a crash mid-pass re-sends at most
one batch. The marks are scoped to the organization and to the database file they describe; change
either and they start over.

A machine that was offline for a week catches up on its own.

## When something goes wrong

- **A rejected key (401) disables the bridge** and records the reason, rather than retrying a dead
  key every five minutes forever. `tj status` and `tj doctor` print it. Re-running
  `tj init --cloud <key> --org <org>` with a current key clears it.
- **5xx, 429 and network failures** back off and resume on the next pass.
- **Any other 4xx** is a batch the receiver refused. The batch is split until the refused record is
  isolated, then that record is skipped and counted, so one bad row cannot wedge the stream.
- The forwarder runs on its own thread and swallows every error. It never blocks local ingest and
  never takes the local database down.

## Checking it

```bash
tj status    # Cloud: connected · N spans, M sessions, K commits sent · last 2m ago
tj doctor    # probes the endpoint and the key without sending any telemetry
```

`tj doctor` reports one of: not connected, forwarding off, unreachable, key rejected, or reachable
and accepted for your organization.

## Configuration

```toml
[cloud]
enabled         = true
endpoint        = "https://tokenjam-cloud-api.onrender.com"
org_id          = "org_..."
ingest_key      = "tj_live_..."
forward_content = false
```

The block holds a live per-organization ingest key. `.tj/config.toml` is untracked for exactly this
reason; keep it that way. See [configuration.md](../configuration.md#tokenjam-cloud-bridge).
