# TypeScript SDK

For any Node.js / TypeScript agent. Sends spans to `tj serve` over HTTP — no in-process state, no patches.

## Install

```bash
npm install @tokenjam/sdk
```

`tj serve` must be running locally for the SDK to send spans. Set `TJ_INGEST_SECRET` to the secret from `~/.config/tj/config.toml`.

## Quick start

```typescript
import { TjClient, SpanBuilder } from "@tokenjam/sdk";

const client = new TjClient({
  baseUrl:      "http://127.0.0.1:7391",
  ingestSecret: process.env.TJ_INGEST_SECRET ?? "",
});

const span = new SpanBuilder("invoke_agent")
  .agentId("my-ts-agent")
  .model("gpt-4o-mini")
  .provider("openai")
  .inputTokens(450)
  .outputTokens(120)
  .build();

await client.send([span]);
```

## Sessions

For long-running agent sessions, use `startSession` / `endSession`:

```typescript
const session = await client.startSession({ agentId: "my-ts-agent" });

// ... agent runs, sends spans tagged with session.id ...

await client.endSession(session.id);
```

## Span builder

`SpanBuilder` follows the OTel GenAI semantic conventions:

```typescript
new SpanBuilder("invoke_agent")
  .agentId("agent-1")
  .sessionId("sess-abc")
  .provider("anthropic")
  .model("claude-opus-4-7")
  .inputTokens(450)
  .outputTokens(120)
  .cacheReadTokens(0)
  .cacheWriteTokens(0)
  .durationMs(1200)
  .attribute("custom.key", "value")
  .build();
```

## Record an outcome

Attach a business outcome to a workflow in one call instead of hand-building an
OTLP event. `recordOutcome` emits the emerging gen_ai outcome event (OTel
semconv issue #2665):

```ts
await client.recordOutcome({
  outcomeType: "ticket_resolved",
  sessionId: "sess-9",
  success: true,
  valueUsd: 25.0, // optional, self-reported
});
```

At least one of `workflowId` / `sessionId` is required. `valueUsd` is
**optional and self-reported** — a value you declare, which TokenJam does not
measure or verify.

**ROI is a TokenJam Cloud feature.** The SDK only *emits* the outcome event;
turning declared value ÷ measured cost into an ROI figure happens in TokenJam
Cloud. There is no local ROI compute.

## Errors and retries

The client buffers up to 1000 spans if `tj serve` is unreachable, retries with exponential backoff (3 attempts, 2s base delay), and drops the buffer on process exit.

On `401 Unauthorized`, the client fails fast (no retries) and logs the configured secret fingerprint so you can spot a mismatch with the daemon's secret.

## API

Full type signatures and parameter docs: see [`sdk-ts/README.md`](../sdk-ts/README.md).
