# @tokenjam/sdk

TypeScript SDK for [TokenJam](https://github.com/Metabuilder-Labs/tokenjam) — local-first, OTel-native observability for AI agents.

Communicates with a running `tj serve` instance via HTTP. No in-process OTel pipeline — spans are built with `SpanBuilder` and sent by `TjClient`.

> **Note:** Provider auto-instrumentation (the `patch_anthropic()`, `patch_openai()`, etc. convenience wrappers from the Python SDK) does not exist in this package. Every LLM call and tool call must be manually instrumented using `SpanBuilder`.

## Install

```bash
npm install @tokenjam/sdk
```

Requires Node.js >= 18. Start the TokenJam server before sending spans:

```bash
pip install tokenjam
tj serve
```

## Quick start

```typescript
import { TjClient, SpanBuilder, SpanStatus } from "@tokenjam/sdk";

const client = new TjClient({
  ingestSecret: "your-ingest-secret",   // from tj.toml security.ingest_secret
  serviceName: "my-agent",              // shown as agent ID in tj status
}).start();

// Record an LLM call
const span = new SpanBuilder("gen_ai.llm.call")
  .agentId("my-agent")
  .agentName("My Agent")
  .provider("anthropic")
  .model("claude-sonnet-4-6")
  .inputTokens(512)
  .outputTokens(128)
  .cacheReadTokens(256)
  .cacheCreateTokens(64)
  .conversationId("conv-abc123")
  .startTime(new Date().toISOString())
  .durationMs(1200)
  .build();

await client.send(span);
await client.shutdown();
```

## TjClient

```typescript
new TjClient(options: TjClientOptions)
```

| Option | Type | Default | Description |
|---|---|---|---|
| `ingestSecret` | `string` | required | Bearer token from `security.ingest_secret` in `tj.toml` |
| `baseUrl` | `string` | `http://127.0.0.1:7391` | `tj serve` base URL |
| `serviceName` | `string` | `"tj-ts-sdk"` | Reported as `service.name` in OTLP resource attributes; used as fallback agent ID |
| `batchSize` | `number` | `50` | Max spans buffered before auto-flush |
| `flushIntervalMs` | `number` | `5000` | Interval between automatic flushes (ms) |
| `maxRetries` | `number` | `3` | Retry attempts on network errors and 5xx responses; 4xx errors are not retried |

### Methods

| Method | Description |
|---|---|
| `client.start()` | Start the automatic flush timer. Returns `this`. |
| `client.send(span)` | Buffer a span; auto-flushes when `batchSize` is reached. |
| `client.flush()` | Immediately send all buffered spans. Returns `IngestResult | null`. |
| `client.shutdown()` | Flush remaining spans and stop the timer. Call before process exit. |
| `client.recordOutcome(options)` | Emit a gen_ai outcome event attaching a business outcome to a workflow. |

### Record an outcome

Attach a business outcome to a workflow in one call — the emerging gen_ai
outcome event (OTel semconv issue #2665) TokenJam Cloud's ROI backend ingests.

```typescript
await client.recordOutcome({
  outcomeType: "ticket_resolved", // required marker
  sessionId: "sess-9",            // at least one of sessionId / workflowId
  success: true,
  valueUsd: 25.0,                 // optional, self-reported
});
```

| Option | Type | Description |
|---|---|---|
| `outcomeType` | `string` | Required. Caller-defined label (e.g. `"ticket_resolved"`). The marker attribute. |
| `workflowId` | `string` | Explicit workflow key. At least one of `workflowId` / `sessionId` is required. |
| `sessionId` | `string` | Session (or root session of a fan-out) the outcome belongs to. |
| `success` | `boolean` | Whether the outcome was achieved. Defaults to `true`. |
| `valueUsd` | `number` | **Optional, self-reported** business value. TokenJam does not measure or verify it. |
| `agentId` | `string` | Emitting agent id. |
| `attributes` | `Record<string, unknown>` | Extra attributes attached verbatim. |

**ROI compute is a TokenJam Cloud feature.** The SDK only emits the event.

## SpanBuilder

Fluent builder for constructing spans with GenAI semantic conventions.

```typescript
new SpanBuilder(name: string)
```

### Agent identity

| Method | Attribute set |
|---|---|
| `.agentId(id)` | `gen_ai.agent.id` + `span.agentId` |
| `.agentName(name)` | `gen_ai.agent.name` |
| `.agentVersion(version)` | `gen_ai.agent.version` |
| `.sessionId(id)` | `gen_ai.session.id` |
| `.conversationId(id)` | `gen_ai.conversation.id` |

### LLM call attributes

| Method | Attribute set |
|---|---|
| `.provider(name)` | `gen_ai.provider.name` |
| `.model(name)` | `gen_ai.request.model` |
| `.inputTokens(n)` | `gen_ai.usage.input_tokens` |
| `.outputTokens(n)` | `gen_ai.usage.output_tokens` |
| `.cacheReadTokens(n)` | `gen_ai.usage.cache_read_tokens` |
| `.cacheCreateTokens(n)` | `gen_ai.usage.cache_creation_tokens` |

### Tool call attributes

| Method | Attribute set |
|---|---|
| `.toolName(name)` | `gen_ai.tool.name` |
| `.toolInput(input)` | `gen_ai.tool.input` |
| `.toolOutput(output)` | `gen_ai.tool.output` |

### Span metadata

| Method | Description |
|---|---|
| `.traceId(id)` | Override auto-generated trace ID |
| `.spanId(id)` | Override auto-generated span ID |
| `.parentSpanId(id)` | Set parent span for trace hierarchy |
| `.kind(SpanKind)` | Span kind (default: `CLIENT`) |
| `.status(SpanStatus, message?)` | Status code (default: `OK`) |
| `.startTime(iso)` | Start time as ISO 8601 string |
| `.endTime(iso)` | End time as ISO 8601 string |
| `.durationMs(ms)` | Duration; used to compute `endTime` if not set |
| `.attribute(key, value)` | Set any arbitrary attribute |
| `.build()` | Returns the completed `Span` object |

## Semantic convention constants

```typescript
import { GenAIAttributes, TjAttributes, ClaudeCodeEvents } from "@tokenjam/sdk";

// GenAI attribute name strings
GenAIAttributes.AGENT_ID           // "gen_ai.agent.id"
GenAIAttributes.REQUEST_MODEL      // "gen_ai.request.model"
GenAIAttributes.CACHE_CREATE_TOKENS // "gen_ai.usage.cache_creation_tokens"
// ...

// tj-specific attribute name strings
TjAttributes.COST_USD             // "tokenjam.cost_usd"
TjAttributes.SANDBOX_EVENT        // "tokenjam.sandbox.event"
// ...

// Claude Code OTel log event names and attribute constants
ClaudeCodeEvents.API_REQUEST       // "claude_code.api_request"
ClaudeCodeEvents.TOOL_RESULT       // "claude_code.tool_result"
ClaudeCodeEvents.COST_USD          // "cost_usd"
ClaudeCodeEvents.INPUT_TOKENS      // "input_tokens"
// ...
```

Use `ClaudeCodeEvents` when writing agents that consume Claude Code's own OTel log output (e.g. via the `tj` MCP server or a log subscriber).

## SpanKind and SpanStatus

```typescript
import { SpanKind, SpanStatus } from "@tokenjam/sdk";

SpanKind.CLIENT   // default for LLM calls
SpanKind.SERVER
SpanKind.INTERNAL
SpanKind.PRODUCER
SpanKind.CONSUMER

SpanStatus.OK     // default
SpanStatus.ERROR
SpanStatus.UNSET
```

## What this SDK does NOT provide

Unlike the Python SDK (`pip install tokenjam`), this package does **not** include:

- **Session management** (`@watch()` decorator / `AgentSession` context manager) — you must manually build and send `invoke_agent` session spans.
- **Provider auto-instrumentation** — no `patchAnthropic()`, `patchOpenAI()`, etc. Every LLM call requires an explicit `SpanBuilder`.
- **Framework patches** — no LangChain JS, OpenAI Agents SDK, or Vercel AI SDK integration.
- **In-process OTel pipeline** — all telemetry goes over HTTP to `tj serve`.

See the [Python SDK docs](https://github.com/Metabuilder-Labs/tokenjam#python-sdk) for the full-featured in-process instrumentation path.
