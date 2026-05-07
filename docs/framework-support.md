# Framework Support

`ocw` is OTel-native. Any framework that emits OpenTelemetry spans works automatically — point its OTLP exporter at `tj serve` and you're done. For everything else, one-line patches exist.

## Python provider patches

Intercept at the API level, framework-agnostic:

```python
from ocw.sdk.integrations.anthropic import patch_anthropic   # Anthropic — Messages.create + streaming
from ocw.sdk.integrations.openai    import patch_openai      # OpenAI — chat completions
from ocw.sdk.integrations.gemini    import patch_gemini      # Google Gemini — GenerativeModel
from ocw.sdk.integrations.bedrock   import patch_bedrock     # AWS Bedrock — boto3 invoke_model/invoke_agent
from ocw.sdk.integrations.litellm   import patch_litellm    # LiteLLM — unified interface for 100+ providers
```

`patch_litellm()` covers all providers LiteLLM routes to (OpenAI, Anthropic, Bedrock, Vertex, Cohere, Mistral, Ollama, etc.) with correct per-provider attribution. If you use LiteLLM, you don't need the individual provider patches above.

OpenAI-compatible providers (Groq, Together, Fireworks, xAI, Azure OpenAI) also work via `patch_openai(base_url=...)` — no separate patches needed.

## Python framework patches

Instrument the framework's own tool and LLM abstractions:

```python
from ocw.sdk.integrations.langchain         import patch_langchain        # BaseLLM + BaseTool
from ocw.sdk.integrations.langgraph         import patch_langgraph        # CompiledGraph
from ocw.sdk.integrations.crewai            import patch_crewai           # Task + Agent
from ocw.sdk.integrations.autogen           import patch_autogen          # ConversableAgent
from ocw.sdk.integrations.llamaindex        import patch_llamaindex       # Native OTel wrapper
from ocw.sdk.integrations.openai_agents_sdk import patch_openai_agents   # Native OTel wrapper
from ocw.sdk.integrations.nemoclaw          import watch_nemoclaw         # NemoClaw Gateway observer
```

## Zero-code via OTLP

Point any of these frameworks' built-in OTel exporter at `tj serve`, no integration code required:

| Framework | OTel support |
|---|---|
| **Claude Code** | **Built-in** — [setup guide](claude-code-integration.md) |
| **OpenClaw** | **Built-in** (`diagnostics-otel` plugin) — [setup guide](openclaw.md) |
| LlamaIndex | `opentelemetry-instrumentation-llama-index` |
| OpenAI Agents SDK | Built-in |
| Google ADK | Built-in |
| Strands Agent SDK (AWS) | Built-in |
| Haystack | Built-in |
| Pydantic AI | Built-in |
| Semantic Kernel | Built-in |

## TypeScript / Node.js

`@tokenjam/sdk` provides `TjClient` and `SpanBuilder` for sending spans to `tj serve` from any TypeScript agent:

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
