# Python SDK

For any Python agent — Anthropic, OpenAI, Gemini, Bedrock, LangChain, CrewAI, and 10+ more frameworks.

## Install

```bash
pipx install tokenjam
tj onboard    # creates config, generates ingest secret
tj doctor     # verify your setup
```

## Quick start

```python
from tokenjam.sdk import watch
from tokenjam.sdk.integrations.anthropic import patch_anthropic

patch_anthropic()    # auto-intercepts all Anthropic API calls

@watch(agent_id="my-agent")
def run(task: str) -> str:
    # your agent code — nothing else to change
    ...
```

## Provider patches

Intercept at the API level. Framework-agnostic.

```python
from tokenjam.sdk.integrations.anthropic import patch_anthropic   # Anthropic
from tokenjam.sdk.integrations.openai    import patch_openai      # OpenAI
from tokenjam.sdk.integrations.gemini    import patch_gemini      # Google Gemini
from tokenjam.sdk.integrations.bedrock   import patch_bedrock     # AWS Bedrock
from tokenjam.sdk.integrations.litellm   import patch_litellm     # LiteLLM (100+ providers)
```

`patch_litellm()` covers all providers LiteLLM routes to (OpenAI, Anthropic, Bedrock, Vertex, Cohere, Mistral, Ollama, etc.). If you use LiteLLM, you don't need individual patches.

OpenAI-compatible providers (Groq, Together, Fireworks, xAI, Azure OpenAI) work via `patch_openai(base_url=...)`.

## Framework patches

Instrument the framework's own abstractions:

```python
from tokenjam.sdk.integrations.langchain         import patch_langchain        # BaseLLM + BaseTool
from tokenjam.sdk.integrations.langgraph         import patch_langgraph        # CompiledGraph
from tokenjam.sdk.integrations.crewai            import patch_crewai           # Task + Agent
from tokenjam.sdk.integrations.autogen           import patch_autogen          # ConversableAgent
from tokenjam.sdk.integrations.llamaindex        import patch_llamaindex       # Native OTel
from tokenjam.sdk.integrations.openai_agents_sdk import patch_openai_agents    # Native OTel
from tokenjam.sdk.integrations.nemoclaw          import watch_nemoclaw         # NemoClaw Gateway
```

Full framework support guide: [docs/framework-support.md](framework-support.md)

## Manual instrumentation

If you can't (or don't want to) use a patch, record spans manually:

```python
from tokenjam.sdk.agent import record_llm_call, record_tool_call

record_llm_call(
    agent_id="my-agent", provider="anthropic", model="claude-opus-4-7",
    input_tokens=450, output_tokens=120, duration_ms=1200,
)

record_tool_call(
    agent_id="my-agent", tool_name="send_email", duration_ms=300,
    success=True,
)
```

## Record an outcome

Attach a business outcome to a workflow in one line instead of hand-POSTing an
OTLP event. `record_outcome` emits the emerging gen_ai outcome event (OTel
semconv issue #2665) alongside your spans:

```python
from tokenjam.sdk import record_outcome

# Inside an active @watch() / AgentSession, the session is inherited:
record_outcome("ticket_resolved", success=True, value_usd=25.00)

# Or attach to a workflow / session you name yourself:
record_outcome(
    "lead_qualified",
    workflow_id="onboarding-run-42",
    success=True,
    value_usd=500.00,
)
```

At least one of `workflow_id` / `session_id` is required (an active session
satisfies it). `value_usd` is **optional and self-reported** — a value you
declare, which TokenJam does not measure or verify.

**ROI is a TokenJam Cloud feature.** The OSS SDK only *emits* the outcome event
(and the local stack ingests it as a normal span). Turning declared value ÷
measured cost into an ROI figure happens in TokenJam Cloud — there is no local
ROI compute.

## Examples

The [`examples/`](../examples/) directory has runnable agents for every integration. See [`examples/README.md`](../examples/README.md) for the full list.
