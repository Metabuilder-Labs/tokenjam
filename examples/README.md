# OCW Examples

Example agents demonstrating [tj](https://github.com/Metabuilder-Labs/TokenJam) integrations, from single-provider basics to complex multi-agent workflows.

## Quick Start

```bash
pip install -e ".[dev]"           # install tokenjam in dev mode
export ANTHROPIC_API_KEY=sk-...   # set your API key
python examples/single_provider/anthropic_agent.py
```

After running any example, inspect the results with:

```bash
tj status       # agent session overview
tj traces       # list all spans from the run
tj cost --since 1h   # cost breakdown
tj alerts       # any fired alerts
```

---

## Single Provider

One integration per file. Simplest way to see tj in action with your provider of choice.

| Example | Env Vars | Extra Deps | Description |
|---|---|---|---|
| [`anthropic_agent.py`](single_provider/anthropic_agent.py) | `ANTHROPIC_API_KEY` | `anthropic` | Tool-use agent with calculator and weather tools |
| [`openai_agent.py`](single_provider/openai_agent.py) | `OPENAI_API_KEY` | `openai` | Function-calling agent with streaming response |
| [`gemini_agent.py`](single_provider/gemini_agent.py) | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | `google-generativeai` | Text summarization via Gemini Flash |
| [`bedrock_agent.py`](single_provider/bedrock_agent.py) | `AWS_DEFAULT_REGION` + AWS creds | `boto3` | Claude on AWS Bedrock (advanced setup) |
| [`openai_agents_sdk_agent.py`](single_provider/openai_agents_sdk_agent.py) | `OPENAI_API_KEY` | `openai-agents httpx` | Multi-agent handoff via OpenAI Agents SDK |
| [`litellm_agent.py`](single_provider/litellm_agent.py) | `OPENAI_API_KEY` `ANTHROPIC_API_KEY` | `litellm` | Multi-provider routing via LiteLLM |

> **Note:** `openai_agents_sdk_agent.py` requires `tj serve` running (`tj serve &`).

---

## Single Framework

One framework integration per file. Shows how tj captures framework-level spans.

| Example | Env Vars | Extra Deps | Description |
|---|---|---|---|
| [`langchain_agent.py`](single_framework/langchain_agent.py) | `OPENAI_API_KEY` | `langchain-core langchain-openai` | Tool-calling agent with calculator and word counter |
| [`langgraph_agent.py`](single_framework/langgraph_agent.py) | `OPENAI_API_KEY` | `langgraph langchain-openai` | Plan-execute-review graph pipeline |
| [`crewai_agent.py`](single_framework/crewai_agent.py) | `OPENAI_API_KEY` | `crewai` | Researcher + writer crew collaboration |
| [`autogen_agent.py`](single_framework/autogen_agent.py) | `OPENAI_API_KEY` | `pyautogen` | Two-agent debate with back-and-forth |
| [`llamaindex_agent.py`](single_framework/llamaindex_agent.py) | `OPENAI_API_KEY` | `llama-index` | RAG query engine over sample documents |

> **Note:** `llamaindex_agent.py` requires `tj serve` running (`tj serve &`).

---

## Multi-Integration

Complex real-world patterns combining multiple providers and frameworks. These showcase tj's ability to track cost, performance, and behavior across a heterogeneous agent stack.

| Example | Env Vars | Extra Deps | Description |
|---|---|---|---|
| [`router_agent.py`](multi/router_agent.py) | `ANTHROPIC_API_KEY` `OPENAI_API_KEY` `GOOGLE_API_KEY` | `anthropic openai google-generativeai` | Routes tasks to the cheapest/best provider |
| [`research_team.py`](multi/research_team.py) | `ANTHROPIC_API_KEY` `OPENAI_API_KEY` | `anthropic crewai langchain-core` | CrewAI agents with LangChain tools |
| [`rag_pipeline.py`](multi/rag_pipeline.py) | `OPENAI_API_KEY` `ANTHROPIC_API_KEY` | `llama-index openai anthropic` | RAG with OpenAI-to-Anthropic fallback |

> **Note:** `rag_pipeline.py` requires `tj serve` running (`tj serve &`).

---

## Alerts and Drift

These examples demonstrate what makes tj unique: real-time alerting and behavioral drift detection. **No API keys required** -- they use simulated instrumentation via `record_llm_call()` and `record_tool_call()`.

| Example | Env Vars | Description |
|---|---|---|
| [`sensitive_actions_demo.py`](alerts_and_drift/sensitive_actions_demo.py) | None | Fires alerts when agent calls sensitive tools |
| [`budget_breach_demo.py`](alerts_and_drift/budget_breach_demo.py) | None | Exceeds budget limits, shows cost alerts |
| [`drift_demo.py`](alerts_and_drift/drift_demo.py) | None | Builds baseline, then triggers drift detection |

These examples include the required `tj.toml` config snippets as comments at the top of each file. Copy the relevant config to your `tj.toml` before running.

```bash
# No API keys needed -- just run:
python examples/alerts_and_drift/budget_breach_demo.py

# Then inspect:
tj alerts       # see budget-breach alerts
tj cost --since 1h   # see cost tracking
```

---

## Which examples need `tj serve`?

Most examples use in-process telemetry (spans go directly to the local DuckDB). These two integrations export via OTLP HTTP and require the server:

- `openai_agents_sdk_agent.py`
- `llamaindex_agent.py`
- `rag_pipeline.py` (uses LlamaIndex)

Start the server before running them:

```bash
tj serve &
```
