"""
LlamaIndex agent example with OCW observability.

Uses LlamaIndex's native OTel support to export spans to ocw serve.
Builds a VectorStoreIndex from sample documents and queries it.

IMPORTANT: This example requires `ocw serve` to be running because
LlamaIndex integration exports spans over HTTP (not in-process).

Extra deps: pip install llama-index opentelemetry-instrumentation-llama-index
Run:        ocw serve &
            python examples/single_framework/llamaindex_agent.py
"""
import os
import sys

if not os.environ.get("OPENAI_API_KEY"):
    sys.exit(
        "OPENAI_API_KEY not set.\n"
        "Export it before running: export OPENAI_API_KEY=sk-..."
    )

import httpx  # noqa: E402

try:
    httpx.get("http://127.0.0.1:8787/api/v1/traces", timeout=2)
except httpx.ConnectError:
    sys.exit(
        "This example requires ocw serve to be running.\n"
        "Start it with: ocw serve &"
    )

from pathlib import Path  # noqa: E402

from llama_index.core import Document, VectorStoreIndex  # noqa: E402

from tj.sdk import watch, patch_llamaindex  # noqa: E402

# Configure LlamaIndex's native OTel to export to ocw serve
patch_llamaindex()

# Sample documents directory: examples/multi/sample_docs/
# If those files exist, load them; otherwise use inline fallback documents.
SAMPLE_DOCS_DIR = Path(__file__).parent.parent / "multi" / "sample_docs"


def load_documents() -> list[Document]:
    """Load documents from sample_docs dir or create inline fallbacks."""
    if SAMPLE_DOCS_DIR.is_dir():
        docs = []
        for path in sorted(SAMPLE_DOCS_DIR.glob("*.txt")):
            text = path.read_text()
            if text.strip():
                docs.append(Document(text=text, metadata={"source": path.name}))
        if docs:
            print(f"Loaded {len(docs)} documents from {SAMPLE_DOCS_DIR}")
            return docs

    # Fallback: inline sample documents
    print("Using inline fallback documents (sample_docs not found)")
    return [
        Document(
            text=(
                "Observability in AI agents involves collecting traces, metrics, "
                "and logs from agent runtimes. Unlike traditional monitoring which "
                "focuses on infrastructure health, agent observability tracks "
                "LLM calls, token usage, tool invocations, and decision chains. "
                "This enables developers to understand why an agent made specific "
                "choices, identify cost anomalies, and detect behavioral drift "
                "over time."
            ),
            metadata={"source": "observability_overview.txt"},
        ),
        Document(
            text=(
                "OpenTelemetry (OTel) provides a vendor-neutral standard for "
                "distributed tracing. When applied to AI agents, each LLM call "
                "becomes a span with attributes like model name, token counts, "
                "and latency. Tool calls are captured as child spans, creating "
                "a full execution tree. This structured telemetry can be exported "
                "to any OTel-compatible backend for analysis and alerting."
            ),
            metadata={"source": "otel_for_agents.txt"},
        ),
    ]


@watch(agent_id="llamaindex-demo")
def main():
    documents = load_documents()

    print("Building vector index...\n")
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    questions = [
        "What is observability in the context of AI agents?",
        "How does OpenTelemetry help with agent monitoring?",
    ]

    for question in questions:
        print(f"Q: {question}")
        response = query_engine.query(question)
        print(f"A: {response}\n")

    # --- Observation ---
    print("--- OCW Observation ---")
    print("LlamaIndex integration captured spans via native OTel support:")
    print("  - Document indexing and embedding calls")
    print("  - Query engine retrieval and LLM synthesis")
    print("  - Spans exported to ocw serve over HTTP")
    print("Run 'ocw traces' to see the captured telemetry.")


if __name__ == "__main__":
    main()
