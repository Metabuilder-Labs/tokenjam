"""
RAG Pipeline with Provider Fallback — LlamaIndex + OpenAI/Anthropic.

Demonstrates a retrieval-augmented generation pipeline that falls back from
OpenAI to Anthropic when the primary provider fails. Requires `ocw serve`
running on localhost:8787 for span ingestion.

Extra deps:
    pip install llama-index openai anthropic

Required env vars:
    OPENAI_API_KEY, ANTHROPIC_API_KEY

Prerequisite:
    ocw serve   (must be running on localhost:8787)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx

from tj.sdk import watch, patch_llamaindex, patch_openai, patch_anthropic
from tj.sdk.agent import record_tool_call

# ---------------------------------------------------------------------------
# Env-var gate
# ---------------------------------------------------------------------------
REQUIRED_KEYS = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
missing = [k for k in REQUIRED_KEYS if not os.environ.get(k)]
if missing:
    sys.exit(f"Missing env vars: {', '.join(missing)}")

# ---------------------------------------------------------------------------
# Connectivity check — ocw serve must be running
# ---------------------------------------------------------------------------
try:
    resp = httpx.get("http://localhost:8787/api/v1/spans", timeout=3)
except httpx.ConnectError:
    sys.exit(
        "Cannot connect to ocw serve on localhost:8787. "
        "Start it first: ocw serve"
    )

# ---------------------------------------------------------------------------
# Activate patches
# ---------------------------------------------------------------------------
patch_llamaindex()
patch_openai()
patch_anthropic()


# ---------------------------------------------------------------------------
# Document loading and index construction
# ---------------------------------------------------------------------------
DOCS_DIR = Path(__file__).parent / "sample_docs"


def build_index():
    """Load sample docs and build an in-memory vector index."""
    from llama_index.core import (
        SimpleDirectoryReader,
        VectorStoreIndex,
    )

    documents = SimpleDirectoryReader(str(DOCS_DIR)).load_data()
    print(f"Loaded {len(documents)} documents from {DOCS_DIR}")
    index = VectorStoreIndex.from_documents(documents)
    return index


# ---------------------------------------------------------------------------
# Query with provider fallback
# ---------------------------------------------------------------------------
def query_with_fallback(
    index,
    question: str,
    force_fallback: bool = False,
) -> str:
    """
    Query the index with OpenAI; fall back to Anthropic on failure.

    When force_fallback is True, the OpenAI path is skipped to demonstrate
    the fallback branch in the trace.
    """
    from llama_index.core import Settings

    # Record the retrieval step
    record_tool_call(
        "retrieval",
        tool_input={"question": question},
        tool_output={"source": "sample_docs", "chunks_retrieved": 3},
    )

    # --- Primary: OpenAI ---
    if not force_fallback:
        try:
            from llama_index.llms.openai import OpenAI as LlamaOpenAI

            Settings.llm = LlamaOpenAI(model="gpt-4o-mini")
            engine = index.as_query_engine()
            response = engine.query(question)
            print(f"  [OpenAI] {response}")
            return str(response)
        except Exception as exc:
            print(f"  [OpenAI failed: {exc}] Falling back to Anthropic...")

    # --- Fallback: Anthropic ---
    try:
        from llama_index.llms.anthropic import Anthropic as LlamaAnthropic

        Settings.llm = LlamaAnthropic(model="claude-sonnet-4-20250514")
        engine = index.as_query_engine()
        response = engine.query(question)
        print(f"  [Anthropic fallback] {response}")
        return str(response)
    except Exception as exc:
        error_msg = f"Both providers failed: {exc}"
        print(f"  {error_msg}")
        return error_msg


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------
@watch(agent_id="rag-pipeline")
def main() -> None:
    index = build_index()

    questions = [
        ("What are the key pillars of observability?", False),
        # Force fallback on the second question to show both providers
        ("How can I reduce LLM API costs?", True),
        ("What safety guardrails should AI agents have?", False),
    ]

    for question, force_fallback in questions:
        label = " [forced fallback]" if force_fallback else ""
        print(f"\nQ: {question}{label}")
        query_with_fallback(index, question, force_fallback=force_fallback)


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------
# After running, inspect the RAG session:
#
#   ocw traces --since 10m
#       -> Single trace showing retrieval + LLM spans
#
#   ocw trace <trace-id>
#       -> Waterfall: session -> retrieval tool calls -> LLM calls
#          The second question shows both OpenAI (skipped) and Anthropic spans
#
#   ocw cost --since 1h
#       -> Cost comparison between OpenAI and Anthropic calls
#
#   ocw tools
#       -> "retrieval" tool call count matching the number of questions
