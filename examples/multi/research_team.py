"""
Multi-Framework Research Team — CrewAI agents with LangChain tools.

Demonstrates deep span trees from combining CrewAI multi-agent orchestration
with LangChain tool abstractions, all captured in a single ocw session.

Extra deps:
    pip install anthropic crewai langchain-core

Required env vars:
    ANTHROPIC_API_KEY, OPENAI_API_KEY
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from tj.sdk import watch, patch_anthropic, patch_crewai, patch_langchain

# ---------------------------------------------------------------------------
# Env-var gate
# ---------------------------------------------------------------------------
REQUIRED_KEYS = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
missing = [k for k in REQUIRED_KEYS if not os.environ.get(k)]
if missing:
    sys.exit(f"Missing env vars: {', '.join(missing)}")

# ---------------------------------------------------------------------------
# Activate patches BEFORE importing framework classes
# ---------------------------------------------------------------------------
patch_anthropic()
patch_crewai()
patch_langchain()

# ---------------------------------------------------------------------------
# LangChain tool stubs
# ---------------------------------------------------------------------------
from langchain_core.tools import BaseTool  # noqa: E402


class WebSearchTool(BaseTool):
    """Simulated web search returning fake results."""

    name: str = "web_search"
    description: str = "Search the web for information on a topic."

    def _run(self, query: str) -> str:
        return (
            f"Search results for '{query}':\n"
            "1. Recent advances in renewable energy storage (Nature, 2025)\n"
            "2. Global renewable capacity grew 15% year-over-year (IEA)\n"
            "3. Battery costs fell below $100/kWh milestone (BloombergNEF)"
        )


class CalculatorTool(BaseTool):
    """Simple calculator that evaluates math expressions."""

    name: str = "calculator"
    description: str = "Evaluate a mathematical expression."

    def _run(self, expression: str) -> str:
        try:
            result = eval(expression, {"__builtins__": {}})  # noqa: S307
            return f"{expression} = {result}"
        except Exception as exc:
            return f"Error evaluating '{expression}': {exc}"


class FileReaderTool(BaseTool):
    """Reads a sample document from the sample_docs directory."""

    name: str = "file_reader"
    description: str = "Read a document file by name from sample_docs."

    def _run(self, filename: str) -> str:
        docs_dir = Path(__file__).parent / "sample_docs"
        target = docs_dir / filename
        if not target.exists():
            available = [f.name for f in docs_dir.glob("*.txt")]
            return f"File '{filename}' not found. Available: {available}"
        return target.read_text()[:500]


# ---------------------------------------------------------------------------
# CrewAI setup
# ---------------------------------------------------------------------------
from crewai import Agent, Task, Crew  # noqa: E402


def build_crew() -> Crew:
    """Create a research crew with three specialized agents."""
    tools = [WebSearchTool(), CalculatorTool(), FileReaderTool()]

    lead_researcher = Agent(
        role="Lead Researcher",
        goal="Coordinate the research team to produce a concise report",
        backstory=(
            "You are an experienced research lead who delegates effectively."
        ),
        tools=tools,
        verbose=True,
    )

    data_analyst = Agent(
        role="Data Analyst",
        goal="Analyze numerical data and compute statistics",
        backstory="You specialize in quantitative analysis and calculations.",
        tools=[CalculatorTool()],
        verbose=True,
    )

    writer = Agent(
        role="Report Writer",
        goal="Synthesize findings into a clear, concise report",
        backstory="You turn raw research into polished written output.",
        tools=[FileReaderTool()],
        verbose=True,
    )

    research_task = Task(
        description=(
            "Search the web for the latest trends in renewable energy "
            "and summarize the top three findings."
        ),
        expected_output="A bullet-point list of 3 key findings.",
        agent=lead_researcher,
    )

    analysis_task = Task(
        description=(
            "If global renewable capacity was 3,500 GW last year and grew "
            "15%, calculate the new total capacity. Also compute the cost "
            "savings if battery prices dropped from $130 to $97 per kWh "
            "for a 100 MWh installation."
        ),
        expected_output="Numerical results with brief explanations.",
        agent=data_analyst,
    )

    report_task = Task(
        description=(
            "Read the cost_management.txt sample document for context, "
            "then combine the research findings and analysis into a "
            "200-word executive summary."
        ),
        expected_output="A polished executive summary paragraph.",
        agent=writer,
    )

    return Crew(
        agents=[lead_researcher, data_analyst, writer],
        tasks=[research_task, analysis_task, report_task],
        verbose=True,
    )


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------
@watch(agent_id="research-team")
def main() -> None:
    crew = build_crew()
    result = crew.kickoff()
    print("\n=== Final Report ===")
    print(result)


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------
# After running, explore the multi-agent session:
#
#   ocw traces --since 10m
#       -> Single trace containing all three agents' activity
#
#   ocw trace <trace-id>
#       -> Deep span tree: session -> agent tasks -> LLM calls + tool calls
#
#   ocw tools
#       -> Breakdown of web_search, calculator, and file_reader usage
#          showing call counts and average durations
#
#   ocw cost --since 1h
#       -> Total session cost across all LLM calls made by the crew
