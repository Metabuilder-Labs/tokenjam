"""
CrewAI agent example with OCW observability.

Creates a 2-agent crew (researcher + writer) that collaborates on a task.
OCW patches Task.execute and Agent.execute_task to capture spans automatically.

Extra deps: pip install crewai
Run:        python examples/single_framework/crewai_agent.py
"""
import os
import sys

if not os.environ.get("OPENAI_API_KEY"):
    sys.exit(
        "OPENAI_API_KEY not set.\n"
        "Export it before running: export OPENAI_API_KEY=sk-..."
    )

from crewai import Agent, Task, Crew  # noqa: E402

from tj.sdk import watch, patch_crewai  # noqa: E402

# Patch CrewAI BEFORE creating agents and tasks
patch_crewai()


@watch(agent_id="crewai-demo")
def main():
    # Create agents
    researcher = Agent(
        role="Senior Researcher",
        goal="Find key facts about the given topic",
        backstory=(
            "You are an experienced researcher who excels at finding "
            "and synthesizing information on technical topics."
        ),
        verbose=True,
    )

    writer = Agent(
        role="Technical Writer",
        goal="Write clear, concise summaries",
        backstory=(
            "You are a skilled technical writer who distills complex "
            "research into accessible summaries."
        ),
        verbose=True,
    )

    # Create tasks
    research_task = Task(
        description=(
            "Research the benefits of observability in AI agent systems. "
            "Identify the top 3 benefits and explain each briefly."
        ),
        expected_output="A list of 3 key benefits with brief explanations.",
        agent=researcher,
    )

    write_task = Task(
        description=(
            "Based on the research findings, write a 3-sentence summary "
            "about why observability matters for AI agents."
        ),
        expected_output="A concise 3-sentence summary paragraph.",
        agent=writer,
    )

    # Create and run the crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        verbose=True,
    )

    print("Starting CrewAI crew...\n")
    result = crew.kickoff()
    print(f"\n--- Crew Result ---\n{result}\n")

    # --- Observation ---
    print("--- OCW Observation ---")
    print("CrewAI integration captured spans for:")
    print("  - Task execution via Task.execute")
    print("  - Agent task execution via Agent.execute_task")
    print("Run 'ocw traces' to see the captured telemetry.")


if __name__ == "__main__":
    main()
