"""
LangGraph agent example with OCW observability.

Builds a simple 3-node StateGraph (plan -> execute -> review) and runs it.
OCW patches CompiledGraph.invoke to capture the graph execution as a span,
and BaseLLM.generate to capture individual LLM calls within each node.

Extra deps: pip install langgraph langchain-openai
Run:        python examples/single_framework/langgraph_agent.py
"""
import os
import sys

if not os.environ.get("OPENAI_API_KEY"):
    sys.exit(
        "OPENAI_API_KEY not set.\n"
        "Export it before running: export OPENAI_API_KEY=sk-..."
    )

from typing import TypedDict  # noqa: E402

from langchain_openai import ChatOpenAI  # noqa: E402
from langgraph.graph import StateGraph, START, END  # noqa: E402

from tj.sdk import watch, patch_langgraph, patch_langchain  # noqa: E402

# Patch both LangGraph and LangChain (LangGraph uses LangChain LLMs)
patch_langgraph()
patch_langchain()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


class WorkflowState(TypedDict):
    task: str
    plan: str
    result: str
    review: str


def plan_node(state: WorkflowState) -> WorkflowState:
    """Use the LLM to create a plan for the given task."""
    prompt = (
        f"Create a brief 3-step plan for the following task. "
        f"Return only the numbered steps.\n\nTask: {state['task']}"
    )
    response = llm.invoke(prompt)
    plan = response.content
    print(f"[plan] {plan}\n")
    return {**state, "plan": plan}


def execute_node(state: WorkflowState) -> WorkflowState:
    """Use the LLM to execute the plan."""
    prompt = (
        f"Execute the following plan and produce the final output.\n\n"
        f"Task: {state['task']}\nPlan:\n{state['plan']}"
    )
    response = llm.invoke(prompt)
    result = response.content
    print(f"[execute] {result}\n")
    return {**state, "result": result}


def review_node(state: WorkflowState) -> WorkflowState:
    """Use the LLM to review the output."""
    prompt = (
        f"Review the following output for the given task. "
        f"Provide a brief quality assessment (2-3 sentences).\n\n"
        f"Task: {state['task']}\nOutput:\n{state['result']}"
    )
    response = llm.invoke(prompt)
    review = response.content
    print(f"[review] {review}\n")
    return {**state, "review": review}


@watch(agent_id="langgraph-demo")
def main():
    # Build the graph: START -> plan -> execute -> review -> END
    graph = StateGraph(WorkflowState)
    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("review", review_node)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "review")
    graph.add_edge("review", END)

    compiled = graph.compile()

    task = "Write a haiku about observability"
    print(f"Task: {task}\n")

    result = compiled.invoke({
        "task": task,
        "plan": "",
        "result": "",
        "review": "",
    })

    print("--- Final State ---")
    print(f"Plan:\n{result['plan']}\n")
    print(f"Result:\n{result['result']}\n")
    print(f"Review:\n{result['review']}\n")

    # --- Observation ---
    print("--- OCW Observation ---")
    print("LangGraph integration captured spans for:")
    print("  - Graph invocation via CompiledGraph.invoke")
    print("  - Individual LLM calls within each node (via LangChain patch)")
    print("Run 'ocw traces' to see the captured telemetry.")


if __name__ == "__main__":
    main()
