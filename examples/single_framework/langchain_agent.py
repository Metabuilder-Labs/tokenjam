"""
LangChain agent example with OCW observability.

Uses ChatOpenAI with tool bindings to demonstrate LangChain integration.
OCW patches BaseLLM.generate and BaseTool.run to capture spans automatically.

Extra deps: pip install langchain-core langchain-openai
Run:        python examples/single_framework/langchain_agent.py
"""
import os
import sys

if not os.environ.get("OPENAI_API_KEY"):
    sys.exit(
        "OPENAI_API_KEY not set.\n"
        "Export it before running: export OPENAI_API_KEY=sk-..."
    )

from langchain_core.tools import BaseTool  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from tj.sdk import watch, patch_langchain  # noqa: E402

# Patch LangChain BEFORE creating any LangChain objects
patch_langchain()


class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "Evaluate a math expression and return the result."

    def _run(self, expression: str) -> str:
        try:
            result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
            return str(result)
        except Exception as exc:
            return f"Error: {exc}"


class WordCounterTool(BaseTool):
    name: str = "word_counter"
    description: str = "Count the number of words in a given string."

    def _run(self, text: str) -> str:
        count = len(text.split())
        return f"{count} words"


@watch(agent_id="langchain-demo")
def main():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [CalculatorTool(), WordCounterTool()]
    llm_with_tools = llm.bind_tools(tools)

    question = (
        "How many words are in 'the quick brown fox jumps over the lazy dog' "
        "and what is 42 * 17?"
    )
    print(f"Question: {question}\n")

    # First LLM call -- may request tool calls
    response = llm_with_tools.invoke(question)
    print(f"LLM response: {response.content}")

    # Process tool calls if present
    tool_map = {t.name: t for t in tools}
    tool_results = []

    if response.tool_calls:
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            print(f"\nTool call: {tool_name}({tool_args})")
            tool = tool_map[tool_name]
            # BaseTool.run is patched by OCW
            result = tool.run(tool_args)
            print(f"Tool result: {result}")
            tool_results.append({"tool_call_id": tc["id"], "result": result})

    # If we got tool results, send them back to the LLM for a final answer
    if tool_results:
        from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

        messages = [
            HumanMessage(content=question),
            AIMessage(content=response.content, tool_calls=response.tool_calls),
        ]
        for tr in tool_results:
            messages.append(
                ToolMessage(content=tr["result"], tool_call_id=tr["tool_call_id"])
            )
        final = llm_with_tools.invoke(messages)
        print(f"\nFinal answer: {final.content}")

    # --- Observation ---
    print("\n--- OCW Observation ---")
    print("LangChain integration captured spans for:")
    print("  - LLM calls via ChatOpenAI")
    print("  - Tool calls via BaseTool.run")
    print("Run 'ocw traces' to see the captured telemetry.")


if __name__ == "__main__":
    main()
