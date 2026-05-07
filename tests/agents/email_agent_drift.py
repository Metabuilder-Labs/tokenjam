"""Agent with 10x normal token usage — simulates drift."""
from tj.sdk import watch, record_llm_call
from tests.agents.mock_llm import MockLLMClient


@watch(agent_id="test-email-agent")
def run(task: str) -> str:
    client = MockLLMClient(
        script=["Very long response..." * 10],
        token_counts=[(1000, 2000)],  # 10x normal
    )
    response, in_tok, out_tok = client.complete(task)
    record_llm_call("claude-haiku-4-5", "anthropic", in_tok, out_tok)
    return response
