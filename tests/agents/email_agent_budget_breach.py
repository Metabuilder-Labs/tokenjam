"""Agent whose session cost exceeds limit — simulates budget breach."""
from tj.sdk import watch, record_llm_call
from tests.agents.mock_llm import MockLLMClient


@watch(agent_id="test-email-agent")
def run(task: str) -> str:
    client = MockLLMClient(
        script=["Expensive response..."] * 10,
        token_counts=[(10000, 5000)] * 10,  # Very high token usage
    )
    for _ in range(10):
        response, in_tok, out_tok = client.complete(task)
        record_llm_call("claude-opus-4-20250514", "anthropic", in_tok, out_tok)
    return "done"
