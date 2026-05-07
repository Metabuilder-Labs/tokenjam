"""Agent that calls the same tool 5 times in a row — simulates retry loop."""
from tj.sdk import watch, record_llm_call, record_tool_call
from tests.agents.mock_llm import MockLLMClient


@watch(agent_id="test-email-agent")
def run(task: str) -> str:
    client = MockLLMClient(
        script=["Trying...", "Retrying...", "Again...", "Once more...", "Last try..."],
        token_counts=[(100, 20)] * 5,
    )
    for _ in range(5):
        response, in_tok, out_tok = client.complete(task)
        record_llm_call("claude-haiku-4-5", "anthropic", in_tok, out_tok)
        record_tool_call("send_email", tool_output={"status": "failed"}, error="SMTP timeout")
    return "failed"
