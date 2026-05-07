"""Normal session — baseline behavior, no anomalies."""
from tj.sdk import watch, record_llm_call, record_tool_call
from tests.agents.mock_llm import MockLLMClient


@watch(agent_id="test-email-agent")
def run(task: str) -> str:
    client = MockLLMClient(
        script=["Drafting email...", "Sending..."],
        token_counts=[(100, 20), (150, 30)],
    )

    response, in_tok, out_tok = client.complete(task)
    record_llm_call("claude-haiku-4-5", "anthropic", in_tok, out_tok)

    response, in_tok, out_tok = client.complete("send it")
    record_llm_call("claude-haiku-4-5", "anthropic", in_tok, out_tok)

    record_tool_call("send_email", tool_output={"status": "sent"})
    return "sent"
