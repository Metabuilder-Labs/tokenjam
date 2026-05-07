"""
Sensitive Actions Alert Demo

Simulates an agent that performs sensitive actions (send_email, delete_file,
submit_form). Shows how ocw detects and alerts on configured sensitive tools.

No API keys required — uses simulated instrumentation.

The demo seeds its own [agents.sensitive-demo] block in the active ocw
config on startup, so it works out of the box on a fresh `ocw onboard`.
The injected config is equivalent to:

    [agents.sensitive-demo]
    [[agents.sensitive-demo.sensitive_actions]]
    name = "send_email"
    severity = "warning"
    [[agents.sensitive-demo.sensitive_actions]]
    name = "delete_file"
    severity = "critical"
    [[agents.sensitive-demo.sensitive_actions]]
    name = "submit_form"
    severity = "warning"
"""
from __future__ import annotations

import json
import os
import tempfile

from tj.sdk.agent import watch, record_llm_call, record_tool_call


# ---------------------------------------------------------------------------
# Fake tool implementations (no real side-effects)
# ---------------------------------------------------------------------------

def send_email(to: str, subject: str, body: str) -> dict:
    """Write an email record to a temp log file."""
    log_path = os.path.join(tempfile.gettempdir(), "ocw_email_log.txt")
    with open(log_path, "a") as f:
        f.write(f"To: {to}\nSubject: {subject}\nBody: {body}\n---\n")
    return {"status": "sent", "log": log_path}


def delete_file(path: str) -> dict:
    """Create a temp file and delete it to simulate file deletion."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix="_demo.txt")
    tmp.write(b"temporary content for demo")
    tmp.close()
    os.unlink(tmp.name)
    return {"status": "deleted", "path": tmp.name}


def submit_form(url: str, data: dict) -> dict:
    """Print form submission to stdout (no network call)."""
    print(f"  [simulated] POST {url} with data={json.dumps(data)}")
    return {"status": "submitted", "url": url}


# ---------------------------------------------------------------------------
# Agent workflow
# ---------------------------------------------------------------------------

@watch(agent_id="sensitive-demo")
def run_sensitive_agent() -> None:
    """Simulate an agent that triggers three sensitive actions."""

    # Step 1 -- LLM decides to send an email
    print("\n[Step 1] Agent decides to send an email...")
    record_llm_call(
        model="claude-sonnet-4-20250514",
        provider="anthropic",
        input_tokens=500,
        output_tokens=150,
    )
    email_input = {
        "to": "alice@example.com",
        "subject": "Weekly report",
        "body": "Attached is the weekly summary.",
    }
    result = send_email(**email_input)
    record_tool_call("send_email", tool_input=email_input, tool_output=result)
    print(f"  -> send_email completed: {result['status']}")

    # Step 2 -- LLM decides to delete a file
    print("\n[Step 2] Agent decides to delete a file...")
    record_llm_call(
        model="claude-sonnet-4-20250514",
        provider="anthropic",
        input_tokens=500,
        output_tokens=150,
    )
    delete_input = {"path": "/tmp/old_report.txt"}
    result = delete_file(delete_input["path"])
    record_tool_call("delete_file", tool_input=delete_input, tool_output=result)
    print(f"  -> delete_file completed: {result['status']}")

    # Step 3 -- LLM decides to submit a form
    print("\n[Step 3] Agent decides to submit a form...")
    record_llm_call(
        model="claude-sonnet-4-20250514",
        provider="anthropic",
        input_tokens=500,
        output_tokens=150,
    )
    form_input = {
        "url": "https://example.com/api/feedback",
        "data": {"rating": 5, "comment": "Great service"},
    }
    result = submit_form(form_input["url"], form_input["data"])
    record_tool_call("submit_form", tool_input=form_input, tool_output=result)
    print(f"  -> submit_form completed: {result['status']}")

    print("\nAgent workflow complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from examples.alerts_and_drift._shared import ensure_demo_agent_config
    ensure_demo_agent_config(
        "sensitive-demo",
        {
            "sensitive_actions": [
                {"name": "send_email", "severity": "warning"},
                {"name": "delete_file", "severity": "critical"},
                {"name": "submit_form", "severity": "warning"},
            ],
        },
    )

    from tj.sdk.bootstrap import ensure_initialised
    ensure_initialised()

    print("=" * 60)
    print("OCW Sensitive Actions Alert Demo")
    print("=" * 60)

    run_sensitive_agent()

    print("\n" + "=" * 60)
    print("What to observe:")
    print("=" * 60)
    print(
        "If your ocw.toml has the sensitive_actions config shown at the\n"
        "top of this file, ocw should have fired alerts for each tool.\n"
        "\n"
        "Run these commands to inspect:\n"
        "\n"
        "  ocw alerts                  # see fired sensitive-action alerts\n"
        "  ocw status                  # agent overview with alert count\n"
        "  ocw traces                  # list recent traces\n"
    )
