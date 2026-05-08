"""
AWS Bedrock agent with OCW observability.

Demonstrates calling Claude via AWS Bedrock's invoke_model API. All LLM calls
are captured by tj via the Bedrock integration patch.

SETUP NOTES:
    This example requires valid AWS credentials configured in your environment.
    Bedrock must be enabled in your AWS account, and you must have requested
    access to the Anthropic Claude models in the Bedrock console for your region.

    Typical setup:
        export AWS_ACCESS_KEY_ID=AKIA...
        export AWS_SECRET_ACCESS_KEY=...
        export AWS_DEFAULT_REGION=us-east-1   # or us-west-2, etc.

    Alternatively, configure credentials via ~/.aws/credentials or an IAM role.
    The region must have Bedrock model access enabled for the model used below.

Requirements:
    pip install boto3 tokenjam

Environment:
    AWS_DEFAULT_REGION or AWS_REGION  — required (must have Bedrock access)

Usage:
    python examples/single_provider/bedrock_agent.py
"""
from __future__ import annotations

import json
import os
import sys

region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")
if not region:
    print("ERROR: AWS_DEFAULT_REGION or AWS_REGION environment variable is required.")
    print("  export AWS_DEFAULT_REGION=us-east-1")
    print()
    print("You also need valid AWS credentials (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)")
    print("and Bedrock model access enabled in the specified region.")
    sys.exit(1)

import boto3  # noqa: E402

from tokenjam.sdk import watch  # noqa: E402
from tokenjam.sdk.integrations.bedrock import patch_bedrock  # noqa: E402

# Monkey-patch the boto3 Bedrock client BEFORE creating any instances.
patch_bedrock()

MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------


@watch(agent_id="bedrock-agent")
def run() -> str:
    client = boto3.client("bedrock-runtime", region_name=region)

    request_body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 256,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Explain the difference between supervised and "
                        "unsupervised learning in two sentences."
                    ),
                },
            ],
        }
    )

    response = client.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=request_body,
    )

    response_body = json.loads(response["body"].read())
    result_text = response_body["content"][0]["text"]
    return result_text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run()
    print(f"\nAgent response:\n{result}")

    print("\n--- OCW Observation ---")
    print("Session and LLM spans have been recorded.")
    print("Run 'tj status --agent bedrock-agent' to view telemetry.")
    print("Run 'tj cost --agent bedrock-agent' to see token costs.")
