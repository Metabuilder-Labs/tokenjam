"""Waste shape: tool-heavy chain.

A "research and report" agent that does one small planning LLM call, then
nine deterministic local tool calls (search -> fetch x3 -> extract x3 ->
format -> save), then one small final LLM call. Most of the session's
step count and latency is tool execution, not model calls; this exercises
tool-span capture and non-LLM cost/latency attribution, which the other
five workloads barely touch (they're mostly LLM-call shaped).

Every tool call uses a fixed, uniform argument SHAPE (a `query` string, a
`doc_id` string, a `section` string, a `sections` array, a `path` that
looks like a filesystem path); only the values vary per doc. That is the
signature `workflow_restructure.py` (the `script` analyzer) clusters on:
"ordered tuple of (tool_name, arg_shape)". One run of this workload is one
instance of that signature.

`script` needs `MIN_CLUSTER_INSTANCES = 20` sessions sharing an identical
signature before it will recommend anything; conservative by design, to
avoid false positives. A single cheap demo invocation (this workload's
default) will NOT clear that bar; it costs ~2 LLM calls regardless of
`--repeat`, so pass `--repeat 20` (or more) to actually build enough
session volume to see `script` fire; each repeat is a full fresh session
with the identical tool signature. Default is `--repeat 1` to stay cheap.
"""
from __future__ import annotations

import argparse
from typing import Callable, TypeVar

from tokenjam.sdk.agent import record_tool_call, watch
from tokenjam.sdk.attribution import attribution

from _shared import guarded_chat_completion, run_workload_main, set_default_environment

set_default_environment("sdk-workload-demo")

AGENT_ID = "sdk-workload-tool-heavy-chain"
TENANT_ID = "umbrella-research"
FEATURE = "research-agent"
PROMPT_TEMPLATE_ID = "research-chain-agent"
PROMPT_TEMPLATE_VERSION = "v3"

# Fixed local "document store"; deterministic, no randomness, so the tool
# outputs (and therefore the whole chain) are identical across repeats and
# across a before/after replay.
_DOCS = {
    "doc-001": {
        "title": "Q3 Incident Postmortems",
        "sections": {"summary": "Three P1 incidents, all resolved within SLA."},
    },
    "doc-002": {
        "title": "Vendor Security Reviews",
        "sections": {"summary": "Two vendors flagged for missing SOC2 renewal."},
    },
    "doc-003": {
        "title": "Cost Optimization Backlog",
        "sections": {"summary": "SDK telemetry corpus identified as the top gap."},
    },
}


def search_docs(query: str) -> list[str]:
    return list(_DOCS.keys())


def fetch_doc(doc_id: str) -> dict:
    return _DOCS[doc_id]


def extract_section(doc_id: str, section: str) -> str:
    return _DOCS[doc_id]["sections"].get(section, "")


def format_report(sections: list[str]) -> str:
    return "\n".join(f"- {s}" for s in sections)


def save_report(report: str, path: str) -> dict:
    return {"saved": True, "path": path, "bytes": len(report)}


_T = TypeVar("_T")


def _call_tool(name: str, fn: Callable[..., _T], **kwargs) -> _T:
    result = fn(**kwargs)
    record_tool_call(tool_name=name, tool_input=kwargs, tool_output={"result": result})
    return result


@watch(agent_id=AGENT_ID, tenant_id=TENANT_ID, feature=FEATURE)
def run_one_session(client, guard, dry_run: bool, model: str, run_index: int) -> None:
    with attribution(
        prompt_template_id=PROMPT_TEMPLATE_ID,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    ):
        guarded_chat_completion(
            client,
            guard,
            dry_run=dry_run,
            model=model,
            messages=[
                {"role": "system", "content": "You are a research-and-report agent."},
                {"role": "user", "content": "Summarize this week's operational reports."},
            ],
            max_tokens=20,
        )

        doc_ids = _call_tool("search_docs", search_docs, query="weekly operational reports")

        summaries: list[str] = []
        for doc_id in doc_ids:
            doc = _call_tool("fetch_doc", fetch_doc, doc_id=doc_id)
            section = _call_tool(
                "extract_section", extract_section, doc_id=doc_id, section="summary",
            )
            summaries.append(f"{doc['title']}: {section}")

        report = _call_tool("format_report", format_report, sections=summaries)
        _call_tool(
            "save_report", save_report, report=report,
            path=f"/reports/weekly-{run_index:03d}.md",
        )

        guarded_chat_completion(
            client,
            guard,
            dry_run=dry_run,
            model=model,
            messages=[
                {"role": "system", "content": "You are a research-and-report agent."},
                {"role": "user", "content": f"Report saved. One-sentence confirmation:\n{report}"},
            ],
            max_tokens=30,
        )
        print(f"  run {run_index + 1}: report saved, {len(doc_ids)} docs, {len(summaries)} sections")


def run(client, guard, dry_run: bool, model: str, repeat: int) -> None:
    for i in range(repeat):
        run_one_session(client, guard, dry_run, model, i)


def _main(client, guard, dry_run, model, args) -> None:
    run(client, guard, dry_run, model, args.repeat)


def _extra_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "Number of independent sessions to run with the identical tool "
            "signature (default: 1, cheap). Pass 20+ to clear the `script` "
            "analyzer's MIN_CLUSTER_INSTANCES gate."
        ),
    )


if __name__ == "__main__":
    run_workload_main(
        "Tool-heavy chain workload; exercises tool spans; --repeat 20+ to also target `script`.",
        _main,
        extra_args=_extra_args,
    )
