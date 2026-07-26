#!/usr/bin/env python3
"""Measurement harness for the SDK workload corpus.

Runs one workload as a fresh subprocess against a scratch, throwaway
tokenjam home + DuckDB (never the operator's real `~/.tj`), then opens
that same scratch DB itself and runs the full `tj optimize` analyzer
suite over the telemetry the workload just produced, plus a check of
whatever alerts fired. Prints a table: which analyzers/alerts fired, how
many findings, and their `past_overspend_usd` where the analyzer carries
one; so "did this analyzer fire at all for SDK data" is answerable at a
glance, without hand-running `tj optimize` and cross-referencing source.

Isolation: a fresh subprocess per run (tokenjam's TracerProvider is a
process-global, set-once singleton. CLAUDE.md Critical Rule 11; so
reusing one interpreter across workloads would silently pin every later
run to the first run's config). `TJ_CONFIG` points the subprocess and
this harness at the same scratch `tj.toml`; `HOME` is also overridden as
defense in depth. `[api] port` is set to a non-default value so that even
if the operator happens to have a real `tj serve` running on the default
port, the workload's SDK bootstrap never mistakes it for the scratch
daemon and routes spans there instead of to the scratch DuckDB file.

Usage:
    python examples/sdk_workloads/runner.py <workload> [--dry-run]
        [--max-spend 2.00] [--model gpt-5.4-mini] [--out report.json]
        [-- <workload-specific args, e.g. --repeat 20>]

    python examples/sdk_workloads/runner.py --list
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import tomli_w

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKLOADS_DIR = Path(__file__).resolve().parent

WORKLOADS: dict[str, dict[str, str]] = {
    "repeated-prefix": {
        "script": "repeated_prefix.py",
        "target": "cache (optimize analyzer)",
        "description": "Repeated stable system-prompt prefix across 25 calls.",
    },
    "growing-context": {
        "script": "growing_context.py",
        "target": "resend (optimize analyzer)",
        "description": "Stateless chat loop re-sending full history every turn.",
    },
    "retry-loop": {
        "script": "retry_loop.py",
        "target": "RETRY_LOOP (alert, not an optimize analyzer)",
        "description": "Same tool call retried with identical arguments.",
    },
    "tool-heavy-chain": {
        "script": "tool_heavy_chain.py",
        "target": "tool spans; script (optimize analyzer, needs --repeat 20+)",
        "description": "Multi-step research agent, 9 tool calls per session.",
    },
    "oversized-model": {
        "script": "oversized_model.py",
        "target": "downsize (optimize analyzer)",
        "description": "Premium model used for trivial one-word answers.",
    },
    "streaming-disconnect": {
        "script": "streaming_disconnect.py",
        "target": "none today; documents an SDK instrumentation gap",
        "description": "Streaming response abandoned before the usage chunk.",
    },
}

# Static, source-verified context for every registered optimize analyzer.
# Printed alongside the LIVE fired/not-fired result so a reader knows WHY,
# not just whether. See examples/sdk_workloads/README.md for the full
# derivation of each of these from tokenjam/core/optimize/analyzers/*.py.
ANALYZER_NOTES: dict[str, str] = {
    "downsize": "Tiny-session case fires on 1+ session: input<5k tok, output<500 tok, tool_calls<=5, known cheaper same-family model.",
    "budget-projection": "Only fires when [budget.<provider>].usd > 0 is configured; the scratch config sets one.",
    "cache": "OpenAI is 'best-effort': needs >=20 calls & >=100k input tokens per model. tj's OpenAI integration never reads cached_tokens, so efficacy always reads 0% (data gap, not a real caching problem).",
    "cache-recommend": "Anthropic-only in v1; skips every non-Anthropic span outright. Never fires on pure OpenAI telemetry.",
    "resend": "Needs >=3 sessions and >=6 total LLM turns in the window.",
    "script": "Needs >=20 sessions sharing an identical (tool, arg_shape) signature; a single cheap run won't clear this.",
    "reuse": "Needs >=3 sessions sharing a planning-output skeleton with >=200 planning tokens each.",
    "trim": "Requires the optional tokenjam[bloat] extra (LLMLingua-2); skipped if not installed.",
    "subagent": "Persona-disabled for 'sdk'. SDK spans never set sub_agent_id (no Task-tool concept).",
    "summarize": "Scans on-disk prompt files at the analysis CWD (repo root here), gated only by window activity, not this workload's content. A fired card in this demo reflects the tokenjam repo's own docs, not something a workload wrote; run the harness from an unrelated CWD to isolate it.",
    "relearn": "Deliberately ignores the report window (its own module docstring: scoping it would cap its horizon) and scans the whole scratch DB's retention period instead, so its count reflects structural repetition across everything this corpus has written to that DB so far, not just one workload's run.",
    "verbosity": "Needs >=5 sessions in a cohort.",
    "deadweight": "Persona-disabled for 'sdk'; reads on-disk Claude Code transcripts / .mcp.json, which don't exist for an SDK-only corpus.",
}

# Attributes to look for, in order, when summarising a finding generically
# (every analyzer names its example/candidate list differently).
_COUNT_ATTRS = [
    "candidate_sessions", "driver_sessions",
    "examples", "candidates", "clusters", "rows", "flagged",
    "uncached_agents", "thrash_agents", "lookback_miss_agents",
]


def _write_scratch_config(scratch_dir: Path, max_spend: float) -> Path:
    db_path = scratch_dir / "telemetry.duckdb"
    config = {
        "version": "1",
        "storage": {"path": str(db_path)},
        # Non-default port: even if a real `tj serve` happens to be running
        # on 7391, the workload's SDK bootstrap must never mistake it for
        # this scratch run and route spans to the operator's real daemon.
        "api": {"port": 58391, "enabled": False},
        # A generous ceiling (well above any --max-spend a demo run would
        # use) purely so the budget-projection analyzer has a configured
        # ceiling to project against; it fires on NOTHING without one.
        # `plan = "api"`: this corpus is genuinely API-billed, not a
        # subscription. Without it ProviderBudget.plan defaults to None and
        # sessions get plan_tier="unknown", which is a real state (some
        # dollar-figure renderers caveat it) but not the honest one here.
        "budget": {"openai": {"usd": max(max_spend * 25, 50.0), "plan": "api"}},
    }
    toml_path = scratch_dir / "tj.toml"
    with toml_path.open("wb") as f:
        tomli_w.dump(config, f)
    return toml_path


def _run_workload_subprocess(
    workload: str, toml_path: Path, scratch_home: Path,
    dry_run: bool, max_spend: float, model: str | None, extra_args: list[str],
) -> int:
    import os

    script = WORKLOADS_DIR / WORKLOADS[workload]["script"]
    cmd = [sys.executable, str(script), "--max-spend", str(max_spend)]
    if dry_run:
        cmd.append("--dry-run")
    if model:
        cmd += ["--model", model]
    cmd += extra_args

    env = os.environ.copy()
    env["TJ_CONFIG"] = str(toml_path)
    env["HOME"] = str(scratch_home)
    scratch_home.mkdir(parents=True, exist_ok=True)

    print(f"[runner] scratch home:   {scratch_home}")
    print(f"[runner] scratch config: {toml_path}")
    print(f"[runner] running: {' '.join(cmd)}\n")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    return proc.returncode


def _finding_summary(finding: Any) -> tuple[bool, str]:
    if finding is None:
        return False, "no data"
    parts: list[str] = []
    total_count = 0
    for attr in _COUNT_ATTRS:
        val = getattr(finding, attr, None)
        if isinstance(val, list):
            total_count += len(val)
        elif isinstance(val, int) and attr in ("candidate_sessions", "driver_sessions"):
            total_count += val
    usd = getattr(finding, "past_overspend_usd", None)
    if total_count:
        parts.append(f"{total_count} candidate(s)")
    if usd is not None:
        parts.append(f"${usd:,.4f} past_overspend_usd")
    fired = total_count > 0 or (usd is not None and usd > 0)
    return fired, ("; ".join(parts) if parts else "ran, nothing cleared threshold")


def _build_and_render_report(toml_path: Path, since, until, out_path: Path | None) -> None:
    from tokenjam.core.config import load_config
    from tokenjam.core.db import open_db
    from tokenjam.core.models import AlertFilters
    from tokenjam.core.optimize import (
        ANALYZER_ORDER,
        build_report,
        disabled_analyzers_for_persona,
        report_to_dict,
    )

    config = load_config(path=str(toml_path))
    db = open_db(config.storage)

    report = build_report(db, config, since=since, until=until)
    disabled = disabled_analyzers_for_persona(report.persona)

    print(f"\n=== Window: {report.window.spans} spans, {report.window.sessions} session(s), "
          f"persona={report.persona!r} ===\n")

    rows: list[tuple[str, str, str, str]] = []
    for name in ANALYZER_ORDER:
        if name in disabled:
            rows.append((name, "SKIPPED", "persona-disabled for this window", ANALYZER_NOTES.get(name, "")))
            continue
        if name == "downsize":
            fired, detail = _finding_summary(report.downgrade)
        elif name == "budget-projection":
            fired = len(report.budgets) > 0
            detail = f"{len(report.budgets)} provider(s) projected" if fired else "no configured budget matched"
        else:
            fired, detail = _finding_summary(report.findings.get(name))
        rows.append((name, "FIRED" if fired else "ran, no finding", detail, ANALYZER_NOTES.get(name, "")))

    _print_table(
        ["analyzer", "status", "detail"],
        [(n, s, d) for n, s, d, _note in rows],
    )
    print()
    for name, _status, _detail, note in rows:
        if note:
            print(f"  - {name}: {note}")

    alerts = db.get_alerts(AlertFilters(since=since, limit=1000))
    print(f"\n=== Alerts fired in window: {len(alerts)} ===")
    if alerts:
        by_type: dict[str, int] = {}
        for a in alerts:
            by_type[a.type.value if hasattr(a.type, "value") else str(a.type)] = (
                by_type.get(a.type.value if hasattr(a.type, "value") else str(a.type), 0) + 1
            )
        for alert_type, count in sorted(by_type.items()):
            print(f"  - {alert_type}: {count}")

    cost_row = db.conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0.0) FROM spans WHERE start_time >= $1 AND start_time < $2",
        [since, until],
    ).fetchone()
    total_cost = cost_row[0] if cost_row else 0.0
    print(f"\n=== Actual spend recorded by tj's own cost engine: ${float(total_cost):.4f} ===")

    if out_path is not None:
        import json

        payload = {
            "report": report_to_dict(report),
            "alerts": [
                {"type": a.type.value if hasattr(a.type, "value") else str(a.type), "detail": a.detail}
                for a in alerts
            ],
            "actual_spend_usd": float(total_cost),
        }
        out_path.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\n[runner] full report written to {out_path}")


def _print_table(headers: list[str], rows: list[tuple]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("workload", nargs="?", choices=sorted(WORKLOADS), help="Which workload to run.")
    parser.add_argument("--list", action="store_true", help="List available workloads and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Zero API calls, zero spend.")
    parser.add_argument("--max-spend", type=float, default=2.00, help="Hard USD ceiling (default: $2.00).")
    parser.add_argument("--model", type=str, default=None, help="Override the workload's default model.")
    parser.add_argument("--out", type=Path, default=None, help="Write the full JSON report to this path.")
    parser.add_argument("--keep-scratch", action="store_true", help="Don't print a cleanup reminder; scratch dirs are never auto-deleted.")
    args, extra_args = parser.parse_known_args()

    if args.list or not args.workload:
        _print_table(
            ["workload", "target", "description"],
            [(name, w["target"], w["description"]) for name, w in sorted(WORKLOADS.items())],
        )
        return

    from tokenjam.utils.time_parse import utcnow
    from datetime import timedelta

    scratch_dir = Path(tempfile.mkdtemp(prefix="tj-sdk-workload-"))
    toml_path = _write_scratch_config(scratch_dir, args.max_spend)
    scratch_home = scratch_dir / "home"

    since = utcnow() - timedelta(seconds=5)
    returncode = _run_workload_subprocess(
        args.workload, toml_path, scratch_home,
        args.dry_run, args.max_spend, args.model, extra_args,
    )
    until = utcnow() + timedelta(seconds=5)

    if returncode == 2:
        print("\n[runner] workload aborted via the spend guard; building a report from partial telemetry.\n")
    elif returncode != 0:
        print(f"\n[runner] WARNING: workload exited with code {returncode}; telemetry may be incomplete.\n")

    _build_and_render_report(toml_path, since, until, args.out)
    print(f"\n[runner] scratch DB left at {scratch_dir} for inspection (not auto-deleted).")


if __name__ == "__main__":
    main()
