"""The zero-install, zero-config first run (issue #6).

The 15-second time-to-first-value path that lets a brand-new user, reached via
``npx tokenjam`` / ``uvx tj`` with **no** pip env, **no** daemon and **no**
onboarding, see how much of their recent Claude Code spend was avoidable,
straight from the JSONL files ccusage already reads
(``~/.claude/projects/*.jsonl``).

Design (what makes it "zero-setup"):

  * It opens a **transient in-memory** DuckDB (``InMemoryBackend``) — nothing is
    written to ``~/.tj``, no config is read or written, no daemon is started or
    contacted. Each run re-reads the JSONL fresh.
  * It backfills the on-disk Claude Code sessions into that transient DB via the
    existing :func:`tokenjam.core.backfill.ingest_claude_code` parser, then runs
    the same ``build_report`` / ``COST_ANALYZERS`` /
    ``inbox_contribution.gather_rollup_population`` path the Review inbox and
    the dashboard use, and reports ONE number: the avoidable dollars summed
    across every analyzer this window's persona can actually act on — minus
    ``relearn``, deliberately dropped here on runtime-cost grounds (see
    ``_OVERSPEND_SKIP_ANALYZERS``), so this screen is a declared LOWER BOUND
    on what the Review inbox can show after ``tj onboard``, never a
    guaranteed match.

**The screen is deliberately minimal, and its shape is a founder decision.** It
is three things and nothing else: what tj read, one avoidable-dollars sentence,
and the pointer to ``npx tokenjam onboard``. It carries no quota composition (no
re-read share, no net-new share, no ``/compact`` counting), no statusline
preview, no per-analyzer titles / evidence / fixes, and no boxes. A first run is
a hook, not a report; the per-finding detail lives in the Review inbox behind
onboarding.
If you are about to add a panel here, that is the decision you are reversing.

Copy rules: no em dashes in user-facing strings; "avoidable", never "wasted" or
"saved"; near-monochrome, with the single ``accent`` reserved for the dollar
figure and the typeable command (see :mod:`tokenjam.utils.theme`).

This has no public/typeable command name — ``cli/main.py``'s no-subcommand
branch invokes ``cmd_quickstart`` directly (via ``ctx.invoke``) when the npm
wrapper's ``TJ_NPX_ZERO_INSTALL_REPORT`` env var is set, so it never opens the
on-disk DB or trips the daemon's write lock either way.

Honesty discipline (CLAUDE.md Rule 14): the figure is a *measured*, already-
incurred avoidable total over the ingested window, never a projected saving.
"""
from __future__ import annotations

import json as _json
import re as _re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import click

from tokenjam.cli.backfill_progress import backfill_progress, transient_status
from tokenjam.core.backfill import CLAUDE_CODE_PROJECTS_ROOT, ingest_claude_code
from tokenjam.core.db import InMemoryBackend
from tokenjam.utils.formatting import console, err_console, format_cost
from tokenjam.utils.theme import ACCENT
from tokenjam.utils.time_parse import parse_since, utcnow

# First-run cap (#13): on a large ~/.claude history a full backfill into the
# transient DB blows past the <30s time-to-first-value goal. We cap the headline
# to the most-recent N sessions (bounded work, well under 30s even on thousands
# of sessions) and disclose the cap; `--full` lifts it for the complete picture.
# ~300 sessions keeps the slowest plausible session shapes comfortably in budget.
DEFAULT_MAX_SESSIONS = 300

#: The one sentence that keeps this screen from reading as Claude-Code-only.
#: tokenjam is not: `tj onboard` configures a Codex flow too, and the daemon
#: mounts an OTLP receiver any OTel-instrumented SDK or API agent can post to.
#:
#: EVERY source named here was verified against the code, because a source on
#: screen that does not work is worse than the framing it was added to fix:
#:   * Codex CLI sessions — `core/ingest_adapters/codex.py` parses
#:     `~/.codex/sessions/**/rollout-*.jsonl`, reached by `tj backfill codex`
#:     and by `tj onboard --codex`, with its own passing test suites.
#:   * OTel spans — `api/routes/otlp.py` mounts `POST /v1/traces` and
#:     `POST /v1/logs` unconditionally in the daemon app (`api/app.py`), and
#:     `core/ingest_adapters/otlp.py` backs the offline import.
#:
#: The OTel half says "your SDK or API agents send it", NOT "any OTel app", and
#: the distinction is load-bearing. `api/routes/_body.py` decodes the request
#: with `json.loads`: the receiver is OTLP/HTTP **JSON only**, there is no
#: protobuf decoder and no gRPC listener at all, so a stock OTel SDK left on its
#: default `http/protobuf` exporter gets a 400. `api/middleware.py` also
#: requires the Bearer ingest secret onboard writes. Both are fine for an agent
#: you point at tokenjam on purpose, which is what the sentence describes;
#: neither supports a drop-in "works with anything OTel" claim, so do not
#: upgrade this wording without adding a protobuf decoder first.
#:
#: **Codex is deliberately NOT named, and this one is not a code question.**
#: A real parser exists (`core/ingest_adapters/codex.py`, reached by
#: `tj backfill codex` and `tj onboard --codex`) and it passes its own suites,
#: which is exactly why a later reader will be tempted to "fix" this omission.
#: Do not. Shipping-readiness is an operator call, not a code-presence one, and
#: the operator's is that tokenjam is Claude Code only for now. Advertising a
#: half-supported source on the FIRST screen a stranger sees is the specific
#: defect this sentence was verified against in the first place.
#:
#: Two more things are deliberately NOT named. **Metrics**: `POST /v1/metrics` is a
#: stub that returns 200 and discards the body, so "OTel" here means spans and
#: says so. **The MCP server**: `mcp/server.py` exposes only read/query and
#: apply tools, with no ingest tool at all, so it is not a source. The copy this
#: replaced claimed SDK traffic arrives "from OTel spans or the tokenjam MCP
#: server", and the second half of that was never true.
#:
#: Langfuse and Helicone backfills also exist (`tj backfill langfuse` /
#: `helicone`) and work. They are left off for length: they import from another
#: tool you already run rather than describing a way tokenjam watches your own
#: agents, and this block must not grow into a feature list.
_OTHER_SOURCES = (
    "It also reads OTel spans your SDK or API agents send it."
)

#: Held on screen while the analyzer pass runs. Present tense, the product's
#: vocabulary, and deliberately claim-free: it says what is being looked for,
#: never how much was found. See `transient_status` for the erase contract.
_ANALYZING_STATUS = "Finding avoidable spend across your sessions…"


@click.command("quickstart")
@click.option("--since", default="30d",
              help="Window for analysis (e.g. 7d, 30d, 2026-03-01). Default 30d.")
@click.option("--root", "root_path", default=None,
              help=f"Override Claude Code projects root (default {CLAUDE_CODE_PROJECTS_ROOT}).")
@click.option("--full", is_flag=True,
              help=f"Process the full history (default caps at the most-recent "
                   f"{DEFAULT_MAX_SESSIONS} sessions for a fast first run).")
@click.option("--json", "output_json", is_flag=True,
              help="Emit machine-readable JSON.")
@click.pass_context
def cmd_quickstart(ctx: click.Context, since: str, root_path: str | None,
                   full: bool, output_json: bool) -> None:
    """Zero-setup first run: how much of your recent spend was avoidable.

    Reads the same ~/.claude/projects/*.jsonl files ccusage does: no pip env,
    no daemon, no onboarding. On a large history the first run caps at the
    most-recent sessions for speed (use `--full` for everything). Run
    `npx tokenjam onboard` afterwards to go deeper (live capture, the dashboard,
    and the zero-token statusline).
    """
    root = Path(root_path).expanduser() if root_path else CLAUDE_CODE_PROJECTS_ROOT
    if not root.exists():
        _render_no_logs(root, output_json)
        return

    try:
        since_dt = parse_since(since)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="'--since'") from exc
    until_dt = utcnow()

    # Transient in-memory DB — nothing persisted, no config, no daemon.
    max_sessions = None if full else DEFAULT_MAX_SESSIONS
    db = InMemoryBackend()

    # Ingest is the only silent stretch in the whole command — on a large
    # history it can run tens of seconds with zero output otherwise. An
    # honest status line lands within ~1s of launch, then the shared
    # streaming counter (#443/#444's `backfill_progress`) advances per
    # session through to render. `--json` must keep stdout byte-for-byte
    # clean, so both route to the stderr console when JSON is requested —
    # never suppressed outright, so a human watching a scripted run still
    # sees it's alive.
    status_console = err_console if output_json else console
    # NO pre-ingest session total, deliberately. The only cheap pre-scan
    # available is a stat()-only count of `.jsonl` FILES, and a Claude Code
    # session is more than one file: every `Task` dispatch writes its own
    # `subagents/agent-*.jsonl` sharing the parent's `session_id`. On a real
    # corpus that made the header announce roughly twice the number the report
    # then printed, which reads as two answers to one question.
    #
    # Filtering the pre-scan down to main-thread files was measured and lands
    # on the report's number today (154 files, 154 sessions), but it cannot be
    # relied on: the pre-scan filters by FILE MTIME before parsing while the
    # report filters by SPAN TIMESTAMP after it, so a transcript touched inside
    # the window whose turns all predate it counts in one and not the other. A
    # missing number is fine; two numbers that disagree is the bug. The counter
    # therefore runs without a denominator, and the report states the one
    # session count this screen makes.
    status_console.print(f"[dim]{_pre_ingest_status(since, max_sessions)}[/dim]")
    with backfill_progress(None, console=status_console) as progress_cb:
        result = ingest_claude_code(db, root=root, since=since_dt,
                                    max_sessions=max_sessions, progress=progress_cb)

    if result.sessions_ingested == 0:
        _render_no_sessions(result, since, output_json)
        return

    if output_json:
        from tokenjam.core.context_diagnostic import (
            compute_context_diagnostic,
            diagnostic_to_dict,
        )
        from tokenjam.core.session_timeline import (
            compute_session_timeline,
            timeline_to_dict,
        )
        payload = {
            "quota_composition": diagnostic_to_dict(
                compute_context_diagnostic(db.conn, since_dt, until_dt)),
            "session_timeline": timeline_to_dict(
                compute_session_timeline(db.conn)),
            "backfill": {
                "sessions_ingested": result.sessions_ingested,
                "spans_ingested": result.spans_ingested,
                "project_count": result.project_count,
                "total_cost_usd": round(result.total_cost_usd, 6),
                "limit_reached": result.limit_reached,
                "max_sessions": max_sessions,
            },
        }
        click.echo(_json.dumps(payload, default=str))
        return

    # Computed only on the human path: `--json` must stay byte-clean AND fast,
    # and the analyzers are the one materially expensive step after ingest.
    #
    # `fallback_sessions` is only that: the analyzed population comes off the
    # report's own window, and `sessions_ingested` stands in ONLY when the
    # computation could not run (there is then no figure to mispair it with).
    # The two genuinely differ: the ingest picks the most-recent N session FILES
    # by mtime, while the report filters by span timestamp, so a real corpus can
    # ingest 300 files and analyze 143 sessions. Printing the ingest count beside
    # a figure summed over the analyzed one is exactly the mixed-population
    # defect this screen must not have.
    #
    # This is the last slow stretch, and it used to be silent: measured on a
    # real corpus, ~14s elapses between the final backfill line and the first
    # line of the report, long enough to read as a hang. `transient_status`
    # holds one self-erasing line built from the SAME `Progress` construction
    # the backfill counter above uses, so the two phases read as one process,
    # and it erases before the report renders. The message names what is
    # happening in the product's own terms and promises NO number: nothing on
    # this screen may claim a figure, a count or an all-clear while the
    # computation that would establish it is still running.
    with transient_status(_ANALYZING_STATUS):
        avoidable = _compute_avoidable_total(
            db, since_dt, until_dt,
            fallback_sessions=result.sessions_ingested,
            population_capped=max_sessions is not None and result.limit_reached,
        )

    _render(avoidable, fallback_sessions=result.sessions_ingested)


# ────────────────────────── avoidable total ────────────────────────────────
#
# ONE number: the ALREADY-INCURRED avoidable dollars over the ingested window,
# summed across every analyzer this window's persona can act on. Quickstart is
# a hook, not a report, so it names no analyzer, shows no evidence and hands
# out no fix; that detail lives in the Review inbox behind `tj onboard`.
#
# Contract (repo CLAUDE.md, "THE per-analyzer dollar-field contract"):
#   * `past_overspend_usd` is the canonical figure, observed over the analyzed
#     window, past tense. It is NEVER paced, projected, or multiplied by a
#     30-day ratio, and `observed_cost_usd` is never summed into it.
#   * `None` means "not measured", never `$0`. A `None` return here renders no
#     sentence at all, NOT an empty state: "we could not compute it" and "we
#     computed it and it is zero" are different claims and must read that way.
#   * The sum goes through `inbox_contribution.gather_rollup_population` — the
#     one gatherer every other surface (Review inbox, dashboard, CLI) reaches
#     `cost_proposals.past_overspend_rollup` through, deduped by proposal
#     `signature`. That makes this screen a LOWER BOUND on the number the user
#     sees after onboarding, not a guaranteed match: see
#     `_OVERSPEND_SKIP_ANALYZERS` for the one analyzer (`relearn`) this screen
#     deliberately omits and why. Do not hand-roll a second sum here; the
#     `inbox_contribution` module docstring records what happened last time
#     there were two.
#
# Analyzer selection is NOT re-implemented here, and there is no second
# persona filter. `build_report` is handed `COST_ANALYZERS` and does the
# persona gating itself (`PERSONA_DISABLED_ANALYZERS`, a TRUE skip before
# dispatch), so a Claude-Code-dominant window never spends query time on a
# finding that persona could not act on, and the enabled set is derived from
# the live registry rather than transcribed here. The one deviation is a
# runtime bound for this zero-install path, at `_OVERSPEND_SKIP_ANALYZERS`.
#
# Pricing: quickstart reads no config, so a plan tier is never declared and
# `pricing_mode_for(dominant_plan(...))` would be `"unknown"`. `cmd_status`'s
# stricter api-only gate is deliberately NOT reused: the product stance is to
# assume API pricing.


#: Plain-English shape phrases, one per cost analyzer, for the explanatory
#: sentence under the figure. The reference for what each analyzer MEANS is the
#: "what each analyzer sees" table in `.claude/product-state/positioning.md`;
#: these are that table's left column said out loud, with no internal analyzer
#: name reaching the screen.
#:
#: Two analyzers deliberately share one phrase: `downsize` (work that should
#: have been delegated off an expensive main thread) and `subagent` (the worker
#: it WAS delegated to was oversized) are different findings, but from the
#: reader's side both are "a model bigger than the job needed", and the sentence
#: is a gesture at the mechanism rather than an inventory. Duplicates collapse.
#:
#: A contributing analyzer with NO entry here does not silently vanish: it
#: forces the non-exhaustive phrasing (see `_shape_clause`), so an analyzer
#: added to `COST_ANALYZERS` without a phrase degrades to a vaguer but still
#: TRUE sentence rather than to a confident list that omits it.
_ANALYZER_SHAPES: dict[str, str] = {
    "downsize":        "oversized models",
    "subagent":        "oversized models",
    "resend":          "context re-sent every turn",
    "relearn":         "mistakes that repeated without a fix",
    "deadweight":      "MCP servers connected but never used",
    "summarize":       "always-loaded files longer than they need to be",
    "trim":            "prompt text that carried no weight",
    "cache":           "requests that never reused a cache",
    "cache-recommend": "requests that never reused a cache",
    "reuse":           "plans re-derived from scratch each time",
    "script":          "tool sequences re-run by an agent instead of a script",
    "verbosity":       "answers longer than the task needed",
}

#: How many shapes the sentence may name. Three is the founder-approved shape;
#: past that the sentence stops being one plain line.
_MAX_SHAPES = 3

#: The half of the sentence that is true for ANY mix, used alone when the
#: contributing analyzers map to no phrasing at all. It still does the more
#: important of the sentence's two jobs: the likeliest misread of a bare dollar
#: figure is that it is what the sessions COST.
_MEANING_ONLY = (
    "That is the part a change to your setup would have removed, not what "
    "your sessions cost."
)


def _shape_clause(contributors: tuple[str, ...]) -> str:
    """The explanatory sentence, derived from what ACTUALLY contributed.

    `contributors` is the analyzers that put a non-zero dollar figure into this
    run's rollup, biggest first. Naming a fixed list instead would describe a
    `deadweight`-dominated corpus by causes that were not its own — the same
    defect class as printing a session count from a different population than
    the dollars.

    The list is never presented as exhaustive unless it genuinely is: more
    contributors than fit, or any contributor this module has no phrase for,
    switches the sentence to "including".
    """
    shapes: list[str] = []
    for name in contributors:
        phrase = _ANALYZER_SHAPES.get(name)
        if phrase and phrase not in shapes:
            shapes.append(phrase)
    if not shapes:
        return _MEANING_ONLY

    shown = shapes[:_MAX_SHAPES]
    exhaustive = (
        len(shown) == len(shapes)
        and all(name in _ANALYZER_SHAPES for name in contributors)
    )
    joined = ", ".join(shown)
    if exhaustive:
        return f"That is the part a change to your setup would have removed: {joined}."
    return (
        f"That is the part a change to your setup would have removed, "
        f"including {joined}."
    )


@dataclass(frozen=True)
class AvoidableTotal:
    """The window's avoidable dollars, and the population they were summed over.

    Constructed only when the rollup actually ran. ``usd`` may legitimately be
    ``0.0`` — that is the honest "nothing avoidable found" state, and it renders
    as a sentence saying so, never as a ``$0.00`` styled like a finding. When
    the computation could not run at all the caller gets ``None`` instead, and
    no sentence is rendered.

    ``sessions`` is the count the analyzers actually reasoned over (the report's
    own window summary), carried on the same object so the render cannot pair
    this figure with a count from another population. The screen prints this one
    number in both places it states a population.

    ``contributors`` is the analyzers that put a non-zero dollar figure into the
    rollup, biggest first. It exists so the explanatory sentence can describe
    THIS run's causes rather than a fixed list; see ``_shape_clause``.
    """
    usd:          float
    sessions:     int
    contributors: tuple[str, ...] = ()


# The ONE analyzer dropped from `COST_ANALYZERS` for this figure, and the
# only one: `relearn`.
#
# This is a runtime bound, not a second persona filter — persona gating stays
# entirely inside `build_report` (`PERSONA_DISABLED_ANALYZERS`, a true skip
# before dispatch), which is why the full tuple is passed through this helper
# rather than rebuilt.
#
# THIS IS A DECLARED SUBSET, NOT AN OVERSIGHT — read this before "fixing" it
# by deleting the skip. `_compute_avoidable_total` folds relearn's open
# clusters into its rollup exactly like the Review inbox does, through
# `inbox_contribution.gather_rollup_population` (the same function the CLI's
# `tj relearn cost-proposals` and the API's `GET /relearn/cost-proposals`
# use) — so relearn is NOT excluded because it has no figure to contribute
# any more; it has one, on the canonical field, same as every other
# analyzer here. It is excluded because `run(ctx)` deliberately scans
# unbounded retained history rather than the report window (relearn's whole
# signal is recurrence across history), which makes it the one analyzer here
# whose cost does not shrink with quickstart's already-capped corpus.
# Measured on a real 300-session corpus it cost 27.9s of the 35.0s the whole
# analyzer set took — on a zero-install path whose own status line exists
# because ~14s of silence already reads as a hang, tripling that is not a
# trade this screen makes for one more contributor's money.
#
# The consequence: `report.findings.get("relearn")` is always `None` here, so
# `gather_rollup_population` always folds in ZERO relearn rows on this
# screen — never a fabricated number, but never relearn's real one either.
# This screen's total is therefore a DELIBERATE SUBSET of what the Review
# inbox can show after `tj onboard` (which runs relearn out of band, on the
# daemon's own cadence, where this runtime cost is amortised rather than
# paid inline): a lower bound, not a guaranteed match. If relearn's runtime
# ever drops enough to affordably run inline (or this screen grows a budget
# for it), delete this helper and pass `COST_ANALYZERS` whole — the rollup
# plumbing already has nothing left to change to pick it up.
_OVERSPEND_SKIP_ANALYZERS = frozenset({"relearn"})

#: Analyzers that reason over the raw on-disk transcript tree directly
#: (`root.rglob(...)`), never `ctx.conn` — so their population is NOT bounded
#: by `--max-sessions`/`DEFAULT_MAX_SESSIONS`. `deadweight` is the one in
#: `COST_ANALYZERS` that does this (`compute_deadweight_finding` walks every
#: matching transcript under the projects root for the window). When
#: quickstart's ingest actually truncated at the session cap
#: (`result.limit_reached`), a disk-scan analyzer's figure covers strictly
#: MORE sessions than the ones ingested into the DB and rendered on screen —
#: a population mismatch the existing magnitude ceiling (`_over_ceiling`)
#: cannot catch, since a SMALLER out-of-population figure still clears it,
#: and which matters more now that the screen SUMS the analyzers rather than
#: picking one: an out-of-population figure would be added to, not compared
#: against, the in-population ones.
#: Excluded outright rather than rescaled: there is no honest way to shrink
#: an unbounded-population figure down to the capped one without inventing a
#: number nothing measured.
_POPULATION_UNBOUNDED_ANALYZERS = frozenset({"deadweight"})


def _overspend_analyzers(names: tuple[str, ...], *, population_capped: bool = False) -> list[str]:
    """`names` minus the analyzers that cannot produce a past-overspend figure,
    and — only when the session ingest was actually capped — minus the ones
    whose own population isn't bounded by that cap either."""
    skip = _OVERSPEND_SKIP_ANALYZERS
    if population_capped:
        skip = skip | _POPULATION_UNBOUNDED_ANALYZERS
    return [n for n in names if n not in skip]


def _compute_avoidable_total(
    db, since: datetime, until: datetime, *,
    fallback_sessions: int = 0, population_capped: bool = False,
) -> AvoidableTotal | None:
    """Avoidable dollars over the ingested window, summed across the analyzers.

    Returns ``None`` — meaning "not measured" — only when the computation could
    not run (anything raised). A window the analyzers examined and found nothing
    in returns ``AvoidableTotal(usd=0.0, ...)``, which is a DIFFERENT state and
    renders differently: unknown prints no sentence, known-and-empty prints one
    saying so. Never raises and never fabricates a figure.

    `population_capped` is True only when the ingest itself truncated at the
    session cap (`result.limit_reached`) — see `_POPULATION_UNBOUNDED_ANALYZERS`
    for why that also has to drop any analyzer whose own data source ignores
    the cap.

    The config is an in-memory `TjConfig()` default: quickstart reads and
    writes NO config file and NO on-disk DB, and that must stay true.
    """
    try:
        from tokenjam.core.config import TjConfig
        from tokenjam.core.optimize import build_report, inbox_contribution
        from tokenjam.core.optimize.cost_proposals import (
            COST_ANALYZERS,
            cost_proposals_from_report,
        )

        # In-memory defaults only. Matches what `load_config` synthesises when
        # no config file exists, so no file is read and none is written.
        config = TjConfig(version="1")
        report = build_report(
            db=db, config=config, since=since, until=until,
            findings=_overspend_analyzers(COST_ANALYZERS, population_capped=population_capped),
        )
        window_days = max((until - since).total_seconds() / 86400.0, 1.0)
        proposals = cost_proposals_from_report(
            report, config, window_days=window_days,
        )

        # Defensibility ceiling. An avoidable figure LARGER than what the whole
        # window cost is self-refuting no matter how the analyzer derived it,
        # and a reader can disprove it from their own billing. (It happens: some
        # analyzers count a population wider than what quickstart ingested, e.g.
        # a per-session tax summed over every transcript on disk while the
        # transient DB holds the capped, most-recent subset.) Matrix rule: show
        # the LARGEST number the derivation can legitimately support. Such a
        # proposal is DROPPED, never rescaled: rescaling would invent a number
        # nothing measured.
        window_cost = float(getattr(getattr(report, "window", None),
                                    "total_cost_usd", 0.0) or 0.0)
        eligible = [
            p for p in proposals
            if not (p.past_overspend_usd is not None
                    and window_cost > 0.0
                    and p.past_overspend_usd > window_cost)
        ]

        # THE rollup, not a local sum: `gather_rollup_population` is the same
        # gatherer the Review inbox CLI/API routes use (dedup-by-signature,
        # canonical field, and — unlike a bare `past_overspend_rollup` call —
        # it also folds in relearn's open clusters as ordinary rows, so a
        # population that includes relearn's money is not something this call
        # site has to remember separately). `window_days` doubles as the label
        # every row's window has to match exactly for relearn's contribution
        # to join; see the module docstring below on why this screen still is
        # NOT a strict guarantee of parity with the Review inbox.
        rollup = inbox_contribution.gather_rollup_population(
            eligible, getattr(report, "findings", {}).get("relearn"),
            window_days=window_days,
        )
        usd = float(rollup.get("past_overspend_usd") or 0.0)

        # The population comes off the REPORT's own window summary, which is
        # the set the analyzers queried, not the set the ingest happened to
        # load. `fallback_sessions` is a last resort for a report shape that
        # carries no count; it is never preferred over the real one.
        analyzed = int(getattr(getattr(report, "window", None), "sessions", 0) or 0)

        # Who actually paid into the total, biggest first. Read off the rollup's
        # own per-analyzer breakdown rather than re-walking the proposals, so
        # the explanation and the figure can never describe different sets.
        # Dollar-bearing entries only: `by_analyzer` also admits a row that
        # contributed tokens alone, and the sentence explains a dollar figure.
        contributors = tuple(
            entry["analyzer"]
            for entry in sorted(
                (e for e in rollup.get("by_analyzer", []) if (e.get("usd") or 0) > 0),
                key=lambda e: float(e.get("usd") or 0.0),
                reverse=True,
            )
        )
    except Exception:
        # A first run must never die on the analyzers. Worst case no sentence
        # is rendered; the rest of the screen is unaffected.
        return None

    return AvoidableTotal(usd=max(usd, 0.0),
                          sessions=analyzed or max(fallback_sessions, 0),
                          contributors=contributors)


# ───────────────────────────── rendering ──────────────────────────────────

_SINCE_UNIT_WORDS = {"d": "days", "h": "hours", "m": "minutes"}


def _describe_window(since: str) -> str:
    """Human-readable window phrasing for the pre-ingest status line.

    Special-cases the relative `Nd`/`Nh`/`Nm` shapes `--since` accepts (the
    default is `30d`) into "last N days"; anything else (a literal date, an
    ISO datetime) falls back to "history since <value>" rather than guessing.
    """
    m = _re.match(r"^(\d+)([mhd])$", since.strip())
    if m:
        amount, unit = m.groups()
        return f"last {amount} {_SINCE_UNIT_WORDS[unit]}"
    return f"history since {since}"


def _pre_ingest_status(since: str, max_sessions: int | None) -> str:
    """Honest status line printed BEFORE ingest starts.

    Ingest was previously the one silent stretch in the whole command —
    ~40s of dead cursor on a large history before any output. This line
    lands within ~1s of launch; `backfill_progress`'s streaming counter
    takes over immediately after.

    It states the cap WITHOUT a number. The cap is a file budget
    (`DEFAULT_MAX_SESSIONS` bounds a glob over `~/.claude/projects`), and a file
    count is not a session count, so printing it here would put a second,
    larger number on screen above the one the report makes. See the call site
    for why the honest-looking fix (count main-thread files only) is not
    reliable enough to print.
    """
    window = _describe_window(since)
    scope = " (capped for a fast first run)" if max_sessions is not None else ""
    return f"Reading your {window} of Claude Code history{scope}…"


def _render_no_logs(root, output_json: bool) -> None:
    if output_json:
        click.echo(_json.dumps({"error": "no_claude_code_logs", "root": str(root)}))
        return
    console.print(
        f"\n[yellow]No Claude Code logs found at {root}.[/yellow]\n"
        "[dim]This reads your ~/.claude/projects/*.jsonl session logs. This is "
        "normal if Claude Code hasn't run on this machine yet. Use it for a "
        "session, then run [bold]npx tokenjam[/bold] again. Ready to go "
        "deeper now? [bold]npx tokenjam onboard[/bold].[/dim]\n"
    )


def _render_no_sessions(result, since: str, output_json: bool) -> None:
    if output_json:
        click.echo(_json.dumps({"error": "no_sessions_in_window", "since": since}))
        return
    console.print(
        f"\n[yellow]No Claude Code sessions in the last {since}.[/yellow]\n"
        "[dim]Run [bold]npx tokenjam onboard[/bold] to go deeper; it wires "
        "up live capture so [bold]tj context[/bold] can show a wider "
        "window.[/dim]\n"
    )


def _sessions_phrase(n: int) -> str:
    """`"1 session"` / `"42 sessions"` — pluralised, never "1 sessions"."""
    return f"{n} session" if n == 1 else f"{n} sessions"


def _population_phrase(n: int) -> str:
    """The finding sentence's population clause: `"your last 42 sessions"`.

    Drops the count entirely at n == 1, because "your last 1 session" reads as
    a bug even though it is arithmetically right.
    """
    return "your last session" if n == 1 else f"your last {n} sessions"


def _avoidable_line(total: AvoidableTotal):
    """The ONE finding sentence: avoidable dollars over the sessions analyzed.

    Three states, three different sentences, and the caller never reaches this
    function in the fourth (`None`, "not measured") — an unknown figure prints
    nothing at all rather than an empty state, because "we have not computed it"
    and "we computed it and found nothing" are different claims.

    `total.sessions` is the population `total.usd` was summed over, carried on
    the same object precisely so the two cannot drift apart.
    """
    from rich.text import Text

    line = Text()
    if total.usd <= 0:
        # Known, and genuinely empty. Never a `$0.00` styled like a finding.
        line.append(
            f"No avoidable spend found in {_population_phrase(total.sessions)}.",
            style="muted",
        )
        return line
    line.append(format_cost(total.usd), style=f"bold {ACCENT}")
    line.append(f" in {_population_phrase(total.sessions)} was avoidable.")
    return line


def _render(avoidable: AvoidableTotal | None, *,
            fallback_sessions: int = 0) -> None:
    """The whole first-run screen.

    Deliberately three beats and nothing else: what tj read, one avoidable
    sentence, and the pointer to `npx tokenjam onboard`. No panels, no borders, no
    per-analyzer detail, no quota composition, no statusline preview. See the
    module docstring — the shape is a founder decision, not an accident of
    what was easy to render.

    ONE session count appears on this screen, and it is the population the
    figure was summed over. The scoping line and the finding sentence quote the
    same number by construction, because two counts on one screen invite a
    reader to pair the money with the wrong one.

    Every command named here is the `npx` form: this screen is only ever reached
    through the npm wrapper, so a bare `tj ...` would be an instruction the
    reader may have no binary for. See `ONBOARD_COMMAND`.

    Colour discipline: everything is `muted` except the dollar figure and the
    typeable command, which carry the single accent. Nothing here is red,
    green, yellow or cyan; there is no genuine state on this screen for a
    state colour to mean.
    """
    from rich.text import Text

    console.print()
    console.print(Text(
        "TokenJam reads your ~/.claude/projects/*.jsonl session logs.",
        style="muted",
    ))

    # Scope, stated before the figure. "most-recent" is the honest word whether
    # or not the first-run cap bit: this is a recent slice of a history that
    # keeps going, which is what the onboard pointer below is for. The
    # pre-ingest status line already names the cap itself.
    scoped = avoidable.sessions if avoidable is not None else max(fallback_sessions, 0)
    console.print(Text(
        f"Showing your most-recent {_sessions_phrase(scoped)}.", style="muted"))

    if avoidable is not None:
        console.print()
        console.print(_avoidable_line(avoidable))
        # The explanation attaches to a FIGURE, so it renders only when there is
        # one. On an unknown or measured-empty window there is nothing to
        # explain, and a sentence about what would have been removed would read
        # as a finding the run never made.
        if avoidable.usd > 0:
            console.print(Text(_shape_clause(avoidable.contributors), style="muted"))

    console.print()
    cta = Text()
    cta.append("Run ", style="muted")
    cta.append(ONBOARD_COMMAND, style=ACCENT)
    cta.append(" to set up TokenJam.", style="muted")
    console.print(cta)
    console.print(Text(
        "You get your full history, live capture as you work, and Lens: the "
        "local dashboard where you review and apply fixes.",
        style="muted",
    ))
    console.print(Text(_OTHER_SOURCES, style="muted"))
    console.print(Text("Runs on your machine. No signup.", style="muted"))
    console.print()


#: The onboard CTA. Unconditionally the zero-install form, and that is a
#: correctness constraint rather than a style choice.
#:
#: This screen is reachable from exactly ONE place: `cli/main.py`'s
#: no-subcommand branch, gated on `TJ_NPX_ZERO_INSTALL_REPORT`, which only the
#: npm wrapper (`npm-wrapper/bin/tj.js`) sets, and only on a bare `npx tokenjam`
#: with no passthrough args. An installed user running bare `tj` gets the home
#: screen instead and never lands here. So the reader arrived through `npx` and
#: `npx tokenjam onboard` is the door they can re-enter by.
#:
#: What was here before branched on `cmd_onboard._is_ephemeral_runner()` and
#: dropped the prefix to a bare `tj onboard` whenever the process was NOT
#: running from a throwaway uvx/pipx cache. That probe answers "was this
#: launched from a persistent install", which is a different question from "does
#: this user have `tj` on PATH": the wrapper's third runner is an
#: already-installed `tj`, so a real npx user could be printed an instruction
#: whose binary they may not have, and which is not how they invoked the tool
#: either way. `npx tokenjam onboard` is correct for every reader of this
#: screen; there is no case here that needs a second form.
ONBOARD_COMMAND = "npx tokenjam onboard"
