from __future__ import annotations

import json as json_mod
import os
import platform
import secrets
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

import click
from rich.markup import escape

from tokenjam.cli.backfill_progress import transient_status
from tokenjam.cli.banner import print_welcome_banner
from tokenjam.cli.onboard_detect import SdkMatch, detect_stack, install_hint
from tokenjam.cli.tj_status import TjCommand
from tokenjam.core.config import (
    align_project_secret_to_global,
    ensure_global_ingest_secret,
    resolve_config_path,
)
# Aliased: several flows below bind a LOCAL `global_config_path` variable, which
# would shadow the imported helper inside those functions.
from tokenjam.core.config import global_config_path as _global_config_path
from tokenjam.core.ingest_adapters.codex import ingest_codex
from tokenjam.otel.semconv import SUBSCRIPTION_PLAN_TIERS
from tokenjam.utils.formatting import console, display_path

# --- Claude Code backfill scope (#443) ---------------------------------------
# `tj onboard --claude-code` used to backfill the ENTIRE on-disk history with
# no cap and no progress output — on a large `~/.claude/projects` (thousands of
# JSONL files) that's 10-30+ minutes of silent, 100%-CPU work right after "tj
# config written to...", indistinguishable from a hang at the exact moment a
# new user's trust is most fragile. Mirrors `tj quickstart`'s DEFAULT_MAX_SESSIONS
# cap (#13) but scopes by *time* (`since`) instead of a session count, since
# onboard's backfill is meant to be completable in full later via
# `tj backfill claude-code` — the prompt below always says so.
DEFAULT_BACKFILL_DAYS = 30

# --- output-trim (`tj hook cap-output`) legacy hook cleanup ------------------
# The output-trim hook itself was removed (measured negative: +5.6% whole-
# session cost on Claude Code, see CLAUDE.md). This matcher/unwire pair stays
# ONLY so onboard/uninstall can strip an already-installed entry from a prior
# release's ~/.claude/settings.json — never re-installed, never re-offered.

_CAP_OUTPUT_MARKER = "hook cap-output"  # tj-managed marker (substring of command)


def _is_tj_cap_output_entry(entry: object) -> bool:
    if not isinstance(entry, dict):
        return False
    for h in entry.get("hooks", []) or []:
        if isinstance(h, dict) and _CAP_OUTPUT_MARKER in str(h.get("command", "")):
            return True
    return False


def _unwire_claude_output_cap_hook(settings: dict) -> bool:
    """Remove any tj-managed cap-output PostToolUse entry. Returns True if one
    was removed (used by `tj onboard` and `tj uninstall`)."""
    hooks = settings.get("hooks")
    if not isinstance(hooks, dict):
        return False
    post = hooks.get("PostToolUse")
    if not isinstance(post, list):
        return False
    kept = [e for e in post if not _is_tj_cap_output_entry(e)]
    if len(kept) == len(post):
        return False
    if kept:
        hooks["PostToolUse"] = kept
    else:
        hooks.pop("PostToolUse", None)
        if not hooks:
            settings.pop("hooks", None)
    return True


# --- resume-brief (`tj resume-brief --last`) SessionStart hook wiring ---------
# Installed into ~/.claude/settings.json out-of-band (zero in-loop token cost).
# On a session that RESUMES or comes back post-COMPACTION, Claude Code fires
# SessionStart with source `resume` / `compact`; this hook runs
# `tj resume-brief --last` and its stdout (the brief) is injected as
# additionalContext, so the continuing session is handed its prior method
# instead of re-investigating. Idempotent + non-destructive: a tj-managed entry
# is detected by the "resume-brief" substring, so re-onboard updates OUR entry
# in place and NEVER clobbers a user's own SessionStart hooks.

_RESUME_BRIEF_MATCHER = "resume|compact"
_RESUME_BRIEF_MARKER = "resume-brief"  # tj-managed marker (substring of command)


def _tj_resume_brief_command() -> str:
    """Absolute `tj resume-brief --from-hook` command, falling back to bare `tj`.

    ``--from-hook`` reads the SessionStart hook's stdin JSON so the brief is
    scoped to the session the hook fired for. The prior ``--last`` wiring
    guessed by global mtime across ALL projects and could cross-leak a
    concurrent session's brief — the exact fan-out scenario this feature is
    for. Re-onboard rewires stale ``--last`` entries in place via the
    ``resume-brief`` substring marker.
    """
    exe = shutil.which("tj")
    return f"{exe} resume-brief --from-hook" if exe else "tj resume-brief --from-hook"


def _is_tj_resume_brief_entry(entry: object) -> bool:
    if not isinstance(entry, dict):
        return False
    for h in entry.get("hooks", []) or []:
        if isinstance(h, dict) and _RESUME_BRIEF_MARKER in str(h.get("command", "")):
            return True
    return False


def _wire_claude_resume_brief_hook(settings: dict) -> str:
    """Install/refresh the SessionStart resume-brief hook in a settings dict.

    Mutates ``settings`` in place; returns one of ``written`` / ``updated`` /
    ``kept`` / ``skipped`` (foreign structure left untouched).
    """
    desired = {
        "matcher": _RESUME_BRIEF_MATCHER,
        "hooks": [{"type": "command", "command": _tj_resume_brief_command()}],
    }
    hooks = settings.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        return "skipped"
    start = hooks.get("SessionStart")
    if start is None:
        hooks["SessionStart"] = [desired]
        return "written"
    if not isinstance(start, list):
        return "skipped"
    for i, entry in enumerate(start):
        if _is_tj_resume_brief_entry(entry):
            if entry == desired:
                return "kept"
            start[i] = desired
            return "updated"
    start.append(desired)          # preserve any foreign SessionStart hooks
    return "written"


def _unwire_claude_resume_brief_hook(settings: dict) -> bool:
    """Remove any tj-managed resume-brief SessionStart entry. Returns True if one
    was removed (used by `tj uninstall`)."""
    hooks = settings.get("hooks")
    if not isinstance(hooks, dict):
        return False
    start = hooks.get("SessionStart")
    if not isinstance(start, list):
        return False
    kept = [e for e in start if not _is_tj_resume_brief_entry(e)]
    if len(kept) == len(start):
        return False
    if kept:
        hooks["SessionStart"] = kept
    else:
        hooks.pop("SessionStart", None)
        if not hooks:
            settings.pop("hooks", None)
    return True


# --- Ephemeral-runner guard (#120) -------------------------------------------
# `npx tokenjam onboard` delegates to `uvx --from tokenjam tj onboard` (or
# `pipx run --spec tokenjam tj onboard`) — both resolve `sys.executable` into a
# throwaway, cache-managed venv that is not kept on PATH once this process
# exits. Onboard wires a background daemon and a Claude Code statusline that
# both invoke `tj` afterward; under an ephemeral runner those references go
# stale the moment the session ends. This guard detects that situation and
# offers (or performs) a persistent install before any wiring happens.

_LOCAL_BIN_DIR = Path.home() / ".local" / "bin"


def _is_ephemeral_path(path: str) -> bool:
    """True when `path` sits inside a throwaway uvx/pipx-run cache rather than
    a stable, named install location.

    Persistent installs resolve into a stable, named location:
      - `uv tool install`  → .../uv/tools/<pkg>/...
      - `pipx install`     → .../pipx/venvs/<pkg>/...  (see `_installed_via_pipx`
        in cmd_uninstall.py — same signature, different concern)
    Ephemeral `uvx` / `pipx run` executions resolve into a cache directory
    instead (e.g. `~/.cache/uv/archive-v0/...` for uvx). That cache is
    routinely swept by `uv cache prune` / `uv cache clean` — a path pointing
    into it is a landmine for anything (like a launchd/systemd unit) that
    expects to find a binary there indefinitely (#155).
    """
    p = path.replace("\\", "/")
    if "/uv/tools/" in p or "/pipx/venvs/" in p:
        return False
    return "/uv/" in p or "/pipx/" in p


def _is_ephemeral_runner() -> bool:
    """True when this process is running from a throwaway uvx/pipx-run venv
    rather than a persistent install."""
    return _is_ephemeral_path(sys.executable)


def _install_tokenjam_persistently() -> str | None:
    """Best-effort persistent install via `uv tool install` (preferred) or
    `pipx install`. Returns the absolute path to the newly installed `tj`, or
    None if neither runner is available or the install failed.
    """
    candidates: list[list[str]] = []
    if shutil.which("uv"):
        candidates.append(["uv", "tool", "install", "tokenjam"])
    if shutil.which("pipx"):
        candidates.append(["pipx", "install", "tokenjam"])
    for cmd in candidates:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except Exception:
            continue
        combined = f"{result.stdout}\n{result.stderr}".lower()
        if result.returncode != 0 and "already" not in combined:
            continue
        # The default shim dir covers a stock install; `UV_TOOL_BIN_DIR` /
        # `PIPX_BIN_DIR` (or a customized PATH) can place `tj` elsewhere, and
        # missing that would silently fall through to the ephemeral path this
        # guard exists to avoid — so also resolve `tj` via PATH.
        tj_path = _LOCAL_BIN_DIR / "tj"
        if tj_path.exists():
            return str(tj_path)
        on_path = shutil.which("tj")
        if on_path:
            return on_path
    return None


def _maybe_guard_ephemeral_runner(ctx: click.Context) -> None:
    """If running under an ephemeral uvx/pipx-run env, offer a persistent
    install and re-exec onboard through it (#120). No-op for the common case
    (already-installed `tj`, plain pip/venv) — zero behavior change there.
    """
    if not _is_ephemeral_runner():
        return

    console.print(
        "\n[warn]Heads up:[/warn] you're running via a temporary "
        "uvx/pipx-run environment. [bold]tj onboard[/bold] wires a "
        "background daemon and a Claude Code statusline that both need a "
        "persistent [bold]tj[/bold] on PATH — those would go stale the "
        "moment this session ends."
    )

    if not _is_interactive():
        console.print(
            "[dim]Non-interactive — continuing without a persistent "
            "install. Run [bold]pipx install tokenjam && tj onboard[/bold] "
            "(or [bold]uv tool install tokenjam[/bold]) for a setup that "
            "survives.[/dim]\n"
        )
        return

    if not click.confirm(
        "Install tokenjam persistently now (uv tool install / pipx "
        "install), then continue onboarding?", default=True,
    ):
        console.print(
            "[dim]Continuing without a persistent install — re-run "
            "[bold]pipx install tokenjam && tj onboard[/bold] "
            "later.[/dim]\n"
        )
        return

    console.print("[dim]Installing tokenjam…[/dim]")
    tj_path = _install_tokenjam_persistently()
    if tj_path is None:
        console.print(
            "[error]Persistent install failed.[/error] Continuing this "
            "session ephemerally — run [bold]pipx install tokenjam && "
            "tj onboard[/bold] yourself afterward.\n"
        )
        return

    console.print(f"[ok]✓[/ok] Installed — re-running onboard via {tj_path}\n")
    result = subprocess.run([tj_path, *sys.argv[1:]])
    ctx.exit(result.returncode)


def _print_generic_instrument_snippet() -> None:
    """The one-size-fits-all Anthropic snippet — fallback when detection finds nothing."""
    console.print("[dim]     from tokenjam.sdk import watch[/dim]")
    console.print("[dim]     from tokenjam.sdk.integrations.anthropic import patch_anthropic[/dim]")
    console.print()
    console.print("[dim]     patch_anthropic()[/dim]")
    console.print()
    console.print('[dim]     @watch(agent_id="my-agent")[/dim]')
    console.print("[dim]     def run(task):[/dim]")
    console.print("[dim]         ...[/dim]")


def _print_matched_instrument_snippet(match: SdkMatch) -> None:
    """One detected SDK/framework's tailored `patch_*()` + `@watch()` snippet."""
    console.print(f"[dim]     # {match.label}[/dim]")
    console.print("[dim]     from tokenjam.sdk import watch[/dim]")
    console.print(f"[dim]     {match.import_line}[/dim]")
    console.print()
    console.print(f"[dim]     {match.patch_call}[/dim]")
    console.print()
    console.print('[dim]     @watch(agent_id="my-agent")[/dim]')
    console.print("[dim]     def run(task):[/dim]")
    console.print("[dim]         ...[/dim]")
    hint = install_hint(match)
    if hint:
        console.print()
        # escape(): the extras bracket (e.g. "tokenjam[langchain]") would
        # otherwise be swallowed as Rich markup — same class of bug as #157.
        console.print(f"[dim]     {escape(hint)}[/dim]")


def _print_capture_disclosure(prompts_captured: bool, tool_inputs_captured: bool = False) -> None:
    """Disclose prompt-text / tool-input capture state at the end of onboarding.

    `capture.prompts` and `capture.tool_inputs` both default on (see
    `CaptureConfig` in `core/config.py`) so `tj optimize trim` /
    `cache-recommend` / `reuse`'s sharper mode, and the `script` / `verbosity`
    analyzers' argument-shape clustering, work without extra setup. Storage
    stays local — the user's own telemetry DB — but that doesn't make it
    exempt from disclosure: this is new data at rest the user didn't have
    before, so every onboarding path says so explicitly rather than leaving
    it to a comment in the config file.
    """
    if not (prompts_captured or tool_inputs_captured):
        return
    # One clear line that leads with local + on-by-default (#643), replacing the
    # two dense per-toggle paragraphs. The default (both on) uses the exact
    # approved wording; the subject phrase narrows honestly when only one toggle
    # is on so the line never claims capture that isn't happening.
    if prompts_captured and tool_inputs_captured:
        subject = "Your prompts and tool inputs are captured"
    elif prompts_captured:
        subject = "Your prompts are captured"
    else:
        subject = "Your tool inputs are captured"
    console.print(
        f"\U0001f512 {subject} and stored [bold]only on this machine[/bold] "
        "(your local telemetry DB — nothing is ever uploaded). That's what "
        "powers the trim, cache, reuse and script analyzers. On by default — "
        "turn it off anytime with [accent]capture.prompts[/accent] / "
        "[accent]capture.tool_inputs = false[/accent] in your config."
    )


def _print_instrument_agent_snippet() -> None:
    """Print the bare-onboard "instrument your agent" snippet (issue #85).

    Detects the current project's declared LLM provider SDKs / agent
    frameworks (via `onboard_detect.detect_stack`) and prints a tailored
    `patch_*()` call + install hint per match, instead of always assuming
    Anthropic. Falls back to the generic Anthropic snippet when nothing is
    detected (unchanged from pre-#85 behavior).
    """
    # Stack detection is a nice-to-have outro, never load-bearing: if anything
    # in manifest reading goes wrong (e.g. a non-UTF-8 manifest surfacing an
    # unexpected error), degrade to the generic snippet rather than crashing
    # a default-run command.
    try:
        matches = detect_stack(".")
    except Exception:
        matches = []
    if not matches:
        _print_generic_instrument_snippet()
        return
    for i, match in enumerate(matches):
        if i > 0:
            console.print()
        _print_matched_instrument_snippet(match)
    if any(m.key == "litellm" for m in matches) and len(matches) > 1:
        console.print()
        console.print(
            "[dim]     # Note: patch_litellm() alone covers every provider "
            "it routes — the individual[/dim]"
        )
        console.print(
            "[dim]     # provider patches above are only needed for calls "
            "made outside LiteLLM.[/dim]"
        )


#: No class-level `status_message`: onboard is heavily interactive (usage-path
#: choice, plan-tier prompts, an optional `--verify` confirm) for most of its
#: body, and a live spinner colliding with a blocking stdin read corrupts
#: both. `transient_status` is called directly around its two genuinely
#: silent stretches instead — see `_try_backfill_codex` and
#: `_run_onboard_verification`.
@click.command("onboard", cls=TjCommand)
@click.option("--claude-code", "claude_code", is_flag=True, default=False,
              help="Configure Claude Code telemetry to flow into tj")
@click.option("--codex", "codex", is_flag=True, default=False,
              help="Configure Codex CLI telemetry to flow into tj")
@click.option("--add-project", "add_project", is_flag=True, default=False,
              help="Register this repo under a project namespace in the "
                   "existing global config, without re-running the full "
                   "wizard (no plan/budget prompt, no backfill, no daemon "
                   "restart). Requires `tj onboard` to have already run once "
                   "(anywhere). The project name is always derived from the "
                   "repo/folder name.")
@click.option("--budget", type=float, default=None,
              help="Daily budget in USD per agent (0 = no limit)")
@click.option("--install-daemon", "install_daemon", is_flag=True, default=False,
              help="(no-op: daemon is installed by default; use --no-daemon to skip)")
@click.option("--no-daemon", is_flag=True, default=False,
              help="Skip background daemon installation")
@click.option("--force", is_flag=True, help="Overwrite existing config")
@click.option("--reconfigure", is_flag=True, default=False,
              help="Re-prompt for plan tier and budget against an existing config. "
                   "Equivalent to onboard but skips agent-runtime re-detection.")
@click.option("--plan",
              type=click.Choice(["api", "pro", "max_5x", "max_20x",
                                  "plus", "team", "enterprise"]),
              default=None,
              help="Plan tier for the provider being onboarded. Skips the "
                   "interactive plan prompt when set. Choices: api / pro / "
                   "max_5x / max_20x (Anthropic), plus / team / enterprise (OpenAI).")
@click.option("--analysis-span", "analysis_span",
              type=click.Choice(["30d", "90d", "all"]),
              default=None,
              help="How far back tj should analyze. Storage retention is "
                   "derived from this — 'all' disables deletion entirely — so "
                   "history the analyzers use can never be deleted underneath "
                   "them. Skips the interactive span prompt when set.")
@click.option("--project", "removed_project_flag", default=None, hidden=True,
              help="Removed — the project name is always derived from the "
                   "repo/folder name now.")
@click.option("--backfill-days", "backfill_days", type=int, default=None,
              help=f"Backfill only the last N days of Claude Code history "
                   f"(default {DEFAULT_BACKFILL_DAYS} when neither this nor "
                   f"--backfill-all is set). Skips the interactive scope prompt.")
@click.option("--backfill-all", "backfill_all", is_flag=True, default=False,
              help="Backfill the entire Claude Code history instead of the "
                   "default recent window. Skips the interactive scope prompt.")
@click.option("--verbose", "-V", "verbose", is_flag=True, default=False,
              help="Show the full connection-details block (settings paths, "
                   "Agent ID, OTLP endpoints, ingest secret, per-terminal "
                   "tiles) on the success screen. Omitted by default to keep "
                   "the completion screen lean; the same details are always "
                   "available via `tj doctor`.")
@click.option("--verify", is_flag=True, default=False,
              help="After setup, poll for the first span from the newly "
                   "configured source and report whether telemetry is flowing "
                   "(distinguishes 'wired and receiving' from 'configured but "
                   "silent'). Runs non-interactively; skips the prompt.")
@click.option("--verify-only", "verify_only", is_flag=True, default=False,
              help="Skip setup and only poll for the first live span against an "
                   "existing config — the lightweight post-restart re-check. "
                   "Does not rewrite config or replay the summary. Pair with "
                   "--claude-code / --codex to select that persona's config.")
@click.pass_context
def cmd_onboard(ctx: click.Context, claude_code: bool, codex: bool, budget: float | None,
                install_daemon: bool, no_daemon: bool, force: bool,
                reconfigure: bool, plan: str | None, analysis_span: str | None,
                removed_project_flag: str | None,
                backfill_days: int | None, backfill_all: bool,
                verify: bool, verify_only: bool, add_project: bool,
                verbose: bool) -> None:
    """Set up tj (interactive)."""
    # --project was removed: the project name is now ALWAYS derived from the
    # repo/folder name (see _derive_project_name) — the flag and the prompt
    # it fed were ceremony that only ever created divergence from what nearly
    # everyone already ended up with. Fail loudly and specifically rather
    # than letting a script that still passes it hit Click's generic
    # "no such option" (the flag stays declared, hidden, purely so we can
    # catch this and say why).
    if removed_project_flag is not None:
        raise click.UsageError(
            "--project has been removed. The project name is now taken "
            "from the repo/folder name automatically."
        )
    # --add-project is the lightweight "register another repo" path: a fresh
    # onboard run per repo re-prompts plan/budget/backfill scope and re-scans
    # the entire Claude Code history just to set one config key
    # (agents.<id>.project). Short-circuit before the banner, the
    # ephemeral-runner guard, and every other prompt — this path writes
    # nothing but the namespace mapping.
    if add_project:
        _onboard_add_project(ctx)
        return
    # --verify-only is the documented post-restart re-check: config already
    # exists, the user just restarted the agent, and re-running the whole wizard
    # (config rewrite + full summary + restart banner) only to poll is wasteful
    # noise (#102). Skip straight to the poll, before the banner and any setup.
    if verify_only:
        _run_verify_only(ctx, claude_code=claude_code, codex=codex)
        return
    if backfill_days is not None and backfill_all:
        raise click.UsageError("Use either --backfill-days or --backfill-all, not both.")
    if backfill_days is not None and backfill_days <= 0:
        raise click.UsageError("--backfill-days must be > 0.")
    # Ephemeral-runner guard (#120): onboard below wires a daemon + Claude Code
    # statusline that both invoke `tj` after this process exits. Under
    # `uvx --from tokenjam tj onboard` / `pipx run --spec tokenjam tj onboard`
    # (what the `npx tokenjam onboard` wrapper delegates to) there is no
    # persistent `tj` left on PATH once this run ends, so those references
    # would go stale. Offer (or perform) a persistent install and re-exec
    # onboard through it before any wiring happens.
    _maybe_guard_ephemeral_runner(ctx)
    # Branded welcome moment (#240) — shown once at the top of every onboard
    # flow (plain / --claude-code / --codex) before any prompt or config check.
    print_welcome_banner()
    if claude_code:
        _onboard_claude_code(ctx, budget, no_daemon, force, reconfigure, plan,
                             verify=verify,
                             backfill_days=backfill_days, backfill_all=backfill_all,
                             analysis_span=analysis_span, verbose=verbose)
        return
    if codex:
        _onboard_codex(ctx, budget, no_daemon, force, reconfigure, plan,
                       verify=verify, analysis_span=analysis_span)
        return

    # Path-branched first run (#448): the bare `tj onboard` no longer assumes an
    # SDK/API user. It opens with "How do you use AI agents?" and routes to the
    # matching flow, so a Claude Code user (the most common case) gets a
    # backfill + statusline rather than an SDK snippet and a live-span verify
    # that can never succeed. `--claude-code` / `--codex` above stay as shortcuts
    # that skip the question. `--reconfigure` is still per-provider only, so it
    # keeps its early error below. We only branch interactively — a non-tty bare
    # invocation (scripts, CI) falls through to the historical generic SDK path
    # so existing automation is byte-for-byte unchanged.
    if not reconfigure and _is_interactive():
        choice = _prompt_usage_path()
        if choice == "claude-code":
            _onboard_claude_code(ctx, budget, no_daemon, force, reconfigure, plan,
                                 verify=verify,
                                 backfill_days=backfill_days,
                                 backfill_all=backfill_all,
                                 analysis_span=analysis_span, verbose=verbose)
            return
        if choice == "codex":
            _onboard_codex(ctx, budget, no_daemon, force, reconfigure, plan,
                           verify=verify, analysis_span=analysis_span)
            return
        if choice == "combination":
            _onboard_combination(ctx, budget, no_daemon, force, plan,
                                  verify=verify,
                                  backfill_days=backfill_days,
                                  backfill_all=backfill_all,
                                  analysis_span=analysis_span)
            return
        # choice == "sdk" → fall through to the generic SDK/API path below.

    # Honor TJ_CONFIG for the EXISTING-config check only (not the write
    # target below, which always writes a new `.tj/config.toml` by design —
    # see the comment at that assignment). This is deliberately asymmetric:
    # a bare search-path find_config_file() here would miss a TJ_CONFIG-
    # pointed config and go on to double-write `.tj/config.toml` alongside
    # it, leaving the user with two configs and the wrong one still active
    # (TJ_CONFIG always wins at read time). Honoring TJ_CONFIG here instead
    # makes onboard correctly report "Config already exists" and no-op.
    existing = resolve_config_path((ctx.obj or {}).get("config_path_override"))
    if existing and not force:
        # --reconfigure is only meaningful with --claude-code / --codex.
        # The bare onboard path writes a generic config and doesn't manage
        # plan tier — that's per-provider, written into [budget.<provider>]
        # sections by the integration-specific onboarders. Silently
        # no-op'ing here would frustrate users who pass --reconfigure --plan
        # expecting their plan field to update (#68 §1).
        if reconfigure:
            console.print(
                "[error]--reconfigure has no effect without --claude-code or "
                "--codex.[/error]"
            )
            console.print(
                "\nThe bare onboard path writes a generic config and doesn't "
                "manage plan tier — that's per-provider, set by the "
                "integration-specific flows.\n"
            )
            console.print(
                "Try [bold]tj onboard --claude-code --reconfigure[/bold] or "
                "[bold]tj onboard --codex --reconfigure[/bold] instead."
            )
            ctx.exit(1)
            return
        console.print(f"[bold]Config already exists:[/bold] {existing}")
        console.print("Use [bold]--force[/bold] to overwrite.")
        return

    console.print()
    console.print("[bold]Setting up TokenJam...[/bold]")
    console.print()

    # Plan tier (#4): the plain path now honors `--plan` and prompts for it
    # interactively, instead of silently ignoring `--plan` and never writing a
    # `[budget.<provider>] plan`. This is a Claude-first tool, so the interactive
    # prompt offers the Anthropic tiers; an OpenAI-only `--plan` (plus/team/
    # enterprise) is routed to its provider section. (The `--claude-code` /
    # `--codex` flows still own the global integration configs.)
    #
    # Plan-first (#240): "How do you pay?" is the more important and more natural
    # opening question, so it comes before the daily-budget prompt below.
    plan_tier = plan
    if plan_tier is None and sys.stdin.isatty():
        plan_tier = _prompt_plan("Claude", _ANTHROPIC_PLAN_CHOICES)

    if budget is None:
        budget = click.prompt(
            "Daily budget in USD per agent (0 = no limit, default 0)",
            type=float, default=0.0, show_default=False,
        )

    plan_provider = (
        "openai" if plan_tier in ("plus", "team", "enterprise") else "anthropic"
    )
    plan_changed = plan_tier is not None
    plan_section = (
        f'\n[budget.{plan_provider}]\nplan = "{plan_tier}"\n' if plan_tier else ""
    )

    # One machine-wide ingest secret, always the global config's. This path
    # writes a project-local `.tj/config.toml`, and minting a secret into it
    # produced a config the daemon (always launched against the global config)
    # rejects: the SDK signs spans with the project secret, the daemon
    # authenticates against the global one, and every push 401s silently. The
    # project file now MIRRORS the global secret rather than owning one.
    ingest_secret, secret_minted = ensure_global_ingest_secret()

    want_daemon = not no_daemon

    config_path = Path(".tj/config.toml")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    prior_plan: str | None = None
    if config_path.exists():
        try:
            from tokenjam.core.config import load_config as _load_prior_config
            prior_cfg = _load_prior_config(str(config_path.resolve()))
            prior_pb = prior_cfg.budgets.get(plan_provider)
            prior_plan = prior_pb.plan if prior_pb else None
        except Exception:
            prior_plan = None

    budget_line = ""
    if budget and budget > 0:
        budget_line = f"daily_usd = {budget}"

    # The one span question, asked here because this path writes its config as
    # TOML text rather than through `write_config` — `_apply_analysis_span`
    # cannot reach it. Same choice, same default, same silence when non-tty.
    from tokenjam.core.analysis_span import DEFAULT_ANALYSIS_SPAN, parse_analysis_span
    if analysis_span is None:
        analysis_span = (
            _prompt_analysis_span() if _is_interactive() else DEFAULT_ANALYSIS_SPAN
        )
    parse_analysis_span(analysis_span)  # reject an unwritable span before it is written

    config_text = f"""\
# TokenJam configuration
# Docs: https://github.com/Metabuilder-Labs/tokenjam#configuration

[defaults.budget]
{budget_line}
{plan_section}
[security]
ingest_secret = "{ingest_secret}"

[capture]
# prompts is on by default: `tj optimize trim` / `cache-recommend` and the
# sharper mode of `reuse` all read captured prompt text, and stay dark
# without it. Storage is local-only (your own telemetry DB below); nothing
# is sent anywhere. Set this to false to turn it off.
prompts = true
completions = false
# tool_inputs is on by default too: it captures the tool call arguments your
# instrumentation records (e.g. via `record_tool_call`, or the declared tool
# schema on an Anthropic/OpenAI/etc. request) so the `script` and `verbosity`
# analyzers can cluster on argument shape instead of tool names alone.
# completions/tool_outputs stay off by default since those would persist
# actual completion/tool-output text; turn them on for deeper analysis.
tool_inputs = true
tool_outputs = false

[storage]
path = "~/.tj/telemetry.duckdb"
# How far back tj analyzes: "30d", "90d", or "all" (keep everything, never
# delete). Storage retention is DERIVED from this, so history the analyzers use
# can never be deleted underneath them. Set retention_days as well only if you
# want to keep MORE than you analyze; a value shorter than the span is raised
# to it rather than honoured.
analysis_span = "{analysis_span}"

# Per-agent overrides (optional):
# [agents.my-agent]
# description = "My email agent"
#   [agents.my-agent.budget]
#   daily_usd = 5.00
#   session_usd = 1.00
#   [[agents.my-agent.sensitive_actions]]
#   name = "send_email"
#   severity = "critical"
"""
    config_path.write_text(config_text)

    from tokenjam.core.config import load_config as _load_plain_config
    plain_config = _load_plain_config(str(config_path.resolve()))
    stopped_for_db = _stop_serve_for_db_write()
    reconcile_plan = (
        prior_plan is not None
        and plan_tier is not None
        and plan_tier != prior_plan
    )
    apply_msg = _try_apply_declared_plans(plain_config, reconcile=reconcile_plan)
    if apply_msg:
        console.print(f"[ok]\u2713[/ok] {apply_msg}")

    daemon_msg = None
    if want_daemon:
        daemon_msg = _finish_onboard_serve(
            str(config_path.resolve()),
            want_daemon=True,
            plan_changed=plan_changed,
            stopped_for_db=stopped_for_db,
            secret_rotated=False,
            no_daemon=no_daemon,
            force=False,
        )

    # Output
    console.print()
    console.print("[ok]\u2713[/ok] Config written to [accent].tj/config.toml[/accent]")
    if secret_minted:
        console.print(f"[ok]\u2713[/ok] Ingest secret generated: "
                      f"[dim]{ingest_secret[:8]}...[/dim]")
    else:
        # Adopted, not generated: saying "generated" about a secret that
        # already existed reads as a rotation, and would send the user looking
        # for integrations to re-key that in fact still work.
        console.print(
            f"[ok]\u2713[/ok] Ingest secret: [dim]{ingest_secret[:8]}... "
            f"(shared with {display_path(_global_config_path())})[/dim]",
            soft_wrap=True,
        )
    if budget and budget > 0:
        console.print(f"[ok]\u2713[/ok] Default daily budget: "
                      f"[bold]${budget:.2f}[/bold] per agent")
    if plan_tier:
        # Escape the TOML section header \u2014 Rich treats `[budget.<provider>]` as a
        # markup tag and would strip it, leaving "(written to )". See issue #157.
        plan_section = escape(f"[budget.{plan_provider}]")
        console.print(f"[ok]\u2713[/ok] Plan tier: "
                      f"[bold]{plan_tier}[/bold] (written to {plan_section})")
    if daemon_msg:
        console.print(f"[ok]\u2713[/ok] {daemon_msg}")
    _print_capture_disclosure(plain_config.capture.prompts, plain_config.capture.tool_inputs)

    console.print()
    console.print("[bold]Next steps:[/bold]")
    console.print()
    console.print("  1. Instrument your agent:")
    console.print()
    _print_instrument_agent_snippet()
    console.print()
    console.print("  2. Run your agent \u2014 spans are recorded automatically")
    console.print()
    console.print("  3. View telemetry:")
    console.print("[dim]     tj status          [/dim]# agent overview")
    console.print("[dim]     tj traces          [/dim]# span history")
    console.print("[dim]     tj serve           [/dim]# web UI at http://127.0.0.1:7391/")
    console.print()
    console.print("  4. Prove a cheaper model still holds:")
    console.print("[dim]     pip install tokenjam-bench[/dim]  # then: tjb")
    console.print()

    if not want_daemon:
        _warn_manual_serve_restart(stopped_for_db=stopped_for_db, no_daemon=True)
        console.print("  Run [bold]tj serve[/bold] to start the web UI "
                      "and enable real-time alerts.")
        console.print()

    console.print(
        "  To configure per-agent budgets, sensitive actions, or drift detection:"
    )
    console.print(
        "  Edit [bold].tj/config.toml[/bold] \u2014 see "
        "[dim]https://github.com/Metabuilder-Labs/tokenjam#configuration[/dim]"
    )
    console.print()

    # First-signal verification (#80): the SDK is fail-open, so a typo'd
    # agent_id / missing patch / dead daemon leaves the user silent. Reload the
    # just-written config so the poller reads the same DB/API the daemon uses.
    try:
        from tokenjam.core.config import load_config as _load_written_config
        written_config = _load_written_config(str(config_path.resolve()))
    except Exception:
        written_config = None
    if written_config is not None:
        _maybe_verify_onboarding(written_config, persona="sdk", verify=verify)

    # Shared closing banner (#448): the SDK path has no backfill, so this is the
    # branded home screen + next-best-actions with no session count.
    _print_setup_complete_home()


def _maybe_verify_onboarding(config: object, *, persona: str, verify: bool) -> None:
    """Run first-signal verification if ``--verify`` was passed, or offer it
    interactively. No-op when neither applies (non-interactive without the flag).

    ``persona`` is one of ``"sdk"``, ``"claude-code"``, ``"codex"`` and drives
    the instruction shown and the not-confirmed cause.

    Restart-dependent personas (claude_code / codex) are never OFFERED the
    interactive poll here: their first span can't arrive until the agent
    runtime restarts, which hasn't happened yet at this point in onboarding —
    saying yes would poll for 60s and always time out. Codex gets a
    verify-after-restarting pointer instead. Claude Code does NOT get one
    here; it's already step 3 of the consolidated restart panel
    (``_print_claude_code_restart_panel``), printed earlier in the same
    completion screen; repeating it here near Connection details would just
    duplicate it. An explicit ``--verify`` still polls (the user asked for it,
    and the poll copy tells them to restart now).
    """
    if not verify:
        if persona == "claude-code":
            return
        if persona == "codex":
            console.print(
                "[dim]Verify after restarting:[/dim]  "
                "tj onboard --codex --verify-only"
            )
            return
        if not sys.stdin.isatty():
            return
        if not click.confirm(
            "\nVerify tj is receiving telemetry now?", default=False
        ):
            return
    _run_onboard_verification(config, persona)


def _run_onboard_verification(
    config: object, persona: str, *, timeout_s: float = 60.0
) -> None:
    """Open a read-only path to spans and poll for the first one after now,
    reporting confirmed / not-confirmed with the per-persona likely cause."""
    from tokenjam.core.onboard_verify import (
        _ReadBackend,
        not_confirmed_cause,
        open_read_backend,
        poll_for_first_span,
    )
    from tokenjam.utils.time_parse import utcnow

    backend, _mode, error = open_read_backend(config)
    if backend is None:
        console.print(f"\n[warn]Can't verify yet[/warn] \u2014 {error}.")
        console.print(
            "Start [bold]tj serve[/bold], then run [bold]tj doctor[/bold] to check."
        )
        return

    try:
        since = utcnow()
        if persona == "sdk":
            console.print(
                "\n[bold]Verifying\u2026[/bold] trigger one span now \u2014 run your agent, "
                "or in another terminal run [bold]tj ping[/bold]."
            )
        else:
            console.print(
                "\n[bold]Verifying\u2026[/bold] waiting for the first telemetry. If you "
                "haven't yet, [bold]restart[/bold] the agent runtime now."
            )
        with transient_status(f"Polling for up to {int(timeout_s)}s\u2026"):
            result = poll_for_first_span(
                cast(_ReadBackend, backend), since, timeout_s=timeout_s
            )
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    if result.confirmed:
        console.print(
            f"[ok]\u2713 Receiving telemetry![/ok] First span arrived after "
            f"{result.elapsed_s:.0f}s; you're wired up."
        )
    elif result.error:
        console.print(f"[warn]Couldn't verify[/warn] \u2014 {result.error}.")
    else:
        console.print(
            f"[warn]\u26a0 No telemetry yet[/warn] after {int(timeout_s)}s. "
            + not_confirmed_cause(persona)
        )


def _run_verify_only(ctx: click.Context, *, claude_code: bool, codex: bool) -> None:
    """Poll an already-configured install for its first live span, skipping setup.

    The persona (and which config to read) follows the same flag the user
    onboarded with: ``--claude-code`` / ``--codex`` read the global config;
    bare reads the nearest project/SDK config via ``resolve_config_path``
    (honoring ``TJ_CONFIG`` like every other read path). Errors out cleanly
    when no config exists yet — that's an "run `tj onboard` first" situation,
    not a verification failure.
    """
    from tokenjam.core.config import load_config

    if claude_code or codex:
        global_path = Path.home() / ".config" / "tj" / "config.toml"
        persona = "claude-code" if claude_code else "codex"
        config_path: Path | None = global_path if global_path.exists() else None
    else:
        found = resolve_config_path((ctx.obj or {}).get("config_path_override"))
        config_path = Path(found) if found else None
        persona = "sdk"

    if config_path is None:
        console.print(
            "[error]No tj config found.[/error] Run [bold]tj onboard[/bold] "
            "(optionally with --claude-code / --codex) first, then "
            "[bold]tj onboard --verify-only[/bold]."
        )
        ctx.exit(1)
        return

    try:
        config = load_config(str(config_path.resolve()))
    except Exception as exc:  # surface a clean message, no traceback
        console.print(f"[error]Could not load config[/error] at {display_path(config_path)}: {exc}")
        ctx.exit(1)
        return

    _run_onboard_verification(config, persona)


_ANTHROPIC_PLAN_CHOICES = [
    ("api",      "API (per-token billing through console.anthropic.com)"),
    ("pro",      "Pro plan ($20/mo subscription)"),
    ("max_5x",   "Max 5x plan ($100/mo subscription)"),
    ("max_20x",  "Max 20x plan ($200/mo subscription)"),
]

_OPENAI_PLAN_CHOICES = [
    ("api",        "API (per-token billing through platform.openai.com)"),
    ("plus",       "ChatGPT Plus ($20/mo subscription)"),
    ("team",       "ChatGPT Team ($25–30/seat subscription)"),
    ("enterprise", "ChatGPT Enterprise"),
]


def _prompt_plan(provider_label: str, choices: list[tuple[str, str]],
                 current: str | None = None) -> str:
    """
    Render a numbered menu and return the chosen plan key. `current` is the
    existing value (shown as the default when reconfiguring).
    """
    console.print(f"\nHow do you pay for {provider_label}?")
    for i, (_key, desc) in enumerate(choices, start=1):
        # The digit is what the user types, so it takes the accent. Rich used
        # to colour it anyway — along with every *other* number on the line,
        # which is how "Max 20x plan" ended up rendering as `Max 2` + a cyan
        # `0x`. Accenting only the choice key restores the affordance and
        # leaves the plan names alone.
        console.print(f"  [accent]{i}[/accent]) {desc}")
    keys = [k for k, _ in choices]
    default_idx = keys.index(current) + 1 if current in keys else 1
    raw = click.prompt(
        "Choose",
        type=click.IntRange(1, len(choices)),
        default=default_idx,
        show_default=True,
    )
    return keys[int(raw) - 1]


_ANALYSIS_SPAN_CHOICES = [
    ("30d", "Last 30 days"),
    ("90d", "Last 90 days"),
    ("all", "All available — keep everything, never delete"),
]


def _prompt_analysis_span(current: str | None = None) -> str:
    """Ask the one span question. Returns a key from ANALYSIS_SPAN_CHOICES.

    Deliberately framed as "how far back should tj analyze" rather than "how
    long should tj keep data": those were two settings that were free to
    disagree, and they did — retention was quietly deleting the oldest history
    the analyzers were still sizing their window against. There is one answer
    now and storage follows it.
    """
    console.print("\nHow far back should tj analyze?")
    for i, (_key, desc) in enumerate(_ANALYSIS_SPAN_CHOICES, start=1):
        console.print(f"  [accent]{i}[/accent]) {desc}")
    console.print(
        "  [muted]Older data is deleted, so this also decides what is kept.[/muted]"
    )
    keys = [k for k, _ in _ANALYSIS_SPAN_CHOICES]
    default_idx = keys.index(current) + 1 if current in keys else keys.index("90d") + 1
    raw = click.prompt(
        "Choose",
        type=click.IntRange(1, len(_ANALYSIS_SPAN_CHOICES)),
        default=default_idx,
        show_default=True,
    )
    return keys[int(raw) - 1]


def _apply_analysis_span(
    config: object, span_override: str | None = None, *, reconfigure: bool = False,
) -> None:
    """Settle the analysis span on `config` before it is written.

    The coupling lives in code, not in a comment: retention is derived from the
    span (`core/analysis_span.retention_days_for`), and an explicit
    `retention_days` shorter than the span is raised to it rather than honoured.
    Lowering storage retention must not be able to silently retract a span the
    product has already promised to analyze over, so the clamp only ever moves
    retention UP, and it says so when it fires.

    Idempotent and prompt-free once a span is on the config, so a flow that
    writes the config more than once (or the combination flow, which runs two
    of them over the same file) asks exactly once.
    """
    from tokenjam.core.analysis_span import (
        DEFAULT_ANALYSIS_SPAN,
        analysis_span_days,
        parse_analysis_span,
        retention_was_raised_to_span,
        span_label,
    )

    storage = config.storage  # type: ignore[attr-defined]
    if span_override:
        # Validate before storing: an unparseable span written to disk would
        # raise on every later read instead of here, where it can be corrected.
        parse_analysis_span(span_override)
        storage.analysis_span = span_override
    elif storage.analysis_span is None:
        if storage.retention_days is not None and not reconfigure:
            # A config written before the coupling existed. Its kept history IS
            # the most the product could ever have analyzed, so adopt it as the
            # span rather than re-asking — nothing about this setup changes.
            storage.analysis_span = f"{int(storage.retention_days)}d"
        elif _is_interactive():
            storage.analysis_span = _prompt_analysis_span(
                f"{int(storage.retention_days)}d"
                if storage.retention_days is not None else None
            )
        else:
            storage.analysis_span = DEFAULT_ANALYSIS_SPAN
    elif reconfigure and _is_interactive():
        storage.analysis_span = _prompt_analysis_span(storage.analysis_span)

    if retention_was_raised_to_span(storage):
        console.print(
            f"  Storage retention was shorter than the {span_label(storage)} "
            f"analysis span; raised to match so nothing tj analyzes can be "
            f"deleted underneath it."
        )
        storage.retention_days = analysis_span_days(storage)


def _is_interactive() -> bool:
    """True when onboard is running against a terminal (a human is present).

    Wrapped in a helper (rather than calling ``sys.stdin.isatty()`` inline) so
    it's a single, testable seam — Click's CliRunner swaps ``sys.stdin`` for a
    non-tty buffer, so tests patch this to force the interactive path.
    """
    try:
        return sys.stdin.isatty()
    except (ValueError, AttributeError):
        return False


_USAGE_PATH_CHOICES = [
    ("claude-code", "Claude Code"),
    ("codex",       "Codex"),
    ("sdk",         "Your own agents (Python/TS SDK or API)"),
    ("combination", "A combination of the above"),
]


def _prompt_usage_path() -> str:
    """Ask the path question that opens the bare `tj onboard` (#448).

    Returns the chosen path key: ``claude_code`` / ``codex`` / ``sdk`` /
    ``combination``. The caller routes to the matching flow. Only called when
    interactive (the non-tty path keeps the historical generic SDK behavior).
    """
    console.print()
    console.print("[bold]How do you use AI agents?[/bold]")
    for i, (_key, desc) in enumerate(_USAGE_PATH_CHOICES, start=1):
        console.print(f"  {i}) {desc}")
    raw = click.prompt(
        "Choose",
        type=click.IntRange(1, len(_USAGE_PATH_CHOICES)),
        default=1,
        show_default=True,
    )
    return _USAGE_PATH_CHOICES[int(raw) - 1][0]


def _try_backfill_codex(config) -> tuple[str | None, bool, int]:
    """Best-effort Codex backfill, called on the `--codex` / combination flows.

    The Codex on-disk backfill adapter (`tj backfill codex`) is imported at
    module scope, so a genuine import failure surfaces rather than being
    swallowed. Runtime ingest errors (e.g. the daemon holding the DB write
    lock) are still handled defensively so onboard never crashes — returning
    forward-only framing ("wired — data flows as you use Codex") instead of
    claiming a backfill that didn't happen (honesty discipline, Rule 14).

    Returns ``(message, has_data, sessions_total)``:
      * ``message`` — a human summary line, or None if nothing to report.
      * ``has_data`` — True only when at least one session was actually ingested.
      * ``sessions_total`` — distinct sessions ingested (0 when unavailable).
    """
    try:
        from tokenjam.core.db import open_db
        db = open_db(config.storage)
        try:
            with transient_status("Backfilling Codex sessions…"):
                result = ingest_codex(db, config=config)
        finally:
            db.close()
    except Exception as exc:
        err = str(exc).lower()
        if "lock" in err or "i/o error" in err or "io error" in err:
            return (
                "skipped — daemon holds the DB write lock. "
                "Stop the daemon (`tj stop`) and re-run `tj backfill codex`.",
                False,
                0,
            )
        return f"skipped ({exc})", False, 0

    def _rf(obj_key: str, dict_key: str, default: float = 0):
        # ingest_codex currently returns a summary dict; a future alignment to
        # BackfillResult (an object) reads the same way — forward-compatible.
        if isinstance(result, dict):
            return result.get(dict_key, default)
        return getattr(result, obj_key, default)

    total = int(_rf("sessions_total", "sessions_seen") or 0)
    if total <= 0:
        return None, False, 0
    new = int(_rf("sessions_new", "sessions_written", total) or 0)
    existing = total - new
    cost = float(_rf("total_cost_usd", "total_cost_usd", 0.0) or 0.0)
    spend = f" · ${cost:.0f} total spend" if cost > 0 else ""
    msg = (
        f"{new} new ({existing} already present) · "
        f"{total} total session{'s' if total != 1 else ''}{spend}"
    )
    return msg, True, total


def _print_setup_complete_home(
    *, sessions_backfilled: int = 0, has_data: bool = False,
    days: int | None = None,
) -> None:
    """Every onboard path ends here (#448): the branded home banner + tailored
    next-best-actions, so the closing screen is consistent no matter which path
    the user took.

    Renders a consistent "You're set up" close + an honest one-line
    "N sessions backfilled" note when a backfill actually happened (the
    branded welcome banner already printed at the top of the flow).
    Deliberately does NOT re-render
    ``cli/home.print_home``'s next-best-actions list — the onboard flows
    print their own curated next-steps block just above, and a second command
    list on the same screen read as duplication (founder review, 2026-07).
    Copy stays honest — no promised savings (Critical Rule 14).
    """
    console.print()
    if has_data and sessions_backfilled > 0:
        span = f" across the last {days} days" if days else ""
        console.print(
            f"[ok]✓ You're set up.[/ok] "
            f"{sessions_backfilled} session"
            f"{'s' if sessions_backfilled != 1 else ''} backfilled"
            f"{span}."
        )
    else:
        console.print("[ok]✓ You're set up.[/ok]")
    console.print("[muted]Full command list:[/muted]  [accent]tj --help[/accent]  "
                  "[muted]· home screen:[/muted]  [accent]tj[/accent]")


def _prompt_daily_budget(budget: float | None, plan_tier: str | None) -> float:
    """Prompt for the per-agent daily-budget alert threshold, unless already
    supplied via --budget or the just-resolved plan tier has no marginal
    per-token cost. Called AFTER the plan prompt so onboard reads plan-first
    (#240).

    Gated on plan_tier (#128): flat-rate subscription plans
    (``SUBSCRIPTION_PLAN_TIERS``) and local inference have a $0/day marginal
    cost, so asking for a USD budget ceiling right after the user just named
    their plan contradicts the framing discipline ``core/framing.py`` already
    applies everywhere else (dollar figures suppressed for those tiers), and
    the ``[budget.<provider>]`` USD monthly ceiling already skips them by
    design. Only ``api`` and ``unknown`` get prompted — and ``None`` (no
    resolved tier, e.g. an existing config with no plan section yet) behaves
    like ``unknown``: still prompt rather than assume $0. ``--budget`` always
    wins regardless of tier.
    """
    if budget is not None:
        return budget
    if plan_tier is not None and (plan_tier in SUBSCRIPTION_PLAN_TIERS or plan_tier == "local"):
        return 0.0
    return click.prompt(
        "Daily budget in USD (0 = no limit, default 0)",
        type=float, default=0.0, show_default=False,
    )


def _resolve_backfill_scope(
    backfill_days: int | None, backfill_all: bool, config: object | None = None,
):
    """Resolve the Claude Code backfill window (#443, #643).

    Returns ``(since, is_full, max_sessions)``:
      - ``is_full=True`` (``since``/``max_sessions`` both ``None``) means
        "everything" — the full `tj backfill claude-code` path, unbounded.
      - Otherwise ``since`` is the cutoff for `ingest_claude_code`, and
        ``max_sessions`` is an additional cap (or ``None`` for none).

    **One "how far back" question (#643).** Onboard used to ask the user twice —
    once "How far back should tj analyze?" (the analysis span) and again
    "Backfill your Claude Code history: 1) Last 30 days 2) Everything". Those
    are the same intent. The backfill window is now DERIVED from the already-
    resolved analysis span on ``config.storage`` (set by `_apply_analysis_span`
    just before this runs): a bounded span ("30d"/"90d") backfills that many
    days; an unbounded span ("all") backfills everything. No second prompt.

    The bounded path still pairs `since` with `max_sessions`: `since` alone
    doesn't reliably bound the work on an actively-used machine (the mtime
    pre-filter barely excludes anything when most session files have recent
    mtimes), so `max_sessions` (mirroring `tj quickstart`'s `DEFAULT_MAX_SESSIONS`
    cap, #13 — kept as the SAME constant so the two don't drift) guarantees
    bounded work regardless of mtime patterns.

    `--backfill-days N` is a scripting flag and means exactly what it says —
    days only, no implicit cap — so automation gets what it asked for.
    `--backfill-all` means everything, uncapped. Both still override the span.
    """
    from datetime import timedelta

    from tokenjam.cli.cmd_quickstart import DEFAULT_MAX_SESSIONS
    from tokenjam.core.analysis_span import analysis_span_days
    from tokenjam.utils.time_parse import utcnow

    def _print_complete_later_tip(days: int) -> None:
        console.print(
            f"[dim]  Backfilling the last {days} days. Run "
            f"`tj backfill claude-code` afterwards for your full history.[/dim]"
        )

    # Explicit scripting flags always win, unchanged from #443.
    if backfill_all:
        return None, True, None
    if backfill_days is not None:
        _print_complete_later_tip(backfill_days)
        return utcnow() - timedelta(days=backfill_days), False, None

    # Otherwise derive from the analysis span the user already answered.
    span_days: int | None = DEFAULT_BACKFILL_DAYS
    if config is not None:
        try:
            span_days = analysis_span_days(config.storage)  # type: ignore[attr-defined]
        except Exception:
            span_days = DEFAULT_BACKFILL_DAYS

    if span_days is None:
        # "all available" span → backfill everything, uncapped. No preamble
        # line: the post-backfill "N new · N total sessions" result line is the
        # honest count (sessionId-deduped), and a "~N sessions in scope"
        # pre-count over transcript FILES over-reports badly (Critical Rule 34).
        return None, True, None

    _print_complete_later_tip(span_days)
    return (
        utcnow() - timedelta(days=span_days), False, DEFAULT_MAX_SESSIONS,
    )


def _print_next_steps_nudge(
    *,
    has_data: bool,
    days: int | None = None,
    persona: str = "sdk",
    daemon_running: bool = False,
    port: int = 7391,
    show_restart: bool = False,
) -> None:
    """Curated post-onboard nudge (#240), persona-aware.

    Commands that work on the just-backfilled data *immediately*. The Claude
    Code list is View the Web UI (Lens) and View your Token Efficiency Card
    (``tj tokenmaxx``) — each an action label plus a concrete instruction —
    followed, when ``show_restart`` is set (a ``claude`` session is currently
    running), by an optional grey "restart Claude Code" line as the third item
    (#675). Restarting is NOT required for completeness: the daemon's transcript
    catch-up ingests running sessions from disk on its interval regardless, so
    the line is stated as optional. ``tj context`` / ``tj quota-audit`` still
    exist as commands; they are just not what a fresh user should be sent to
    first. ``tjb`` is an SDK-persona workflow (re-run your own agent on a
    cheaper model) so it only appears on the generic (non-Claude-Code) list. On
    that list, when onboarding just installed the daemon, the dashboard is
    already serving — suggesting ``tj serve`` there invites a port conflict, so
    that line says "already running" instead. Copy stays honest (no promised
    savings — Critical Rule 14).
    """
    console.print()
    lens_url = f"http://127.0.0.1:{port}/"
    if persona == "claude-code":
        # Two action/instruction rows, plain header (founder review, demo trim).
        console.print("[heading]▸ Next steps[/heading]")
        console.print()
        console.print(
            "  [label]View the Web UI (Lens)[/label]  "
            f"[go]open {lens_url} in a browser[/go]"
        )
        console.print(
            "  [label]View your Token Efficiency Card[/label]  "
            "[go]run tj tokenmaxx[/go]"
        )
        if show_restart:
            console.print(
                "  [muted]Optional: restart Claude Code to stream today's "
                "active sessions live -- otherwise tj picks them up from disk "
                "within a few minutes.[/muted]"
            )
        console.print()
        _warn_if_tj_path_unresolved()
        return
    if has_data:
        span = f"last {days} days" if days else "history"
        console.print(
            f"[heading]▸ Next steps[/heading]  [muted]your {span} "
            "already loaded; these work right now:[/muted]"
        )
    else:
        console.print(
            "[heading]▸ Next steps[/heading]  [muted]these work right now:[/muted]"
        )
    console.print()
    if daemon_running:
        lens_line = (
            f"  [label]web dashboard[/label]  [muted]already running →[/muted] [url]{lens_url}[/url]"
        )
    else:
        lens_line = (
            f"  [accent]tj serve[/accent]       "
            f"[muted]open the web dashboard at[/muted] [url]{lens_url}[/url]"
        )
    console.print(
        "  [accent]tj tokenmaxx[/accent]   [muted]your shareable efficiency tier[/muted]"
    )
    console.print(
        "  [accent]tj optimize[/accent]    "
        "[muted]cost-saving candidates from your usage[/muted]"
    )
    console.print(
        "  [accent]tjb[/accent]            "
        "[muted]prove a cheaper model still holds "
        "(pip install tokenjam-bench)[/muted]"
    )
    console.print(lens_line)
    console.print()
    _warn_if_tj_path_unresolved()


# --- PATH resolution guard ---------------------------------------------------
# Onboard installs a persistent `tj` (the ephemeral-runner guard above, or a
# prior pip/pipx/uv install) and then writes artifacts — this next-steps
# nudge, the statusline command, the claude() shell wrapper below — that
# invoke bare `tj` later, in whatever shell the user happens to be in at the
# time. That shell's PATH is NOT guaranteed to resolve `tj` to the install
# onboard manages: it may have no `~/.local/bin` on PATH at all (uv tool
# install's default shim dir), or an older `tj` (a stale pip install, a
# different venv) earlier on PATH shadowing it — confirmed in the wild in a
# VS Code integrated terminal whose PATH orders a Python.framework `tj`
# ahead of the uv-tool shim, while Terminal.app on the same machine resolved
# the shim fine minutes later. Detect both cases so the summary can fix
# (PATH missing) or warn (shadowed) instead of silently leaving next-steps
# commands that fail later.


def _current_tj_binary() -> str:
    """Absolute path to the ``tj`` binary onboard is running as right now.

    Derived from ``sys.executable`` (this process's interpreter), NOT
    ``shutil.which("tj")`` — the interpreter path is fixed at process start
    and can't be shadowed by a PATH change in some other, later shell, so
    it's the authoritative "what tj did onboard actually just set up"
    answer. Falls back to a PATH lookup, then the bare command, for the rare
    layout where no sibling ``tj`` sits next to the interpreter.
    """
    sibling = Path(sys.executable).with_name("tj")
    if sibling.exists():
        return str(sibling)
    return shutil.which("tj") or "tj"


def _probe_tj_path_resolution() -> tuple[str, str, str | None]:
    """Check whether bare ``tj`` on PATH (as this onboard process's shell
    sees it) resolves to the binary onboard manages.

    Returns ``(status, expected, shadow_path)``:
      * ``"ok"``         — bare ``tj`` resolves to *expected*.
      * ``"unresolved"`` — nothing named ``tj`` is on PATH.
      * ``"shadowed"``   — bare ``tj`` resolves to *shadow_path*, a
        different file than *expected*.
    """
    expected = _current_tj_binary()
    on_path = shutil.which("tj")
    if on_path is None:
        return "unresolved", expected, None
    try:
        same = Path(on_path).resolve() == Path(expected).resolve()
    except OSError:
        same = on_path == expected
    return ("ok", expected, None) if same else ("shadowed", expected, on_path)


def _tj_binary_version(path: str) -> str | None:
    """Best-effort ``<path> --version`` — names the shadowing binary in the
    PATH warning below. None on any failure (missing file, crash, timeout)."""
    try:
        result = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=5)
    except Exception:
        return None
    text = (result.stdout or result.stderr or "").strip()
    return text or None


_ZSHRC_PATH_START = "# >>> tokenjam PATH (managed) >>>"
_ZSHRC_PATH_END = "# <<< tokenjam PATH <<<"


def _zshrc_tj_path_block(bin_dir: str) -> str:
    """Idempotent PATH-export block ensuring *bin_dir* precedes the rest of
    PATH — the fallback when ``uv tool update-shell`` isn't available or
    doesn't cover the user's shell."""
    return (
        f"{_ZSHRC_PATH_START}\n"
        f'export PATH="{bin_dir}:$PATH"\n'
        f"{_ZSHRC_PATH_END}\n"
    )


def _ensure_tj_on_path(expected: str) -> str:
    """Best-effort fix for "nothing named tj is on PATH at all".

    Tries ``uv tool update-shell`` first — uv's own shell integration, which
    covers zsh/bash/fish profiles in one shot. Falls back to appending a
    small marker-delimited PATH export to ~/.zshrc, the same idempotent-block
    pattern already used for the OTEL export + claude wrapper below. Neither
    action can change THIS already-running process's PATH — the effect only
    lands in the next shell the user opens — so callers should still warn
    that a new terminal (or ``source ~/.zshrc``) is needed.
    """
    bin_dir = str(Path(expected).parent)
    if not Path(bin_dir).is_absolute():
        # A bare command name (the `_current_tj_binary` last-resort fallback)
        # carries no directory: Path("tj").parent is "." and exporting
        # `.:$PATH` would put the shell's CWD first on PATH — a classic
        # privilege-escalation footgun. Nothing safe to write; let the
        # caller's warning tell the user to fix PATH themselves.
        return "no-absolute-path"
    if shutil.which("uv"):
        try:
            result = subprocess.run(
                ["uv", "tool", "update-shell"], capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                return "ran-uv-update-shell"
        except Exception:
            pass
    zshrc = Path.home() / ".zshrc"
    zshrc.touch(exist_ok=True)
    text = zshrc.read_text()
    # Marker-only check (like every other zshrc block this module writes) —
    # NOT a raw `bin_dir in text` substring check: the claude() wrapper block
    # embeds this same absolute path for direct invocation (not a PATH
    # export), so that heuristic false-positived "already on PATH" and
    # skipped writing the export block entirely.
    if _ZSHRC_PATH_START in text:
        return "already-managed"
    block = _zshrc_tj_path_block(bin_dir)
    with zshrc.open("a") as f:
        f.write(("" if text.endswith("\n") or not text else "\n") + "\n" + block)
    return "wrote-zshrc-block"


def _print_tj_path_warning(
    status: str, expected: str, shadow_path: str | None, *, fix_status: str | None,
) -> None:
    """Surface a PATH-resolution problem in the onboard summary so bare `tj`
    commands don't silently hit the wrong (or no) binary later."""
    if status == "unresolved":
        console.print(
            "[warn]Heads up:[/warn] [accent]tj[/accent] isn't resolvable on "
            "PATH in a fresh shell yet, so the commands above will fail "
            "until you open a [label]new terminal[/label] (or run "
            "[accent]source ~/.zshrc[/accent])."
        )
        if fix_status in ("ran-uv-update-shell", "wrote-zshrc-block"):
            console.print(
                f"[muted]  Fixed for next time: added[/muted] "
                f"[accent]{expected}[/accent] [muted]to PATH.[/muted]"
            )
        if Path(expected).is_absolute():
            console.print(
                f"[muted]  Full path meanwhile:[/muted]  [accent]{expected}[/accent]"
            )
    elif status == "shadowed":
        # A "shadowed" status always carries the shadowing binary's path.
        assert shadow_path is not None
        shadow_version = _tj_binary_version(shadow_path) or "an older tj"
        console.print(
            f"[warn]Heads up:[/warn] [label]{shadow_version}[/label] at "
            f"[accent]{shadow_path}[/accent] shadows the tj just installed at "
            f"[accent]{expected}[/accent]. Bare [accent]tj[/accent] in this "
            "shell resolves to the older one."
        )
        console.print(
            f"[muted]  Use the full path, or move[/muted] "
            f"[accent]{Path(expected).parent}[/accent] "
            f"[muted]earlier on PATH:[/muted]  [accent]{expected}[/accent]"
        )


def _warn_if_tj_path_unresolved() -> None:
    """Probe bare `tj` PATH resolution and fix-or-warn before the next-steps
    commands above rely on it. No-op in the common case (already resolves to
    the tj onboard manages)."""
    status, expected, shadow_path = _probe_tj_path_resolution()
    if status == "ok":
        return
    fix_status = _ensure_tj_on_path(expected) if status == "unresolved" else None
    _print_tj_path_warning(status, expected, shadow_path, fix_status=fix_status)


def _tj_statusline_command() -> str:
    """Return the command Claude Code should invoke for the tj statusline.

    Prefer an absolute path to the installed ``tj`` (robust against Claude Code
    running the statusline with a minimal PATH); fall back to the bare ``tj``.
    """
    tj_bin = shutil.which("tj")
    return f"{tj_bin} statusline" if tj_bin else "tj statusline"


def _is_tj_statusline(entry: object) -> bool:
    """True if *entry* is a statusLine config that already points at tj."""
    if isinstance(entry, dict):
        cmd = entry.get("command", "")
        return isinstance(cmd, str) and "tj statusline" in cmd
    return False


def _wire_claude_statusline(settings: dict) -> str:
    """Idempotently wire the tj statusline into a Claude Code settings dict.

    MUTATES *settings* in place (the caller writes it once). Returns a status:
      * ``"written"``  — no statusLine before; ours was added.
      * ``"updated"``  — an existing tj statusLine was refreshed (e.g. path).
      * ``"kept"``     — already exactly ours; nothing changed.
      * ``"skipped"``  — a foreign/human-authored statusLine exists; left intact.

    settings.json is a user contract: we never clobber a statusLine the user (or
    another tool like ccstatusline) authored.
    """
    desired = {"type": "command", "command": _tj_statusline_command()}
    existing = settings.get("statusLine")
    if existing is None:
        settings["statusLine"] = desired
        return "written"
    if _is_tj_statusline(existing):
        if existing == desired:
            return "kept"
        settings["statusLine"] = desired
        return "updated"
    return "skipped"


# --- zshrc OTEL export block (harness observability) ----------------------
# Installed into ~/.zshrc so harness (Docker) sessions pick up the OTLP env
# vars automatically. Delimited by a STABLE, content-based sentinel pair that
# is never renamed — earlier revisions keyed on a bare single-line comment
# marker instead ("# ocw harness observability" pre-rebrand, then "# tj
# harness observability" after), so a block written under an older marker was
# invisible to both re-onboard's replace-in-place and `tj uninstall`'s
# removal. Consequence: a real ~/.zshrc could carry BOTH an old-marker block
# and a new-marker block, each with a different bearer token — re-onboard
# APPENDED a second block instead of replacing the first (stale secrets
# accumulate in the user's shell rc), and uninstall only removed the current
# marker's block, leaving the old one behind (#118). onboard now strips ALL
# managed blocks (current sentinel + every legacy marker) before writing
# exactly one fresh block; uninstall strips the same set.

_ZSHRC_OTEL_START = "# >>> tokenjam OTEL (managed) >>>"
_ZSHRC_OTEL_END = "# <<< tokenjam OTEL <<<"
# Legacy single-line markers used before the sentinel pair (pre-#118): each
# preceded a run of `export ...` lines with no closing delimiter.
_ZSHRC_OTEL_LEGACY_MARKERS = (
    "# tj harness observability",
    "# ocw harness observability",
)


def _report_secret_alignment(secret: str) -> None:
    """Repair a project-local config carrying a stale ingest secret.

    The global secret is the survivor, never the project-local one: the managed
    shell block, the daemon unit file and every already-working integration
    carry it, so adopting a project value would break all of them at once. A
    left-over `.tj/config.toml` from an older install is silently rewritten to
    match, and the repair is reported (a config file edited without a word is
    exactly how a divergence goes unnoticed in the first place).
    """
    repaired = align_project_secret_to_global(secret)
    if repaired is not None:
        console.print(
            f"  Aligned the ingest secret in {display_path(repaired)} with "
            f"{display_path(_global_config_path())}.",
            soft_wrap=True,
        )


_OTEL_HOST_ENV = "TJ_OTEL_HOST"
_DOCKER_GATEWAY_HOST = "host.docker.internal"


def _in_container() -> bool:
    """True when this process is itself running inside a container.

    Docker writes `/.dockerenv`; the cgroup path names the runtime under
    Docker/Podman/containerd. Best-effort and never raising: an unreadable
    cgroup file simply means "not detected", which lands on the loopback
    default that is correct on a host.
    """
    if Path("/.dockerenv").exists():
        return True
    try:
        cgroup = Path("/proc/self/cgroup").read_text()
    except OSError:
        return False
    return any(rt in cgroup for rt in ("docker", "containerd", "podman", "kubepods"))


def _otel_endpoint_host() -> str:
    """The host an OTLP exporter configured by onboarding should send to.

    `host.docker.internal` is the address a process INSIDE a container uses to
    reach a service on its host. It was written unconditionally, including into
    a host user's ~/.zshrc, where it does not resolve at all: every span an
    agent session emitted failed at DNS and was dropped silently, leaving only
    the delayed transcript backfill. Two event types have no transcript
    equivalent (tool decisions and API errors), so those were lost outright.

    So it is now conditional on actually being in that environment, and
    overridable by `TJ_OTEL_HOST` for a setup this cannot detect (an exported
    shell rc consumed inside a container, a remote daemon).
    """
    override = os.environ.get(_OTEL_HOST_ENV, "").strip()
    if override:
        return override
    if _in_container() and _host_resolves(_DOCKER_GATEWAY_HOST):
        return _DOCKER_GATEWAY_HOST
    return "127.0.0.1"


def _host_resolves(host: str) -> bool:
    """True when `host` resolves via DNS on this machine. Never raises."""
    import socket

    try:
        socket.getaddrinfo(host, None)
    except OSError:
        return False
    return True


def _zshrc_otel_block(port: int, secret: str) -> str:
    """Build one fresh sentinel-delimited OTEL export block for ~/.zshrc."""
    host = _otel_endpoint_host()
    return (
        f"{_ZSHRC_OTEL_START}\n"
        f"export CLAUDE_CODE_ENABLE_TELEMETRY=1\n"
        f"export OTEL_LOGS_EXPORTER=otlp\n"
        f"export OTEL_EXPORTER_OTLP_PROTOCOL=http/json\n"
        f"export OTEL_EXPORTER_OTLP_ENDPOINT=http://{host}:{port}\n"
        f'export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer {secret}"\n'
        f"{_ZSHRC_OTEL_END}\n"
    )


def _strip_zshrc_otel_blocks(text: str) -> str:
    """Remove every tj-managed OTEL block from ~/.zshrc content — the current
    sentinel-delimited block AND any legacy single-marker block, however many
    accumulated. Shared by onboard (replace-all before writing one fresh
    block) and uninstall (removal only). Idempotent: text with no managed
    blocks is returned unchanged.

    Both patterns treat the trailing newline after the block as optional
    (``(?:\\n|$)``/``\\n?``) — requiring a hard ``\\n`` missed a managed block
    that happens to be the last line of a file with no final newline, leaving
    a bearer token behind after "removal"."""
    import re as _re

    cleaned = _re.sub(
        rf"{_re.escape(_ZSHRC_OTEL_START)}\n.*?{_re.escape(_ZSHRC_OTEL_END)}(?:\n|$)",
        "",
        text,
        flags=_re.DOTALL,
    )
    for marker in _ZSHRC_OTEL_LEGACY_MARKERS:
        cleaned = _re.sub(
            rf"{_re.escape(marker)}\n(?:export [^\n]+(?:\n|$))*",
            "",
            cleaned,
        )
    return cleaned


def _onboard_claude_code(
    ctx: click.Context,
    budget: float | None,
    no_daemon: bool,
    force: bool,
    reconfigure: bool = False,
    plan_override: str | None = None,
    verify: bool = False,
    standalone: bool = True,
    backfill_days: int | None = None,
    backfill_all: bool = False,
    plan_usd_override: float | None = None,
    analysis_span: str | None = None,
    verbose: bool = False,
) -> None:
    """Configure Claude Code to send telemetry to tj.

    ``standalone`` is True on the single-path flow (`tj onboard --claude-code`)
    and False when this runs as one leg of the combination flow (#432). The
    Claude Code flow itself ends right after the Next steps block (no closing
    home banner — founder review, demo trim); the combination flow prints its
    own closing banner once at the end of `_onboard_combination`. The inline
    Claude Code backfill still runs (it is only ever invoked from here).

    ``plan_usd_override`` is the pre-collected API monthly spend ceiling that
    pairs with ``plan_override``: when the combination flow hoists the billing
    questions up front (so every leg's plan is asked before any leg's
    long-running backfill), it threads the ceiling it already collected in here
    instead of the leg re-prompting for it. It is only consulted when
    ``plan_override`` is set, so the standalone flow (no overrides) is unchanged.
    """
    from tokenjam.core.config import (
        AgentConfig, BudgetConfig, ProviderBudget, TjConfig, SecurityConfig,
        load_config, write_config,
    )

    # --claude-code always uses the global config so that all projects share one
    # ingest secret and one running daemon. Per-project configs cause the secret in
    # ~/.claude/settings.json to rotate on every project onboard, breaking auth for
    # every other project.
    global_config_path = _global_config_path()

    project_name = _derive_project_name()
    agent_id = f"claude-code-{project_name}"

    # Plan-first (#240): resolve the plan tier before prompting for the daily
    # budget — "How do you pay?" is the more important, more natural opener.
    if global_config_path.exists() and not force:
        config = load_config(str(global_config_path))
        is_new_agent = agent_id not in config.agents
        if is_new_agent:
            config.agents[agent_id] = AgentConfig()

        # A config written before prompt/tool-input capture defaulted on has
        # an explicit `prompts = false` / `tool_inputs = false` baked in —
        # don't skip a stale-value rewrite just because the key is already
        # present (the same dotfile-managed-block stale-marker trap CLAUDE.md
        # documents elsewhere). `--reconfigure` is a deliberate "redo my
        # setup" action, so treat it as license to pick up the current
        # defaults rather than leaving those stale values in place forever.
        if reconfigure:
            config.capture.prompts = True
            config.capture.tool_inputs = True

        existing_plan = (
            config.budgets["anthropic"].plan
            if "anthropic" in config.budgets else None
        )
        plan_changed = False
        plan = existing_plan
        # Prompt for plan tier when:
        #   - this is a fresh onboard for this agent (no existing plan), or
        #   - the user passed --reconfigure to explicitly re-prompt
        # `plan_override` (from --plan CLI flag) bypasses the prompt entirely.
        if existing_plan is None or reconfigure or plan_override:
            if plan_override:
                plan = plan_override
            else:
                plan = _prompt_plan("Claude", _ANTHROPIC_PLAN_CHOICES, current=existing_plan)
            plan_changed = plan != existing_plan
            # Subscription plans don't get an auto-written budget ceiling.
            # A pre-collected ceiling (combination flow) rides in via
            # plan_usd_override; standalone leaves it None and prompts below.
            usd: float | None = plan_usd_override
            if plan == "api" and not plan_override:
                # API users may want a self-imposed soft ceiling — only prompt
                # when interactive (no --plan flag).
                ceiling = click.prompt(
                    "Monthly Anthropic API spend ceiling in USD (0 = no limit)",
                    type=float, default=0.0, show_default=False,
                )
                if ceiling > 0:
                    usd = ceiling
            existing_budget = config.budgets.get("anthropic")
            if existing_budget is not None:
                existing_budget.plan = plan
                if usd is not None:
                    existing_budget.usd = usd
            else:
                config.budgets["anthropic"] = ProviderBudget(
                    usd=usd, cycle_start_day=1, plan=plan,
                )
        budget = _prompt_daily_budget(budget, plan)
        if budget and budget > 0:
            config.agents[agent_id].budget.daily_usd = budget
        # Server-side project mapping so already-running sessions group by
        # project without restarting the agent (see AgentConfig.project).
        # Only set it for a genuinely NEW agent — an existing one may carry a
        # hand-entered name that differs from the derived one (the prompt
        # this replaced let a user type anything), and silently overwriting
        # that would orphan its dashboard history under a new identity, a
        # data-loss-shaped bug even though no rows are deleted.
        if is_new_agent or not config.agents[agent_id].project:
            config.agents[agent_id].project = project_name
        config_path = global_config_path
        _apply_analysis_span(config, analysis_span, reconfigure=reconfigure)
        write_config(config, config_path)
        console.print(f"  tj config updated: {display_path(config_path)}", soft_wrap=True)
    else:
        ingest_secret = secrets.token_hex(32)
        if plan_override:
            plan = plan_override
        else:
            plan = _prompt_plan("Claude", _ANTHROPIC_PLAN_CHOICES)
        plan_changed = False
        usd: float | None = plan_usd_override  # type: ignore[no-redef]
        if plan == "api" and not plan_override:
            ceiling = click.prompt(
                "Monthly Anthropic API spend ceiling in USD (0 = no limit)",
                type=float, default=0.0, show_default=False,
            )
            if ceiling > 0:
                usd = ceiling
        budget = _prompt_daily_budget(budget, plan)
        daily_usd = budget if budget and budget > 0 else None
        # No existing config at all — nothing stored to respect.
        agents = {agent_id: AgentConfig(budget=BudgetConfig(daily_usd=daily_usd), project=project_name)}
        config = TjConfig(
            version="1",
            agents=agents,
            security=SecurityConfig(ingest_secret=ingest_secret),
            budgets={"anthropic": ProviderBudget(
                usd=usd, cycle_start_day=1, plan=plan,
            )},
        )
        config_path = global_config_path
        config_path.parent.mkdir(parents=True, exist_ok=True)
        _apply_analysis_span(config, analysis_span, reconfigure=reconfigure)
        write_config(config, config_path)
        console.print(
            f"[check]\u2713[/check] Config written to "
            f"[accent]{display_path(config_path)}[/accent]",
            soft_wrap=True,
        )
        if _sync_secret_to_codex(ingest_secret):
            console.print("  Codex config updated to match new ingest secret.")

    stopped_for_db = _stop_serve_for_db_write()
    if stopped_for_db:
        console.print(
            "[dim]  Server:              stopped briefly for "
            "config/DB update[/dim]"
        )

    apply_msg = _try_apply_declared_plans(
        config, reconcile=reconfigure and plan_changed,
    )
    if apply_msg:
        console.print(f"  {apply_msg}")

    # --- Backfill existing Claude Code session logs ---
    # Run before daemon install so a freshly-started serve does not hold the
    # DuckDB write lock (#71). Idempotent — safe to re-run.
    backfill_msg: str | None = None
    backfill_has_data = False
    backfill_span_days: int | None = None
    backfill_sessions_total = 0
    try:
        from tokenjam.cli.backfill_progress import backfill_progress
        from tokenjam.core.backfill import (
            CLAUDE_CODE_PROJECTS_ROOT,
            count_claude_code_sessions_in_scope,
            ingest_claude_code,
        )
        if CLAUDE_CODE_PROJECTS_ROOT.exists():
            from tokenjam.core.db import open_db
            try:
                since, _backfill_is_full, max_sessions = _resolve_backfill_scope(
                    backfill_days, backfill_all, config,
                )
                # Used only to size the progress bar below. Deliberately NOT
                # printed as a "~N sessions in scope" heads-up: it counts
                # transcript FILES, which over-reports the true (sessionId-
                # deduped) session count badly (Critical Rule 34) — the
                # post-backfill "N new · N total sessions" line is the honest
                # figure.
                total_in_scope = count_claude_code_sessions_in_scope(
                    since=since, max_sessions=max_sessions,
                )
                db = open_db(config.storage)
                with backfill_progress(total_in_scope) as backfill_progress_cb:
                    result = ingest_claude_code(
                        db, config=config, since=since,
                        max_sessions=max_sessions,
                        progress=backfill_progress_cb,
                    )
                if result.limit_reached and max_sessions is not None:
                    console.print(
                        f"[muted]  Showing your most-recent {max_sessions} "
                        f"sessions for a fast first run; run[/muted] "
                        f"[accent]tj backfill claude-code[/accent] "
                        f"[muted]for your full history.[/muted]"
                    )
                post_apply = _try_apply_declared_plans(
                    config, reconcile=reconfigure and plan_changed,
                )
                if post_apply and not apply_msg:
                    console.print(f"  {post_apply}")
                db.close()
                if result.sessions_total > 0:
                    backfill_has_data = True
                    days = None
                    if result.earliest and result.latest:
                        days = (result.latest - result.earliest).days
                    backfill_span_days = days
                    backfill_sessions_total = result.sessions_total
                    # Report new / already-present / total so a re-run reads as
                    # "13 total" rather than "1 session" (#238).
                    total = result.sessions_total
                    pieces = [
                        f"{result.sessions_new} new "
                        f"({result.sessions_existing} already present) · "
                        f"{total} total session{'s' if total != 1 else ''}"
                    ]
                    if days is not None:
                        pieces.append(f"over {days} day{'s' if days != 1 else ''}")
                    # Dollar spend only for per-token billing: flat-fee
                    # subscription and local tiers get dollar figures
                    # suppressed everywhere else (core/framing.py), and a
                    # "$N total spend" two prompts after the user declared a
                    # $100/mo subscription reads as tj ignoring its own
                    # question.
                    if not (plan and (plan in SUBSCRIPTION_PLAN_TIERS or plan == "local")):
                        pieces.append(f"${result.total_cost_usd:.0f} total spend")
                    backfill_msg = ", ".join(pieces)
            except Exception as exc:
                _err = str(exc).lower()
                if "lock" in _err or "i/o error" in _err or "io error" in _err:
                    backfill_msg = (
                        "skipped — daemon holds the DB write lock. "
                        "Stop the daemon (`tj stop`) and re-run "
                        "`tj backfill claude-code`."
                    )
                else:
                    backfill_msg = f"skipped ({exc})"
    except Exception:
        pass

    # --- tj is out-of-band for Claude Code: statusline, NOT MCP ---
    # An in-loop MCP server is a per-turn quota burden on CC subscription power
    # users (a measured A/B showed tj-MCP-in-loop cost +36% model-weighted quota
    # vs a no-tj control). So we deliberately do NOT run `claude mcp add tj`
    # here. Claude Code gets tj via the zero-token statusline wired below plus
    # the existing out-of-band OTel telemetry ingest. The MCP is reserved for
    # SDK / API users, where tj sits in the request path for real-time
    # enforcement. (`tj mcp` still works for them; we just don't default CC
    # users into it.)

    # --- Global settings (~/.claude/settings.json) ---
    claude_dir = Path.home() / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    global_settings_path = claude_dir / "settings.json"

    global_settings: dict = {}
    if global_settings_path.exists():
        try:
            global_settings = json_mod.loads(global_settings_path.read_text())
        except (json_mod.JSONDecodeError, OSError):
            global_settings = {}

    # Write global OTLP config — always overwrite endpoint vars so reinstall stays in sync.
    # Custom headers (non-TokenJam) are preserved; only TokenJam-generated "Authorization=Bearer"
    # headers are replaced when the secret rotates.
    port = config.api.port
    secret = config.security.ingest_secret
    _report_secret_alignment(secret)
    global_env: dict = global_settings.get("env", {})
    global_env["CLAUDE_CODE_ENABLE_TELEMETRY"] = "1"
    global_env["OTEL_LOGS_EXPORTER"] = "otlp"
    global_env["OTEL_EXPORTER_OTLP_PROTOCOL"] = "http/json"
    global_env["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"http://127.0.0.1:{port}"
    existing_header = global_env.get("OTEL_EXPORTER_OTLP_HEADERS", "")
    if secret and (not existing_header or "Authorization=Bearer" in existing_header):
        global_env["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Bearer {secret}"
    global_settings["env"] = global_env
    # Wire the zero-token statusline (non-destructively — a human-authored or
    # third-party statusLine is a contract we never clobber).
    statusline_status = _wire_claude_statusline(global_settings)

    # Wire the SessionStart resume-brief hook (zero in-loop token cost). Rides
    # this same read-merge-write. Installed by default: on a resumed /
    # post-compaction session, `tj resume-brief --from-hook` reads the
    # session_id / transcript_path Claude Code pipes on stdin and re-injects
    # THAT session's prior method as additionalContext (a global-mtime scan
    # could cross-leak a concurrent session's brief). Idempotent +
    # non-destructive (foreign SessionStart hooks preserved); removed by
    # `tj uninstall`.
    # Installed silently — no printed status line (founder review, demo trim).
    _wire_claude_resume_brief_hook(global_settings)

    # The output-trim PostToolUse hook was removed (measured negative: +5.6%
    # whole-session cost on Claude Code, see CLAUDE.md). Best-effort cleanup
    # only, for users who opted into a prior release's hook — never installs.
    cap_removed = _unwire_claude_output_cap_hook(global_settings)

    global_settings_path.write_text(json_mod.dumps(global_settings, indent=2) + "\n")
    if cap_removed:
        console.print(
            "[ok]✓[/ok] Removed the legacy output-trim hook "
            "(tj hook cap-output) — this feature was removed."
        )
    # --- Project settings (<cwd>/.claude/settings.json) ---
    project_claude_dir = Path.cwd() / ".claude"
    project_claude_dir.mkdir(parents=True, exist_ok=True)
    project_settings_path = project_claude_dir / "settings.json"

    project_settings: dict = {}
    if project_settings_path.exists():
        try:
            project_settings = json_mod.loads(project_settings_path.read_text())
        except (json_mod.JSONDecodeError, OSError):
            project_settings = {}

    # The `claude` shell wrapper (installed below) now owns
    # OTEL_RESOURCE_ATTRIBUTES, exporting a distinct service.instance.id per
    # terminal. Claude Code's settings.json `env` block OVERRIDES shell env, so
    # a value hardcoded here would clobber the wrapper's per-terminal value and
    # silently collapse every terminal back into one dashboard tile. Do not
    # write it, and delete any pre-existing one to migrate older setups (other
    # env keys are left untouched).
    project_env: dict = project_settings.get("env", {})
    removed_resource_attr = project_env.pop("OTEL_RESOURCE_ATTRIBUTES", None) is not None
    project_settings["env"] = project_env
    project_settings_path.write_text(json_mod.dumps(project_settings, indent=2) + "\n")

    # --- Track onboarded project paths for clean uninstall ---
    projects_index = config_path.parent / "projects.json"
    try:
        known: list[str] = json_mod.loads(projects_index.read_text()) if projects_index.exists() else []
    except (json_mod.JSONDecodeError, OSError):
        known = []
    cwd_str = str(Path.cwd())
    if cwd_str not in known:
        known.append(cwd_str)
        projects_index.write_text(json_mod.dumps(known, indent=2) + "\n")

    # --- Shell env (~/.zshrc) ---
    # The endpoint host is chosen by `_otel_endpoint_host()` for the machine
    # being written to: loopback on a host, the Docker gateway only when this
    # is genuinely running inside a container, and `TJ_OTEL_HOST` overrides
    # both. Native Claude Code uses settings.json (127.0.0.1) written above.
    zshrc = Path.home() / ".zshrc"
    zshrc.touch(exist_ok=True)
    zshrc_text = zshrc.read_text()
    # Replace-all (#118): strip every managed block — current sentinel AND any
    # legacy marker left over from a pre-rebrand or pre-sentinel install —
    # then write exactly one fresh block. A bare `not in` check on a single
    # marker string missed blocks written under an older marker, so re-onboard
    # would append a second block with a stale bearer token instead of
    # replacing it.
    stripped = _strip_zshrc_otel_blocks(zshrc_text)
    new_block = _zshrc_otel_block(port, secret)
    updated = (stripped.rstrip("\n") + "\n\n" + new_block) if stripped.strip() else new_block
    zshrc.write_text(updated)

    # --- Per-terminal naming: install the `claude` shell wrapper ---
    # Tags each terminal with a distinct service.instance.id so concurrent
    # Claude Code sessions render as separate dashboard tiles, without the user
    # hand-editing their shell rc. Idempotent across re-onboards.
    wrapper_files = _install_claude_wrapper()

    want_daemon = not no_daemon
    daemon_result = _finish_onboard_serve(
        str(config_path.resolve()),
        want_daemon=want_daemon,
        plan_changed=plan_changed,
        stopped_for_db=stopped_for_db,
        secret_rotated=False,
        no_daemon=no_daemon,
        force=force,
        # --claude-code completion (#675): the payoff screen reports via three
        # `✓` lines, so suppress the internal Daemon/Server-restart status lines
        # and defer the macOS "Background Items Added" heads-up to the bottom.
        quiet_daemon=True,
    )
    # `quiet_daemon` suppresses the routine daemon lines, but a daemon
    # install/restart FAILURE must never be silently swallowed — a green
    # completion screen while the server is actually down is worse than no line
    # (Greptile, #675). A clean install returns "Daemon installed at ..."; a
    # clean restart "restarted to pick up ..."; a skip "daemon already running".
    # Anything else off the restart path (could-not-auto-restart / stopped-stale
    # / restart-attempted-verify) means the server may not be up — surface it.
    daemon_needs_attention = daemon_result is not None and daemon_result.startswith(
        ("could not", "stopped stale server", "restart attempted")
    )
    if daemon_needs_attention:
        console.print()
        console.print(
            f"[warn]Daemon:[/warn] {daemon_result}. "
            "Verify with [accent]tj status[/accent]."
        )

    # ── Completion screen (founder review, 2026-07): what got wired → the one
    # required action (restart, prominent, one why-first panel) → next steps →
    # connection details (dim footer). The old screen led with a config dump
    # and buried the restart box below the next-steps list; a later pass split
    # the restart guidance across four spots (the panel, a stray paragraph, an
    # "after restarting" pointer, and a "verify after restarting" line near
    # Connection details); consolidated back into one panel below.
    console.print()
    # Lean payoff screen (#675): three green `\u2713` status lines (config written /
    # observability configured / sessions backfilled), then the privacy note,
    # then Next steps (with the optional restart as its third item), and the
    # macOS "Background Items Added" heads-up last of all. The Daemon / Server-
    # restart / Statusline / verbose "Backfilled: N new \u00b7 N total" status lines
    # that used to interleave here are gone from the default screen; the
    # connection-details block behind `--verbose` (and `tj doctor`) still has
    # the mechanics for anyone who wants them.
    console.print("[check]\u2713[/check] Claude Code observability configured.")
    if backfill_has_data and backfill_sessions_total > 0:
        # Omit the span for a same-day corpus (0 days or unknown) — "over 0
        # days" reads odd and adds nothing.
        span = (
            f", over {backfill_span_days} day"
            f"{'s' if backfill_span_days != 1 else ''}"
            if backfill_span_days else ""
        )
        console.print(
            f"[check]\u2713[/check] "
            f"[bold]Claude Code sessions backfilled[/bold]: "
            f"{backfill_sessions_total} session"
            f"{'s' if backfill_sessions_total != 1 else ''}{span}."
        )
    elif backfill_msg:
        # Backfill ran but found no sessions, or was skipped (lock / error).
        # Surface the reason as one plain line rather than a bare `\u2713`.
        console.print(f"  [muted]Backfill: {backfill_msg}[/muted]")
    _print_capture_disclosure(config.capture.prompts, config.capture.tool_inputs)
    # (`_print_next_steps_nudge` prints its own leading blank line.)
    _print_next_steps_nudge(
        has_data=backfill_has_data, days=backfill_span_days,
        persona="claude-code", daemon_running=want_daemon, port=port,
        # The optional "restart Claude Code" line is now the third Next-steps
        # item (#675); shown only when `claude` is actually running, since a
        # fresh terminal with no open session needs no restart.
        show_restart=_claude_code_is_running(),
    )
    if not want_daemon:
        _warn_manual_serve_restart(stopped_for_db=stopped_for_db, no_daemon=True)
    # --verbose (#643): the connection-details block + internal mechanics.
    # Off by default -- this is debug/reference noise on the first-run payoff
    # moment, and `tj doctor` surfaces the same information on demand.
    if verbose:
        console.print("[dim]Connection details[/dim]")
        console.print(
            "  Telemetry:           Claude Code -> tj, out-of-band "
            "(global settings + ~/.zshrc)"
        )
        if wrapper_files:
            console.print(
                "  claude wrapper:      per-terminal dashboard tiles "
                "(~/.zshrc); claude --as <name> labels a terminal"
            )
        if statusline_status == "skipped":
            console.print(
                "[dim]  Statusline:          left untouched -- set your "
                "Claude Code statusLine command to[/dim]  tj statusline"
            )
        console.print(
            f"[dim]  Global settings:    {display_path(global_settings_path)}[/dim]",
            soft_wrap=True,
        )
        console.print(
            f"[dim]  Project settings:   {display_path(project_settings_path)}[/dim]",
            soft_wrap=True,
        )
        if removed_resource_attr:
            console.print(
                "[muted]  Removed a hardcoded OTEL_RESOURCE_ATTRIBUTES from "
                "project settings (the claude wrapper now sets it per "
                "terminal).[/muted]"
            )
        console.print(f"[dim]  Agent ID:           {agent_id}[/dim]")
        if budget and budget > 0:
            console.print(f"[dim]  Daily budget:       ${budget:.2f}[/dim]")
        console.print(
            f"[dim]  OTLP endpoint:      http://127.0.0.1:{port} (native) . "
            f"http://{_otel_endpoint_host()}:{port} (shell env)[/dim]"
        )
        if secret:
            console.print(f"[dim]  Ingest secret:      {secret[:8]}...[/dim]")
        from tokenjam.core.config import load_config as _lc
        try:
            _cfg = _lc(str(global_config_path))
            _ab = _cfg.budgets.get("anthropic")
            if _ab is not None and _ab.plan:
                _line = f"[budget.anthropic] plan = \"{_ab.plan}\""
                if _ab.usd:
                    _line += f", usd = {_ab.usd}"
                console.print(f"[dim]  Budget projection:  {escape(_line)}[/dim]")
        except Exception:
            pass
        console.print()

    _maybe_verify_onboarding(config, persona="claude-code", verify=verify)

    # macOS "Background Items Added" heads-up last of all (#675): it was
    # suppressed at daemon-install time (`quiet_daemon`) so it lands here, after
    # Next steps, at the very bottom of the payoff screen. Gate it on a NEW
    # background item having actually been registered — a fresh `_install_launchd`
    # returns "Daemon installed at ..." — not merely on `want_daemon` (Greptile,
    # #675). A skipped reinstall ("daemon already running"), a pure restart, or a
    # failed install registers no new item, so no notification fires and the
    # heads-up would be a lie.
    daemon_installed_new = daemon_result is not None and daemon_result.startswith(
        "Daemon installed"
    )
    if daemon_installed_new:
        console.print()
        _print_background_items_headsup()

    # The Claude Code flow ends right after the Next steps block (founder
    # review, demo trim): no "Recurring mistakes" pointer and no "You're set up
    # / full command list" footer. The other onboard paths keep those via
    # `_print_setup_complete_home`. The backfilled-session count is reported on
    # the "✓ Claude Code sessions backfilled" line above.


def _onboard_codex(
    ctx: click.Context,
    budget: float | None,
    no_daemon: bool,
    force: bool,
    reconfigure: bool = False,
    plan_override: str | None = None,
    plan_usd_override: float | None = None,
    verify: bool = False,
    standalone: bool = True,
    analysis_span: str | None = None,
) -> None:
    """Configure Codex CLI to send telemetry to tj.

    ``standalone`` is True on the single-path flow (`tj onboard --codex`) and
    False when this runs as one leg of the combination flow (#432). When not
    standalone we skip BOTH the internal Codex backfill and the closing home
    banner: `_onboard_combination` runs the Codex backfill exactly once itself
    and prints the banner exactly once at the very end. Running the backfill
    here too would double-run it, and printing the banner here would show it up
    to three times.

    ``plan_usd_override`` mirrors the same param on ``_onboard_claude_code``:
    the combination flow hoists the OpenAI billing questions up front and
    threads the pre-collected API spend ceiling in here (consulted only when
    ``plan_override`` is set), so this leg re-asks nothing on the combination
    path. Standalone (no overrides) is unchanged.
    """
    try:
        import tomllib  # type: ignore[import]
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    from tokenjam.core.config import (
        AgentConfig, BudgetConfig, ProviderBudget, TjConfig, SecurityConfig,
        load_config, write_config,
    )

    # Codex hardcodes service.name="codex_exec" in its binary regardless of
    # what [otel.resource] says, so all Codex traces land under "codex_exec".
    agent_id = "codex_exec"

    # `--codex` always writes to the global config, mirroring `--claude-code`.
    # Codex's own config (~/.codex/config.toml) is global and the agent_id
    # `codex_exec` is project-agnostic by design (Codex hardcodes service.name
    # in its binary). Per-project TokenJam configs would rotate the secret on every
    # onboard, breaking the running server.
    config_path = _global_config_path()

    previous_secret: str | None = None
    if config_path.exists():
        try:
            prev_cfg = load_config(str(config_path))
            previous_secret = prev_cfg.security.ingest_secret
        except Exception:
            previous_secret = None

    # Plan-first (#240): resolve the plan tier before prompting for the daily
    # budget — "How do you pay?" is the more important, more natural opener.
    if config_path.exists():
        config = load_config(str(config_path))
        if agent_id not in config.agents:
            config.agents[agent_id] = AgentConfig()

        # See the matching comment in `_onboard_claude_code`: `--reconfigure`
        # is license to pick up the current capture defaults rather than
        # leaving stale pre-default `prompts = false` / `tool_inputs = false`
        # in place forever.
        if reconfigure:
            config.capture.prompts = True
            config.capture.tool_inputs = True

        existing_plan = (
            config.budgets["openai"].plan
            if "openai" in config.budgets else None
        )
        plan_changed = False
        plan = existing_plan
        if existing_plan is None or reconfigure or plan_override:
            if plan_override:
                plan = plan_override
            else:
                plan = _prompt_plan("OpenAI / Codex", _OPENAI_PLAN_CHOICES, current=existing_plan)
            plan_changed = plan != existing_plan
            usd: float | None = plan_usd_override
            if plan == "api" and not plan_override:
                ceiling = click.prompt(
                    "Monthly OpenAI API spend ceiling in USD (0 = no limit)",
                    type=float, default=0.0, show_default=False,
                )
                if ceiling > 0:
                    usd = ceiling
            existing_budget = config.budgets.get("openai")
            if existing_budget is not None:
                existing_budget.plan = plan
                if usd is not None:
                    existing_budget.usd = usd
            else:
                config.budgets["openai"] = ProviderBudget(
                    usd=usd, cycle_start_day=1, plan=plan,
                )
        budget = _prompt_daily_budget(budget, plan)
        if budget and budget > 0:
            config.agents[agent_id].budget.daily_usd = budget
        _apply_analysis_span(config, analysis_span, reconfigure=reconfigure)
        write_config(config, config_path)
        console.print(f"  tj config updated: {display_path(config_path)}", soft_wrap=True)
    else:
        ingest_secret = secrets.token_hex(32)
        if plan_override:
            plan = plan_override
        else:
            plan = _prompt_plan("OpenAI / Codex", _OPENAI_PLAN_CHOICES)
        plan_changed = False
        usd: float | None = plan_usd_override  # type: ignore[no-redef]
        if plan == "api" and not plan_override:
            ceiling = click.prompt(
                "Monthly OpenAI API spend ceiling in USD (0 = no limit)",
                type=float, default=0.0, show_default=False,
            )
            if ceiling > 0:
                usd = ceiling
        budget = _prompt_daily_budget(budget, plan)
        daily_usd = budget if budget and budget > 0 else None
        agents = {agent_id: AgentConfig(budget=BudgetConfig(daily_usd=daily_usd))}
        config = TjConfig(
            version="1",
            agents=agents,
            security=SecurityConfig(ingest_secret=ingest_secret),
            budgets={"openai": ProviderBudget(
                usd=usd, cycle_start_day=1, plan=plan,
            )},
        )
        config_path.parent.mkdir(parents=True, exist_ok=True)
        _apply_analysis_span(config, analysis_span, reconfigure=reconfigure)
        write_config(config, config_path)
        console.print(
            f"[ok]\u2713[/ok] Config written to "
            f"[accent]{display_path(config_path)}[/accent]",
            soft_wrap=True,
        )
        if _sync_secret_to_claude_code(ingest_secret):
            console.print("  Claude Code config updated to match new ingest secret.")

    stopped_for_db = _stop_serve_for_db_write()
    if stopped_for_db:
        console.print(
            "[dim]  Server:              stopped briefly for "
            "config/DB update[/dim]"
        )

    apply_msg = _try_apply_declared_plans(
        config, reconcile=reconfigure and plan_changed,
    )
    if apply_msg:
        console.print(f"  {apply_msg}")

    port = config.api.port
    secret = config.security.ingest_secret
    _report_secret_alignment(secret)
    secret_rotated = bool(previous_secret) and previous_secret != secret
    want_daemon = not no_daemon

    # --- Write Codex CLI OTel config (~/.codex/config.toml) ---
    codex_config_path = Path.home() / ".codex" / "config.toml"
    codex_config_path.parent.mkdir(parents=True, exist_ok=True)

    existing_content = codex_config_path.read_text() if codex_config_path.exists() else ""

    # Purge any legacy `ocw`-managed sections left over from pre-rebrand
    # onboards. If anything was stripped, persist the cleaned file now so the
    # "already configured" early-return path below doesn't leave the legacy
    # sections sitting in the file forever.
    cleaned = _codex_purge_legacy_ocw(existing_content)
    if cleaned != existing_content:
        codex_config_path.write_text(cleaned)
        existing_content = cleaned

    # Check whether an [otel] section already exists
    existing_codex: dict = {}
    if existing_content:
        try:
            existing_codex = tomllib.loads(existing_content)
        except Exception:
            pass

    # Codex gets tj purely out-of-band: the OTel telemetry export (below) that
    # tj ingests with zero per-turn model cost. We deliberately do NOT register
    # the tj MCP server for Codex — an in-loop MCP is a per-turn token burden on
    # subscription power users (a measured A/B showed +36% model-weighted tokens
    # vs a no-tj control). Codex has no
    # statusline / status-hook surface to carry tj's re-read line the way Claude
    # Code does, so out-of-band OTel + `tj` CLI reports is the whole surface. The
    # MCP is reserved for SDK / API users where tj sits in the request path.
    already_has_otel = "otel" in existing_codex
    if already_has_otel and not force:
        # Use plain print() for messages containing TOML section headers like
        # [otel] — Rich treats square brackets as markup tags and would strip
        # them, leaving the message unintelligible ("already has ").
        click.echo(
            "~/.codex/config.toml already has an [otel] section."
        )
        click.echo("Use --force to overwrite, or add manually:")
        click.echo("")
        _print_codex_otel_block(port, secret, agent_id)
        # Retire a previously tj-registered MCP block so a re-onboard actually
        # stops the per-turn burden (only touches our own managed block).
        mcp_removed = _codex_retire_tj_mcp(codex_config_path)
        if mcp_removed:
            click.echo("")
            click.echo(
                "Removed the previously-registered [mcp_servers.tj] block "
                "(tj is out-of-band for Codex — see `tj` CLI reports)."
            )
        # Even on the "already configured" fast path, attempt a defensive Codex
        # backfill so re-onboarding picks up any newly-written on-disk logs (#448).
        # On the combination path the backfill is run once by the caller, so skip
        # it here to avoid double-running (#432).
        cx_has, cx_total = False, 0
        if standalone:
            cx_msg, cx_has, cx_total = _try_backfill_codex(config)
            if cx_msg:
                console.print(f"  Backfilled:          {cx_msg}")
        _finish_onboard_serve(
            str(config_path.resolve()),
            want_daemon=not no_daemon,
            plan_changed=plan_changed,
            stopped_for_db=stopped_for_db,
            secret_rotated=secret_rotated,
            no_daemon=no_daemon,
            force=force,
        )
        _warn_manual_serve_restart(stopped_for_db=stopped_for_db, no_daemon=no_daemon)
        if standalone:
            _print_setup_complete_home(
                sessions_backfilled=cx_total, has_data=cx_has,
            )
        return

    otel_block = _codex_otel_toml_block(port, secret, agent_id)

    # Build the new file content by replacing/appending the [otel] section only.
    base_content = existing_content
    if force:
        # Fully wipe previous OTEL sections so nested tables don't duplicate.
        base_content = _codex_strip_otel_sections(base_content)
    new_content = _codex_apply_block(
        base_content, r"\[otel\]", already_has_otel, otel_block, force,
    )
    # Retire any previously tj-registered MCP block — tj is out-of-band for Codex
    # now; leaving [mcp_servers.tj] would keep taxing every turn. Only strips our
    # own managed block, never a user-authored one.
    new_content, mcp_was_removed = _codex_strip_tj_mcp_from_content(new_content)
    codex_config_path.write_text(new_content)

    # --- Attempt a Codex backfill (#448) ---
    # The per-Codex flow is OTel/forward wiring; a separate workstream is adding
    # an on-disk Codex backfill adapter (`tj backfill codex`). Call it
    # defensively — if the adapter hasn't landed yet, we fall back to
    # forward-only framing rather than claiming a backfill that didn't happen
    # (honesty discipline, Rule 14). Run before daemon install so a freshly
    # started serve doesn't hold the DuckDB write lock (#71). On the combination
    # path the caller runs this backfill exactly once, so skip it here (#432).
    codex_backfill_msg: str | None = None
    codex_has_data, codex_sessions_total = False, 0
    if standalone:
        codex_backfill_msg, codex_has_data, codex_sessions_total = (
            _try_backfill_codex(config)
        )

    # --- Install / restart serve after DB writes ---
    _finish_onboard_serve(
        str(config_path.resolve()),
        want_daemon=not no_daemon,
        plan_changed=plan_changed,
        stopped_for_db=stopped_for_db,
        secret_rotated=secret_rotated,
        no_daemon=no_daemon,
        force=force,
    )

    console.print()
    console.print("[ok]\u2713 Codex CLI observability configured.[/ok]")
    console.print(
        f"  Codex config:        {display_path(codex_config_path)}", soft_wrap=True
    )
    console.print(
        f"  TokenJam config:     {display_path(config_path)}", soft_wrap=True
    )
    if budget and budget > 0:
        console.print(f"  Daily budget:        ${budget:.2f}")
    console.print(f"  OTLP endpoint:       http://127.0.0.1:{port}/v1/logs")
    if secret:
        console.print(f"  Ingest secret:       {secret[:8]}...")
    console.print("  Integration:         out-of-band (OTel telemetry — zero token cost)")
    if codex_backfill_msg:
        console.print(f"  Backfilled:          {codex_backfill_msg}")
    _print_capture_disclosure(config.capture.prompts, config.capture.tool_inputs)
    if mcp_was_removed:
        # Plain echo: Rich would treat [mcp_servers.tj] as a markup tag and strip it.
        click.echo(
            "  Removed [mcp_servers.tj] — tj is out-of-band for Codex now "
            "(no per-turn quota burden)."
        )
    # Lead with the wins that need no restart, THEN the restart note (#240).
    # `has_data` is True only when the Codex backfill actually ingested sessions
    # (adapter present + on-disk history); otherwise forward-only framing.
    _print_next_steps_nudge(
        has_data=codex_has_data, daemon_running=want_daemon, port=port,
    )
    if not want_daemon:
        _warn_manual_serve_restart(stopped_for_db=stopped_for_db, no_daemon=True)
        console.print("[dim]Start the server:[/dim]  tj serve")
    # Codex has no statusline / status-hook surface (as of Codex CLI today), so
    # tj can't put its re-read line inline the way it does for Claude Code. tj
    # stays fully out-of-band: it ingests Codex's OTel telemetry, and you read
    # it with the `tj` CLI. No in-loop MCP burden.
    if not codex_has_data:
        # Forward-only: no on-disk history was ingested (adapter not shipped yet,
        # or no Codex logs) — say so honestly rather than implying past data.
        console.print(
            "[dim]Wired — Codex telemetry flows to tj as you use Codex.[/dim]"
        )
    console.print(
        "[dim]Codex has no statusline surface, so tj stays out-of-band: your "
        "Codex telemetry flows to tj automatically. Run[/dim]  tj tokenmaxx  "
        "[dim]/[/dim]  tj traces  [dim]for the deep dive.[/dim]"
    )
    console.print(
        "[dim]Codex gets a smaller subset of tj than Claude Code (no statusline, "
        "no per-terminal split) — see[/dim] "
        "docs/agent-capability-matrix.md [dim]for the full breakdown.[/dim]"
    )
    _print_restart_banner("Codex")
    console.print("[dim]After restarting, run:[/dim]  tj traces")

    _maybe_verify_onboarding(config, persona="codex", verify=verify)

    # Shared closing banner (#448). On the combination path this is deferred to
    # `_onboard_combination` so the banner prints exactly once (#432).
    if standalone:
        _print_setup_complete_home(
            sessions_backfilled=codex_sessions_total,
            has_data=codex_has_data,
        )


def _global_provider_plan(provider_key: str) -> str | None:
    """Return the plan tier already stored for ``provider_key`` (``anthropic`` /
    ``openai``) in the global tj config, or None when there's no config or no
    plan for that provider yet.

    The combination flow uses this to decide whether a leg would prompt for its
    plan at all — a leg with an already-stored plan keeps it without asking. So
    the flow only hoists the billing question up front when the leg would
    actually ask it (a fresh config, or an explicit ``--plan``), leaving an
    existing-config re-run byte-for-byte unchanged.
    """
    path = Path.home() / ".config" / "tj" / "config.toml"
    if not path.exists():
        return None
    try:
        from tokenjam.core.config import load_config as _lc
        cfg = _lc(str(path))
        pb = cfg.budgets.get(provider_key)
        return pb.plan if pb else None
    except Exception:
        return None


def _collect_combination_billing(
    provider_label: str,
    choices: list[tuple[str, str]],
    ceiling_prompt: str,
    plan_override: str | None,
    budget: float | None,
) -> tuple[str, float | None, float]:
    """Collect one leg's billing answers up front for the combination flow.

    Returns ``(plan, usd_ceiling, daily_budget)`` — the same three values a leg
    resolves inline — so a multi-surface run can ask every billing question
    before any leg's long-running backfill runs, then thread the answers back
    into the leg (via ``plan_override`` / ``plan_usd_override`` / ``budget``) so
    the leg re-asks nothing. Mirrors the legs' own logic exactly: ``--plan``
    short-circuits the tier prompt, and the API spend-ceiling sub-prompt only
    fires for the interactive ``api`` tier.
    """
    if plan_override:
        plan = plan_override
    else:
        plan = _prompt_plan(provider_label, choices)
    usd: float | None = None
    if plan == "api" and not plan_override:
        ceiling = click.prompt(
            ceiling_prompt, type=float, default=0.0, show_default=False,
        )
        if ceiling > 0:
            usd = ceiling
    daily_budget = _prompt_daily_budget(budget, plan)
    return plan, usd, daily_budget


def _onboard_combination(
    ctx: click.Context,
    budget: float | None,
    no_daemon: bool,
    force: bool,
    plan_override: str | None = None,
    verify: bool = False,
    backfill_days: int | None = None,
    backfill_all: bool = False,
    analysis_span: str | None = None,
) -> None:
    """The "combination" path (#448): the user runs more than one kind of agent.

    Asks which surfaces they use (Claude Code, Codex, custom SDK/API agents),
    runs the necessary backfills for the coding-agent surfaces, reports what was
    done, then shows the SDK instrumentation snippet for their custom agents —
    one coherent sequence ending in the shared home banner.

    Wiring is delegated to the existing per-path onboarders so config/statusline/
    OTel setup stays DRY and correct; this function only orchestrates the
    ordering and the closing summary.
    """
    console.print()
    console.print("[bold]Which do you use?[/bold] [dim](answer each)[/dim]")
    uses_cc = click.confirm("  Claude Code", default=True)
    uses_codex = click.confirm("  Codex", default=False)
    uses_sdk = click.confirm(
        "  Your own agents (Python/TS SDK or API)", default=True
    )

    if not (uses_cc or uses_codex or uses_sdk):
        console.print(
            "[warn]Nothing selected.[/warn] Re-run [bold]tj onboard[/bold] "
            "and pick at least one."
        )
        ctx.exit(1)
        return

    # --- Ask every billing question up front, before any long-running work ---
    # The old shape ran each leg to completion in turn: the Claude leg asked its
    # plan, then ran its (slow) backfill, and only THEN did the Codex leg ask
    # its plan — so a combination run interleaved a question with minutes of
    # ingest. We hoist each selected leg's billing (plan tier → API ceiling →
    # daily budget) to the top, Claude first then Codex, then run the legs with
    # the answers threaded in so neither leg re-prompts. Project name and
    # backfill scope stay inside the Claude leg — they already run there before
    # its backfill, which now precedes the (prompt-free) Codex leg, so no
    # cross-leg interleaving remains: Claude billing → Codex billing → project →
    # backfill scope → execution. A leg whose plan is already stored (an
    # existing-config re-run, no --plan) keeps it without asking, exactly as the
    # single-persona flow does — so we only hoist a leg's billing when the leg
    # would actually prompt.
    cc_plan, cc_usd, cc_budget = plan_override, None, budget
    if uses_cc and (_global_provider_plan("anthropic") is None or plan_override):
        cc_plan, cc_usd, cc_budget = _collect_combination_billing(
            "Claude", _ANTHROPIC_PLAN_CHOICES,
            "Monthly Anthropic API spend ceiling in USD (0 = no limit)",
            plan_override, budget,
        )
    codex_plan, codex_usd, codex_budget = plan_override, None, budget
    if uses_codex and (_global_provider_plan("openai") is None or plan_override):
        codex_plan, codex_usd, codex_budget = _collect_combination_billing(
            "OpenAI / Codex", _OPENAI_PLAN_CHOICES,
            "Monthly OpenAI API spend ceiling in USD (0 = no limit)",
            plan_override, budget,
        )

    done: list[str] = []

    # --- Claude Code (full flow: config + statusline + auto-backfill) ---
    if uses_cc:
        console.print()
        console.print("[bold]Setting up Claude Code…[/bold]")
        _onboard_claude_code(
            ctx, cc_budget, no_daemon, force, reconfigure=False,
            plan_override=cc_plan, plan_usd_override=cc_usd,
            verify=False, standalone=False,
            backfill_days=backfill_days, backfill_all=backfill_all,
            analysis_span=analysis_span,
        )
        done.append("Claude Code")

    # --- Codex (full flow: OTel wiring) then a defensive backfill ---
    if uses_codex:
        console.print()
        console.print("[bold]Setting up Codex…[/bold]")
        _onboard_codex(
            ctx, codex_budget, no_daemon, force, reconfigure=False,
            plan_override=codex_plan, plan_usd_override=codex_usd,
            verify=False, standalone=False, analysis_span=analysis_span,
        )
        # Run the on-disk Codex backfill exactly once here. Passing
        # standalone=False above suppressed the per-path onboarder's own backfill
        # so it doesn't double-run (#432).
        try:
            from tokenjam.core.config import load_config as _lc
            global_path = Path.home() / ".config" / "tj" / "config.toml"
            if global_path.exists():
                codex_cfg = _lc(str(global_path))
                cx_msg, _cx_has, _cx_n = _try_backfill_codex(codex_cfg)
                if cx_msg:
                    console.print(f"  Codex backfilled:    {cx_msg}")
        except Exception:
            pass
        done.append("Codex")

    # --- Custom SDK / API agents: show the instrument snippet ---
    if uses_sdk:
        console.print()
        console.print("[bold]Instrument your own agents:[/bold]")
        console.print()
        _print_instrument_agent_snippet()
        console.print()
        console.print(
            "  Then run your agent and verify a live span is flowing with "
            "[bold]tj ping[/bold] (in another terminal)."
        )
        done.append("SDK/API")

    console.print()
    console.print(
        f"[ok]\u2713 Combination setup complete[/ok] "
        f"[dim]({', '.join(done)})[/dim]"
    )
    _print_setup_complete_home()


def _print_restart_banner(app_name: str) -> None:
    """Render a prominent restart-required banner at the end of onboarding.

    Coding agents (Claude Code, Codex) read their OTLP exporter env vars once
    at startup, not per request. After onboard rewrites the endpoint/ingest
    secret, an already-running instance keeps exporting to the stale endpoint
    and today's spans silently never reach TokenJam (issue #179). A single dim
    one-liner was too easy to miss, so make this a Rich panel.

    Generic variant used by runtimes without session-resume guidance to fold in
    (currently: Codex). Claude Code uses the more detailed, why-first
    ``_print_claude_code_restart_panel`` instead, which also covers resume
    semantics and the verify step.
    """
    from rich.panel import Panel
    from rich.text import Text

    body = Text()
    body.append("Restart ", style="bold")
    body.append(app_name, style="warn.strong")
    body.append(" now for the new settings to take effect.\n", style="bold")
    body.append(
        f"A {app_name} session already running will keep sending telemetry to "
        "the old endpoint — today's activity won't reach TokenJam until you "
        "restart it.",
        style="dim",
    )
    console.print(
        Panel(
            body,
            title="[warn.strong]Action required[/warn.strong]",
            border_style="warn",
            padding=(1, 2),
        )
    )


def _claude_code_is_running() -> bool:
    """True when at least one ``claude`` process appears to be running (#643).

    The "restart Claude Code" block only matters for sessions that are ALREADY
    open — they keep exporting to the pre-onboard endpoint until relaunched. A
    fresh terminal with nothing open needs no restart, so we suppress the block
    there. Detected via ``pgrep -f claude``.

    Fail-OPEN: if ``pgrep`` is missing (not on PATH), errors, or times out, we
    can't prove there are no running sessions, so we return True and show the
    block rather than risk silently dropping a needed restart instruction.
    """
    import shutil
    import subprocess

    if shutil.which("pgrep") is None:
        return True
    try:
        result = subprocess.run(
            ["pgrep", "-f", "claude"],
            capture_output=True, text=True, timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return True
    # pgrep exits 0 when a match is found, 1 when none, >1 on error. Treat any
    # non-"clean no-match" outcome as "can't be sure" → show the block.
    if result.returncode == 1:
        return False
    return True


def _print_claude_code_restart_panel() -> None:
    """One optional line for the Claude Code restart (#643; demo trim 2026-07-30).

    Restarting is NOT required for completeness: the daemon's transcript
    catch-up ingests running sessions from disk on its interval regardless
    (the ``[ingest]`` auto_catch_up path). A restart only makes today's
    currently-active sessions stream in live immediately, so this is stated as
    optional, not an action gate.
    """
    console.print(
        "[muted]Optional: restart Claude Code to stream today's active sessions "
        "live -- otherwise tj picks them up from disk within a few minutes.[/muted]"
    )


def _print_background_items_headsup() -> None:
    """The macOS "Background Items Added" heads-up (#675).

    Only macOS surfaces this OS-level notification (launchd LaunchAgents), so
    the line is gated on Darwin. Printed at the very bottom of the completion
    screen — after Next steps — rather than mid daemon-install, so the payoff
    screen reads top-to-bottom.
    """
    if platform.system() != "Darwin":
        return
    console.print(
        "[warn]Heads up:[/warn] macOS will show a 'Background Items Added' "
        "notification -- this is normal."
    )


def _warn_manual_serve_restart(*, stopped_for_db: bool, no_daemon: bool) -> None:
    """Tell the user to restart serve when onboard stopped it without daemon mode."""
    if stopped_for_db and no_daemon:
        console.print(
            "[dim]  Serve was stopped for DB update — "
            "run `tj serve` to start again.[/dim]"
        )


def _finish_onboard_serve(
    config_path: str,
    *,
    want_daemon: bool,
    plan_changed: bool,
    stopped_for_db: bool,
    secret_rotated: bool,
    no_daemon: bool,
    force: bool,
    quiet_daemon: bool = False,
) -> str | None:
    """Install or restart ``tj serve`` after onboard DB/config writes.

    ``quiet_daemon`` (the --claude-code completion, #675) suppresses the
    internal "Daemon: installing.../already running" and "Server restart:"
    status lines — the completion screen reports success via three `✓` lines
    instead — and withholds the macOS "Background Items Added" heads-up so it
    can print at the bottom of the payoff screen.
    """
    if not config_path or not Path(config_path).exists():
        return None

    restart_msg: str | None = None
    need_restart = secret_rotated or plan_changed or stopped_for_db

    if want_daemon and not (need_restart and not force):
        if (
            not force
            and _daemon_already_running()
            and not stopped_for_db
            and not need_restart
        ):
            if not quiet_daemon:
                console.print(
                    "  Daemon:              already running (skipped reinstall)"
                )
            restart_msg = "daemon already running"
        else:
            if not quiet_daemon:
                console.print("  Daemon:              installing...")
            install_msg = _install_daemon(config_path, suppress_headsup=quiet_daemon)
            if install_msg:
                restart_msg = install_msg

    if want_daemon and need_restart and not force:
        if secret_rotated:
            reason = "secret"
        elif plan_changed:
            reason = "plan"
        else:
            reason = "db_update"
        restart_msg = _restart_tj_server(
            config_path, no_daemon, reason=reason, suppress_headsup=quiet_daemon,
        )
        if not quiet_daemon:
            console.print(f"  Server restart:      {restart_msg}")

    return restart_msg


def _stop_serve_for_db_write() -> bool:
    """Stop a running tj serve so onboard can write DuckDB. Returns True if stopped."""
    from tokenjam.cli.cmd_stop import stop_tj_serve

    stopped, _ = stop_tj_serve(quiet=True)
    return stopped


def _try_apply_declared_plans(config, *, reconcile: bool = False) -> str | None:
    """Stamp sessions from ``[budget.<provider>].plan``. Best-effort.

    When *reconcile* is True (plan changed on reconfigure), all sessions for
    each provider are updated to match config — not only ``unknown`` rows.
    """
    from tokenjam.core.db import open_db
    from tokenjam.core.framing import apply_declared_plans_to_sessions

    try:
        db = open_db(config.storage)
        try:
            n = apply_declared_plans_to_sessions(
                db.conn, config, reconcile=reconcile,
            )
            if n:
                return f"Plan tier applied to {n} session(s)."
        finally:
            db.close()
    except Exception as exc:
        err = str(exc).lower()
        if "lock" in err or "i/o error" in err or "io error" in err:
            return (
                "Plan tier backfill deferred — serve holds the DB write lock. "
                "Plans apply on next `tj serve` startup, or stop serve "
                "(`tj stop`) and re-run onboard."
            )
    return None


def _restart_tj_server(
    config_path: str,
    no_daemon: bool,
    *,
    reason: str = "secret",
    suppress_headsup: bool = False,
) -> str:
    """Restart running tj serve to pick up config changes.

    *reason* selects the user-facing message: ``secret``, ``plan``, or
    ``db_update``.
    """
    stopped = False
    try:
        from tokenjam.cli.cmd_stop import stop_tj_serve
        stopped, _ = stop_tj_serve(quiet=True)
    except Exception:
        stopped = False

    if no_daemon:
        if reason == "secret":
            hint = "run `tj serve` to start with new secret"
        elif reason == "plan":
            hint = "run `tj serve` to pick up plan tier change"
        else:
            hint = "run `tj serve` to pick up config changes"
        if stopped:
            return f"stopped stale server; {hint}"
        return f"could not auto-restart in --no-daemon mode; {hint}"

    daemon_msg = _install_daemon(config_path, suppress_headsup=suppress_headsup)
    if daemon_msg:
        if reason == "secret":
            return "restarted to pick up new ingest secret"
        if reason == "plan":
            return "restarted to pick up plan tier change"
        return "restarted to pick up config changes"
    if stopped:
        return "stopped stale server; please run `tj serve` manually"
    return "restart attempted; verify with `tj status`"


def _codex_strip_tj_mcp_from_content(content: str) -> tuple[str, bool]:
    """Remove a tj-owned [mcp_servers.tj] section from Codex config *content*.

    Only strips the block when it is unmistakably tj's own — the "Managed by tj"
    marker or a ``command = "tj"`` line — so a user's unrelated same-named server
    is never touched. Returns ``(new_content, removed)``.
    """
    import re as _re

    m = _re.search(
        r"\[mcp_servers\.tj\].*?(?=\n\[|\Z)",
        content,
        flags=_re.DOTALL,
    )
    if not m:
        return content, False
    block = m.group(0)
    is_tj = ("Managed by tj" in block) or bool(
        _re.search(r'command\s*=\s*"tj"', block)
    )
    if not is_tj:
        return content, False
    stripped = content[: m.start()] + content[m.end():]
    # Collapse the blank-line run left behind by the removal.
    stripped = _re.sub(r"\n{3,}", "\n\n", stripped)
    return stripped.strip() + ("\n" if stripped.strip() else ""), True


def _codex_retire_tj_mcp(codex_config_path: Path) -> bool:
    """Strip a tj-owned [mcp_servers.tj] block from the Codex config file.

    Reads, strips, and rewrites the file only when a tj-owned block is present.
    Returns True if a block was removed. Best-effort and fail-safe.
    """
    try:
        if not codex_config_path.exists():
            return False
        content = codex_config_path.read_text()
        new_content, removed = _codex_strip_tj_mcp_from_content(content)
        if removed:
            codex_config_path.write_text(new_content)
        return removed
    except Exception:
        return False


def _codex_apply_block(
    content: str,
    section_pattern: str,
    section_exists: bool,
    block: str,
    force: bool,
) -> str:
    """Replace or append a TOML section identified by *section_pattern* (regex).

    If the section is absent, the block is appended.
    If the section exists and *force* is True, it is replaced.
    If the section exists and *force* is False, the content is unchanged.
    """
    import re as _re

    if section_exists:
        if not force:
            return content
        # Replace from the section header to the next section header or EOF.
        stripped = _re.sub(
            section_pattern + r".*?(?=\n\[|\Z)",
            "",
            content,
            flags=_re.DOTALL,
        ).rstrip()
        return (stripped + "\n\n" + block) if stripped else block
    # Section absent — append.
    return (content.rstrip() + "\n\n" + block) if content.strip() else block


def _codex_purge_legacy_ocw(content: str) -> str:
    """Remove ocw-managed sections from ~/.codex/config.toml left behind by
    pre-rebrand onboards.

    Before the project was renamed, the legacy `ocw` CLI wrote a
    `[mcp_servers.ocw]` block (pointing `command = "ocw"`) and an `[otel]`
    block with a `# Managed by ocw` comment. After the rebrand, the new
    onboard appends the `[mcp_servers.tj]` block but does not touch the
    legacy `ocw` sections — so Codex ends up with both registered, and
    tries to spawn a non-existent `ocw` MCP server on every launch.

    This unconditionally strips the legacy sections so the normal onboard
    flow can write fresh tj-managed blocks in their place.
    """
    import re as _re

    # Drop the entire [mcp_servers.ocw] section (and any nested tables).
    content = _re.sub(
        r"\[mcp_servers\.ocw\].*?(?=\n\[|\Z)",
        "",
        content,
        flags=_re.DOTALL,
    )

    # If the existing [otel] block is marked "Managed by ocw", strip the
    # whole [otel] tree so the new tj-managed block is written cleanly.
    # We use a non-anchored search across the whole content because the
    # comment may sit a few lines below the section header.
    has_legacy_otel = bool(
        _re.search(
            r"\[otel\][^\[]*?#\s*Managed by ocw",
            content,
            flags=_re.IGNORECASE,
        )
    )
    if has_legacy_otel:
        for pat in (
            r"\[otel\.exporter\.\"otlp-http\"\.headers\].*?(?=\n\[|\Z)",
            r"\[otel\.exporter\.\"otlp-http\"\].*?(?=\n\[|\Z)",
            r"\[otel\.exporter\.\"otlp-grpc\"\.headers\].*?(?=\n\[|\Z)",
            r"\[otel\.exporter\.\"otlp-grpc\"\].*?(?=\n\[|\Z)",
            r"\[otel\.resource\].*?(?=\n\[|\Z)",
            r"\[otel\].*?(?=\n\[|\Z)",
        ):
            content = _re.sub(pat, "", content, flags=_re.DOTALL)

    # Collapse runs of 3+ blank lines left behind by removals.
    content = _re.sub(r"\n{3,}", "\n\n", content)
    return content.strip() + ("\n" if content.strip() else "")


def _codex_strip_otel_sections(content: str) -> str:
    """Remove all [otel] sections (including nested exporter/resource tables)."""
    import re as _re

    patterns = (
        r"\[otel\].*?(?=\n\[|\Z)",
        r"\[otel\.resource\].*?(?=\n\[|\Z)",
        r"\[otel\.exporter\.\"otlp-http\"\].*?(?=\n\[|\Z)",
        r"\[otel\.exporter\.\"otlp-http\"\.headers\].*?(?=\n\[|\Z)",
        r"\[otel\.exporter\.\"otlp-grpc\"\].*?(?=\n\[|\Z)",
        r"\[otel\.exporter\.\"otlp-grpc\"\.headers\].*?(?=\n\[|\Z)",
    )
    stripped = content
    for pat in patterns:
        stripped = _re.sub(pat, "", stripped, flags=_re.DOTALL)
    return stripped.strip()


def _codex_otel_toml_block(port: int, secret: str, agent_id: str) -> str:
    """Return the [otel] TOML block to append to ~/.codex/config.toml."""
    endpoint = f"http://127.0.0.1:{port}/v1/logs"
    # Note: Codex CLI hardcodes service.name="codex_exec" in the binary and
    # ignores [otel.resource] entirely, so we don't write a resource block.
    return (
        f'[otel]\n'
        f'# Managed by tj — do not edit this block manually\n'
        f'log_user_prompt = false\n'
        f'\n'
        f'[otel.exporter."otlp-http"]\n'
        f'endpoint = "{endpoint}"\n'
        f'protocol = "json"\n'
        f'\n'
        f'[otel.exporter."otlp-http".headers]\n'
        f'Authorization = "Bearer {secret}"\n'
    )


def _print_codex_otel_block(port: int, secret: str, agent_id: str = "codex_exec") -> None:
    # Plain print — the TOML block contains [otel] and other section headers
    # that Rich would interpret as markup tags and strip from the output.
    click.echo("Add this to ~/.codex/config.toml:")
    click.echo("")
    block = _codex_otel_toml_block(port, secret, agent_id)
    for line in block.splitlines():
        click.echo(f"  {line}")
    click.echo("")


def _sync_secret_to_codex(secret: str) -> bool:
    """Update Authorization header in ~/.codex/config.toml if an [otel] section exists."""
    import re as _re
    codex_path = Path.home() / ".codex" / "config.toml"
    if not codex_path.exists():
        return False
    content = codex_path.read_text()
    if "Authorization" not in content:
        return False
    updated = _re.sub(
        r'(Authorization\s*=\s*"Bearer\s+)[^"]+(")',
        rf'\g<1>{secret}\g<2>',
        content,
    )
    if updated == content:
        return False
    codex_path.write_text(updated)
    return True


def _sync_secret_to_claude_code(secret: str) -> bool:
    """Update OTLP Authorization header in ~/.claude/settings.json if present."""
    settings_path = Path.home() / ".claude" / "settings.json"
    if not settings_path.exists():
        return False
    try:
        settings = json_mod.loads(settings_path.read_text())
    except (json_mod.JSONDecodeError, OSError):
        return False
    env = settings.get("env", {})
    if "Authorization=Bearer" not in env.get("OTEL_EXPORTER_OTLP_HEADERS", ""):
        return False
    env["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Bearer {secret}"
    settings["env"] = env
    settings_path.write_text(json_mod.dumps(settings, indent=2) + "\n")
    return True


_WRAPPER_MARKER = "# tj per-terminal naming"
_WRAPPER_END_MARKER = "# end tj per-terminal naming"


def _claude_wrapper_block() -> str:
    """Return the idempotent ``claude`` shell-wrapper block.

    The wrapper tags each terminal with a distinct ``service.instance.id`` so
    concurrent Claude Code sessions show as separate dashboard tiles. It:

    - consumes an optional ``--as <name>`` flag and passes the rest through,
    - derives the instance id from ``--as`` else the tty basename else
      ``unknown``,
    - exports ``OTEL_RESOURCE_ATTRIBUTES`` (project attrs from
      ``tj otel-resource-attrs`` + the instance id),
    - runs the real binary via ``command claude`` so it never recurses,
    - reports the session closed (``tj session-end``) when claude exits or is
      interrupted, so the dashboard archives the tile (Claude Code emits no
      close event of its own). Best-effort and idempotent.

    Invokes ``tj`` by absolute path (like ``_tj_statusline_command`` already
    does), not the bare command: this block runs whenever the user
    later types ``claude`` in SOME terminal, which may have a different PATH
    than the one onboard itself ran in (e.g. VS Code's integrated terminal
    vs. Terminal.app) — an absolute path can't be shadowed by an older `tj`
    earlier on that terminal's PATH, or fail outright if `tj`'s directory
    isn't on PATH at all.

    Every ``tj`` invocation here is stderr-silenced (``2>/dev/null``) and
    failure-tolerant: a stale/shadowed/half-uninstalled ``tj`` ahead on PATH
    must never flash a traceback into an interactive ``claude`` launch, and a
    failed or empty ``otel-resource-attrs`` lookup falls back to just
    ``service.instance.id=...`` rather than a malformed value with a leading
    comma.

    Written portably so it works in both zsh and bash. Re-running
    ``_install_claude_wrapper`` always regenerates this block fresh and
    replaces the prior one in place (matched by ``_WRAPPER_MARKER`` ..
    ``_WRAPPER_END_MARKER``, content-agnostic) — so a content-only fix like
    this one reaches existing users automatically on their next ``tj
    onboard``, with no marker version bump needed.
    """
    tj_bin = _current_tj_binary()
    return (
        f"{_WRAPPER_MARKER}\n"
        f"# Tags each terminal with a distinct service.instance.id so concurrent\n"
        f"# Claude Code sessions appear as separate TokenJam dashboard tiles.\n"
        f"# Override the label with: claude --as <name>\n"
        f"claude() {{\n"
        f'  local _tj_as=""\n'
        f"  local -a _tj_args=()\n"
        f'  while [ "$#" -gt 0 ]; do\n'
        f'    case "$1" in\n'
        f'      --as) _tj_as="$2"; shift 2 ;;\n'
        f'      --as=*) _tj_as="${{1#--as=}}"; shift ;;\n'
        f'      *) _tj_args+=("$1"); shift ;;\n'
        f"    esac\n"
        f"  done\n"
        f'  local _tj_inst="$_tj_as"\n'
        f'  if [ -z "$_tj_inst" ]; then\n'
        f'    local _tj_tty\n'
        f'    _tj_tty="$(tty 2>/dev/null)"\n'
        f'    case "$_tj_tty" in\n'
        f'      /dev/*) _tj_inst="${{_tj_tty#/dev/}}"; _tj_inst="${{_tj_inst//\\//-}}" ;;\n'
        f"    esac\n"
        f"  fi\n"
        f'  [ -z "$_tj_inst" ] && _tj_inst="unknown"\n'
        f"  # Silence stderr and tolerate failure: a stale/shadowed tj on PATH\n"
        f"  # must never leak a traceback into every claude launch, and an empty\n"
        f"  # or failing lookup must not produce a malformed leading-comma value.\n"
        f'  local _tj_attrs\n'
        f'  _tj_attrs="$("{tj_bin}" otel-resource-attrs 2>/dev/null)" || _tj_attrs=""\n'
        f'  if [ -n "$_tj_attrs" ]; then\n'
        f'    export OTEL_RESOURCE_ATTRIBUTES="$_tj_attrs,service.instance.id=$_tj_inst"\n'
        f"  else\n"
        f'    export OTEL_RESOURCE_ATTRIBUTES="service.instance.id=$_tj_inst"\n'
        f"  fi\n"
        f"  # Report this terminal's session closed on exit/interrupt so the\n"
        f"  # dashboard archives its tile. Idempotent — double-fire is harmless.\n"
        f"  trap '\"{tj_bin}\" session-end --instance \"$_tj_inst\" >/dev/null 2>&1 || true' INT TERM HUP\n"
        f'  command claude "${{_tj_args[@]}}"\n'
        f"  local _tj_status=$?\n"
        f"  trap - INT TERM HUP\n"
        f'  "{tj_bin}" session-end --instance "$_tj_inst" >/dev/null 2>&1 || true\n'
        f"  return $_tj_status\n"
        f"}}\n"
        f"{_WRAPPER_END_MARKER}\n"
    )


def _install_claude_wrapper() -> list[str]:
    """Install the idempotent ``claude`` wrapper into the user's shell rc files.

    Always writes ``~/.zshrc`` (created if absent); also writes ``~/.bashrc``
    only when it already exists. Re-running replaces the existing block in place
    (matched between the begin/end markers) so onboards never duplicate it.

    Returns the list of rc files that were written.
    """
    import re as _re

    block = _claude_wrapper_block()
    written: list[str] = []

    zshrc = Path.home() / ".zshrc"
    zshrc.touch(exist_ok=True)
    targets = [zshrc]
    bashrc = Path.home() / ".bashrc"
    if bashrc.exists():
        targets.append(bashrc)

    for rc in targets:
        text = rc.read_text()
        if _WRAPPER_MARKER not in text:
            with rc.open("a") as f:
                # Ensure a blank line before the block for readability.
                f.write(("" if text.endswith("\n") or not text else "\n") + "\n" + block)
        else:
            # Replace the existing block (begin marker .. end marker) in place.
            updated = _re.sub(
                _re.escape(_WRAPPER_MARKER) + r".*?" + _re.escape(_WRAPPER_END_MARKER) + r"\n",
                block,
                text,
                flags=_re.DOTALL,
            )
            rc.write_text(updated)
        written.append(str(rc))

    return written


def _unwire_claude_wrapper() -> list[str]:
    """Remove the ``claude`` shell-wrapper block from the user's shell rc files.

    The counterpart to ``_install_claude_wrapper()`` — strips the exact
    ``_WRAPPER_MARKER`` .. ``_WRAPPER_END_MARKER`` block (marker-delimited,
    not a fragile regex on the body, so it can't accidentally eat unrelated
    rc content) from ``~/.zshrc`` and ``~/.bashrc`` (whichever exist).
    Idempotent — a no-op when the block is absent. Used by ``tj uninstall``
    (#117: uninstall previously left this wrapper behind, breaking every
    subsequent ``claude`` launch once the tj package was removed).

    The trailing newline after the end marker is optional (``(?:\n|$)``) —
    requiring a hard ``\n`` silently no-opped on a block that happens to be
    the last line of a file with no final newline.

    Returns the list of rc files that were actually modified.
    """
    import re as _re

    removed: list[str] = []
    for rc in (Path.home() / ".zshrc", Path.home() / ".bashrc"):
        if not rc.exists():
            continue
        text = rc.read_text()
        if _WRAPPER_MARKER not in text:
            continue
        updated = _re.sub(
            _re.escape(_WRAPPER_MARKER) + r".*?" + _re.escape(_WRAPPER_END_MARKER) + r"(?:\n|$)",
            "",
            text,
            flags=_re.DOTALL,
        )
        if updated != text:
            rc.write_text(updated)
            removed.append(str(rc))

    return removed


def _derive_project_name() -> str:
    """
    Derive a meaningful project name for the agent ID.
    Priority: git remote origin repo name > current folder name.
    """
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            # Extract repo name from URL — handles both https and ssh forms
            # e.g. https://github.com/org/my-repo.git  -> my-repo
            #      git@github.com:org/my-repo.git       -> my-repo
            name = url.rstrip("/").split("/")[-1].split(":")[-1]
            name = name.removesuffix(".git").lower()
            if name:
                return name
    except Exception:
        pass
    return Path.cwd().name.lower()


def _onboard_add_project(ctx: click.Context) -> None:
    """Register the current repo's Claude Code agent under a project
    namespace, without running any of the rest of the onboarding wizard.

    The dashboard groups sessions by `service.namespace`, which comes from
    `config.agents[<agent_id>].project`. Registering the Nth repo previously
    meant re-running the full `--claude-code` flow: plan prompt, budget
    prompt, backfill-scope prompt, a full backfill pass over ALL of
    `~/.claude/projects`, a relearn scan, and a daemon reinstall — none of
    which is needed just to set one key. This writes only that key.

    Resolves the config to write via `load_config()` — the same TJ_CONFIG /
    project-local / global search order every other `tj` invocation uses
    (`config.config_path`, set by `load_config` off of that same
    resolution). This is deliberate: writing into whichever config
    `tj otel-resource-attrs` (and hence the `claude` wrapper) will actually
    load from this directory is what makes the mapping take effect, rather
    than always targeting the default global path regardless of an active
    override.

    Requires an existing config (from a prior `tj onboard` run, anywhere) —
    there is no ingest secret or default plan to invent here, so a missing
    config is a hard error pointing at plain `tj onboard` rather than a
    silently half-configured agent.
    """
    from tokenjam.core.config import AgentConfig, load_config, write_config

    config = load_config()
    config_path = config.config_path
    if config_path is None:
        console.print(
            "[error]No tj config found — nothing to add a project to.[/error]"
        )
        console.print(
            "Run [bold]tj onboard[/bold] first to set up Claude Code "
            "telemetry, then use [bold]tj onboard --add-project[/bold] from "
            "any other repo to register it."
        )
        ctx.exit(1)

    project_name = _derive_project_name()
    agent_id = f"claude-code-{project_name}"

    is_new_agent = agent_id not in config.agents
    if is_new_agent:
        config.agents[agent_id] = AgentConfig()

    # Only set it for a genuinely NEW agent — an existing one may carry a
    # hand-entered name from before --project was removed; respect it rather
    # than silently overwriting (would orphan its dashboard history).
    if is_new_agent or not config.agents[agent_id].project:
        config.agents[agent_id].project = project_name
    namespace = config.agents[agent_id].project
    # Not a fresh onboard, so no span question — but the clamp still runs,
    # so a config hand-edited to a shorter retention is corrected here too.
    _apply_analysis_span(config)
    write_config(config, config_path)

    console.print(
        f"[bold]{agent_id}[/bold] registered under project "
        f"[bold]{namespace}[/bold] in {display_path(config_path)}."
    )
    console.print(
        "New [bold]claude[/bold] launches in this directory will pick this "
        "up immediately."
    )
    if _daemon_already_running():
        console.print(
            "[dim]Note: the running tj daemon reads this mapping only as a "
            "server-side fallback for spans that arrive without "
            "service.namespace already set — it won't pick up this change "
            "until it restarts. New `claude` launches are unaffected since "
            "they read the config fresh at launch.[/dim]"
        )


def _daemon_already_running() -> bool:
    """Check if the TokenJam daemon is already installed and loaded."""
    system = platform.system()
    if system == "Darwin":
        plist = Path.home() / "Library/LaunchAgents/com.tokenjam.serve.plist"
        if not plist.exists():
            return False
        result = subprocess.run(
            ["launchctl", "list", "com.tokenjam.serve"],
            capture_output=True, text=True,
        )
        return result.returncode == 0
    elif system == "Linux":
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "tokenjam"],
            capture_output=True, text=True,
        )
        return result.stdout.strip() == "active"
    return False


def _install_daemon(config_path: str, *, suppress_headsup: bool = False) -> str | None:
    """Install background daemon. Returns success message or None.

    ``suppress_headsup`` (macOS only) withholds the inline "Background Items
    Added" heads-up so the --claude-code completion can print it at the bottom
    of the payoff screen instead (#675).
    """
    system = platform.system()
    try:
        if system == "Darwin":
            return _install_launchd(config_path, suppress_headsup=suppress_headsup)
        elif system == "Linux":
            return _install_systemd(config_path)
        else:
            console.print(f"[warn]Background daemon not supported on {system}. "
                          "Run `tj serve` manually.[/warn]")
            return None
    except Exception as e:
        console.print(f"[warn]Daemon installation failed: {e}[/warn]")
        console.print("[dim]You can run `tj serve` manually instead.[/dim]")
        return None


def _resolve_tj_binary() -> str:
    """Resolve the absolute path to the ``tj`` executable for daemon units.

    Prefer the sibling ``tj`` next to the running interpreter — derived by
    path, not string substitution — the same PATH-independent priority
    ``_current_tj_binary`` uses: it's fixed at process start and reflects the
    binary onboard is actually running as, so it can't be shadowed by an
    older/other ``tj`` earlier on PATH at the moment onboard installs the
    daemon. Without this, a PATH shadow at install time permanently pins the
    launchd/systemd unit to the wrong binary — surviving even after the
    shadowing PATH entry is later removed.

    Falls back to ``shutil.which("tj")`` only when no sibling exists (e.g. a
    venv whose bin dir isn't exported), and as a last resort to the
    constructed sibling path anyway — NOT a bare ``"tj"`` — so
    ``_daemon_program_args``'s ephemeral-cache detection (uv/pipx cache
    paths) still has a real path shape to inspect. The old
    ``sys.executable.replace("/python", "/tj")`` rewrote a ``python3``-named
    interpreter to a nonexistent ``tj3`` (``/python`` matches inside
    ``/python3``), so launchd/systemd pointed at a binary that doesn't exist
    while ``launchctl load`` still returned 0 (#340).
    """
    sibling = Path(sys.executable).with_name("tj")
    if sibling.exists():
        return str(sibling)
    return shutil.which("tj") or str(sibling)


def _daemon_program_args(config_path: str) -> list[str] | None:
    """Resolve the argv the daemon unit (launchd/systemd) should launch.

    Prefers a direct path to `tj`. When the only resolvable `tj` sits inside
    an ephemeral uv/pipx cache (`uvx`/`pipx run` — see `_is_ephemeral_path`),
    writing that raw path into launchd/systemd is a landmine: `uv cache
    prune` / `uv cache clean` (routine maintenance, also run by some CI/cleanup
    tools) deletes it and silently kills the daemon on next load, and the
    path pins the daemon to whatever version was resolved at onboard time
    forever, independent of the wrapper's `--refresh` freshness logic (#111,
    #155). Fall back to invoking through the stable `uvx`/`pipx` shim itself
    instead, so launchd/systemd re-resolves `tj` through uv/pipx on every
    start rather than a path that can vanish out from under it. Returns None
    when no durable entrypoint exists at all — the caller should warn and
    skip installing rather than silently write a cache path.
    """
    tj_path = _resolve_tj_binary()
    if not _is_ephemeral_path(tj_path):
        return [tj_path, "--config", config_path, "serve"]

    uvx_path = shutil.which("uvx")
    if uvx_path and not _is_ephemeral_path(uvx_path):
        return [uvx_path, "--from", "tokenjam", "tj", "--config", config_path, "serve"]

    pipx_path = shutil.which("pipx")
    if pipx_path and not _is_ephemeral_path(pipx_path):
        return [pipx_path, "run", "--spec", "tokenjam", "tj", "--config", config_path, "serve"]

    return None


def _warn_no_durable_daemon_entrypoint(unit_kind: str) -> None:
    console.print(
        f"[warn]No durable `tj` entrypoint found — skipping {unit_kind} "
        "install rather than pointing it at a throwaway uvx/pipx cache path "
        "that `uv cache prune` would silently delete.[/warn]"
    )
    console.print(
        "[dim]Install tokenjam persistently (`uv tool install tokenjam` or "
        "`pipx install tokenjam`) and re-run `tj onboard`, or run "
        "`tj serve` manually.[/dim]"
    )


def _install_launchd(config_path: str, *, suppress_headsup: bool = False) -> str | None:
    program_args = _daemon_program_args(config_path)
    if program_args is None:
        _warn_no_durable_daemon_entrypoint("launchd")
        return None
    args_xml = "\n".join(f"        <string>{arg}</string>" for arg in program_args)
    plist_path = Path.home() / "Library/LaunchAgents/com.tokenjam.serve.plist"
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_content = f"""\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.tokenjam.serve</string>
    <key>ProgramArguments</key>
    <array>
{args_xml}
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardErrorPath</key>
    <string>/tmp/tj-serve.err</string>
    <key>StandardOutPath</key>
    <string>/tmp/tj-serve.out</string>
</dict>
</plist>"""
    plist_path.write_text(plist_content)
    # Unload any existing registration before loading the updated plist.
    # Ignore errors — the service may not be registered yet on first install.
    subprocess.run(
        ["launchctl", "unload", "-w", str(plist_path)],
        capture_output=True, text=True,
    )
    # `-w` clears the Disabled=true flag that `tj stop` writes via
    # `launchctl unload -w`. Without `-w` here, the daemon stays disabled
    # in launchd's database and load is a no-op even though it returns 0.
    result = subprocess.run(
        ["launchctl", "load", "-w", str(plist_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        console.print(f"[warn]Daemon plist written to {plist_path} but "
                      f"launchctl load failed.[/warn]")
        console.print("[dim]Try loading manually:[/dim]")
        console.print(f"  launchctl load {plist_path}")
        console.print("[dim]Or run the server directly:[/dim]")
        console.print("  tj serve &")
        return None
    # A zero exit from `launchctl load` proves only that launchctl accepted the
    # request. Verify the label is actually resident: the failure behind #614
    # left a valid RunAtLoad/KeepAlive plist on disk while `launchctl list`
    # showed no job, so onboarding reported a daemon that would never restart.
    loaded = subprocess.run(
        ["launchctl", "list", "com.tokenjam.serve"],
        capture_output=True,
        text=True,
    )
    if loaded.returncode != 0:
        console.print(
            f"[warn]Daemon plist written to {plist_path}, but launchd did not "
            "load com.tokenjam.serve.[/warn]"
        )
        console.print("[dim]Re-run `tj onboard` or run `tj serve` manually.[/dim]")
        return None
    # The --claude-code completion (#675) moves this heads-up to the very BOTTOM
    # of the payoff screen (after Next steps) so it reads top-to-bottom, hence
    # `suppress_headsup` there; every other onboard path prints it inline here as
    # before.
    if not suppress_headsup:
        _print_background_items_headsup()
    return f"Daemon installed at {plist_path}"


def _install_systemd(config_path: str) -> str | None:
    program_args = _daemon_program_args(config_path)
    if program_args is None:
        _warn_no_durable_daemon_entrypoint("systemd")
        return None
    exec_start = " ".join(program_args)
    service_path = Path.home() / ".config/systemd/user/tokenjam.service"
    service_path.parent.mkdir(parents=True, exist_ok=True)
    service_content = f"""\
[Unit]
Description=TokenJam observability server
After=network.target

[Service]
ExecStart={exec_start}
Restart=on-failure

[Install]
WantedBy=default.target"""
    service_path.write_text(service_content)
    subprocess.run(
        ["systemctl", "--user", "enable", "--now", "tokenjam"],
        check=True,
    )
    enabled = subprocess.run(
        ["systemctl", "--user", "is-enabled", "tokenjam"],
        capture_output=True,
        text=True,
    )
    if enabled.returncode != 0:
        console.print(
            f"[warn]Daemon unit written to {service_path}, but systemd did not "
            "enable tokenjam.[/warn]"
        )
        console.print("[dim]Re-run `tj onboard` or run `tj serve` manually.[/dim]")
        return None
    return f"Daemon installed at {service_path}"
