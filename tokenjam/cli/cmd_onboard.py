from __future__ import annotations

import json as json_mod
import platform
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

import click
from rich.markup import escape

from tokenjam.cli.banner import print_welcome_banner
from tokenjam.cli.onboard_detect import SdkMatch, detect_stack, install_hint
from tokenjam.core.config import find_config_file
from tokenjam.utils.formatting import console, display_path

# --- output-trim (`tj hook cap-output`) PostToolUse hook wiring --------------
# Installed into ~/.claude/settings.json out-of-band (zero in-loop token cost).
# Idempotent + non-destructive: a tj-managed entry is detected by the
# "hook cap-output" substring in its command, so re-onboard updates OUR entry in
# place and NEVER clobbers a user's own PostToolUse hooks.

_CAP_OUTPUT_MATCHER = "Bash|Grep|Glob|WebFetch"
_CAP_OUTPUT_MARKER = "hook cap-output"  # tj-managed marker (substring of command)


def _tj_cap_output_command() -> str:
    """Absolute `tj hook cap-output` command, falling back to bare `tj`."""
    exe = shutil.which("tj")
    return f"{exe} hook cap-output" if exe else "tj hook cap-output"


def _is_tj_cap_output_entry(entry: object) -> bool:
    if not isinstance(entry, dict):
        return False
    for h in entry.get("hooks", []) or []:
        if isinstance(h, dict) and _CAP_OUTPUT_MARKER in str(h.get("command", "")):
            return True
    return False


def _wire_claude_output_cap_hook(settings: dict) -> str:
    """Install/refresh the PostToolUse cap-output hook in a settings dict.

    Mutates ``settings`` in place; returns one of ``written`` / ``updated`` /
    ``kept`` / ``skipped`` (foreign structure left untouched).
    """
    desired = {
        "matcher": _CAP_OUTPUT_MATCHER,
        "hooks": [{"type": "command", "command": _tj_cap_output_command()}],
    }
    hooks = settings.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        return "skipped"
    post = hooks.get("PostToolUse")
    if post is None:
        hooks["PostToolUse"] = [desired]
        return "written"
    if not isinstance(post, list):
        return "skipped"
    for i, entry in enumerate(post):
        if _is_tj_cap_output_entry(entry):
            if entry == desired:
                return "kept"
            post[i] = desired
            return "updated"
    post.append(desired)          # preserve any foreign PostToolUse hooks
    return "written"


def _unwire_claude_output_cap_hook(settings: dict) -> bool:
    """Remove any tj-managed cap-output PostToolUse entry. Returns True if one
    was removed (used by `tj uninstall`)."""
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


def _is_ephemeral_runner() -> bool:
    """True when this process is running from a throwaway uvx/pipx-run venv
    rather than a persistent install.

    Persistent installs resolve `sys.executable` into a stable, named
    location:
      - `uv tool install`  → .../uv/tools/<pkg>/...
      - `pipx install`     → .../pipx/venvs/<pkg>/...  (see `_installed_via_pipx`
        in cmd_uninstall.py — same signature, different concern)
    Ephemeral `uvx` / `pipx run` executions resolve into a cache directory
    instead (e.g. `~/.cache/uv/archive-v0/...` for uvx).
    """
    exe = sys.executable.replace("\\", "/")
    if "/uv/tools/" in exe or "/pipx/venvs/" in exe:
        return False
    return "/uv/" in exe or "/pipx/" in exe


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
        "\n[yellow]Heads up:[/yellow] you're running via a temporary "
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
            "[red]Persistent install failed.[/red] Continuing this "
            "session ephemerally — run [bold]pipx install tokenjam && "
            "tj onboard[/bold] yourself afterward.\n"
        )
        return

    console.print(f"[green]✓[/green] Installed — re-running onboard via {tj_path}\n")
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


@click.command("onboard")
@click.option("--claude-code", "claude_code", is_flag=True, default=False,
              help="Configure Claude Code telemetry to flow into tj")
@click.option("--codex", "codex", is_flag=True, default=False,
              help="Configure Codex CLI telemetry to flow into tj")
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
@click.option("--project", "project_override", default=None,
              help="Project name to group this repo under in the dashboard "
                   "(OTel service.namespace — e.g. all Aquanodeio/* repos under "
                   "'aquanode'). Defaults to the git org. Used with --claude-code.")
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
                reconfigure: bool, plan: str | None, project_override: str | None,
                verify: bool, verify_only: bool) -> None:
    """Interactive setup wizard for tj."""
    # --verify-only is the documented post-restart re-check: config already
    # exists, the user just restarted the agent, and re-running the whole wizard
    # (config rewrite + full summary + restart banner) only to poll is wasteful
    # noise (#102). Skip straight to the poll, before the banner and any setup.
    if verify_only:
        _run_verify_only(ctx, claude_code=claude_code, codex=codex)
        return
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
                             project_override, verify=verify)
        return
    if codex:
        _onboard_codex(ctx, budget, no_daemon, force, reconfigure, plan, verify=verify)
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
        if choice == "claude_code":
            _onboard_claude_code(ctx, budget, no_daemon, force, reconfigure, plan,
                                 project_override, verify=verify)
            return
        if choice == "codex":
            _onboard_codex(ctx, budget, no_daemon, force, reconfigure, plan,
                           verify=verify)
            return
        if choice == "combination":
            _onboard_combination(ctx, budget, no_daemon, force, plan,
                                  project_override, verify=verify)
            return
        # choice == "sdk" → fall through to the generic SDK/API path below.

    existing = find_config_file()
    if existing and not force:
        # --reconfigure is only meaningful with --claude-code / --codex.
        # The bare onboard path writes a generic config and doesn't manage
        # plan tier — that's per-provider, written into [budget.<provider>]
        # sections by the integration-specific onboarders. Silently
        # no-op'ing here would frustrate users who pass --reconfigure --plan
        # expecting their plan field to update (#68 §1).
        if reconfigure:
            console.print(
                "[red]--reconfigure has no effect without --claude-code or "
                "--codex.[/red]"
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

    ingest_secret = secrets.token_hex(32)

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

    config_text = f"""\
# TokenJam configuration
# Docs: https://github.com/Metabuilder-Labs/tokenjam#configuration

[defaults.budget]
{budget_line}
{plan_section}
[security]
ingest_secret = "{ingest_secret}"

[capture]
prompts = false
completions = false
tool_inputs = false
tool_outputs = false

[storage]
path = "~/.tj/telemetry.duckdb"
retention_days = 90

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
        console.print(f"[green]\u2713[/green] {apply_msg}")

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
    console.print("[green]\u2713[/green] Config written to [bold].tj/config.toml[/bold]")
    console.print(f"[green]\u2713[/green] Ingest secret generated: "
                  f"[dim]{ingest_secret[:8]}...[/dim]")
    if budget and budget > 0:
        console.print(f"[green]\u2713[/green] Default daily budget: "
                      f"[bold]${budget:.2f}[/bold] per agent")
    if plan_tier:
        # Escape the TOML section header \u2014 Rich treats `[budget.<provider>]` as a
        # markup tag and would strip it, leaving "(written to )". See issue #157.
        plan_section = escape(f"[budget.{plan_provider}]")
        console.print(f"[green]\u2713[/green] Plan tier: "
                      f"[bold]{plan_tier}[/bold] (written to {plan_section})")
    if daemon_msg:
        console.print(f"[green]\u2713[/green] {daemon_msg}")

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

    ``persona`` is one of ``"sdk"``, ``"claude_code"``, ``"codex"`` and drives
    the instruction shown and the not-confirmed cause.
    """
    if not verify:
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
        not_confirmed_cause,
        open_read_backend,
        poll_for_first_span,
    )
    from tokenjam.utils.time_parse import utcnow

    backend, _mode, error = open_read_backend(config)
    if backend is None:
        console.print(f"\n[yellow]Can't verify yet[/yellow] \u2014 {error}.")
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
        console.print(f"[dim]Polling for up to {int(timeout_s)}s\u2026[/dim]")
        result = poll_for_first_span(backend, since, timeout_s=timeout_s)
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    if result.confirmed:
        console.print(
            f"[green]\u2713 Receiving telemetry![/green] First span arrived after "
            f"{result.elapsed_s:.0f}s \u2014 you're wired up."
        )
    elif result.error:
        console.print(f"[yellow]Couldn't verify[/yellow] \u2014 {result.error}.")
    else:
        console.print(
            f"[yellow]\u26a0 No telemetry yet[/yellow] after {int(timeout_s)}s. "
            + not_confirmed_cause(persona)
        )


def _run_verify_only(ctx: click.Context, *, claude_code: bool, codex: bool) -> None:
    """Poll an already-configured install for its first live span, skipping setup.

    The persona (and which config to read) follows the same flag the user
    onboarded with: ``--claude-code`` / ``--codex`` read the global config;
    bare reads the nearest project/SDK config via ``find_config_file``. Errors
    out cleanly when no config exists yet — that's an "run `tj onboard` first"
    situation, not a verification failure.
    """
    from tokenjam.core.config import load_config

    if claude_code or codex:
        global_path = Path.home() / ".config" / "tj" / "config.toml"
        persona = "claude_code" if claude_code else "codex"
        config_path: Path | None = global_path if global_path.exists() else None
    else:
        found = find_config_file()
        config_path = Path(found) if found else None
        persona = "sdk"

    if config_path is None:
        console.print(
            "[red]No tj config found.[/red] Run [bold]tj onboard[/bold] "
            "(optionally with --claude-code / --codex) first, then "
            "[bold]tj onboard --verify-only[/bold]."
        )
        ctx.exit(1)
        return

    try:
        config = load_config(str(config_path.resolve()))
    except Exception as exc:  # noqa: BLE001 — surface a clean message, no traceback
        console.print(f"[red]Could not load config[/red] at {display_path(config_path)}: {exc}")
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
        console.print(f"  {i}) {desc}")
    keys = [k for k, _ in choices]
    default_idx = keys.index(current) + 1 if current in keys else 1
    raw = click.prompt(
        "Choose",
        type=click.IntRange(1, len(choices)),
        default=default_idx,
        show_default=True,
    )
    return keys[int(raw) - 1]


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
    ("claude_code", "Claude Code"),
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
    """Best-effort Codex backfill, called defensively (#448).

    The Codex on-disk backfill adapter (`tj backfill codex`) is being added by
    another workstream and may not have landed yet. This resolves it at runtime
    via a guarded import so the combination / --codex flows work whether or not
    that adapter exists. When it's unavailable, returns forward-only framing so
    the caller can say "wired — data flows as you use Codex" instead of claiming
    a backfill that didn't happen (honesty discipline, Rule 14).

    Returns ``(message, has_data, sessions_total)``:
      * ``message`` — a human summary line, or None if nothing to report.
      * ``has_data`` — True only when at least one session was actually ingested.
      * ``sessions_total`` — distinct sessions ingested (0 when unavailable).
    """
    try:
        from tokenjam.core.ingest_adapters.codex import (  # type: ignore[import]
            ingest_codex,
        )
    except Exception:
        # Adapter not shipped yet (or import error) — forward-only, no claim.
        return None, False, 0

    try:
        from tokenjam.core.db import open_db
        db = open_db(config.storage)
        try:
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

    Reuses ``cli/home.print_home`` for the "You're set up" next-best-actions
    block (config is always written by the time we reach here), prefixed with an
    honest one-line "N sessions backfilled" note when a backfill actually
    happened. Copy stays honest — no promised savings (Critical Rule 14).
    """
    from tokenjam.cli.home import print_home

    console.print()
    if has_data and sessions_backfilled > 0:
        span = f" across the last {days} days" if days else ""
        console.print(
            f"[bold green]You're set up.[/bold green] "
            f"[green]{sessions_backfilled} session"
            f"{'s' if sessions_backfilled != 1 else ''} backfilled"
            f"{span}.[/green]"
        )
    else:
        console.print("[bold green]You're set up.[/bold green]")
    # print_home() re-renders the banner + the configured next-best-actions
    # (tj status / tokenmaxx / optimize / serve).
    print_home()


def _prompt_daily_budget(budget: float | None) -> float:
    """Prompt for the per-agent daily-budget alert threshold, unless already
    supplied via --budget. Called AFTER the plan prompt so onboard reads
    plan-first (#240)."""
    if budget is not None:
        return budget
    return click.prompt(
        "Daily budget in USD (0 = no limit, default 0)",
        type=float, default=0.0, show_default=False,
    )


def _print_next_steps_nudge(*, has_data: bool, days: int | None = None) -> None:
    """Curated post-onboard nudge (#240).

    Leads with the commands that work on the just-backfilled data *immediately*
    — no Claude Code restart required — because onboard otherwise ends on
    "Restart Claude Code", the exact point we lose new users. The restart note
    (for live telemetry) is printed separately, after this. Curated to ~3
    high-wow commands rather than a `--help` wall; copy stays honest (no
    promised savings — Critical Rule 14).
    """
    console.print()
    if has_data:
        span = f"last {days} days" if days else "history"
        console.print(
            f"[bold]▸ Next steps[/bold]  [dim]your {span} "
            "already loaded — these work right now:[/dim]"
        )
    else:
        console.print(
            "[bold]▸ Next steps[/bold]  [dim]these work right now:[/dim]"
        )
    console.print()
    console.print("  [bold]tj tokenmaxx[/bold]   [dim]your shareable spend tier[/dim]")
    console.print("  [bold]tj optimize[/bold]    [dim]cost-saving candidates from your usage[/dim]")
    console.print("  [bold]tjb[/bold]            [dim]prove a cheaper model still holds (pip install tokenjam-bench)[/dim]")
    console.print("  [bold]tj serve[/bold]       [dim]open Lens (web UI) at http://127.0.0.1:7391/[/dim]")
    console.print()


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


def _print_statusline_status(status: str) -> None:
    """Render the statusLine wiring outcome in the onboard summary block."""
    if status in ("written", "updated", "kept"):
        verb = {"written": "wired", "updated": "updated", "kept": "already set"}[status]
        console.print(
            f"  Statusline:          {verb} (tj statusline — zero token cost)"
        )
    elif status == "skipped":
        console.print(
            "  [yellow]Statusline:          left your existing statusLine "
            "untouched[/yellow] (set it to `tj statusline` to enable tj's line)."
        )


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


def _zshrc_otel_block(port: int, secret: str) -> str:
    """Build one fresh sentinel-delimited OTEL export block for ~/.zshrc."""
    return (
        f"{_ZSHRC_OTEL_START}\n"
        f"export CLAUDE_CODE_ENABLE_TELEMETRY=1\n"
        f"export OTEL_LOGS_EXPORTER=otlp\n"
        f"export OTEL_EXPORTER_OTLP_PROTOCOL=http/json\n"
        f"export OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:{port}\n"
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
    project_override: str | None = None,
    verify: bool = False,
    standalone: bool = True,
) -> None:
    """Configure Claude Code to send telemetry to tj.

    ``standalone`` is True on the single-path flow (`tj onboard --claude-code`)
    and False when this runs as one leg of the combination flow (#432). On the
    combination path the closing home banner must print exactly once, at the end
    of `_onboard_combination` — so we suppress it here when not standalone. The
    inline Claude Code backfill still runs (it is only ever invoked from here).
    """
    from tokenjam.core.config import (
        AgentConfig, BudgetConfig, ProviderBudget, TjConfig, SecurityConfig,
        load_config, write_config,
    )

    # --claude-code always uses the global config so that all projects share one
    # ingest secret and one running daemon. Per-project configs cause the secret in
    # ~/.claude/settings.json to rotate on every project onboard, breaking auth for
    # every other project.
    global_config_path = Path.home() / ".config" / "tj" / "config.toml"

    project_name = _derive_project_name()
    agent_id = f"claude-code-{project_name}"
    # Project name = OTel service.namespace, the key the dashboard groups by.
    # A meta-repo (e.g. git repo "harness" holding all of "aquanode") wants a
    # human project name, so prompt with the repo name as default. --project
    # skips the prompt for non-interactive use.
    if project_override:
        namespace = project_override
    else:
        namespace = click.prompt(
            "Project name (groups related repos under one dashboard tile)",
            default=project_name, show_default=True,
        ).strip() or project_name

    # Plan-first (#240): resolve the plan tier before prompting for the daily
    # budget — "How do you pay?" is the more important, more natural opener.
    if global_config_path.exists() and not force:
        config = load_config(str(global_config_path))
        if agent_id not in config.agents:
            config.agents[agent_id] = AgentConfig()
        # Server-side project mapping so already-running sessions group by
        # project without restarting the agent (see AgentConfig.project).
        config.agents[agent_id].project = namespace

        existing_plan = (
            config.budgets["anthropic"].plan
            if "anthropic" in config.budgets else None
        )
        plan_changed = False
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
            usd: float | None = None
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
        budget = _prompt_daily_budget(budget)
        if budget and budget > 0:
            config.agents[agent_id].budget.daily_usd = budget
        config_path = global_config_path
        write_config(config, config_path)
        console.print(f"  tj config updated: {display_path(config_path)}", soft_wrap=True)
    else:
        ingest_secret = secrets.token_hex(32)
        if plan_override:
            plan = plan_override
        else:
            plan = _prompt_plan("Claude", _ANTHROPIC_PLAN_CHOICES)
        plan_changed = False
        usd: float | None = None  # type: ignore[no-redef]
        if plan == "api" and not plan_override:
            ceiling = click.prompt(
                "Monthly Anthropic API spend ceiling in USD (0 = no limit)",
                type=float, default=0.0, show_default=False,
            )
            if ceiling > 0:
                usd = ceiling
        budget = _prompt_daily_budget(budget)
        daily_usd = budget if budget and budget > 0 else None
        agents = {agent_id: AgentConfig(budget=BudgetConfig(daily_usd=daily_usd), project=namespace)}
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
        write_config(config, config_path)
        console.print(f"  tj config written to: {display_path(config_path)}", soft_wrap=True)
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
    backfill_days: int | None = None
    backfill_sessions_total = 0
    try:
        from tokenjam.core.backfill import (
            CLAUDE_CODE_PROJECTS_ROOT, ingest_claude_code,
        )
        if CLAUDE_CODE_PROJECTS_ROOT.exists():
            from tokenjam.core.db import open_db
            try:
                db = open_db(config.storage)
                result = ingest_claude_code(db, config=config)
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
                    backfill_days = days
                    # Report new / already-present / total so a re-run reads as
                    # "13 total" rather than "1 session" (#238).
                    total = result.sessions_total
                    backfill_sessions_total = total
                    pieces = [
                        f"{result.sessions_new} new "
                        f"({result.sessions_existing} already present) · "
                        f"{total} total session{'s' if total != 1 else ''}"
                    ]
                    if days is not None:
                        pieces.append(f"over {days} day{'s' if days != 1 else ''}")
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
    resume_brief_status = _wire_claude_resume_brief_hook(global_settings)

    # Install the output-trim PostToolUse hook (zero in-loop token cost). Rides
    # this same read-merge-write. DEFAULT-OFF opt-in (config gate): the A/B gate
    # failed — Claude Code already truncates Bash output to ~30 KB before the
    # PostToolUse hook sees it, so the addressable bloat is tiny and treatment
    # cost was +5.6% (noise). See the `[hooks.output_cap]` docstring in
    # core/config.py. So onboarding does NOT auto-install it: a fresh config has
    # enabled=False → we take the else branch and remove any previously-installed
    # tj entry, keeping config the source of truth. Users opt in by setting
    # `[hooks.output_cap].enabled = true` and re-running onboard. Idempotent +
    # non-destructive (foreign hooks preserved).
    cap_cfg = getattr(getattr(config, "hooks", None), "output_cap", None)
    cap_enabled = bool(getattr(cap_cfg, "enabled", False)) and not bool(
        getattr(cap_cfg, "killswitch", False)
    )
    if cap_enabled:
        cap_status = _wire_claude_output_cap_hook(global_settings)
    else:
        removed = _unwire_claude_output_cap_hook(global_settings)
        cap_status = "removed" if removed else "disabled"

    global_settings_path.write_text(json_mod.dumps(global_settings, indent=2) + "\n")
    _CAP_STATUS_MSG = {
        "written": "installed (Bash/Grep/Glob/WebFetch)",
        "updated": "updated to current path",
        "kept": "already installed",
        "skipped": "left alone (custom PostToolUse hooks present)",
        "removed": "removed (opt-in; disabled in config)",
        "disabled": "off (opt-in — set [hooks.output_cap].enabled = true)",
    }
    console.print(
        f"[green]✓[/green] Output-trim hook (tj hook cap-output): "
        f"{_CAP_STATUS_MSG.get(cap_status, cap_status)}"
    )
    _RESUME_BRIEF_STATUS_MSG = {
        "written": "installed (SessionStart: resume|compact)",
        "updated": "updated to current path",
        "kept": "already installed",
        "skipped": "skipped — ~/.claude/settings.json has malformed hooks "
                   "(expected object with SessionStart list); fix and re-run",
    }
    console.print(
        f"[green]✓[/green] Resume-brief hook (tj resume-brief --from-hook): "
        f"{_RESUME_BRIEF_STATUS_MSG.get(resume_brief_status, resume_brief_status)}"
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
    # Writes host.docker.internal endpoint so harness sessions (Docker) pick up
    # the vars automatically via compose.yml passthrough — no manual setup needed.
    # Native Claude Code uses settings.json (127.0.0.1) written above instead.
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
    _finish_onboard_serve(
        str(config_path.resolve()),
        want_daemon=want_daemon,
        plan_changed=plan_changed,
        stopped_for_db=stopped_for_db,
        secret_rotated=False,
        no_daemon=no_daemon,
        force=force,
    )

    console.print()
    console.print("[bold green]Claude Code observability configured.[/bold green]")
    console.print(
        f"  Global settings:     {display_path(global_settings_path)}", soft_wrap=True
    )
    console.print(
        f"  Project settings:    {display_path(project_settings_path)}", soft_wrap=True
    )
    _print_statusline_status(statusline_status)
    if removed_resource_attr:
        console.print(
            "  [yellow]Removed a hardcoded OTEL_RESOURCE_ATTRIBUTES from project "
            "settings[/yellow] (the claude wrapper now sets it per terminal)."
        )
    console.print("  Shell env:           ~/.zshrc (harness-compatible endpoint)")
    if wrapper_files:
        console.print(
            f"  claude wrapper:      "
            f"{', '.join(display_path(w) for w in wrapper_files)} "
            "(per-terminal naming)",
            soft_wrap=True,
        )
        console.print(
            "  [dim]The claude wrapper controls OTEL_RESOURCE_ATTRIBUTES per "
            "terminal (service.instance.id); project settings.json no longer "
            "sets it.[/dim]"
        )
    console.print(f"  Agent ID:            {agent_id}")
    if budget and budget > 0:
        console.print(f"  Daily budget:        ${budget:.2f}")
    console.print(f"  OTLP endpoint:       http://127.0.0.1:{port} (native)")
    console.print(f"                       http://host.docker.internal:{port} (harness)")
    if secret:
        console.print(f"  Ingest secret:       {secret[:8]}...")
    if backfill_msg:
        console.print(f"  Backfilled:          {backfill_msg}")
    # Surface what we wrote for [budget.anthropic]: the user's plan tier, and
    # the spending ceiling only when one is set (API users may opt in to one).
    from tokenjam.core.config import load_config as _lc
    try:
        _cfg = _lc(str(global_config_path))
        _ab = _cfg.budgets.get("anthropic")
        if _ab is not None and _ab.plan:
            _line = f"[budget.anthropic] plan = \"{_ab.plan}\""
            if _ab.usd:
                _line += f", usd = {_ab.usd}"
            console.print(f"  Budget projection:   {_line}")
    except Exception:
        pass
    console.print()
    # tj is out-of-band for Claude Code: the statusline (zero model tokens),
    # not an in-loop MCP server. Say so explicitly so users know where tj lives.
    if statusline_status == "skipped":
        console.print(
            "[dim]tj did not touch your existing statusLine. To see tj's "
            "re-read/quota line, set your Claude Code statusLine command to[/dim]  "
            "tj statusline"
        )
    else:
        console.print(
            "[dim]tj is now in your Claude Code statusline "
            "([bold]zero token cost[/bold]) — it shows this session's re-read "
            "share and nudges [bold]/compact[/bold] when re-reading eats your "
            "quota.[/dim]"
        )
    console.print(
        "[dim]For the deep dive, run[/dim]  tj tokenmaxx  [dim]/[/dim]  tj optimize"
    )
    console.print()
    # Lead with the wins that need no restart, THEN the restart note (#240).
    _print_next_steps_nudge(has_data=backfill_has_data, days=backfill_days)
    if not want_daemon:
        _warn_manual_serve_restart(stopped_for_db=stopped_for_db, no_daemon=True)
        console.print("[dim]Start the server:[/dim]  tj serve")
    _print_restart_banner("Claude Code")
    console.print(
        "[dim]Open a new terminal, then launch with[/dim]  claude  "
        "[dim](each terminal becomes its own dashboard tile;[/dim] "
        "claude --as <name> [dim]to label it).[/dim]"
    )
    console.print(f"[dim]After restarting, run:[/dim]  tj status --agent {agent_id}")

    _maybe_verify_onboarding(config, persona="claude_code", verify=verify)

    # Shared closing banner (#448): every onboard path ends on the branded home
    # screen + tailored next-best-actions. For Claude Code the success signal is
    # the backfill ("N sessions backfilled"), NOT a live span — the log parse
    # already ran above. On the combination path this is deferred to
    # `_onboard_combination` so the banner prints exactly once (#432).
    if standalone:
        _print_setup_complete_home(
            sessions_backfilled=backfill_sessions_total,
            has_data=backfill_has_data,
            days=backfill_days,
        )


def _onboard_codex(
    ctx: click.Context,
    budget: float | None,
    no_daemon: bool,
    force: bool,
    reconfigure: bool = False,
    plan_override: str | None = None,
    verify: bool = False,
    standalone: bool = True,
) -> None:
    """Configure Codex CLI to send telemetry to tj.

    ``standalone`` is True on the single-path flow (`tj onboard --codex`) and
    False when this runs as one leg of the combination flow (#432). When not
    standalone we skip BOTH the internal Codex backfill and the closing home
    banner: `_onboard_combination` runs the Codex backfill exactly once itself
    and prints the banner exactly once at the very end. Running the backfill
    here too would double-run it, and printing the banner here would show it up
    to three times.
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
    config_path = Path.home() / ".config" / "tj" / "config.toml"

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

        existing_plan = (
            config.budgets["openai"].plan
            if "openai" in config.budgets else None
        )
        plan_changed = False
        if existing_plan is None or reconfigure or plan_override:
            if plan_override:
                plan = plan_override
            else:
                plan = _prompt_plan("OpenAI / Codex", _OPENAI_PLAN_CHOICES, current=existing_plan)
            plan_changed = plan != existing_plan
            usd: float | None = None
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
        budget = _prompt_daily_budget(budget)
        if budget and budget > 0:
            config.agents[agent_id].budget.daily_usd = budget
        write_config(config, config_path)
        console.print(f"  tj config updated: {display_path(config_path)}", soft_wrap=True)
    else:
        ingest_secret = secrets.token_hex(32)
        if plan_override:
            plan = plan_override
        else:
            plan = _prompt_plan("OpenAI / Codex", _OPENAI_PLAN_CHOICES)
        plan_changed = False
        usd: float | None = None  # type: ignore[no-redef]
        if plan == "api" and not plan_override:
            ceiling = click.prompt(
                "Monthly OpenAI API spend ceiling in USD (0 = no limit)",
                type=float, default=0.0, show_default=False,
            )
            if ceiling > 0:
                usd = ceiling
        budget = _prompt_daily_budget(budget)
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
        write_config(config, config_path)
        console.print(f"  tj config written to: {display_path(config_path)}", soft_wrap=True)
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
    # the tj MCP server for Codex — an in-loop MCP is a per-turn quota burden on
    # subscription power users (see the +36% A/B in ticket #59). Codex has no
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
    console.print("[bold green]Codex CLI observability configured.[/bold green]")
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
    if mcp_was_removed:
        # Plain echo: Rich would treat [mcp_servers.tj] as a markup tag and strip it.
        click.echo(
            "  Removed [mcp_servers.tj] — tj is out-of-band for Codex now "
            "(no per-turn quota burden)."
        )
    # Lead with the wins that need no restart, THEN the restart note (#240).
    # `has_data` is True only when the Codex backfill actually ingested sessions
    # (adapter present + on-disk history); otherwise forward-only framing.
    _print_next_steps_nudge(has_data=codex_has_data)
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


def _onboard_combination(
    ctx: click.Context,
    budget: float | None,
    no_daemon: bool,
    force: bool,
    plan_override: str | None = None,
    project_override: str | None = None,
    verify: bool = False,
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
            "[yellow]Nothing selected.[/yellow] Re-run [bold]tj onboard[/bold] "
            "and pick at least one."
        )
        ctx.exit(1)
        return

    done: list[str] = []

    # --- Claude Code (full flow: config + statusline + auto-backfill) ---
    if uses_cc:
        console.print()
        console.print("[bold]Setting up Claude Code…[/bold]")
        _onboard_claude_code(
            ctx, budget, no_daemon, force, reconfigure=False,
            plan_override=plan_override, project_override=project_override,
            verify=False, standalone=False,
        )
        done.append("Claude Code")

    # --- Codex (full flow: OTel wiring) then a defensive backfill ---
    if uses_codex:
        console.print()
        console.print("[bold]Setting up Codex…[/bold]")
        _onboard_codex(
            ctx, budget, no_daemon, force, reconfigure=False,
            plan_override=plan_override, verify=False, standalone=False,
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
        f"[bold green]Combination setup complete[/bold green] "
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
    """
    from rich.panel import Panel
    from rich.text import Text

    body = Text()
    body.append("Restart ", style="bold")
    body.append(app_name, style="bold yellow")
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
            title="[bold]Action required[/bold]",
            border_style="yellow",
            padding=(1, 2),
        )
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
) -> str | None:
    """Install or restart ``tj serve`` after onboard DB/config writes."""
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
            console.print(
                "  Daemon:              already running (skipped reinstall)"
            )
            restart_msg = "daemon already running"
        else:
            console.print("  Daemon:              installing...")
            install_msg = _install_daemon(config_path)
            if install_msg:
                restart_msg = install_msg

    if want_daemon and need_restart and not force:
        if secret_rotated:
            reason = "secret"
        elif plan_changed:
            reason = "plan"
        else:
            reason = "db_update"
        restart_msg = _restart_tj_server(config_path, no_daemon, reason=reason)
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

    daemon_msg = _install_daemon(config_path)
    if daemon_msg:
        if reason == "secret":
            return "restarted to pick up new ingest secret"
        if reason == "plan":
            return "restarted to pick up plan tier change"
        return "restarted to pick up config changes"
    if stopped:
        return "stopped stale server; please run `tj serve` manually"
    return "restart attempted; verify with `tj status`"


def _restart_tj_server_for_secret_rotation(config_path: str, no_daemon: bool) -> str:
    """Backward-compatible alias for secret-rotation restarts."""
    return _restart_tj_server(config_path, no_daemon, reason="secret")


def _codex_mcp_toml_block() -> str:
    """Return the legacy [mcp_servers.tj] TOML block for ~/.codex/config.toml.

    Retained only to *recognize* a previously tj-written block so onboard can
    retire it (see ``_codex_strip_tj_mcp_from_content``). tj no longer registers
    an MCP server for Codex — an in-loop MCP is a per-turn quota burden on
    subscription users (ticket #59); tj is out-of-band for Codex via OTel.
    """
    return (
        "[mcp_servers.tj]\n"
        "# Managed by tj — gives Codex access to TokenJam observability tools\n"
        'command = "tj"\n'
        'args = ["mcp"]\n'
    )


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

    Written portably so it works in both zsh and bash.
    """
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
        f'  export OTEL_RESOURCE_ATTRIBUTES="$(tj otel-resource-attrs),service.instance.id=$_tj_inst"\n'
        f"  # Report this terminal's session closed on exit/interrupt so the\n"
        f"  # dashboard archives its tile. Idempotent — double-fire is harmless.\n"
        f"  trap 'tj session-end --instance \"$_tj_inst\" >/dev/null 2>&1 || true' INT TERM HUP\n"
        f'  command claude "${{_tj_args[@]}}"\n'
        f"  local _tj_status=$?\n"
        f"  trap - INT TERM HUP\n"
        f'  tj session-end --instance "$_tj_inst" >/dev/null 2>&1 || true\n'
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


def _install_daemon(config_path: str) -> str | None:
    """Install background daemon. Returns success message or None."""
    system = platform.system()
    try:
        if system == "Darwin":
            return _install_launchd(config_path)
        elif system == "Linux":
            return _install_systemd(config_path)
        else:
            console.print(f"[yellow]Background daemon not supported on {system}. "
                          "Run `tj serve` manually.[/yellow]")
            return None
    except Exception as e:
        console.print(f"[yellow]Daemon installation failed: {e}[/yellow]")
        console.print("[dim]You can run `tj serve` manually instead.[/dim]")
        return None


def _resolve_tj_binary() -> str:
    """Resolve the absolute path to the ``tj`` executable for daemon units.

    Prefer the copy on PATH. When ``tj`` isn't on PATH (e.g. a venv whose bin
    dir isn't exported), fall back to the sibling ``tj`` next to the running
    interpreter — derived by path, not string substitution. The old
    ``sys.executable.replace("/python", "/tj")`` rewrote a ``python3``-named
    interpreter to a nonexistent ``tj3`` (``/python`` matches inside
    ``/python3``), so launchd/systemd pointed at a binary that doesn't exist
    while ``launchctl load`` still returned 0 (#340).
    """
    return shutil.which("tj") or str(Path(sys.executable).with_name("tj"))


def _install_launchd(config_path: str) -> str | None:
    tj_path = _resolve_tj_binary()
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
        <string>{tj_path}</string>
        <string>--config</string>
        <string>{config_path}</string>
        <string>serve</string>
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
        console.print(f"[yellow]Daemon plist written to {plist_path} but "
                      f"launchctl load failed.[/yellow]")
        console.print("[dim]Try loading manually:[/dim]")
        console.print(f"  launchctl load {plist_path}")
        console.print("[dim]Or run the server directly:[/dim]")
        console.print("  tj serve &")
        return None
    console.print(
        "  [dim]macOS will show a 'Background Items Added' notification "
        "-- this is normal.[/dim]"
    )
    return f"Daemon installed at {plist_path}"


def _install_systemd(config_path: str) -> str | None:
    tj_path = _resolve_tj_binary()
    service_path = Path.home() / ".config/systemd/user/tokenjam.service"
    service_path.parent.mkdir(parents=True, exist_ok=True)
    service_content = f"""\
[Unit]
Description=TokenJam observability server
After=network.target

[Service]
ExecStart={tj_path} --config {config_path} serve
Restart=on-failure

[Install]
WantedBy=default.target"""
    service_path.write_text(service_content)
    subprocess.run(
        ["systemctl", "--user", "enable", "--now", "tokenjam"],
        check=True,
    )
    return f"Daemon installed at {service_path}"
