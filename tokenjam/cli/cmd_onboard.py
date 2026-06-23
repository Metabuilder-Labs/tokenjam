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
from tokenjam.core.config import find_config_file
from tokenjam.utils.formatting import console


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
@click.pass_context
def cmd_onboard(ctx: click.Context, claude_code: bool, codex: bool, budget: float | None,
                install_daemon: bool, no_daemon: bool, force: bool,
                reconfigure: bool, plan: str | None) -> None:
    """Interactive setup wizard for tj."""
    # Branded welcome moment (#240) — shown once at the top of every onboard
    # flow (plain / --claude-code / --codex) before any prompt or config check.
    print_welcome_banner()
    if claude_code:
        _onboard_claude_code(ctx, budget, no_daemon, force, reconfigure, plan)
        return
    if codex:
        _onboard_codex(ctx, budget, no_daemon, force, reconfigure, plan)
        return
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
    console.print("[dim]     from tokenjam.sdk import watch[/dim]")
    console.print("[dim]     from tokenjam.sdk.integrations.anthropic import patch_anthropic[/dim]")
    console.print()
    console.print("[dim]     patch_anthropic()[/dim]")
    console.print()
    console.print('[dim]     @watch(agent_id="my-agent")[/dim]')
    console.print("[dim]     def run(task):[/dim]")
    console.print("[dim]         ...[/dim]")
    console.print()
    console.print("  2. Run your agent \u2014 spans are recorded automatically")
    console.print()
    console.print("  3. View telemetry:")
    console.print("[dim]     tj status          [/dim]# agent overview")
    console.print("[dim]     tj traces          [/dim]# span history")
    console.print("[dim]     tj serve           [/dim]# web UI at http://127.0.0.1:7391/")
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
    console.print("  [bold]tj serve[/bold]       [dim]open Lens (web UI) at http://127.0.0.1:7391/[/dim]")
    console.print()


def _onboard_claude_code(
    ctx: click.Context,
    budget: float | None,
    no_daemon: bool,
    force: bool,
    reconfigure: bool = False,
    plan_override: str | None = None,
) -> None:
    """Configure Claude Code to send telemetry to tj."""
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

    # Plan-first (#240): resolve the plan tier before prompting for the daily
    # budget — "How do you pay?" is the more important, more natural opener.
    if global_config_path.exists() and not force:
        config = load_config(str(global_config_path))
        if agent_id not in config.agents:
            config.agents[agent_id] = AgentConfig()

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
        console.print(f"  tj config updated: {config_path}")
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
        agents = {agent_id: AgentConfig(budget=BudgetConfig(daily_usd=daily_usd))}
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
        console.print(f"  tj config written to: {config_path}")
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

    # --- Register MCP server with Claude Code ---
    if shutil.which("claude"):
        subprocess.run(
            ["claude", "mcp", "add", "tj", "--scope", "user", "--", "tj", "mcp"],
            capture_output=True,
        )

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
    global_settings_path.write_text(json_mod.dumps(global_settings, indent=2) + "\n")

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

    project_env: dict = project_settings.get("env", {})
    project_env["OTEL_RESOURCE_ATTRIBUTES"] = f"service.name={agent_id}"
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
    marker = "# tj harness observability"
    zshrc_text = zshrc.read_text()
    new_block = (
        f"\n{marker}\n"
        f"export CLAUDE_CODE_ENABLE_TELEMETRY=1\n"
        f"export OTEL_LOGS_EXPORTER=otlp\n"
        f"export OTEL_EXPORTER_OTLP_PROTOCOL=http/json\n"
        f"export OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:{port}\n"
        f'export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer {secret}"\n'
    )
    if marker not in zshrc_text:
        with zshrc.open("a") as f:
            f.write(new_block)
    else:
        # Marker already present — replace the entire block to keep the secret in sync.
        import re as _re
        updated = _re.sub(
            r"# tj harness observability\n(?:export [^\n]+\n)*",
            new_block.lstrip("\n"),
            zshrc_text,
        )
        zshrc.write_text(updated)

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
    console.print(f"  Global settings:     {global_settings_path}")
    console.print(f"  Project settings:    {project_settings_path}")
    console.print("  Shell env:           ~/.zshrc (harness-compatible endpoint)")
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
    # Lead with the wins that need no restart, THEN the restart note (#240).
    _print_next_steps_nudge(has_data=backfill_has_data, days=backfill_days)
    if not want_daemon:
        _warn_manual_serve_restart(stopped_for_db=stopped_for_db, no_daemon=True)
        console.print("[dim]Start the server:[/dim]  tj serve")
    _print_restart_banner("Claude Code")
    console.print(f"[dim]After restarting, run:[/dim]  tj status --agent {agent_id}")


def _onboard_codex(
    ctx: click.Context,
    budget: float | None,
    no_daemon: bool,
    force: bool,
    reconfigure: bool = False,
    plan_override: str | None = None,
) -> None:
    """Configure Codex CLI to send telemetry to tj."""
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
        console.print(f"  tj config updated: {config_path}")
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
        console.print(f"  tj config written to: {config_path}")
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

    already_has_otel = "otel" in existing_codex
    already_has_mcp = bool(existing_codex.get("mcp_servers", {}).get("tj"))
    if already_has_otel and already_has_mcp and not force:
        # Use plain print() for messages containing TOML section headers like
        # [otel] — Rich treats square brackets as markup tags and would strip
        # them, leaving the message unintelligible ("already has  and ").
        click.echo(
            "~/.codex/config.toml already has [otel] and [mcp_servers.tj] sections."
        )
        click.echo("Use --force to overwrite, or add manually:")
        click.echo("")
        _print_codex_otel_block(port, secret, agent_id)
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
        return

    otel_block = _codex_otel_toml_block(port, secret, agent_id)
    mcp_block = _codex_mcp_toml_block()

    # Build the new file content by replacing/appending each managed section.
    base_content = existing_content
    if force:
        # Fully wipe previous OTEL sections so nested tables don't duplicate.
        base_content = _codex_strip_otel_sections(base_content)
    new_content = _codex_apply_block(
        base_content, r"\[otel\]", already_has_otel, otel_block, force,
    )
    # Re-parse after the otel update so section detection is accurate.
    try:
        existing_after_otel = tomllib.loads(new_content)
    except Exception:
        existing_after_otel = existing_codex
    existing_mcp = existing_after_otel.get("mcp_servers", {}).get("tj")
    new_content = _codex_apply_block(
        new_content,
        r"\[mcp_servers\.tj\]",
        bool(existing_mcp),
        mcp_block,
        force,
    )
    codex_config_path.write_text(new_content)

    # --- Try registering via the Codex CLI as well (best-effort) ---
    if shutil.which("codex"):
        subprocess.run(
            ["codex", "mcp", "add", "tj", "--", "tj", "mcp"],
            capture_output=True,
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
    console.print(f"  Codex config:        {codex_config_path}")
    console.print(f"  TokenJam config:     {config_path}")
    if budget and budget > 0:
        console.print(f"  Daily budget:        ${budget:.2f}")
    console.print(f"  OTLP endpoint:       http://127.0.0.1:{port}/v1/logs")
    if secret:
        console.print(f"  Ingest secret:       {secret[:8]}...")
    console.print("  MCP server:          tj mcp (registered in Codex config)")
    # Lead with the wins that need no restart, THEN the restart note (#240).
    # Codex onboarding doesn't backfill, so there's no "already loaded" claim.
    _print_next_steps_nudge(has_data=False)
    if not want_daemon:
        _warn_manual_serve_restart(stopped_for_db=stopped_for_db, no_daemon=True)
        console.print("[dim]Start the server:[/dim]  tj serve")
    console.print(
        "[dim]Codex can now call TokenJam tools (open_dashboard, get_status, etc.) directly.[/dim]"
    )
    _print_restart_banner("Codex")
    console.print("[dim]After restarting, run:[/dim]  tj traces")


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
    """Return the [mcp_servers.tj] TOML block for ~/.codex/config.toml."""
    return (
        "[mcp_servers.tj]\n"
        "# Managed by tj — gives Codex access to TokenJam observability tools\n"
        'command = "tj"\n'
        'args = ["mcp"]\n'
    )


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


def _install_launchd(config_path: str) -> str | None:
    tj_path = shutil.which("tj") or sys.executable.replace("/python", "/tj").replace("/python3", "/tj")
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
    tj_path = shutil.which("tj") or sys.executable.replace("/python", "/tj").replace("/python3", "/tj")
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
