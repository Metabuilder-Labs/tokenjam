from __future__ import annotations

import json as json_mod
import platform
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

import click

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
@click.option("--project", "project_override", default=None,
              help="Project name to group this repo under in the dashboard "
                   "(OTel service.namespace — e.g. all Aquanodeio/* repos under "
                   "'aquanode'). Defaults to the git org. Used with --claude-code.")
@click.pass_context
def cmd_onboard(ctx: click.Context, claude_code: bool, codex: bool, budget: float | None,
                install_daemon: bool, no_daemon: bool, force: bool,
                reconfigure: bool, plan: str | None, project_override: str | None) -> None:
    """Interactive setup wizard for tj."""
    if claude_code:
        _onboard_claude_code(ctx, budget, no_daemon, force, reconfigure, plan, project_override)
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

    if budget is None:
        budget = click.prompt(
            "Daily budget in USD per agent (0 = no limit, default 0)",
            type=float, default=0.0, show_default=False,
        )

    # Plan tier (#4): the plain path now honors `--plan` and prompts for it
    # interactively, instead of silently ignoring `--plan` and never writing a
    # `[budget.<provider>] plan`. This is a Claude-first tool, so the interactive
    # prompt offers the Anthropic tiers; an OpenAI-only `--plan` (plus/team/
    # enterprise) is routed to its provider section. (The `--claude-code` /
    # `--codex` flows still own the global integration configs.)
    plan_tier = plan
    if plan_tier is None and sys.stdin.isatty():
        plan_tier = _prompt_plan("Claude", _ANTHROPIC_PLAN_CHOICES)
    plan_provider = (
        "openai" if plan_tier in ("plus", "team", "enterprise") else "anthropic"
    )
    plan_section = (
        f'\n[budget.{plan_provider}]\nplan = "{plan_tier}"\n' if plan_tier else ""
    )

    ingest_secret = secrets.token_hex(32)

    want_daemon = not no_daemon

    config_path = Path(".tj/config.toml")
    config_path.parent.mkdir(parents=True, exist_ok=True)

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

    daemon_msg = None
    if want_daemon:
        daemon_msg = _install_daemon(str(config_path.resolve()))

    # Output
    console.print()
    console.print("[green]\u2713[/green] Config written to [bold].tj/config.toml[/bold]")
    console.print(f"[green]\u2713[/green] Ingest secret generated: "
                  f"[dim]{ingest_secret[:8]}...[/dim]")
    if budget and budget > 0:
        console.print(f"[green]\u2713[/green] Default daily budget: "
                      f"[bold]${budget:.2f}[/bold] per agent")
    if plan_tier:
        console.print(f"[green]\u2713[/green] Plan tier: "
                      f"[bold]{plan_tier}[/bold] (written to [budget.{plan_provider}])")
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


def _onboard_claude_code(
    ctx: click.Context,
    budget: float | None,
    no_daemon: bool,
    force: bool,
    reconfigure: bool = False,
    plan_override: str | None = None,
    project_override: str | None = None,
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

    if budget is None:
        budget = click.prompt(
            "Daily budget in USD (0 = no limit, default 0)",
            type=float, default=0.0, show_default=False,
        )

    if global_config_path.exists() and not force:
        config = load_config(str(global_config_path))
        if agent_id not in config.agents:
            config.agents[agent_id] = AgentConfig()
        if budget and budget > 0:
            config.agents[agent_id].budget.daily_usd = budget
        # Server-side project mapping so already-running sessions group by
        # project without restarting the agent (see AgentConfig.project).
        config.agents[agent_id].project = namespace

        existing_plan = (
            config.budgets["anthropic"].plan
            if "anthropic" in config.budgets else None
        )
        # Prompt for plan tier when:
        #   - this is a fresh onboard for this agent (no existing plan), or
        #   - the user passed --reconfigure to explicitly re-prompt
        # `plan_override` (from --plan CLI flag) bypasses the prompt entirely.
        if existing_plan is None or reconfigure or plan_override:
            if plan_override:
                plan = plan_override
            else:
                plan = _prompt_plan("Claude", _ANTHROPIC_PLAN_CHOICES, current=existing_plan)
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
        config_path = global_config_path
        write_config(config, config_path)
        console.print(f"  tj config updated: {config_path}")
    else:
        ingest_secret = secrets.token_hex(32)
        daily_usd = budget if budget and budget > 0 else None
        agents = {agent_id: AgentConfig(budget=BudgetConfig(daily_usd=daily_usd), project=namespace)}
        if plan_override:
            plan = plan_override
        else:
            plan = _prompt_plan("Claude", _ANTHROPIC_PLAN_CHOICES)
        usd: float | None = None  # type: ignore[no-redef]
        if plan == "api" and not plan_override:
            ceiling = click.prompt(
                "Monthly Anthropic API spend ceiling in USD (0 = no limit)",
                type=float, default=0.0, show_default=False,
            )
            if ceiling > 0:
                usd = ceiling
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

    # --- Per-terminal naming: install the `claude` shell wrapper ---
    # Tags each terminal with a distinct service.instance.id so concurrent
    # Claude Code sessions render as separate dashboard tiles, without the user
    # hand-editing their shell rc. Idempotent across re-onboards.
    wrapper_files = _install_claude_wrapper()

    want_daemon = not no_daemon
    if want_daemon:
        if not force and _daemon_already_running():
            console.print("  Daemon:              already running (skipped reinstall)")
        else:
            console.print("  Daemon:              installing...")
            _install_daemon(str(config_path.resolve()))

    # --- Backfill existing Claude Code session logs ---
    # First-time users have no history yet; this populates the DB so that
    # `tj optimize` returns useful output on first run. Idempotent — safe to
    # re-run; will skip already-ingested spans.
    backfill_msg: str | None = None
    try:
        from tokenjam.core.backfill import (
            CLAUDE_CODE_PROJECTS_ROOT, ingest_claude_code,
        )
        if CLAUDE_CODE_PROJECTS_ROOT.exists():
            from tokenjam.core.db import open_db
            try:
                db = open_db(config.storage)
                result = ingest_claude_code(db)
                db.close()
                if result.sessions_ingested > 0:
                    days = None
                    if result.earliest and result.latest:
                        days = (result.latest - result.earliest).days
                    pieces = [f"{result.sessions_ingested} sessions"]
                    if days is not None:
                        pieces.append(f"over {days} day{'s' if days != 1 else ''}")
                    pieces.append(f"${result.total_cost_usd:.0f} total")
                    backfill_msg = ", ".join(pieces)
                elif result.sessions_seen > 0:
                    backfill_msg = "history already up to date"
            except Exception as exc:
                # Friendly message for the most common case: daemon holds
                # the DB write lock. Backfill is a writer and can't share
                # the lock; raw DuckDB IO error is unhelpful (#71 finding 2).
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

    console.print()
    console.print("[bold green]Claude Code observability configured.[/bold green]")
    console.print(f"  Global settings:     {global_settings_path}")
    console.print(f"  Project settings:    {project_settings_path}")
    if removed_resource_attr:
        console.print(
            "  [yellow]Removed a hardcoded OTEL_RESOURCE_ATTRIBUTES from project "
            "settings[/yellow] (the claude wrapper now sets it per terminal)."
        )
    console.print("  Shell env:           ~/.zshrc (harness-compatible endpoint)")
    if wrapper_files:
        console.print(
            f"  claude wrapper:      {', '.join(wrapper_files)} "
            "(per-terminal naming)"
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
    if not want_daemon:
        console.print("[dim]Start the server:[/dim]  tj serve")
    console.print("[dim]Restart Claude Code for settings to take effect.[/dim]")
    console.print(
        "[dim]Open a new terminal, then launch with[/dim]  claude  "
        "[dim](each terminal becomes its own dashboard tile;[/dim] "
        "claude --as <name> [dim]to label it).[/dim]"
    )
    console.print(f"[dim]Then run:[/dim]  tj status --agent {agent_id}")


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

    if budget is None:
        budget = click.prompt(
            "Daily budget in USD (0 = no limit, default 0)",
            type=float, default=0.0, show_default=False,
        )

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

    if config_path.exists():
        config = load_config(str(config_path))
        if agent_id not in config.agents:
            config.agents[agent_id] = AgentConfig()
        if budget and budget > 0:
            config.agents[agent_id].budget.daily_usd = budget

        existing_plan = (
            config.budgets["openai"].plan
            if "openai" in config.budgets else None
        )
        if existing_plan is None or reconfigure or plan_override:
            if plan_override:
                plan = plan_override
            else:
                plan = _prompt_plan("OpenAI / Codex", _OPENAI_PLAN_CHOICES, current=existing_plan)
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
        write_config(config, config_path)
        console.print(f"  tj config updated: {config_path}")
    else:
        ingest_secret = secrets.token_hex(32)
        daily_usd = budget if budget and budget > 0 else None
        agents = {agent_id: AgentConfig(budget=BudgetConfig(daily_usd=daily_usd))}
        if plan_override:
            plan = plan_override
        else:
            plan = _prompt_plan("OpenAI / Codex", _OPENAI_PLAN_CHOICES)
        usd: float | None = None  # type: ignore[no-redef]
        if plan == "api" and not plan_override:
            ceiling = click.prompt(
                "Monthly OpenAI API spend ceiling in USD (0 = no limit)",
                type=float, default=0.0, show_default=False,
            )
            if ceiling > 0:
                usd = ceiling
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

    port = config.api.port
    secret = config.security.ingest_secret
    secret_rotated = bool(previous_secret) and previous_secret != secret

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

    # --- If ingest secret rotated, restart running server first ---
    restart_msg: str | None = None
    if secret_rotated:
        restart_msg = _restart_tj_server_for_secret_rotation(
            str(config_path.resolve()), no_daemon=no_daemon
        )

    # --- Install daemon if requested ---
    want_daemon = not no_daemon
    if want_daemon and not secret_rotated:
        console.print("  Daemon:              auto-installing (use --no-daemon to skip)")
        _install_daemon(str(config_path.resolve()))

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
    if restart_msg:
        console.print(f"  Server restart:      {restart_msg}")
    console.print()
    if not want_daemon:
        console.print("[dim]Start the server:[/dim]  tj serve")
    console.print(
        "[dim]Codex can now call TokenJam tools (open_dashboard, get_status, etc.) directly.[/dim]"
    )
    console.print("[dim]Then run:[/dim]  tj traces")


def _restart_tj_server_for_secret_rotation(config_path: str, no_daemon: bool) -> str:
    """Restart running tj serve to pick up a rotated ingest secret.

    We stop first to ensure any manually-started server process with stale config
    is terminated, then (when daemon mode is enabled) start it again.
    """
    stopped = False
    try:
        # Best-effort: stop launchd/systemd daemon or background serve process.
        stop_result = subprocess.run(
            ["tj", "stop"], capture_output=True, text=True, timeout=10
        )
        stopped = stop_result.returncode == 0
    except Exception:
        stopped = False

    if no_daemon:
        if stopped:
            return "stopped stale server; run `tj serve` to start with new secret"
        return "could not auto-restart in --no-daemon mode; run `tj serve` manually"

    daemon_msg = _install_daemon(config_path)
    if daemon_msg:
        return "restarted to pick up new ingest secret"
    if stopped:
        return "stopped stale server; please run `tj serve` manually"
    return "restart attempted; verify with `tj status`"


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
