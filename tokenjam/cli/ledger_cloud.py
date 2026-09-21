"""`tj init --cloud <key> --org <org>`: connect this machine to TokenJam Cloud
(ledger W5; contracts §6, §9).

One flag on the same command as the rest of onboarding, like `--hooks` and
`--enforce`. It writes the `[cloud]` block into the config `tj init`
resolves, prints the contracts §9 emission list and asks before the first
byte leaves (`--yes` skips the question), then refills the repo context of
sessions ingested without it, matches sessions to commits, and runs one
forwarding pass, in that order, so Cloud fills with history its ledger can
join instead of waiting for the daemon's next tick. `tj init --cloud off` turns forwarding off in place.

The config block carries a live per-org ingest key, which is why it lives
in a file that is never tracked (Critical Rule 20) and why the key is
never echoed back in full.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from rich.markup import escape

from tokenjam.core import cloud_sync
from tokenjam.core.config import CloudConfig, TjConfig, load_config, write_config
from tokenjam.core.repo_context import ensure_install_id, tj_home
from tokenjam.utils.formatting import console
from tokenjam.utils.humanize import display_path

#: TOML section names, escaped once: bare `[cloud]` inside Rich markup reads
#: as a style tag and vanishes from the rendered line.
_CLOUD_SECTION = escape("[cloud]")
_CAPTURE_SECTION = escape("[capture]")

#: The literal that turns forwarding off: `tj init --cloud off`.
CLOUD_OFF = "off"


def print_emission_list(config: TjConfig) -> None:
    """Contracts §9, printed before anything is written or sent."""
    console.print("[bold]What leaves this machine when Cloud forwarding is on[/bold]")
    console.print("  " + ", ".join(cloud_sync.EMISSION_LEAVES) + ".", soft_wrap=True)
    console.print("[bold]What never leaves by default[/bold]")
    console.print("  " + ", ".join(cloud_sync.EMISSION_NEVER_BY_DEFAULT) + ".", soft_wrap=True)
    if config.cloud.forward_content:
        console.print(
            f"  [warn]forward_content is on:[/warn] whatever your [accent]{_CAPTURE_SECTION}[/accent] "
            "toggles keep locally is forwarded too.", soft_wrap=True,
        )
    else:
        console.print(
            "  Content forwarding is off ([accent]forward_content = false[/accent] under "
            f"[accent]{_CLOUD_SECTION}[/accent]); prompts, completions and tool outputs are "
            "stripped before sending even when captured locally.", soft_wrap=True,
        )
    console.print(
        "  Developers are pseudonymous on Cloud by default; per-developer views are "
        "admin-only and hidden below the cohort floor. Subscription OAuth credentials "
        "are never proxied or forwarded.", soft_wrap=True,
    )


def _key_display(key: str) -> str:
    return key[:12] + "…" if len(key) > 12 else key


def _resolve_target(ctx: click.Context, config_path: Path | None) -> tuple[TjConfig, Path] | None:
    from tokenjam.core.config import resolve_config_path

    ctx.ensure_object(dict)
    path = config_path or resolve_config_path(ctx.obj.get("config_path_override"))
    if path is None:
        console.print(
            "[warn]Cloud not connected:[/warn] no tj config found. Run "
            "[accent]tj init[/accent] first, then [accent]tj init --cloud <key> --org <org>[/accent]."
        )
        return None
    return load_config(str(path)), Path(path)


def run_cloud_init(
    ctx: click.Context,
    value: str,
    *,
    org: str | None,
    endpoint: str | None,
    yes: bool,
    config_path: Path | None = None,
) -> bool:
    """The `--cloud` step. Returns True when the config was written."""
    resolved = _resolve_target(ctx, config_path)
    if resolved is None:
        return False
    config, path = resolved

    console.print()
    console.print("[bold]TokenJam Cloud[/bold]")
    if value.strip().lower() == CLOUD_OFF:
        return _turn_off(config, path)

    try:
        key = cloud_sync.parse_ingest_key(value)
    except ValueError as exc:
        raise click.UsageError(f"--cloud: {exc}") from exc
    org_id = (org or "").strip() or config.cloud.org_id.strip()
    if not org_id:
        raise click.UsageError(
            "--cloud needs the organization the key belongs to: "
            "tj init --cloud <key> --org <org_id> (both are on Cloud's Connect screen)."
        )
    target_endpoint = (endpoint or "").strip() or config.cloud.endpoint or CloudConfig.endpoint

    if is_git_tracked(path):
        console.print(
            f"[warn]Cloud not connected:[/warn] [accent]{display_path(path)}[/accent] is tracked "
            "by git, and the ingest key is a live secret. Add the file to .gitignore (or point "
            "tj at an untracked config with [accent]--config[/accent]) and re-run.",
            soft_wrap=True,
        )
        return False

    print_emission_list(config)
    console.print(
        f"  Forwarding to [accent]{target_endpoint}[/accent] as org [bold]{org_id}[/bold] "
        f"with key {_key_display(key)}.", soft_wrap=True,
    )
    if not yes:
        try:
            agreed = click.confirm("Connect and forward?", default=False)
        except click.Abort:
            agreed = False
        if not agreed:
            console.print("  Nothing written. Re-run with [accent]--yes[/accent] to skip the question.")
            return False

    config.cloud = CloudConfig(
        enabled=True,
        endpoint=target_endpoint,
        org_id=org_id,
        ingest_key=key,
        forward_content=config.cloud.forward_content,
    )
    write_config(config, path)
    cloud_sync.reset_state(org_id=org_id, db_path=cloud_sync.storage_identity(config))
    console.print(f"[ok]✓[/ok] {_CLOUD_SECTION} written to [accent]{display_path(path)}[/accent]")
    # The §3 install id rides every forwarded session and spans resource;
    # Cloud counts connected developers off it. Minted here if `tj init`
    # never got to, and its absence is said out loud rather than discovered
    # as a zero on Cloud's Connect screen (issue #770, fix 2).
    if not ensure_install_id():
        console.print(
            f"[warn]No install id could be written under {display_path(tj_home())};[/warn] "
            "history is forwarded without one and Cloud will not count this machine "
            "as a connected developer until it can be.", soft_wrap=True,
        )

    _initial_push(config, path)
    return True


def is_git_tracked(path: Path) -> bool:
    """Whether git tracks `path` (Critical Rule 20: a live key never lands
    in a committable file). Read-only, 2s, `check=False`; anything short of
    a clean "yes" is "no", so a machine without git is never blocked."""
    try:
        target = path.resolve()
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", target.name],
            cwd=target.parent, capture_output=True, text=True, timeout=2, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _turn_off(config: TjConfig, path: Path) -> bool:
    if not config.cloud.configured:
        console.print("  Cloud forwarding was never configured here; nothing to turn off.")
        return False
    if not config.cloud.enabled:
        console.print("  Cloud forwarding is already off.")
        return False
    config.cloud.enabled = False
    write_config(config, path)
    console.print(
        f"[ok]✓[/ok] Cloud forwarding off ([accent]enabled = false[/accent] under "
        f"{_CLOUD_SECTION} in [accent]{display_path(path)}[/accent]). The key stays so "
        "[accent]tj init --cloud <key>[/accent] turns it back on."
    )
    _restart_daemon_if_running(path, reason="stop forwarding")
    return True


#: The three steps of the first push, in the order they must run and in the
#: words the spinner shows for each (issue #770, fixes 1 and 5).
PREP_REFILL_MSG = "Filling repo context on existing sessions…"
PREP_MATCH_MSG = "Matching sessions to commits…"
PUSH_MSG = "Forwarding history to Cloud…"


@dataclass
class PushPrep:
    """What the two steps before the push found, for the summary line."""
    refill: Any = None   # transcript_sync.RefillReport | None
    match: Any = None    # shipped.MatchResult | None
    error: str | None = None


def _prepare_then_push(db: Any, config: TjConfig) -> tuple[PushPrep, cloud_sync.SyncReport]:
    """Refill, match, forward, in that order, under one status line.

    The order is the whole point (issue #770, fix 1): on the first real run
    every local session had NULL repo context (the daemon that ingested them
    ran a build that never derived it) so `session_commits` was empty and
    the push carried spans and sessions but zero commits. Each step here is
    the same function the daemon runs on its schedule; the push comes last
    so what leaves is what the ledger can join on. A refill or match
    failure is recorded and the push still runs: stale context is one pass
    behind, an unsent history is not.
    """
    from tokenjam.cli.backfill_progress import phase_status
    from tokenjam.core.shipped import match_sessions_to_commits
    from tokenjam.core.transcript_sync import refill_session_context

    prep = PushPrep()
    with phase_status(PREP_REFILL_MSG, console=console) as update:
        try:
            prep.refill = refill_session_context(db, config=config)
            update(PREP_MATCH_MSG)
            prep.match = match_sessions_to_commits(db, config)
        except Exception as exc:  # noqa: BLE001 - a fatal propagates, the rest is reported
            from tokenjam.core.db import is_fatal_db_error

            if is_fatal_db_error(exc):
                raise
            prep.error = str(exc)
        update(PUSH_MSG)
        report = cloud_sync.run_sync(db, config)
    return prep, report


def _print_prep(prep: PushPrep) -> None:
    parts: list[str] = []
    if prep.refill is not None and prep.refill.candidates:
        parts.append(
            f"repo context filled on {prep.refill.filled} of {prep.refill.candidates} "
            "session(s) that had none"
        )
    if prep.match is not None and prep.match.rows_written:
        parts.append(f"{prep.match.rows_written} new session-commit join(s)")
    if parts:
        console.print("  Before sending: " + "; ".join(parts) + ".")
    if prep.error:
        console.print(
            f"  Repo context or commit matching could not run ({prep.error}); "
            "the daemon retries on its next pass."
        )


def _initial_push(config: TjConfig, path: Path) -> None:
    """Refill context, match commits, then one forwarding pass, so Cloud
    fills with history it can join. Needs the DuckDB write lock, so a
    running daemon is stopped for it and restarted after (the same dance
    every onboard DB write does).

    Output discipline (issue #770, fix 5): the status line stops and every
    result line prints BEFORE the daemon restart starts, so the user reads
    what was sent while the restart runs, not a spinner until it ends.
    """
    from tokenjam.cli.cmd_onboard import _stop_serve_for_db_write
    from tokenjam.core.db import open_db

    stopped = _stop_serve_for_db_write()
    report: cloud_sync.SyncReport | None = None
    prep: PushPrep | None = None
    try:
        db = open_db(config.storage)
    except Exception as exc:  # noqa: BLE001 - a locked or unreadable DB is reported, not raised
        console.print(
            f"  Initial push skipped ({exc}); the daemon forwards on its next pass."
        )
    else:
        try:
            prep, report = _prepare_then_push(db, config)
        except Exception as exc:  # noqa: BLE001 - classified: a fatal is recovered, the rest reported
            from tokenjam.core.db import handle_if_fatal

            handle_if_fatal(exc, what="cloud initial push")
            console.print(f"  Initial push failed ({exc}); the daemon retries on its next pass.")
        finally:
            from tokenjam.core.db import recover_if_fatal_noted

            recover_if_fatal_noted(what="cloud initial push")
            try:
                db.close()
            except Exception:
                pass
    # The status line is gone by here (its `with` block ended inside
    # `_prepare_then_push`); everything below lands on a clean line, and
    # all of it lands before the restart below is even attempted.
    if prep is not None:
        _print_prep(prep)
    if report is not None:
        _print_report(report)
    if stopped:
        _restart_daemon_if_running(path, reason="pick up the cloud block", already_stopped=True)
    else:
        console.print(
            "  The daemon forwards new sessions every "
            f"{cloud_sync.SYNC_INTERVAL_MINUTES} minutes while [accent]tj serve[/accent] runs."
        )


def _print_report(report: cloud_sync.SyncReport) -> None:
    if report.skipped_reason:
        console.print(f"  Initial push skipped: {report.skipped_reason}")
        return
    counts = (f"{report.spans_sent} spans, {report.sessions_sent} sessions, "
              f"{report.commits_sent} commits")
    if report.stopped == cloud_sync.Outcome.UNAUTHORIZED:
        console.print(
            f"[warn]Cloud rejected the key[/warn] ({counts} sent before it did). "
            "Forwarding is disabled until [accent]tj init --cloud <key> --org <org>[/accent] "
            "runs with a current key.", soft_wrap=True,
        )
        return
    if report.stopped == cloud_sync.Outcome.UNAVAILABLE:
        console.print(
            f"  Cloud was unreachable part-way ({counts} sent); the daemon resumes "
            "from where it stopped."
        )
        return
    tail = f" ({report.rejected} refused by the receiver)" if report.rejected else ""
    if report.install_id_missing:
        tail += " (without an install id: Cloud will not count this machine as a developer)"
    console.print(f"[ok]✓[/ok] Forwarded {counts}{tail}.")


def _restart_daemon_if_running(path: Path, *, reason: str, already_stopped: bool = False) -> None:
    from tokenjam.cli.cmd_onboard import _daemon_already_running, _restart_tj_server

    if not already_stopped and not _daemon_already_running():
        return
    msg = _restart_tj_server(str(path), False, reason="db_update")
    console.print(f"  Daemon: {msg} ({reason}).")


_NOT_CONNECTED = (
    "Cloud: not connected (forward to TokenJam Cloud with: tj init --cloud <key> --org <org>)"
)


def cloud_summary_line(config: TjConfig | None) -> str:
    """One line for the `tj init` end-of-run summary and `tj status`."""
    if config is None:
        return _NOT_CONNECTED
    line = cloud_sync.status_line(config)
    if line is None:
        return _NOT_CONNECTED
    return line


__all__ = [
    "CLOUD_OFF",
    "cloud_summary_line",
    "is_git_tracked",
    "print_emission_list",
    "run_cloud_init",
]
