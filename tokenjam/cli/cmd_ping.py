"""
``tj ping`` — emit one clearly-labeled test span through the real SDK exporter
path so an SDK user can prove their wiring without running a whole agent (#80).

Modeled on Sentry's wizard-generated test event and OTel's console-exporter
proof-of-life: the span is emitted through the same ``@watch()`` / provider-patch
machinery real agents use, and a local in-process exporter prints a "the patch IS
intercepting" confirmation that fires even when the daemon is down (the span is
still captured locally). The command then reports where the span was delivered —
a running ``tj serve`` over HTTP, or the local DuckDB.

Exit code contract: 0 means delivery was *confirmed*, not merely attempted. In
HTTP mode the span is POSTed asynchronously via ``BatchSpanProcessor``, and
``TjHttpExporter.export()`` swallows network/auth failures into a logged
``SpanExportResult.FAILURE`` rather than raising — so ``force_flush()`` returning
without an exception does not mean the daemon actually stored the span. To make
the exit code trustworthy for scripts/CI, HTTP-mode delivery is confirmed by
polling the daemon's read API for the just-emitted span (reusing
``onboard_verify``'s ``open_read_backend``/``poll_for_first_span``) before
exiting 0. Direct (local DuckDB) mode writes synchronously inside
``force_flush()`` (``TjSpanExporter.export()`` calls ``pipeline.process()``
in-line), so a clean flush already means the span landed in the DB.
"""
from __future__ import annotations

import json as _json

import click

from tokenjam.utils.formatting import console

# Clearly-labeled test identity so a ping span is never mistaken for real
# telemetry in the dashboard or analyzers.
PING_AGENT_ID = "tj-ping"
PING_MODEL = "tj-ping-test"

# How long to wait for the daemon to confirm it stored the ping span before
# giving up. Short: the span was already POSTed by the time we start polling,
# so a healthy daemon confirms almost immediately.
PING_CONFIRM_TIMEOUT_S = 8.0
PING_CONFIRM_INTERVAL_S = 1.0


class _ProofExporter:
    """Minimal in-process SpanExporter that records exported spans so we can
    print a local proof-of-life. Attached via a SimpleSpanProcessor so it fires
    synchronously on span end, independent of network / DB delivery."""

    def __init__(self) -> None:
        self.captured: list[str] = []

    def export(self, spans):  # type: ignore[no-untyped-def]
        from opentelemetry.sdk.trace.export import SpanExportResult

        for span in spans:
            self.captured.append(span.name)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:  # pragma: no cover - trivial
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # pragma: no cover
        return True


@click.command("ping")
@click.option("--agent", "agent_id", default=PING_AGENT_ID, show_default=True,
              help="agent_id to stamp on the test span.")
@click.option("--json", "output_json", is_flag=True,
              help="Emit a machine-readable result instead of prose.")
@click.pass_context
def cmd_ping(ctx: click.Context, agent_id: str, output_json: bool) -> None:
    """Emit a labeled test span to prove SDK instrumentation is wired up.

    Exits 0 only once delivery is *confirmed* — HTTP mode polls the daemon's
    read API for the span before exiting; exit 0 in HTTP mode never means
    just "attempted". See the module docstring for why that distinction
    matters.
    """
    config = ctx.obj["config"]

    from opentelemetry import trace as trace_api
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    from tokenjam.sdk import bootstrap
    from tokenjam.sdk.agent import AgentSession, record_llm_call
    from tokenjam.utils.time_parse import utcnow

    bootstrap.ensure_initialised()

    # Attach the local proof-of-life exporter to whichever provider bootstrap
    # set as global. If bootstrap failed entirely there is no real provider, so
    # the interception can't be proven — reported honestly below.
    proof = _ProofExporter()
    provider = trace_api.get_tracer_provider()
    proof_attached = False
    if hasattr(provider, "add_span_processor"):
        provider.add_span_processor(SimpleSpanProcessor(proof))
        proof_attached = True

    # Captured before emission so the confirmation poll below only matches
    # this ping's span, not a stale one from an earlier run.
    since = utcnow()

    # Emit one labeled test span through the real @watch()/record_llm_call path.
    with AgentSession(agent_id=agent_id):
        record_llm_call(
            model=PING_MODEL,
            provider="anthropic",
            input_tokens=1,
            output_tokens=1,
        )

    if hasattr(provider, "force_flush"):
        try:
            provider.force_flush()
        except Exception:  # noqa: BLE001 — best-effort delivery
            pass

    mode = bootstrap.get_mode()
    intercepted = proof_attached and bool(proof.captured)

    confirmed = False
    confirm_error: str | None = None
    if mode == "direct":
        # Written synchronously above (see module docstring) — no read-back
        # needed, and re-opening the DB here would contend with the write
        # connection this same process already holds.
        confirmed = True
    elif mode == "http":
        confirmed, confirm_error = _confirm_delivery(config, agent_id, since)

    if output_json:
        console.print_json(_json.dumps({
            "intercepted": intercepted,
            "delivery_mode": mode,
            "confirmed": confirmed,
            "agent_id": agent_id,
            "model": PING_MODEL,
        }))
        _exit_for_confirmation(ctx, confirmed)
        return

    if intercepted:
        console.print(
            "[green]✓[/green] tj intercepted a test span "
            f"([bold]llm_call[/bold] model={PING_MODEL} agent={agent_id}) "
            "— the SDK export path is working."
        )
    else:
        console.print(
            "[yellow]⚠[/yellow] Could not confirm local interception "
            "(the SDK failed to initialise a tracer)."
        )

    base_url = f"http://{config.api.host}:{config.api.port}"
    if mode == "http" and confirmed:
        console.print(
            f"[green]✓[/green] Delivered to [bold]tj serve[/bold] at {base_url} "
            "(confirmed received). Run [bold]tj status[/bold] to see it."
        )
    elif mode == "http":
        reason = f" ({confirm_error})" if confirm_error else ""
        console.print(
            f"[yellow]⚠[/yellow] Span emitted to [bold]tj serve[/bold] at {base_url} "
            f"but not confirmed received{reason} — is [bold]tj serve[/bold] healthy? "
            "Run [bold]tj status[/bold] to check."
        )
    elif mode == "direct":
        console.print(
            f"[green]✓[/green] Written to the local database "
            f"([dim]{config.storage.path}[/dim]). Run [bold]tj status[/bold] to see it."
        )
    else:
        console.print(
            "[yellow]⚠[/yellow] Could not reach [bold]tj serve[/bold] or open the "
            "local database, so the span was not stored. Start [bold]tj serve[/bold] "
            "and re-run [bold]tj ping[/bold]."
        )

    _exit_for_confirmation(ctx, confirmed)


def _confirm_delivery(
    config: object, agent_id: str, since: object
) -> tuple[bool, str | None]:
    """Poll the daemon's read API for the just-emitted ping span.

    Returns ``(confirmed, error)``. ``error`` is set only when the read path
    itself couldn't be resolved (daemon unreachable and the local DB is
    locked/unavailable) — a clean "didn't arrive within the timeout" is
    ``(False, None)``.
    """
    from tokenjam.core.onboard_verify import open_read_backend, poll_for_first_span

    backend, _mode, error = open_read_backend(config)
    if backend is None:
        return False, error

    try:
        result = poll_for_first_span(
            backend,
            since,
            agent_id=agent_id,
            timeout_s=PING_CONFIRM_TIMEOUT_S,
            interval_s=PING_CONFIRM_INTERVAL_S,
        )
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass

    if result.error:
        return False, result.error
    return result.confirmed, None


def _exit_for_confirmation(ctx: click.Context, confirmed: bool) -> None:
    """Exit non-zero unless delivery was confirmed, so scripts/CI can gate on it."""
    ctx.exit(0 if confirmed else 1)
