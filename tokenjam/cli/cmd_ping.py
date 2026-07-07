"""
``tj ping`` — emit one clearly-labeled test span through the real SDK exporter
path so an SDK user can prove their wiring without running a whole agent (#80).

Modeled on Sentry's wizard-generated test event and OTel's console-exporter
proof-of-life: the span is emitted through the same ``@watch()`` / provider-patch
machinery real agents use, and a local in-process exporter prints a "the patch IS
intercepting" confirmation that fires even when the daemon is down (the span is
still captured locally). The command then reports where the span was delivered —
a running ``tj serve`` over HTTP, or the local DuckDB.
"""
from __future__ import annotations

import json as _json

import click

from tokenjam.utils.formatting import console

# Clearly-labeled test identity so a ping span is never mistaken for real
# telemetry in the dashboard or analyzers.
PING_AGENT_ID = "tj-ping"
PING_MODEL = "tj-ping-test"


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
    """Emit a labeled test span to prove SDK instrumentation is wired up."""
    config = ctx.obj["config"]

    from opentelemetry import trace as trace_api
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    from tokenjam.sdk import bootstrap
    from tokenjam.sdk.agent import AgentSession, record_llm_call

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

    if output_json:
        console.print_json(_json.dumps({
            "intercepted": intercepted,
            "delivery_mode": mode,
            "agent_id": agent_id,
            "model": PING_MODEL,
        }))
        _exit_for_mode(ctx, mode)
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
    if mode == "http":
        console.print(
            f"[green]✓[/green] Delivered to [bold]tj serve[/bold] at {base_url}. "
            "Run [bold]tj status[/bold] to see it."
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

    _exit_for_mode(ctx, mode)


def _exit_for_mode(ctx: click.Context, mode: str) -> None:
    """Exit non-zero when the span was not stored, so scripts/CI can gate on it."""
    if mode in ("http", "direct"):
        ctx.exit(0)
    ctx.exit(1)
