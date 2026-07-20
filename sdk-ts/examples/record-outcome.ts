/**
 * Record an outcome (TypeScript) — attach a business outcome to a workflow in
 * one call with `client.recordOutcome(...)` instead of hand-building an OTLP
 * outcome event.
 *
 * Requires `tj serve` running (the TS SDK sends over HTTP):
 *
 *     tj serve &
 *     npx tsx sdk-ts/examples/record-outcome.ts   # or compile with tsc
 *
 * Honesty note: `valueUsd` is OPTIONAL and SELF-REPORTED. It is a value YOU
 * declare for the outcome; TokenJam does not measure or verify it. ROI compute
 * (declared value ÷ measured cost) is a TokenJam Cloud feature — this SDK only
 * EMITS the outcome event.
 */
import { TjClient, SpanBuilder } from "@tokenjam/sdk";

async function main(): Promise<void> {
  const client = new TjClient({
    ingestSecret: process.env.TJ_INGEST_SECRET ?? "your-ingest-secret",
    serviceName: "support-agent",
  }).start();

  const sessionId = "sess-" + Math.random().toString(36).slice(2, 10);

  // A tiny support workflow: one LLM call attributed to the session.
  await client.send(
    new SpanBuilder("gen_ai.llm.call")
      .agentId("support-agent")
      .provider("anthropic")
      .model("claude-haiku-4-5")
      .inputTokens(350)
      .outputTokens(120)
      .attribute("session.id", sessionId)
      .durationMs(900)
      .build()
  );

  // Attach the business outcome to that session's workflow.
  await client.recordOutcome({
    outcomeType: "ticket_resolved",
    sessionId,
    success: true,
    valueUsd: 25.0, // self-reported: what resolving this ticket is worth
    agentId: "support-agent",
  });

  await client.shutdown();
  console.log(`Recorded a 'ticket_resolved' outcome for ${sessionId}.`);
  console.log("ROI compute is a TokenJam Cloud feature — the SDK only emits the event.");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
