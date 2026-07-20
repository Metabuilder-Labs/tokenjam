import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { TjClient } from "./client.js";
import { GenAIAttributes, TjAttributes } from "./semconv.js";

/**
 * Tests for TjClient.recordOutcome — asserts the emitted outcome span carries
 * the exact attribute names/shape TokenJam Cloud's ROI ingest keys off
 * (roi.is_outcome_event / write_outcome_from_span), and that argument
 * validation mirrors the Cloud OutcomeIn validator.
 */
function createMockServer(): {
  server: ReturnType<typeof createServer>;
  port: () => number;
  requests: Array<{ body: string }>;
  start: () => Promise<void>;
  stop: () => Promise<void>;
} {
  const requests: Array<{ body: string }> = [];
  const server = createServer((req: IncomingMessage, res: ServerResponse) => {
    let body = "";
    req.on("data", (chunk: Buffer) => {
      body += chunk.toString();
    });
    req.on("end", () => {
      requests.push({ body });
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ ingested: 1, rejected: 0, rejections: [] }));
    });
  });

  return {
    server,
    port: () => {
      const addr = server.address();
      if (addr && typeof addr === "object") return addr.port;
      throw new Error("Server not started");
    },
    requests,
    async start() {
      await new Promise<void>((resolve) => {
        server.listen(0, "127.0.0.1", resolve);
      });
    },
    async stop() {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    },
  };
}

/** Extract the first span's attribute map from a captured OTLP request body. */
function attrMapOf(body: string): Map<string, unknown> {
  const parsed = JSON.parse(body);
  const span = parsed.resourceSpans[0].scopeSpans[0].spans[0];
  return new Map(
    span.attributes.map((a: { key: string; value: Record<string, unknown> }) => [
      a.key,
      a.value,
    ])
  );
}

function spanNameOf(body: string): string {
  return JSON.parse(body).resourceSpans[0].scopeSpans[0].spans[0].name;
}

describe("TjClient.recordOutcome", () => {
  let mock: ReturnType<typeof createMockServer>;

  beforeEach(async () => {
    mock = createMockServer();
    await mock.start();
  });

  afterEach(async () => {
    await mock.stop();
  });

  function newClient() {
    return new TjClient({
      baseUrl: `http://127.0.0.1:${mock.port()}`,
      ingestSecret: "secret",
      batchSize: 1,
    });
  }

  it("emits an outcome span with the Cloud marker attributes", async () => {
    await newClient().recordOutcome({
      outcomeType: "ticket_resolved",
      workflowId: "wf-123",
    });

    assert.equal(mock.requests.length, 1);
    const body = mock.requests[0].body;
    assert.equal(spanNameOf(body), GenAIAttributes.SPAN_OUTCOME);

    const attrs = attrMapOf(body);
    assert.deepEqual(attrs.get(GenAIAttributes.EVENT_NAME), {
      stringValue: GenAIAttributes.OUTCOME_EVENT_NAME,
    });
    assert.deepEqual(attrs.get(GenAIAttributes.OUTCOME_TYPE), {
      stringValue: "ticket_resolved",
    });
    assert.deepEqual(attrs.get(GenAIAttributes.OUTCOME_SUCCESS), {
      boolValue: true,
    });
    assert.deepEqual(attrs.get(TjAttributes.WORKFLOW_ID), {
      stringValue: "wf-123",
    });
    // No value declared -> value_usd attribute absent (never fabricated).
    assert.equal(attrs.has(GenAIAttributes.OUTCOME_VALUE_USD), false);
  });

  it("uses canonical attribute name strings matching the Cloud roi.py", () => {
    assert.equal(GenAIAttributes.OUTCOME_EVENT_NAME, "gen_ai.outcome");
    assert.equal(GenAIAttributes.EVENT_NAME, "event.name");
    assert.equal(GenAIAttributes.OUTCOME_TYPE, "gen_ai.outcome.type");
    assert.equal(GenAIAttributes.OUTCOME_SUCCESS, "gen_ai.outcome.success");
    assert.equal(GenAIAttributes.OUTCOME_VALUE_USD, "gen_ai.outcome.value_usd");
    assert.equal(TjAttributes.WORKFLOW_ID, "tokenjam.workflow_id");
    assert.equal(TjAttributes.SESSION_ID, "session.id");
  });

  it("stamps session.id and self-reported value_usd", async () => {
    await newClient().recordOutcome({
      outcomeType: "lead_qualified",
      sessionId: "sess-9",
      valueUsd: 42.5,
      agentId: "sales-bot",
    });

    const attrs = attrMapOf(mock.requests[0].body);
    assert.deepEqual(attrs.get(TjAttributes.SESSION_ID), { stringValue: "sess-9" });
    assert.deepEqual(attrs.get(GenAIAttributes.OUTCOME_VALUE_USD), {
      doubleValue: 42.5,
    });
    assert.deepEqual(attrs.get(GenAIAttributes.AGENT_ID), {
      stringValue: "sales-bot",
    });
  });

  it("carries success=false and extra attributes", async () => {
    await newClient().recordOutcome({
      outcomeType: "checkout",
      workflowId: "wf-1",
      success: false,
      attributes: { customer_tier: "enterprise" },
    });

    const attrs = attrMapOf(mock.requests[0].body);
    assert.deepEqual(attrs.get(GenAIAttributes.OUTCOME_SUCCESS), {
      boolValue: false,
    });
    assert.deepEqual(attrs.get("customer_tier"), { stringValue: "enterprise" });
  });

  it("rejects an empty outcomeType", async () => {
    await assert.rejects(
      () => newClient().recordOutcome({ outcomeType: "", workflowId: "wf-1" }),
      /outcomeType/
    );
    assert.equal(mock.requests.length, 0);
  });

  it("rejects when neither workflowId nor sessionId is given", async () => {
    await assert.rejects(
      () => newClient().recordOutcome({ outcomeType: "ticket_resolved" }),
      /workflowId or sessionId/
    );
    assert.equal(mock.requests.length, 0);
  });
});
