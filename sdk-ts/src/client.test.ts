import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { TjClient } from "./client.js";
import { SpanBuilder } from "./span-builder.js";
import { SpanKind, SpanStatus } from "./types.js";

/**
 * Spin up a local HTTP server that captures requests, so we can test
 * the client without a real OCW server.
 */
function createMockServer(): {
  server: ReturnType<typeof createServer>;
  port: () => number;
  requests: Array<{ method: string; url: string; headers: Record<string, string>; body: string }>;
  setResponse: (status: number, body: unknown) => void;
  start: () => Promise<void>;
  stop: () => Promise<void>;
} {
  const requests: Array<{
    method: string;
    url: string;
    headers: Record<string, string>;
    body: string;
  }> = [];
  let responseStatus = 200;
  let responseBody: unknown = { ingested: 1, rejected: 0, rejections: [] };

  const server = createServer((req: IncomingMessage, res: ServerResponse) => {
    let body = "";
    req.on("data", (chunk: Buffer) => {
      body += chunk.toString();
    });
    req.on("end", () => {
      requests.push({
        method: req.method ?? "",
        url: req.url ?? "",
        headers: req.headers as Record<string, string>,
        body,
      });
      res.writeHead(responseStatus, { "Content-Type": "application/json" });
      res.end(JSON.stringify(responseBody));
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
    setResponse(status: number, body: unknown) {
      responseStatus = status;
      responseBody = body;
    },
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

describe("TjClient", () => {
  let mock: ReturnType<typeof createMockServer>;

  beforeEach(async () => {
    mock = createMockServer();
    await mock.start();
  });

  afterEach(async () => {
    await mock.stop();
  });

  it("sends a span with correct auth header", async () => {
    const client = new TjClient({
      baseUrl: `http://127.0.0.1:${mock.port()}`,
      ingestSecret: "test-secret-123",
      batchSize: 1,
    });

    const span = new SpanBuilder("gen_ai.llm.call")
      .agentId("test-agent")
      .provider("anthropic")
      .model("claude-haiku-4-5")
      .inputTokens(1000)
      .outputTokens(200)
      .build();

    await client.send(span);

    assert.equal(mock.requests.length, 1);
    const req = mock.requests[0];
    assert.equal(req.method, "POST");
    assert.equal(req.url, "/api/v1/spans");
    assert.equal(req.headers["authorization"], "Bearer test-secret-123");
    assert.equal(req.headers["content-type"], "application/json");
  });

  it("sends OTLP JSON format", async () => {
    const client = new TjClient({
      baseUrl: `http://127.0.0.1:${mock.port()}`,
      ingestSecret: "secret",
      batchSize: 1,
    });

    const span = new SpanBuilder("gen_ai.llm.call")
      .agentId("agent-1")
      .inputTokens(500)
      .build();

    await client.send(span);

    const body = JSON.parse(mock.requests[0].body);
    assert.ok(body.resourceSpans);
    assert.ok(Array.isArray(body.resourceSpans));
    assert.equal(body.resourceSpans.length, 1);

    const scopeSpans = body.resourceSpans[0].scopeSpans;
    assert.ok(Array.isArray(scopeSpans));
    assert.equal(scopeSpans[0].spans.length, 1);

    const otlpSpan = scopeSpans[0].spans[0];
    assert.equal(otlpSpan.name, "gen_ai.llm.call");
    assert.ok(otlpSpan.traceId);
    assert.ok(otlpSpan.spanId);
    assert.equal(otlpSpan.kind, 3); // CLIENT
    assert.equal(otlpSpan.status.code, 1); // OK
  });

  it("batches spans and flushes at batchSize", async () => {
    mock.setResponse(200, { ingested: 3, rejected: 0, rejections: [] });

    const client = new TjClient({
      baseUrl: `http://127.0.0.1:${mock.port()}`,
      ingestSecret: "secret",
      batchSize: 3,
    });

    // Send 2 spans — should not flush yet
    await client.send(new SpanBuilder("span-1").build());
    await client.send(new SpanBuilder("span-2").build());
    assert.equal(mock.requests.length, 0);

    // Third span triggers flush
    await client.send(new SpanBuilder("span-3").build());
    assert.equal(mock.requests.length, 1);

    const body = JSON.parse(mock.requests[0].body);
    const spans = body.resourceSpans[0].scopeSpans[0].spans;
    assert.equal(spans.length, 3);
  });

  it("flush sends remaining spans", async () => {
    const client = new TjClient({
      baseUrl: `http://127.0.0.1:${mock.port()}`,
      ingestSecret: "secret",
      batchSize: 100, // won't auto-flush
    });

    await client.send(new SpanBuilder("span-1").build());
    await client.send(new SpanBuilder("span-2").build());
    assert.equal(mock.requests.length, 0);

    const result = await client.flush();
    assert.equal(mock.requests.length, 1);
    assert.ok(result);
    assert.equal(result.ingested, 1); // mock default
  });

  it("flush on empty buffer returns null", async () => {
    const client = new TjClient({
      baseUrl: `http://127.0.0.1:${mock.port()}`,
      ingestSecret: "secret",
    });

    const result = await client.flush();
    assert.equal(result, null);
    assert.equal(mock.requests.length, 0);
  });

  it("shutdown flushes and stops timer", async () => {
    const client = new TjClient({
      baseUrl: `http://127.0.0.1:${mock.port()}`,
      ingestSecret: "secret",
      batchSize: 100,
      flushIntervalMs: 60000, // won't auto-flush in test
    });
    client.start();

    await client.send(new SpanBuilder("span-1").build());
    await client.shutdown();

    assert.equal(mock.requests.length, 1);
  });

  it("throws on server error", async () => {
    mock.setResponse(401, { detail: "Invalid ingest secret" });

    const client = new TjClient({
      baseUrl: `http://127.0.0.1:${mock.port()}`,
      ingestSecret: "wrong-secret",
      batchSize: 1,
    });

    await assert.rejects(
      () => client.send(new SpanBuilder("span-1").build()),
      (err: Error) => {
        assert.ok(err.message.includes("401"));
        return true;
      }
    );
  });

  it("converts span attributes to OTLP format", async () => {
    const client = new TjClient({
      baseUrl: `http://127.0.0.1:${mock.port()}`,
      ingestSecret: "secret",
      batchSize: 1,
    });

    const span = new SpanBuilder("gen_ai.llm.call")
      .attribute("string.attr", "hello")
      .attribute("int.attr", 42)
      .attribute("float.attr", 3.14)
      .attribute("bool.attr", true)
      .build();

    await client.send(span);

    const body = JSON.parse(mock.requests[0].body);
    const attrs = body.resourceSpans[0].scopeSpans[0].spans[0].attributes;
    const attrMap = new Map(attrs.map((a: { key: string; value: unknown }) => [a.key, a.value]));

    assert.deepEqual(attrMap.get("string.attr"), { stringValue: "hello" });
    assert.deepEqual(attrMap.get("int.attr"), { intValue: "42" });
    assert.deepEqual(attrMap.get("float.attr"), { doubleValue: 3.14 });
    assert.deepEqual(attrMap.get("bool.attr"), { boolValue: true });
  });
});
