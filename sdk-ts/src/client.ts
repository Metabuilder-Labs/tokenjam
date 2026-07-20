/**
 * TjClient — sends spans to the TokenJam REST API.
 * Communicates via HTTP POST to /api/v1/spans in OTLP JSON format.
 */
import { GenAIAttributes, TjAttributes } from "./semconv.js";
import { SpanBuilder } from "./span-builder.js";
import type { IngestResult, OtlpSpan, OtlpValue, Span, SpanBatch } from "./types.js";
import { SpanKind, SpanStatus } from "./types.js";

export interface RecordOutcomeOptions {
  /**
   * The kind of outcome, a caller-defined label
   * (e.g. "ticket_resolved", "lead_qualified", "pr_merged"). Required — this is
   * the marker attribute; without it the event is not recognised downstream.
   */
  outcomeType: string;
  /**
   * An explicit workflow key to attach the outcome to. Optional if sessionId is
   * given. At least one of workflowId / sessionId is required.
   */
  workflowId?: string;
  /**
   * The session (or root session of a fan-out) the outcome belongs to. At least
   * one of workflowId / sessionId is required.
   */
  sessionId?: string;
  /** Whether the outcome was achieved (execution succeeded). Defaults to true. */
  success?: boolean;
  /**
   * An OPTIONAL, SELF-REPORTED business value for the outcome in USD. A value
   * YOU declare — TokenJam does not measure or verify it. Negative values are
   * treated as undeclared. ROI compute (declared value / measured cost) is a
   * TokenJam Cloud feature; the SDK only emits the event.
   */
  valueUsd?: number;
  /** The emitting agent id (stamped as gen_ai.agent.id). */
  agentId?: string;
  /** Extra attributes attached verbatim to the event. */
  attributes?: Record<string, unknown>;
}

export interface TjClientOptions {
  /** Base URL of the TokenJam server (default: http://127.0.0.1:7391) */
  baseUrl?: string;
  /** Ingest secret for authentication */
  ingestSecret: string;
  /** Maximum batch size before auto-flush (default: 50) */
  batchSize?: number;
  /** Flush interval in milliseconds (default: 5000) */
  flushIntervalMs?: number;
  /** Service name reported in OTLP resource attributes (default: "tj-ts-sdk") */
  serviceName?: string;
  /** Maximum retry attempts on network errors or 5xx responses (default: 3) */
  maxRetries?: number;
}

const SPAN_KIND_TO_OTLP: Record<string, number> = {
  [SpanKind.INTERNAL]: 1,
  [SpanKind.SERVER]: 2,
  [SpanKind.CLIENT]: 3,
  [SpanKind.PRODUCER]: 4,
  [SpanKind.CONSUMER]: 5,
};

const STATUS_CODE_TO_OTLP: Record<string, number> = {
  [SpanStatus.UNSET]: 0,
  [SpanStatus.OK]: 1,
  [SpanStatus.ERROR]: 2,
};

function isoToUnixNano(iso: string): string {
  const ms = new Date(iso).getTime();
  // Represent as nanoseconds in a string (BigInt-safe)
  return `${ms}000000`;
}

function toOtlpValue(value: unknown): OtlpValue {
  if (typeof value === "string") return { stringValue: value };
  if (typeof value === "number") {
    if (Number.isInteger(value)) return { intValue: String(value) };
    return { doubleValue: value };
  }
  if (typeof value === "boolean") return { boolValue: value };
  if (Array.isArray(value)) {
    return { arrayValue: { values: value.map(toOtlpValue) } };
  }
  return { stringValue: String(value) };
}

function spanToOtlp(span: Span): OtlpSpan {
  const attributes = Object.entries(span.attributes).map(([key, value]) => ({
    key,
    value: toOtlpValue(value),
  }));

  const otlp: OtlpSpan = {
    traceId: span.traceId,
    spanId: span.spanId,
    name: span.name,
    kind: SPAN_KIND_TO_OTLP[span.kind] ?? 1,
    startTimeUnixNano: isoToUnixNano(span.startTime),
    status: {
      code: STATUS_CODE_TO_OTLP[span.statusCode] ?? 0,
      message: span.statusMessage,
    },
    attributes,
  };

  if (span.parentSpanId) otlp.parentSpanId = span.parentSpanId;
  if (span.endTime) otlp.endTimeUnixNano = isoToUnixNano(span.endTime);
  if (span.events?.length) {
    otlp.events = span.events.map(e => ({
      timeUnixNano: isoToUnixNano(e.time as string),
      name: e.name as string,
      attributes: Object.entries(e.attributes ?? {}).map(([key, value]) => ({ key, value: toOtlpValue(value) })),
    }));
  }

  return otlp;
}

export class TjClient {
  private readonly baseUrl: string;
  private readonly ingestSecret: string;
  private readonly batchSize: number;
  private readonly flushIntervalMs: number;
  private readonly serviceName: string;
  private readonly maxRetries: number;
  private buffer: Span[] = [];
  private timer: ReturnType<typeof setInterval> | null = null;

  constructor(options: TjClientOptions) {
    this.baseUrl = (options.baseUrl ?? "http://127.0.0.1:7391").replace(
      /\/$/,
      ""
    );
    this.ingestSecret = options.ingestSecret;
    this.batchSize = options.batchSize ?? 50;
    this.flushIntervalMs = options.flushIntervalMs ?? 5000;
    this.serviceName = options.serviceName ?? "tj-ts-sdk";
    this.maxRetries = options.maxRetries ?? 3;
  }

  /**
   * Start the automatic flush timer.
   * Call this once after creating the client.
   */
  start(): this {
    if (this.timer) return this;
    this.timer = setInterval(() => {
      void this.flush();
    }, this.flushIntervalMs);
    // Allow process to exit even if timer is running
    if (this.timer.unref) this.timer.unref();
    return this;
  }

  /**
   * Add a span to the buffer. Auto-flushes when batchSize is reached.
   */
  async send(span: Span): Promise<void> {
    this.buffer.push(span);
    if (this.buffer.length >= this.batchSize) {
      await this.flush();
    }
  }

  /**
   * Emit a gen_ai outcome event attaching a business outcome to a workflow.
   *
   * A thin wrapper that builds and sends one outcome span carrying the emerging
   * gen_ai outcome-event attributes (OTel semconv issue #2665) that TokenJam
   * Cloud's ROI ingest keys off — one call instead of hand-POSTing OTLP JSON.
   * The span is buffered like any other (auto-flushes at batchSize).
   *
   * ROI compute is a TokenJam Cloud feature; the SDK only emits the event.
   * `valueUsd` is self-reported — TokenJam does not measure or verify it.
   *
   * @throws if outcomeType is empty, or neither workflowId nor sessionId is set.
   */
  async recordOutcome(options: RecordOutcomeOptions): Promise<void> {
    if (!options.outcomeType) {
      throw new Error("recordOutcome requires a non-empty outcomeType");
    }
    if (!options.workflowId && !options.sessionId) {
      throw new Error(
        "recordOutcome requires at least one of workflowId or sessionId"
      );
    }

    const builder = new SpanBuilder(GenAIAttributes.SPAN_OUTCOME)
      .kind(SpanKind.CLIENT)
      // The marker attrs the Cloud ROI ingest keys off, plus the stock
      // event.name so the event can also ride the OTLP event path.
      .attribute(GenAIAttributes.EVENT_NAME, GenAIAttributes.OUTCOME_EVENT_NAME)
      .attribute(GenAIAttributes.OUTCOME_TYPE, options.outcomeType)
      .attribute(GenAIAttributes.OUTCOME_SUCCESS, options.success ?? true);

    if (options.workflowId) {
      builder.attribute(TjAttributes.WORKFLOW_ID, options.workflowId);
    }
    if (options.sessionId) {
      // session.id is the key the canonical OTLP parser reads (not the
      // SpanBuilder.sessionId() gen_ai.session.id form).
      builder.attribute(TjAttributes.SESSION_ID, options.sessionId);
    }
    if (options.agentId) {
      builder.agentId(options.agentId);
    }
    if (options.valueUsd != null) {
      builder.attribute(GenAIAttributes.OUTCOME_VALUE_USD, options.valueUsd);
    }
    for (const [key, value] of Object.entries(options.attributes ?? {})) {
      builder.attribute(key, value);
    }

    await this.send(builder.build());
  }

  /**
   * Flush all buffered spans to the server.
   * Returns the ingest result, or null if the buffer was empty.
   */
  async flush(): Promise<IngestResult | null> {
    if (this.buffer.length === 0) return null;

    const spans = this.buffer.splice(0);
    const batch = this.toBatch(spans);
    return this.post(batch);
  }

  /**
   * Flush remaining spans and stop the timer.
   */
  async shutdown(): Promise<void> {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
    await this.flush();
  }

  /**
   * Convert SDK spans to OTLP JSON batch format.
   */
  private toBatch(spans: Span[]): SpanBatch {
    return {
      resourceSpans: [
        {
          resource: {
            attributes: [
              {
                key: "service.name",
                value: { stringValue: this.serviceName },
              },
            ],
          },
          scopeSpans: [
            {
              spans: spans.map(spanToOtlp),
            },
          ],
        },
      ],
    };
  }

  /**
   * POST a span batch to the ingest endpoint.
   * Retries up to maxRetries times on network errors or 5xx responses using
   * exponential backoff (base 2s, matching Python HttpTransport behaviour).
   * 4xx errors are not retried — they indicate auth or validation failures.
   */
  private async post(batch: SpanBatch): Promise<IngestResult> {
    const url = `${this.baseUrl}/api/v1/spans`;
    const body = JSON.stringify(batch);
    const headers = {
      "Content-Type": "application/json",
      Authorization: `Bearer ${this.ingestSecret}`,
    };

    let lastError: Error | null = null;
    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      if (attempt > 0) {
        // Exponential backoff: 2s, 4s, 8s
        await new Promise(resolve => setTimeout(resolve, 2000 * Math.pow(2, attempt - 1)));
      }

      let response: Response;
      try {
        response = await fetch(url, { method: "POST", headers, body });
      } catch (err) {
        // Network-level error — retry
        lastError = err instanceof Error ? err : new Error(String(err));
        continue;
      }

      if (response.ok) {
        return (await response.json()) as IngestResult;
      }

      const text = await response.text().catch(() => "");
      const error = new Error(
        `TokenJam ingest failed: ${response.status} ${response.statusText} — ${text}`
      );

      // 4xx: not retriable (auth/validation failure)
      if (response.status >= 400 && response.status < 500) {
        throw error;
      }

      // 5xx: retriable
      lastError = error;
    }

    throw lastError ?? new Error("TokenJam ingest failed after retries");
  }
}
