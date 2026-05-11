/**
 * Core types for the TokenJam TypeScript SDK.
 */

export enum SpanKind {
  INTERNAL = "internal",
  CLIENT = "client",
  SERVER = "server",
  PRODUCER = "producer",
  CONSUMER = "consumer",
}

export enum SpanStatus {
  OK = "ok",
  ERROR = "error",
  UNSET = "unset",
}

export interface Span {
  spanId: string;
  traceId: string;
  parentSpanId?: string;
  name: string;
  kind: SpanKind;
  statusCode: SpanStatus;
  statusMessage?: string;
  startTime: string; // ISO 8601
  endTime?: string; // ISO 8601
  durationMs?: number;
  attributes: Record<string, unknown>;
  events?: Array<Record<string, unknown>>;
  agentId?: string;
  sessionId?: string;
  conversationId?: string;
}

export interface SpanBatch {
  resourceSpans: Array<{
    resource: {
      attributes: Array<{ key: string; value: { stringValue: string } }>;
    };
    scopeSpans: Array<{
      spans: Array<OtlpSpan>;
    }>;
  }>;
}

export interface OtlpSpan {
  traceId: string;
  spanId: string;
  parentSpanId?: string;
  name: string;
  kind: number;
  startTimeUnixNano: string;
  endTimeUnixNano?: string;
  status: { code: number; message?: string };
  attributes: Array<{ key: string; value: OtlpValue }>;
  events?: Array<{
    timeUnixNano: string;
    name: string;
    attributes?: Array<{ key: string; value: OtlpValue }>;
  }>;
}

export type OtlpValue =
  | { stringValue: string }
  | { intValue: string }
  | { doubleValue: number }
  | { boolValue: boolean }
  | { arrayValue: { values: OtlpValue[] } };

export interface IngestResult {
  ingested: number;
  rejected: number;
  rejections: Array<{ span_id: string; reason: string }>;
}
