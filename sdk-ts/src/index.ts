export {
  TjClient,
  type TjClientOptions,
  type RecordOutcomeOptions,
} from "./client.js";
export { type Span, type SpanBatch, type IngestResult, SpanKind, SpanStatus } from "./types.js";
export { SpanBuilder } from "./span-builder.js";
export { GenAIAttributes, TjAttributes, ClaudeCodeEvents } from "./semconv.js";
