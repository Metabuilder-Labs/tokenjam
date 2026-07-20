/**
 * OpenTelemetry GenAI Semantic Convention attribute names.
 * Mirrors tokenjam/otel/semconv.py — keep in sync.
 */
export const GenAIAttributes = {
  AGENT_ID: "gen_ai.agent.id",
  AGENT_NAME: "gen_ai.agent.name",
  AGENT_VERSION: "gen_ai.agent.version",
  PROVIDER_NAME: "gen_ai.provider.name",
  REQUEST_MODEL: "gen_ai.request.model",
  REQUEST_TYPE: "gen_ai.request.type",
  INPUT_TOKENS: "gen_ai.usage.input_tokens",
  OUTPUT_TOKENS: "gen_ai.usage.output_tokens",
  CACHE_READ_TOKENS: "gen_ai.usage.cache_read_tokens",
  CACHE_CREATE_TOKENS: "gen_ai.usage.cache_creation_tokens",
  TOOL_NAME: "gen_ai.tool.name",
  TOOL_DESCRIPTION: "gen_ai.tool.description",
  TOOL_INPUT: "gen_ai.tool.input",
  TOOL_OUTPUT: "gen_ai.tool.output",
  CONVERSATION_ID: "gen_ai.conversation.id",
  PROMPT_CONTENT: "gen_ai.prompt.content",
  COMPLETION_CONTENT: "gen_ai.completion.content",
  SPAN_INVOKE_AGENT: "invoke_agent",
  SPAN_CREATE_AGENT: "create_agent",
  SPAN_TOOL_CALL: "gen_ai.tool.call",
  SPAN_LLM_CALL: "gen_ai.llm.call",

  // Outcome / feedback event (emerging gen_ai outcome-event semconv, OTel
  // semconv issue #2665). recordOutcome() emits a span carrying these; the
  // outcome-type attribute is the marker TokenJam Cloud's ROI ingest keys off.
  SPAN_OUTCOME: "gen_ai.outcome",
  OUTCOME_EVENT_NAME: "gen_ai.outcome",
  EVENT_NAME: "event.name",
  OUTCOME_TYPE: "gen_ai.outcome.type",
  OUTCOME_SUCCESS: "gen_ai.outcome.success",
  OUTCOME_VALUE_USD: "gen_ai.outcome.value_usd",
} as const;

export const TjAttributes = {
  COST_USD: "tokenjam.cost_usd",
  ALERT_TYPE: "tokenjam.alert.type",
  ALERT_SEVERITY: "tokenjam.alert.severity",
  // session.id is the key the canonical OTLP parser reads for a span's session.
  SESSION_ID: "session.id",
  // Explicit workflow key for an outcome event (see recordOutcome).
  WORKFLOW_ID: "tokenjam.workflow_id",
  SANDBOX_EVENT: "tokenjam.sandbox.event",
  EGRESS_HOST: "tokenjam.sandbox.egress_host",
  EGRESS_PORT: "tokenjam.sandbox.egress_port",
  FILESYSTEM_PATH: "tokenjam.sandbox.filesystem_path",
  SYSCALL_NAME: "tokenjam.sandbox.syscall_name",
} as const;

/**
 * Event names and attribute constants from Claude Code's OTel log exporter.
 * Mirrors ClaudeCodeEvents in tokenjam/otel/semconv.py — keep in sync.
 */
export const ClaudeCodeEvents = {
  // Event names (logRecord body values)
  API_REQUEST: "claude_code.api_request",
  TOOL_RESULT: "claude_code.tool_result",
  API_ERROR: "claude_code.api_error",
  USER_PROMPT: "claude_code.user_prompt",
  TOOL_DECISION: "claude_code.tool_decision",

  // Standard context attributes present on all events
  SESSION_ID: "session.id",
  PROMPT_ID: "prompt.id",
  EVENT_SEQUENCE: "event.sequence",

  // api_request attributes
  COST_USD: "cost_usd",
  DURATION_MS: "duration_ms",
  SPEED: "speed",
  INPUT_TOKENS: "input_tokens",
  OUTPUT_TOKENS: "output_tokens",
  CACHE_READ_TOKENS: "cache_read_tokens",
  CACHE_CREATION_TOKENS: "cache_creation_tokens",

  // tool_result attributes
  TOOL_NAME: "tool_name",
  SUCCESS: "success",
  ERROR: "error",
  TOOL_PARAMETERS: "tool_parameters",
  TOOL_INPUT: "tool_input",
  DECISION_TYPE: "decision_type",
  TOOL_RESULT_SIZE: "tool_result_size_bytes",

  // api_error attributes
  STATUS_CODE_HTTP: "status_code",
  ATTEMPT: "attempt",

  // tool_decision attributes
  DECISION: "decision",
  DECISION_SOURCE: "source",
} as const;
