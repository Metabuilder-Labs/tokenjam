"""
OpenTelemetry GenAI Semantic Conventions attribute names.
Based on OTel GenAI SemConv v1.37+.
"""


class GenAIAttributes:
    # Agent identity
    AGENT_ID      = "gen_ai.agent.id"
    AGENT_NAME    = "gen_ai.agent.name"
    AGENT_VERSION = "gen_ai.agent.version"

    # Provider (anthropic | openai | aws.bedrock | google | hud | ...)
    PROVIDER_NAME = "gen_ai.provider.name"

    # LLM request
    REQUEST_MODEL = "gen_ai.request.model"
    REQUEST_TYPE  = "gen_ai.request.type"

    # Token usage
    INPUT_TOKENS        = "gen_ai.usage.input_tokens"
    OUTPUT_TOKENS       = "gen_ai.usage.output_tokens"
    CACHE_READ_TOKENS   = "gen_ai.usage.cache_read_tokens"
    CACHE_CREATE_TOKENS = "gen_ai.usage.cache_creation_tokens"

    # Tool calls
    TOOL_NAME        = "gen_ai.tool.name"
    TOOL_DESCRIPTION = "gen_ai.tool.description"
    TOOL_INPUT       = "gen_ai.tool.input"
    TOOL_OUTPUT      = "gen_ai.tool.output"

    # Conversation / session continuity
    CONVERSATION_ID = "gen_ai.conversation.id"

    # Prompt / completion capture (off by default)
    PROMPT_CONTENT     = "gen_ai.prompt.content"
    COMPLETION_CONTENT = "gen_ai.completion.content"

    # Standard span names
    SPAN_INVOKE_AGENT = "invoke_agent"
    SPAN_CREATE_AGENT = "create_agent"
    SPAN_TOOL_CALL    = "gen_ai.tool.call"
    SPAN_LLM_CALL     = "gen_ai.llm.call"


class ClaudeCodeEvents:
    """Event names and attributes from Claude Code's OTel log exporter."""
    # Event names (logRecord body values)
    API_REQUEST   = "claude_code.api_request"
    TOOL_RESULT   = "claude_code.tool_result"
    API_ERROR     = "claude_code.api_error"
    USER_PROMPT   = "claude_code.user_prompt"
    TOOL_DECISION = "claude_code.tool_decision"

    # Standard context attributes on all events
    SESSION_ID     = "session.id"
    PROMPT_ID      = "prompt.id"
    EVENT_SEQUENCE = "event.sequence"

    # api_request attributes
    COST_USD              = "cost_usd"
    DURATION_MS           = "duration_ms"
    SPEED                 = "speed"
    INPUT_TOKENS          = "input_tokens"
    OUTPUT_TOKENS         = "output_tokens"
    CACHE_READ_TOKENS     = "cache_read_tokens"
    CACHE_CREATION_TOKENS = "cache_creation_tokens"

    # tool_result attributes
    TOOL_NAME        = "tool_name"
    SUCCESS          = "success"
    ERROR            = "error"
    TOOL_PARAMETERS  = "tool_parameters"
    TOOL_INPUT       = "tool_input"
    DECISION_TYPE    = "decision_type"
    TOOL_RESULT_SIZE = "tool_result_size_bytes"

    # api_error attributes
    STATUS_CODE_HTTP = "status_code"
    ATTEMPT          = "attempt"

    # tool_decision attributes
    DECISION         = "decision"
    DECISION_SOURCE  = "source"


class CodexEvents:
    """Event names and attributes from Codex CLI's OTel log exporter."""
    # Event names (logRecord body values)
    API_REQUEST   = "codex.api_request"
    SSE_EVENT     = "codex.sse_event"
    USER_PROMPT   = "codex.user_prompt"
    TOOL_DECISION = "codex.tool_decision"
    TOOL_RESULT   = "codex.tool_result"

    # Standard context attributes on all events
    CONVERSATION_ID = "conversation.id"
    APP_VERSION     = "app.version"
    MODEL           = "model"
    SLUG            = "slug"
    EVENT_TIMESTAMP = "event.timestamp"  # ISO-8601 UTC; Codex sets timeUnixNano=0

    # api_request attributes
    ATTEMPT      = "attempt"
    DURATION_MS  = "duration_ms"
    HTTP_STATUS  = "http.response.status_code"
    ERROR_MESSAGE = "error.message"

    # sse_event attributes
    EVENT_KIND            = "event.kind"
    INPUT_TOKEN_COUNT     = "input_token_count"
    OUTPUT_TOKEN_COUNT    = "output_token_count"
    CACHED_TOKEN_COUNT    = "cached_token_count"
    REASONING_TOKEN_COUNT = "reasoning_token_count"
    TOOL_TOKEN_COUNT      = "tool_token_count"

    # user_prompt attributes
    PROMPT_LENGTH = "prompt_length"
    PROMPT        = "prompt"

    # tool_decision attributes
    TOOL_NAME       = "tool_name"
    CALL_ID         = "call_id"
    DECISION        = "decision"
    DECISION_SOURCE = "source"

    # tool_result attributes  (also uses TOOL_NAME, CALL_ID, DURATION_MS, ERROR_MESSAGE)
    ARGUMENTS = "arguments"
    SUCCESS   = "success"
    OUTPUT    = "output"


class ResourceAttributes:
    """OTel standard resource attributes (set per process / service)."""
    SERVICE_NAME      = "service.name"
    # Logical grouping above service.name. tj uses it as the "project" a
    # service belongs to, so the dashboard can roll up every repo under one
    # project tile (e.g. all `Aquanodeio/*` repos -> namespace "aquanode").
    SERVICE_NAMESPACE = "service.namespace"
    # Per-instance identifier (one process / terminal). tj uses it as the
    # human label for a session's terminal (e.g. "founder-os") when set at
    # launch via OTEL_RESOURCE_ATTRIBUTES.
    SERVICE_INSTANCE_ID = "service.instance.id"


class TjAttributes:
    """tj-specific span attributes (non-standard extensions)."""
    COST_USD         = "tokenjam.cost_usd"
    SESSION_ID       = "session.id"
    ALERT_TYPE       = "tokenjam.alert.type"
    ALERT_SEVERITY   = "tokenjam.alert.severity"

    # Billing / plan classification
    # `billing_account` is provider-only (anthropic, openai, google, bedrock,
    # local.ollama). It's a span-level attribute set by each integration.
    # `plan_tier` is set on the session record at session creation by reading
    # ProviderBudget.plan for the matching billing_account; it does NOT live
    # on individual spans. Analyzers JOIN through SessionRecord to read it.
    BILLING_ACCOUNT  = "tokenjam.billing_account"
    PLAN_TIER        = "tokenjam.plan_tier"

    # NemoClaw / OpenShell sandbox events
    SANDBOX_EVENT    = "tokenjam.sandbox.event"
    EGRESS_HOST      = "tokenjam.sandbox.egress_host"
    EGRESS_PORT      = "tokenjam.sandbox.egress_port"
    FILESYSTEM_PATH  = "tokenjam.sandbox.filesystem_path"
    SYSCALL_NAME     = "tokenjam.sandbox.syscall_name"


# Valid plan_tier values. `unknown` is the default for backfilled or pre-onboard
# sessions; `tj optimize` suppresses dollar figures for unknown sessions.
VALID_PLAN_TIERS = frozenset({
    "api", "pro", "max_5x", "max_20x", "plus", "team", "enterprise", "local", "unknown",
})

# plan_tier values that mean "flat-rate subscription with allocation/cap."
# pricing_mode = "subscription" for these.
SUBSCRIPTION_PLAN_TIERS = frozenset({
    "pro", "max_5x", "max_20x", "plus", "team", "enterprise",
})
