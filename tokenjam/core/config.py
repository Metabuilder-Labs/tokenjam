from __future__ import annotations
import os
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]

import tomli_w


# -- Nested config dataclasses --

@dataclass
class SensitiveAction:
    name:     str
    severity: str = "warning"   # critical | warning | info


@dataclass
class BudgetConfig:
    daily_usd:   float | None = None
    session_usd: float | None = None


@dataclass
class GroupBudgetConfig:
    """A coding-tool GROUP's daily cap — e.g. one ceiling covering every
    claude-code-<project> variant summed together. Daily-only: a per-session
    cap has no meaning at group scope (there is no single session to cap),
    and the UI never offers a per-session field for these rows. Kept as its
    own dataclass rather than reusing BudgetConfig so a `session_usd` can
    never be silently written here and silently ignored."""
    daily_usd: float | None = None


@dataclass
class CodingGroupConfig:
    budget: GroupBudgetConfig = field(default_factory=GroupBudgetConfig)


@dataclass
class DriftConfig:
    enabled:            bool  = True
    baseline_sessions:  int   = 10
    token_threshold:    float = 2.0
    tool_sequence_diff: float = 0.4


@dataclass
class AgentConfig:
    description:      str                  = ""
    budget:           BudgetConfig         = field(default_factory=BudgetConfig)
    sensitive_actions: list[SensitiveAction] = field(default_factory=list)
    output_schema:    str | None           = None
    drift:            DriftConfig          = field(default_factory=DriftConfig)
    # Project this agent rolls up under in the dashboard (server-side fallback
    # for OTel service.namespace). Lets already-running sessions group by
    # project without restarting the agent — the mapping is applied by tj, so
    # no service.namespace needs to arrive on the wire.
    project:          str | None           = None
    # Local checkout this agent's code lives in, registered BY THE USER
    # ([agents.<id>] source_path). Opt-in and never inferred: tokenjam does not
    # scan the filesystem looking for an agent's source. Its only consumer is
    # the gated model-id swap in `core.optimize.relearn_apply` (a deterministic
    # string substitution in a clean git repo); with no source_path a downsize
    # card stays advise-only and just prints the one-paste fix.
    source_path:      str | None           = None


@dataclass
class DefaultsConfig:
    # SDK-workflow zone default (per-agent daily + session cap).
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    # Coding-agent zone default: the daily group cap a newly-appearing coding
    # tool (a claude-code/codex group with no [coding_agents.<id>] entry yet)
    # inherits, so it starts capped instead of arriving uncapped.
    coding_budget: GroupBudgetConfig = field(default_factory=GroupBudgetConfig)


@dataclass
class StorageConfig:
    path:           str = "~/.tj/telemetry.duckdb"
    # THE user-chosen analysis span — "30d" / "90d" / "all" — written by
    # `tj onboard`. Retention is derived from it, never chosen independently,
    # so deletion cannot remove history the product is offering to analyze. See
    # `core/analysis_span.py` for the derivation and the one-directional clamp;
    # read the span through that module, never off this field, so the
    # back-compat path below is applied everywhere.
    analysis_span:  str | None = None
    # None means "derive from analysis_span". A config that sets this and
    # nothing else — every config written before the coupling existed — has its
    # value read AS the span, so nothing about that setup changes.
    retention_days: int | None = None
    # DuckDB's buffer pool, bounded. Left unset, DuckDB sizes `memory_limit` at
    # 80% of physical RAM — 19.1 GiB on a 24 GB Mac — and its buffer manager has
    # no reason to evict below that, so a daemon that should idle near 200 MB
    # instead grows until the machine is swapping. This is a *ceiling*, not a
    # reservation: past it DuckDB spills to `temp_directory` rather than
    # failing, so a small number costs disk I/O on the rare large scan and
    # nothing the rest of the time. `threads` is capped for the same reason —
    # each carries its own allocation arenas, and the daemon opens a second
    # backend for transcript catch-up while the request path is live.
    # 2GB rather than 1GB because 1GB was measured failing here: on a 768k-span
    # database the daemon OOMed on a request-path aggregate while the startup
    # transcript catch-up already held the pool ("1.0 GiB/953.6 MiB used" — the
    # query needed 16 KB more). The ceiling has to fit the request path AND the
    # catch-up backend that runs beside it, not either alone. Raise it if a scan
    # ever spills enough to feel slow.
    memory_limit:   str = "2GB"
    threads:        int = 4
    # Runtime provenance, never read from or written to TOML: True when `path`
    # came from an explicit `--db` rather than config discovery. The
    # filesystem-reading analyzers scope themselves off it (see
    # `core/optimize/scope.py`) — a config file that happens to name the same
    # path is a normal run, so the two cases have to stay distinguishable
    # after the override has been applied.
    path_is_explicit: bool = field(default=False, repr=False, compare=False)


@dataclass
class OtlpConfig:
    enabled:  bool        = False
    endpoint: str         = "http://localhost:4318"
    protocol: str         = "http"   # http | grpc
    headers:  dict        = field(default_factory=dict)
    insecure: bool        = True


@dataclass
class PrometheusConfig:
    enabled: bool = True


@dataclass
class ExportConfig:
    otlp:       OtlpConfig       = field(default_factory=OtlpConfig)
    prometheus: PrometheusConfig = field(default_factory=PrometheusConfig)


@dataclass
class AlertChannelConfig:
    type: str
    # stdout / file
    path: str | None = None
    # ntfy
    topic:        str | None = None
    server:       str        = "https://ntfy.sh"
    token:        str        = ""
    # webhook
    url:     str | None = None
    method:  str        = "POST"
    headers: dict       = field(default_factory=dict)
    # discord
    webhook_url: str | None = None
    # telegram
    bot_token: str | None = None
    chat_id:   str | None = None
    # shared
    min_severity: str = "info"


@dataclass
class AlertsConfig:
    cooldown_seconds:        int  = 60
    include_captured_content: bool = False
    async_hooks:             bool = False
    channels: list[AlertChannelConfig] = field(default_factory=lambda: [
        AlertChannelConfig(type="stdout"),
    ])


@dataclass
class SecurityConfig:
    ingest_secret:          str = ""
    max_attribute_bytes:    int = 65536
    max_attributes_per_span: int = 256
    max_attribute_depth:    int = 10
    webhook_allowed_domains: list[str] = field(default_factory=list)


@dataclass
class ApiAuthConfig:
    enabled: bool = False
    api_key: str  = ""


@dataclass
class ApiConfig:
    enabled: bool         = True
    host:    str          = "127.0.0.1"
    port:    int          = 7391
    auth:    ApiAuthConfig = field(default_factory=ApiAuthConfig)


@dataclass
class ProxyConfig:
    """Optional in-process enforcement-plane proxy (#219), off by default.

    When ``enabled``, ``tj serve`` runs a second listener on ``port`` that sits
    between an agent and its LLM provider, speaking the Anthropic
    (``/v1/messages``) and OpenAI (``/v1/chat/completions``) APIs. It ships in
    SUGGEST MODE ONLY — it records what a policy *would* do and enforces nothing.

    The pricing-mode gate is a built-in invariant (not a toggle): subscription
    and ``unknown`` traffic is always forwarded unmodified (observe-only), and
    only api/usage-billed traffic reaches the policy path. ``killswitch`` flips
    the proxy to pass-through-everything while keeping the listener alive.
    """
    enabled:            bool = False
    host:               str  = "127.0.0.1"
    port:               int  = 7392
    # "suggest" only for now; the enforce-mode path lands behind a later gate (#220).
    mode:               str  = "suggest"
    killswitch:         bool = False
    anthropic_base_url: str  = "https://api.anthropic.com"
    openai_base_url:    str  = "https://api.openai.com"


@dataclass
class PolicyConfig:
    """A data-driven enforcement-plane policy (#220), defined in `[[policies]]`.

    A policy is DATA, not code: it binds a ``kind`` (a registered evaluator) to
    a target (provider / agent) with kind-specific ``params``. The proxy's
    policy engine loads these and evaluates eligible (api/usage-billed) requests.

    ``mode`` is ``suggest`` (evaluate + record what it WOULD do, enforce nothing)
    or ``enforce`` (gated OFF in the OSS rails — scaffolded, never acts). All
    OSS policies are user-authored and run **unvalidated** — there is no
    certification engine in the open tree, so no policy decision is ever implied
    to have been validated as safe.
    """
    name:            str
    kind:            str
    enabled:         bool = True
    mode:            str  = "suggest"          # suggest | enforce (enforce gated off)
    target_provider: str | None = None          # anthropic | openai | None (any)
    target_agent:    str | None = None          # agent id | None (any)
    params:          dict = field(default_factory=dict)


@dataclass
class CaptureConfig:
    """What raw content gets persisted alongside token counts / model names.

    ``prompts`` defaults ON: without prompt text, `cache-recommend` and
    `trim` never fire and `reuse` never reaches its prompt-prefix mode (all
    three stay dark for every onboarded user otherwise). Storage is
    local-only (the user's own telemetry DB), which is why this defaults on
    rather than opt-in. ``completions`` and ``tool_outputs`` stay off by
    default — no analyzer needs them yet, and completion text is the
    largest, least useful payload to store.

    ``tool_inputs`` also defaults ON: without tool-call arguments, the
    `script` (workflow-restructure) and `verbosity` analyzers fall back to
    tool-names-only clustering — the same "dark by default" problem
    ``prompts`` had. Claude Code's JSONL backfill is the persona that
    actually populates this (Read/Grep/Glob file paths and search queries,
    not their content), and it's also what `tj context` / the statusline
    read from. SDK/API callers gain little from it today (no automatic
    instrumentation records ``gen_ai.tool.input`` there yet), but the
    default stays uniform across onboarding paths rather than carving out
    an exception per persona.
    """
    prompts:      bool = True
    completions:  bool = False
    tool_inputs:  bool = True
    tool_outputs: bool = False


@dataclass
class ProviderBudget:
    """
    Per-provider periodic spending budget used by `tj optimize` projections.

    Distinct from BudgetConfig (per-agent daily/session alert thresholds).
    ProviderBudget is a recurring monthly ceiling — typed against a provider
    so projection scopes to the spend that actually counts toward that budget.

    `plan` is the user's declared plan tier for this provider, written by
    `tj onboard`. SessionRecord.plan_tier is set at session creation by
    reading this field for the matching billing_account. Valid values: see
    VALID_PLAN_TIERS in tokenjam.otel.semconv.
    """
    usd:                  float | None      = None
    cycle_start_day:      int               = 1
    # service.name values that count toward this budget. Empty = all services
    # billed by this provider.
    applies_to_services:  list[str]         = field(default_factory=list)
    # Declared plan tier (api | pro | max_5x | max_20x | plus | team |
    # enterprise | local). Defaults to None so missing config produces
    # plan_tier='unknown' on sessions rather than a silent 'api' guess.
    plan:                 str | None        = None


@dataclass
class SummarizeConfig:
    """`[summarize]` — config for structure-aware prompt summarization.

    `api_model` is the model `tj summarize prep --via api` calls (with the user's own
    `TJ_ANTHROPIC_API_KEY`). There is NO default: only frontier models are validated to
    preserve structure (DEC-029 / DEF-010), and a weak model just fails the structure
    check and stages nothing — so the user must choose one explicitly.

    `allow_outbound_run` gates `POST /summarize/run` — the only server route that
    spends the user's money / drives their subscription (DEC-031). Default OFF: the
    outbound run surface is inert on a fresh install until the user knowingly turns it
    on. The manual (prep + paste-back check) path never goes outbound and is unaffected.
    """
    api_model: str | None = None
    allow_outbound_run: bool = False


@dataclass
class OptimizeConfig:
    """`[optimize]` — sensitivity thresholds for `tj optimize` analyzers.

    Every analyzer's "does this even fire" bar used to be a bare module-level
    constant no user could see or change. A savings opportunity that exists in
    a user's data but never clears the bar is a saving never surfaced — this
    section exists so a user can trade "possible noise" for "possible visibility"
    on their own data, without a code change or a fork.

    Every field default matches the historical module-level constant it
    replaces exactly, so an unset `[optimize]` section (or a config predating
    this section entirely) reproduces today's behaviour byte-for-byte. Lowering
    a field only ever makes an analyzer MORE willing to surface a finding on
    the same data; raising it only makes it more conservative. See each
    analyzer module's docstring/comment at the named constant for the
    false-positive reasoning behind its default — this class does not repeat
    that reasoning, only the wiring.
    """
    # script (analyzers/workflow_restructure.py MIN_CLUSTER_INSTANCES): a
    # tool-call-signature cluster needs at least this many member sessions
    # before it's recommended for script-replacement.
    min_cluster_instances: int = 20
    # The transcript/config root the filesystem-reading analyzers (deadweight,
    # relearn, summarize) may read — the `--projects-root` flag writes here.
    # `None` defers to the precedence chain in `core/optimize/scope.py`: the
    # TJ_CLAUDE_PROJECTS_ROOT env var, then suppression under an explicit
    # `--db`, then `~/.claude/projects`. Unlike every other field in this
    # section this is a SCOPE, not a sensitivity threshold — it decides which
    # filesystem is evidence, not how eager an analyzer is about it.
    projects_root: str | None = None
    # deadweight (core/optimize/mcp_probe.py): whether the analyzer may MEASURE
    # each configured MCP server's tool schemas, which means STARTING that
    # server, sending it `initialize` + `tools/list`, and terminating it. That is
    # the only way to know what a server actually injects, and the figure it
    # replaced was a flat assumption charged per-server.
    #
    # OFF BY DEFAULT, deliberately. The protocol surface is read-only (no
    # `tools/call` is ever issued), but starting a process is not nothing: the
    # command is the user's own, and an `npx -y` spec fetches from the network
    # while a server may open connections or refresh a token on startup. The
    # analyzer pass is reachable from `tj optimize`, the daemon's scheduled scan,
    # AND tj's own MCP tool — so an agent asking tj for a report could otherwise
    # spawn a batch of the user's servers as a side effect of a question. Making
    # the first exposure opt-in is the difference between a user choosing that
    # and discovering it.
    #
    # Leaving it off does NOT restore the old assumption. Unmeasured servers are
    # excluded from every priced figure and the finding's own coverage note says
    # how many were excluded and why.
    measure_mcp_schemas: bool = False
    # cache (analyzers/cache_efficacy.py MIN_INPUT_TOKENS): minimum
    # (provider, model) input-token volume in the window before a low
    # cache-efficacy ratio is even worth surfacing.
    min_cache_input_tokens: int = 100_000
    # cache (analyzers/cache_efficacy.py EFFICACY_THRESHOLD): a (provider,
    # model) row is flagged when its cache-read efficacy falls below this.
    cache_efficacy_threshold: float = 0.30
    # cache (analyzers/cache_efficacy.py MIN_CALLS_FOR_ROOT_CAUSE): minimum
    # call volume for one agent before the A1/A2/A3 root-cause classifiers
    # (uncached / thrash / lookback-miss) consider it.
    min_calls_for_root_cause: int = 20
    # verbosity (analyzers/output_verbosity.py MIN_COHORT_SESSIONS): a
    # task-shape cohort needs at least this many sessions before its output-
    # token median is a meaningful baseline to flag outliers against.
    min_cohort_sessions: int = 5
    # cache-recommend (analyzers/cache_recommend.py MIN_PREFIX_OCCURRENCES):
    # minimum occurrences of the same prompt prefix before a cache_control
    # breakpoint is recommended for it.
    min_prefix_occurrences: int = 3
    # trim (analyzers/prompt_bloat.py SIGNIFICANCE_THRESHOLD): LLMLingua-2
    # tokens scored below this are considered low-significance ("bloat").
    trim_significance_threshold: float = 0.40
    # reuse (analyzers/plan_reuse.py MIN_REPETITIONS): minimum sessions
    # sharing a planning skeleton before the cluster is surfaced.
    min_reuse_repetitions: int = 3
    # downsize quota-audit (analyzers/model_downgrade.py MIN_STRETCH_TURNS):
    # minimum contiguous cheap-shaped turns before a mid-session stretch
    # counts toward the premium-quota-misallocation audit.
    min_stretch_turns: int = 2
    # subagent (analyzers/subagent_rightsizing.py MIN_FLAG_COST_USD): noise
    # floor below which a subagent's spend isn't worth a right-sizing flag
    # regardless of its shape.
    min_flag_cost_usd: float = 0.05
    # relearn (analyzers/relearn.py MIN_RECURRING_SESSIONS): minimum distinct
    # sessions a failure signature must recur across before it's proposed.
    min_recurring_sessions: int = 3
    # placement (analyzers/batch_placement.py MIN_SESSIONS_FOR_CADENCE):
    # minimum sessions in a workload group before its start-time cadence is
    # even checked for regularity.
    min_sessions_for_cadence: int = 5
    # placement (analyzers/batch_placement.py MIN_GROUP_COST_USD): minimum
    # window spend for a cadence-regular workload group before batch-lane
    # placement is worth suggesting.
    min_group_cost_usd: float = 1.0

    # --- Scheduled analyzer scan (core/optimize/report_store.py) -------------
    # No HTTP request path runs analyzers any more: the full report is computed
    # by the `tj serve` daemon (once at boot, then on an interval, plus on a
    # user-pressed rescan) and every route serves the STORED result. These are
    # the always-on rails for that scan — a kill switch, a cadence, the window
    # the scan observes, and a floor on how often a rescan request may actually
    # re-run the analyzers.
    #
    # scan_enabled=False keeps the daemon from ever scanning on its own; the
    # stored report then only ever changes when a human presses rescan. It does
    # NOT re-enable inline computation on a request — nothing does.
    scan_enabled: bool = True
    # Cadence of the daemon's background scan.
    scan_interval_hours: float = 6.0
    # Lookback the scan observes. Stored alongside the result so every surface
    # labels the figures with the window they were actually computed over
    # rather than whatever picker the reader's screen is set to.
    scan_window_days: int = 30
    # Floor between two rescans that actually re-run the analyzers. A rescan
    # request inside this window is answered with the stored result and
    # `throttled: true` rather than stacking another full-corpus pass.
    scan_min_rescan_seconds: int = 60
    # How often a UI surface re-reads the stored result (NOT how often the scan
    # runs). Zero disables the UI's auto-refresh entirely.
    scan_ui_poll_seconds: int = 300
    # Ingestion watermark that gates a SCHEDULED tick (never an explicit
    # rescan — see `core/optimize/scan_cycle.py`). A tick only re-runs the
    # expensive analyzer pass when at least this many new spans have landed
    # since the last pass; below it, the tick is a no-op on an idle machine.
    # 1 preserves today's "scan on every tick" behaviour for any machine that
    # saw ANY telemetry; only a genuinely idle tick is skipped.
    scan_watermark_min_new_spans: int = 1
    # Ceiling on the above: a scheduled tick re-runs the pass regardless of
    # the watermark once this many hours have passed since the last one, so a
    # quiet machine still refreshes eventually rather than never rescanning
    # again.
    scan_watermark_max_staleness_hours: float = 24.0


@dataclass
class LoopConfig:
    """`[loop]` — the self-improve loop's workspace binding.

    `transcript_path` points the loop at a session-transcript root other than
    Claude Code's `~/.claude/projects`. A Claude Agent SDK app writes its own
    transcripts wherever the app puts them, so without this the loop simply
    can't see it: the detector globs the projects root and finds nothing.

    Setting it also decides which lane an agent lands in, and that seam is
    load-bearing for honesty:

      * a transcript root + a repo cwd = a WORKSPACE agent. The loop can write a
        fix into that repo's own `.claude/` (note / skill / hook, human-gated and
        reversible), so it gets the full detect -> propose -> approve -> apply ->
        verify path.
      * no workspace (a plain OTel service) = the ADVISE lane. Detect + advise +
        verify only, off stored spans; no auto-apply path exists at all. See
        `core/optimize/relearn_otel.py`.

    `None` (the default) keeps the historical behaviour: `resolve_projects_root`
    falls back to `TJ_CLAUDE_PROJECTS_ROOT` and then `~/.claude/projects`.
    """
    transcript_path: str | None = None


@dataclass
class IngestConfig:
    """`[ingest]` — continuous transcript ingestion by the `tj serve` daemon.

    Claude Code's OTLP exporter has no retry and no buffer, so a session whose
    shell lacked the telemetry env vars, or that ran while the daemon was down
    or pointed at a stale port, is dropped permanently by the live path. The
    on-disk transcript still exists — but only for ~30 days, after which Claude
    Code prunes it and the session is unrecoverable.

    So the daemon re-runs the (idempotent) Claude Code backfill over a bounded
    recent window: once shortly after startup, so downtime self-heals, and then
    on an interval, so an ongoing live-path miss is closed within minutes
    instead of waiting for a human to remember `tj backfill claude-code`.

    `startup_lookback_days` is deliberately wider than `lookback_hours`: the
    startup pass has to cover however long the daemon was down, while the
    steady-state pass only has to cover one interval plus slack.
    """
    auto_catch_up:        bool = True
    interval_minutes:     int  = 30
    lookback_hours:       int  = 48
    startup_lookback_days: int = 14


@dataclass
class TjConfig:
    version:  str
    defaults: DefaultsConfig          = field(default_factory=DefaultsConfig)
    agents:   dict[str, AgentConfig]  = field(default_factory=dict)
    # Coding-tool GROUP caps ([coding_agents.<group_id>.budget] in TOML), keyed
    # by group id ("claude-code" / "codex" — see core/agent_kind.py). A
    # deliberately SEPARATE namespace from `agents`: a group id like
    # "claude-code" would otherwise collide with a literal per-agent
    # [agents.claude-code] entry (the bare agent_id some setups still emit).
    # Keeping groups in their own top-level TOML table means both can be
    # configured independently with no ambiguity about which one a given
    # section name refers to.
    coding_agents: dict[str, CodingGroupConfig] = field(default_factory=dict)
    storage:  StorageConfig           = field(default_factory=StorageConfig)
    export:   ExportConfig            = field(default_factory=ExportConfig)
    alerts:   AlertsConfig            = field(default_factory=AlertsConfig)
    security: SecurityConfig          = field(default_factory=SecurityConfig)
    api:      ApiConfig               = field(default_factory=ApiConfig)
    proxy:    ProxyConfig             = field(default_factory=ProxyConfig)
    capture:  CaptureConfig           = field(default_factory=CaptureConfig)
    summarize: SummarizeConfig        = field(default_factory=SummarizeConfig)
    optimize: OptimizeConfig          = field(default_factory=OptimizeConfig)
    loop:     LoopConfig              = field(default_factory=LoopConfig)
    ingest:   IngestConfig            = field(default_factory=IngestConfig)
    budgets:  dict[str, ProviderBudget] = field(default_factory=dict)
    policies: list[PolicyConfig]      = field(default_factory=list)
    # Manual session_id -> human label overrides ([session_labels] in TOML).
    # Keys may be a full session_id or a prefix (e.g. the 8-char short id shown
    # on the dashboard). Lets you name already-running terminals immediately;
    # for durable naming prefer OTel service.instance.id.
    session_labels: dict[str, str]    = field(default_factory=dict)
    # Idle window (minutes) for the session lifecycle ([sessions] idle_minutes).
    # An active session quieter than SESSION_STALE_THRESHOLD (5 min) but within
    # this window renders as "idle"; beyond it as "stale" (archived). 4h default.
    session_idle_minutes: int         = 240
    # Path to the config file on disk; set by load_config() so that relative
    # paths in the config (e.g. output_schema) can be resolved correctly.
    config_path: Path | None          = field(default=None, repr=False, compare=False)


# -- File discovery --

SEARCH_PATHS = [
    Path("tokenjam.toml"),
    Path(".tj/config.toml"),
    Path.home() / ".config" / "tj" / "config.toml",
]


def global_config_path() -> Path:
    """The user-scoped config file: the single source of truth for the ingest
    secret.

    Resolved at call time rather than import time, because ``SEARCH_PATHS``
    bakes ``Path.home()`` in at import and a test (or a sandboxed run) that
    moves ``HOME`` afterwards would otherwise still be pointed at the real
    user's file.

    Why this file and no other: the daemon is launched from a unit file that
    names this path explicitly, so the daemon ALWAYS authenticates with the
    secret stored here, whatever directory anything else runs from. A secret
    minted into a project-local ``.tj/config.toml`` is therefore a secret the
    daemon will never accept, and every span pushed with it 401s silently.
    """
    return Path.home() / ".config" / "tj" / "config.toml"


def global_ingest_secret() -> str:
    """The ingest secret recorded in the global config, or "" when there is
    no global config or it carries none. Never raises."""
    path = global_config_path()
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return ""
    secret = (raw.get("security") or {}).get("ingest_secret")
    return secret if isinstance(secret, str) else ""


def ensure_global_ingest_secret() -> tuple[str, bool]:
    """Return ``(secret, minted)`` for the machine-wide ingest secret.

    Adopts the secret already in the global config when there is one, and only
    mints (and stores) a fresh one when the global config has none. Every
    onboarding path routes its secret through here, so no flow can create a
    second, divergent secret: the shell block that every already-configured
    integration carries holds the GLOBAL secret, and overwriting that would
    break every working integration.
    """
    import secrets as _secrets

    existing = global_ingest_secret()
    if existing:
        return existing, False

    path = global_config_path()
    config = load_config(str(path)) if path.exists() else TjConfig(version="1")
    secret = _secrets.token_hex(32)
    config.security.ingest_secret = secret
    write_config(config, path)
    return secret, True


def align_project_secret_to_global(secret: str, path: Path | None = None) -> Path | None:
    """Rewrite a project-local config's ``ingest_secret`` to the global one.

    The global secret is always the survivor: the managed shell block, the
    daemon unit file, and therefore every already-working integration carry
    it. Returns the path repaired, or None when there was nothing to repair
    (no project-local config, or it already agrees).
    """
    target = path if path is not None else Path(".tj/config.toml")
    if not secret or not target.exists():
        return None
    try:
        with open(target, "rb") as f:
            raw = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    current = (raw.get("security") or {}).get("ingest_secret")
    if not isinstance(current, str) or not current or current == secret:
        return None
    # Textual edit of the one key, not a `write_config` round-trip: the plain
    # onboarding path writes this file as commented TOML by hand, and
    # re-serialising it would silently delete every one of those comments as a
    # side effect of a secret repair.
    import re as _re

    try:
        text = target.read_text()
    except OSError:
        return None
    patched, count = _re.subn(
        r'(?m)^(\s*ingest_secret\s*=\s*)"[^"]*"',
        lambda m: m.group(1) + f'"{secret}"',
        text,
        count=1,
    )
    if not count:
        return None
    try:
        target.write_text(patched)
    except OSError:
        return None
    # Resolved, not relative: the caller prints this, and ".tj/config.toml" on
    # its own does not say WHICH project directory was just edited.
    return target.resolve()


def find_diverged_secret_config(active_path: Path, active_raw: dict) -> tuple[Path, str, str] | None:
    """Return `(other_path, active_secret, other_secret)` for the first
    shadowed config on `SEARCH_PATHS` that carries a DIFFERENT non-empty
    `ingest_secret` than `active_raw`'s, or None when none diverges.

    The single source of truth for the footgun `tj doctor`'s ingest-secret
    check reports: project-local `.tj/config.toml` has
    secret A; global `~/.config/tj/config.toml` has secret B; the SDK uses A;
    the daemon (started with global config) uses B; span pushes 401 silently.
    Best-effort — an unreadable/unparseable candidate is skipped, never raised.
    """
    active_secret = (active_raw.get("security") or {}).get("ingest_secret")
    if not active_secret:
        return None
    try:
        active_resolved = active_path.resolve()
    except OSError:
        return None
    for candidate in SEARCH_PATHS:
        try:
            cand_resolved = candidate.resolve()
        except OSError:
            continue
        if cand_resolved == active_resolved:
            continue
        if not candidate.exists():
            continue
        try:
            with open(candidate, "rb") as f:
                other_raw = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError):
            continue
        other_secret = (other_raw.get("security") or {}).get("ingest_secret")
        if not other_secret or other_secret == active_secret:
            continue
        return candidate, active_secret, other_secret
    return None


# `load_config` deliberately does NOT warn about a diverged secret. It used to
# print one to stderr on every invocation, in front of every command, which is
# noise on a fault the user cannot act on from there and trains people to read
# past warnings. The divergence is now reported where a verdict belongs and can
# be re-checked on demand: `tj doctor`'s ingest-secret check (which calls
# `find_diverged_secret_config` directly) alongside its live endpoint probe, and
# onboarding no longer creates the divergence in the first place
# (`ensure_global_ingest_secret` / `align_project_secret_to_global`).


def find_config_file(override: str | None = None) -> Path | None:
    if override:
        p = Path(override)
        if p.exists():
            return p
        raise FileNotFoundError(f"Config file not found: {override}")
    for path in SEARCH_PATHS:
        if path.exists():
            return path
    return None


def resolve_config_path(override: str | None = None) -> Path | None:
    """
    The single source of truth for "which config file is this process using".

    Precedence: an explicit ``override`` wins, then the ``TJ_CONFIG``
    environment variable, then ``find_config_file``'s ``SEARCH_PATHS`` walk.
    ``load_config`` resolves through here, so any call site that independently
    needs the config path — to write back to the file config was read from
    (budget updates), to report it (``tj doctor``), or to hand it to a
    subprocess (``tj mcp``, the MCP server) — must call this too, never bare
    ``find_config_file()``. A bare call ignores ``TJ_CONFIG`` and silently
    splits reads (TJ_CONFIG-aware, via ``load_config``) from writes/reports
    (search-path only) across two different files.

    Like ``find_config_file``, raises ``FileNotFoundError`` when an explicit
    override or ``TJ_CONFIG`` points at a path that doesn't exist — this
    matches ``load_config``'s fail-loud contract. Callers that must stay
    resilient to a bad ``TJ_CONFIG`` (e.g. the bare ``tj`` landing screen,
    which renders before any config is validated) should not use this
    function directly; see ``cli/home.py``.
    """
    if override is None:
        override = os.environ.get("TJ_CONFIG") or None
    return find_config_file(override)


def active_config_path(config: "TjConfig | None") -> Path | None:
    """The file this already-loaded config was actually read FROM, if known.

    ``resolve_config_path`` answers "which file WOULD this process discover",
    which is a different question once a per-invocation ``--config`` override
    is in play: the override never reaches the environment, so a rediscovery
    silently falls through to ``TJ_CONFIG`` or the search path and names some
    other file. Every site that writes the config back — or reports which one
    is live — must ask about the config it is holding, not re-run discovery.

    Returns ``None`` when the config did not come from a file (defaults only,
    or a caller-constructed object); the call site then falls back to
    ``resolve_config_path()`` for the genuine no-config-yet case.
    """
    return getattr(config, "config_path", None)


def load_config(path: str | None = None) -> TjConfig:
    """
    Load config from file, merge with defaults, return TjConfig.

    When no explicit ``path`` is given, honor the ``TJ_CONFIG`` environment
    variable before falling back to the search-path discovery order (via
    ``resolve_config_path``). This keeps SDK-bootstrapped processes
    (``ensure_initialised`` and the SDK integrations, which call
    ``load_config()`` with no argument) consistent with the CLI — the CLI
    already resolves ``TJ_CONFIG`` via Click's ``envvar`` and passes the path
    in, so without this the SDK silently wrote spans to the global DB even
    when ``TJ_CONFIG`` pointed elsewhere (#196). An explicit ``path`` argument
    still wins over the env var.

    IMPORTANT: tomllib requires binary mode "rb" -- not text mode "r".
    Using "r" raises TypeError at runtime.
    """
    config_path = resolve_config_path(path)
    if config_path is None:
        return TjConfig(version="1")

    with open(config_path, "rb") as f:   # "rb" is REQUIRED
        raw = tomllib.load(f)

    cfg = _parse(raw)
    cfg.config_path = config_path.resolve()
    return cfg


def write_config(config: TjConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(_serialise(config), f)


def _parse(raw: dict) -> TjConfig:
    """Convert raw TOML dict to TjConfig, applying defaults for missing keys."""
    agents = {}
    for agent_id, agent_raw in raw.get("agents", {}).items():
        budget = BudgetConfig(**agent_raw.get("budget", {}))
        sensitive_actions = [
            SensitiveAction(**sa) for sa in agent_raw.get("sensitive_actions", [])
        ]
        drift = DriftConfig(**agent_raw.get("drift", {}))
        agents[agent_id] = AgentConfig(
            description=agent_raw.get("description", ""),
            budget=budget,
            sensitive_actions=sensitive_actions,
            output_schema=agent_raw.get("output_schema"),
            drift=drift,
            project=agent_raw.get("project"),
            source_path=agent_raw.get("source_path"),
        )

    storage_raw = raw.get("storage", {})
    storage = StorageConfig(
        path=storage_raw.get("path", StorageConfig.path),
        # Absence is meaningful for both of these and must survive the load:
        # `analysis_span` absent + `retention_days` present is the pre-coupling
        # config whose kept history IS its span. Defaulting either one here
        # would erase that distinction.
        analysis_span=storage_raw.get("analysis_span"),
        retention_days=storage_raw.get("retention_days"),
        memory_limit=storage_raw.get("memory_limit", StorageConfig.memory_limit),
        threads=storage_raw.get("threads", StorageConfig.threads),
    )

    export_raw = raw.get("export", {})
    otlp_raw = export_raw.get("otlp", {})
    otlp = OtlpConfig(
        enabled=otlp_raw.get("enabled", False),
        endpoint=otlp_raw.get("endpoint", OtlpConfig.endpoint),
        protocol=otlp_raw.get("protocol", OtlpConfig.protocol),
        headers=otlp_raw.get("headers", {}),
        insecure=otlp_raw.get("insecure", True),
    )
    prom_raw = export_raw.get("prometheus", {})
    prometheus = PrometheusConfig(
        enabled=prom_raw.get("enabled", True)
    )
    export = ExportConfig(otlp=otlp, prometheus=prometheus)

    alerts_raw = raw.get("alerts", {})
    channels = []
    for ch_raw in alerts_raw.get("channels", []):
        channels.append(AlertChannelConfig(**ch_raw))
    alerts = AlertsConfig(
        cooldown_seconds=alerts_raw.get("cooldown_seconds", AlertsConfig.cooldown_seconds),
        include_captured_content=alerts_raw.get("include_captured_content", False),
        async_hooks=alerts_raw.get("async_hooks", False),
        channels=channels if channels else [AlertChannelConfig(type="stdout")],
    )

    security_raw = raw.get("security", {})
    security = SecurityConfig(
        ingest_secret=security_raw.get("ingest_secret", ""),
        max_attribute_bytes=security_raw.get("max_attribute_bytes", 65536),
        max_attributes_per_span=security_raw.get("max_attributes_per_span", 256),
        max_attribute_depth=security_raw.get("max_attribute_depth", 10),
        webhook_allowed_domains=security_raw.get("webhook_allowed_domains", []),
    )

    api_raw = raw.get("api", {})
    api_auth_raw = api_raw.get("auth", {})
    api_auth = ApiAuthConfig(
        enabled=api_auth_raw.get("enabled", False),
        api_key=api_auth_raw.get("api_key", ""),
    )
    api = ApiConfig(
        enabled=api_raw.get("enabled", True),
        host=api_raw.get("host", ApiConfig.host),
        port=api_raw.get("port", ApiConfig.port),
        auth=api_auth,
    )

    proxy_raw = raw.get("proxy", {})
    proxy = ProxyConfig(
        enabled=proxy_raw.get("enabled", False),
        host=proxy_raw.get("host", ProxyConfig.host),
        port=proxy_raw.get("port", ProxyConfig.port),
        mode=proxy_raw.get("mode", ProxyConfig.mode),
        killswitch=proxy_raw.get("killswitch", False),
        anthropic_base_url=proxy_raw.get("anthropic_base_url", ProxyConfig.anthropic_base_url),
        openai_base_url=proxy_raw.get("openai_base_url", ProxyConfig.openai_base_url),
    )

    # Fall back to the CaptureConfig defaults (not a hardcoded False) so a
    # config that predates a given flag — or a hand-authored one that omits
    # `[capture]` entirely — picks up the current default rather than being
    # frozen at whatever the flag defaulted to when the file was written.
    capture_raw = raw.get("capture", {})
    capture = CaptureConfig(
        prompts=capture_raw.get("prompts", CaptureConfig.prompts),
        completions=capture_raw.get("completions", CaptureConfig.completions),
        tool_inputs=capture_raw.get("tool_inputs", CaptureConfig.tool_inputs),
        tool_outputs=capture_raw.get("tool_outputs", CaptureConfig.tool_outputs),
    )

    # [loop] — optional transcript root for non-Claude-Code workspace agents
    # (a Claude Agent SDK app). Absent/None keeps the env + ~/.claude default.
    loop_raw = raw.get("loop", {})
    loop_cfg = LoopConfig(
        transcript_path=loop_raw.get("transcript_path") or None,
    )

    # [ingest] — the daemon's continuous transcript catch-up. Defaults are on:
    # without it, completeness depends on a human running a CLI command.
    ingest_raw = raw.get("ingest", {})
    ingest_cfg = IngestConfig(
        auto_catch_up=bool(ingest_raw.get("auto_catch_up", IngestConfig.auto_catch_up)),
        interval_minutes=int(
            ingest_raw.get("interval_minutes", IngestConfig.interval_minutes)
        ),
        lookback_hours=int(
            ingest_raw.get("lookback_hours", IngestConfig.lookback_hours)
        ),
        startup_lookback_days=int(
            ingest_raw.get("startup_lookback_days", IngestConfig.startup_lookback_days)
        ),
    )

    summarize = SummarizeConfig(
        api_model=raw.get("summarize", {}).get("api_model"),
        allow_outbound_run=bool(raw.get("summarize", {}).get("allow_outbound_run", False)),
    )

    # [optimize] — analyzer sensitivity thresholds. Fall back to the
    # OptimizeConfig defaults (not a hardcoded literal) so a config that
    # predates this section, or a hand-authored one that omits a given key,
    # picks up the current module-constant-equivalent default rather than
    # being frozen at whatever value existed when the file was written —
    # same discipline as the `capture` block above.
    optimize_raw = raw.get("optimize", {})
    optimize = OptimizeConfig(
        min_cluster_instances=optimize_raw.get(
            "min_cluster_instances", OptimizeConfig.min_cluster_instances),
        measure_mcp_schemas=bool(optimize_raw.get(
            "measure_mcp_schemas", OptimizeConfig.measure_mcp_schemas)),
        min_cache_input_tokens=optimize_raw.get(
            "min_cache_input_tokens", OptimizeConfig.min_cache_input_tokens),
        cache_efficacy_threshold=optimize_raw.get(
            "cache_efficacy_threshold", OptimizeConfig.cache_efficacy_threshold),
        min_calls_for_root_cause=optimize_raw.get(
            "min_calls_for_root_cause", OptimizeConfig.min_calls_for_root_cause),
        min_cohort_sessions=optimize_raw.get(
            "min_cohort_sessions", OptimizeConfig.min_cohort_sessions),
        min_prefix_occurrences=optimize_raw.get(
            "min_prefix_occurrences", OptimizeConfig.min_prefix_occurrences),
        trim_significance_threshold=optimize_raw.get(
            "trim_significance_threshold", OptimizeConfig.trim_significance_threshold),
        min_reuse_repetitions=optimize_raw.get(
            "min_reuse_repetitions", OptimizeConfig.min_reuse_repetitions),
        min_stretch_turns=optimize_raw.get(
            "min_stretch_turns", OptimizeConfig.min_stretch_turns),
        min_flag_cost_usd=optimize_raw.get(
            "min_flag_cost_usd", OptimizeConfig.min_flag_cost_usd),
        min_recurring_sessions=optimize_raw.get(
            "min_recurring_sessions", OptimizeConfig.min_recurring_sessions),
        min_sessions_for_cadence=optimize_raw.get(
            "min_sessions_for_cadence", OptimizeConfig.min_sessions_for_cadence),
        min_group_cost_usd=optimize_raw.get(
            "min_group_cost_usd", OptimizeConfig.min_group_cost_usd),
        scan_enabled=bool(optimize_raw.get(
            "scan_enabled", OptimizeConfig.scan_enabled)),
        scan_interval_hours=float(optimize_raw.get(
            "scan_interval_hours", OptimizeConfig.scan_interval_hours)),
        scan_window_days=int(optimize_raw.get(
            "scan_window_days", OptimizeConfig.scan_window_days)),
        scan_min_rescan_seconds=int(optimize_raw.get(
            "scan_min_rescan_seconds", OptimizeConfig.scan_min_rescan_seconds)),
        scan_ui_poll_seconds=int(optimize_raw.get(
            "scan_ui_poll_seconds", OptimizeConfig.scan_ui_poll_seconds)),
        scan_watermark_min_new_spans=int(optimize_raw.get(
            "scan_watermark_min_new_spans", OptimizeConfig.scan_watermark_min_new_spans)),
        scan_watermark_max_staleness_hours=float(optimize_raw.get(
            "scan_watermark_max_staleness_hours",
            OptimizeConfig.scan_watermark_max_staleness_hours)),
        projects_root=optimize_raw.get("projects_root") or None,
    )

    defaults_raw = raw.get("defaults", {})
    defaults_budget_raw = defaults_raw.get("budget", {})
    defaults_coding_budget_raw = defaults_raw.get("coding_budget", {})
    defaults = DefaultsConfig(
        budget=BudgetConfig(**defaults_budget_raw),
        coding_budget=GroupBudgetConfig(**defaults_coding_budget_raw),
    )

    # [coding_agents.<group_id>.budget] — daily-only ceilings for a coding
    # TOOL group ("claude-code" / "codex"), summed across every member
    # agent_id. Separate top-level table from [agents.*] on purpose: see
    # TjConfig.coding_agents docstring for the collision this avoids.
    coding_agents: dict[str, CodingGroupConfig] = {}
    for group_id, group_raw in raw.get("coding_agents", {}).items():
        if not isinstance(group_raw, dict):
            continue
        group_budget_raw = group_raw.get("budget", {})
        coding_agents[group_id] = CodingGroupConfig(
            budget=GroupBudgetConfig(**group_budget_raw)
        )

    # [budget.<provider>] sections — periodic monthly ceilings used by tj optimize.
    # Distinct from [defaults.budget] / [agents.X.budget] (per-agent alert thresholds).
    budgets: dict[str, ProviderBudget] = {}
    for provider, prov_raw in raw.get("budget", {}).items():
        if not isinstance(prov_raw, dict):
            continue
        budgets[provider] = ProviderBudget(
            usd=prov_raw.get("usd"),
            cycle_start_day=int(prov_raw.get("cycle_start_day", 1)),
            applies_to_services=list(prov_raw.get("applies_to_services", [])),
            plan=prov_raw.get("plan"),
        )

    # [[policies]] — data-driven enforcement-plane policies (#220). Each binds a
    # registered evaluator `kind` to a target with kind-specific params.
    policies: list[PolicyConfig] = []
    for pol_raw in raw.get("policies", []):
        if not isinstance(pol_raw, dict) or "name" not in pol_raw or "kind" not in pol_raw:
            continue
        policies.append(PolicyConfig(
            name=str(pol_raw["name"]),
            kind=str(pol_raw["kind"]),
            enabled=bool(pol_raw.get("enabled", True)),
            mode=str(pol_raw.get("mode", PolicyConfig.mode)),
            target_provider=pol_raw.get("target_provider"),
            target_agent=pol_raw.get("target_agent"),
            params=dict(pol_raw.get("params", {})),
        ))

    sessions_raw = raw.get("sessions", {})

    return TjConfig(
        version=raw.get("version", "1"),
        defaults=defaults,
        agents=agents,
        coding_agents=coding_agents,
        storage=storage,
        export=export,
        alerts=alerts,
        security=security,
        api=api,
        proxy=proxy,
        capture=capture,
        summarize=summarize,
        optimize=optimize,
        loop=loop_cfg,
        ingest=ingest_cfg,
        budgets=budgets,
        policies=policies,
        session_labels=dict(raw.get("session_labels", {})),
        session_idle_minutes=int(
            sessions_raw.get("idle_minutes", TjConfig.session_idle_minutes)
        ),
    )


def _serialise(config: TjConfig) -> dict:
    """Convert TjConfig back to a plain dict suitable for tomli_w."""
    def _dc_to_dict(obj: object) -> dict:
        result: dict = {}
        for f in fields(obj):  # type: ignore[arg-type]
            val = getattr(obj, f.name)
            if isinstance(val, dict):
                result[f.name] = val
            elif isinstance(val, list):
                result[f.name] = [
                    _dc_to_dict(item) if hasattr(item, "__dataclass_fields__") else item
                    for item in val
                ]
            elif hasattr(val, "__dataclass_fields__"):
                result[f.name] = _dc_to_dict(val)
            elif val is not None and not isinstance(val, Path):
                result[f.name] = val
        return result

    d = _dc_to_dict(config)
    # `budgets` (dataclass field) maps to `[budget.*]` (TOML key); strip raw form.
    d.pop("budgets", None)

    # `session_idle_minutes` (scalar field) maps to the `[sessions]` table.
    idle_minutes = d.pop("session_idle_minutes", None)
    if idle_minutes is not None:
        d["sessions"] = {"idle_minutes": idle_minutes}

    # agents is a dict of str -> AgentConfig, handle specially
    agents_out = {}
    for agent_id, agent_cfg in config.agents.items():
        agents_out[agent_id] = _dc_to_dict(agent_cfg)
    d["agents"] = agents_out

    # coding_agents is a dict of str -> CodingGroupConfig, handle specially
    # (same reason as `agents`/`budgets`: the generic dict branch above would
    # assign the raw dataclass objects instead of recursing into them).
    d.pop("coding_agents", None)
    coding_agents_out: dict = {}
    for group_id, group_cfg in config.coding_agents.items():
        coding_agents_out[group_id] = _dc_to_dict(group_cfg)
    if coding_agents_out:
        d["coding_agents"] = coding_agents_out

    # budgets is a dict of str -> ProviderBudget, handle specially
    budgets_out: dict = {}
    for provider, prov_cfg in config.budgets.items():
        budgets_out[provider] = _dc_to_dict(prov_cfg)
    if budgets_out:
        d["budget"] = budgets_out
    elif "budgets" in d:
        d.pop("budgets", None)

    return d


def resolve_effective_budget(agent_id: str, config: TjConfig) -> BudgetConfig:
    """
    Return the effective budget for an agent, merging per-agent overrides
    with global defaults on a per-field basis.

    Each field (daily_usd, session_usd) independently uses the agent value
    if set, otherwise falls back to the defaults value.
    """
    defaults = config.defaults.budget
    agent_cfg = config.agents.get(agent_id)
    if agent_cfg is None:
        return BudgetConfig(
            daily_usd=defaults.daily_usd,
            session_usd=defaults.session_usd,
        )
    ab = agent_cfg.budget
    return BudgetConfig(
        daily_usd=ab.daily_usd if ab.daily_usd is not None else defaults.daily_usd,
        session_usd=ab.session_usd if ab.session_usd is not None else defaults.session_usd,
    )


def resolve_group_budget(group_id: str, config: TjConfig) -> GroupBudgetConfig:
    """Return the effective daily cap for a coding-tool GROUP (see
    core/agent_kind.py for what a "group" is), merging any
    [coding_agents.<group_id>.budget] override over
    [defaults.coding_budget] — same per-field-fallback shape as
    resolve_effective_budget, just for the group namespace instead of the
    per-agent one.
    """
    defaults = config.defaults.coding_budget
    group_cfg = config.coding_agents.get(group_id)
    if group_cfg is None:
        return GroupBudgetConfig(daily_usd=defaults.daily_usd)
    gb = group_cfg.budget
    return GroupBudgetConfig(
        daily_usd=gb.daily_usd if gb.daily_usd is not None else defaults.daily_usd,
    )


def validate_budget_value(value: float, field_name: str) -> float | None:
    """
    Validate and normalise a budget value from user input.

    Positive values are returned as-is. Zero means 'remove limit' (returns None).
    Negative values raise ValueError.
    """
    if value < 0:
        raise ValueError(f"Budget {field_name} must be non-negative, got {value}")
    return value if value > 0 else None


def validate_cycle_start_day(value: int) -> int:
    """
    Validate a `[budget.<provider>] cycle_start_day` value from user input.

    Must be 1..28 (see `core.cycle.cycle_bounds` — clamped there too, but this
    surfaces a clear error at the API boundary instead of silently clamping a
    typo'd value the user never intended).
    """
    if value < 1 or value > 28:
        raise ValueError(f"cycle_start_day must be between 1 and 28, got {value}")
    return value
