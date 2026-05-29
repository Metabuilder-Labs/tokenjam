from __future__ import annotations
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


@dataclass
class DefaultsConfig:
    budget: BudgetConfig = field(default_factory=BudgetConfig)


@dataclass
class StorageConfig:
    path:           str = "~/.tj/telemetry.duckdb"
    retention_days: int = 90


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
    port:    int  = 9464
    path:    str  = "/metrics"


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
class CaptureConfig:
    prompts:      bool = False
    completions:  bool = False
    tool_inputs:  bool = False
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
class TjConfig:
    version:  str
    defaults: DefaultsConfig          = field(default_factory=DefaultsConfig)
    agents:   dict[str, AgentConfig]  = field(default_factory=dict)
    storage:  StorageConfig           = field(default_factory=StorageConfig)
    export:   ExportConfig            = field(default_factory=ExportConfig)
    alerts:   AlertsConfig            = field(default_factory=AlertsConfig)
    security: SecurityConfig          = field(default_factory=SecurityConfig)
    api:      ApiConfig               = field(default_factory=ApiConfig)
    capture:  CaptureConfig           = field(default_factory=CaptureConfig)
    budgets:  dict[str, ProviderBudget] = field(default_factory=dict)
    # Path to the config file on disk; set by load_config() so that relative
    # paths in the config (e.g. output_schema) can be resolved correctly.
    config_path: Path | None          = field(default=None, repr=False, compare=False)


# -- File discovery --

SEARCH_PATHS = [
    Path("tokenjam.toml"),
    Path(".tj/config.toml"),
    Path.home() / ".config" / "tj" / "config.toml",
]


def _warn_if_secrets_diverge(active_path: Path, active_raw: dict) -> None:
    """
    Emit a stderr warning if a shadowed config exists with a different
    ingest_secret. Tracks the common footgun (#68 §5): project-local
    .tj/config.toml has secret A; global ~/.config/tj/config.toml has
    secret B; the SDK uses A; the daemon (started with global config)
    uses B; span pushes 401 silently.

    Fires at most once per process via the module-level guard so this
    doesn't spam multi-call test environments.
    """
    global _SECRET_DIVERGENCE_WARNED
    if _SECRET_DIVERGENCE_WARNED:
        return
    active_secret = (active_raw.get("security") or {}).get("ingest_secret")
    if not active_secret:
        return
    try:
        active_resolved = active_path.resolve()
    except OSError:
        return
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
        # Diverged. Warn once.
        print(
            f"warning: ingest_secret differs between {active_path} "
            f"and {candidate}. The SDK will use the secret from "
            f"{active_path} but a daemon launched from a different cwd "
            f"may use the other one — span pushes will 401 silently. "
            f"Align them (copy one secret into the other config) or "
            f"delete the unused config.",
            file=sys.stderr,
        )
        _SECRET_DIVERGENCE_WARNED = True
        return


# Module-level guard. Reset for tests via the helper exposed below.
_SECRET_DIVERGENCE_WARNED = False


def _reset_secret_divergence_warning() -> None:
    """Test helper — reset the once-per-process warning guard."""
    global _SECRET_DIVERGENCE_WARNED
    _SECRET_DIVERGENCE_WARNED = False


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


def load_config(path: str | None = None) -> TjConfig:
    """
    Load config from file, merge with defaults, return TjConfig.

    IMPORTANT: tomllib requires binary mode "rb" -- not text mode "r".
    Using "r" raises TypeError at runtime.
    """
    config_path = find_config_file(path)
    if config_path is None:
        return TjConfig(version="1")

    with open(config_path, "rb") as f:   # "rb" is REQUIRED
        raw = tomllib.load(f)

    # Diverged-secret detection (#68 §5). When a project-local config
    # shadows a global one with a different ingest_secret, the SDK and
    # daemon end up with different secrets and span pushes silently 401.
    # Warn at config-load time so the user gets a chance to align them
    # before debugging mysterious 401s.
    _warn_if_secrets_diverge(config_path, raw)

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
        )

    storage_raw = raw.get("storage", {})
    storage = StorageConfig(
        path=storage_raw.get("path", StorageConfig.path),
        retention_days=storage_raw.get("retention_days", StorageConfig.retention_days),
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
        enabled=prom_raw.get("enabled", True),
        port=prom_raw.get("port", PrometheusConfig.port),
        path=prom_raw.get("path", PrometheusConfig.path),
    )
    export = ExportConfig(otlp=otlp, prometheus=prometheus)

    alerts_raw = raw.get("alerts", {})
    channels = []
    for ch_raw in alerts_raw.get("channels", []):
        channels.append(AlertChannelConfig(**ch_raw))
    alerts = AlertsConfig(
        cooldown_seconds=alerts_raw.get("cooldown_seconds", AlertsConfig.cooldown_seconds),
        include_captured_content=alerts_raw.get("include_captured_content", False),
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

    capture_raw = raw.get("capture", {})
    capture = CaptureConfig(
        prompts=capture_raw.get("prompts", False),
        completions=capture_raw.get("completions", False),
        tool_inputs=capture_raw.get("tool_inputs", False),
        tool_outputs=capture_raw.get("tool_outputs", False),
    )

    defaults_raw = raw.get("defaults", {})
    defaults_budget_raw = defaults_raw.get("budget", {})
    defaults = DefaultsConfig(budget=BudgetConfig(**defaults_budget_raw))

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

    return TjConfig(
        version=raw.get("version", "1"),
        defaults=defaults,
        agents=agents,
        storage=storage,
        export=export,
        alerts=alerts,
        security=security,
        api=api,
        capture=capture,
        budgets=budgets,
    )


def _serialise(config: TjConfig) -> dict:
    """Convert TjConfig back to a plain dict suitable for tomli_w."""
    def _dc_to_dict(obj: object) -> dict:
        result = {}
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

    # agents is a dict of str -> AgentConfig, handle specially
    agents_out = {}
    for agent_id, agent_cfg in config.agents.items():
        agents_out[agent_id] = _dc_to_dict(agent_cfg)
    d["agents"] = agents_out

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


def validate_budget_value(value: float, field_name: str) -> float | None:
    """
    Validate and normalise a budget value from user input.

    Positive values are returned as-is. Zero means 'remove limit' (returns None).
    Negative values raise ValueError.
    """
    if value < 0:
        raise ValueError(f"Budget {field_name} must be non-negative, got {value}")
    return value if value > 0 else None
