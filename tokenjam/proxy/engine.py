"""The data-driven policy engine + envelope (#220) — suggest mode only.

A policy is **data, not code**: `[[policies]]` in config bind a registered
``kind`` (an evaluator) to a target with kind-specific ``params``. This module
loads them and, for **eligible** requests, evaluates each applicable policy and
produces a :class:`PolicyEnvelope` — an inspectable, round-trippable record of
what every policy *would* do.

Two invariants are enforced here, belt-and-suspenders with the gate (#219):

1. **API-only guard.** :meth:`PolicyEngine.evaluate` REFUSES to run on anything
   but a ``GateDecision.path == POLICY`` request (api/usage-billed). Observe-only
   traffic (subscription / local / unknown) never reaches the engine; calling it
   with such a decision raises :class:`PolicyGuardError`.
2. **Suggest mode only.** The engine evaluates and RECORDS; it never modifies a
   request. The enforce-mode path is scaffolded but GATED OFF behind
   ``ENFORCE_GATE_OPEN`` (False) — actual request modification lands later behind
   the certification gate (a separate, private track), NOT here.

Honesty: there is no certification engine in the open tree, so every envelope
carries the explicit ``unvalidated`` label. A suggestion is never implied to
have been validated as safe.

This module is intentionally HTTP-free (like the gate) so the safety-critical
logic stays inspectable and is unit-tested directly.
"""
from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from tokenjam.proxy.gate import POLICY, GateDecision
from tokenjam.utils.time_parse import utcnow

# The cert gate. ENFORCE never acts in the OSS rails — request modification is a
# separate, private track. This stays False here; do not flip it in OSS.
ENFORCE_GATE_OPEN = False

# Every OSS policy decision is unvalidated — there is no certification engine in
# the open tree. Surfaced on every envelope and in `tj policy` output.
UNVALIDATED_LABEL = "unvalidated"

# would_action vocabulary (suggest mode — these are what a policy WOULD do).
ACTION_NOOP = "noop"
ACTION_ALLOW = "allow"
ACTION_WOULD_MODIFY = "would_modify"
ACTION_WOULD_BLOCK = "would_block"
ACTION_ERROR = "error"

# Strength ordering for summarising an envelope's overall would-action.
_ACTION_STRENGTH = {
    ACTION_NOOP: 0, ACTION_ALLOW: 0, ACTION_ERROR: 1,
    ACTION_WOULD_MODIFY: 2, ACTION_WOULD_BLOCK: 3,
}


class PolicyGuardError(RuntimeError):
    """Raised when the engine is asked to evaluate non-POLICY (observe-only) traffic."""


@dataclass(frozen=True)
class PolicyRequest:
    """The pure evaluation context for one request (no HTTP types)."""
    provider: str
    path:     str
    agent:    str | None = None
    body:     dict | None = None


@dataclass(frozen=True)
class PolicyOutcome:
    """What a single evaluator decided — the minimal return from a `kind`."""
    would_action: str = ACTION_NOOP
    reason:       str = ""
    details:      dict = field(default_factory=dict)


# A policy `kind` evaluator. Pure function of (policy config, request) → outcome,
# or (policy, request, context) for kinds that need shared state — the engine
# inspects the arity and passes `context` only when the evaluator accepts it, so
# stateless kinds (noop) keep the 2-arg form.
PolicyEvaluator = Callable[..., PolicyOutcome]

POLICY_REGISTRY: dict[str, PolicyEvaluator] = {}


def register_policy(kind: str) -> Callable[[PolicyEvaluator], PolicyEvaluator]:
    """Register an evaluator for a policy ``kind`` (mirrors the analyzer registry)."""
    def _decorator(fn: PolicyEvaluator) -> PolicyEvaluator:
        POLICY_REGISTRY[kind] = fn
        return fn
    return _decorator


@dataclass(frozen=True)
class PolicyContext:
    """Shared, engine-level state an evaluator may need beyond the request.

    Stateless kinds (``noop``) ignore it; stateful kinds (``budget_cap``) read
    the per-provider ceiling from ``config.budgets`` and current-cycle spend from
    the telemetry DB. HTTP-free: ``db`` is the shared ``tj serve`` DuckDB backend
    (the proxy runs in-process, so it reuses that connection — per-thread cursors
    make this concurrency-safe, #124). ``spend_fn`` is an injectable override so
    evaluators are unit-testable without a DB; ``now_fn`` injects the clock for
    deterministic cycle bounds in tests.
    """
    config:   Any = None
    db:       Any = None
    spend_fn: Callable[[str], float | None] | None = None
    now_fn:   Callable[[], Any] | None = None

    def provider_budget(self, provider: str) -> Any:
        budgets = getattr(self.config, "budgets", None) or {}
        return budgets.get(provider)

    def cycle_spend_usd(self, provider: str) -> float | None:
        """Current billing-cycle USD spend for ``provider`` (None if unavailable).

        Honors the provider's ``cycle_start_day`` + ``applies_to_services`` from
        config, mirroring the budget-projection analyzer's window so the proxy
        and `tj optimize` agree on what counts. Best-effort: returns None when no
        spend source is wired (a budget_cap with no data is a no-op, not a guess).
        """
        if self.spend_fn is not None:
            return self.spend_fn(provider)
        conn = getattr(self.db, "conn", None) if self.db is not None else None
        if conn is None:
            return None
        from tokenjam.core.cycle import cycle_bounds
        budget = self.provider_budget(provider)
        start_day = getattr(budget, "cycle_start_day", 1) if budget is not None else 1
        services = list(getattr(budget, "applies_to_services", None) or []) if budget else []
        now = self.now_fn() if self.now_fn is not None else utcnow()
        cs, ce = cycle_bounds(now, start_day)
        # Parameterised SQL only (Rule 7); provider/cost_usd/start_time per semconv.
        clauses = ["start_time >= $1", "start_time < $2", "provider = $3"]
        params: list = [cs, ce, provider]
        if services:
            ph = ",".join("$" + str(len(params) + i + 1) for i in range(len(services)))
            clauses.append(f"agent_id IN ({ph})")  # agent_id holds service.name in tj's model
            params.extend(services)
        sql = "SELECT COALESCE(SUM(cost_usd), 0.0) FROM spans WHERE " + " AND ".join(clauses)
        try:
            row = conn.execute(sql, params).fetchone()
        except Exception:  # noqa: BLE001 — a spend-query failure must not break eval
            return None
        return float(row[0] or 0.0) if row else 0.0


@register_policy("noop")
def _noop_policy(policy: Any, request: PolicyRequest) -> PolicyOutcome:
    """The reference example `kind`: applies to eligible traffic, does nothing.

    Ships so the engine has a working built-in and `[[policies]]` round-trips
    end-to-end. It is deliberately NOT concrete policy logic (budget_cap is
    #222) — it only records that it observed the request.
    """
    return PolicyOutcome(
        would_action=ACTION_NOOP,
        reason="noop example policy: observed, no action",
        details={"name": getattr(policy, "name", "")},
    )


# Default fraction of the ceiling at which budget_cap raises a soft "approaching
# ceiling" warning (still a no-op action). Overridable per policy via params.
_BUDGET_CAP_DEFAULT_WARN_AT = 0.8


@register_policy("budget_cap")
def _budget_cap_policy(policy: Any, request: PolicyRequest, context: PolicyContext) -> PolicyOutcome:
    """The first concrete policy (#222): per-provider cycle-spend ceiling.

    Reads the existing ``[budget.<provider>] usd`` ceiling (``ProviderBudget.usd``)
    and compares it to the provider's current-cycle spend:

    - spend **over** the ceiling → ``would_block`` (it says it WOULD block further
      requests; SUGGEST MODE — it does not act).
    - spend at/above ``warn_at`` × ceiling (default 0.8) but under → a no-op that
      flags ``near_ceiling`` (a soft warning).
    - under → ``noop``.

    A ceiling is deterministic, so budget_cap has NO validation/certification
    dependency — that's why it's first, and why it's the policy that can graduate
    to enforce-mode first when that gate opens (NOT built here; ENFORCE stays
    closed). The envelope's ``unvalidated`` label + suggest-only framing are
    applied by the engine; this evaluator never says "blocked", only "would
    block".
    """
    provider = request.provider
    budget = context.provider_budget(provider) if context is not None else None
    ceiling = getattr(budget, "usd", None) if budget is not None else None
    if not ceiling or ceiling <= 0:
        return PolicyOutcome(
            would_action=ACTION_NOOP,
            reason=f"no [budget.{provider}] usd ceiling configured — nothing to cap",
            details={"provider": provider, "ceiling_usd": None},
        )

    spend = context.cycle_spend_usd(provider) if context is not None else None
    if spend is None:
        return PolicyOutcome(
            would_action=ACTION_NOOP,
            reason=f"current-cycle spend for {provider} unavailable — no evaluation",
            details={"provider": provider, "ceiling_usd": ceiling, "cycle_spend_usd": None},
        )

    try:
        warn_at = float((getattr(policy, "params", None) or {}).get("warn_at", _BUDGET_CAP_DEFAULT_WARN_AT))
    except (TypeError, ValueError):
        warn_at = _BUDGET_CAP_DEFAULT_WARN_AT
    pct = (spend / ceiling) if ceiling else 0.0
    base = {
        "provider": provider,
        "ceiling_usd": round(float(ceiling), 6),
        "cycle_spend_usd": round(float(spend), 6),
        "pct_of_ceiling": round(pct * 100, 1),
    }

    if spend > ceiling:
        # SUGGEST MODE: report what it WOULD do, never "blocked".
        return PolicyOutcome(
            would_action=ACTION_WOULD_BLOCK,
            reason=(f"{provider} cycle spend ${spend:.2f} is over the "
                    f"${ceiling:.2f} ceiling — would block until cycle reset "
                    "(suggest mode: not enforced)"),
            details={**base, "over_by_usd": round(float(spend - ceiling), 6),
                     "would_do": "block further requests until the budget cycle resets"},
        )
    if pct >= warn_at:
        return PolicyOutcome(
            would_action=ACTION_NOOP,
            reason=(f"{provider} cycle spend ${spend:.2f} is approaching the "
                    f"${ceiling:.2f} ceiling ({base['pct_of_ceiling']}%)"),
            details={**base, "near_ceiling": True, "warn_at_pct": round(warn_at * 100, 1)},
        )
    return PolicyOutcome(
        would_action=ACTION_NOOP,
        reason=f"{provider} cycle spend ${spend:.2f} is under the ${ceiling:.2f} ceiling",
        details={**base, "near_ceiling": False},
    )


@dataclass(frozen=True)
class PolicyEvaluation:
    """What ONE policy decided for a request — a row in the envelope."""
    policy_name:       str
    kind:              str
    mode:              str          # suggest | enforce
    would_action:      str
    reason:            str
    enforcement_gated: bool         # True when mode=enforce (gated off in OSS)
    details:           dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PolicyEvaluation":
        return cls(
            policy_name=d["policy_name"], kind=d["kind"], mode=d["mode"],
            would_action=d["would_action"], reason=d.get("reason", ""),
            enforcement_gated=bool(d.get("enforcement_gated", False)),
            details=dict(d.get("details", {})),
        )


@dataclass(frozen=True)
class PolicyEnvelope:
    """The inspectable, round-trippable record of a request's policy evaluation.

    Records what each policy WOULD do in suggest vs enforce mode. ``enforced`` is
    always False in the OSS rails (the enforce path is gated off); ``validated``
    is always False (no certification engine) and ``label`` is ``unvalidated``.
    """
    ts:                 str
    provider:           str
    path:               str
    agent:              str | None
    gate_path:          str          # always "policy" — the api-only guard holds
    evaluations:        list[PolicyEvaluation]
    overall_action:     str
    enforced:           bool = False       # OSS: never enforces
    enforcement_gated:  bool = False       # True if any enforce-mode policy applied
    validated:          bool = False       # OSS: no certification engine
    label:              str = UNVALIDATED_LABEL
    suggest_only:       bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["evaluations"] = [e.to_dict() for e in self.evaluations]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PolicyEnvelope":
        return cls(
            ts=d["ts"], provider=d["provider"], path=d["path"], agent=d.get("agent"),
            gate_path=d.get("gate_path", POLICY),
            evaluations=[PolicyEvaluation.from_dict(e) for e in d.get("evaluations", [])],
            overall_action=d.get("overall_action", ACTION_NOOP),
            enforced=bool(d.get("enforced", False)),
            enforcement_gated=bool(d.get("enforcement_gated", False)),
            validated=bool(d.get("validated", False)),
            label=d.get("label", UNVALIDATED_LABEL),
            suggest_only=bool(d.get("suggest_only", True)),
        )


def _policy_applies(policy: Any, request: PolicyRequest) -> bool:
    """A policy applies when its target filters match the request."""
    tp = getattr(policy, "target_provider", None)
    ta = getattr(policy, "target_agent", None)
    if tp is not None and tp != request.provider:
        return False
    if ta is not None and ta != request.agent:
        return False
    return True


def _accepts_context(fn: PolicyEvaluator) -> bool:
    """True if an evaluator takes a 3rd (context) positional arg."""
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return False
    if any(p.kind == p.VAR_POSITIONAL for p in params):
        return True
    positional = [p for p in params
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return len(positional) >= 3


def _call_evaluator(fn: PolicyEvaluator, policy: Any, request: PolicyRequest,
                    context: PolicyContext) -> PolicyOutcome:
    """Call an evaluator, passing ``context`` only to kinds that accept it."""
    if _accepts_context(fn):
        return fn(policy, request, context)
    return fn(policy, request)


class PolicyEngine:
    """Loads `[[policies]]` and evaluates eligible requests into envelopes."""

    def __init__(self, policies: list[Any], context: PolicyContext | None = None):
        # Only enabled policies participate.
        self.policies = [p for p in (policies or []) if getattr(p, "enabled", True)]
        # Shared state for stateful kinds (budget_cap). Empty by default so the
        # engine works with no config/DB (stateful kinds then no-op gracefully).
        self.context = context or PolicyContext()

    @classmethod
    def from_config(cls, config: Any, *, db: Any = None) -> "PolicyEngine":
        # `db` is the shared tj-serve DuckDB backend, threaded through so
        # budget_cap can read current-cycle spend (the proxy runs in-process).
        return cls(
            list(getattr(config, "policies", []) or []),
            context=PolicyContext(config=config, db=db),
        )

    def evaluate(self, gate_decision: GateDecision, request: PolicyRequest) -> PolicyEnvelope:
        """Evaluate all applicable policies for an eligible request.

        The API-ONLY GUARD is the first line: this refuses to run on anything but
        a POLICY-path (api/usage-billed) decision — belt-and-suspenders with the
        gate. Observe-only traffic must NEVER reach the engine.
        """
        if gate_decision.path != POLICY:
            raise PolicyGuardError(
                f"policy engine refused non-POLICY traffic (path={gate_decision.path!r}); "
                "observe-only traffic is never evaluated"
            )

        evaluations: list[PolicyEvaluation] = []
        enforcement_gated = False
        for policy in self.policies:
            if not _policy_applies(policy, request):
                continue
            evaluations.append(self._evaluate_one(policy, request))
            if getattr(policy, "mode", "suggest") == "enforce":
                enforcement_gated = True

        overall = ACTION_NOOP
        for ev in evaluations:
            if _ACTION_STRENGTH.get(ev.would_action, 0) > _ACTION_STRENGTH.get(overall, 0):
                overall = ev.would_action

        return PolicyEnvelope(
            ts=utcnow().isoformat(),
            provider=request.provider,
            path=request.path,
            agent=request.agent,
            gate_path=gate_decision.path,
            evaluations=evaluations,
            overall_action=overall,
            # Suggest mode only: nothing is ever enforced in the OSS rails. Even a
            # mode=enforce policy is gated off behind ENFORCE_GATE_OPEN.
            enforced=False,
            enforcement_gated=enforcement_gated and not ENFORCE_GATE_OPEN,
            validated=False,
            label=UNVALIDATED_LABEL,
        )

    def _evaluate_one(self, policy: Any, request: PolicyRequest) -> PolicyEvaluation:
        kind = getattr(policy, "kind", "")
        mode = getattr(policy, "mode", "suggest")
        name = getattr(policy, "name", "")
        gated = mode == "enforce"  # enforce is scaffolded but never acts (OSS)

        evaluator = POLICY_REGISTRY.get(kind)
        if evaluator is None:
            return PolicyEvaluation(
                policy_name=name, kind=kind, mode=mode, would_action=ACTION_ERROR,
                reason=f"unknown policy kind {kind!r} (no registered evaluator)",
                enforcement_gated=gated, details={},
            )
        try:
            outcome = _call_evaluator(evaluator, policy, request, self.context)
        except Exception as exc:  # noqa: BLE001 — one bad policy never breaks the rest
            return PolicyEvaluation(
                policy_name=name, kind=kind, mode=mode, would_action=ACTION_ERROR,
                reason=f"evaluator raised: {exc}", enforcement_gated=gated, details={},
            )
        return PolicyEvaluation(
            policy_name=name, kind=kind, mode=mode,
            would_action=outcome.would_action, reason=outcome.reason,
            enforcement_gated=gated, details=dict(outcome.details),
        )
