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


# A policy `kind` evaluator: pure function of (policy config, request) → outcome.
PolicyEvaluator = Callable[[Any, PolicyRequest], PolicyOutcome]

POLICY_REGISTRY: dict[str, PolicyEvaluator] = {}


def register_policy(kind: str) -> Callable[[PolicyEvaluator], PolicyEvaluator]:
    """Register an evaluator for a policy ``kind`` (mirrors the analyzer registry)."""
    def _decorator(fn: PolicyEvaluator) -> PolicyEvaluator:
        POLICY_REGISTRY[kind] = fn
        return fn
    return _decorator


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


class PolicyEngine:
    """Loads `[[policies]]` and evaluates eligible requests into envelopes."""

    def __init__(self, policies: list[Any]):
        # Only enabled policies participate.
        self.policies = [p for p in (policies or []) if getattr(p, "enabled", True)]

    @classmethod
    def from_config(cls, config: Any) -> "PolicyEngine":
        return cls(list(getattr(config, "policies", []) or []))

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
            outcome = evaluator(policy, request)
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
