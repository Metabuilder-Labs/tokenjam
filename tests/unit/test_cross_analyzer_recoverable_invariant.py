"""Generic, analyzer-agnostic guard: the cost-analyzer registry must never
claim more of a window's spend than the window actually spent.

Critical Rule 27 (`.claude/rules/optimize-cost-figures.md`) already states the
principle per-pair ("name the exact span population a new recoverable figure
claims and prove no shipped analyzer already claims those rows") and
`tests/unit/test_rollup_subagent_downsize_dedup.py` pins it for two known
pairs (cache family; downsize/subagent; resend/downsize). Both are
per-analyzer-test-shaped: they can only catch an overlap between analyzers
they were specifically written to compare. This file is the generalisation
Rule 27's own text calls for — it iterates ``ANALYZER_REGISTRY`` rather than a
hardcoded pair list, so a THIRD (or Nth) analyzer added later that reaches
back into another analyzer's spans is caught by CI the first time its finding
and a sibling's both fire over the same window, not by a UI investigation.

Seeded window has three deliberately overlapping shapes on the SAME spans:
  - a session with a Task-dispatch subagent turn (trips ``downsize`` +
    ``subagent``, per the existing pinned pair)
  - a context-heavy in-thread session (trips ``resend``)
  - 20 sessions sharing one planning-call + tool-sequence skeleton (trips
    ``reuse`` AND ``script`` simultaneously — the same repeated structure is
    exactly what both cluster on, and neither one currently excludes the
    other's claim)
Two different models are used across disjoint session groups so the
window-wide check cannot pass by accident on a single-model window (a
per-(provider, model) bug can cancel out across models in a window-wide sum
even though it cannot within a single model's own total).

NOTE: `_collect_recoverable` (`tokenjam/api/routes/cost.py`) is deliberately
NOT covered here. Its `total_recoverable_usd` is a documented, non-additive
gross ceiling (`recoverable_additive: False`, `_recoverable_overlap_note`) —
see the A1 analyzer-audit comment above it — precisely because summing
overlapping analyzer angles can't be made honest by subtraction alone without
either re-deriving every pairwise overlap or under-stating a real, actionable
per-analyzer figure. `past_overspend_rollup` is the one total this file
guards: it is the canonical ADDITIVE figure every surface (Review inbox,
Dashboard hero, CLI, MCP) renders as an amount a reader can act on, so it is
the one that must never exceed reality.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tokenjam.core.config import CaptureConfig, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize import ANALYZER_REGISTRY, build_report
from tokenjam.core.optimize.analyzers.context_resend import MIN_SESSION_CONTEXT_TOKENS
from tokenjam.core.optimize.cost_proposals import (
    cost_proposals_from_report,
    past_overspend_rollup,
)
from tokenjam.otel.semconv import GenAIAttributes
from tests.factories import make_llm_span, make_session, make_tool_span

UTC = timezone.utc
BASE = datetime(2026, 5, 10, tzinfo=UTC)
SINCE = datetime(2026, 5, 1, tzinfo=UTC)
UNTIL = datetime(2026, 5, 30, tzinfo=UTC)

MODEL_A = "claude-opus-4-7"   # subagent/downsize + resend groups
MODEL_B = "claude-haiku-4-5"  # reuse/script cluster group


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def _config() -> TjConfig:
    return TjConfig(version="1", capture=CaptureConfig(prompts=True, tool_inputs=True))


def _seed_subagent_dispatch(db) -> None:
    """One session: small main thread + one premium-model Task dispatch —
    the shape `test_rollup_subagent_downsize_dedup.py` already pins for
    downsize/subagent, replicated here so the GENERIC test also covers it."""
    start = BASE
    db.upsert_session(make_session(session_id="dispatch", plan_tier="api"))
    db.insert_span(make_llm_span(
        model=MODEL_A, provider="anthropic",
        input_tokens=500, output_tokens=50, cost_usd=0.02,
        session_id="dispatch", sub_agent_id=None, start_time=start,
    ))
    db.insert_span(make_llm_span(
        model=MODEL_A, provider="anthropic",
        input_tokens=4_000, output_tokens=300, cost_usd=0.30,
        session_id="dispatch", sub_agent_id="researcher", start_time=start,
    ))


def _seed_context_heavy_session(db) -> None:
    """A main-thread session with growing cache-read context, the shape
    `resend` prices its offload/right-size claim over."""
    for i in range(4):
        db.upsert_session(make_session(session_id="heavy", plan_tier="api"))
        db.insert_span(make_llm_span(
            model=MODEL_A, provider="anthropic",
            input_tokens=MIN_SESSION_CONTEXT_TOKENS,
            cache_tokens=MIN_SESSION_CONTEXT_TOKENS * i,
            output_tokens=500, cost_usd=1.0,
            session_id="heavy", sub_agent_id=None,
            start_time=BASE + timedelta(hours=1, minutes=i),
        ))


def _seed_repeated_skeleton_cluster(db, *, count: int = 20) -> None:
    """`count` sessions sharing one planning-call + tool-sequence skeleton —
    the exact shape both `reuse` (cache-reuse over the repeated plan) and
    `script` (deterministic tool-call cluster) key off. Neither adapter
    currently excludes the other's claim, so this is the pair the test is
    actually expected to exercise."""
    tool_names = ["read", "edit", "run"]
    for i in range(count):
        sid = f"cluster-{i}"
        # `script` aggregates its cluster's cost/tokens off the SESSIONS
        # table (unlike downsize/subagent/resend, which read spans directly),
        # so the session record has to carry the same totals the plan span
        # below does or the cluster surfaces with a real `instances` count
        # but a zero dollar/token figure.
        db.upsert_session(make_session(
            session_id=sid, plan_tier="api",
            input_tokens=1_000, output_tokens=200, total_cost_usd=0.20,
        ))
        t0 = BASE + timedelta(hours=2, minutes=i)
        plan = make_llm_span(
            model=MODEL_B, provider="anthropic",
            session_id=sid, start_time=t0, cost_usd=0.20,
            input_tokens=1_000, output_tokens=200,
            extra_attributes={
                GenAIAttributes.PROMPT_CONTENT: "cut release on monday",
            },
        )
        db.insert_span(plan)
        for j, tn in enumerate(tool_names):
            ts = make_tool_span(tool_name=tn)
            ts.session_id = sid
            ts.start_time = t0 + timedelta(seconds=j + 1)
            # Raw dict, not a JSON-encoded string — `script`'s arg-shape
            # classifier reads the attribute as a dict directly (matches
            # `test_workflow_restructure._seed_deterministic_cluster`).
            ts.attributes = {GenAIAttributes.TOOL_INPUT: {"path": f"/repo/f{j}.py"}}
            db.insert_span(ts)


def _seed_window(db) -> None:
    _seed_subagent_dispatch(db)
    _seed_context_heavy_session(db)
    _seed_repeated_skeleton_cluster(db)


def _window_ground_truth(db) -> dict:
    """The window's real, measured totals — independent of any analyzer —
    window-wide and broken out per (provider, model)."""
    rows = db.conn.execute(
        "SELECT provider, model, "
        "SUM(input_tokens + output_tokens + cache_tokens + cache_write_tokens) AS tok, "
        "SUM(cost_usd) AS usd "
        "FROM spans WHERE start_time >= ? AND start_time <= ? "
        "GROUP BY provider, model",
        [SINCE, UNTIL],
    ).fetchall()
    per_model: dict[tuple[str, str], tuple[int, float]] = {}
    total_tok, total_usd = 0, 0.0
    for provider, model, tok, usd in rows:
        tok, usd = int(tok or 0), float(usd or 0.0)
        per_model[(provider, model)] = (tok, usd)
        total_tok += tok
        total_usd += usd
    return {"total_tokens": total_tok, "total_usd": total_usd, "per_model": per_model}


def _proposal_model_key(p) -> tuple[str, str] | None:
    """The single (provider, model) a proposal's claim can be pinned to, when
    its target/baseline names exactly one — several analyzers (downsize's
    window-wide card, script/reuse clusters spanning many sessions) make no
    single-model claim and are correctly excluded from the per-model check;
    the window-wide check still covers them."""
    for src in (p.target_key, p.baseline):
        model = src.get("model") if isinstance(src, dict) else None
        if isinstance(model, str) and model:
            provider = src.get("provider") if isinstance(src, dict) else None
            return (str(provider or "anthropic"), model)
    return None


def test_rollup_never_exceeds_the_windows_real_spend(db):
    """Window-wide: whatever combination of analyzers fires, the ONE additive
    total every surface renders (`past_overspend_rollup`) can claim at most
    what the window actually spent — never more, however many analyzers
    priced angles on the same underlying spans."""
    _seed_window(db)
    truth = _window_ground_truth(db)
    assert truth["total_tokens"] > 0 and truth["total_usd"] > 0

    report = build_report(
        db=db, config=_config(), since=SINCE, until=UNTIL,
        findings=list(ANALYZER_REGISTRY.keys()),
    )
    proposals = cost_proposals_from_report(report)
    fired = sorted({p.analyzer for p in proposals})
    # Sanity: the seeded shapes actually tripped the analyzers this test is
    # about. If a future refactor stops one of these from firing, the
    # invariant below would pass vacuously and stop meaning anything.
    assert "downsize" in fired or "subagent" in fired
    assert "reuse" in fired
    assert "script" in fired

    rollup = past_overspend_rollup(proposals)
    assert rollup["past_overspend_tokens"] <= truth["total_tokens"], (
        f"rollup claimed {rollup['past_overspend_tokens']} tokens across "
        f"{fired}, more than the window's real {truth['total_tokens']}"
    )
    assert rollup["past_overspend_usd"] <= truth["total_usd"] + 1e-6, (
        f"rollup claimed ${rollup['past_overspend_usd']} across {fired}, "
        f"more than the window's real ${truth['total_usd']:.4f}"
    )


def test_rollup_never_exceeds_real_spend_per_provider_model(db):
    """Per (provider, model): a window-wide pass can hide an overlap that
    happens to net out across two different models. Every proposal that
    names exactly one (provider, model) must sum to no more than that
    model's own real spend."""
    _seed_window(db)
    truth = _window_ground_truth(db)
    assert len(truth["per_model"]) >= 2, "fixture must mix models to be a real check"

    report = build_report(
        db=db, config=_config(), since=SINCE, until=UNTIL,
        findings=list(ANALYZER_REGISTRY.keys()),
    )
    proposals = cost_proposals_from_report(report)

    claimed: dict[tuple[str, str], dict[str, float]] = {}
    for p in proposals:
        key = _proposal_model_key(p)
        if key is None:
            continue
        acc = claimed.setdefault(key, {"usd": 0.0, "tokens": 0})
        if p.past_overspend_usd is not None:
            acc["usd"] += float(p.past_overspend_usd)
        if p.past_overspend_tokens is not None:
            acc["tokens"] += int(p.past_overspend_tokens)

    for key, acc in claimed.items():
        model_tok, model_usd = truth["per_model"].get(key, (0, 0.0))
        assert acc["tokens"] <= model_tok, (
            f"{key}: claimed {acc['tokens']} tokens, model only saw {model_tok}"
        )
        assert acc["usd"] <= model_usd + 1e-6, (
            f"{key}: claimed ${acc['usd']}, model only spent ${model_usd:.4f}"
        )
