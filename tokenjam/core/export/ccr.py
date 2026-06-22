"""
Generate a claude-code-router (CCR) advisory routing config from the downsize
analyzer.

Output shape — a JSONC document under a `tokenjam.routing_recommendations`
namespace, with the mandatory honesty comments (`//`) baked in. CCR's own
config (`~/.claude-code-router/config.json`) expresses *scenario* routing
(default / background / think / longContext), not original->alternative model
rules, so — exactly like `claude_code.py` — this is an advisory block the user
reads and translates into their CCR config by hand. There is no `--apply`:
TokenJam writes only to `~/.config/tokenjam/exports/` and never touches CCR's
config. Standard JSON parsers reject `//`, so the file is served as JSONC.

Plan-tier-aware (same doctrine as claude_code.py):
  - API users:          rules carry `estimated_savings_usd_month`.
  - Subscription/local: rules carry `estimated_tokens_freed` (never dollars
                        against a flat-rate plan).
  - Unknown plan-tier:  rules carry a reconfigure note, no figure.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from tokenjam.core.export.common import (
    EVIDENCE_LEVEL,
    caveat_lines,
    header_lines,
    rule_evidence_comment,
    select_figure_value,
)
from tokenjam.core.optimize.types import DowngradeFinding


def _comment_block(lines: list[str], indent: int) -> str:
    pad = " " * indent
    return "\n".join(f"{pad}// {line}" if line else f"{pad}//" for line in lines)


def render_ccr_config(
    *,
    downgrade: DowngradeFinding | None,
    pricing_mode: str,
    plan_tier: str,
    since: str,
    until: str,
    agent_id: str | None = None,
) -> str:
    """Return a JSONC string ready to write to disk for claude-code-router."""
    timestamp = datetime.now(tz=timezone.utc).isoformat()
    caveat = downgrade.caveat if downgrade is not None else (
        # No finding yet — still embed the canonical caveat so an empty export
        # carries the honesty framing.
        DowngradeFinding.__dataclass_fields__["caveat"].default
    )

    header = header_lines(
        target="ccr",
        generated_at=timestamp,
        since=since,
        until=until,
        agent_id=agent_id,
        plan_tier=plan_tier,
        pricing_mode=pricing_mode,
    )

    rule_blocks: list[str] = []
    if downgrade is not None and downgrade.suggestions:
        for original_model, alt_model in sorted(downgrade.suggestions.items()):
            key, value = select_figure_value(
                pricing_mode=pricing_mode,
                monthly_savings_usd=downgrade.monthly_savings_usd,
                monthly_tokens_in_candidates=downgrade.monthly_tokens_in_candidates,
            )
            figure_line = f'        {json.dumps(key)}: {json.dumps(value)}'
            rule_blocks.append(
                f"      // {rule_evidence_comment()}\n"
                "      {\n"
                f'        "match": {{"original_model": {json.dumps(original_model)}}},\n'
                f'        "suggested_model": {json.dumps(alt_model)},\n'
                f'        "evidence": {json.dumps(EVIDENCE_LEVEL)},\n'
                f"{figure_line}\n"
                "      }"
            )
    rules_array = ",\n".join(rule_blocks)

    return (
        "{\n"
        '  "tokenjam": {\n'
        '    "routing_recommendations": {\n'
        f"{_comment_block(header, 6)}\n"
        "      //\n"
        f"{_comment_block(caveat_lines(caveat), 6)}\n"
        f'      "target": "claude-code-router",\n'
        f'      "generated_at": {json.dumps(timestamp)},\n'
        f'      "derivation_window": {json.dumps(f"{since} -> {until}")},\n'
        f'      "plan_tier": {json.dumps(plan_tier)},\n'
        f'      "pricing_mode": {json.dumps(pricing_mode)},\n'
        f'      "evidence_level": {json.dumps(EVIDENCE_LEVEL)},\n'
        '      "rules": [\n'
        f"{rules_array}\n"
        "      ]\n"
        "    }\n"
        "  }\n"
        "}\n"
    )
