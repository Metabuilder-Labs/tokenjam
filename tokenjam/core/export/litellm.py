"""
Generate a LiteLLM router advisory config from the downsize analyzer.

Output shape — a YAML document with the mandatory honesty comments (`#`) baked
in (YAML supports native comments, so no JSONC trick is needed here). The
recommendations live under a `tokenjam_routing_recommendations` key rather than
LiteLLM's own `model_list` / `router_settings`, because a downgrade
*recommendation* is not a drop-in LiteLLM routing primitive — like
`claude_code.py` and `ccr.py`, this is an advisory block the user reads and
translates into their LiteLLM config by hand. No `--apply`: TokenJam writes
only to `~/.config/tokenjam/exports/` and never touches the LiteLLM config.

Plan-tier-aware (same doctrine as claude_code.py):
  - API users:          rules carry `estimated_savings_usd_month`.
  - Subscription/local: rules carry `estimated_tokens_freed`.
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
from tokenjam.core.optimize.types import MODEL_DOWNGRADE_CAVEAT, DowngradeFinding


def _comment_block(lines: list[str], indent: int = 0) -> str:
    pad = " " * indent
    return "\n".join(f"{pad}# {line}" if line else f"{pad}#" for line in lines)


def _yaml_scalar(value: object) -> str:
    """Render a Python value as a quoted YAML scalar (numbers stay bare)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    # json.dumps gives a correctly-escaped double-quoted string — valid YAML.
    return json.dumps(str(value))


def render_litellm_config(
    *,
    downgrade: DowngradeFinding | None,
    pricing_mode: str,
    plan_tier: str,
    since: str,
    until: str,
    agent_id: str | None = None,
) -> str:
    """Return a YAML string ready to write to disk for the LiteLLM router."""
    timestamp = datetime.now(tz=timezone.utc).isoformat()
    caveat = downgrade.caveat if downgrade is not None else MODEL_DOWNGRADE_CAVEAT

    header = header_lines(
        target="litellm",
        generated_at=timestamp,
        since=since,
        until=until,
        agent_id=agent_id,
        plan_tier=plan_tier,
        pricing_mode=pricing_mode,
    )

    lines: list[str] = []
    lines.append(_comment_block(header))
    lines.append("#")
    lines.append(_comment_block(caveat_lines(caveat)))
    lines.append("tokenjam_routing_recommendations:")
    lines.append('  target: "litellm"')
    lines.append(f"  generated_at: {_yaml_scalar(timestamp)}")
    lines.append(f"  derivation_window: {_yaml_scalar(f'{since} -> {until}')}")
    lines.append(f"  plan_tier: {_yaml_scalar(plan_tier)}")
    lines.append(f"  pricing_mode: {_yaml_scalar(pricing_mode)}")
    lines.append(f"  evidence_level: {_yaml_scalar(EVIDENCE_LEVEL)}")
    lines.append("  rules:")

    if downgrade is not None and downgrade.suggestions:
        for original_model, alt_model in sorted(downgrade.suggestions.items()):
            key, value = select_figure_value(
                pricing_mode=pricing_mode,
                monthly_savings_usd=downgrade.monthly_savings_usd,
                monthly_tokens_in_candidates=downgrade.monthly_tokens_in_candidates,
            )
            lines.append(f"    # {rule_evidence_comment()}")
            lines.append("    - match:")
            lines.append(f"        original_model: {_yaml_scalar(original_model)}")
            lines.append(f"      suggested_model: {_yaml_scalar(alt_model)}")
            lines.append(f"      evidence: {_yaml_scalar(EVIDENCE_LEVEL)}")
            lines.append(f"      {key}: {_yaml_scalar(value)}")
    else:
        lines.append("    []")

    return "\n".join(lines) + "\n"
