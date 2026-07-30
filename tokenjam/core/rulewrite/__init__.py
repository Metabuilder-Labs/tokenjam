"""One rule-write lifecycle, shared by every analyzer whose fix is a rule.

``downsize``, ``resend``, ``subagent`` and ``relearn`` all end in the same
artifact: a block appended to a ``CLAUDE.md``. They used to hand-roll that
independently, each offering exactly one destination — the workspace/root file
— which is wrong in both directions: a root-level rule is re-sent in every
session of every project, including the ones that never exhibited the problem,
and it dilutes the one file whose dilution makes rules start being ignored.

``core/optimize/rule_placement`` answers WHERE from the working directories the
sessions already recorded, and ``core/optimize/write_budget`` nets each
destination's own standing cost — so a rule that reads net-negative against the
global file can flip to net-positive by landing in the right place. This
package is the lifecycle that follows: list -> stage -> check -> apply -> undo,
per destination, dry-run by default.
"""
from __future__ import annotations

from tokenjam.core.rulewrite.apply import apply_staged, check_staged, stage_rule, undo
from tokenjam.core.rulewrite.delivery import (
    DEFAULT_DELIVERY,
    DELIVERY_CLAUDE_MD_RULE,
    DELIVERY_KINDS,
    DeliveryKind,
    resolve_delivery,
)
from tokenjam.core.rulewrite.plan import (
    RULE_WRITING_ANALYZERS,
    find_rule,
    all_rule_writes,
    list_rule_writes,
)
from tokenjam.core.rulewrite.types import (
    RuleDestination,
    RuleWrite,
    RuleWriteRefused,
    StagedRuleWrite,
)

__all__ = [
    "DEFAULT_DELIVERY",
    "DELIVERY_CLAUDE_MD_RULE",
    "DELIVERY_KINDS",
    "DeliveryKind",
    "RULE_WRITING_ANALYZERS",
    "RuleDestination",
    "RuleWrite",
    "RuleWriteRefused",
    "StagedRuleWrite",
    "apply_staged",
    "check_staged",
    "find_rule",
    "all_rule_writes",
    "list_rule_writes",
    "resolve_delivery",
    "stage_rule",
    "undo",
]
