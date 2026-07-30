"""Single source of truth for classifying an ``agent_id`` into a budget KIND:
a coding-tool GROUP (one row per tool, never per project or session) or an
SDK workflow (one row per literal ``agent_id``).

Why this exists: a real corpus renders one ``sessions.agent_id`` row per
Claude Code *project*, because that is how the id is minted
(``core/backfill.py::_agent_id_from_cwd`` derives ``claude-code-<cwd-basename>``
per project). From a budget-ceiling standpoint that is one tool, not N
projects, so this module collapses all of a tool's ids into one GROUP id
before any cap is applied or displayed.

Two coding-tool groups are recognized today, each via its OWN naming
convention — do not merge their matching rules, they come from different
runtimes with different id-minting behavior:

  - ``claude-code``: the bare ``"claude-code"`` id (used before
    ``_agent_id_from_cwd`` runs, or by onboarding paths that attach no cwd),
    plus every ``"claude-code-<project>"`` variant it derives. Prefix match.
  - ``codex``: Codex hardcodes ``service.name=codex_exec`` globally in the
    OTel resource it emits — there is no per-project variant to collapse, so
    the match is an EXACT id, not a prefix.

Anything that matches neither pattern is an SDK workflow: an arbitrary id a
caller declared via ``@watch()``. SDK workflows are never grouped — each
keeps its own row, keyed by its own ``agent_id``, exactly as configured.

This module is intentionally NOT the same thing as
``tokenjam.core.alerts.is_interactive_coding_agent``, an older, broader,
prefix-only boolial predicate (``"claude-code"`` OR ``"codex"`` prefix) with
five existing call sites (``alerts.py``, ``drift.py``, ``framing.py``,
``relearn_otel.py``, ``api/routes/status.py``) and a pinned margin-case test
(``tests/unit/test_alerts.py::test_is_interactive_coding_agent_margin_cases``,
which asserts ``"codex-cli-session"`` classifies as a coding agent under
THAT predicate). Retightening codex to an exact match there would silently
change behavior for all five unrelated call sites and break that pinned
test. Budget-group classification needs the tighter, more accurate rule
(Codex truly never varies its service name), so it lives here, scoped to the
one feature that needs it, rather than mutating the older shared predicate.
"""
from __future__ import annotations

from dataclasses import dataclass

# claude-code: bare id, or "claude-code-<project>" (core/backfill.py::_agent_id_from_cwd).
_CLAUDE_CODE_BARE_ID = "claude-code"
_CLAUDE_CODE_PROJECT_PREFIX = "claude-code-"

# codex: Codex hardcodes service.name=codex_exec globally — exact match only,
# no per-project variant exists to collapse.
_CODEX_EXACT_ID = "codex_exec"

# Ordered so "claude-code" always renders before "codex" when both are present.
CODING_AGENT_GROUPS: tuple[str, ...] = ("claude-code", "codex")


@dataclass(frozen=True)
class AgentKind:
    """Classification of one agent_id.

    ``group`` is set if and only if ``is_coding`` is True, and names the
    coding TOOL ("claude-code" / "codex") — never a project, never a session.
    An SDK workflow's identity IS its own agent_id; callers use ``agent_id``
    directly for that case rather than reading a group field here.
    """
    is_coding: bool
    group: str | None = None


def classify_agent_kind(agent_id: str | None) -> AgentKind:
    """Classify a single agent_id into a coding-tool group or an SDK workflow."""
    if not agent_id:
        return AgentKind(is_coding=False, group=None)
    if agent_id == _CLAUDE_CODE_BARE_ID or agent_id.startswith(_CLAUDE_CODE_PROJECT_PREFIX):
        return AgentKind(is_coding=True, group="claude-code")
    if agent_id == _CODEX_EXACT_ID:
        return AgentKind(is_coding=True, group="codex")
    return AgentKind(is_coding=False, group=None)


def is_coding_agent(agent_id: str | None) -> bool:
    """True for an agent_id belonging to a coding-tool group (see module docstring)."""
    return classify_agent_kind(agent_id).is_coding


def coding_group_id(agent_id: str | None) -> str | None:
    """The coding-tool group id for `agent_id`, or None (including for SDK workflows)."""
    return classify_agent_kind(agent_id).group


def group_agent_ids(agent_ids: list[str], group_id: str) -> list[str]:
    """The subset of `agent_ids` that belong to coding-tool group `group_id`."""
    return [a for a in agent_ids if classify_agent_kind(a).group == group_id]


def sdk_agent_ids(agent_ids: list[str]) -> list[str]:
    """The subset of `agent_ids` that are SDK workflows (not any coding group)."""
    return [a for a in agent_ids if not classify_agent_kind(a).is_coding]


def present_coding_groups(agent_ids: list[str]) -> list[str]:
    """Coding-tool groups that have at least one member in `agent_ids`, in
    `CODING_AGENT_GROUPS` order."""
    present = {classify_agent_kind(a).group for a in agent_ids}
    return [g for g in CODING_AGENT_GROUPS if g in present]
