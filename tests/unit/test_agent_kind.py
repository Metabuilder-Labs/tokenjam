"""Unit tests for tokenjam.core.agent_kind — the single source of truth for
classifying an agent_id into a coding-tool GROUP (claude-code / codex) or an
SDK workflow, used by both the budget API route and the alert engine's
group-scoped daily-cap enforcement.
"""
from __future__ import annotations

import pytest

from tokenjam.core.agent_kind import (
    AgentKind,
    classify_agent_kind,
    coding_group_id,
    group_agent_ids,
    is_coding_agent,
    present_coding_groups,
    sdk_agent_ids,
)


@pytest.mark.parametrize("agent_id,expected", [
    ("claude-code", AgentKind(is_coding=True, group="claude-code")),
    ("claude-code-my-project", AgentKind(is_coding=True, group="claude-code")),
    ("claude-code-tokenjam", AgentKind(is_coding=True, group="claude-code")),
    ("codex_exec", AgentKind(is_coding=True, group="codex")),
    # Codex hardcodes codex_exec globally — no per-project variant exists, so
    # a hypothetical "codex-something" must NOT collapse into the codex group.
    ("codex-myrepo", AgentKind(is_coding=False, group=None)),
    ("codex", AgentKind(is_coding=False, group=None)),
    ("billing-agent", AgentKind(is_coding=False, group=None)),
    ("sdk-workflow-7", AgentKind(is_coding=False, group=None)),
    (None, AgentKind(is_coding=False, group=None)),
    ("", AgentKind(is_coding=False, group=None)),
])
def test_classify_agent_kind_margin_cases(agent_id, expected):
    assert classify_agent_kind(agent_id) == expected


def test_is_coding_agent_matches_classify():
    assert is_coding_agent("claude-code-foo") is True
    assert is_coding_agent("codex_exec") is True
    assert is_coding_agent("sdk-thing") is False


def test_coding_group_id():
    assert coding_group_id("claude-code-foo") == "claude-code"
    assert coding_group_id("codex_exec") == "codex"
    assert coding_group_id("sdk-thing") is None


def test_group_agent_ids_filters_to_the_named_group():
    ids = ["claude-code-a", "claude-code-b", "codex_exec", "sdk-x", "claude-code"]
    assert set(group_agent_ids(ids, "claude-code")) == {
        "claude-code-a", "claude-code-b", "claude-code",
    }
    assert group_agent_ids(ids, "codex") == ["codex_exec"]


def test_sdk_agent_ids_excludes_both_coding_groups():
    ids = ["claude-code-a", "codex_exec", "sdk-x", "sdk-y"]
    assert set(sdk_agent_ids(ids)) == {"sdk-x", "sdk-y"}


def test_present_coding_groups_only_lists_groups_actually_seen():
    assert present_coding_groups(["claude-code-a", "sdk-x"]) == ["claude-code"]
    assert present_coding_groups(["codex_exec"]) == ["codex"]
    assert present_coding_groups(["sdk-x"]) == []
    # Order is always claude-code before codex when both present.
    assert present_coding_groups(["codex_exec", "claude-code"]) == ["claude-code", "codex"]
