"""Four shipped defects, each one an instance of the same framing failure.

`past_overspend_usd` is the maximum its analyzer knows: applying the fix should
ERASE the number it found. Each of these broke that in a different way — a key
the harness ignores, a file we declined to create, an action that does not
exist for the persona, and a fix whose own text says to do nothing.
"""
from __future__ import annotations

from datetime import timedelta

from tokenjam.utils.time_parse import utcnow


# --- D1: a frontmatter key the harness does not read -------------------------#

def test_the_effort_key_is_the_one_claude_code_actually_reads():
    """`reasoning_effort` is not a field Claude Code reads. The user pasted it,
    believed effort was pinned, and nothing changed — a silently-ignored key is
    worse than no line, because it converts an unfixed problem into one the
    user believes is fixed."""
    from tokenjam.core.optimize.cost_proposals import (
        AGENT_EFFORT_LEVELS,
        _rightsize_frontmatter_snippet,
    )

    snippet = _rightsize_frontmatter_snippet({
        "agent_name": "explorer", "proposed_model": "claude-haiku-4-5",
        "target_path": ".claude/agents/explorer.md",
        "over_provisioned": True,
    })
    assert "effort: low" in snippet
    assert "reasoning_effort" not in snippet
    assert "low" in AGENT_EFFORT_LEVELS


def test_effort_is_omitted_when_the_observation_does_not_support_a_value():
    """The old snippet hardcoded `low` for every dispatch, which on this
    analyzer's population is usually the WRONG guess: the flagged rows are the
    large, tool-heavy, long-output ones, so pinning them to low effort
    recommends making the expensive work worse. Omitting a line we cannot
    derive is the honest default."""
    from tokenjam.core.optimize.cost_proposals import _rightsize_frontmatter_snippet

    snippet = _rightsize_frontmatter_snippet({
        "agent_name": "worker", "proposed_model": "claude-haiku-4-5",
        "over_provisioned": False,
    })
    assert "effort:" not in snippet
    assert "model: claude-haiku-4-5" in snippet


# --- D2: declining to create a file we are allowed to create -----------------#

class _Row:
    def __init__(self, name: str, model: str = "claude-opus-4-7") -> None:
        self.session_id = "s1"
        self.sub_agent_id = "a" + "0" * 16
        self.sub_agent_type = name
        self.model = model
        self.provider = "anthropic"
        self.flags = ["over_powered"]
        self.input_tokens = 50_000
        self.output_tokens = 8_000
        self.cache_tokens = 0
        self.cache_write_tokens = 0
        self.cost_usd = 4.0


def test_a_builtin_dispatch_type_is_offered_a_file_to_CREATE(monkeypatch):
    """The big one. Every Claude Code dispatch is a built-in, none of which has
    a definition file, so requiring the file to EXIST made this branch
    unreachable on the dominant corpus and pushed the whole claim down to prose.
    A user subagent of the same name overrides the built-in and keeps its own
    `model`, so the file can simply be created — the product was not failing to
    find one, it was declining to create one."""
    from tokenjam.core.optimize import cost_proposals as cp

    monkeypatch.setattr(cp, "_session_cwds", lambda ids, config: {"s1": "/repo/alpha"})
    plumbing = cp._agent_model_plumbing([_Row("Explore")], config=object())

    assert plumbing, "a built-in must still yield a fix"
    assert plumbing["creates_file"] is True
    assert plumbing["agent_name"] == "Explore"
    assert plumbing["target_path"].endswith("Explore.md")


def test_a_capitalised_builtin_name_is_not_rejected_by_the_shape_check():
    """`Explore` and `Plan` are capitalised, and they are exactly the built-ins
    the define-an-override fix targets. A lowercase-only slug check silently
    rejected them, so the fix that needs them most could never reach its own
    branch. The dispatch-id guard is what does the real filtering."""
    from tokenjam.core.optimize.cost_proposals import _names_agent_definition

    for name in ("Explore", "Plan", "general-purpose", "fork"):
        assert _names_agent_definition(name), name
    # A per-dispatch id still must not be mistaken for a definition name.
    assert not _names_agent_definition("af8b26e872b7184a7")
    assert not _names_agent_definition("aw-ratehistory-7e1dd2a1642d7c29")


def test_the_create_card_explains_the_inherit_default(monkeypatch):
    """The waste's origin belongs in the evidence: `model` defaults to
    `inherit`, so an Opus-driven session hands every Task dispatch an Opus
    worker unless something pins otherwise. Nobody decided that."""
    from tokenjam.core.optimize import cost_proposals as cp

    monkeypatch.setattr(cp, "_session_cwds", lambda ids, config: {"s1": "/repo/alpha"})

    class _Finding:
        flagged = [_Row("Explore")]
        percent_of_cost = 0.4
        flagged_cost_usd = 100.0
        subagent_cost_usd = 200.0
        past_overspend_usd = 90.0
        past_overspend_tokens = 300_000
        estimate_basis = "basis"
        caveat = "caveat"

    prop = cp._subagent_to_proposals(_Finding(), config=object())[0]
    assert "inherit" in prop.advise_text
    assert "built-in" in prop.advise_text
    # The one-paste artifact is a CREATE, not an edit to a file the user lacks.
    assert prop.one_paste_fix.startswith("# Create ")
    assert "model: " in prop.one_paste_fix


# --- D3: an action that does not exist for this persona ----------------------#

def _agent_rows():
    from tokenjam.core.optimize.analyzers.downsize_agents import build_agent_price_rows

    return build_agent_price_rows([{
        "session_id": f"s{i}", "agent_id": "claude-code-tokenjam",
        "provider": "anthropic", "model": "claude-opus-4-7",
        "alt_model": "claude-haiku-4-5",
        "input_tokens": 1, "output_tokens": 48,
        "cache_tokens": 3142, "cache_write_tokens": 6716,
        "started_at": utcnow() - timedelta(days=1),
    } for i in range(32)], window_days=30.0)


def test_claude_code_is_never_told_to_redeploy_a_project_directory():
    """On Claude Code `agent_id` is `claude-code-<cwd-basename>`: it names a
    PROJECT DIRECTORY whose sessions are ephemeral. There is no service, no
    process, and no model id written down — "redeploy or restart the agent"
    names three things that do not exist."""
    from tokenjam.core.optimize.cost_proposals import _downsize_agent_proposals

    class _Finding:
        per_agent = _agent_rows()
        candidate_sessions = 32
        suggestions: dict = {}

    prop = _downsize_agent_proposals(_Finding(), config=None, persona="claude-code")[0]
    text = (prop.advise_text + " " + prop.one_paste_fix).lower()
    assert "redeploy" not in text
    assert "restart the agent" not in text
    assert "project directory, not a deployed service" in prop.advise_text


def test_the_persona_gate_withholds_the_offer_and_keeps_the_observation():
    """Critical Rule 32 again: the observation is just as real for a Claude
    Code window, so the figure is identical across personas. Only the
    redeploy-shaped OFFER differs."""
    from tokenjam.core.optimize.cost_proposals import _downsize_agent_proposals

    class _Finding:
        per_agent = _agent_rows()
        candidate_sessions = 32
        suggestions: dict = {}

    cc = _downsize_agent_proposals(_Finding(), config=None, persona="claude-code")[0]
    sdk = _downsize_agent_proposals(_Finding(), config=None, persona="sdk")[0]
    assert cc.past_overspend_usd == sdk.past_overspend_usd
    assert cc.past_overspend_tokens == sdk.past_overspend_tokens
    assert repr(cc.past_overspend_usd) == repr(sdk.past_overspend_usd)
    # An SDK service genuinely does redeploy, so it keeps that instruction.
    assert "redeploy" in sdk.one_paste_fix.lower()


# --- D4: a fix whose own text says to do nothing -----------------------------#

def test_an_advisory_only_family_never_offers_a_write():
    """The harness already errors clearly and agents self-correct next turn, so
    there is no action to sell. `is_placeholder_fix` cannot see this — the text
    is a real, specific, non-placeholder sentence that happens to say there is
    nothing to do."""
    from tokenjam.core.optimize.analyzers.relearn import _FAMILY_BY_KEY

    family = _FAMILY_BY_KEY["edit_before_read"]
    assert family["advisory_only"] is True
    assert "no rule or hook is needed" in family["fix"]


def test_only_the_swept_family_is_advisory_only():
    """The sweep result, pinned. If a later family acquires a do-nothing fix
    text, the catalog lint catches the text and this catches the flag."""
    from tokenjam.core.optimize.analyzers.relearn import _KNOWN_FAMILIES

    advisory = {f["key"] for f in _KNOWN_FAMILIES if f.get("advisory_only")}
    assert advisory == {"edit_before_read"}


# --- one finding, one fix, across surfaces -----------------------------------#

def test_compaction_is_never_listed_as_an_actionable_lever():
    """`/compact` is transient, persists nothing past the session, and users
    who feel a session filling already do it unprompted — it sits below the
    actionable ceiling. Listing it beside three durable levers implies it is
    the same kind of thing, and pads a list whose length is itself a cost to
    adherence."""
    from tokenjam.core.optimize.cost_proposals import (
        _DOWNSIZE_CC_LEVER,
        _DOWNSIZE_MIXED_CC_NOTE,
    )

    for text in (_DOWNSIZE_CC_LEVER, _DOWNSIZE_MIXED_CC_NOTE):
        assert "/compact" not in text
    # The durable levers survive — this is a removal, not a gutting.
    assert "tj route export" in _DOWNSIZE_CC_LEVER
    assert "tj optimize subagent" in _DOWNSIZE_CC_LEVER
    assert "CLAUDE.md" in _DOWNSIZE_CC_LEVER


def test_the_cli_and_the_card_lead_with_the_same_fix_for_one_finding():
    """They disagreed: the card led with the durable offload rule while the CLI
    led with `/compact`, for the identical finding, and the CLI led with the
    weaker one. Both now render through the same helper, so they cannot drift
    into different wordings again."""
    from tokenjam.core.optimize.cost_proposals import compound_offload_fix

    class _Finding:
        repeat_share = 0.6
        sessions_examined = 12
        repeat_tokens = 1_000_000
        fix_compaction = "Run /compact."
        fix_cache_control = ""
        fix_subagent_offload = "OFFLOAD-TEXT"
        fix_rightsize = "RIGHTSIZE-TEXT"
        past_overspend_usd = 10.0
        past_overspend_tokens = 100
        coverage_note = ""
        estimate_basis = ""
        caveat = ""

    from tokenjam.core.optimize.cost_proposals import _resend_to_proposals

    card = _resend_to_proposals(_Finding(), persona="claude-code")[0]
    shared = compound_offload_fix({}, "OFFLOAD-TEXT", "RIGHTSIZE-TEXT")
    # The card's advise leads with the shared compound fix...
    assert card.advise_text.startswith(shared)
    # ...and demotes compaction to the same explicitly-secondary position.
    assert "Immediate relief in an already-full session" in card.advise_text
    assert card.advise_text.index("OFFLOAD-TEXT") < card.advise_text.index("Run /compact.")
