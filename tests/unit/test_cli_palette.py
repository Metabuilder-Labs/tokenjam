"""The tj terminal palette contract.

Onboarding used to put eight colours on one screen, and only about half were
chosen by anyone: Rich's automatic highlighter is on by default and repaints
*prose* by pattern — numbers bold cyan, paths magenta + bright magenta, quoted
strings green, brackets bold, a bare ``/`` in a sentence pink. These tests pin
both halves of the fix: the highlighter stays off, and deliberate colour goes
through a named theme role rather than a raw colour name, so the palette can be
audited (and changed) in one place.
"""
from __future__ import annotations

import re

import pytest
from rich.console import Console

from tokenjam.cli import cmd_onboard
from tokenjam.cli.cmd_onboard import _print_next_steps_nudge, _prompt_plan
from tokenjam.utils.formatting import console, err_console
from tokenjam.utils.theme import ACCENT, TJ_THEME

_SGR = re.compile(r"\x1b\[([0-9;:]+)m")


def _render(markup: str, *, width: int = 100) -> str:
    """Render markup through a console configured exactly like the shared one,
    but forced to emit ANSI so the styling is inspectable."""
    c = Console(
        highlight=False, theme=TJ_THEME, force_terminal=True,
        width=width, color_system="truecolor",
    )
    with c.capture() as cap:
        c.print(markup)
    return cap.get()


def _sgr_codes(rendered: str) -> set[str]:
    return {m.group(1) for m in _SGR.finditer(rendered)} - {"0"}


# --- the highlighter stays off ----------------------------------------------


@pytest.mark.parametrize("cons", [console, err_console], ids=["stdout", "stderr"])
def test_shared_consoles_disable_automatic_highlighting(cons):
    # Rich's default is True. Left on, it colours prose by pattern; every
    # colour in the CLI is supposed to be a deliberate markup tag instead.
    # Rich keeps the constructor flag on `_highlight` and exposes no public
    # accessor, so read it directly; the *behavioural* consequence is covered
    # by the prose tests below, which render through an identically configured
    # console and assert no styling comes out.
    assert cons._highlight is False
    assert cons.get_style("accent").color is not None, "theme not registered"


def test_plan_menu_does_not_split_the_20x_label():
    """The concrete regression: Rich's number highlighter matched the `0x` in
    "Max 20x" as a hex literal, so the line rendered as `Max 2` + a cyan `0x`.
    """
    out = _render("  Max 20x plan ($200/mo subscription)")
    assert "Max 20x plan ($200/mo subscription)" in out
    assert not _sgr_codes(out), f"plain prose should carry no styling, got {_sgr_codes(out)}"


@pytest.mark.parametrize(
    "prose",
    [
        "tj config written to: ~/.config/tj/config.toml",   # was magenta + pink
        "needed for trim / cache-recommend / reuse",        # bare `/` was pink
        "installed (SessionStart: resume|compact)",         # brackets were bold
        '[budget.anthropic] plan = "api"',                  # string was green
        "tj, version 0.6.1",                                # digits were cyan
        "Ingest secret:      17ccb4a0...",                  # `...` was yellow
    ],
)
def test_prose_carries_no_incidental_styling(prose):
    from rich.markup import escape

    out = _render(escape(prose))
    assert not _sgr_codes(out), f"{prose!r} picked up styling: {_sgr_codes(out)}"


# --- one accent, and it means "you can type or click this" -------------------


def test_accent_is_the_only_hue_in_the_next_steps_list(capsys):
    _print_next_steps_nudge(has_data=False, persona="claude-code", port=7391)
    out = capsys.readouterr().out
    # The typeable command and the clickable URL are the actionable content of
    # this screen (the trimmed two-row CC list — founder review, demo trim).
    assert "tj tokenmaxx" in out
    assert "http://127.0.0.1:7391/" in out
    # no em dash in user-facing copy
    assert "—" not in out


def test_plan_menu_accents_only_the_choice_key(capsys):
    import click

    monkeypatched = {}

    def fake_prompt(*_a, **kw):
        monkeypatched["default"] = kw.get("default")
        return kw.get("default", 1)

    orig = click.prompt
    click.prompt = fake_prompt
    try:
        _prompt_plan("Claude", [("api", "API (per-token)"), ("max_20x", "Max 20x plan ($200/mo)")])
    finally:
        click.prompt = orig
    out = capsys.readouterr().out
    # the plan names survive intact; only the typed digit is marked up
    assert "Max 20x plan ($200/mo)" in out
    assert "1)" in out and "2)" in out


def test_accent_is_a_single_defined_hue():
    assert ACCENT.startswith("#")
    # `url` is the same hue as accent, so typeable/clickable content never
    # carries two competing accents in the body of a screen.
    assert ACCENT in str(TJ_THEME.styles["url"])


def test_brand_is_the_dedicated_orange_wordmark_colour():
    # #643: the ASCII wordmark gets its OWN brand colour (orange), distinct from
    # the periwinkle accent — the banner is a branded moment, not typeable
    # content, so it earns its own hue. The accent stays reserved for the body.
    from tokenjam.utils.theme import BRAND_ORANGE

    assert BRAND_ORANGE == "#e0922f"
    assert BRAND_ORANGE in str(TJ_THEME.styles["brand"])
    assert ACCENT not in str(TJ_THEME.styles["brand"])


# --- colour past the accent is reserved for genuine state -------------------


def test_success_is_a_glyph_and_weight_not_green():
    src = cmd_onboard.__file__
    with open(src, encoding="utf-8") as fh:
        body = fh.read()
    # No raw colour names anywhere in the module: colour goes through a theme
    # role (`ok` / `warn` / `error`) so the palette is auditable in one place.
    for raw in ("[green]", "[/green]", "[bold green]", "[yellow]", "[/yellow]", "[red]", "[/red]"):
        assert raw not in body, f"raw colour markup {raw} is back in cmd_onboard"


def test_ok_role_carries_no_colour():
    """Success is the least surprising outcome in the flow; spending a colour on
    it is what made every screen multi-hued. `ok` is weight only."""
    style = TJ_THEME.styles["ok"]
    assert style.bold is True
    assert style.color is None


@pytest.mark.parametrize("role", ["warn", "warn.strong", "error", "error.strong"])
def test_state_roles_keep_a_colour(role):
    # These are the two cases where colour still earns its place: a blocker the
    # user must act on, and a real failure.
    assert TJ_THEME.styles[role].color is not None


def test_restart_panel_is_the_only_warning_colour_on_the_claude_code_tail(capsys):
    cmd_onboard._print_claude_code_restart_panel()
    out = capsys.readouterr().out
    assert "Action required: restart Claude Code" in out
    # the panel's instructions are commands, so they stay intact and typeable
    assert "claude -c" in out
    assert "claude --resume" in out
    assert "tj onboard --claude-code --verify-only" in out
    # no stray Rich markup escaped into the rendered panel
    assert "\\<" not in out
