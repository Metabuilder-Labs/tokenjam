"""Docs must advertise the plugin's slash commands namespaced, not bare.

Claude Code namespaces every plugin-provided command as `/<plugin-name>:<command>`,
where the prefix is the `name` field in `plugin.json` (not the repo name, and not
hand-picked by the docs author). The bare form (`/onboard`) matches nothing once
the plugin is installed: it was documented that way in both READMEs and only
caught by manual reproduction, not by any test.

This guard reads the real command set straight from `plugin/commands/*.md` and
the real prefix straight from `plugin/.claude-plugin/plugin.json`, so a renamed
plugin or an added/removed command is caught automatically rather than by
remembering to update a hand-kept list here.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLUGIN_DIR = _REPO_ROOT / "plugin"
_COMMANDS_DIR = _PLUGIN_DIR / "commands"
_PLUGIN_MANIFEST = _PLUGIN_DIR / ".claude-plugin" / "plugin.json"
_DOC_FILES = [_REPO_ROOT / "README.md", _PLUGIN_DIR / "README.md"]


def _plugin_prefix() -> str:
    manifest = json.loads(_PLUGIN_MANIFEST.read_text())
    return manifest["name"]


def _command_stems() -> list[str]:
    return sorted(p.stem for p in _COMMANDS_DIR.glob("*.md"))


def _bare_command_references(text: str, stem: str) -> list[str]:
    """Occurrences of `/<stem>` NOT preceded by `:` (i.e. not `...:<stem>`)."""
    pattern = re.compile(r"(?<![:\w])/" + re.escape(stem) + r"\b")
    return pattern.findall(text)


# --------------------------------------------------------------------------- #
# The guard has to be able to fail before it can be trusted to pass
# --------------------------------------------------------------------------- #

def test_the_guard_finds_at_least_one_command():
    """If extraction ever finds zero commands, the checks below pass
    vacuously — that must itself be a failure, not silent success."""
    stems = _command_stems()
    assert stems, f"no command files found under {_COMMANDS_DIR}"


def test_the_guard_flags_a_bare_command_reference():
    assert _bare_command_references("See /onboard for details.", "onboard") == [
        "/onboard"
    ]
    # namespaced form must not trip the bare-form detector
    assert _bare_command_references("See /tokenjam:onboard for details.", "onboard") == []


# --------------------------------------------------------------------------- #
# The actual docs
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "doc_path", _DOC_FILES, ids=[str(p.relative_to(_REPO_ROOT)) for p in _DOC_FILES]
)
def test_docs_use_the_namespaced_command_form(doc_path):
    prefix = _plugin_prefix()
    stems = _command_stems()
    assert stems, f"no command files found under {_COMMANDS_DIR}"
    text = doc_path.read_text()

    for stem in stems:
        namespaced = f"/{prefix}:{stem}"
        bare_hits = _bare_command_references(text, stem)
        assert not bare_hits, (
            f"{doc_path.relative_to(_REPO_ROOT)} documents the bare command "
            f"`/{stem}`, which Claude Code will not resolve — it must be "
            f"`{namespaced}` (prefix comes from plugin.json's `name`: "
            f"{prefix!r})."
        )
