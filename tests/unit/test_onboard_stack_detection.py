"""Bare `tj onboard` prints a stack-tailored instrument-your-agent snippet (#85).

Previously the outro always printed the same hardcoded `patch_anthropic()` +
`@watch()` snippet regardless of the project's actual dependencies. These
tests exercise the wiring in `cmd_onboard.py` end-to-end via the CLI, on top
of the pure-logic coverage in `test_onboard_detect.py`.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from tokenjam.cli.cmd_onboard import cmd_onboard


@pytest.fixture(autouse=True)
def _no_existing_config(monkeypatch):
    monkeypatch.setattr("tokenjam.cli.cmd_onboard.find_config_file", lambda: None)


def _run(tmp_path, manifest_name=None, manifest_text=None):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        if manifest_name:
            Path(manifest_name).write_text(manifest_text)
        res = runner.invoke(cmd_onboard, ["--no-daemon", "--budget", "0"], obj={})
        return res


def test_unknown_stack_falls_back_to_generic_anthropic_snippet(tmp_path):
    res = _run(tmp_path)
    assert res.exit_code == 0, res.output
    assert "from tokenjam.sdk.integrations.anthropic import patch_anthropic" in res.output
    assert "patch_anthropic()" in res.output


def test_langchain_project_prints_langchain_snippet_and_extra(tmp_path):
    res = _run(tmp_path, "requirements.txt", "langchain>=0.2\n")
    assert res.exit_code == 0, res.output
    assert "patch_langchain()" in res.output
    assert "from tokenjam.sdk.integrations.langchain import patch_langchain" in res.output
    assert "pip install 'tokenjam[langchain]'" in res.output
    # Generic Anthropic snippet should not also be shown once a match is found.
    assert "patch_anthropic()" not in res.output


def test_openai_pyproject_project_prints_openai_snippet(tmp_path):
    res = _run(tmp_path, "pyproject.toml", '[project]\ndependencies = ["openai>=1.0"]\n')
    assert res.exit_code == 0, res.output
    assert "patch_openai()" in res.output
    assert "from tokenjam.sdk.integrations.openai import patch_openai" in res.output


def test_multiple_matches_all_printed(tmp_path):
    res = _run(tmp_path, "requirements.txt", "anthropic\ncrewai\n")
    assert res.exit_code == 0, res.output
    assert "patch_anthropic()" in res.output
    assert "patch_crewai()" in res.output


def test_litellm_with_providers_prints_supersedes_note(tmp_path):
    res = _run(tmp_path, "requirements.txt", "litellm\nopenai\n")
    assert res.exit_code == 0, res.output
    assert "patch_litellm()" in res.output
    assert "supersede" in res.output.lower() or "alone covers" in res.output.lower()


def test_litellm_alone_has_no_supersedes_note(tmp_path):
    res = _run(tmp_path, "requirements.txt", "litellm\n")
    assert res.exit_code == 0, res.output
    assert "patch_litellm()" in res.output
    assert "alone covers" not in res.output.lower()
