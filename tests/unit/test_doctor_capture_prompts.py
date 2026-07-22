"""Doctor "Prompt capture" check (E33 follow-up): `capture.prompts` now
defaults on, so this check flags the degraded case — off — rather than
silently letting `trim` / `cache-recommend` / `reuse` go dark with no
signal in `tj doctor`."""
from __future__ import annotations

from types import SimpleNamespace

from tokenjam.cli.cmd_doctor import _check_capture_prompts


def _config(prompts: bool):
    return SimpleNamespace(capture=SimpleNamespace(prompts=prompts))


def test_ok_when_prompts_captured():
    check = _check_capture_prompts(_config(True))
    assert check["level"] == "ok"


def test_info_when_prompts_off():
    check = _check_capture_prompts(_config(False))
    assert check["level"] == "info"
    assert "trim" in check["message"]
    assert "cache-recommend" in check["message"]
    assert "reuse" in check["message"]
    assert "capture.prompts = true" in check["message"]
