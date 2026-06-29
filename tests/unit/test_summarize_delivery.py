"""Unit tests for the delivery layer (DEC-027/029): `claude -p` + `api`, and `summarize_via`.

`subprocess.run` / `httpx.post` are mocked in every test — nothing here launches a real `claude`
or hits the network. The fakes echo every `<tj-keep>` marker back from the wrapped prompt they were
fed, so the verdict exercises the real wrap/restore path. The MCP front door never reaches this
module (Claude rewrites in-session); these tests cover only the CLI's automated path.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tokenjam.core.config import StorageConfig, SummarizeConfig, TjConfig
from tokenjam.core.summarize.delivery import DeliveryError, deliver, summarize_via
from tokenjam.core.summarize.session import SummarizeRefused, read_staged

RUN = "tokenjam.core.summarize.delivery.subprocess.run"
POST = "httpx.post"
PROSE = "Always act carefully and never drop a required step when you respond. " * 30
_MARKER_RE = re.compile(r'<tj-keep id="\d+"[^>]*?(?:/>|>.*?</tj-keep>)', re.DOTALL)


@pytest.fixture
def cfg(tmp_path):
    """Config whose summarize anchor is tmp — staged results never touch the real ~/.tj."""
    return TjConfig(version="1", storage=StorageConfig(path=str(tmp_path / "t.duckdb")))


def _api_cfg(tmp_path, model="claude-opus-4-8"):
    """Config with an `api_model` set, for the `--via api` path."""
    return TjConfig(version="1", storage=StorageConfig(path=str(tmp_path / "t.duckdb")),
                    summarize=SummarizeConfig(api_model=model))


def _write(tmp_path, name, text) -> str:
    f = tmp_path / name
    f.write_text(text, encoding="utf-8")
    return str(f)


# --------------------------------------------------------------------------- #
# claude -p (subprocess)
# --------------------------------------------------------------------------- #

def _fake_claude(new_prose="Be careful; never skip a step.", *, returncode=0, stderr="",
                 stdout=None, mutate_path=None, mutate_text=""):
    """A `subprocess.run` stand-in. Echoes the markers it was fed (so restore() succeeds)."""
    def _run(cmd, *, input, capture_output, text, timeout=None):
        if mutate_path is not None:
            Path(mutate_path).write_text(mutate_text, encoding="utf-8")
        out = stdout if stdout is not None else new_prose + " " + " ".join(_MARKER_RE.findall(input))
        return SimpleNamespace(returncode=returncode, stdout=out, stderr=stderr)
    return _run


def test_summarize_via_claude_roundtrips_and_stages(cfg, tmp_path):
    path = _write(tmp_path, "CLAUDE.md", PROSE + "\n```\nkeep = 'me'\n```\n")
    with patch(RUN, _fake_claude()):
        result = summarize_via(cfg, path, "claude-p")
    assert result is not None and result.verdict.structure_ok and result.verdict.staged
    assert result.verdict.produced_by == "claude-p"
    assert result.amortization is None                # claude-p = subscription/local, no marginal $
    assert "keep = 'me'" in result.verdict.restored
    assert read_staged(cfg, path)["produced_by"] == "claude-p"


def test_summarize_via_feeds_claude_p_the_prep_output(cfg, tmp_path):
    """`claude -p` is fed exactly prep's `system_rules\\n\\nwrapped_prompt` (ratio plumbed through)."""
    from tokenjam.core.summarize import session
    path = _write(tmp_path, "CLAUDE.md", PROSE)
    prep = session.prepare(path=path, ratio=0.1)      # prepare is pure → identical rules+wrap to re-derive
    captured: dict = {}

    def _run(cmd, *, input, capture_output, text, timeout=None):
        captured["cmd"], captured["input"] = cmd, input
        return SimpleNamespace(returncode=0, stdout="Short. " + " ".join(_MARKER_RE.findall(input)), stderr="")

    with patch(RUN, _run):
        summarize_via(cfg, path, "claude-p", ratio=0.1)
    assert captured["cmd"] == ["claude", "-p"]
    assert captured["input"] == f"{prep.system_rules}\n\n{prep.wrapped_prompt}"


def test_summarize_via_below_gate_skips_model_and_returns_note(cfg, tmp_path):
    path = _write(tmp_path, "tiny.md", "short prompt, only a few words here")
    with patch(RUN) as run:
        result = summarize_via(cfg, path, "claude-p")
    assert result.verdict is None and "gate" in (result.skipped_note or "")   # note from the one prep (#3)
    run.assert_not_called()                           # nothing worth summarizing → no spend


def test_via_claude_not_installed_points_at_api_and_manual(cfg, tmp_path):
    path = _write(tmp_path, "CLAUDE.md", PROSE)
    with patch(RUN, side_effect=FileNotFoundError()):
        with pytest.raises(DeliveryError) as exc:
            summarize_via(cfg, path, "claude-p")
    msg = str(exc.value)
    assert "isn't installed" in msg and "--via api" in msg and "manual mode" in msg
    assert read_staged(cfg, path) is None


def test_via_claude_nonzero_exit_raises(cfg, tmp_path):
    path = _write(tmp_path, "CLAUDE.md", PROSE)
    with patch(RUN, _fake_claude(returncode=2, stderr="boom")):
        with pytest.raises(DeliveryError) as exc:
            summarize_via(cfg, path, "claude-p")
    assert "exit 2" in str(exc.value) and "boom" in str(exc.value)
    assert read_staged(cfg, path) is None


def test_via_claude_timeout_raises(cfg, tmp_path):
    """A stuck `claude -p` (auth / permission / update) hits the timeout → DeliveryError, nothing staged (#2)."""
    path = _write(tmp_path, "CLAUDE.md", PROSE)
    timeout = subprocess.TimeoutExpired(cmd=["claude", "-p"], timeout=300)
    with patch(RUN, side_effect=timeout):
        with pytest.raises(DeliveryError, match="timed out"):
            summarize_via(cfg, path, "claude-p")
    assert read_staged(cfg, path) is None


def test_via_claude_empty_output_raises(cfg, tmp_path):
    path = _write(tmp_path, "CLAUDE.md", PROSE)
    with patch(RUN, _fake_claude(stdout="   ")):      # whitespace-only → nothing usable
        with pytest.raises(DeliveryError, match="returned nothing"):
            summarize_via(cfg, path, "claude-p")
    assert read_staged(cfg, path) is None


def test_deliver_unknown_mode_raises(cfg):
    with pytest.raises(DeliveryError, match="unknown delivery mode"):
        deliver(cfg, "bogus", "wrapped", "rules")     # claude-p + api are valid; anything else isn't


def test_summarize_via_refuses_on_drift_and_never_stages(cfg, tmp_path):
    """If the file changes during the model call, check()'s hash guard fires — nothing is staged."""
    path = _write(tmp_path, "CLAUDE.md", PROSE + "\n```\nx = 1\n```\n")
    with patch(RUN, _fake_claude(mutate_path=path, mutate_text="edited mid-flight")):
        with pytest.raises(SummarizeRefused, match="changed since"):
            summarize_via(cfg, path, "claude-p")
    assert read_staged(cfg, path) is None


def test_summarize_via_reports_progress(cfg, tmp_path):
    """`on_progress` fires at each phase so the CLI isn't silent during the (slow) model call."""
    path = _write(tmp_path, "CLAUDE.md", PROSE + "\n```\nk = 1\n```\n")
    seen: list[str] = []
    with patch(RUN, _fake_claude()):
        summarize_via(cfg, path, "claude-p", on_progress=seen.append)
    assert any("Wrapping" in m for m in seen)
    assert any("Rewriting" in m for m in seen)
    assert any("Verifying" in m for m in seen)


# --------------------------------------------------------------------------- #
# api (httpx) — DEC-029: own-key, real-charge cost, "pays for itself"
# --------------------------------------------------------------------------- #

def _fake_post(*, in_tok=1200, out_tok=400, status=200, text_override=None, body_text="boom"):
    """An `httpx.post` stand-in. Echoes markers from the wrapped prompt; reports usage for the cost."""
    def _post(url, *, timeout, headers, json):
        wrapped = json["messages"][0]["content"]
        text = (text_override if text_override is not None
                else "Be careful; never skip a step. " + " ".join(_MARKER_RE.findall(wrapped)))
        return SimpleNamespace(
            status_code=status, text=body_text,
            json=lambda: {"content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
                          "usage": {"input_tokens": in_tok, "output_tokens": out_tok}})
    return _post


def test_via_api_roundtrips_stages_and_amortizes(tmp_path, monkeypatch):
    monkeypatch.setenv("TJ_ANTHROPIC_API_KEY", "sk-test")
    cfg = _api_cfg(tmp_path)
    path = _write(tmp_path, "CLAUDE.md", PROSE + "\n```\nkeep = 'me'\n```\n")
    with patch(POST, _fake_post()):
        result = summarize_via(cfg, path, "api")
    assert result is not None and result.verdict.structure_ok and result.verdict.staged
    assert result.verdict.produced_by == "api"
    assert "keep = 'me'" in result.verdict.restored
    a = result.amortization
    assert a is not None and a.rewrite_usd > 0 and a.saving_usd_per_call > 0
    assert a.break_even_calls is not None and a.break_even_calls >= 1
    assert a.model == "claude-opus-4-8" and a.rates_known is True   # priced model → "real charge"
    assert read_staged(cfg, path)["produced_by"] == "api"


def test_via_api_unknown_model_flags_estimate(tmp_path, monkeypatch):
    """A model not in the pricing table → cost still computed (default rates) but flagged not-real (#4)."""
    monkeypatch.setenv("TJ_ANTHROPIC_API_KEY", "sk-test")
    cfg = _api_cfg(tmp_path, model="totally-made-up-model-2099")
    path = _write(tmp_path, "CLAUDE.md", PROSE + "\n```\nx = 1\n```\n")
    with patch(POST, _fake_post()):
        result = summarize_via(cfg, path, "api")
    assert result is not None and result.verdict.structure_ok
    a = result.amortization
    assert a is not None and a.rewrite_usd > 0        # computed from default rates
    assert a.rates_known is False                     # …but labeled an estimate, not "real charge"


def test_via_api_failed_structure_still_reports_rewrite_cost(tmp_path, monkeypatch):
    """A paid API rewrite can fail the gate; report spend, but no staged saving / break-even."""
    monkeypatch.setenv("TJ_ANTHROPIC_API_KEY", "sk-test")
    cfg = _api_cfg(tmp_path)
    path = _write(tmp_path, "CLAUDE.md", PROSE + "\n```\nx = 1\n```\n")
    with patch(POST, _fake_post(text_override="summary with no structure markers")):
        result = summarize_via(cfg, path, "api")
    assert result is not None
    assert result.verdict.structure_ok is False and result.verdict.staged is False
    a = result.amortization
    assert a is not None and a.rewrite_usd > 0
    assert a.rates_known is True
    assert a.saving_usd_per_call == 0.0
    assert a.break_even_calls is None
    assert read_staged(cfg, path) is None


def test_via_api_missing_usage_uses_no_amortization_fallback(tmp_path, monkeypatch):
    """If Anthropic returns text but no usage, cost is unknown — do not fabricate $0 or defaults."""
    monkeypatch.setenv("TJ_ANTHROPIC_API_KEY", "sk-test")
    cfg = _api_cfg(tmp_path)
    path = _write(tmp_path, "CLAUDE.md", PROSE + "\n```\nx = 1\n```\n")

    def _post(url, *, timeout, headers, json):
        wrapped = json["messages"][0]["content"]
        return SimpleNamespace(
            status_code=200, text="",
            json=lambda: {"content": [{"type": "text",
                                       "text": "Careful. " + " ".join(_MARKER_RE.findall(wrapped))}],
                          "stop_reason": "end_turn"})

    with patch(POST, _post):
        result = summarize_via(cfg, path, "api")
    assert result is not None and result.verdict.structure_ok
    assert result.amortization is None and result.cost_unknown is True   # api billed, usage absent (#2)


def test_via_api_truncated_response_rejected(tmp_path, monkeypatch):
    """stop_reason=max_tokens is a truncated summary — reject it even with ALL markers present (a
    prose-only prompt would have no markers to fail the structure gate). (#1)"""
    monkeypatch.setenv("TJ_ANTHROPIC_API_KEY", "sk-test")
    cfg = _api_cfg(tmp_path)
    path = _write(tmp_path, "CLAUDE.md", PROSE + "\n```\nx = 1\n```\n")

    def _post(url, *, timeout, headers, json):
        wrapped = json["messages"][0]["content"]
        markers = _MARKER_RE.findall(wrapped)         # echo every marker → structure would otherwise pass
        return SimpleNamespace(
            status_code=200, text="",
            json=lambda: {"content": [{"type": "text", "text": "Careful. " + " ".join(markers)}],
                          "stop_reason": "max_tokens",
                          "usage": {"input_tokens": 100, "output_tokens": 8192}})

    with patch(POST, _post):
        with pytest.raises(DeliveryError, match="max_tokens|truncated") as exc:
            summarize_via(cfg, path, "api")
    assert "billed" in str(exc.value)              # B-light: the rejected call cost money — say so
    assert read_staged(cfg, path) is None


def test_via_api_non_end_turn_stop_reason_rejected(tmp_path, monkeypatch):
    """A non-end_turn stop (e.g. refusal) on a PROSE-ONLY prompt has no markers to fail the structure
    gate — the stop_reason allowlist rejects it before staging. (round-3 #1)"""
    monkeypatch.setenv("TJ_ANTHROPIC_API_KEY", "sk-test")
    cfg = _api_cfg(tmp_path)
    path = _write(tmp_path, "notes.md", PROSE)        # prose only — NO structure, so NO markers to catch it

    def _post(url, *, timeout, headers, json):
        return SimpleNamespace(
            status_code=200, text="",
            json=lambda: {"content": [{"type": "text", "text": "I can't help with that."}],
                          "stop_reason": "refusal",
                          "usage": {"input_tokens": 100, "output_tokens": 20}})

    with patch(POST, _post):
        with pytest.raises(DeliveryError, match="end_turn|didn't complete") as exc:
            summarize_via(cfg, path, "api")
    assert "billed" in str(exc.value)              # B-light
    assert read_staged(cfg, path) is None


def test_via_api_no_key_refuses_without_network(tmp_path, monkeypatch):
    monkeypatch.delenv("TJ_ANTHROPIC_API_KEY", raising=False)
    cfg = _api_cfg(tmp_path)
    path = _write(tmp_path, "CLAUDE.md", PROSE)
    with patch(POST) as post:
        with pytest.raises(DeliveryError, match="TJ_ANTHROPIC_API_KEY"):
            summarize_via(cfg, path, "api")
    post.assert_not_called()                          # no key → never hits the network
    assert read_staged(cfg, path) is None


def test_via_api_no_model_refuses_without_network(tmp_path, monkeypatch):
    monkeypatch.setenv("TJ_ANTHROPIC_API_KEY", "sk-test")
    cfg = TjConfig(version="1", storage=StorageConfig(path=str(tmp_path / "t.duckdb")))  # no api_model
    path = _write(tmp_path, "CLAUDE.md", PROSE)
    with patch(POST) as post:
        with pytest.raises(DeliveryError, match="api_model"):
            summarize_via(cfg, path, "api")
    post.assert_not_called()
    assert read_staged(cfg, path) is None


def test_via_api_http_error_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("TJ_ANTHROPIC_API_KEY", "sk-test")
    cfg = _api_cfg(tmp_path)
    path = _write(tmp_path, "CLAUDE.md", PROSE)
    with patch(POST, _fake_post(status=500, body_text="server boom")):
        with pytest.raises(DeliveryError, match="500"):
            summarize_via(cfg, path, "api")
    assert read_staged(cfg, path) is None


def test_via_api_empty_text_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("TJ_ANTHROPIC_API_KEY", "sk-test")
    cfg = _api_cfg(tmp_path)
    path = _write(tmp_path, "CLAUDE.md", PROSE)
    with patch(POST, _fake_post(text_override="")):
        with pytest.raises(DeliveryError, match="no text"):
            summarize_via(cfg, path, "api")
    assert read_staged(cfg, path) is None


def test_via_api_malformed_json_raises_delivery_error(tmp_path, monkeypatch):
    """A 200 with an unparseable body raises DeliveryError (not a raw decode error); stages nothing."""
    monkeypatch.setenv("TJ_ANTHROPIC_API_KEY", "sk-test")
    cfg = _api_cfg(tmp_path)
    path = _write(tmp_path, "CLAUDE.md", PROSE)

    def _bad_json():
        raise ValueError("Expecting value: line 1 column 1 (char 0)")

    def _post(url, *, timeout, headers, json):
        return SimpleNamespace(status_code=200, text="<html>oops</html>", json=_bad_json)

    with patch(POST, _post):
        with pytest.raises(DeliveryError, match="unparseable"):
            summarize_via(cfg, path, "api")
    assert read_staged(cfg, path) is None


def test_via_api_non_dict_json_raises_delivery_error(tmp_path, monkeypatch):
    """A 200 whose JSON isn't an object (e.g. a list) raises DeliveryError, not AttributeError."""
    monkeypatch.setenv("TJ_ANTHROPIC_API_KEY", "sk-test")
    cfg = _api_cfg(tmp_path)
    path = _write(tmp_path, "CLAUDE.md", PROSE)

    def _post(url, *, timeout, headers, json):
        return SimpleNamespace(status_code=200, text="[]", json=lambda: [])

    with patch(POST, _post):
        with pytest.raises(DeliveryError, match="unexpected JSON"):
            summarize_via(cfg, path, "api")
    assert read_staged(cfg, path) is None


def test_via_api_timeout_raises(tmp_path, monkeypatch):
    import httpx
    monkeypatch.setenv("TJ_ANTHROPIC_API_KEY", "sk-test")
    cfg = _api_cfg(tmp_path)
    path = _write(tmp_path, "CLAUDE.md", PROSE)
    with patch(POST, side_effect=httpx.TimeoutException("slow")):
        with pytest.raises(DeliveryError, match="timed out"):
            summarize_via(cfg, path, "api")
    assert read_staged(cfg, path) is None


def test_via_api_below_gate_skips_network(tmp_path, monkeypatch):
    monkeypatch.setenv("TJ_ANTHROPIC_API_KEY", "sk-test")
    cfg = _api_cfg(tmp_path)
    path = _write(tmp_path, "tiny.md", "short prompt, only a few words here")
    with patch(POST) as post:
        result = summarize_via(cfg, path, "api")
    assert result.verdict is None and "gate" in (result.skipped_note or "")   # note from the one prep (#3)
    post.assert_not_called()                          # below the prose floor → no spend


def test_via_api_refuses_on_drift_and_never_stages(tmp_path, monkeypatch):
    monkeypatch.setenv("TJ_ANTHROPIC_API_KEY", "sk-test")
    cfg = _api_cfg(tmp_path)
    path = _write(tmp_path, "CLAUDE.md", PROSE + "\n```\nx = 1\n```\n")

    def _post(url, *, timeout, headers, json):
        Path(path).write_text("edited mid-flight", encoding="utf-8")    # change the file during the call
        wrapped = json["messages"][0]["content"]
        return SimpleNamespace(
            status_code=200, text="",
            json=lambda: {"content": [{"type": "text", "text": "x " + " ".join(_MARKER_RE.findall(wrapped))}],
                          "stop_reason": "end_turn",
                          "usage": {"input_tokens": 100, "output_tokens": 50}})

    with patch(POST, _post):
        with pytest.raises(SummarizeRefused, match="changed since"):
            summarize_via(cfg, path, "api")
    assert read_staged(cfg, path) is None


# --------------------------------------------------------------------------- #
# [summarize] config wiring
# --------------------------------------------------------------------------- #

def test_summarize_config_parses_api_model(tmp_path):
    from tokenjam.core.config import load_config
    cfg_file = tmp_path / "tj.toml"
    cfg_file.write_text('version = "1"\n[summarize]\napi_model = "claude-opus-4-8"\n')
    cfg = load_config(str(cfg_file))
    assert cfg.summarize.api_model == "claude-opus-4-8"


def test_summarize_config_defaults_to_no_model():
    assert TjConfig(version="1").summarize.api_model is None   # no default — DEC-029 / DEF-010
