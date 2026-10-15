"""Mast alert channel. httpx is faked, nothing leaves the process."""
from __future__ import annotations

import logging
from typing import Any

import pytest

from tokenjam.core.alerts import MastChannel, _build_channel
from tokenjam.core.config import AlertChannelConfig
from tokenjam.core.models import Alert, AlertType, Severity
from tokenjam.utils.time_parse import utcnow

CHANNEL_URL = "https://mast.tissue.dev/mk_0123456789abcdef0123456789abcdef01234567"


class _FakeResponse:
    def __init__(self, status_code: int = 202, body: Any = None) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> Any:
        if self._body is None:
            raise ValueError("no JSON body")
        return self._body


class _FakeClient:
    """Stands in for httpx.Client, recording the one POST a send makes."""

    calls: list[dict[str, Any]] = []
    status_code = 202
    body: Any = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def post(self, url: str, json: dict[str, Any] | None = None, **kwargs: Any) -> _FakeResponse:
        type(self).calls.append({"url": url, "json": json})
        return _FakeResponse(type(self).status_code, type(self).body)


@pytest.fixture
def fake_http(monkeypatch: pytest.MonkeyPatch) -> type[_FakeClient]:
    _FakeClient.calls = []
    _FakeClient.status_code = 202
    _FakeClient.body = None
    monkeypatch.setattr("tokenjam.core.alerts.httpx.Client", _FakeClient)
    return _FakeClient


def _alert(
    severity: Severity = Severity.CRITICAL,
    detail: dict[str, Any] | None = None,
    agent_id: str | None = "my-agent",
    title: str = "Budget exceeded",
    alert_type: AlertType = AlertType.COST_BUDGET_DAILY,
) -> Alert:
    return Alert(
        alert_id="al_1",
        fired_at=utcnow(),
        type=alert_type,
        severity=severity,
        title=title,
        detail=detail if detail is not None else {"message": "daily spend over limit"},
        agent_id=agent_id,
    )


def _channel(**overrides: Any) -> MastChannel:
    config = AlertChannelConfig(type="mast", channel_url=CHANNEL_URL, **overrides)
    return MastChannel(config, include_captured_content=False)


# ── Factory registration ──────────────────────────────────────────────────

def test_build_channel_returns_a_mast_channel_for_the_mast_type():
    ch = _build_channel(
        AlertChannelConfig(type="mast", channel_url=CHANNEL_URL),
        include_captured_content=False,
    )
    assert isinstance(ch, MastChannel)
    assert ch.channel_url == CHANNEL_URL


def test_min_severity_is_set_for_the_dispatcher():
    # AlertDispatcher.dispatch() reads min_severity off the channel; without
    # it every severity gets through.
    ch = _channel(min_severity="critical")
    assert ch.min_severity is Severity.CRITICAL


# ── Priority mapping ──────────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("severity", "expected"),
    [
        (Severity.INFO, "quiet"),
        (Severity.WARNING, "normal"),
        (Severity.CRITICAL, "loud"),
    ],
)
def test_severity_maps_onto_a_mast_priority(fake_http, severity, expected):
    _channel().send(_alert(severity=severity))
    assert fake_http.calls[0]["json"]["priority"] == expected


def test_no_severity_maps_to_page(fake_http):
    # page repeats until acked and ignores quiet hours, so you have to ask
    # for it.
    for severity in (Severity.INFO, Severity.WARNING, Severity.CRITICAL):
        _channel().send(_alert(severity=severity))
    assert all(call["json"]["priority"] != "page" for call in fake_http.calls)


def test_configured_priority_overrides_the_severity_mapping(fake_http):
    _channel(priority="page").send(_alert(severity=Severity.INFO))
    assert fake_http.calls[0]["json"]["priority"] == "page"


def test_a_misspelt_priority_fails_at_startup():
    # Otherwise it's a 400 in the log the first time something fires.
    with pytest.raises(ValueError, match="quiet, normal, loud, page"):
        _channel(priority="high")


# ── Payload ───────────────────────────────────────────────────────────────

def test_send_posts_to_the_channel_url_with_title_and_body(fake_http):
    _channel().send(_alert())
    call = fake_http.calls[0]
    assert call["url"] == CHANNEL_URL
    assert call["json"]["title"] == "Budget exceeded"
    assert call["json"]["body"] == "daily spend over limit"


def test_dedupe_key_folds_repeats_of_the_same_agent_and_type(fake_http):
    _channel().send(_alert())
    assert fake_http.calls[0]["json"]["key"] == "cost_budget_daily:my-agent"


def test_dedupe_key_survives_an_alert_with_no_agent(fake_http):
    _channel().send(_alert(agent_id=None))
    assert fake_http.calls[0]["json"]["key"] == "cost_budget_daily:unknown"


def test_title_and_body_are_cut_to_mast_limits(fake_http):
    # Past the limit Mast answers 400 and the alert is lost.
    _channel().send(_alert(title="t" * 500, detail={"message": "b" * 9000}))
    payload = fake_http.calls[0]["json"]
    assert len(payload["title"]) == MastChannel.TITLE_MAXLEN
    assert len(payload["body"]) == MastChannel.BODY_MAXLEN


def test_a_channel_with_no_url_sends_nothing(fake_http):
    MastChannel(AlertChannelConfig(type="mast"), include_captured_content=False).send(_alert())
    assert fake_http.calls == []


# ── Critical Rule 5: captured-content stripping ───────────────────────────

def test_captured_content_is_stripped_by_default(fake_http):
    detail = {"prompt_content": "secret prompt", "tool_output": "secret output", "cost": 1.5}
    _channel().send(_alert(detail=detail))
    body = fake_http.calls[0]["json"]["body"]
    assert "secret prompt" not in body
    assert "secret output" not in body
    assert "1.5" in body


def test_captured_content_is_included_when_the_flag_is_set(fake_http):
    detail = {"prompt_content": "secret prompt", "cost": 1.5}
    config = AlertChannelConfig(type="mast", channel_url=CHANNEL_URL)
    MastChannel(config, include_captured_content=True).send(_alert(detail=detail))
    assert "secret prompt" in fake_http.calls[0]["json"]["body"]


# ── Delivery failure is reported ──────────────────────────────────────────

def test_a_rejected_send_logs_the_status_and_message(fake_http, caplog):
    # Mast's error body is {"error": {"code", "message"}}; the message is
    # the part that tells you it was a rotated key.
    fake_http.status_code = 404
    fake_http.body = {"error": {"code": "not_found", "message": "No such channel."}}
    with caplog.at_level(logging.WARNING, logger="tokenjam.core.alerts"):
        _channel().send(_alert())
    assert "rejected" in caplog.text
    assert "404" in caplog.text
    assert "No such channel." in caplog.text


def test_a_rejection_without_a_json_body_still_logs_the_status(fake_http, caplog):
    fake_http.status_code = 429
    fake_http.body = None
    with caplog.at_level(logging.WARNING, logger="tokenjam.core.alerts"):
        _channel().send(_alert())
    assert "429" in caplog.text


def test_a_5xx_is_not_logged_as_a_rejection(fake_http, caplog):
    # A 502 may have come after the message was stored. Call it rejected
    # and whoever reads the log retries and pages twice.
    fake_http.status_code = 502
    fake_http.body = {"error": {"code": "storage_error", "message": "The message could not be stored."}}
    with caplog.at_level(logging.WARNING, logger="tokenjam.core.alerts"):
        _channel().send(_alert())
    assert "502" in caplog.text
    assert "may or may not have gone out" in caplog.text
    assert "rejected" not in caplog.text


def test_an_accepted_send_logs_nothing(fake_http, caplog):
    fake_http.status_code = 202
    with caplog.at_level(logging.WARNING, logger="tokenjam.core.alerts"):
        _channel().send(_alert())
    assert caplog.text == ""
