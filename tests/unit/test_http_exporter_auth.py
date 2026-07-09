"""`TjHttpExporter` Authorization-header construction.

Regression guard for the `tj ping` footgun where an empty ingest secret
produced ``Authorization: Bearer `` — an illegal HTTP header value (leading/
trailing whitespace is forbidden by RFC 9110). httpx/h11 raise
``LocalProtocolError: Illegal header value b'Bearer '`` at send time, so the
OTLP span export failed outright and `tj ping` reported "emitted … but not
confirmed received".

Contract enforced here:
- non-empty secret  -> ``Authorization: Bearer <secret>`` (Critical Rule 4)
- empty / missing   -> no ``Authorization`` header at all (never ``Bearer ``)
"""
from __future__ import annotations

import pytest

from tokenjam.sdk.http_exporter import TjHttpExporter

ENDPOINT = "http://127.0.0.1:7391/api/v1/spans"


def test_header_is_well_formed_bearer_when_secret_present() -> None:
    secret = "0af032cb1234567890abcdef"
    exporter = TjHttpExporter(ENDPOINT, secret)

    assert exporter._headers["Authorization"] == f"Bearer {secret}"
    # Well-formed: exactly one space, no trailing/leading whitespace in value.
    value = exporter._headers["Authorization"]
    assert value == value.strip()
    assert value.startswith("Bearer ")
    assert value[len("Bearer "):] == secret


@pytest.mark.parametrize("empty_secret", ["", None, " ", "   ", "\t", "\n"])
def test_no_authorization_header_when_secret_missing(empty_secret) -> None:
    # ``None`` can slip through if a caller passes an unset config field, and a
    # whitespace-only secret is treated as absent (#431) — otherwise
    # ``Bearer  `` (a stray space) is the same illegal header value.
    exporter = TjHttpExporter(ENDPOINT, empty_secret)  # type: ignore[arg-type]

    # Never emit an empty / whitespace Bearer — omit the header entirely.
    assert "Authorization" not in exporter._headers
    assert all(not v.startswith("Bearer ") for v in exporter._headers.values())
    assert exporter._headers.get("Content-Type") == "application/json"
