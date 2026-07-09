"""OTLP body parsing utilities."""
from __future__ import annotations

import gzip
import io
import json
from typing import Any

from fastapi import Request

_MAX_DECOMPRESSED = 32 * 1024 * 1024  # 32 MB

def _decompress_gzip(data: bytes) -> bytes:
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as f:
        result = f.read(_MAX_DECOMPRESSED + 1)
    if len(result) > _MAX_DECOMPRESSED:
        raise ValueError("decompressed body exceeds size limit")
    return result

async def read_otlp_body(request: Request) -> Any:
    """Return the parsed JSON body, decompressing gzip if needed.

    Decompresses when Content-Encoding: gzip is set, or when the body starts
    with the gzip magic bytes (\x1f\x8b) as a fallback for exporters that
    compress without setting the header. Raises ValueError on any failure.
    """
    raw = await request.body()

    # 1. Honour explicit Content-Encoding header
    if request.headers.get("content-encoding", "").lower() == "gzip":
        try:
            raw = _decompress_gzip(raw)
        except Exception as exc:
            # gzip.decompress can raise OSError, EOFError, or zlib.error
            raise ValueError(f"gzip decompression failed: {exc}") from exc
    # 2. Sniff fallback — gzip magic bytes present but no Content-Encoding header
    elif raw[:2] == b"\x1f\x8b":
        try:
            raw = _decompress_gzip(raw)
        except Exception as exc:
            # gzip.decompress can raise OSError, EOFError, or zlib.error
            raise ValueError(
                f"body appears gzip-compressed but decompression failed: {exc}"
            ) from exc

    try:
        return json.loads(raw)
    except Exception as exc:
        raise ValueError(f"JSON decode failed: {exc}") from exc
