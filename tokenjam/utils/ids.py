import uuid


def new_uuid() -> str:
    return str(uuid.uuid4())


def new_trace_id() -> str:
    """Generate a 32-char hex trace ID (OTel format)."""
    return uuid.uuid4().hex


def new_span_id() -> str:
    """Generate a 16-char hex span ID (OTel format)."""
    return uuid.uuid4().hex[:16]
