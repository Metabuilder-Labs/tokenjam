from datetime import datetime, timedelta, timezone
import re


_RELATIVE_PATTERN = re.compile(r"^(\d+)([mhd])$")


def parse_since(value: str) -> datetime:
    """
    Parse a --since value and return the corresponding UTC datetime.
    Raises ValueError with a descriptive message for unrecognised formats.

    Supported formats:
    - 30m, 1h, 12h, 1d, 7d  (relative)
    - 2026-03-01             (date, treated as start of day UTC)
    - 2026-03-01T10:00:00Z   (ISO datetime)
    """
    value = value.strip()

    match = _RELATIVE_PATTERN.match(value)
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        if amount == 0:
            raise ValueError(f"Invalid --since value: {value!r} (amount must be > 0)")
        delta_map = {"m": "minutes", "h": "hours", "d": "days"}
        delta = timedelta(**{delta_map[unit]: amount})
        return utcnow() - delta

    # Try ISO datetime with timezone
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        pass

    # Try date-only format
    try:
        dt = datetime.strptime(value, "%Y-%m-%d")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass

    raise ValueError(
        f"Unrecognised --since format: {value!r}. "
        f"Expected: 30m, 1h, 7d, 2026-03-01, or 2026-03-01T10:00:00Z"
    )


def utcnow() -> datetime:
    """Return current UTC time, timezone-aware."""
    return datetime.now(tz=timezone.utc)
