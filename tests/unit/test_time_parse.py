from datetime import datetime, timezone, timedelta
from unittest.mock import patch
import pytest

from tj.utils.time_parse import parse_since, utcnow


class TestParseSinceRelative:
    def test_minutes(self):
        with patch("tj.utils.time_parse.utcnow") as mock_now:
            now = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)
            mock_now.return_value = now
            result = parse_since("30m")
            assert result == now - timedelta(minutes=30)

    def test_hours(self):
        with patch("tj.utils.time_parse.utcnow") as mock_now:
            now = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)
            mock_now.return_value = now
            result = parse_since("1h")
            assert result == now - timedelta(hours=1)

    def test_hours_large(self):
        with patch("tj.utils.time_parse.utcnow") as mock_now:
            now = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)
            mock_now.return_value = now
            result = parse_since("12h")
            assert result == now - timedelta(hours=12)

    def test_days(self):
        with patch("tj.utils.time_parse.utcnow") as mock_now:
            now = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)
            mock_now.return_value = now
            result = parse_since("7d")
            assert result == now - timedelta(days=7)

    def test_single_day(self):
        with patch("tj.utils.time_parse.utcnow") as mock_now:
            now = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)
            mock_now.return_value = now
            result = parse_since("1d")
            assert result == now - timedelta(days=1)


class TestParseSinceAbsolute:
    def test_date_only(self):
        result = parse_since("2026-03-01")
        assert result == datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)

    def test_iso_datetime_with_z(self):
        result = parse_since("2026-03-01T10:00:00+00:00")
        assert result == datetime(2026, 3, 1, 10, 0, 0, tzinfo=timezone.utc)

    def test_iso_datetime_roundtrip(self):
        dt = datetime(2026, 6, 15, 8, 30, 0, tzinfo=timezone.utc)
        iso_str = dt.isoformat()
        result = parse_since(iso_str)
        assert result == dt

    def test_date_only_is_utc(self):
        result = parse_since("2026-03-01")
        assert result.tzinfo == timezone.utc


class TestParseSinceInvalid:
    def test_empty_string(self):
        with pytest.raises(ValueError, match="Unrecognised"):
            parse_since("")

    def test_garbage(self):
        with pytest.raises(ValueError, match="Unrecognised"):
            parse_since("foobar")

    def test_invalid_unit(self):
        with pytest.raises(ValueError, match="Unrecognised"):
            parse_since("10x")

    def test_zero_amount(self):
        with pytest.raises(ValueError, match="amount must be > 0"):
            parse_since("0m")

    def test_whitespace_handled(self):
        with patch("tj.utils.time_parse.utcnow") as mock_now:
            now = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)
            mock_now.return_value = now
            result = parse_since("  30m  ")
            assert result == now - timedelta(minutes=30)


class TestUtcnow:
    def test_returns_aware_datetime(self):
        result = utcnow()
        assert result.tzinfo is not None

    def test_returns_utc(self):
        result = utcnow()
        assert result.tzinfo == timezone.utc
