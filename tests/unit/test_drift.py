"""Unit tests for drift detection pure functions."""
from unittest.mock import MagicMock

from tokenjam.core.config import TjConfig
from tokenjam.core.drift import DriftDetector, jaccard_similarity, z_score
from tests.factories import make_session


class TestZScore:
    def test_standard_values(self):
        # value=12, mean=10, stddev=2 => z=1.0
        assert z_score(12.0, 10.0, 2.0) == 1.0

    def test_negative_z(self):
        assert z_score(8.0, 10.0, 2.0) == -1.0

    def test_zero_stddev_nonzero_deviation_returns_inf(self):
        assert z_score(100.0, 10.0, 0.0) == float("inf")

    def test_zero_stddev_zero_deviation_returns_zero(self):
        assert z_score(10.0, 10.0, 0.0) == 0.0

    def test_large_deviation(self):
        z = z_score(10000.0, 1000.0, 200.0)
        assert z == 45.0


class TestJaccardSimilarity:
    def test_identical_sets(self):
        assert jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint_sets(self):
        assert jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial_overlap(self):
        # intersection=1, union=3 => 1/3
        result = jaccard_similarity({"a", "b"}, {"b", "c"})
        assert abs(result - 1 / 3) < 0.001

    def test_both_empty(self):
        assert jaccard_similarity(set(), set()) == 1.0

    def test_one_empty(self):
        assert jaccard_similarity(set(), {"a"}) == 0.0

    def test_subset(self):
        # intersection=2, union=3 => 2/3
        result = jaccard_similarity({"a", "b"}, {"a", "b", "c"})
        assert abs(result - 2 / 3) < 0.001


class TestDriftDetectorAgentFallback:
    """Drift detection should work for agents that aren't explicitly configured."""

    def test_unconfigured_agent_builds_baseline(self):
        db = MagicMock()
        db.get_baseline.return_value = None
        db.get_completed_session_count.return_value = 10
        db.get_completed_sessions.return_value = [
            make_session(
                agent_id="ad-hoc-agent", session_id=f"s{i}",
                input_tokens=100, output_tokens=50, tool_call_count=2,
            ) for i in range(10)
        ]
        alert_engine = MagicMock()
        # Empty config — no [agents.<id>] block for "ad-hoc-agent"
        config = TjConfig(version="1")

        detector = DriftDetector(db=db, alert_engine=alert_engine, config=config)
        session = make_session(
            agent_id="ad-hoc-agent", session_id="latest",
            input_tokens=100, output_tokens=50, tool_call_count=2,
        )

        detector.on_session_end("ad-hoc-agent", session)

        # Baseline should have been built despite no agent config
        assert db.upsert_baseline.called


class TestApiBackendGetBaseline:
    def test_get_baseline_computed_at_fallback(self, monkeypatch):
        from tokenjam.core.api_backend import ApiBackend
        from datetime import datetime
        from tokenjam.utils.time_parse import utcnow

        backend = ApiBackend("http://localhost:8000")

        # Test case 1: computed_at is present in the API response
        mock_data = {
            "baseline": {
                "sessions_sampled": 5,
                "computed_at": "2026-06-25T19:00:00+00:00",
                "avg_input_tokens": 100
            }
        }
        monkeypatch.setattr(backend, "_get", lambda *args, **kwargs: mock_data)
        baseline = backend.get_baseline("test-agent")
        assert baseline is not None
        assert baseline.sessions_sampled == 5
        assert baseline.computed_at == datetime.fromisoformat("2026-06-25T19:00:00+00:00")

        # Test case 2: computed_at is missing from the API response
        mock_data_missing = {
            "baseline": {
                "sessions_sampled": 3,
                "avg_input_tokens": 50
            }
        }
        monkeypatch.setattr(backend, "_get", lambda *args, **kwargs: mock_data_missing)

        now = utcnow()
        baseline_missing = backend.get_baseline("test-agent")
        assert baseline_missing is not None
        assert baseline_missing.sessions_sampled == 3
        # Should be timezone-aware (tzinfo is not None)
        assert baseline_missing.computed_at.tzinfo is not None
        # Difference should be very small
        assert abs((baseline_missing.computed_at - now).total_seconds()) < 5

