"""Unit tests for gaffer/analytics/defensive_line.py."""
import pytest


class TestDefensiveLineTracker:
    def test_returns_second_to_last(self):
        from gaffer.analytics.defensive_line import compute_defensive_line
        positions_y = [5.0, 20.0, 35.0, 50.0, 60.0]
        result = compute_defensive_line(positions_y, ball_y=60.0)
        assert result == pytest.approx(50.0), (
            f"Expected second-to-last (50.0), got {result}"
        )

    def test_two_players_returns_first(self):
        from gaffer.analytics.defensive_line import compute_defensive_line
        result = compute_defensive_line([10.0, 30.0], ball_y=60.0)
        assert result == pytest.approx(10.0)

    def test_single_player_returns_none(self):
        from gaffer.analytics.defensive_line import compute_defensive_line
        result = compute_defensive_line([20.0], ball_y=60.0)
        assert result is None

    def test_empty_returns_none(self):
        from gaffer.analytics.defensive_line import compute_defensive_line
        result = compute_defensive_line([], ball_y=60.0)
        assert result is None
