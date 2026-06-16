"""Unit tests for gaffer/analytics/pressing.py."""
import numpy as np
import pytest


class TestPressingIntensityComputer:
    def test_no_opponents_means_zero_intensity(self):
        from gaffer.analytics.pressing import compute_pressing_intensity
        result = compute_pressing_intensity(
            ball_pos=(52.5, 34.0),
            ball_team="teamA",
            all_positions={"teamA": [(52.5, 34.0)], "teamB": []},
        )
        assert result["intensity"] == 0

    def test_three_opponents_within_radius(self):
        from gaffer.analytics.pressing import compute_pressing_intensity
        # 3 opponents 5m away from ball
        ball = (52.5, 34.0)
        pressers = [(52.5, 39.0), (57.5, 34.0), (52.5, 29.0)]  # 5m each
        result = compute_pressing_intensity(
            ball_pos=ball,
            ball_team="teamA",
            all_positions={"teamA": [ball], "teamB": pressers},
            radius_m=10.0,
        )
        assert result["intensity"] == 3

    def test_opponents_outside_radius_not_counted(self):
        from gaffer.analytics.pressing import compute_pressing_intensity
        ball = (52.5, 34.0)
        far_opponents = [(30.0, 34.0), (20.0, 10.0)]  # >20m away
        result = compute_pressing_intensity(
            ball_pos=ball,
            ball_team="teamA",
            all_positions={"teamA": [ball], "teamB": far_opponents},
            radius_m=10.0,
        )
        assert result["intensity"] == 0

    def test_intensity_capped_at_five(self):
        from gaffer.analytics.pressing import compute_pressing_intensity
        ball = (52.5, 34.0)
        # 8 opponents all within 3m
        pressers = [(52.5 + i*0.5, 34.0) for i in range(8)]
        result = compute_pressing_intensity(
            ball_pos=ball,
            ball_team="teamA",
            all_positions={"teamA": [ball], "teamB": pressers},
            radius_m=10.0,
        )
        assert result["intensity"] <= 5, "Intensity should be capped at 5"
