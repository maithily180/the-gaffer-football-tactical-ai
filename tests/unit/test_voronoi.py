"""Unit tests for gaffer/analytics/voronoi.py."""
import numpy as np
import pytest


class TestVoronoiComputer:
    def test_areas_sum_to_100(self, symmetric_positions):
        from gaffer.analytics.voronoi import compute_voronoi_control
        team_a, team_b = symmetric_positions
        result = compute_voronoi_control(team_a, team_b)
        total = result["teamA_pct"] + result["teamB_pct"]
        assert abs(total - 100.0) < 0.5, f"Areas sum to {total:.2f}%, expected ~100%"

    def test_symmetric_gives_roughly_50_50(self, symmetric_positions):
        from gaffer.analytics.voronoi import compute_voronoi_control
        team_a, team_b = symmetric_positions
        result = compute_voronoi_control(team_a, team_b)
        assert abs(result["teamA_pct"] - 50.0) < 10.0, (
            f"Expected ~50% but got {result['teamA_pct']:.1f}%"
        )

    def test_one_sided_dominance(self):
        from gaffer.analytics.voronoi import compute_voronoi_control
        # Team A completely dominates right half
        team_a = np.array([[60 + i*3, 10 + j*10] for i in range(4) for j in range(3)],
                          dtype=float)[:11]
        team_b = np.array([[10, 34]], dtype=float)  # single player deep
        result = compute_voronoi_control(team_a, team_b)
        assert result["teamA_pct"] > result["teamB_pct"], "Team A should dominate"

    def test_returns_required_keys(self, symmetric_positions):
        from gaffer.analytics.voronoi import compute_voronoi_control
        team_a, team_b = symmetric_positions
        result = compute_voronoi_control(team_a, team_b)
        assert "teamA_pct" in result
        assert "teamB_pct" in result
        assert "cells" in result
