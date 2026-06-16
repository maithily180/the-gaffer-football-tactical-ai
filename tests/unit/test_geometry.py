"""Unit tests for gaffer/utils/geometry.py."""
import numpy as np
import pytest


class TestDistanceM:
    def test_zero_distance(self):
        from gaffer.utils.geometry import dist_m
        assert dist_m((0, 0), (0, 0)) == pytest.approx(0.0)

    def test_horizontal(self):
        from gaffer.utils.geometry import dist_m
        assert dist_m((0, 0), (3, 0)) == pytest.approx(3.0)

    def test_pythagorean(self):
        from gaffer.utils.geometry import dist_m
        assert dist_m((0, 0), (3, 4)) == pytest.approx(5.0)

    def test_numpy_arrays(self):
        from gaffer.utils.geometry import dist_m
        a = np.array([10.0, 20.0])
        b = np.array([13.0, 24.0])
        assert dist_m(a, b) == pytest.approx(5.0)
