"""Unit tests for gaffer/calibration/homography.py."""
import numpy as np
import pytest


class TestHomographyEstimator:
    def _make_H(self):
        """Build a simple H matrix from known correspondences."""
        import cv2
        image_pts = np.float32([
            [100, 400], [500, 400], [500, 100], [100, 100]
        ])
        world_pts = np.float32([
            [0, 0], [52.5, 0], [52.5, 34], [0, 34]
        ])
        H, _ = cv2.findHomography(image_pts, world_pts, cv2.RANSAC)
        return H, image_pts, world_pts

    def test_homography_computes(self):
        from gaffer.calibration.homography import HomographyEstimator
        estimator = HomographyEstimator()
        image_pts = np.float32([[100,400],[500,400],[500,100],[100,100]])
        world_pts = np.float32([[0,0],[52.5,0],[52.5,34],[0,34]])
        H, valid = estimator.compute(image_pts, world_pts)
        assert valid, "Homography should be valid for non-collinear points"
        assert H is not None
        assert H.shape == (3, 3)

    def test_project_known_point(self):
        from gaffer.calibration.homography import HomographyEstimator
        estimator = HomographyEstimator()
        image_pts = np.float32([[100,400],[500,400],[500,100],[100,100]])
        world_pts = np.float32([[0,0],[52.5,0],[52.5,34],[0,34]])
        H, _ = estimator.compute(image_pts, world_pts)

        # Project the top-left corner (should be near 0,0)
        result = estimator.project((100, 400), H)
        assert result is not None
        x, y = result
        assert abs(x - 0.0) < 2.0, f"X should be ~0 but got {x:.2f}"
        assert abs(y - 0.0) < 2.0, f"Y should be ~0 but got {y:.2f}"

    def test_invalid_for_collinear_points(self):
        from gaffer.calibration.homography import HomographyEstimator
        estimator = HomographyEstimator()
        # All points on same horizontal line — collinear, H is degenerate
        image_pts = np.float32([[100,400],[200,400],[300,400],[400,400]])
        world_pts = np.float32([[0,0],[10,0],[20,0],[30,0]])
        H, valid = estimator.compute(image_pts, world_pts)
        assert not valid, "Collinear points should return valid=False"
