"""Homography estimation and point projection."""

import cv2
import numpy as np


class HomographyEstimator:
    """
    Wraps cv2.findHomography and cv2.perspectiveTransform.
    Provides validity checking and collinearity detection.
    """

    REPROJECTION_ERROR_THRESHOLD = 5.0  # metres

    def compute(
        self,
        image_pts: np.ndarray,
        world_pts: np.ndarray,
        method: int = cv2.RANSAC,
    ) -> tuple[np.ndarray | None, bool]:
        """
        Compute homography matrix from N point correspondences.

        Args:
            image_pts: [N, 2] pixel coordinates
            world_pts: [N, 2] real-world coordinates (metres)
            method: cv2.RANSAC (default) or 0 for exact (4 points)

        Returns:
            H: [3, 3] homography matrix, or None if computation failed
            valid: bool — False if degenerate (collinear, insufficient inliers)
        """
        image_pts = np.asarray(image_pts, dtype=np.float32)
        world_pts = np.asarray(world_pts, dtype=np.float32)

        if len(image_pts) < 4:
            return None, False

        if self._are_collinear(image_pts):
            return None, False

        try:
            H, mask = cv2.findHomography(image_pts, world_pts, method, 5.0)
        except cv2.error:
            return None, False

        if H is None:
            return None, False

        # Check inlier ratio (RANSAC)
        if mask is not None:
            inlier_ratio = mask.sum() / len(mask)
            if inlier_ratio < 0.5:
                return H, False

        return H, True

    def project(
        self, pixel_pt: tuple[float, float], H: np.ndarray
    ) -> tuple[float, float] | None:
        """
        Project a pixel coordinate to world (pitch) coordinates.

        Returns:
            (x_m, y_m) in metres, or None if H is None
        """
        if H is None:
            return None
        pt = np.array([[[float(pixel_pt[0]), float(pixel_pt[1])]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, H)
        x, y = result[0][0]
        return (float(x), float(y))

    def project_to_image(
        self, world_pt: tuple[float, float], H: np.ndarray
    ) -> tuple[float, float] | None:
        """
        Inverse projection: world (pitch metres) → pixel, using H^-1.
        Returns None if H is None or non-invertible.
        """
        if H is None:
            return None
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return None
        pt = np.array([[[float(world_pt[0]), float(world_pt[1])]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, H_inv)
        x, y = result[0][0]
        return (float(x), float(y))

    def reprojection_error(
        self,
        image_pts: np.ndarray,
        world_pts: np.ndarray,
        H: np.ndarray,
    ) -> tuple[float, float, np.ndarray]:
        """
        Validate H by reprojecting the world landmarks back into the image
        (world → pixel via H^-1) and comparing to the clicked image points.

        This is the trust gate: hand-clicked calibration is only as good as its
        reprojection error. Measured in IMAGE PIXELS (what we clicked in).

        Returns:
            (mean_error_px, max_error_px, per_point_errors[N])
        """
        image_pts = np.asarray(image_pts, dtype=np.float64)
        world_pts = np.asarray(world_pts, dtype=np.float64)
        errs = np.full(len(world_pts), np.inf)
        for i, w in enumerate(world_pts):
            reproj = self.project_to_image((w[0], w[1]), H)
            if reproj is None:
                continue
            errs[i] = float(np.hypot(reproj[0] - image_pts[i][0],
                                     reproj[1] - image_pts[i][1]))
        return float(np.mean(errs)), float(np.max(errs)), errs

    def _are_collinear(self, pts: np.ndarray, tol: float = 1e-6) -> bool:
        """Return True if all points lie on a single line."""
        if len(pts) < 3:
            return True
        v1 = pts[1] - pts[0]
        for i in range(2, len(pts)):
            cross = abs(v1[0] * (pts[i][1] - pts[0][1]) - v1[1] * (pts[i][0] - pts[0][0]))
            if cross > tol:
                return False
        return True
