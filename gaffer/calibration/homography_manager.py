"""
gaffer/calibration/homography_manager.py
─────────────────────────────────────────
Owns the current homography matrix and provides the single
    project(pixel_pt) -> (x_m, y_m) | None
interface the rest of the system uses. Keeps H state in one place instead of
scattered across the pipeline.

v0.5 scope: a single STATIC H loaded from a calibration JSON (produced by
scripts/collect_calibration.py). It is valid only while the camera roughly
matches the calibration frame. Camera-motion compensation / recompute-on-drift
is a later upgrade — until then is_valid() reflects only whether an H is loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gaffer import config
from gaffer.calibration.homography import HomographyEstimator


class HomographyManager:
    def __init__(self, H: np.ndarray | None = None, frame_idx: int | None = None):
        self._est = HomographyEstimator()
        self.H = np.asarray(H, dtype=np.float64) if H is not None else None
        self.calibration_frame = frame_idx

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_calibration(cls, json_path: str | Path) -> "HomographyManager":
        """
        Load from a calibration JSON. Uses the stored homography if present;
        otherwise recomputes it from image_points + config.PITCH_KEYPOINTS.
        """
        data = json.loads(Path(json_path).read_text())

        if data.get("homography") is not None:
            return cls(np.array(data["homography"], dtype=np.float64),
                       data.get("frame_idx"))

        # Recompute from points
        names = list(data["image_points"].keys())
        image_pts = np.array([data["image_points"][n] for n in names], dtype=np.float32)
        world_pts = np.array([config.PITCH_KEYPOINTS[n] for n in names], dtype=np.float32)
        est = HomographyEstimator()
        H, valid = est.compute(image_pts, world_pts)
        if not valid or H is None:
            raise ValueError(f"Could not compute a valid homography from {json_path}")
        return cls(H, data.get("frame_idx"))

    # ── Use ───────────────────────────────────────────────────────────────────

    def is_valid(self) -> bool:
        return self.H is not None

    def project(self, pixel_pt: tuple[float, float]) -> tuple[float, float] | None:
        """Pixel → pitch metres. None if no H."""
        return self._est.project(pixel_pt, self.H)

    def on_pitch(self, x_m: float, y_m: float, margin_m: float = 5.0) -> bool:
        """True if a projected point falls within the pitch (+ margin)."""
        return (-margin_m <= x_m <= config.PITCH_LENGTH_M + margin_m and
                -margin_m <= y_m <= config.PITCH_WIDTH_M + margin_m)
