"""
gaffer/calibration/pitch_model.py
─────────────────────────────────
Pure geometry of the 2D pitch — no computer vision here.

Owns the single mapping between real-world pitch coordinates (metres, origin at
the top-left corner, x = length 0..105, y = width 0..68) and template/minimap
pixel coordinates. Every consumer (generate_pitch_template.py, the minimap
renderer, the homography validation notebook) goes through this class so the
coordinate convention can never drift.
"""

from __future__ import annotations

import cv2
import numpy as np

from gaffer import config


class PitchModel:
    """
    Pitch landmarks + metres↔pixels conversion + a canonical pitch drawing.

    pitch coords:  (x_m, y_m)  x: 0=left goal line .. 105=right goal line
                                y: 0=top touchline  .. 68=bottom touchline
    pixel coords:  template/minimap canvas pixels (see config scale/margin)
    """

    LINE_COLOR = (255, 255, 255)
    PITCH_GREEN = (34, 139, 34)        # BGR forest green
    LINE_THICKNESS = 2

    def __init__(
        self,
        scale: int = config.PITCH_SCALE_PX_PER_M,
        margin: int = config.PITCH_MARGIN_PX,
        length_m: float = config.PITCH_LENGTH_M,
        width_m: float = config.PITCH_WIDTH_M,
    ):
        self.scale = scale
        self.margin = margin
        self.length_m = length_m
        self.width_m = width_m

    # ── Canvas geometry ─────────────────────────────────────────────────────

    @property
    def canvas_size(self) -> tuple[int, int]:
        """(width, height) of the pitch canvas in pixels."""
        w = int(self.length_m * self.scale) + 2 * self.margin
        h = int(self.width_m * self.scale) + 2 * self.margin
        return (w, h)

    # ── Coordinate conversion ───────────────────────────────────────────────

    def pitch_to_pixels(self, x_m: float, y_m: float) -> tuple[int, int]:
        """Pitch metres → template pixel (int)."""
        return (int(round(self.margin + x_m * self.scale)),
                int(round(self.margin + y_m * self.scale)))

    def pixels_to_pitch(self, px: float, py: float) -> tuple[float, float]:
        """Template pixel → pitch metres (inverse of pitch_to_pixels)."""
        return ((px - self.margin) / self.scale,
                (py - self.margin) / self.scale)

    # ── Landmarks ─────────────────────────────────────────────────────────────

    def get_landmarks(self) -> dict[str, tuple[float, float]]:
        """Named pitch landmarks in metres (from config.PITCH_KEYPOINTS)."""
        return dict(config.PITCH_KEYPOINTS)

    def world_point(self, name: str) -> tuple[float, float]:
        """World (metres) coordinate of a named landmark."""
        return config.PITCH_KEYPOINTS[name]

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw_pitch(self) -> np.ndarray:
        """
        Render the canonical 2D pitch (white lines on green). This is the single
        source of the pitch drawing — generate_pitch_template.py just saves it.
        """
        w, h = self.canvas_size
        canvas = np.full((h, w, 3), self.PITCH_GREEN, dtype=np.uint8)
        c, t = self.LINE_COLOR, self.LINE_THICKNESS
        p = self.pitch_to_pixels
        rad = lambda r_m: int(round(r_m * self.scale))  # noqa: E731

        # Outer boundary + halfway line
        cv2.rectangle(canvas, p(0, 0), p(105, 68), c, t)
        cv2.line(canvas, p(52.5, 0), p(52.5, 68), c, t)

        # Centre circle (9.15m) + spot
        cv2.circle(canvas, p(52.5, 34), rad(9.15), c, t)
        cv2.circle(canvas, p(52.5, 34), 3, c, -1)

        # Left penalty area / six-yard box / spot / arc / goal
        cv2.rectangle(canvas, p(0, 13.84), p(16.5, 54.16), c, t)
        cv2.rectangle(canvas, p(0, 24.84), p(5.5, 43.16), c, t)
        cv2.circle(canvas, p(11, 34), 3, c, -1)
        cv2.ellipse(canvas, p(11, 34), (rad(9.15), rad(9.15)), 0, -53, 53, c, t)
        cv2.rectangle(canvas, p(-2, 30.34), p(0, 37.66), c, t)

        # Right penalty area / six-yard box / spot / arc / goal
        cv2.rectangle(canvas, p(88.5, 13.84), p(105, 54.16), c, t)
        cv2.rectangle(canvas, p(99.5, 24.84), p(105, 43.16), c, t)
        cv2.circle(canvas, p(94, 34), 3, c, -1)
        cv2.ellipse(canvas, p(94, 34), (rad(9.15), rad(9.15)), 0, 127, 233, c, t)
        cv2.rectangle(canvas, p(105, 30.34), p(107, 37.66), c, t)

        # Corner arcs (1m)
        cv2.ellipse(canvas, p(0, 0),    (rad(1), rad(1)), 0,   0,  90, c, t)
        cv2.ellipse(canvas, p(105, 0),  (rad(1), rad(1)), 0,  90, 180, c, t)
        cv2.ellipse(canvas, p(0, 68),   (rad(1), rad(1)), 0, 270, 360, c, t)
        cv2.ellipse(canvas, p(105, 68), (rad(1), rad(1)), 0, 180, 270, c, t)

        return canvas
