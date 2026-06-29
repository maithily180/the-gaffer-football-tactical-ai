"""
gaffer/output/minimap.py
────────────────────────
Render projected players onto the 2D pitch and composite the result as a corner
inset on the video frame.

Foot points (Detection.foot_point) are projected through the HomographyManager
to pitch metres, then to template pixels via PitchModel. Players are coloured by
team, goalkeepers ringed, referees grey, ball yellow.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from gaffer import config
from gaffer.calibration.homography_manager import HomographyManager
from gaffer.calibration.pitch_model import PitchModel
from gaffer.detection.detector import Detection

_TEAM_CLR = {0: config.TEAM_A_COLOR_BGR, 1: config.TEAM_B_COLOR_BGR}
_UNKNOWN_CLR = (160, 160, 160)


class MinimapRenderer:
    """Draws projected detections on the pitch template."""

    def __init__(self, manager: HomographyManager, pitch_model: PitchModel | None = None):
        self.mgr = manager
        self.pm = pitch_model or PitchModel()
        self._base = self.pm.draw_pitch()

    # ── Full-size pitch render ────────────────────────────────────────────────

    def render(self, detections: List[Detection], label: str | None = None,
              label_color: tuple[int, int, int] = (60, 220, 60)) -> np.ndarray:
        """Return a fresh pitch canvas with all projectable detections drawn."""
        img = self._base.copy()
        if not self.mgr.is_valid():
            self._banner(img, "CALIBRATION LOST", (60, 60, 230))
            return img

        for det in detections:
            anchor = det.center if det.class_name == "ball" else det.foot_point
            world = self.mgr.project(anchor)
            if world is None:
                continue
            x_m, y_m = world
            if not self.mgr.on_pitch(x_m, y_m):
                continue                       # skip crowd / spurious off-pitch points
            cx, cy = self.pm.pitch_to_pixels(x_m, y_m)
            self._draw_marker(img, det, cx, cy)

        if label:
            self._banner(img, label, label_color)
        return img

    def _draw_marker(self, img: np.ndarray, det: Detection, cx: int, cy: int) -> None:
        if det.class_name == "ball":
            cv2.circle(img, (cx, cy), 5, config.BALL_COLOR_BGR, -1)
            cv2.circle(img, (cx, cy), 5, (0, 0, 0), 1)
            return
        if det.class_name == "referee":
            cv2.circle(img, (cx, cy), 6, config.REFEREE_COLOR_BGR, -1)
            return
        clr = _TEAM_CLR.get(det.team_id, _UNKNOWN_CLR)
        cv2.circle(img, (cx, cy), 8, clr, -1)
        cv2.circle(img, (cx, cy), 8, (255, 255, 255), 1)
        if det.class_name == "goalkeeper":             # ring to distinguish GK
            cv2.circle(img, (cx, cy), 11, (0, 255, 0), 2)

    def _banner(self, img: np.ndarray, text: str, color: tuple[int, int, int] = (60, 220, 60)) -> None:
        cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(img, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    color, 2, cv2.LINE_AA)

    # ── Composite as a corner inset ───────────────────────────────────────────

    def composite(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        *,
        width: int = config.MINIMAP_WIDTH,
        alpha: float = config.MINIMAP_ALPHA,
        margin: int = 12,
        label: str | None = None,
        label_color: tuple[int, int, int] = (60, 220, 60),
        corner: str = "bottom_right",
    ) -> np.ndarray:
        """Draw the minimap and overlay it onto `frame` in `corner`
        (bottom_right | top_right | bottom_left | top_left)."""
        mini = self.render(detections, label=label, label_color=label_color)
        h0, w0 = mini.shape[:2]
        height = int(width * h0 / w0)
        mini = cv2.resize(mini, (width, height), interpolation=cv2.INTER_AREA)

        out = frame.copy()
        H, W = out.shape[:2]
        right = "right" in corner
        top = "top" in corner
        x2 = (W - margin) if right else (margin + width)
        x1 = x2 - width
        y1 = margin if top else (H - margin - height)
        y2 = y1 + height
        if x1 < 0 or y1 < 0:
            return out
        roi = out[y1:y2, x1:x2]
        cv2.addWeighted(mini, alpha, roi, 1 - alpha, 0, roi)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 1)
        return out
