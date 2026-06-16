from __future__ import annotations

from collections import deque
from typing import List

import cv2
import numpy as np

from gaffer import config
from gaffer.detection.detector import Detection

# ── Colour palette ────────────────────────────────────────────────────────────
# 32 visually-distinct BGR colours for track IDs
_TRACK_PALETTE: list[tuple[int, int, int]] = [
    (255, 56,  56), (56,  56, 255), (56, 200,  56), (200,  56, 200),
    (56,  200,200), (200,200,  56), (150,  0, 150), (0,  150, 150),
    (150,150,   0), (100,  0, 220), (0,  100, 220), (220,100,   0),
    (230,130,   0), (0,  130, 230), (130,  0, 230), (190,100, 100),
    (100,190, 100), (100,100, 190), (190,190, 100), (100,190, 190),
    (190,100, 190), (255,150,   0), (0,  255, 150), (150,  0, 255),
    (255,  0, 150), (0,  150, 255), (150,255,   0), (255, 80,  80),
    (80,  255,  80), (80,  80, 255), (220,220,   0), (0,  220, 220),
]

# Team box colours (BGR)
_TEAM_CLR: dict[int, tuple[int, int, int]] = {
    0: config.TEAM_A_COLOR_BGR,
    1: config.TEAM_B_COLOR_BGR,
}
_UNKNOWN_CLR: tuple[int, int, int] = (160, 160, 160)
_BALL_CLR:    tuple[int, int, int] = config.BALL_COLOR_BGR

_FONT = cv2.FONT_HERSHEY_DUPLEX


def _track_colour(track_id: int) -> tuple[int, int, int]:
    if track_id < 0:
        return _UNKNOWN_CLR
    return _TRACK_PALETTE[track_id % len(_TRACK_PALETTE)]


def _draw_label(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    fg: tuple[int, int, int],
    bg: tuple[int, int, int] = (20, 20, 20),
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    """Draw text on a filled dark background badge."""
    (tw, th), _ = cv2.getTextSize(text, _FONT, scale, thickness)
    pad = 4
    x1 = max(0, x)
    y1 = max(0, y - th - pad * 2)
    x2 = min(img.shape[1] - 1, x + tw + pad * 2)
    y2 = min(img.shape[0] - 1, y)
    cv2.rectangle(img, (x1, y1), (x2, y2), bg, -1)
    cv2.putText(img, text, (x1 + pad, y2 - pad),
                _FONT, scale, fg, thickness, cv2.LINE_AA)


class Annotator:
    """
    Draws detection boxes, track IDs, team labels, track age, and a stats HUD.

    Visual convention
    -----------------
    Box colour   = team  (red=T0, blue=T1, grey=unknown)  — 3px thick
    Label colour = track-unique colour from _TRACK_PALETTE
    Label text   = #ID · Tx · conf · age
    """

    def __init__(self, fps_history: int = 30):
        self._fps_times: deque[float]    = deque(maxlen=fps_history)
        self._track_first_frame: dict[int, int] = {}   # track_id → first frame_idx seen

    # ── Main entry point ──────────────────────────────────────────────────────

    def annotate(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        frame_idx: int,
        wall_time_s: float,
        total_processed: int,
        *,
        fps_hint: float | None = None,
    ) -> np.ndarray:
        out      = frame.copy()
        clip_fps = fps_hint or config.DEFAULT_FPS
        self._fps_times.append(wall_time_s)

        # Record first appearance of each new track ID
        for det in detections:
            if det.track_id >= 0 and det.track_id not in self._track_first_frame:
                self._track_first_frame[det.track_id] = frame_idx

        for det in detections:
            if det.class_name == "ball":
                self._draw_ball(out, det)
            else:
                self._draw_player(out, det, frame_idx, clip_fps)

        self._draw_hud(out, frame_idx, clip_fps, detections)
        return out

    # ── Per-detection drawing ─────────────────────────────────────────────────

    def _draw_player(
        self,
        img: np.ndarray,
        det: Detection,
        frame_idx: int,
        clip_fps: float,
    ) -> None:
        x1, y1, x2, y2 = det.bbox
        box_c = _TEAM_CLR.get(det.team_id, _UNKNOWN_CLR)
        cv2.rectangle(img, (x1, y1), (x2, y2), box_c, 3)

        tid_str  = f"#{det.track_id}" if det.track_id >= 0 else "?"
        team_str = f"T{det.team_id}"  if det.team_id  >= 0 else "ref"
        conf_str = f"{det.confidence:.2f}"

        # Track age: how long this ID has been alive
        if det.track_id >= 0 and det.track_id in self._track_first_frame:
            age_s   = (frame_idx - self._track_first_frame[det.track_id]) / clip_fps
            age_str = f"{age_s:.0f}s"
        else:
            age_str = ""

        label = f"{tid_str} {team_str} {conf_str}"
        if age_str:
            label += f"  {age_str}"

        lbl_c = _track_colour(det.track_id)
        _draw_label(img, label, x1, y1, fg=lbl_c, scale=0.60, thickness=1)

    def _draw_ball(self, img: np.ndarray, det: Detection) -> None:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), _BALL_CLR, 2)
        _draw_label(img, f"ball {det.confidence:.2f}",
                    x1, y1, fg=_BALL_CLR, scale=0.50)

    # ── HUD ───────────────────────────────────────────────────────────────────

    def _draw_hud(
        self,
        img: np.ndarray,
        frame_idx: int,
        clip_fps: float,
        detections: List[Detection],
    ) -> None:
        active   = sum(1 for d in detections if d.track_id >= 0)
        t0_count = sum(1 for d in detections if d.team_id == 0)
        t1_count = sum(1 for d in detections if d.team_id == 1)
        ts       = frame_idx / clip_fps

        proc_fps = (len(self._fps_times) / max(sum(self._fps_times), 1e-6)
                    if self._fps_times else 0.0)

        lines = [
            "GAFFER  v0.1",
            f"Frame {frame_idx:>5d}  t={ts:>5.1f}s",
            f"Tracks  {active:>2d} active",
            f"T0 {t0_count:>2d}   T1 {t1_count:>2d}",
            f"Proc  {proc_fps:>5.1f} fps",
        ]

        scale   = 0.52
        pad     = 8
        line_h  = 20
        panel_w = 210
        panel_h = pad * 2 + line_h * len(lines)
        margin  = 10

        overlay = img.copy()
        cv2.rectangle(overlay, (margin, margin),
                      (margin + panel_w, margin + panel_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.70, img, 0.30, 0, img)
        cv2.rectangle(img, (margin, margin),
                      (margin + panel_w, margin + panel_h), (80, 80, 80), 1)

        for i, line in enumerate(lines):
            colour = (120, 220, 120) if i == 0 else (255, 255, 255)
            cv2.putText(img, line,
                        (margin + pad, margin + pad + line_h * (i + 1) - 4),
                        _FONT, scale, colour, 1, cv2.LINE_AA)

        # Team colour swatches next to the T0/T1 line
        ty  = margin + pad + line_h * 4 - 4 - 12
        tx  = margin + pad + 95
        cv2.rectangle(img, (tx, ty), (tx + 12, ty + 12), config.TEAM_A_COLOR_BGR, -1)
        cv2.rectangle(img, (tx + 45, ty), (tx + 57, ty + 12), config.TEAM_B_COLOR_BGR, -1)
