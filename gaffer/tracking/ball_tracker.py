"""
gaffer/tracking/ball_tracker.py
────────────────────────────────
Single-instance tracker for the football.

Why the ball needs its own tracker
───────────────────────────────────
ByteTrack is designed for objects that move < ~30px between detection frames.
The ball at skip-3 (3 × 40ms = 120ms) moves 30-90px on a normal pass and
100-200px on a shot — ByteTrack's IoU association simply fails. Additionally
the ball has a mAP50 of ~0.37 vs ~0.88 for players, so it's frequently
missed by the detector, leaving gaps ByteTrack can't bridge.

This tracker handles both problems:
  1. EMA smoothing   — jitter reduction on low-confidence detections
  2. Velocity-based extrapolation — predicts ball position for frames where
     the detector returns nothing, up to BALL_MAX_INTERP_FRAMES frames.

The ball always gets track_id = 0. Extrapolated positions are marked with
confidence = 0.0 so the annotator can render them differently if desired.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import List

from gaffer import config
from gaffer.detection.detector import Detection


class BallTracker:
    """
    Tracks a single ball across frames.

    Usage
    -----
    Call update() on every detection frame (even if ball_dets is empty).
    The returned Detection (or None) is the best estimate of ball position
    for that frame — either a smoothed live detection or an extrapolation.
    """

    TRACK_ID = 0  # ball always occupies track slot 0

    def __init__(
        self,
        max_interp_frames: int = config.BALL_MAX_INTERP_FRAMES,
        smooth_alpha: float = config.BALL_SMOOTH_ALPHA,
    ):
        self._max_interp = max_interp_frames
        self._alpha = smooth_alpha

        # History: list of (frame_idx, Detection) for frames where ball was
        # actually detected (not extrapolated). Kept trim to max_interp_frames.
        self._history: list[tuple[int, Detection]] = []

        # Smoothed centre tracked independently of bbox (sub-pixel precision).
        self._smooth_cx: float | None = None
        self._smooth_cy: float | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, ball_dets: List[Detection], frame_idx: int) -> Detection | None:
        """
        Feed all ball detections for this frame.

        Returns the best position estimate (smoothed detection or extrapolation),
        or None if the ball has been lost beyond the interpolation window.
        """
        best = self._best(ball_dets)

        if best is not None:
            cx, cy = best.center
            self._smooth_cx, self._smooth_cy = self._ema(cx, cy)
            self._history.append((frame_idx, best))
            self._trim_history(frame_idx)
            return self._make_result(best, frame_idx, interpolated=False)

        return self._extrapolate(frame_idx)

    def last_velocity_px(self) -> float | None:
        """Pixel speed (magnitude) in px/detection-frame from the last two detections."""
        v = self.last_velocity_vector()
        if v is None:
            return None
        return math.sqrt(v[0] ** 2 + v[1] ** 2)

    def last_velocity_vector(self) -> tuple[float, float] | None:
        """
        Velocity vector (vx, vy) in px/detection-frame from the last two actual
        detections.  Used by BallCandidateFilter to predict the ball's next
        position and centre the spatial gate correctly — not just on the last
        known position, but on where it should be now.
        """
        if len(self._history) < 2:
            return None
        f1, d1 = self._history[-2]
        f2, d2 = self._history[-1]
        dt = f2 - f1
        if dt <= 0:
            return None
        x1, y1 = d1.center
        x2, y2 = d2.center
        return (x2 - x1) / dt, (y2 - y1) / dt

    def last_position_px(self) -> tuple[int, int] | None:
        """Pixel centre of the last ACTUALLY DETECTED (not extrapolated) ball.
        Used by BallCandidateFilter to build the spatial gate."""
        if not self._history:
            return None
        _, det = self._history[-1]
        return det.center

    def last_detection_frame(self) -> int | None:
        """Frame index of the last actual detection (not an extrapolation)."""
        return self._history[-1][0] if self._history else None

    def reset(self) -> None:
        self._history.clear()
        self._smooth_cx = None
        self._smooth_cy = None

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _best(dets: List[Detection]) -> Detection | None:
        """Highest-confidence ball detection, or None."""
        balls = [d for d in dets if d.class_name == "ball"]
        return max(balls, key=lambda d: d.confidence) if balls else None

    def _ema(self, cx: float, cy: float) -> tuple[float, float]:
        """Exponential moving average. First call seeds the filter."""
        if self._smooth_cx is None:
            return float(cx), float(cy)
        a = self._alpha
        return (
            a * cx + (1 - a) * self._smooth_cx,
            a * cy + (1 - a) * self._smooth_cy,
        )

    def _extrapolate(self, frame_idx: int) -> Detection | None:
        """
        Predict ball position using constant-velocity extrapolation from the
        last two detected positions. Returns None if:
          - fewer than 2 detections in history, or
          - the last detection is older than max_interp_frames.
        """
        if len(self._history) < 1:
            return None
        last_f, last_d = self._history[-1]
        gap = frame_idx - last_f
        if gap > self._max_interp:
            return None  # ball lost too long to extrapolate reliably

        if len(self._history) < 2:
            # No velocity — return last known position as-is
            return self._make_result(last_d, frame_idx, interpolated=True)

        prev_f, prev_d = self._history[-2]
        dt = last_f - prev_f
        if dt <= 0:
            return self._make_result(last_d, frame_idx, interpolated=True)

        lx, ly = last_d.center
        px, py = prev_d.center
        vx = (lx - px) / dt
        vy = (ly - py) / dt

        pred_cx = lx + vx * gap
        pred_cy = ly + vy * gap

        # Clamp to frame bounds (crude but avoids absurd out-of-bounds)
        pred_cx = max(0.0, pred_cx)
        pred_cy = max(0.0, pred_cy)

        w, h = last_d.width, last_d.height
        x1 = int(pred_cx - w / 2)
        y1 = int(pred_cy - h / 2)
        x2 = int(pred_cx + w / 2)
        y2 = int(pred_cy + h / 2)

        return self._make_result(
            replace(last_d, bbox=(x1, y1, x2, y2)),
            frame_idx,
            interpolated=True,
        )

    def _make_result(
        self, det: Detection, frame_idx: int, interpolated: bool
    ) -> Detection:
        """
        Return a Detection with:
          - track_id = TRACK_ID (0) so PositionStore records it
          - confidence = 0.0 when interpolated (annotator can render differently)
          - centre shifted to smoothed position when live
        """
        if not interpolated and self._smooth_cx is not None:
            # Rebuild bbox centred on the EMA-smoothed position
            scx = int(round(self._smooth_cx))
            scy = int(round(self._smooth_cy))
            hw, hh = det.width // 2, det.height // 2
            det = replace(det, bbox=(scx - hw, scy - hh, scx + hw, scy + hh))

        conf = 0.0 if interpolated else det.confidence
        return replace(det, track_id=self.TRACK_ID, confidence=conf)

    def _trim_history(self, frame_idx: int) -> None:
        cutoff = frame_idx - self._max_interp - 5
        self._history = [(f, d) for f, d in self._history if f >= cutoff]
