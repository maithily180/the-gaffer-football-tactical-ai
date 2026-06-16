from __future__ import annotations

import math
from collections import defaultdict
from typing import List

from gaffer import config
from gaffer.detection.detector import Detection


class PositionEntry:
    """One recorded position for a track."""
    __slots__ = ("frame_idx", "center", "timestamp")

    def __init__(self, frame_idx: int, center: tuple[int, int], timestamp: float):
        self.frame_idx = frame_idx
        self.center    = center        # (cx, cy) pixels
        self.timestamp = timestamp     # seconds


class PositionStore:
    """
    Maintains per-track position history built from Detection lists.

    Usage:
        store = PositionStore(fps=25.0)
        store.update(frame_idx, detections)   # call every rendered frame
        hist = store.get_track_history(track_id)
        v    = store.get_velocity(track_id)   # px/s
    """

    def __init__(self, fps: float = config.DEFAULT_FPS):
        self._fps = fps
        self._history: dict[int, list[PositionEntry]] = defaultdict(list)

    # ── Write ─────────────────────────────────────────────────────────────────

    def update(self, frame_idx: int, detections: List[Detection]) -> None:
        """Append a PositionEntry for every detection that has a confirmed track_id."""
        ts = frame_idx / self._fps
        for det in detections:
            if det.track_id < 0:
                continue
            self._history[det.track_id].append(
                PositionEntry(frame_idx, det.center, ts)
            )

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_track_history(self, track_id: int) -> list[PositionEntry]:
        """
        Return the full position history for a track, oldest first.
        Returns an empty list if the track is unknown.
        """
        return list(self._history.get(track_id, []))

    def get_velocity(
        self,
        track_id: int,
        window: int = config.VELOCITY_WINDOW_FRAMES,
    ) -> float | None:
        """
        Estimate instantaneous speed (px/s) over the last `window` entries.

        Returns None if fewer than 2 samples exist for the track.
        The window is defined in number of history entries (detection frames),
        not raw video frames.
        """
        hist = self._history.get(track_id)
        if not hist or len(hist) < 2:
            return None

        recent = hist[-window:] if len(hist) >= window else hist
        if len(recent) < 2:
            return None

        dx = recent[-1].center[0] - recent[0].center[0]
        dy = recent[-1].center[1] - recent[0].center[1]
        dt = recent[-1].timestamp - recent[0].timestamp
        if dt < 1e-6:
            return None
        return math.sqrt(dx * dx + dy * dy) / dt

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def track_ids(self) -> list[int]:
        """All track IDs seen so far, in order of first appearance."""
        return list(self._history.keys())

    def track_length(self, track_id: int) -> int:
        """Number of recorded frames for a track."""
        return len(self._history.get(track_id, []))

    def first_frame(self, track_id: int) -> int | None:
        h = self._history.get(track_id)
        return h[0].frame_idx if h else None

    def last_frame(self, track_id: int) -> int | None:
        h = self._history.get(track_id)
        return h[-1].frame_idx if h else None

    def persistence(self, track_id: int) -> float | None:
        """
        Fraction of frames between first and last appearance where the track
        was actually recorded. 1.0 = never dropped; lower = fragmented.
        """
        h = self._history.get(track_id)
        if not h or len(h) < 2:
            return None
        span = h[-1].frame_idx - h[0].frame_idx + 1
        return len(h) / span
