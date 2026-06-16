from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from dataclasses import replace
from typing import List

import numpy as np
import supervision as sv

from gaffer import config
from gaffer.detection.detector import Detection

# Classes that get tracked (ball has its own trajectory; referee = 1 person, low priority)
_TRACKABLE: frozenset[str] = frozenset({"player", "goalkeeper", "person"})


class PlayerTracker:
    """
    Wraps supervision.ByteTrack to assign stable integer track_ids to players.

    Why ByteTrack and NOT BoT-SORT+CMC
    ----------------------------------
    Day 5 audit (scripts/tracking_audit.py) measured ID *switches* — how often
    the number on a player flips, the thing you actually see — not just unique-ID
    count. Plain ByteTrack at skip-3 was the clear winner (47 switches at conf
    .35) vs BoT-SORT+CMC (148 at skip-3, 176 every-frame). BoT-SORT's looser
    association swaps IDs between players in crowds and its camera-motion
    compensation mis-warps on textureless grass; every-frame detection adds
    association wobble that skip-3's carry-forward avoids. Unique-ID count was a
    misleading target: most of those IDs are players re-entering frame after a
    camera cut, not mid-track instability.

    Only players/goalkeepers with confidence >= config.TRACK_MIN_CONF enter
    tracking (audit: conf .35 → 47 switches vs .25 → 60). Lower-confidence and
    non-trackable detections pass through with track_id=-1.

    Call update() on detection frames; carry_forward() on skip frames.
    """

    def __init__(
        self,
        fps: float = config.DEFAULT_FPS,
        min_conf: float = config.TRACK_MIN_CONF,
        match_thresh: float = config.MATCH_THRESH,
    ):
        self._min_conf = min_conf
        # lost_track_buffer is counted in update() calls, and update() runs once
        # per detection frame → convert the wall-clock buffer to update-steps.
        buffer_steps = max(1, round(config.TRACK_BUFFER_FRAMES / config.DETECT_EVERY_N_FRAMES))
        self._byte = sv.ByteTrack(
            track_activation_threshold=min_conf,
            lost_track_buffer=buffer_steps,
            minimum_matching_threshold=match_thresh,   # IoU COST ceiling: 0.7 → IoU ≥ 0.3
            frame_rate=fps,
            minimum_consecutive_frames=1,
        )
        self._last_result: List[Detection] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, detections: List[Detection], frame: np.ndarray | None = None) -> List[Detection]:
        """
        Feed detections into ByteTrack. Call on detection frames only.

        `frame` is accepted for call-site compatibility (and a future CMC option)
        but unused — supervision ByteTrack has no camera-motion compensation.

        Returns a new list where tracked players carry their track_id. Players
        below min_conf and non-trackable detections pass through with track_id=-1.
        """
        players: List[Detection] = []
        passthrough: List[Detection] = []
        for d in detections:
            if d.class_name in _TRACKABLE and d.confidence >= self._min_conf:
                players.append(d)
            else:
                passthrough.append(d)

        if not players:
            result = list(passthrough)
            self._last_result = result
            return result

        sv_in = sv.Detections(
            xyxy       = np.array([d.bbox for d in players], dtype=float),
            confidence = np.array([d.confidence for d in players], dtype=float),
            class_id   = np.array([d.class_id   for d in players], dtype=int),
        )

        sv_out = self._byte.update_with_detections(sv_in)

        # supervision returns only confirmed tracks, reordered → map by bbox key.
        tid_map: dict[tuple, int] = {}
        if sv_out.tracker_id is not None:
            for i in range(len(sv_out)):
                tid_map[tuple(sv_out.xyxy[i].astype(int))] = int(sv_out.tracker_id[i])

        tracked = [
            replace(det, track_id=tid_map.get(tuple(int(v) for v in det.bbox), -1))
            for det in players
        ]
        result = tracked + list(passthrough)
        self._last_result = result
        return result

    def carry_forward(self) -> List[Detection]:
        """Return the last update() result unchanged (for skip frames)."""
        return self._last_result

    def reset(self) -> None:
        self._byte.reset()
        self._last_result = []
