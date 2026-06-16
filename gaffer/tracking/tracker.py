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

    Supervision's update_with_detections() returns only *confirmed* tracks
    (those that survived minimum_consecutive_frames detection cycles). This
    class maps confirmed bboxes back onto the original Detection list so that:
      - confirmed detections carry their track_id
      - unconfirmed detections keep track_id=-1
      - on skip frames (detect_every_n cache frames), the last confirmed
        result is carried forward unchanged

    Call update() once per detection frame (not every frame).
    On cache frames, call carry_forward() to propagate the last result.
    """

    def __init__(
        self,
        fps: float = config.DEFAULT_FPS,
        track_thresh: float = config.TRACK_THRESH,
        track_buffer: int = config.TRACK_BUFFER_FRAMES,
        match_thresh: float = config.MATCH_THRESH,
        min_consecutive: int = 1,
    ):
        self._byte = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh,
            frame_rate=fps,
            minimum_consecutive_frames=min_consecutive,
        )
        self._last_result: List[Detection] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, detections: List[Detection]) -> List[Detection]:
        """
        Feed detections into ByteTrack. Call on detection frames only.

        Returns a new list of Detections where confirmed players have their
        track_id set. Unconfirmed players and non-player detections pass
        through with track_id=-1.
        """
        players  = [d for d in detections if d.class_name in _TRACKABLE]
        others   = [d for d in detections if d.class_name not in _TRACKABLE]

        if not players:
            result = list(others)
            self._last_result = result
            return result

        sv_in = sv.Detections(
            xyxy       = np.array([d.bbox for d in players], dtype=float),
            confidence = np.array([d.confidence for d in players], dtype=float),
            class_id   = np.array([d.class_id   for d in players], dtype=int),
        )

        sv_out = self._byte.update_with_detections(sv_in)

        # Build xyxy → tracker_id lookup for confirmed tracks
        tid_map: dict[tuple, int] = {}
        if sv_out.tracker_id is not None and len(sv_out) > 0:
            for i in range(len(sv_out)):
                key = tuple(sv_out.xyxy[i].astype(int))
                tid_map[key] = int(sv_out.tracker_id[i])

        tracked: List[Detection] = []
        for det in players:
            key = tuple(int(v) for v in det.bbox)
            tid = tid_map.get(key, -1)
            tracked.append(replace(det, track_id=tid))

        result = tracked + list(others)
        self._last_result = result
        return result

    def carry_forward(self) -> List[Detection]:
        """
        Return the last update() result unchanged.
        Call on skip frames so callers always get a Detection list.
        """
        return self._last_result

    def reset(self) -> None:
        self._byte.reset()
        self._last_result = []
