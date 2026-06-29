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

v?.? — calibration JSONs can now hold multiple anchor frames (one per distinct
camera shot in a multi-shot clip; see scripts/collect_calibration.py --append).
self.anchors holds all of them sorted by frame_idx; self.H/self.calibration_frame
default to anchors[0] so every consumer that only reads .H (engine.py,
minimap.py, pitch_visibility.py, ball_candidate_filter.py, world_model*.py,
pipeline_runner.py) keeps working exactly as before, unaware multi-anchor exists.
Only a render loop that wants to snap between anchors (gaffer/analyst/
commentary_video.py) needs to know about .anchors / nearest_anchor().
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gaffer import config
from gaffer.calibration.homography import HomographyEstimator


class HomographyManager:
    def __init__(self, H: np.ndarray | None = None, frame_idx: int | None = None,
                anchors: list[tuple[int, np.ndarray]] | None = None):
        self._est = HomographyEstimator()
        self.H = np.asarray(H, dtype=np.float64) if H is not None else None
        self.calibration_frame = frame_idx
        # [(frame_idx, H), ...] sorted by frame_idx -- always at least the
        # primary (H, frame_idx) pair above when one is loaded, so single-
        # anchor callers that never touch .anchors see no behavior change.
        self.anchors: list[tuple[int, np.ndarray]] = anchors or (
            [(frame_idx, self.H)] if self.H is not None and frame_idx is not None else []
        )

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_calibration(cls, json_path: str | Path) -> "HomographyManager":
        """
        Load from a calibration JSON. Supports two shapes:
          - legacy: top-level frame_idx/image_points/homography (single anchor)
          - multi-anchor: {"anchors": [{frame_idx, image_points, homography}, ...]}
        Uses each anchor's stored homography if present; otherwise recomputes
        it from image_points + config.PITCH_KEYPOINTS.
        """
        data = json.loads(Path(json_path).read_text())
        raw_anchors = data["anchors"] if "anchors" in data else [data]

        est = HomographyEstimator()
        anchors: list[tuple[int, np.ndarray]] = []
        for a in raw_anchors:
            if a.get("homography") is not None:
                H = np.array(a["homography"], dtype=np.float64)
            else:
                names = list(a["image_points"].keys())
                image_pts = np.array([a["image_points"][n] for n in names], dtype=np.float32)
                world_pts = np.array([config.PITCH_KEYPOINTS[n] for n in names], dtype=np.float32)
                H, valid = est.compute(image_pts, world_pts)
                if not valid or H is None:
                    raise ValueError(f"Could not compute a valid homography from {json_path}")
            anchors.append((a["frame_idx"], H))

        anchors.sort(key=lambda pair: pair[0])

        # The single-H consumers (engine.py, pipeline_runner.py, ...) must keep
        # using the SAME anchor they always have, even after --append adds more
        # anchors later -- "primary" is "whichever calibration was already
        # trusted," not "whichever happens to be earliest in the video," which
        # silently changed pipeline_runner.py's whole-match analytics (5
        # episodes -> 3, confirmed by a forced rebuild) the first time this
        # mattered. primary_frame_idx is set once by collect_calibration.py
        # when a file is first created and never moved by later appends;
        # legacy single-anchor files (and any file predating this field) just
        # fall back to anchors[0], identical to before since there's only one.
        primary_idx = data.get("primary_frame_idx", anchors[0][0])
        primary_frame, primary_H = next((a for a in anchors if a[0] == primary_idx), anchors[0])
        return cls(primary_H, primary_frame, anchors=anchors)

    # ── Use ───────────────────────────────────────────────────────────────────

    def is_valid(self) -> bool:
        return self.H is not None

    def nearest_anchor(self, frame_idx: int) -> tuple[int, np.ndarray]:
        """The anchor whose frame_idx is closest to `frame_idx` (ties favor
        the earlier one). Pure lookup -- does not touch self.H."""
        return min(self.anchors, key=lambda pair: (abs(pair[0] - frame_idx), pair[0]))

    def project(self, pixel_pt: tuple[float, float]) -> tuple[float, float] | None:
        """Pixel → pitch metres. None if no H."""
        return self._est.project(pixel_pt, self.H)

    def on_pitch(self, x_m: float, y_m: float, margin_m: float = 5.0) -> bool:
        """True if a projected point falls within the pitch (+ margin)."""
        return (-margin_m <= x_m <= config.PITCH_LENGTH_M + margin_m and
                -margin_m <= y_m <= config.PITCH_WIDTH_M + margin_m)
