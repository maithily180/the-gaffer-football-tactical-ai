"""
gaffer/tracking/ball_candidate_filter.py
─────────────────────────────────────────
Post-YOLO filter chain that removes ball false positives and enforces
physical continuity before detections reach BallTracker.

Four gates + one scoring step (applied in order):

1. Stationary-object gate
   Penalty spots / pitch markings appear at the same pixel every detection
   frame.  A real ball always moves.  Reject candidates that keep appearing
   within BALL_STATIONARY_MOVE_PX for BALL_STATIONARY_MIN_FRAMES frames.

2. Spatial continuity gate   ← primary false-positive killer
   Gate is centred on the PREDICTED ball position (last_pos + velocity * dt),
   not just on last_pos.  This is the critical fix for fast passes: if the
   ball was moving rightward at 60 px/frame, the gate moves rightward with it,
   so a valid detection 120 px ahead is accepted rather than rejected.

   Gate radius = BASE + speed * SPEED_MULT + missed_frames * MISS_GROWTH,
   capped at MAX.

   After a scene cut (histogram comparison) the gate resets so the first
   valid detection in the new shot isn't rejected.

3. Recovery mode   ← handles long losses and reacquisition
   After BALL_RECOVERY_FRAMES consecutive missed detection frames the tracker
   has lost the ball.  Instead of blocking everything with a tight gate,
   switch to cluster-based reacquisition: accept any candidate within
   BALL_CLUSTER_RADIUS_PX of the tightest player cluster.  The spatial gate
   re-arms once a candidate is accepted.

4. Candidate scoring + selection
   When multiple candidates survive the gates, rank by a combined score:
       score = confidence × cluster_affinity
   where cluster_affinity = 1 / (1 + dist_to_nearest_player / 100).
   Return only the best-scoring candidate (YOLO often returns duplicates and
   slightly-off detections of the same ball).

5. On-pitch gate  (optional — requires HomographyManager)
   Project candidate via H into pitch metres; reject if outside pitch + margin.
   Kills ad boards and tunnel blobs.
"""

from __future__ import annotations

import math
from collections import deque
from typing import List

import cv2
import numpy as np

from gaffer import config
from gaffer.detection.detector import Detection


class BallCandidateFilter:

    def __init__(
        self,
        # Stationary gate
        stationary_move_px: int    = config.BALL_STATIONARY_MOVE_PX,
        stationary_min_frames: int = config.BALL_STATIONARY_MIN_FRAMES,
        stationary_window: int     = config.BALL_STATIONARY_WINDOW,
        # Spatial gate
        gate_base_px: float        = config.BALL_SPATIAL_GATE_BASE_PX,
        gate_speed_mult: float     = config.BALL_SPATIAL_GATE_SPEED_MULT,
        gate_miss_growth: float    = config.BALL_SPATIAL_GATE_MISS_GROWTH,
        gate_max_px: float         = config.BALL_SPATIAL_GATE_MAX_PX,
        scene_cut_threshold: float = config.BALL_SCENE_CUT_THRESHOLD,
        # Recovery mode
        recovery_frames: int       = config.BALL_RECOVERY_FRAMES,
        cluster_radius_px: int     = config.BALL_CLUSTER_RADIUS_PX,
        # Proximity gate (pre-acquisition)
        proximity_threshold_px: int = config.BALL_PROXIMITY_THRESHOLD_PX,
        in_air_speed_px: float      = config.BALL_IN_AIR_SPEED_PX,
    ):
        # Stationary gate
        self._stat_move = stationary_move_px
        self._stat_min  = stationary_min_frames
        self._candidate_history: deque[tuple[int, int]] = deque(maxlen=stationary_window)

        # Spatial gate
        self._gate_base        = gate_base_px
        self._gate_speed_mult  = gate_speed_mult
        self._gate_miss_growth = gate_miss_growth
        self._gate_max         = gate_max_px
        self._cut_threshold    = scene_cut_threshold
        self._gate_active      = False

        # Recovery mode
        self._recovery_frames  = recovery_frames
        self._cluster_radius   = cluster_radius_px

        # Proximity gate
        self._proximity  = proximity_threshold_px
        self._in_air_spd = in_air_speed_px

        # Rejection / event counters
        self.n_rejected_stationary = 0
        self.n_rejected_spatial    = 0
        self.n_rejected_proximity  = 0
        self.n_rejected_off_pitch  = 0
        self.n_scene_cuts_detected = 0
        self.n_recovery_accepted   = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def filter(
        self,
        ball_dets: List[Detection],
        field_dets: List[Detection],
        frame_idx: int,
        *,
        prev_frame: np.ndarray | None          = None,
        curr_frame: np.ndarray | None          = None,
        homography_manager                     = None,
        last_ball_pos: tuple[int, int] | None  = None,
        last_ball_vel: tuple[float, float] | None = None,   # (vx, vy) px/det-frame
        last_detection_frame: int | None       = None,
    ) -> List[Detection]:
        """
        Return at most one ball Detection: the best candidate that passes all
        gates.  Returns empty list if none survive.

        Parameters
        ----------
        ball_dets            : raw ball detections from YOLO this frame
        field_dets           : player / GK / referee detections (for cluster)
        frame_idx            : current video frame index
        prev_frame / curr_frame : consecutive frames for scene-cut detection
        homography_manager   : optional; enables on-pitch gate
        last_ball_pos        : pixel centre of last ACTUAL detection
        last_ball_vel        : (vx, vy) velocity in px/detection-frame
        last_detection_frame : frame index of last actual detection
        """
        # ── Scene cut → reset spatial gate ───────────────────────────────────
        if prev_frame is not None and curr_frame is not None:
            if self._is_scene_cut(prev_frame, curr_frame):
                self._gate_active = False
                self.n_scene_cuts_detected += 1

        if not ball_dets:
            return []

        candidates = self._deduplicate(ball_dets)

        # Determine how many detection frames we've been without the ball
        missed_dframes = self._missed_dframes(last_detection_frame, frame_idx)
        in_recovery    = self._gate_active and missed_dframes >= self._recovery_frames
        in_air         = self._in_air(last_ball_vel)

        # Predicted ball position for spatial gate
        pred_pos = self._predict(last_ball_pos, last_ball_vel,
                                 last_detection_frame, frame_idx)
        gate_r   = self._gate_radius(last_ball_vel, missed_dframes)

        # Player cluster centre (used for scoring + recovery reacquisition)
        cluster  = self._player_cluster(field_dets)

        passed: List[Detection] = []

        for det in candidates:
            cx, cy = det.center

            # ── Gate 1: stationary ────────────────────────────────────────────
            self._candidate_history.append((cx, cy))
            if self._is_stationary(cx, cy):
                self.n_rejected_stationary += 1
                continue

            # ── Gate 2: spatial continuity ────────────────────────────────────
            if self._gate_active:
                if pred_pos is not None:
                    dist = math.dist((cx, cy), pred_pos)
                    if dist > gate_r:
                        self.n_rejected_spatial += 1
                        continue
            else:
                # Pre-acquisition: require proximity to players
                if not in_air and not self._near_players(cx, cy, field_dets):
                    self.n_rejected_proximity += 1
                    continue

            # ── Gate 3: on-pitch (optional) ───────────────────────────────────
            if homography_manager is not None and homography_manager.is_valid():
                world = homography_manager.project((cx, cy))
                if world is None or not homography_manager.on_pitch(*world, margin_m=3.0):
                    self.n_rejected_off_pitch += 1
                    continue

            passed.append(det)

        if not passed:
            return []

        # ── Select best candidate ──────────────────────────────────────────────
        # Confidence is the primary signal. Cluster proximity is a tiebreaker
        # only among candidates within 15% of the best confidence — never
        # overrides a clearly higher-confidence detection (avoids wrong-candidate
        # poisoning of the tracker position).
        best_conf = max(d.confidence for d in passed)
        top_tier  = [d for d in passed if d.confidence >= best_conf * 0.85]
        if len(top_tier) == 1:
            best_det = top_tier[0]
        else:
            best_det = min(top_tier,
                           key=lambda d: self._dist_to_cluster(d, cluster))

        if in_recovery:
            self.n_recovery_accepted += 1
        self._gate_active = True
        return [best_det]

    def rejection_summary(self) -> dict[str, int]:
        return {
            "stationary":     self.n_rejected_stationary,
            "spatial":        self.n_rejected_spatial,
            "proximity":      self.n_rejected_proximity,
            "off_pitch":      self.n_rejected_off_pitch,
            "scene_cuts":     self.n_scene_cuts_detected,
            "recovery_accept":self.n_recovery_accepted,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _deduplicate(ball_dets: List[Detection]) -> List[Detection]:
        sorted_dets = sorted(ball_dets, key=lambda d: d.confidence, reverse=True)
        kept: List[Detection] = []
        for det in sorted_dets:
            cx, cy = det.center
            if not any(
                abs(cx - k.center[0]) < 20 and abs(cy - k.center[1]) < 20
                for k in kept
            ):
                kept.append(det)
        return kept

    def _is_stationary(self, cx: int, cy: int) -> bool:
        nearby = sum(
            1 for (px, py) in self._candidate_history
            if abs(px - cx) < self._stat_move and abs(py - cy) < self._stat_move
        )
        return nearby >= self._stat_min

    def _predict(
        self,
        last_pos: tuple[int, int] | None,
        last_vel: tuple[float, float] | None,
        last_det_frame: int | None,
        curr_frame: int,
    ) -> tuple[float, float] | None:
        """
        Predict current ball position using a decayed velocity blend.

        Pure constant-velocity overshoots badly in football — the ball stops
        at a player's feet, deflects, or is controlled within 1-2 frames of a
        pass ending.  Instead of projecting the full velocity, we blend between
        last_pos and the full-velocity prediction with a decay factor that
        weakens as dt grows:

            blend = 1 / (1 + dt * 0.25)

        At dt=3 (1 missed frame):   blend=0.57 → small bias toward velocity dir
        At dt=6 (2 missed frames):  blend=0.40 → further weakened
        At dt=9 (3 missed frames):  blend=0.31 → mostly stays at last_pos

        This keeps the gate anchored near the last known position while still
        giving a small directional nudge so the gate can catch fast passes.
        The gate RADIUS (not centre) does the heavy lifting for range.
        """
        if last_pos is None:
            return None
        if last_vel is None or last_det_frame is None:
            return float(last_pos[0]), float(last_pos[1])
        dt = curr_frame - last_det_frame
        if dt <= 0:
            return float(last_pos[0]), float(last_pos[1])
        vx, vy = last_vel
        speed = math.sqrt(vx ** 2 + vy ** 2)
        # Only project the gate center when the ball is clearly in flight
        # (a pass or shot). During possession / slow play the ball often stops
        # within a frame, so projecting the gate overshoots and rejects the
        # real detection at last_pos.  Threshold ≈ 8 px/raw-frame ≈ 12 km/h.
        if speed < 8.0:
            return float(last_pos[0]), float(last_pos[1])
        # Fast ball: blend toward predicted position, decaying with time so we
        # don't overcommit if the ball slows down after 1-2 frames.
        blend = 1.0 / (1.0 + dt * 0.20)
        return (
            last_pos[0] + blend * vx * dt,
            last_pos[1] + blend * vy * dt,
        )

    def _gate_radius(
        self,
        last_vel: tuple[float, float] | None,
        missed_dframes: int,
    ) -> float:
        if not self._gate_active:
            return self._gate_max
        speed = math.sqrt(last_vel[0] ** 2 + last_vel[1] ** 2) if last_vel else 0.0
        r = self._gate_base + speed * self._gate_speed_mult + missed_dframes * self._gate_miss_growth
        return min(r, self._gate_max)

    def _missed_dframes(
        self,
        last_det_frame: int | None,
        curr_frame: int,
    ) -> int:
        if last_det_frame is None:
            return 0
        return max(0, (curr_frame - last_det_frame) // max(config.DETECT_EVERY_N_FRAMES, 1) - 1)

    @staticmethod
    def _player_cluster(field_dets: List[Detection]) -> tuple[float, float] | None:
        """
        Find the centre of the tightest player cluster — the 5 players most
        tightly grouped together.  This is where the ball most likely is.
        Falls back to overall centroid if fewer than 5 players detected.
        """
        if not field_dets:
            return None
        pts = [d.center for d in field_dets if d.class_name in ("player", "goalkeeper")]
        if not pts:
            pts = [d.center for d in field_dets]
        if not pts:
            return None
        if len(pts) <= 5:
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            return cx, cy

        # Find the 5 players whose mutual distances are smallest (tightest group)
        n = len(pts)
        best_score = float("inf")
        best_centre = (float(pts[0][0]), float(pts[0][1]))
        # Sample: for each player, compute spread of nearest 5 including self
        for i, (px, py) in enumerate(pts):
            dists = sorted(
                math.dist((px, py), (qx, qy)) for j, (qx, qy) in enumerate(pts)
            )[:5]
            score = sum(dists)
            if score < best_score:
                best_score = score
                # Centre of those 5 nearest
                near5 = sorted(
                    pts, key=lambda q: math.dist((px, py), q)
                )[:5]
                best_centre = (
                    sum(q[0] for q in near5) / 5,
                    sum(q[1] for q in near5) / 5,
                )
        return best_centre

    @staticmethod
    def _dist_to_cluster(det: Detection, cluster: tuple[float, float] | None) -> float:
        """Distance from detection centre to player cluster. 0 if no cluster."""
        if cluster is None:
            return 0.0
        return math.dist(det.center, cluster)

    def _near_players(self, cx: int, cy: int, field_dets: List[Detection]) -> bool:
        if not field_dets:
            return True
        thr2 = self._proximity ** 2
        for d in field_dets:
            px, py = d.center
            if (px - cx) ** 2 + (py - cy) ** 2 <= thr2:
                return True
        return False

    def _in_air(self, last_vel: tuple[float, float] | None) -> bool:
        if last_vel is None:
            return False
        return math.sqrt(last_vel[0] ** 2 + last_vel[1] ** 2) >= self._in_air_spd

    def _is_scene_cut(self, prev: np.ndarray, curr: np.ndarray) -> bool:
        try:
            prev_s = cv2.resize(prev, (160, 90))
            curr_s = cv2.resize(curr, (160, 90))
            prev_h = cv2.calcHist([prev_s], [0, 1, 2], None, [8, 8, 8],
                                  [0, 256, 0, 256, 0, 256])
            curr_h = cv2.calcHist([curr_s], [0, 1, 2], None, [8, 8, 8],
                                  [0, 256, 0, 256, 0, 256])
            cv2.normalize(prev_h, prev_h)
            cv2.normalize(curr_h, curr_h)
            return cv2.compareHist(prev_h, curr_h, cv2.HISTCMP_BHATTACHARYYA) > self._cut_threshold
        except Exception:
            return False
