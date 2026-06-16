"""
gaffer/tracking/ball_state_estimator.py
────────────────────────────────────────
Football context model: player velocities, approach voting, ball state machine.

Players are far easier to detect than the ball and carry rich spatial
information. This class fuses player motion into a "ball attention field"
that makes candidate selection football-aware:

    "Which blob has players actively converging toward it?"

rather than the naive:

    "Which blob did YOLO score highest?"

Three outputs consumed by BallCandidateFilter
──────────────────────────────────────────────
state
    IN_POSSESSION — ball is near a player's feet
    PASS          — ball is moving fast between players
    AIRBORNE      — ball is moving fast with no nearby players (aerial)
    LOOSE_BALL    — ball is on the ground near contested players
    UNKNOWN       — no ball history yet / just after a cut

approach_voters(pos, field_dets) → int
    Count of nearby players whose velocity vector has a positive dot
    product toward pos.  Used as a tiebreaker in candidate selection
    and as the primary signal when re-acquiring after a suspect discard.

is_suspect() → bool
    True when the current ball hypothesis has had zero approach voters
    for SUSPECT_NO_APPROACH_FRAMES consecutive detection frames.
    Signals BallCandidateFilter to discard and re-acquire.
    Suppressed during AIRBORNE (players are supposed to be far away).
"""

from __future__ import annotations

import math
from collections import deque
from enum import Enum
from typing import List

from gaffer import config
from gaffer.detection.detector import Detection


class BallState(Enum):
    UNKNOWN       = "UNKNOWN"
    IN_POSSESSION = "IN_POSSESSION"
    PASS          = "PASS"
    AIRBORNE      = "AIRBORNE"
    LOOSE_BALL    = "LOOSE_BALL"


class BallStateEstimator:
    """
    Call update() once per detection frame with the current ball Detection
    (or None) and the field player detections.

    Important: field_dets must have stable track_ids (from ByteTrack).
    Without consistent IDs the velocity history cannot be built and
    approach_voters() will return 0 for all candidates.
    """

    def __init__(
        self,
        possession_dist_px:   int   = config.BALL_POSSESSION_DIST_PX,
        pass_speed_px_frame:  float = config.BALL_PASS_SPEED_PX_FRAME,
        approach_dist_px:     int   = config.BALL_APPROACH_DIST_PX,
        approach_dot_thresh:  float = config.BALL_APPROACH_DOT_THRESH,
        suspect_no_approach:  int   = config.BALL_SUSPECT_NO_APPROACH_FRAMES,
        suspect_min_track:    int   = config.BALL_SUSPECT_MIN_TRACK_FRAMES,
    ):
        self._possess_dist  = possession_dist_px
        self._pass_speed    = pass_speed_px_frame
        self._approach_dist = approach_dist_px
        self._approach_dot  = approach_dot_thresh
        self._suspect_n     = suspect_no_approach
        self._suspect_min   = suspect_min_track

        self.state: BallState        = BallState.UNKNOWN
        self.possessor_id: int | None = None

        # Per-track position history: track_id → deque[(frame_idx, cx, cy)]
        # maxlen=8 gives ~1 second of history at skip-3 / 25fps.
        self._player_hist: dict[int, deque[tuple[int, int, int]]] = {}

        # Suspect tracking
        self._no_approach_streak: int = 0
        self._frames_tracked:     int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        ball_det:   Detection | None,
        ball_vel:   tuple[float, float] | None,
        field_dets: List[Detection],
        frame_idx:  int,
    ) -> BallState:
        """
        Update player velocity histories and transition the ball state.
        Must be called once per detection frame (even when ball is None).
        Returns the new BallState.
        """
        self._update_player_history(field_dets, frame_idx)

        if ball_det is None:
            # Ball lost — reset counters; keep state so annotator can display it
            self._frames_tracked = 0
            self._no_approach_streak = 0
            return self.state

        self._frames_tracked += 1
        ball_pos = ball_det.center
        speed    = (
            math.sqrt(ball_vel[0] ** 2 + ball_vel[1] ** 2)
            if ball_vel else 0.0
        )

        # Suspect counter: reset when players are approaching, increment otherwise
        voters = self.approach_voters(ball_pos, field_dets)
        if voters > 0 or self._frames_tracked < self._suspect_min:
            self._no_approach_streak = 0
        else:
            self._no_approach_streak += 1

        # ── State transitions ─────────────────────────────────────────────────
        if speed >= self._pass_speed:
            # Fast ball — airborne if no one nearby, otherwise a pass
            near = self._players_within(ball_pos, field_dets, 150)
            new_state = BallState.AIRBORNE if near == 0 else BallState.PASS
        else:
            nearest = self._nearest_player(ball_pos, field_dets)
            if nearest and math.dist(ball_pos, nearest.center) < self._possess_dist:
                contested = self._players_within(
                    ball_pos, field_dets, self._possess_dist * 2
                )
                if contested >= 3:
                    new_state = BallState.LOOSE_BALL
                else:
                    new_state = BallState.IN_POSSESSION
                    self.possessor_id = nearest.track_id
            else:
                new_state = BallState.LOOSE_BALL

        self.state = new_state
        return self.state

    def approach_voters(
        self,
        candidate_pos: tuple[int, int],
        field_dets:    List[Detection],
    ) -> int:
        """
        Return how many players are actively moving toward candidate_pos.

        A player "votes" when:
          1. They are within APPROACH_DIST_PX of the candidate
          2. Their velocity vector has a dot product > APPROACH_DOT_THRESH
             with the unit vector from the player toward the candidate

        Requires stable ByteTrack IDs; players with no velocity history
        (< 2 observations) are skipped.
        """
        cx, cy = candidate_pos
        votes = 0
        for det in field_dets:
            px, py = det.center
            dist = math.dist((px, py), (cx, cy))
            if dist < 5 or dist > self._approach_dist:
                continue
            vel = self._player_vel(det.track_id)
            if vel is None:
                continue
            vx, vy = vel
            speed = math.sqrt(vx ** 2 + vy ** 2)
            if speed < 0.5:          # stationary player — no vote
                continue
            # Unit vector from player toward candidate
            dx, dy = (cx - px) / dist, (cy - py) / dist
            dot = (vx / speed) * dx + (vy / speed) * dy
            if dot > self._approach_dot:
                votes += 1
        return votes

    def is_suspect(self) -> bool:
        """
        True when the tracked ball is likely a false positive.

        Condition:
          - Ball has been tracked for at least SUSPECT_MIN_TRACK_FRAMES
          - No player has been moving toward it for SUSPECT_NO_APPROACH_FRAMES
            consecutive REAL detection frames

        Suppressed when:
          - AIRBORNE: players are supposed to be far from the ball (aerial)
          - IN_POSSESSION: the ball carrier has the ball — other players run
            AWAY to create options, so approach voters = 0 is normal and not
            evidence of a false positive
        """
        if self.state in (BallState.AIRBORNE, BallState.IN_POSSESSION):
            return False
        return (
            self._frames_tracked >= self._suspect_min
            and self._no_approach_streak >= self._suspect_n
        )

    def reset(self) -> None:
        """
        Reset after a scene cut or after discarding a suspect hypothesis.
        Clears the state machine but keeps player velocity history (still valid).
        """
        self._no_approach_streak = 0
        self._frames_tracked = 0
        self.state = BallState.UNKNOWN
        self.possessor_id = None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _update_player_history(
        self, field_dets: List[Detection], frame_idx: int
    ) -> None:
        """Append current player positions; prune tracks that disappeared."""
        seen: set[int] = set()
        for det in field_dets:
            tid = det.track_id
            if tid < 0:
                continue                 # untracked detection — no stable ID
            seen.add(tid)
            if tid not in self._player_hist:
                self._player_hist[tid] = deque(maxlen=8)
            cx, cy = det.center
            self._player_hist[tid].append((frame_idx, cx, cy))
        # Remove tracks that vanished (avoids stale velocity ghosts)
        for tid in list(self._player_hist):
            if tid not in seen:
                del self._player_hist[tid]

    def _player_vel(self, track_id: int) -> tuple[float, float] | None:
        """Instantaneous velocity (px/raw-frame) from the last two positions."""
        hist = self._player_hist.get(track_id)
        if hist is None or len(hist) < 2:
            return None
        f1, x1, y1 = hist[-2]
        f2, x2, y2 = hist[-1]
        dt = f2 - f1
        if dt <= 0:
            return None
        return (x2 - x1) / dt, (y2 - y1) / dt

    def _nearest_player(
        self, ball_pos: tuple[int, int], field_dets: List[Detection]
    ) -> Detection | None:
        if not field_dets:
            return None
        return min(field_dets, key=lambda d: math.dist(ball_pos, d.center))

    def _players_within(
        self,
        pos:        tuple[int, int],
        field_dets: List[Detection],
        radius:     float,
    ) -> int:
        return sum(1 for d in field_dets if math.dist(pos, d.center) < radius)
