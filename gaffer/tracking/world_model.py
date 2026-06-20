"""
gaffer/tracking/world_model.py
────────────────────────────────
Football World Model — closes the analytics→tracking feedback loop.

Most tracking systems do:
    Video → Detection → Tracking → Analytics

Gaffer adds the inverse path:
    Analytics → WorldModel → BallCandidateFilter  (better ball tracking)

The WorldModel maintains a per-frame tactical prior about where the ball
SHOULD be given everything we know about the match state:

    1. Possession anchor   — nearest player of possession team to last ball;
                            treated as the ball carrier
    2. Trajectory prior    — linear extrapolation of the pitch-space ball
                            history (where is the ball heading?)
    3. Space control       — is this candidate in the possession team's
                            Voronoi space? (adapted from Voronoi pitch-control
                            literature and SoccerNet tactical models)
    4. Event context       — LINE_BREAK says don't look behind the backline;
                            COUNTER says expect fast forward movement;
                            HIGH_PRESS says ball is in a dense cluster

The signals are combined into a score in [0, 1] that modulates YOLO
confidence during candidate selection.  A ±40% swing means a tactically
implausible high-confidence detection can lose to a plausible lower-conf one.

The world model also overrides the recovery-mode anchor (normally the
tightest player cluster) with the possession player's known position —
much more targeted when we know who has the ball.

Inspired by:
    PathCRF (Xia 2026) — ball state inferred from player trajectories alone
    Voronoi pitch-control models (Spearman 2018, Fernandez 2019)
    Kalman-with-velocity-prior for sports ball tracking (IEEE 2009)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from gaffer import config
from gaffer.calibration.homography_manager import HomographyManager
from gaffer.events.base import (
    COUNTER_ATTACK,
    HIGH_PRESS,
    HIGH_PRESS_ENDED,
    LINE_BREAK,
    POSSESSION_CHANGE,
    POSSESSION_RECOVERY,
)

if TYPE_CHECKING:
    from gaffer.analytics.engine import AnalyticsSnapshot

# ── Tuning ────────────────────────────────────────────────────────────────────
_ANCHOR_MAX_DIST_M  = 6.0    # player must be within this of ball to be "carrier"
_ANCHOR_DECAY_PX    = 18.0   # pitch metres — score falls to 0.0 at this dist
_TRAJ_DECAY_M       = 12.0   # score falls to 0.0 this far from trajectory extrapolation
_WM_WEIGHT          = 0.8    # how strongly world model modulates effective confidence
                              # ±WM_WEIGHT/2 swing: 0.8 → ±40% of base confidence
_SPACE_SCORE_OWN    = 0.78   # score when candidate is in possession team's space
_SPACE_SCORE_OPP    = 0.22   # score when clearly in opponent space
_PRESS_DECAY_FRAMES = 30     # frames HIGH_PRESS modifier lingers after onset
_COUNTER_DECAY_FRAMES = 75   # frames COUNTER_ATTACK modifier lingers (~3s)
_LINEBK_DECAY_FRAMES  = 50   # frames LINE_BREAK modifier lingers (~2s)


@dataclass
class WorldContext:
    """Tactical priors for one frame — consumed by BallCandidateFilter."""
    possession_anchor_m:  tuple[float, float] | None = None  # ball carrier in pitch m
    possession_anchor_px: tuple[float, float] | None = None  # same in image px
    expected_ball_m:      tuple[float, float] | None = None  # trajectory extrapolation
    possession_team:      str | None = None
    in_high_press:        bool = False
    in_counter:           bool = False
    line_break_x:         float | None = None  # absolute pitch x — ball should be past here
    line_break_sign:      int = 0              # +1 = expect cx > x, -1 = cx < x


class BallWorldModel:
    """
    Stateful world model.  Call update() every detection frame AFTER
    PitchAnalyticsEngine.update().  The updated context is then available
    via score_candidate_px() and recovery_anchor_px() for the NEXT frame's
    BallCandidateFilter call.
    """

    def __init__(self, fps: float = config.DEFAULT_FPS):
        self._fps     = fps
        self._ctx     = WorldContext()
        # Pitch-space ball history (detection frames only, real detections)
        self._ball_hist_m: deque[tuple[float, float]] = deque(maxlen=12)
        # Event modifier decay counters  (key → frames_remaining)
        self._decay:  dict[str, int] = {}
        # Line-break details for spatial scoring
        self._lb_x:   float | None = None
        self._lb_sign: int = 0
        # Last snapshot kept for space-control lookups
        self._snap: AnalyticsSnapshot | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, snap: "AnalyticsSnapshot", mgr: HomographyManager) -> None:
        """
        Consume the latest AnalyticsSnapshot.  Call once per detection frame,
        AFTER PitchAnalyticsEngine.update(), BEFORE the next frame's filter call.
        """
        self._snap = snap

        # Tick down event decays
        for k in list(self._decay):
            self._decay[k] -= 1
            if self._decay[k] <= 0:
                del self._decay[k]

        # Absorb new events
        for ev in snap.events:
            if ev.event_type == HIGH_PRESS:
                self._decay['press'] = _PRESS_DECAY_FRAMES
            elif ev.event_type == HIGH_PRESS_ENDED:
                self._decay.pop('press', None)
            elif ev.event_type == COUNTER_ATTACK:
                self._decay['counter'] = _COUNTER_DECAY_FRAMES
            elif ev.event_type in (POSSESSION_CHANGE, POSSESSION_RECOVERY):
                self._decay.pop('counter', None)   # possession change ends the counter
            elif ev.event_type == LINE_BREAK:
                self._lb_x    = ev.data.get('def_line_x')
                # Attacking team crossed from low-x side → ball should be at high x
                att = ev.team
                snap_a_dir = snap.team_a.attack_dir if att == "teamA" else snap.team_b.attack_dir
                self._lb_sign = +1 if snap_a_dir == +1 else -1
                self._decay['linebk'] = _LINEBK_DECAY_FRAMES

        # Update pitch-space ball history (only real positions)
        if snap.ball_xy is not None:
            self._ball_hist_m.append(snap.ball_xy)

        # Compute possession anchor (ball carrier)
        anchor_m, anchor_px = self._find_carrier(snap, mgr)

        self._ctx = WorldContext(
            possession_anchor_m  = anchor_m,
            possession_anchor_px = anchor_px,
            expected_ball_m      = self._extrapolate_m(),
            possession_team      = snap.possession.owner,
            in_high_press        = 'press'  in self._decay,
            in_counter           = 'counter' in self._decay,
            line_break_x         = self._lb_x  if 'linebk' in self._decay else None,
            line_break_sign      = self._lb_sign if 'linebk' in self._decay else 0,
        )

    def score_candidate_px(
        self,
        candidate_px: tuple[float, float],
        mgr: HomographyManager,
    ) -> float:
        """
        Tactical plausibility of a candidate ball pixel.  Returns 0.0–1.0.
        Combine with YOLO confidence:
            effective_conf = conf * (1 + WM_WEIGHT * (score - 0.5))
        """
        if self._snap is None or not mgr.is_valid():
            return 0.5

        candidate_m = mgr.project(candidate_px)
        if candidate_m is None:
            return 0.15   # off-pitch — not zero in case H is slightly stale

        signals = self._signals(candidate_m)
        return sum(signals) / len(signals) if signals else 0.5

    def _signals(self, candidate_m: tuple[float, float]) -> list[float]:
        """
        Core v1.0 signal set.  Subclasses (e.g. WorldModelV2) extend this by
        calling super()._signals() and appending additional signals — each
        appended signal gets equal weight in the final average, so only
        append when the signal is actually informative for this candidate
        (return early / skip otherwise, mirroring signals 1-2 below).
        """
        signals: list[float] = []

        # ── Signal 1: possession anchor proximity (strongest)
        if self._ctx.possession_anchor_m is not None:
            d = math.hypot(
                candidate_m[0] - self._ctx.possession_anchor_m[0],
                candidate_m[1] - self._ctx.possession_anchor_m[1],
            )
            signals.append(max(0.0, 1.0 - d / _ANCHOR_DECAY_PX))

        # ── Signal 2: trajectory extrapolation
        if self._ctx.expected_ball_m is not None:
            d = math.hypot(
                candidate_m[0] - self._ctx.expected_ball_m[0],
                candidate_m[1] - self._ctx.expected_ball_m[1],
            )
            signals.append(max(0.0, 1.0 - d / _TRAJ_DECAY_M))

        # ── Signal 3: Voronoi space control (nearest-player proxy)
        signals.append(self._space_score(candidate_m))

        # ── Signal 4: event-context modifiers
        signals.append(self._event_score(candidate_m))

        return signals

    def effective_confidence(
        self,
        base_conf: float,
        candidate_px: tuple[float, float],
        mgr: HomographyManager,
    ) -> float:
        """
        YOLO confidence boosted/penalised by world-model score.
        Use this as the sort key during candidate selection.
        """
        wm = self.score_candidate_px(candidate_px, mgr)
        return base_conf * (1.0 + _WM_WEIGHT * (wm - 0.5))

    def recovery_anchor_px(
        self,
        cluster_px: tuple[float, float] | None,
        mgr: HomographyManager,
    ) -> tuple[float, float] | None:
        """
        Best pixel anchor to search around when ball is lost (recovery mode).
        Priority: possession carrier > trajectory extrapolation > cluster.
        """
        if self._ctx.possession_anchor_px is not None:
            return self._ctx.possession_anchor_px
        if self._ctx.expected_ball_m is not None:
            px = _pitch_to_image(self._ctx.expected_ball_m, mgr)
            if px is not None:
                return px
        return cluster_px

    @property
    def context(self) -> WorldContext:
        return self._ctx

    # ── Internal ──────────────────────────────────────────────────────────────

    def _find_carrier(
        self,
        snap: "AnalyticsSnapshot",
        mgr: HomographyManager,
    ) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        """Find the possession player in both pitch-m and image-px."""
        if snap.ball_xy is None or snap.possession.owner is None:
            return None, None
        positions = snap.positions.get(snap.possession.owner, [])
        if not positions:
            return None, None
        ball = snap.ball_xy
        nearest = min(positions, key=lambda p: math.hypot(p[0]-ball[0], p[1]-ball[1]))
        dist = math.hypot(nearest[0]-ball[0], nearest[1]-ball[1])
        if dist > _ANCHOR_MAX_DIST_M:
            return None, None
        px = _pitch_to_image(nearest, mgr)
        return nearest, px

    def _extrapolate_m(self) -> tuple[float, float] | None:
        """Linear extrapolation of the pitch-space ball trajectory, 1 frame ahead."""
        if len(self._ball_hist_m) < 3:
            return None
        pts = list(self._ball_hist_m)
        p1, p3 = pts[-3], pts[-1]
        vx = (p3[0] - p1[0]) / 2.0
        vy = (p3[1] - p1[1]) / 2.0
        speed = math.hypot(vx, vy)
        if speed > 25.0:   # >25 m per detection-frame is unphysical (pitch = 105m)
            return None
        return (p3[0] + vx, p3[1] + vy)

    def _space_score(self, candidate_m: tuple[float, float]) -> float:
        """
        Nearest-player Voronoi proxy: is the candidate in the possession team's
        space?  Pure distance comparison — no full cell decomposition needed.
        """
        if self._snap is None or self._snap.possession.owner is None:
            return 0.5
        owner = self._snap.possession.owner
        opp   = "teamB" if owner == "teamA" else "teamA"
        own_pos = self._snap.positions.get(owner, [])
        opp_pos = self._snap.positions.get(opp,   [])
        if not own_pos and not opp_pos:
            return 0.5
        cx, cy = candidate_m
        d_own = min((math.hypot(p[0]-cx, p[1]-cy) for p in own_pos), default=999.0)
        d_opp = min((math.hypot(p[0]-cx, p[1]-cy) for p in opp_pos), default=999.0)
        if d_own < d_opp:
            return _SPACE_SCORE_OWN
        if d_opp < d_own * 0.75:   # clearly in opponent space
            return _SPACE_SCORE_OPP
        return 0.5

    def _event_score(self, candidate_m: tuple[float, float]) -> float:
        """Additional score modifier from active tactical events."""
        score = 0.5

        # After LINE_BREAK: ball should be past the defensive line
        if self._ctx.line_break_x is not None and self._ctx.line_break_sign != 0:
            cx = candidate_m[0]
            lx = self._ctx.line_break_x
            sgn = self._ctx.line_break_sign
            if sgn == +1:
                # Attacking team moved past high x — candidate should be > lx
                score += 0.25 if cx > lx else -0.20
            else:
                # Attacking team moved past low x — candidate should be < lx
                score += 0.25 if cx < lx else -0.20

        # During COUNTER: prefer candidates further along the pitch than average
        # (rough heuristic — the ball is moving forward quickly)
        if self._ctx.in_counter and self._snap is not None:
            ball = self._snap.ball_xy
            if ball is not None:
                owner = self._snap.possession.owner
                if owner:
                    att_dir = (self._snap.team_a.attack_dir if owner == "teamA"
                               else self._snap.team_b.attack_dir)
                    fwd = (candidate_m[0] - ball[0]) * att_dir
                    score += min(0.2, fwd / 20.0)

        return max(0.0, min(1.0, score))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pitch_to_image(
    pos_m: tuple[float, float],
    mgr: HomographyManager,
) -> tuple[float, float] | None:
    """Invert H to map pitch-metres → image-pixels."""
    if not mgr.is_valid():
        return None
    try:
        H_inv = np.linalg.inv(mgr.H)
    except np.linalg.LinAlgError:
        return None
    pt = np.array([pos_m[0], pos_m[1], 1.0], dtype=np.float64)
    r  = H_inv @ pt
    if abs(r[2]) < 1e-9:
        return None
    return (float(r[0] / r[2]), float(r[1] / r[2]))
