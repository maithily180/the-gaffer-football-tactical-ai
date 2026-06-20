"""
gaffer/tracking/world_model_v2.py
───────────────────────────────────
World Model v2 — turns the v1.0 reactive prior (where IS the ball, given
possession/trajectory/space/events) into a predictive one (where is the ball
GOING, given the football intelligence Gaffer has already built).

v1.0 (BallWorldModel) answers "is this candidate near the ball carrier / on
the extrapolated trajectory / in the right team's space."  Those are all
present-tense, position-only signals.  v2 adds signals that come from facts
only the analytics layers above it know:

    Signal A — Pass corridor prediction
        PassDetector has been recording sender->receiver pairs all match.
        Once we know who the carrier is, we know (empirically, per-player)
        who they tend to release the ball to.  When the ball goes missing,
        search ALONG that corridor instead of in a generic radius.

    Signal B — Possession-chain transition probabilities
        The corridor target above IS the learned transition; this is just
        naming it.  Falls back to "nearest forward teammate" for players
        with no pass history yet (early match / cold start).

    Signal C — Space-control zone prior
        v1.0's space score is a cheap nearest-player proxy.  This adds the
        real Voronoi-clipped zone control % (space_control.py) for the exact
        zone the candidate falls in — a candidate sitting in a zone the
        possession team controls 80% of is far more plausible than one in a
        contested 50/50 zone, independent of who's nearest.

    Signal D — Overload zone prior
        OVERLOAD events mark where a team currently has a sustained numbers
        advantage.  That's where the ball is statistically likely to be —
        independent of exact player distance.  Decays a few seconds after
        the advantage fires, same pattern as the press/counter decay in v1.

    Signal E — Press locality
        HIGH_PRESS means dense, short, local play.  While a press is active,
        candidates far from the last known ball position are penalised —
        the ball doesn't travel far under a press.

All five are added as EXTRA terms in the same 0-1 averaged scoring scheme
v1.0 uses; each is only appended when actually informative for the
candidate (no contribution → no neutral 0.5 padding → doesn't dilute the
existing v1.0 signals when nothing new applies).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from gaffer import config
from gaffer.calibration.homography_manager import HomographyManager
from gaffer.events.base import OVERLOAD
from gaffer.tracking.world_model import _ANCHOR_MAX_DIST_M, BallWorldModel, _pitch_to_image

if TYPE_CHECKING:
    from gaffer.analytics.engine import AnalyticsSnapshot

# ── Tuning ────────────────────────────────────────────────────────────────────
_CORRIDOR_DECAY_M       = 10.0   # score falls to 0 this far perpendicular from the corridor line
_OVERLOAD_ZONE_DECAY_FRAMES = 90   # ~3.6s at DETECT_EVERY_N_FRAMES=3, 25fps — sustained relevance
_OVERLOAD_ZONE_RADIUS_M = 14.0   # score blends from boost -> neutral over this distance
_OVERLOAD_ZONE_BOOST    = 0.80
_PRESS_LOCAL_DECAY_M    = 10.0   # during a press, ball rarely travels further than this per frame
_ZONE_N_THIRDS          = 3      # must match overload.py / space_control.py grid
_ZONE_N_LANES           = 5


class WorldModelV2(BallWorldModel):
    """
    Drop-in replacement for BallWorldModel — same update()/score_candidate_px()/
    recovery_anchor_px() contract, richer internal signals.  Anywhere a
    BallWorldModel is constructed and passed to BallCandidateFilter.filter(),
    a WorldModelV2 instance works unchanged.
    """

    def __init__(self, fps: float = config.DEFAULT_FPS):
        super().__init__(fps=fps)
        # sender_track_id -> {receiver_track_id: completed_pass_count}
        self._transitions: dict[int, dict[int, int]] = {}
        # predicted (sender_pos_m, receiver_pos_m) corridor for the current carrier
        self._corridor: tuple[tuple[float, float], tuple[float, float]] | None = None
        # zone-key -> centre_m, zone-key -> frames_remaining (decay)
        self._overload_zone_centre: dict[tuple, tuple[float, float]] = {}
        self._overload_zone_decay:  dict[tuple, int] = {}

    # ── Public API (override) ───────────────────────────────────────────────

    def update(self, snap: "AnalyticsSnapshot", mgr: HomographyManager) -> None:
        super().update(snap, mgr)

        # Learn sender->receiver frequencies from confirmed passes (Signals A/B)
        pe = snap.pass_event
        if pe is not None:
            counts = self._transitions.setdefault(pe.sender_id, {})
            counts[pe.receiver_id] = counts.get(pe.receiver_id, 0) + 1

        # Tick down overload-zone decay, drop expired zones (Signal D)
        for key in list(self._overload_zone_decay):
            self._overload_zone_decay[key] -= 1
            if self._overload_zone_decay[key] <= 0:
                del self._overload_zone_decay[key]
                self._overload_zone_centre.pop(key, None)
        for ev in snap.events:
            if ev.event_type == OVERLOAD and ev.location_m is not None:
                key = (ev.team, ev.data.get("lane"), ev.data.get("third"))
                self._overload_zone_centre[key] = ev.location_m
                self._overload_zone_decay[key]  = _OVERLOAD_ZONE_DECAY_FRAMES

        # Predict the pass corridor from whoever currently has the ball
        carrier_id = self._find_carrier_id(snap)
        self._corridor = (
            self._predict_corridor_m(snap, carrier_id) if carrier_id is not None else None
        )

    def recovery_anchor_px(
        self,
        cluster_px: tuple[float, float] | None,
        mgr: HomographyManager,
    ) -> tuple[float, float] | None:
        """Priority: possession carrier > predicted pass receiver > trajectory > cluster."""
        if self._ctx.possession_anchor_px is not None:
            return self._ctx.possession_anchor_px
        if self._corridor is not None:
            px = _pitch_to_image(self._corridor[1], mgr)
            if px is not None:
                return px
        if self._ctx.expected_ball_m is not None:
            px = _pitch_to_image(self._ctx.expected_ball_m, mgr)
            if px is not None:
                return px
        return cluster_px

    @property
    def corridor_m(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """(carrier_pos_m, predicted_receiver_pos_m), for HUD/overlay drawing."""
        return self._corridor

    # ── Signal aggregation (override) ───────────────────────────────────────

    def _signals(self, candidate_m: tuple[float, float]) -> list[float]:
        signals = super()._signals(candidate_m)

        corridor_s = self._corridor_score(candidate_m)
        if corridor_s is not None:
            signals.append(corridor_s)

        overload_s = self._overload_zone_score(candidate_m)
        if overload_s is not None:
            signals.append(overload_s)

        sc_zone_s = self._space_control_zone_score(candidate_m)
        if sc_zone_s is not None:
            signals.append(sc_zone_s)

        press_s = self._press_local_score(candidate_m)
        if press_s is not None:
            signals.append(press_s)

        return signals

    # ── Internal: pass corridor (Signals A/B) ───────────────────────────────

    def _find_carrier_id(self, snap: "AnalyticsSnapshot") -> int | None:
        """track_id of the nearest possession-team player to the ball, if close enough."""
        if snap.ball_xy is None or snap.possession.owner is None:
            return None
        ball = snap.ball_xy
        owner_team = snap.possession.owner
        best_id, best_dist = None, math.inf
        for tid, pos in snap.player_positions_m.items():
            if snap.player_teams.get(tid) != owner_team:
                continue
            d = math.hypot(pos[0] - ball[0], pos[1] - ball[1])
            if d < best_dist:
                best_dist, best_id = d, tid
        if best_id is None or best_dist > _ANCHOR_MAX_DIST_M:
            return None
        return best_id

    def _predict_corridor_m(
        self, snap: "AnalyticsSnapshot", carrier_id: int,
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        carrier_pos = snap.player_positions_m.get(carrier_id)
        if carrier_pos is None:
            return None
        team = snap.player_teams.get(carrier_id)

        # Learned transition: this exact sender's most frequent receiver so far
        receiver_pos = None
        history = self._transitions.get(carrier_id)
        if history:
            receiver_id = max(history, key=history.get)
            receiver_pos = snap.player_positions_m.get(receiver_id)

        # Cold start: nearest teammate ahead of the carrier (likely pass target)
        if receiver_pos is None:
            attack_dir = (snap.team_a.attack_dir if team == "teamA"
                          else snap.team_b.attack_dir)
            teammates = [
                (tid, pos) for tid, pos in snap.player_positions_m.items()
                if tid != carrier_id and snap.player_teams.get(tid) == team
            ]
            if attack_dir != 0:
                ahead = [tp for tp in teammates
                         if (tp[1][0] - carrier_pos[0]) * attack_dir > 0]
                pool = ahead or teammates
            else:
                pool = teammates
            if pool:
                receiver_pos = min(
                    pool, key=lambda tp: math.hypot(
                        tp[1][0] - carrier_pos[0], tp[1][1] - carrier_pos[1]
                    )
                )[1]

        if receiver_pos is None:
            return None
        return carrier_pos, receiver_pos

    def _corridor_score(self, candidate_m: tuple[float, float]) -> float | None:
        if self._corridor is None:
            return None
        a, b = self._corridor
        d = _point_segment_dist_m(candidate_m, a, b)
        if d > _CORRIDOR_DECAY_M:
            return None
        return max(0.0, 1.0 - d / _CORRIDOR_DECAY_M)

    # ── Internal: overload zone prior (Signal D) ────────────────────────────

    def _overload_zone_score(self, candidate_m: tuple[float, float]) -> float | None:
        if not self._overload_zone_centre:
            return None
        best_d = min(
            math.hypot(candidate_m[0] - c[0], candidate_m[1] - c[1])
            for c in self._overload_zone_centre.values()
        )
        if best_d > _OVERLOAD_ZONE_RADIUS_M:
            return None
        frac = best_d / _OVERLOAD_ZONE_RADIUS_M
        return _OVERLOAD_ZONE_BOOST - (_OVERLOAD_ZONE_BOOST - 0.5) * frac

    # ── Internal: real space-control zone (Signal C) ────────────────────────

    def _space_control_zone_score(self, candidate_m: tuple[float, float]) -> float | None:
        if (self._snap is None or self._snap.space_control is None
                or self._ctx.possession_team is None):
            return None
        third_w = config.PITCH_LENGTH_M / _ZONE_N_THIRDS
        lane_w  = config.PITCH_WIDTH_M / _ZONE_N_LANES
        ti = min(max(int(candidate_m[0] // third_w), 0), _ZONE_N_THIRDS - 1)
        li = min(max(int(candidate_m[1] // lane_w), 0), _ZONE_N_LANES - 1)
        for z in self._snap.space_control.zones:
            if z.third_idx == ti and z.lane_idx == li:
                pct = (z.teamA_pct if self._ctx.possession_team == "teamA"
                       else z.teamB_pct)
                return pct / 100.0
        return None

    # ── Internal: press locality (Signal E) ─────────────────────────────────

    def _press_local_score(self, candidate_m: tuple[float, float]) -> float | None:
        if not self._ctx.in_high_press or self._snap is None or self._snap.ball_xy is None:
            return None
        d = math.hypot(
            candidate_m[0] - self._snap.ball_xy[0],
            candidate_m[1] - self._snap.ball_xy[1],
        )
        return max(0.2, 1.0 - d / _PRESS_LOCAL_DECAY_M)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _point_segment_dist_m(
    p: tuple[float, float], a: tuple[float, float], b: tuple[float, float],
) -> float:
    """Perpendicular distance from point p to the segment a-b (clamped to the segment)."""
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    if length2 < 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length2))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)
