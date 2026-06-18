"""
gaffer/analytics/passing.py
──────────────────────────────
Pass detection — the missing edges in Gaffer's possession graph.

Everything built so far describes STATE: who has the ball, how compact a
team is, who controls which zone.  This module describes FLOW: the ball
moved from this player to that one.  That's the first signal in Gaffer that
captures intent rather than position.

Mechanics
─────────
Tracks per-player ball ownership (nearest player within OWN_DIST_M, with
hysteresis exactly like PossessionTracker but at player granularity instead
of team granularity).  When the confirmed owner changes:

    same team, far enough apart, quick enough  -> a completed PASS
    different team                              -> a turnover; closes the
                                                    current side's possession
                                                    sequence and starts a new
                                                    one for the new team

The MIN_PASS_DISTANCE_M gate is the key noise filter: two teammates standing
near each other will jitter the "nearest player" id back and forth on every
frame without the ball having gone anywhere — that's not a pass, it's ID
noise, so possession changes under this distance are silently ignored for
both pass and sequence purposes (ownership state still updates, just nothing
is recorded as a pass).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from gaffer import config

if TYPE_CHECKING:
    from gaffer.analytics.engine import AnalyticsSnapshot

_OWN_DIST_M           = 3.0    # must be this close to the ball to be considered touching it
_HOLD_FRAMES          = 2      # consecutive detection frames as nearest before owner confirmed
_MIN_PASS_DISTANCE_M  = 4.0    # sender/receiver must be at least this far apart -- noise filter
_MAX_PASS_DURATION_S  = 4.0    # longer gaps mean the ball was loose, not cleanly passed
_PROGRESSIVE_MIN_M    = 15.0   # ball advances this far toward goal -> "progressive" (roadmap spec)


@dataclass
class PassEvent:
    frame_idx:      int
    time_s:         float
    team:           str
    sender_id:      int
    receiver_id:    int
    sender_pos_m:   tuple[float, float]
    receiver_pos_m: tuple[float, float]
    distance_m:     float
    duration_s:     float
    speed_ms:       float
    direction:      str     # "forward" | "backward" | "lateral"
    progressive:    bool


class PassDetector:
    """
    Stateful.  Call update() every detection frame with the latest
    AnalyticsSnapshot and the current time_s.  Returns a PassEvent on the
    frame a clean pass completes, else None.
    """

    def __init__(
        self,
        own_dist_m: float = _OWN_DIST_M,
        hold_frames: int = _HOLD_FRAMES,
        min_pass_distance_m: float = _MIN_PASS_DISTANCE_M,
        max_pass_duration_s: float = _MAX_PASS_DURATION_S,
        progressive_min_m: float = _PROGRESSIVE_MIN_M,
    ):
        self._own_dist     = own_dist_m
        self._hold         = hold_frames
        self._min_dist     = min_pass_distance_m
        self._max_duration = max_pass_duration_s
        self._prog_min     = progressive_min_m

        self._owner_id:   int | None = None
        self._owner_team: str | None = None
        self._owner_pos:  tuple[float, float] | None = None   # freshest position while owner
        self._owner_time_s: float = 0.0

        self._challenger_id:     int | None = None
        self._challenger_streak: int = 0

        self.passes:    list[PassEvent] = []
        self.sequences: list[list[int]] = []     # completed possession sequences (track_id chains)
        self._current_sequence: list[int] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, snap: "AnalyticsSnapshot", time_s: float) -> PassEvent | None:
        if snap.ball_xy is None:
            return None

        nearest_id, nearest_team, nearest_dist = self._nearest_player(snap)
        if nearest_id is None or nearest_dist > self._own_dist:
            return None

        if self._owner_id is None:
            self._set_owner(nearest_id, nearest_team, snap, time_s)
            return None

        if nearest_id == self._owner_id:
            # Same player still has it -- refresh freshest position/time, clear any challenger
            self._owner_pos    = snap.player_positions_m.get(nearest_id, self._owner_pos)
            self._owner_time_s = time_s
            self._challenger_id     = None
            self._challenger_streak = 0
            return None

        # A different player is nearest -- needs to sustain it to be confirmed
        if nearest_id == self._challenger_id:
            self._challenger_streak += 1
        else:
            self._challenger_id     = nearest_id
            self._challenger_streak = 1

        if self._challenger_streak < self._hold:
            return None

        return self._confirm_new_owner(nearest_id, nearest_team, snap, time_s)

    def pass_network(self) -> dict[tuple[int, int], int]:
        """{(sender_id, receiver_id): completed_pass_count}."""
        counts: dict[tuple[int, int], int] = {}
        for p in self.passes:
            key = (p.sender_id, p.receiver_id)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def current_sequence(self) -> list[int]:
        return list(self._current_sequence)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _set_owner(self, tid: int, team: str | None, snap: "AnalyticsSnapshot", time_s: float) -> None:
        self._owner_id      = tid
        self._owner_team    = team
        self._owner_pos      = snap.player_positions_m.get(tid)
        self._owner_time_s  = time_s
        self._challenger_id     = None
        self._challenger_streak = 0
        if not self._current_sequence:
            self._current_sequence = [tid]

    def _confirm_new_owner(
        self, new_id: int, new_team: str | None, snap: "AnalyticsSnapshot", time_s: float
    ) -> PassEvent | None:
        old_id, old_team, old_pos, old_time = (
            self._owner_id, self._owner_team, self._owner_pos, self._owner_time_s
        )
        new_pos = snap.player_positions_m.get(new_id)
        pass_event: PassEvent | None = None

        if old_pos is not None and new_pos is not None:
            dist = math.hypot(new_pos[0] - old_pos[0], new_pos[1] - old_pos[1])
            dt   = time_s - old_time

            if (new_team == old_team and dist >= self._min_dist
                    and 0 < dt <= self._max_duration):
                attack_dir = (snap.team_a.attack_dir if new_team == "teamA"
                              else snap.team_b.attack_dir)
                fwd       = (new_pos[0] - old_pos[0]) * attack_dir if attack_dir != 0 else 0.0
                direction = _classify_direction(fwd)
                pass_event = PassEvent(
                    frame_idx=snap.frame_idx, time_s=time_s, team=new_team,
                    sender_id=old_id, receiver_id=new_id,
                    sender_pos_m=old_pos, receiver_pos_m=new_pos,
                    distance_m=round(dist, 1), duration_s=round(dt, 2),
                    speed_ms=round(dist / dt, 1) if dt > 0 else 0.0,
                    direction=direction,
                    progressive=(fwd >= self._prog_min),
                )
                self.passes.append(pass_event)
                self._current_sequence.append(new_id)

            elif new_team != old_team:
                # Turnover -- close out the old side's sequence, start a fresh one
                if self._current_sequence:
                    self.sequences.append(self._current_sequence)
                self._current_sequence = [new_id]
            # else: same team, too close/slow to count -- likely ID jitter, ignore for
            # both pass and sequence purposes (ownership still moves on below)

        self._owner_id      = new_id
        self._owner_team    = new_team
        self._owner_pos      = new_pos
        self._owner_time_s  = time_s
        self._challenger_id     = None
        self._challenger_streak = 0

        return pass_event

    @staticmethod
    def _nearest_player(
        snap: "AnalyticsSnapshot",
    ) -> tuple[int | None, str | None, float]:
        ball = snap.ball_xy
        best_id, best_team, best_dist = None, None, math.inf
        for tid, pos in snap.player_positions_m.items():
            d = math.hypot(pos[0] - ball[0], pos[1] - ball[1])
            if d < best_dist:
                best_dist, best_id, best_team = d, tid, snap.player_teams.get(tid)
        return best_id, best_team, best_dist


def _classify_direction(fwd_component_m: float) -> str:
    if fwd_component_m > 3.0:
        return "forward"
    if fwd_component_m < -3.0:
        return "backward"
    return "lateral"
