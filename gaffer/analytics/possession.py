"""
gaffer/analytics/possession.py
───────────────────────────────
Ball ownership + running possession percentage.

Possession is the simplest useful football fact: which team is nearest the
ball.  Two complications are handled:

1. Loose-ball jitter
   The ball detection wobbles and players cluster — naive "nearest player"
   flips team every frame.  We require the nearest player to be within
   OWN_DIST_M and apply hysteresis: once a team owns the ball it keeps it
   until an opponent is clearly closer (within OWN_DIST_M) for HOLD_FRAMES
   consecutive detection frames.

2. Ball lost
   When the ball position is unknown we hold the last owner rather than
   resetting to contested — possession in football persists through brief
   occlusions.

Percentages are computed over detection frames where SOME team owned the ball
(contested / ball-lost-too-long frames are excluded from the denominator).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PossessionState:
    owner:      str | None          # "teamA" | "teamB" | None (contested)
    owner_dist_m: float | None      # nearest-player distance when owned
    pct_a:      float               # running possession % for team A
    pct_b:      float               # running possession % for team B


class PossessionTracker:
    def __init__(
        self,
        own_dist_m:  float = 3.5,    # nearest player within this → can own the ball
        hold_frames: int   = 3,      # opponent must be closer this many frames to steal
    ):
        self._own_dist  = own_dist_m
        self._hold      = hold_frames
        self._owner: str | None = None
        self._challenge: str | None = None
        self._challenge_streak = 0
        self._count = {"teamA": 0, "teamB": 0}

    def update(
        self,
        ball_xy:     tuple[float, float] | None,
        team_a_pos:  list[tuple[float, float]],
        team_b_pos:  list[tuple[float, float]],
    ) -> PossessionState:
        """
        Feed the current ball position (pitch metres) and both teams' player
        positions.  Call once per detection frame.  Returns the current
        PossessionState including running percentages.
        """
        nearest_team, nearest_dist = self._nearest(ball_xy, team_a_pos, team_b_pos)

        if nearest_team is None:
            # Ball unknown or no player within own_dist → hold last owner,
            # do not count this frame toward either team.
            return self._state(nearest_dist)

        if self._owner is None:
            # Uncontested acquisition
            self._owner = nearest_team
            self._challenge = None
            self._challenge_streak = 0
        elif nearest_team == self._owner:
            self._challenge = None
            self._challenge_streak = 0
        else:
            # An opponent is now closest — needs to sustain it to steal
            if nearest_team == self._challenge:
                self._challenge_streak += 1
            else:
                self._challenge = nearest_team
                self._challenge_streak = 1
            if self._challenge_streak >= self._hold:
                self._owner = nearest_team
                self._challenge = None
                self._challenge_streak = 0

        self._count[self._owner] += 1
        return self._state(nearest_dist)

    def summary(self) -> dict:
        total = self._count["teamA"] + self._count["teamB"]
        if total == 0:
            return {"teamA_pct": 50.0, "teamB_pct": 50.0, "frames": 0}
        return {
            "teamA_pct": round(100.0 * self._count["teamA"] / total, 1),
            "teamB_pct": round(100.0 * self._count["teamB"] / total, 1),
            "frames":    total,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _nearest(
        self,
        ball_xy:    tuple[float, float] | None,
        team_a_pos: list[tuple[float, float]],
        team_b_pos: list[tuple[float, float]],
    ) -> tuple[str | None, float | None]:
        if ball_xy is None:
            return None, None
        best_team, best_dist = None, math.inf
        for team, positions in (("teamA", team_a_pos), ("teamB", team_b_pos)):
            for (px, py) in positions:
                d = math.hypot(px - ball_xy[0], py - ball_xy[1])
                if d < best_dist:
                    best_dist, best_team = d, team
        if best_team is None or best_dist > self._own_dist:
            return None, (None if best_team is None else best_dist)
        return best_team, best_dist

    def _state(self, dist: float | None) -> PossessionState:
        s = self.summary()
        return PossessionState(
            owner        = self._owner,
            owner_dist_m = (round(dist, 2) if dist is not None else None),
            pct_a        = s["teamA_pct"],
            pct_b        = s["teamB_pct"],
        )
