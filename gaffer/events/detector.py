"""
gaffer/events/detector.py
──────────────────────────
Stateful football event detector.  Call update() once per detection frame
with the current AnalyticsSnapshot; it returns every FootballEvent that
fired in that frame.

Detected events
───────────────
  possession_change    owner transitions between teamA ↔ teamB
  possession_recovery  team regains ball within RECOVERY_WINDOW_S after losing it
  counter_attack       recovery + ball advances ≥ COUNTER_FWD_M toward goal
  high_press           press intensity ≥ threshold for HIGH_PRESS_MIN_FRAMES frames (onset)
  high_press_ended     press intensity drops below threshold (offset)
  line_break           ball x-coord crosses the defending team's backline
  sprint_start/end     per-player speed crosses SPRINT_THRESHOLD_MS
  compact_block        defending team enters width < COMPACT_WIDTH_M + area < COMPACT_AREA_M2
  progressive_pass     ball advances ≥ PROG_PASS_MIN_M toward goal between consecutive frames

All measurements are in pitch metres; times in seconds at the supplied fps.
"""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING

from gaffer import config
from gaffer.analytics.overload import compute_overloads, significant_overloads, third_label, zone_centre_m
from gaffer.events.base import (
    COMPACT_BLOCK,
    COUNTER_ATTACK,
    HIGH_PRESS,
    HIGH_PRESS_ENDED,
    LINE_BREAK,
    OVERLOAD,
    POSSESSION_CHANGE,
    POSSESSION_RECOVERY,
    PROGRESSIVE_PASS,
    SPRINT_END,
    SPRINT_START,
    FootballEvent,
)

if TYPE_CHECKING:
    # Avoid circular import; engine imports us after we import base
    from gaffer.analytics.engine import AnalyticsSnapshot

# ── Tuning constants ─────────────────────────────────────────────────────────
_HIGH_PRESS_MIN_FRAMES = 2           # detection frames at intensity≥threshold → onset
_RECOVERY_WINDOW_S     = 4.0         # seconds: if you regain possession this soon it's a recovery
_COUNTER_FWD_M         = 6.0         # metres ball must advance toward goal (within 3s) → counter
_COUNTER_WINDOW_S      = 3.0
_SPRINT_THRESHOLD_MS   = config.SPRINT_THRESHOLD_MS  # 7.0 m/s
_SPRINT_MIN_FRAMES     = config.SPRINT_MIN_FRAMES     # 3 consecutive frames
_COMPACT_WIDTH_M       = 35.0        # metres
_COMPACT_AREA_M2       = 950.0       # m² — low block threshold
_COMPACT_MIN_FRAMES    = 3           # must be compact this long before we emit the event
_PROG_PASS_MIN_M       = 12.0        # ball advances this far toward goal in one detect frame
_OVERLOAD_THRESHOLD    = 2           # min player-count advantage in a zone to flag
_OVERLOAD_MIN_FRAMES   = 2           # sustained detection frames before firing (debounce)


class EventDetector:
    """
    Consumes AnalyticsSnapshot objects in detection-frame order and emits
    FootballEvent objects whenever a football event is detected.
    """

    def __init__(self, fps: float = config.DEFAULT_FPS):
        self._fps = fps

        # Possession state
        self._prev_owner: str | None = None
        self._prev_owner_at_s: float = 0.0    # when current owner gained possession
        self._lost_at: dict[str, float] = {}   # team → time_s when they last lost possession

        # High press
        self._press_streak: int = 0
        self._press_active: bool = False

        # Line-break: track last side of def-line for each team's ball relationship
        # key: "attacking_teamA" or "attacking_teamB"
        # value: True if ball was BEYOND the defending team's line last frame
        self._past_def_line: dict[str, bool] = {}

        # Sprint: per track_id consecutive-frame counter
        self._sprint_streak:   dict[int, int]  = {}
        self._sprinting:       set[int]        = set()
        # Per-player position history for speed computation
        # track_id → deque of (frame_idx, (x, y))
        self._player_pos_hist: dict[int, deque[tuple[int, tuple[float, float]]]] = {}
        self._HIST_LEN = 4

        # Compact block
        self._compact_streak: dict[str, int] = {"teamA": 0, "teamB": 0}
        self._compact_active: dict[str, bool] = {"teamA": False, "teamB": False}

        # Progressive pass
        self._prev_ball_xy: tuple[float, float] | None = None
        self._prev_ball_attack_dir: int = 0   # +1 or -1 from last snap

        # Overload: per-zone (third_idx, lane_idx) state
        self._overload_active:    dict[tuple[int, int], str] = {}  # team we've already fired for
        self._overload_streak:    dict[tuple[int, int], int] = {}  # consecutive frames same team has advantage
        self._overload_last_team: dict[tuple[int, int], str] = {}  # team with advantage last frame

        # Counter-attack: track recent ball positions for forward-movement check
        self._ball_hist: deque[tuple[float, float, float]] = deque(maxlen=128)
        # (time_s, x, y) — kept for COUNTER_WINDOW_S window

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, snap: "AnalyticsSnapshot") -> list[FootballEvent]:
        """
        Process one AnalyticsSnapshot.  Returns a list of events that fired
        this frame (may be empty).  Call once per detection frame.
        """
        time_s = snap.frame_idx / self._fps
        events: list[FootballEvent] = []

        self._update_player_speeds(snap)

        events += self._check_possession(snap, time_s)
        events += self._check_high_press(snap, time_s)
        events += self._check_line_break(snap, time_s)
        events += self._check_sprint(snap, time_s)
        events += self._check_compact_block(snap, time_s)
        events += self._check_progressive_pass(snap, time_s)
        events += self._check_overload(snap, time_s)

        # Update ball history AFTER checks (so we compare prev→curr)
        if snap.ball_xy is not None:
            self._ball_hist.append((time_s, snap.ball_xy[0], snap.ball_xy[1]))
        self._prev_ball_xy = snap.ball_xy
        self._prev_ball_attack_dir = (
            snap.team_a.attack_dir if snap.team_a.attack_dir != 0 else self._prev_ball_attack_dir
        )

        return events

    # ── Possession ────────────────────────────────────────────────────────────

    def _check_possession(
        self, snap: "AnalyticsSnapshot", time_s: float
    ) -> list[FootballEvent]:
        owner = snap.possession.owner
        events: list[FootballEvent] = []

        if owner == self._prev_owner:
            return events

        if self._prev_owner is not None and owner is not None:
            # Real ownership transition
            new_team  = owner
            lost_team = self._prev_owner
            loc = snap.ball_xy

            events.append(FootballEvent(
                frame_idx=snap.frame_idx, time_s=time_s,
                event_type=POSSESSION_CHANGE,
                team=new_team, location_m=loc,
                data={"from": lost_team, "to": new_team},
            ))

            # Recovery: regaining within RECOVERY_WINDOW_S
            lost_ago = time_s - self._lost_at.get(new_team, -999)
            if lost_ago <= _RECOVERY_WINDOW_S:
                events.append(FootballEvent(
                    frame_idx=snap.frame_idx, time_s=time_s,
                    event_type=POSSESSION_RECOVERY,
                    team=new_team, location_m=loc,
                    data={"seconds_ago": round(lost_ago, 1)},
                ))
                # Check for counter-attack: did the ball move forward rapidly?
                events += self._check_counter(snap, new_team, time_s, loc)

            self._lost_at[lost_team] = time_s

        elif owner is not None:
            # Transitioning from None (contested) → owned
            self._lost_at.setdefault(owner, 0.0)

        self._prev_owner = owner
        return events

    def _check_counter(
        self,
        snap: "AnalyticsSnapshot",
        team: str,
        time_s: float,
        loc: tuple[float, float] | None,
    ) -> list[FootballEvent]:
        """Ball moved ≥ COUNTER_FWD_M toward goal within COUNTER_WINDOW_S."""
        if not self._ball_hist:
            return []
        # Attack direction for the recovering team
        attack_dir = (
            snap.team_a.attack_dir if team == "teamA" else snap.team_b.attack_dir
        )
        if attack_dir == 0:
            return []

        cutoff = time_s - _COUNTER_WINDOW_S
        # oldest ball position within window
        oldest = None
        for (t, bx, by) in self._ball_hist:
            if t >= cutoff:
                oldest = (bx, by)
                break
        if oldest is None or snap.ball_xy is None:
            return []

        fwd = (snap.ball_xy[0] - oldest[0]) * attack_dir
        if fwd >= _COUNTER_FWD_M:
            return [FootballEvent(
                frame_idx=snap.frame_idx, time_s=time_s,
                event_type=COUNTER_ATTACK,
                team=team, location_m=loc,
                data={"fwd_m": round(fwd, 1), "window_s": round(time_s - cutoff, 1)},
            )]
        return []

    # ── High press ────────────────────────────────────────────────────────────

    def _check_high_press(
        self, snap: "AnalyticsSnapshot", time_s: float
    ) -> list[FootballEvent]:
        events: list[FootballEvent] = []
        if snap.pressing is None:
            return events

        intensity = snap.pressing.get("intensity", 0)
        threshold = config.HIGH_PRESS_INTENSITY_THRESHOLD

        if intensity >= threshold:
            self._press_streak += 1
            if self._press_streak == _HIGH_PRESS_MIN_FRAMES and not self._press_active:
                # Determine pressing team = the opponent of possession owner
                owner = snap.possession.owner
                pressing_team = (
                    ("teamB" if owner == "teamA" else "teamA") if owner else None
                )
                self._press_active = True
                events.append(FootballEvent(
                    frame_idx=snap.frame_idx, time_s=time_s,
                    event_type=HIGH_PRESS,
                    team=pressing_team, location_m=snap.ball_xy,
                    data={"intensity": intensity},
                ))
        else:
            if self._press_active:
                pressing_team = (
                    ("teamB" if snap.possession.owner == "teamA" else "teamA")
                    if snap.possession.owner else None
                )
                events.append(FootballEvent(
                    frame_idx=snap.frame_idx, time_s=time_s,
                    event_type=HIGH_PRESS_ENDED,
                    team=pressing_team, location_m=snap.ball_xy,
                    data={"peak_streak": self._press_streak},
                ))
            self._press_streak = 0
            self._press_active = False

        return events

    # ── Line break ────────────────────────────────────────────────────────────

    def _check_line_break(
        self, snap: "AnalyticsSnapshot", time_s: float
    ) -> list[FootballEvent]:
        """
        Emit LINE_BREAK when the ball crosses the defending team's backline.
        Checks for both teams.
        """
        if snap.ball_xy is None:
            return []

        events: list[FootballEvent] = []
        bx = snap.ball_xy[0]

        for att_team, def_team, shape in (
            ("teamA", "teamB", snap.team_b),
            ("teamB", "teamA", snap.team_a),
        ):
            attack_dir = (snap.team_a.attack_dir if att_team == "teamA"
                          else snap.team_b.attack_dir)
            if attack_dir == 0 or shape.def_line_m is None:
                continue

            # Convert def_line_m (distance from own goal) to absolute x
            if attack_dir == +1:
                # defending team attacks -x, own goal at x=105
                # def_line_m is 105 - backline_x
                def_x = config.PITCH_LENGTH_M - shape.def_line_m
                # ball is "past the line" if bx > def_x
                past = bx > def_x
            else:
                # defending team attacks +x, own goal at x=0
                # def_line_m is backline_x
                def_x = shape.def_line_m
                past = bx < def_x

            key = f"att_{att_team}"
            was_past = self._past_def_line.get(key, False)
            if past and not was_past:
                events.append(FootballEvent(
                    frame_idx=snap.frame_idx, time_s=time_s,
                    event_type=LINE_BREAK,
                    team=att_team, location_m=snap.ball_xy,
                    data={"def_team": def_team, "def_line_x": round(def_x, 1)},
                ))
            self._past_def_line[key] = past

        return events

    # ── Sprint ────────────────────────────────────────────────────────────────

    def _update_player_speeds(self, snap: "AnalyticsSnapshot") -> None:
        """Keep per-player pitch-position history for speed computation."""
        for track_id, pos in snap.player_positions_m.items():
            if track_id < 0:
                continue
            if track_id not in self._player_pos_hist:
                self._player_pos_hist[track_id] = deque(maxlen=self._HIST_LEN)
            self._player_pos_hist[track_id].append((snap.frame_idx, pos))

    def _check_sprint(
        self, snap: "AnalyticsSnapshot", time_s: float
    ) -> list[FootballEvent]:
        events: list[FootballEvent] = []

        for track_id, hist in self._player_pos_hist.items():
            if len(hist) < 2:
                continue
            f1, p1 = hist[0]
            f2, p2 = hist[-1]
            dt_s = (f2 - f1) / self._fps
            if dt_s < 1e-6:
                continue
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            speed_ms = dist / dt_s

            if speed_ms >= _SPRINT_THRESHOLD_MS:
                self._sprint_streak[track_id] = self._sprint_streak.get(track_id, 0) + 1
                if (self._sprint_streak[track_id] == _SPRINT_MIN_FRAMES
                        and track_id not in self._sprinting):
                    self._sprinting.add(track_id)
                    team = snap.player_teams.get(track_id)
                    events.append(FootballEvent(
                        frame_idx=snap.frame_idx, time_s=time_s,
                        event_type=SPRINT_START,
                        team=team, location_m=snap.player_positions_m.get(track_id),
                        data={"track_id": track_id, "speed_ms": round(speed_ms, 1)},
                    ))
            else:
                self._sprint_streak[track_id] = 0
                if track_id in self._sprinting:
                    self._sprinting.discard(track_id)
                    team = snap.player_teams.get(track_id)
                    events.append(FootballEvent(
                        frame_idx=snap.frame_idx, time_s=time_s,
                        event_type=SPRINT_END,
                        team=team, location_m=snap.player_positions_m.get(track_id),
                        data={"track_id": track_id},
                    ))

        return events

    # ── Compact block ─────────────────────────────────────────────────────────

    def _check_compact_block(
        self, snap: "AnalyticsSnapshot", time_s: float
    ) -> list[FootballEvent]:
        events: list[FootballEvent] = []

        for team, shape in (("teamA", snap.team_a), ("teamB", snap.team_b)):
            c = shape.compactness
            is_compact = (c.n_players >= 5
                          and c.width_m <= _COMPACT_WIDTH_M
                          and c.hull_area_m2 <= _COMPACT_AREA_M2)

            if is_compact:
                self._compact_streak[team] += 1
                if (self._compact_streak[team] == _COMPACT_MIN_FRAMES
                        and not self._compact_active[team]):
                    self._compact_active[team] = True
                    events.append(FootballEvent(
                        frame_idx=snap.frame_idx, time_s=time_s,
                        event_type=COMPACT_BLOCK,
                        team=team, location_m=None,
                        data={"width_m": round(c.width_m, 1),
                              "area_m2": round(c.hull_area_m2, 0)},
                    ))
            else:
                self._compact_streak[team] = 0
                self._compact_active[team] = False

        return events

    # ── Progressive pass ──────────────────────────────────────────────────────

    def _check_progressive_pass(
        self, snap: "AnalyticsSnapshot", time_s: float
    ) -> list[FootballEvent]:
        """
        Ball advances ≥ PROG_PASS_MIN_M toward the possession team's goal
        in one detection-frame cycle while the same team retains possession.
        """
        if (snap.ball_xy is None or self._prev_ball_xy is None
                or snap.possession.owner is None):
            return []

        attack_dir = (snap.team_a.attack_dir if snap.possession.owner == "teamA"
                      else snap.team_b.attack_dir)
        if attack_dir == 0:
            return []

        fwd = (snap.ball_xy[0] - self._prev_ball_xy[0]) * attack_dir
        if fwd >= _PROG_PASS_MIN_M:
            return [FootballEvent(
                frame_idx=snap.frame_idx, time_s=time_s,
                event_type=PROGRESSIVE_PASS,
                team=snap.possession.owner, location_m=snap.ball_xy,
                data={"fwd_m": round(fwd, 1)},
            )]
        return []

    # ── Overload ──────────────────────────────────────────────────────────────

    def _check_overload(
        self, snap: "AnalyticsSnapshot", time_s: float
    ) -> list[FootballEvent]:
        """
        Numerical superiority in a pitch zone, sustained for OVERLOAD_MIN_FRAMES.
        Only fires for the team's middle/attacking third — a defensive-third
        overload is just normal defensive numbers, not a tactical event.
        """
        team_a_pos = snap.positions.get("teamA", [])
        team_b_pos = snap.positions.get("teamB", [])
        if not team_a_pos and not team_b_pos:
            return []

        zones = compute_overloads(team_a_pos, team_b_pos)
        sig   = significant_overloads(zones, threshold=_OVERLOAD_THRESHOLD)

        events: list[FootballEvent] = []
        seen_keys: set[tuple[int, int]] = set()

        for z in sig:
            key = (z.third_idx, z.lane_idx)
            adv_team   = "teamA" if z.diff > 0 else "teamB"
            attack_dir = (snap.team_a.attack_dir if adv_team == "teamA"
                          else snap.team_b.attack_dir)
            tlabel = third_label(z.third_idx, attack_dir)

            if tlabel == "defensive" or tlabel == "unknown":
                self._overload_streak.pop(key, None)
                self._overload_active.pop(key, None)
                self._overload_last_team.pop(key, None)
                continue

            seen_keys.add(key)
            if self._overload_last_team.get(key) == adv_team:
                self._overload_streak[key] = self._overload_streak.get(key, 0) + 1
            else:
                self._overload_streak[key] = 1
                self._overload_active.pop(key, None)   # advantage flipped teams — reset
            self._overload_last_team[key] = adv_team

            if (self._overload_streak[key] == _OVERLOAD_MIN_FRAMES
                    and self._overload_active.get(key) != adv_team):
                self._overload_active[key] = adv_team
                count_for     = z.teamA_count if adv_team == "teamA" else z.teamB_count
                count_against = z.teamB_count if adv_team == "teamA" else z.teamA_count
                events.append(FootballEvent(
                    frame_idx=snap.frame_idx, time_s=time_s,
                    event_type=OVERLOAD,
                    team=adv_team, location_m=zone_centre_m(z.third_idx, z.lane_idx),
                    data={"lane": z.lane_name, "third": tlabel,
                          "count_for": count_for, "count_against": count_against},
                ))

        # Clear state for zones that are no longer significant
        for key in list(self._overload_last_team.keys()):
            if key not in seen_keys:
                self._overload_streak.pop(key, None)
                self._overload_active.pop(key, None)
                self._overload_last_team.pop(key, None)

        return events
