"""
gaffer/analytics/episodes.py
───────────────────────────────
v1.8 — Tactical episodes.

Everything built through v1.7 describes MOMENTS: a recovery here, a line
break there, a pass network aggregated over the whole match. A coach
doesn't think in moments or match-long aggregates — they think in
*possessions*: one team had the ball, here's what they did with it, here's
how it ended.

    EPISODE #14 -- Team B
    Recovery -> Counter -> Line Break -> Progressive Pass
    Duration: 4.0s | Distance advanced: 27m | Outcome: Attacking Third Entry

An episode is exactly the span of one team's continuous possession, as
already debounced by PossessionTracker (snap.possession.owner) — no new
hysteresis logic here, just listen for when the owner changes and bucket
the FootballEvents that landed inside that window.  One exception: a
HIGH_PRESS event is tagged to the *pressing* team, who by definition don't
have the ball yet — so on its own it would never land inside any episode.
A short look-back at the moment an episode opens recovers that causal link
(the press that forced the turnover) so "High Press -> Recovery" can show
up as a single coherent story instead of two unrelated log lines.

Stateful: EpisodeDetector.update(snap, events) once per detection frame,
fed the exact `events` list EventDetector.update(snap) just returned.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from gaffer import config
from gaffer.analytics.overload import third_label
from gaffer.analytics.roles import label_for
from gaffer.events.base import (
    COUNTER_ATTACK,
    HIGH_PRESS,
    LINE_BREAK,
    PASS,
    POSSESSION_RECOVERY,
    FootballEvent,
)

OUTCOME_COUNTER               = "Counter"
OUTCOME_LINE_BREAK            = "Line Break"
OUTCOME_ATTACKING_THIRD_ENTRY = "Attacking Third Entry"
OUTCOME_PRESS_SUCCESS         = "Press Success"
OUTCOME_SUSTAINED_POSSESSION  = "Sustained Possession"
OUTCOME_LOST_POSSESSION       = "Lost Possession"

_PRESS_LOOKBACK_S = 4.0    # a HIGH_PRESS this far before episode start still counts as its cause
_SUSTAINED_MIN_S  = 12.0   # held the ball this long with no breakthrough -> still a story
_N_THIRDS         = 3      # matches overload.py / space_control.py grid


@dataclass
class Episode:
    episode_id:          int
    team:                str                            # "teamA" | "teamB"
    start_time_s:         float
    end_time_s:           float
    events:              list[FootballEvent] = field(default_factory=list)
    possession_chain:    list[str]           = field(default_factory=list)   # role labels, in order
    players:             set[str]            = field(default_factory=set)    # role labels involved
    outcome:             str                  = OUTCOME_LOST_POSSESSION
    distance_advanced_m: float | None        = None     # signed, +toward goal, along this team's attack_dir

    @property
    def duration_s(self) -> float:
        return round(self.end_time_s - self.start_time_s, 1)

    def narrative(self) -> str:
        """'Recovery -> Counter -> Line Break -> Progressive Pass'. Routine
        passes/state events are tracked in `events` but skipped here — only
        the same beats the live event ticker highlights make the story."""
        beats = [ev.label() for ev in self.events if ev.is_highlight]
        return " -> ".join(beats) if beats else "(quiet possession)"

    def report(self) -> str:
        dist = f"{self.distance_advanced_m:+.0f}m" if self.distance_advanced_m is not None else "?"
        tlabel = "Team A" if self.team == "teamA" else "Team B"
        return (f"EPISODE #{self.episode_id} -- {tlabel}\n"
                f"  {self.narrative()}\n"
                f"  Duration: {self.duration_s}s | Distance advanced: {dist} | "
                f"Outcome: {self.outcome}")


class EpisodeDetector:
    def __init__(self, fps: float = config.DEFAULT_FPS):
        self._fps = fps
        self._next_id = 1
        self._open:    dict[str, Episode] = {}     # team -> currently-open episode
        self._scratch: dict[str, dict]    = {}      # team -> {"ball_start", "reached_attacking_third"}
        self._recent:  deque[FootballEvent] = deque(maxlen=64)   # for the press look-back
        self.episodes: list[Episode] = []            # closed episodes, chronological

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, snap, events: list[FootballEvent]) -> list[Episode]:
        """Feed this frame's AnalyticsSnapshot plus the events EventDetector
        just emitted for it.  Returns any episodes that CLOSED this frame
        (usually 0, occasionally 1 — two teams can't both be open at once)."""
        closed: list[Episode] = []
        time_s = snap.frame_idx / self._fps
        owner = snap.possession.owner

        # Open/close BEFORE attaching this frame's events: COUNTER_ATTACK,
        # LINE_BREAK and POSSESSION_RECOVERY are all tagged to the team that
        # just regained the ball, which fires on this exact turnover frame —
        # if events were attached first, that team's episode wouldn't exist
        # yet and the events would be silently dropped.
        if owner is not None and owner not in self._open:
            for other_team in list(self._open):
                closed.append(self._close(other_team, time_s, snap))
            self._open[owner] = self._start(owner, time_s, snap)

        for ev in events:
            self._recent.append(ev)
            if ev.team in self._open:
                self._attach(self._open[ev.team], ev, snap)

        if owner is not None and owner in self._open:
            self._track_zone(owner, snap)

        return closed

    # ── Internal: lifecycle ──────────────────────────────────────────────────

    def _start(self, team: str, time_s: float, snap) -> Episode:
        ep = Episode(episode_id=self._next_id, team=team,
                     start_time_s=time_s, end_time_s=time_s)
        self._next_id += 1
        self._scratch[team] = {"ball_start": snap.ball_xy, "reached_attacking_third": False}

        # The press that caused this turnover is tagged to `team` while they
        # didn't have the ball yet, so it never otherwise reaches this episode.
        for ev in self._recent:
            if (ev.event_type == HIGH_PRESS and ev.team == team
                    and time_s - ev.time_s <= _PRESS_LOOKBACK_S):
                ep.events.append(ev)
        return ep

    def _close(self, team: str, time_s: float, snap) -> Episode:
        ep = self._open.pop(team)
        ep.end_time_s = time_s
        scr = self._scratch.pop(team, {})
        ep.distance_advanced_m = self._distance_advanced(team, scr, snap)
        ep.outcome = self._classify(ep, scr)
        self.episodes.append(ep)
        return ep

    def _attach(self, ep: Episode, ev: FootballEvent, snap) -> None:
        ep.events.append(ev)
        if ev.event_type == PASS:
            sender_lbl   = label_for(ev.data.get("sender_id"), snap.roles)
            receiver_lbl = label_for(ev.data.get("receiver_id"), snap.roles)
            if not ep.possession_chain:
                ep.possession_chain.append(sender_lbl)
            ep.possession_chain.append(receiver_lbl)
            ep.players.add(sender_lbl)
            ep.players.add(receiver_lbl)

    def _track_zone(self, team: str, snap) -> None:
        """Mark whether the ball reaches this team's attacking third at any
        point during the episode -- checked every frame, not just on events,
        since the ball can sit in the final third with nothing else firing."""
        if snap.ball_xy is None:
            return
        attack_dir = snap.team_a.attack_dir if team == "teamA" else snap.team_b.attack_dir
        if attack_dir == 0:
            return
        third_w = config.PITCH_LENGTH_M / _N_THIRDS
        ti = min(max(int(snap.ball_xy[0] // third_w), 0), _N_THIRDS - 1)
        if third_label(ti, attack_dir) == "attacking":
            self._scratch.setdefault(team, {})["reached_attacking_third"] = True

    # ── Internal: outcome classification ────────────────────────────────────

    def _distance_advanced(self, team: str, scr: dict, snap) -> float | None:
        start, end = scr.get("ball_start"), snap.ball_xy
        if start is None or end is None:
            return None
        attack_dir = snap.team_a.attack_dir if team == "teamA" else snap.team_b.attack_dir
        if attack_dir == 0:
            return None
        return round((end[0] - start[0]) * attack_dir, 1)

    def _classify(self, ep: Episode, scr: dict) -> str:
        """Priority order: the most specific/notable outcome wins when an
        episode satisfies more than one (e.g. a counter that also breaks the
        line is told as a Counter, the more dramatic of the two)."""
        types = {e.event_type for e in ep.events}
        if COUNTER_ATTACK in types:
            return OUTCOME_COUNTER
        if LINE_BREAK in types:
            return OUTCOME_LINE_BREAK
        if scr.get("reached_attacking_third"):
            return OUTCOME_ATTACKING_THIRD_ENTRY
        if HIGH_PRESS in types and POSSESSION_RECOVERY in types:
            return OUTCOME_PRESS_SUCCESS
        if ep.duration_s >= _SUSTAINED_MIN_S:
            return OUTCOME_SUSTAINED_POSSESSION
        return OUTCOME_LOST_POSSESSION
