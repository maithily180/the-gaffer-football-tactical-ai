"""
gaffer/analytics/match_report.py
───────────────────────────────────
v1.9 — Structured match report.

Fully factual, no LLM: a single end-of-match summary built mostly from
totals the rest of the pipeline already tracks (possession, passes,
episodes), plus two running aggregates nothing upstream accumulates on its
own — space control is a per-frame Voronoi snapshot, and LINE_BREAK /
OVERLOAD / DOMINANCE are momentary events with no master count anywhere
else.

    MATCH REPORT
    Possession: A 58% B 42%
    Space Control: A 61% B 39%
    Passes: 11
    Progressive Passes: 4
    Line Breaks: 7
    Overloads: 12
    Dominance Periods: 3

    Top Tactical Episodes
    #1 Recovery -> Counter -> Line Break
    #2 Overload -> Progressive Pass
    #3 High Press -> Recovery
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gaffer.analytics.episodes import Episode
from gaffer.events.base import DOMINANCE, LINE_BREAK, OVERLOAD

_TRACKED_COUNTS = (LINE_BREAK, OVERLOAD, DOMINANCE)


@dataclass
class MatchReport:
    possession_pct:      tuple[float, float]    # (teamA, teamB)
    space_control_pct:   tuple[float, float]    # (teamA, teamB), match-mean
    total_passes:        int
    progressive_passes:  int
    line_breaks:         int
    overloads:           int
    dominance_periods:   int
    top_episodes:        list[Episode] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            "MATCH REPORT",
            f"Possession: A {self.possession_pct[0]:.0f}% B {self.possession_pct[1]:.0f}%",
            f"Space Control: A {self.space_control_pct[0]:.0f}% B {self.space_control_pct[1]:.0f}%",
            f"Passes: {self.total_passes}",
            f"Progressive Passes: {self.progressive_passes}",
            f"Line Breaks: {self.line_breaks}",
            f"Overloads: {self.overloads}",
            f"Dominance Periods: {self.dominance_periods}",
            "",
            "Top Tactical Episodes",
        ]
        if self.top_episodes:
            lines += [f"#{i} {ep.narrative()}" for i, ep in enumerate(self.top_episodes, start=1)]
        else:
            lines.append("(none)")
        return "\n".join(lines)


class MatchStatsAccumulator:
    """
    Stateful running totals the rest of the pipeline doesn't already keep.
    Possession / passes / episodes are cumulative inside their own trackers
    -- this only covers what isn't: space control is a per-frame Voronoi
    snapshot (needs averaging over the match), and LINE_BREAK / OVERLOAD /
    DOMINANCE are momentary FootballEvents with no master tally upstream
    (AnalyticsSnapshot.events holds only the current frame's events).
    """

    def __init__(self):
        self._sc_sum = [0.0, 0.0]
        self._sc_n = 0
        self._event_counts: dict[str, int] = {t: 0 for t in _TRACKED_COUNTS}

    def update(self, snap) -> None:
        if snap.space_control is not None:
            self._sc_sum[0] += snap.space_control.teamA_pct
            self._sc_sum[1] += snap.space_control.teamB_pct
            self._sc_n += 1
        for ev in snap.events:
            if ev.event_type in self._event_counts:
                self._event_counts[ev.event_type] += 1

    def space_control_pct(self) -> tuple[float, float]:
        if self._sc_n == 0:
            return (50.0, 50.0)
        return (round(self._sc_sum[0] / self._sc_n, 1),
                round(self._sc_sum[1] / self._sc_n, 1))

    def event_count(self, event_type: str) -> int:
        return self._event_counts.get(event_type, 0)
