"""
gaffer/analyst/narrative_memory.py
─────────────────────────────────────
v3.4 — tracks match-long context (which team has had the better of recent
play, what just happened) so commentate_match() can phrase continuity
("Team B continue to dominate...") instead of narrating every episode as if
it's the first of the match. Stateful and match-spanning, unlike the rest of
gaffer/analyst/commentary.py's pure per-episode functions -- kept in its own
file for that reason.

context_line() is deliberately a single, clearly-labeled line handed to the
LLM as continuity flavor, not a new source of facts -- the prompt templates
explicitly tell the model not to treat it as something to add new detail
from, so a wrong inference here can only ever affect PHRASING, never invent
a claim the fact-check pass would need to catch.
"""

from __future__ import annotations

from collections import deque

from gaffer.analyst.commentary import importance_score
from gaffer.analytics.episodes import Episode

_RECENT_WINDOW_S = 60.0   # how far back "recent play" looks
_RECENT_MAXLEN = 8        # episodes are short; 8 comfortably spans 60s
_DOMINANCE_SHARE = 0.65   # one team's share of recent importance to call it "dominance"

_TEAM_LABEL = {"teamA": "Team A", "teamB": "Team B"}


class NarrativeMemory:
    def __init__(self) -> None:
        self._recent: deque[tuple[float, str, float]] = deque(maxlen=_RECENT_MAXLEN)
        self._last_summary: str | None = None

    def update(self, ep: Episode, facts: list[str]) -> None:
        """Call once per narrated episode, in chronological order."""
        self._recent.append((ep.start_time_s, ep.team, importance_score(ep)))
        if facts:
            self._last_summary = f"{_TEAM_LABEL.get(ep.team, ep.team)} {facts[-1]}"

    def _dominance_line(self) -> str | None:
        if len(self._recent) < 2:
            return None
        cutoff = self._recent[-1][0] - _RECENT_WINDOW_S
        window = [r for r in self._recent if r[0] >= cutoff]
        if len(window) < 2:
            return None
        by_team: dict[str, float] = {}
        for _, team, imp in window:
            by_team[team] = by_team.get(team, 0.0) + imp
        total = sum(by_team.values())
        if total <= 0:
            return None
        leader, leader_score = max(by_team.items(), key=lambda kv: kv[1])
        if leader_score / total < _DOMINANCE_SHARE:
            return None  # not clearly one-sided -- nothing honest to say
        return f"{_TEAM_LABEL.get(leader, leader)} have had the better of the last {_RECENT_WINDOW_S:.0f}s of play."

    def context_line(self) -> str | None:
        """A single line of continuity context, or None if there isn't
        enough history yet / nothing one-sided enough to say."""
        parts = [p for p in (self._dominance_line(), self._last_summary) if p]
        if not parts:
            return None
        if self._last_summary:
            parts[-1] = f"Most recently: {self._last_summary}."
        return " ".join(parts)
