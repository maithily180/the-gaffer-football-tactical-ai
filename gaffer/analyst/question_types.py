"""
gaffer/analyst/question_types.py
─────────────────────────────────────
Intent detection -- step 1 of Intent Detection -> Retrieval -> Structured
Evidence -> LLM Explanation. Plain keyword rules, no LLM and no classifier
model: the six question categories Gaffer needs to answer are distinct
enough in phrasing that training or prompting a classifier would be
solving a problem that doesn't exist yet. Unmatched phrasing defaults to
MATCH_SUMMARY -- a safe generic answer beats a crash on an unanticipated
question.

UNSUPPORTED exists because "default to MATCH_SUMMARY" turned out to be too
generous: questions asking for something Gaffer structurally cannot
compute (half/period splits -- match_report is a whole-match aggregate with
no time-windowing; scores/cards -- there's no goal or disciplinary
detection anywhere in the CV layer; pass accuracy -- only completed passes
are ever logged, there's no attempted-but-failed-pass concept; "best
player" -- hub centrality measures network connectivity, not skill) were
falling through to MATCH_SUMMARY, which always has *some* real evidence in
it, so the LLM would free-associate a confident-sounding but fabricated
answer instead of saying "I can't answer that." Caught by manually
stress-testing v2.0 with questions outside the original 8 examples.
"""

from __future__ import annotations

from enum import Enum

from gaffer.analyst.time_parse import find_time_window
from gaffer.events.base import (
    COMPACT_BLOCK,
    COUNTER_ATTACK,
    DOMINANCE,
    HIGH_PRESS,
    HIGH_PRESS_ENDED,
    LINE_BREAK,
    OVERLOAD,
    POSSESSION_CHANGE,
    POSSESSION_RECOVERY,
    PROGRESSIVE_PASS,
    SPRINT_START,
)


class QuestionType(Enum):
    MATCH_SUMMARY = "match_summary"
    DOMINANCE_EXPLANATION = "dominance_explanation"
    EVENT_SEARCH = "event_search"
    PLAYER_INFLUENCE = "player_influence"
    PASSING_ANALYSIS = "passing_analysis"
    TIME_WINDOW = "time_window"
    UNSUPPORTED = "unsupported"


# Order matters: _find_event_type returns the FIRST matching key, so
# multi-word phrases that are substrings of a shorter, more general keyword
# (e.g. "press ended" contains "press") must be listed before it.
_EVENT_KEYWORDS: dict[str, str] = {
    "progressive pass":     PROGRESSIVE_PASS,
    "press ended":          HIGH_PRESS_ENDED,
    "press ending":         HIGH_PRESS_ENDED,
    "press end":            HIGH_PRESS_ENDED,
    "counter":              COUNTER_ATTACK,
    "overload":             OVERLOAD,
    "line break":           LINE_BREAK,
    "line-break":           LINE_BREAK,
    "press":                HIGH_PRESS,
    "dominance":            DOMINANCE,
    "sprint":               SPRINT_START,
    "compact block":        COMPACT_BLOCK,
    "low block":            COMPACT_BLOCK,
    "recovered possession": POSSESSION_RECOVERY,
    "possession recovery":  POSSESSION_RECOVERY,
    "won back":             POSSESSION_RECOVERY,
    "won the ball back":    POSSESSION_RECOVERY,
    "turnover":             POSSESSION_CHANGE,
    "lost possession":      POSSESSION_CHANGE,
    "possession change":    POSSESSION_CHANGE,
}

_SEARCH_VERBS = ("show", "list", "find", "how many")

_INFLUENCE_KEYWORDS = ("influential", "important", "key player", "centra", "hub")

_PASSING_KEYWORDS = (
    "build up", "build-up", "buildup", "possession chain",
    "longest possession", "passing",
)

_TEAM_KEYWORDS = {"team a": "teamA", "team b": "teamB"}

# Phrases asking for something Gaffer structurally cannot compute today --
# checked before everything else so a mixed question (e.g. "why did Team A
# dominate in the first half?") honestly declines rather than silently
# answering the whole-match version of the question.
_UNSUPPORTED_PATTERNS: dict[str, str] = {
    "first half":      "match_report is a whole-match aggregate -- Gaffer doesn't break stats down by half or time window",
    "second half":     "match_report is a whole-match aggregate -- Gaffer doesn't break stats down by half or time window",
    "1st half":        "match_report is a whole-match aggregate -- Gaffer doesn't break stats down by half or time window",
    "2nd half":        "match_report is a whole-match aggregate -- Gaffer doesn't break stats down by half or time window",
    "half-time":       "match_report is a whole-match aggregate -- Gaffer doesn't break stats down by half or time window",
    "halftime":        "match_report is a whole-match aggregate -- Gaffer doesn't break stats down by half or time window",
    "final score":     "Gaffer's detection layer doesn't track goals or scorelines",
    "the score":       "Gaffer's detection layer doesn't track goals or scorelines",
    "who won":         "Gaffer's detection layer doesn't track goals or scorelines, so it has no notion of a match result",
    "who lost":        "Gaffer's detection layer doesn't track goals or scorelines, so it has no notion of a match result",
    "yellow card":     "Gaffer's detection layer doesn't track fouls or disciplinary cards",
    "red card":        "Gaffer's detection layer doesn't track fouls or disciplinary cards",
    "pass accuracy":   "Gaffer only logs completed passes via possession-transfer detection -- there's no attempted-but-failed-pass concept, so accuracy/completion rate can't be computed",
    "passing accuracy": "Gaffer only logs completed passes via possession-transfer detection -- there's no attempted-but-failed-pass concept, so accuracy/completion rate can't be computed",
    "completion rate": "Gaffer only logs completed passes via possession-transfer detection -- there's no attempted-but-failed-pass concept, so accuracy/completion rate can't be computed",
    "best player":     "player ranking here is hub centrality (how connected a player is in the passing network), not a skill or quality rating",
    "greatest player": "player ranking here is hub centrality (how connected a player is in the passing network), not a skill or quality rating",
    "goals":           "Gaffer's detection layer doesn't track goals -- there's no shot or scoring-event concept anywhere in the pipeline",
    "a goal":          "Gaffer's detection layer doesn't track goals -- there's no shot or scoring-event concept anywhere in the pipeline",
    "scored":          "Gaffer's detection layer doesn't track goals -- there's no shot or scoring-event concept anywhere in the pipeline",
    "shot":            "Gaffer's detection layer doesn't track shots -- ball events only cover passes, possession changes, and pitch-position analytics",
    "win probability": "Gaffer has no outcome-prediction model -- it only reports what was observed, not a forecast of the result",
    "chance of winning": "Gaffer has no outcome-prediction model -- it only reports what was observed, not a forecast of the result",
}


def _find_event_type(q: str) -> str | None:
    return next((v for k, v in _EVENT_KEYWORDS.items() if k in q), None)


def _find_unsupported_reason(q: str) -> str | None:
    return next((v for k, v in _UNSUPPORTED_PATTERNS.items() if k in q), None)


def classify(question: str) -> tuple[QuestionType, str | None, str | None]:
    """question -> (question type, team or None, event_type-or-unsupported-reason or None)."""
    q = question.lower()
    team = next((v for k, v in _TEAM_KEYWORDS.items() if k in q), None)

    unsupported_reason = _find_unsupported_reason(q)
    if unsupported_reason is not None:
        return QuestionType.UNSUPPORTED, team, unsupported_reason

    if find_time_window(q) is not None:
        return QuestionType.TIME_WINDOW, team, None

    if "domina" in q:
        return QuestionType.DOMINANCE_EXPLANATION, team, None

    event_type = _find_event_type(q)
    if event_type is not None and (any(v in q for v in _SEARCH_VERBS) or "all " in q):
        return QuestionType.EVENT_SEARCH, team, event_type

    if any(k in q for k in _INFLUENCE_KEYWORDS):
        return QuestionType.PLAYER_INFLUENCE, team, None

    if any(k in q for k in _PASSING_KEYWORDS):
        return QuestionType.PASSING_ANALYSIS, team, None

    if event_type is not None:
        return QuestionType.EVENT_SEARCH, team, event_type

    return QuestionType.MATCH_SUMMARY, team, None
