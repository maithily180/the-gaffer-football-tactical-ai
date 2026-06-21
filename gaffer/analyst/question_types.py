"""
gaffer/analyst/question_types.py
─────────────────────────────────────
Intent detection -- step 1 of Intent Detection -> Retrieval -> Structured
Evidence -> LLM Explanation. Plain keyword rules, no LLM and no classifier
model: the five question categories Gaffer needs to answer are distinct
enough in phrasing that training or prompting a classifier would be
solving a problem that doesn't exist yet. Unmatched phrasing defaults to
MATCH_SUMMARY -- a safe generic answer beats a crash on an unanticipated
question.
"""

from __future__ import annotations

from enum import Enum

from gaffer.events.base import (
    COUNTER_ATTACK,
    DOMINANCE,
    HIGH_PRESS,
    LINE_BREAK,
    OVERLOAD,
    PROGRESSIVE_PASS,
)


class QuestionType(Enum):
    MATCH_SUMMARY = "match_summary"
    DOMINANCE_EXPLANATION = "dominance_explanation"
    EVENT_SEARCH = "event_search"
    PLAYER_INFLUENCE = "player_influence"
    PASSING_ANALYSIS = "passing_analysis"


_EVENT_KEYWORDS: dict[str, str] = {
    "progressive pass": PROGRESSIVE_PASS,   # checked before the bare "pass"-style keys
    "counter":          COUNTER_ATTACK,
    "overload":         OVERLOAD,
    "line break":       LINE_BREAK,
    "line-break":       LINE_BREAK,
    "press":            HIGH_PRESS,
    "dominance":        DOMINANCE,
}

_SEARCH_VERBS = ("show", "list", "find", "how many")

_INFLUENCE_KEYWORDS = ("influential", "important", "key player", "centra", "hub")

_PASSING_KEYWORDS = (
    "build up", "build-up", "buildup", "possession chain",
    "longest possession", "passing",
)

_TEAM_KEYWORDS = {"team a": "teamA", "team b": "teamB"}


def _find_event_type(q: str) -> str | None:
    return next((v for k, v in _EVENT_KEYWORDS.items() if k in q), None)


def classify(question: str) -> tuple[QuestionType, str | None, str | None]:
    """question -> (question type, team or None, event_type or None)."""
    q = question.lower()
    team = next((v for k, v in _TEAM_KEYWORDS.items() if k in q), None)

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
