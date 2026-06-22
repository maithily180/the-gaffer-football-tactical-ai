"""
gaffer/analyst/time_parse.py
───────────────────────────────
Shared by question_types.py (to recognize a time-anchored question) and
retrieval.py (to re-derive the same window for filtering). Plain regex --
events and episodes already carry absolute time_s / start_time_s / end_time_s
(gaffer/events/base.py, gaffer/analytics/episodes.py), so resolving "around
94s" to a window is deterministic lookup, not inference.

Only explicit absolute-time phrasing is matched ("94s", "1:34", "between 90
and 100 seconds"). Relative phrasing ("the last 10 seconds", "first half") is
deliberately NOT handled here -- "first half"/"second half" already decline
via question_types._UNSUPPORTED_PATTERNS, and "last N seconds" would need to
know total match duration, which this module doesn't have access to.
"""

from __future__ import annotations

import re

_RANGE_RE = re.compile(r"between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)")
_MMSS_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_POINT_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:s\b|sec\b|secs\b|seconds?\b)")

_POINT_PAD_S = 5.0


def find_time_window(question: str) -> tuple[float, float] | None:
    """question -> (start_s, end_s), or None if no explicit time is mentioned.

    Explicit ranges ("between 90 and 100 seconds") are returned as-is. A
    single point in time ("94s", "around 1:34", "near 75 seconds") is padded
    +/- _POINT_PAD_S so a nearby event isn't missed by a few tenths of a
    second of clock drift.
    """
    q = question.lower()

    m = _RANGE_RE.search(q)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return (min(a, b), max(a, b))

    m = _MMSS_RE.search(q)
    if m:
        t = int(m.group(1)) * 60 + int(m.group(2))
        return (max(0.0, t - _POINT_PAD_S), t + _POINT_PAD_S)

    m = _POINT_RE.search(q)
    if m:
        t = float(m.group(1))
        return (max(0.0, t - _POINT_PAD_S), t + _POINT_PAD_S)

    return None
