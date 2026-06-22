"""
gaffer/analyst/clip_finder.py
─────────────────────────────────
Layers 2+3 of Temporal Evidence Retrieval: resolve a reference like "the
counter attack", "overload #4", "the longest episode", or "the first line
break" to a concrete (start_s, end_s) window, then cut that window out of
the source video. Deliberately not a QuestionType/LLM round trip like
Layer 1 -- "which timestamp is this" is a deterministic lookup over data
that already exists (FootballEvent.time_s, Episode.start_time_s/end_time_s),
not something an LLM call would improve.

Reuses question_types.py's existing keyword tables (_find_event_type,
_TEAM_KEYWORDS) rather than duplicating them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from gaffer import config
from gaffer.analyst.match_bundle import MatchBundle
from gaffer.analyst.question_types import _TEAM_KEYWORDS, _find_event_type
from gaffer.events.base import FootballEvent
from gaffer.video.loader import VideoLoader
from gaffer.video.writer import VideoWriter

_ORDINAL_WORDS: dict[str, int] = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "last": -1,
}
_ORDINAL_NUM_RE = re.compile(r"#(\d+)|\bnumber\s+(\d+)\b")


@dataclass
class ClipReference:
    start_s:    float
    end_s:      float
    label:      str                  # e.g. "LINE BREAK A @ 6.0s"
    team:       str | None
    event_type: str | None


def _find_ordinal(q: str) -> int | None:
    for word, n in _ORDINAL_WORDS.items():
        if word in q:
            return n
    m = _ORDINAL_NUM_RE.search(q)
    if m:
        return int(m.group(1) or m.group(2))
    return None


def _enclosing_episode(bundle: MatchBundle, event: FootballEvent):
    return next(
        (ep for ep in bundle.run.episodes
         if ep.team == event.team and ep.start_time_s <= event.time_s <= ep.end_time_s),
        None,
    )


def _reference_for_event(bundle: MatchBundle, event: FootballEvent, suffix: str) -> ClipReference:
    enclosing = _enclosing_episode(bundle, event)
    if enclosing is not None:
        start_s, end_s = enclosing.start_time_s, enclosing.end_time_s
    else:
        start_s = end_s = event.time_s
    return ClipReference(
        start_s=start_s, end_s=end_s,
        label=f"{event.label()} @ {event.time_s:.1f}s{suffix}",
        team=event.team, event_type=event.event_type,
    )


def _find_longest_episode(bundle: MatchBundle, team: str | None, event_type: str | None) -> ClipReference | None:
    episodes = [
        ep for ep in bundle.run.episodes
        if (team is None or ep.team == team)
        and (event_type is None or any(e.event_type == event_type for e in ep.events))
    ]
    if not episodes:
        return None
    chosen = max(episodes, key=lambda ep: ep.duration_s)
    label = f"{chosen.team}: {chosen.narrative()} ({chosen.outcome}, {chosen.duration_s:.1f}s)"
    return ClipReference(start_s=chosen.start_time_s, end_s=chosen.end_time_s, label=label,
                          team=chosen.team, event_type=event_type)


def find_clip_reference(bundle: MatchBundle, query: str) -> ClipReference | None:
    """Resolve a free-text reference to a single ClipReference, or None if
    nothing in the match matches the query."""
    q = query.lower()
    team = next((v for k, v in _TEAM_KEYWORDS.items() if k in q), None)
    event_type = _find_event_type(q)

    if "longest" in q:
        return _find_longest_episode(bundle, team, event_type)

    if event_type is None:
        return None

    matches = sorted(
        (e for e in bundle.run.events if e.event_type == event_type and (team is None or e.team == team)),
        key=lambda e: e.time_s,
    )
    if not matches:
        return None

    ordinal = _find_ordinal(q)
    if ordinal is not None:
        idx = ordinal - 1 if ordinal > 0 else ordinal
        if idx >= len(matches) or idx < -len(matches):
            return None
        return _reference_for_event(bundle, matches[idx], "")

    suffix = f" (1 of {len(matches)})" if len(matches) > 1 else ""
    return _reference_for_event(bundle, matches[0], suffix)


def export_clip(clip_path: str | Path, start_s: float, end_s: float, *,
                 pad_s: float = 2.0,
                 out_dir: Path = config.OUTPUTS_DIR / "clips") -> Path:
    """Cut [start_s - pad_s, end_s + pad_s] (clamped to the clip's bounds)
    out of clip_path and write it to out_dir. Reuses the same VideoLoader/
    VideoWriter pattern every other render script in gaffer/video uses."""
    with VideoLoader(clip_path) as loader:
        s = max(0.0, start_s - pad_s)
        e = min(loader.duration_s, end_s + pad_s)
        start_frame = int(s * loader.fps)
        n_frames = max(1, int(round((e - s) * loader.fps)))
        out_path = out_dir / f"{Path(clip_path).stem}_{s:.1f}-{e:.1f}.mp4"
        with VideoWriter(out_path, fps=loader.fps, width=loader.width, height=loader.height) as writer:
            for _, frame in loader.frames(start=start_frame, count=n_frames):
                writer.write(frame)
    return out_path
