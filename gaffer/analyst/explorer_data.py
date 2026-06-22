"""
gaffer/analyst/explorer_data.py
─────────────────────────────────────
v3.0 — Match Explorer: pure-Python glue between a cached MatchBundle and
the gradio UI in app/gradio_app.py. No gradio imports here on purpose --
every function is independently testable and reusable, the same way
clip_finder.py/highlight_reel.py are plain functions reused by both
library code and CLI scripts.

Generalizes v2.3's report-to-clip bridge (top 3 episodes only) to every
episode in the match. No new detection, analytics, or LLM call -- every
fact and timestamp here already exists on the cached bundle.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from gaffer import config
from gaffer.analyst.match_bundle import CACHE_DIR, MatchBundle
from gaffer.analytics.episodes import (
    OUTCOME_ATTACKING_THIRD_ENTRY,
    OUTCOME_COUNTER,
    OUTCOME_LINE_BREAK,
    OUTCOME_LOST_POSSESSION,
    OUTCOME_PRESS_SUCCESS,
    OUTCOME_SUSTAINED_POSSESSION,
    Episode,
)


def list_available_clips() -> list[str]:
    """Clip stems with a cached MatchBundle AND a matching source file in
    data/test_clips/ -- only matches the app can load instantly, no
    11-minute rebuild trapped behind a button click."""
    stems = []
    for cache_file in sorted(CACHE_DIR.glob("*.json")):
        clip_path = config.DATA_DIR / "test_clips" / f"{cache_file.stem}.mp4"
        if clip_path.exists():
            stems.append(cache_file.stem)
    return stems


def episode_rows(bundle: MatchBundle) -> tuple[list[list], list[int]]:
    """Dataframe rows (id, team, start, end, duration, outcome, narrative),
    sorted by start_time_s, plus the parallel list of episode_ids in that
    same order -- lets a Gradio .select() row index map back to a real
    episode_id without parsing displayed text."""
    episodes = sorted(bundle.run.episodes, key=lambda ep: ep.start_time_s)
    rows = [
        [ep.episode_id, "Team A" if ep.team == "teamA" else "Team B",
         f"{ep.start_time_s:.1f}s", f"{ep.end_time_s:.1f}s", f"{ep.duration_s}s",
         ep.outcome, ep.narrative()]
        for ep in episodes
    ]
    return rows, [ep.episode_id for ep in episodes]


def find_episode(bundle: MatchBundle, episode_id: int) -> Episode | None:
    return next((ep for ep in bundle.run.episodes if ep.episode_id == episode_id), None)


def explain_outcome(ep: Episode) -> str:
    """One sentence grounding ep.outcome in what's actually on the episode.
    Branches on the already-recorded ep.outcome/ep.events rather than
    re-deriving EpisodeDetector._classify()'s exact decision -- some of its
    inputs (e.g. reached_attacking_third) are transient scratch state never
    persisted onto Episode, so this explains the recorded fact instead of
    replaying possibly-stale logic."""
    if ep.outcome == OUTCOME_COUNTER:
        return "Classified as Counter -- this possession contains a counter-attack event (rapid forward move right after winning the ball back)."
    if ep.outcome == OUTCOME_LINE_BREAK:
        return "Classified as Line Break -- the ball crossed the defending team's backline during this possession."
    if ep.outcome == OUTCOME_ATTACKING_THIRD_ENTRY:
        return "Classified as Attacking Third Entry -- the ball reached the attacking third at some point, with no counter or line break recorded."
    if ep.outcome == OUTCOME_PRESS_SUCCESS:
        return "Classified as Press Success -- the ball was won back during or just after a high press from this team."
    if ep.outcome == OUTCOME_SUSTAINED_POSSESSION:
        return f"Classified as Sustained Possession -- held the ball for {ep.duration_s}s with no breakthrough."
    if ep.outcome == OUTCOME_LOST_POSSESSION:
        return "Classified as Lost Possession -- none of the more specific patterns (counter, line break, attacking-third entry, press success, sustained possession) applied."
    return f"Outcome: {ep.outcome}."


def evidence_lines(ep: Episode) -> list[str]:
    """Same highlight-event filter Episode.narrative() already uses,
    unjoined -- the episode's evidence rendered as a flat list instead of
    an arrow-chained string."""
    return [f"{ev.label()} @ {ev.time_s:.1f}s" for ev in ep.events if ev.is_highlight]


def preceding_episode_summary(bundle: MatchBundle, episode_id: int) -> str:
    """The chronologically-previous episode (either team) by start_time_s,
    or an honest 'first episode in the match' if there isn't one."""
    episodes = sorted(bundle.run.episodes, key=lambda ep: ep.start_time_s)
    ids = [ep.episode_id for ep in episodes]
    idx = ids.index(episode_id)
    if idx == 0:
        return "This is the first episode in the match."
    prev = episodes[idx - 1]
    tlabel = "Team A" if prev.team == "teamA" else "Team B"
    return f"Episode #{prev.episode_id} ({tlabel}, {prev.start_time_s:.1f}s-{prev.end_time_s:.1f}s): {prev.narrative()} ({prev.outcome})"


def render_timeline_png(bundle: MatchBundle, out_path: Path) -> Path:
    """A single labeled horizontal bar image: one team-colored rectangle
    per episode, positioned/sized proportionally to start/end time over
    match duration. Plain cv2 drawing -- same library every other visual
    feature in this repo already uses. Static, not click-interactive; the
    Dataframe in the UI is the actual navigation control."""
    W, H, MARGIN = 1200, 140, 20
    duration_s = bundle.run.duration_s or 1.0
    img = np.full((H, W, 3), 30, dtype=np.uint8)

    track_y0, track_y1 = 50, 100
    cv2.rectangle(img, (MARGIN, track_y0), (W - MARGIN, track_y1), (60, 60, 60), -1)

    usable_w = W - 2 * MARGIN
    for ep in sorted(bundle.run.episodes, key=lambda e: e.start_time_s):
        x0 = MARGIN + int((ep.start_time_s / duration_s) * usable_w)
        x1 = MARGIN + int((ep.end_time_s / duration_s) * usable_w)
        x1 = max(x1, x0 + 2)
        clr = config.TEAM_A_COLOR_BGR if ep.team == "teamA" else config.TEAM_B_COLOR_BGR
        cv2.rectangle(img, (x0, track_y0), (x1, track_y1), clr, -1)

    cv2.putText(img, "0s", (MARGIN, track_y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    end_label = f"{duration_s:.0f}s"
    (tw, _), _ = cv2.getTextSize(end_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(img, end_label, (W - MARGIN - tw, track_y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(img, "Match Timeline (red = Team A, blue = Team B)", (MARGIN, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return out_path
