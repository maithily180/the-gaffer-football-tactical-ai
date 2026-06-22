"""
gaffer/analyst/highlight_reel.py
─────────────────────────────────────
v2.3 — Highlight Reels: compile a match report's already-selected top
episodes into one captioned video. The next step in the same "Temporal
Evidence Retrieval" lineage as clip_finder.py -- closes the gap named in
docs/V2_PRODUCT_REVIEW.md (report <-> episode <-> clip don't link to each
other) by reusing report.top_episodes exactly as the report already ranked
them, not a new "highlight-worthiness" score.

No new detection, no new analytics, no LLM call -- every fact and every
timestamp already exists on the cached MatchBundle.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from gaffer import config
from gaffer.analyst.clip_finder import _clamped_window
from gaffer.analyst.match_bundle import MatchBundle
from gaffer.analytics.episodes import Episode
from gaffer.video.loader import VideoLoader
from gaffer.video.writer import VideoWriter

_CAPTION_S = 5.0   # seconds each clip's caption stays visible
_FADE_S = 1.0      # tail of _CAPTION_S over which it fades out
_CAPTION_CLR = (230, 120, 230)   # magenta -- matches analytics_overlay.py's episode banner


def _caption_for(rank: int, ep: Episode) -> str:
    tlabel = "TEAM A" if ep.team == "teamA" else "TEAM B"
    return f"REPORT #{rank}  EPISODE #{ep.episode_id}  {tlabel}  {ep.narrative()}  -> {ep.outcome}"


def _draw_caption(frame: np.ndarray, text: str, t_into_clip: float) -> None:
    """Burn `text` into the top-center of `frame`, visible for _CAPTION_S
    seconds with a _FADE_S fade-out -- same visual language as
    analytics_overlay.py's _draw_episode_banner (top-center, translucent
    black box, magenta text), reimplemented here since that file's drawing
    logic is private instance methods on a stateful class, not a reusable
    free function."""
    if t_into_clip >= _CAPTION_S:
        return
    fade = min(1.0, (_CAPTION_S - t_into_clip) / _FADE_S)
    H, W = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    x, y = (W - tw) // 2, 40

    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 12, y - th - 10), (x + tw + 12, y + 8), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6 * fade, frame, 1 - 0.6 * fade, 0, frame)
    clr = tuple(int(c * fade) for c in _CAPTION_CLR)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, clr, 2, cv2.LINE_AA)


def render_highlight_reel(bundle: MatchBundle, clip_path: str | Path, *,
                           pad_s: float = 2.0,
                           out_path: Path | None = None) -> Path | None:
    """Compile bundle.match_report.top_episodes into one captioned video,
    in chronological playback order. Returns None (writes nothing) if there
    are no top episodes -- no silent/empty file."""
    episodes = sorted(bundle.match_report.top_episodes, key=lambda ep: ep.start_time_s)
    if not episodes:
        return None

    ranks = {ep.episode_id: i + 1 for i, ep in enumerate(bundle.match_report.top_episodes)}
    out_path = out_path or (config.OUTPUTS_DIR / "highlights" / f"{bundle.clip_name}_highlights.mp4")

    with VideoLoader(clip_path) as loader:
        with VideoWriter(out_path, fps=loader.fps, width=loader.width, height=loader.height) as writer:
            for ep in episodes:
                s, e, start_frame, n_frames = _clamped_window(loader, ep.start_time_s, ep.end_time_s, pad_s)
                caption = _caption_for(ranks[ep.episode_id], ep)
                for i, (_, frame) in enumerate(loader.frames(start=start_frame, count=n_frames)):
                    _draw_caption(frame, caption, t_into_clip=i / loader.fps)
                    writer.write(frame)
    return out_path
