"""
scripts/render_highlight_reel.py
─────────────────────────────────────
Gaffer v2.3 — compile a match report's top episodes into one captioned
highlight reel.

Builds (or loads the cached) MatchBundle for a clip, then hands
bundle.match_report.top_episodes to gaffer.analyst.highlight_reel to cut
and caption each one into a single output video.

Usage:
    uv run python scripts/render_highlight_reel.py data/test_clips/tactical_playlist_1.mp4
    uv run python scripts/render_highlight_reel.py clip.mp4 --pad 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gaffer import config
from gaffer.analyst.highlight_reel import render_highlight_reel
from gaffer.analyst.match_bundle import build_bundle


def main() -> None:
    ap = argparse.ArgumentParser(description="Gaffer v2.3 — compile a match report's top episodes into a highlight reel")
    ap.add_argument("clip")
    ap.add_argument("--calib", default=None,
                     help="Calibration JSON (auto-detected from clip stem if omitted)")
    ap.add_argument("--pad", type=float, default=2.0, help="seconds of padding before/after each episode")
    ap.add_argument("--force", action="store_true", help="bypass the bundle cache, re-run detection")
    args = ap.parse_args()

    clip_path = Path(args.clip)
    calib_path = Path(args.calib) if args.calib else config.DATA_DIR / "calibration" / f"{clip_path.stem}.json"

    bundle = build_bundle(clip_path, calib_path, force=args.force)
    top_episodes = bundle.match_report.top_episodes
    if not top_episodes:
        print("No notable episodes to compile -- this match report has no top episodes.")
        return

    for i, ep in enumerate(top_episodes, start=1):
        tlabel = "TEAM A" if ep.team == "teamA" else "TEAM B"
        print(f"  #{i} [{ep.start_time_s:.1f}s-{ep.end_time_s:.1f}s] {tlabel}: {ep.narrative()} ({ep.outcome})")

    out_path = render_highlight_reel(bundle, clip_path, pad_s=args.pad)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
