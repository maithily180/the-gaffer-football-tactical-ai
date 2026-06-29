"""
scripts/commentate.py
─────────────────────────────────────
Gaffer v3.1 — analyst commentary over a match's tactical episodes.

Builds (or loads the cached) MatchBundle for a clip, then narrates each
episode in chronological order using gaffer.analyst.commentary -- grounded
in the events Gaffer actually detected, never invented play-by-play.

Usage:
    uv run python scripts/commentate.py data/test_clips/tactical_playlist_1.mp4
    uv run python scripts/commentate.py clip.mp4 --notable-only
    uv run python scripts/commentate.py clip.mp4 --no-llm     # deterministic only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gaffer import config
from gaffer.analyst.commentary import commentate_match
from gaffer.analyst.match_bundle import build_bundle


def main() -> None:
    ap = argparse.ArgumentParser(description="Gaffer v3.1 — analyst commentary over tactical episodes")
    ap.add_argument("clip")
    ap.add_argument("--calib", default=None,
                     help="Calibration JSON (auto-detected from clip stem if omitted)")
    ap.add_argument("--notable-only", action="store_true",
                     help="skip quiet possessions with no notable events")
    ap.add_argument("--no-llm", action="store_true",
                     help="deterministic commentary only (no Ollama call)")
    ap.add_argument("--force", action="store_true", help="bypass the bundle cache, re-run detection")
    ap.add_argument("--min-importance", type=float, default=0.0,
                     help="skip episodes below this importance_score() (see commentary.py)")
    ap.add_argument("--style", choices=["broadcast", "tactical", "casual"], default="broadcast",
                     help="commentary register")
    ap.add_argument("--no-memory", action="store_true",
                     help="disable narrative-memory continuity context between episodes")
    args = ap.parse_args()

    clip_path = Path(args.clip)
    calib_path = Path(args.calib) if args.calib else config.DATA_DIR / "calibration" / f"{clip_path.stem}.json"

    bundle = build_bundle(clip_path, calib_path, force=args.force)
    lines = commentate_match(bundle, use_llm=not args.no_llm, notable_only=args.notable_only,
                             min_importance=args.min_importance, style=args.style,
                             use_narrative_memory=not args.no_memory)
    if not lines:
        print("No episodes to commentate.")
        return

    for ep, text in lines:
        tlabel = "TEAM A" if ep.team == "teamA" else "TEAM B"
        print(f"\n[{ep.start_time_s:.1f}s-{ep.end_time_s:.1f}s] {tlabel} ({ep.outcome})")
        print(f"  {text}")


if __name__ == "__main__":
    main()
