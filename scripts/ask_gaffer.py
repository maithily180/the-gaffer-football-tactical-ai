"""
scripts/ask_gaffer.py
─────────────────────────
Gaffer v2.0 — ask a natural-language question about a match.

Builds (or loads the cached) MatchBundle for a clip, then routes the
question through gaffer.analyst.query_engine.ask(): classify -> retrieve ->
build evidence -> LLM explanation.

Usage:
    uv run python scripts/ask_gaffer.py data/test_clips/tactical_playlist_1.mp4 "Summarize the match"
    uv run python scripts/ask_gaffer.py clip.mp4 "Which player was most influential?" --calib data/calibration/clip.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gaffer import config
from gaffer.analyst.match_bundle import build_bundle
from gaffer.analyst.query_engine import ask


def main() -> None:
    ap = argparse.ArgumentParser(description="Gaffer v2.0 — ask a question about a match")
    ap.add_argument("clip")
    ap.add_argument("question")
    ap.add_argument("--calib", default=None,
                     help="Calibration JSON (auto-detected from clip stem if omitted)")
    ap.add_argument("--force", action="store_true", help="bypass the bundle cache, re-run detection")
    args = ap.parse_args()

    clip_path = Path(args.clip)
    calib_path = Path(args.calib) if args.calib else config.DATA_DIR / "calibration" / f"{clip_path.stem}.json"

    bundle = build_bundle(clip_path, calib_path, force=args.force)
    print(ask(bundle, args.question))


if __name__ == "__main__":
    main()
