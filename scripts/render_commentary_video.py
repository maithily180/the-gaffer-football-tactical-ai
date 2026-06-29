"""
scripts/render_commentary_video.py
─────────────────────────────────────
Gaffer v3.2 — render a match as one mp4 with the bird's-eye minimap, timed
commentary subtitles, and a spoken narration track.

Needs a perception pass (the minimap needs per-frame positions), so a full
clip takes a few minutes; use --duration to render a short slice first.

Usage:
    uv run python scripts/render_commentary_video.py data/test_clips/arsenal_newcastle_highlights.mp4
    uv run python scripts/render_commentary_video.py clip.mp4 --duration 20
    uv run python scripts/render_commentary_video.py clip.mp4 --no-llm --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gaffer import config
from gaffer.analyst.commentary_video import build_commentary_video


def main() -> None:
    ap = argparse.ArgumentParser(description="Gaffer v3.2 — commentary video (minimap + subtitles + narration)")
    ap.add_argument("clip")
    ap.add_argument("--calib", default=None, help="Calibration JSON (auto-detected from clip stem if omitted)")
    ap.add_argument("--duration", type=float, default=None, help="render only the first N seconds (for testing)")
    ap.add_argument("--no-llm", action="store_true", help="deterministic commentary only (no Ollama)")
    ap.add_argument("--all", action="store_true", help="narrate every possession, not just notable ones")
    ap.add_argument("--min-importance", type=float, default=0.0,
                     help="skip episodes below this importance_score() (see commentary.py)")
    ap.add_argument("--style", choices=["broadcast", "tactical", "casual"], default="broadcast",
                     help="commentary register")
    args = ap.parse_args()

    clip_path = Path(args.clip)
    calib_path = Path(args.calib) if args.calib else config.DATA_DIR / "calibration" / f"{clip_path.stem}.json"

    out = build_commentary_video(
        clip_path, calib_path,
        use_llm=not args.no_llm,
        notable_only=not args.all,
        min_importance=args.min_importance,
        style=args.style,
        duration=args.duration,
    )
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
