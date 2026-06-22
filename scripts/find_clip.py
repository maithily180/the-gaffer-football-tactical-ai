"""
scripts/find_clip.py
─────────────────────────
Gaffer v2.2 — find and export a clip for a specific event/episode reference.

Builds (or loads the cached) MatchBundle for a clip, resolves the query to
a (start_s, end_s) window via gaffer.analyst.clip_finder.find_clip_reference(),
then cuts that window out of the source video.

Usage:
    uv run python scripts/find_clip.py data/test_clips/tactical_playlist_1.mp4 "the counter attack"
    uv run python scripts/find_clip.py clip.mp4 "the first line break" --pad 3
    uv run python scripts/find_clip.py clip.mp4 "overload #4"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gaffer import config
from gaffer.analyst.clip_finder import export_clip, find_clip_reference
from gaffer.analyst.match_bundle import build_bundle


def main() -> None:
    ap = argparse.ArgumentParser(description="Gaffer v2.2 — find and export a clip for an event/episode reference")
    ap.add_argument("clip")
    ap.add_argument("query")
    ap.add_argument("--calib", default=None,
                     help="Calibration JSON (auto-detected from clip stem if omitted)")
    ap.add_argument("--pad", type=float, default=2.0, help="seconds of padding before/after the window")
    ap.add_argument("--force", action="store_true", help="bypass the bundle cache, re-run detection")
    args = ap.parse_args()

    clip_path = Path(args.clip)
    calib_path = Path(args.calib) if args.calib else config.DATA_DIR / "calibration" / f"{clip_path.stem}.json"

    bundle = build_bundle(clip_path, calib_path, force=args.force)
    ref = find_clip_reference(bundle, args.query)
    if ref is None:
        print(f"Couldn't find a match for {args.query!r} in this match.")
        return

    print(f"Found: {ref.label}  [{ref.start_s:.1f}s - {ref.end_s:.1f}s]")
    out_path = export_clip(clip_path, ref.start_s, ref.end_s, pad_s=args.pad)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
