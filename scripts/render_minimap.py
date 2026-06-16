"""
scripts/render_minimap.py
─────────────────────────
v0.5 Phase 6 + 7 — project tracked players onto the pitch using the calibrated
homography and render a minimap.

  static : detect+assign on the calibration frame, project foot points, save a
           side-by-side (source frame with foot dots | pitch minimap).
           → outputs/minimap_frame.png   (the trust check — do these dots land
             where the players actually are?)

  video  : composite the minimap into the corner of a short window around the
           calibration frame.  → outputs/minimap_demo.mp4
           NOTE: a single STATIC homography is only accurate while the camera
           matches the calibration frame; positions drift as it pans.

Usage:
    uv run python scripts/render_minimap.py data/test_clips/tactical_playlist_1.mp4 --mode static
    uv run python scripts/render_minimap.py data/test_clips/tactical_playlist_1.mp4 --mode video --window 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from gaffer import config
from gaffer.calibration.homography_manager import HomographyManager
from gaffer.detection.detector import FootballDetector
from gaffer.detection.team_assigner import TeamAssigner
from gaffer.output.minimap import MinimapRenderer
from gaffer.video.loader import VideoLoader
from gaffer.video.writer import VideoWriter

_TEAM_CLR = {0: config.TEAM_A_COLOR_BGR, 1: config.TEAM_B_COLOR_BGR}
_UNKNOWN = (160, 160, 160)


def _draw_foot_markers(frame, dets):
    """Draw the foot point that gets projected, so the source↔minimap mapping is visible."""
    out = frame.copy()
    for d in dets:
        if d.class_name == "ball":
            cv2.circle(out, d.center, 5, config.BALL_COLOR_BGR, -1)
        else:
            clr = _TEAM_CLR.get(d.team_id, _UNKNOWN)
            fx, fy = d.foot_point
            cv2.circle(out, (fx, fy), 5, clr, -1)
            cv2.circle(out, (fx, fy), 5, (255, 255, 255), 1)
    return out


def _fit_assigner(detector, loader, start_frame, n_frames, n=10):
    assigner = TeamAssigner()
    frames = loader.sample_frames(n, start=start_frame, count=n_frames)
    dets = [detector.detect_raw(f) for f in frames]
    assigner.fit(frames, dets)
    return assigner


def main():
    ap = argparse.ArgumentParser(description="Render pitch minimap from calibrated homography")
    ap.add_argument("clip")
    ap.add_argument("--mode", choices=["static", "video"], default="static")
    ap.add_argument("--calib", default=None, help="Calibration JSON (default data/calibration/<clip>.json)")
    ap.add_argument("--window", type=float, default=20.0, help="video mode: seconds around calib frame")
    args = ap.parse_args()

    calib = Path(args.calib) if args.calib else \
        config.DATA_DIR / "calibration" / f"{Path(args.clip).stem}.json"
    mgr = HomographyManager.from_calibration(calib)
    print(f"Loaded H from {calib}  (valid={mgr.is_valid()}, calib frame={mgr.calibration_frame})")

    loader = VideoLoader(args.clip)
    detector = FootballDetector(verbose=False)
    renderer = MinimapRenderer(mgr)
    calib_frame = mgr.calibration_frame or int(55 * loader.fps)
    label = f"STATIC CALIB f{calib_frame}"

    if args.mode == "static":
        # Fit team colours on frames around the calibration frame
        win = int(5 * loader.fps)
        assigner = _fit_assigner(detector, loader, max(0, calib_frame - win), 2 * win)

        loader.seek(calib_frame)
        ok, frame = loader.read()
        if not ok:
            raise RuntimeError(f"Could not read calibration frame {calib_frame}")
        dets = assigner.assign(frame, detector.detect_raw(frame))

        src = _draw_foot_markers(frame, dets)
        mini = renderer.render(dets, label=label)
        # side-by-side: scale minimap to source height
        mh = frame.shape[0]
        mw = int(mini.shape[1] * mh / mini.shape[0])
        mini_r = cv2.resize(mini, (mw, mh), interpolation=cv2.INTER_AREA)
        combo = np.hstack([src, mini_r])

        out = config.OUTPUTS_DIR / "minimap_frame.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), combo)
        n_proj = sum(1 for d in dets if d.class_name != "ball"
                     and (w := mgr.project(d.foot_point)) and mgr.on_pitch(*w))
        print(f"Saved {out}  ({len(dets)} detections, {n_proj} players projected on-pitch)")
        loader.close()
        return

    # ── video mode ────────────────────────────────────────────────────────────
    half = int(args.window * loader.fps / 2)
    start = max(0, calib_frame - half)
    n_frames = min(int(args.window * loader.fps), loader.total_frames - start)
    assigner = _fit_assigner(detector, loader, start, n_frames)

    out = config.OUTPUTS_DIR / "minimap_demo.mp4"
    print(f"Rendering {n_frames} frames ({start/loader.fps:.0f}-{(start+n_frames)/loader.fps:.0f}s) → {out}")
    with VideoWriter(out, fps=loader.fps, width=loader.width, height=loader.height) as w:
        for fidx, frame in loader.frames(start=start, count=n_frames):
            dets = assigner.assign(frame, detector.detect(frame, fidx))
            framed = _draw_foot_markers(frame, dets)
            composed = renderer.composite(framed, dets, label=label)
            w.write(composed)
    loader.close()
    print(f"Saved {out}  ({out.stat().st_size/1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
