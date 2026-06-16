"""
scripts/render_ball_demo.py
───────────────────────────
Ball tracking demo — annotated video showing live detections, extrapolated
predictions, fading trajectory trail, and per-filter rejection stats.

Visual legend
─────────────
  Filled yellow circle  = live YOLO detection (shows confidence %)
  Hollow orange circle  = BallTracker extrapolation / prediction (shows PRED)
  Fading grey trail     = recent ball positions (brighter = more recent)
  Green HUD (top-left)  = running stats + filter rejection counts

Filters applied before BallTracker:
  1. Stationary-object gate  — kills penalty spots / white lines
  2. Player-proximity gate   — kills lone crowd blobs (relaxed for in-air balls)
  3. On-pitch gate           — kills ad-boards (only if calibration JSON found)

Usage:
    uv run python scripts/render_ball_demo.py data/test_clips/arsenal_newcastle.mp4
    uv run python scripts/render_ball_demo.py data/test_clips/arsenal_newcastle.mp4 --start 0 --duration 30
    uv run python scripts/render_ball_demo.py data/test_clips/clip.mp4 --calib data/calibration/clip.json
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from gaffer import config
from gaffer.detection.detector import Detection, FootballDetector
from gaffer.tracking.ball_candidate_filter import BallCandidateFilter
from gaffer.tracking.ball_state_estimator import BallStateEstimator
from gaffer.tracking.ball_tracker import BallTracker
from gaffer.tracking.tracker import PlayerTracker
from gaffer.video.loader import VideoLoader
from gaffer.video.writer import VideoWriter

# ── Visual constants ──────────────────────────────────────────────────────────
TRAIL_LENGTH  = 40
BALL_LIVE_CLR = (0, 255, 255)    # yellow BGR
BALL_PRED_CLR = (0, 180, 255)    # orange-yellow BGR
HUD_CLR       = (0, 220, 60)     # green


@dataclass
class TrailPoint:
    cx: int
    cy: int
    is_live: bool


def _draw_ball(frame: np.ndarray, det: Detection, is_live: bool) -> None:
    cx, cy = det.center
    clr = BALL_LIVE_CLR if is_live else BALL_PRED_CLR
    if is_live:
        cv2.circle(frame, (cx, cy), 9,  clr, -1)
        cv2.circle(frame, (cx, cy), 9,  (255, 255, 255), 2)
        cv2.circle(frame, (cx, cy), 14, clr, 1)
        cv2.putText(frame, f"{det.confidence:.2f}", (cx + 12, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, clr, 1, cv2.LINE_AA)
    else:
        cv2.circle(frame, (cx, cy), 9,  clr, 2)
        cv2.circle(frame, (cx, cy), 14, clr, 1)
        cv2.putText(frame, "PRED", (cx + 12, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, clr, 1, cv2.LINE_AA)


def _draw_trail(frame: np.ndarray, trail: deque) -> None:
    pts = list(trail)
    n = len(pts)
    for i in range(1, n):
        alpha = i / n
        intensity = int(60 + 130 * alpha)
        cv2.line(frame,
                 (pts[i-1].cx, pts[i-1].cy),
                 (pts[i].cx,   pts[i].cy),
                 (intensity, intensity, intensity), 1, cv2.LINE_AA)
    for i, tp in enumerate(pts):
        alpha = (i + 1) / n
        r = max(2, int(5 * alpha))
        intensity = int(80 + 120 * alpha)
        cv2.circle(frame, (tp.cx, tp.cy), r, (intensity, intensity, intensity), -1)


def _draw_hud(frame, frame_idx, fps, n_det, n_ext, n_lost, ball_result,
              is_live, rej: dict, ball_state: str = "UNKNOWN",
              approach_voters: int = 0) -> None:
    total = n_det + n_ext + n_lost
    ts = frame_idx / fps
    status = ("LIVE" if (ball_result and is_live)
               else "PRED" if ball_result
               else "LOST")

    lines = [
        f"t = {ts:.1f}s       Ball: {status}",
        f"State: {ball_state}   Voters: {approach_voters}",
        "",
        f"Detected   : {n_det:4d} ({100*n_det/max(total,1):.0f}%)",
        f"Predicted  : {n_ext:4d} ({100*n_ext/max(total,1):.0f}%)",
        f"Lost       : {n_lost:4d} ({100*n_lost/max(total,1):.0f}%)",
        "",
        "-- Filters --",
        f"Spatial    : {rej.get('spatial',0)} killed",
        f"Stationary : {rej.get('stationary',0)} killed",
        f"Off-pitch  : {rej.get('off_pitch',0)} killed",
        f"Suspect    : {rej.get('suspect_discards',0)} discarded",
        f"Recovery   : {rej.get('recovery_accept',0)} reacquired",
        f"Cuts       : {rej.get('scene_cuts',0)} detected",
    ]

    pad, line_h = 8, 19
    box_h = pad * 2 + line_h * len(lines) + 4
    box_w = 245
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + box_w, 8 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, line in enumerate(lines):
        if not line:
            continue
        clr = (160, 160, 160) if line.startswith("--") else HUD_CLR
        cv2.putText(frame, line,
                    (8 + pad, 8 + pad + (i + 1) * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, clr, 1, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ball tracking demo with false-positive filters")
    ap.add_argument("clip")
    ap.add_argument("--start",    type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--calib",    default=None,
                    help="Calibration JSON for on-pitch gate (auto-detected if omitted)")
    ap.add_argument("--out",      default=None)
    args = ap.parse_args()

    clip_path = Path(args.clip)
    out_path  = Path(args.out) if args.out else \
                config.OUTPUTS_DIR / "ball_tracking_demo.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Auto-detect calibration JSON (same stem as clip)
    mgr = None
    calib_path = Path(args.calib) if args.calib else \
                 config.DATA_DIR / "calibration" / f"{clip_path.stem}.json"
    if calib_path.exists():
        from gaffer.calibration.homography_manager import HomographyManager
        mgr = HomographyManager.from_calibration(calib_path)
        print(f"Calibration : {calib_path}  (on-pitch gate ACTIVE)")
    else:
        print(f"Calibration : not found at {calib_path}  (on-pitch gate DISABLED)")

    loader          = VideoLoader(str(clip_path))
    detector        = FootballDetector(verbose=False)
    player_tracker  = PlayerTracker(fps=loader.fps)
    ball_tracker    = BallTracker()
    ball_filter     = BallCandidateFilter()
    state_estimator = BallStateEstimator()

    start_frame = int(args.start * loader.fps)
    n_frames    = min(int(args.duration * loader.fps),
                      loader.total_frames - start_frame)

    print(f"Clip    : {clip_path.name}")
    print(f"Window  : {args.start:.0f}s – {args.start + args.duration:.0f}s  ({n_frames} frames)")
    print(f"Output  : {out_path}")
    print("Processing …")

    trail: deque[TrailPoint] = deque(maxlen=TRAIL_LENGTH)
    n_detected = n_extrap = n_lost = 0
    timings: list[float] = []
    prev_frame: np.ndarray | None = None

    with VideoWriter(out_path, fps=loader.fps,
                     width=loader.width, height=loader.height) as writer:

        for frame_idx, frame in loader.frames(start=start_frame, count=n_frames):
            t0 = time.perf_counter()

            dets       = detector.detect(frame, frame_idx)
            ball_dets  = [d for d in dets if d.class_name == "ball"]
            field_dets = [d for d in dets if d.class_name != "ball"]

            is_detect_frame = (frame_idx == detector._last_detect_idx)

            if is_detect_frame:
                field_dets  = player_tracker.update(field_dets, frame)
                ball_dets = ball_filter.filter(
                    ball_dets, field_dets, frame_idx,
                    prev_frame=prev_frame,
                    curr_frame=frame,
                    homography_manager=mgr,
                    last_ball_pos=ball_tracker.last_position_px(),
                    last_ball_vel=ball_tracker.last_velocity_vector(),
                    last_detection_frame=ball_tracker.last_detection_frame(),
                    state_estimator=state_estimator,
                )
                ball_result = ball_tracker.update(ball_dets, frame_idx)
                # Extrapolated positions have confidence=0.0 — treat as no detection
                # so the suspect counter only increments on actual YOLO observations.
                state_det = (
                    ball_result
                    if ball_result is not None and ball_result.confidence > 0
                    else None
                )
                state_estimator.update(
                    state_det,
                    ball_tracker.last_velocity_vector(),
                    field_dets,
                    frame_idx,
                )
                is_live     = bool(ball_dets)
            else:
                field_dets  = player_tracker.carry_forward()
                ball_result = ball_tracker.update([], frame_idx)
                is_live     = False

            prev_frame = frame

            if ball_result is not None:
                n_detected += is_live
                n_extrap   += not is_live
                cx, cy = ball_result.center
                trail.append(TrailPoint(cx, cy, is_live))
            else:
                n_lost += 1

            # ── Draw ──────────────────────────────────────────────────────────
            out = frame.copy()
            _draw_trail(out, trail)
            if ball_result is not None:
                _draw_ball(out, ball_result, is_live)

            rej = ball_filter.rejection_summary()
            voters = (
                state_estimator.approach_voters(
                    ball_result.center, field_dets
                ) if ball_result and field_dets else 0
            )
            _draw_hud(out, frame_idx, loader.fps,
                      n_detected, n_extrap, n_lost, ball_result, is_live, rej,
                      ball_state=state_estimator.state.value,
                      approach_voters=voters)

            writer.write(out)
            timings.append(time.perf_counter() - t0)

            total_so_far = n_detected + n_extrap + n_lost
            if total_so_far % 150 == 0 and total_so_far > 0:
                pct = 100 * total_so_far / n_frames
                fps_p = 1.0 / max(float(np.mean(timings[-50:])), 1e-6)
                print(f"  {pct:5.1f}%  frame {frame_idx}  {fps_p:.1f} fps")

    loader.close()

    total = n_detected + n_extrap + n_lost
    rej   = ball_filter.rejection_summary()
    size_mb   = out_path.stat().st_size / 1024 ** 2
    mean_ms   = float(np.mean(timings)) * 1000

    print()
    print("=" * 52)
    print("  Ball Tracking Demo — Results")
    print("=" * 52)
    print(f"  Frames processed   : {total}")
    print(f"  Processing speed   : {mean_ms:.1f} ms/frame  ({1000/mean_ms:.1f} fps)")
    print()
    print(f"  Live detections    : {n_detected:4d} ({100*n_detected/max(total,1):.1f}%)")
    print(f"  Extrapolated       : {n_extrap:4d} ({100*n_extrap/max(total,1):.1f}%)")
    print(f"  Ball lost          : {n_lost:4d} ({100*n_lost/max(total,1):.1f}%)")
    print(f"  Ball visible total : {n_detected+n_extrap:4d} ({100*(n_detected+n_extrap)/max(total,1):.1f}%)")
    print()
    print("  -- Filter rejections --")
    print(f"  Spatial gate       : {rej['spatial']:4d} detections killed")
    print(f"  Stationary gate    : {rej['stationary']:4d} detections killed")
    print(f"  Proximity gate     : {rej['proximity']:4d} detections killed")
    print(f"  On-pitch gate      : {rej['off_pitch']:4d} detections killed")
    print(f"  Suspect discards   : {rej['suspect_discards']:4d} hypotheses dropped")
    print(f"  Recovery accept    : {rej['recovery_accept']:4d} reacquired")
    print(f"  Scene cuts detected: {rej['scene_cuts']:4d}")
    print()
    print(f"  Output : {out_path}  ({size_mb:.1f} MB)")
    print("=" * 52)


if __name__ == "__main__":
    main()
