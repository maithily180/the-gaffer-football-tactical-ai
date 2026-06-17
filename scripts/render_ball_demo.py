"""
scripts/render_ball_demo.py
────────────────────────────
Ball tracking demo — annotated video with the complete v1.0 tracking stack.

Full pipeline per frame:
  detect → team-assign → track players → propagate H → filter ball
  → track ball → state estimate → analytics snap → world model update

Visual legend
─────────────
  Filled yellow circle    = live YOLO detection (confidence %)
  Hollow orange circle    = BallTracker extrapolation / prediction (PRED)
  Fading grey/white trail = recent ball positions (brighter = more recent)
  Cyan diamond            = world-model possession anchor (ball carrier)
  White cross             = world-model trajectory extrapolation target
  Event badge (bottom)    = active world-model tactical event context

HUD (top-left)
──────────────
  Ball status / timestamp
  Ball state machine state + approach voters
  World model: WM score, active signals
  Filter rejection breakdown

Usage:
    uv run python scripts/render_ball_demo.py data/test_clips/tactical_playlist_1.mp4
    uv run python scripts/render_ball_demo.py clip.mp4 --start 65 --duration 25
    uv run python scripts/render_ball_demo.py clip.mp4 --calib data/calibration/clip.json
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
from gaffer.analytics.engine import PitchAnalyticsEngine
from gaffer.calibration.homography_manager import HomographyManager
from gaffer.calibration.homography_propagator import HomographyPropagator
from gaffer.detection.detector import Detection, FootballDetector
from gaffer.detection.team_assigner import TeamAssigner
from gaffer.tracking.ball_candidate_filter import BallCandidateFilter
from gaffer.tracking.ball_state_estimator import BallStateEstimator
from gaffer.tracking.ball_tracker import BallTracker
from gaffer.tracking.tracker import PlayerTracker
from gaffer.tracking.world_model import BallWorldModel, _pitch_to_image
from gaffer.video.loader import VideoLoader
from gaffer.video.writer import VideoWriter

# ── Visual constants ──────────────────────────────────────────────────────────
TRAIL_LENGTH  = 45
BALL_LIVE_CLR = (0, 255, 255)     # yellow
BALL_PRED_CLR = (0, 180, 255)     # orange
HUD_CLR       = (0, 220, 60)      # green
ANCHOR_CLR    = (255, 220, 0)     # cyan
EXPECT_CLR    = (255, 255, 255)   # white
EVENT_CLR     = (0, 220, 220)     # yellow-ish for event badge


@dataclass
class TrailPoint:
    cx: int
    cy: int
    is_live: bool


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_ball(frame: np.ndarray, det: Detection, is_live: bool,
               wm_score: float | None = None) -> None:
    cx, cy = det.center
    clr = BALL_LIVE_CLR if is_live else BALL_PRED_CLR
    if is_live:
        cv2.circle(frame, (cx, cy), 9,  clr, -1)
        cv2.circle(frame, (cx, cy), 9,  (255, 255, 255), 2)
        cv2.circle(frame, (cx, cy), 15, clr, 1)
        label = f"{det.confidence:.2f}"
        if wm_score is not None:
            label += f"  wm:{wm_score:.2f}"
        cv2.putText(frame, label, (cx + 14, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, clr, 1, cv2.LINE_AA)
    else:
        cv2.circle(frame, (cx, cy), 9,  clr, 2)
        cv2.circle(frame, (cx, cy), 15, clr, 1)
        cv2.putText(frame, "PRED", (cx + 14, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, clr, 1, cv2.LINE_AA)


def _draw_trail(frame: np.ndarray, trail: deque) -> None:
    pts = list(trail)
    n   = len(pts)
    for i in range(1, n):
        alpha = i / n
        intensity = int(60 + 150 * alpha)
        cv2.line(frame,
                 (pts[i-1].cx, pts[i-1].cy),
                 (pts[i].cx,   pts[i].cy),
                 (intensity, intensity, intensity), 1, cv2.LINE_AA)
    for i, tp in enumerate(pts):
        alpha = (i + 1) / n
        r     = max(2, int(5 * alpha))
        intensity = int(80 + 140 * alpha)
        cv2.circle(frame, (tp.cx, tp.cy), r,
                   (intensity, intensity, intensity), -1)


def _draw_wm_markers(frame: np.ndarray, wm: BallWorldModel, mgr: HomographyManager) -> None:
    """Draw possession anchor (cyan diamond) and trajectory target (white cross)."""
    ctx = wm.context

    # Possession anchor — cyan diamond
    if ctx.possession_anchor_px is not None:
        ax, ay = int(ctx.possession_anchor_px[0]), int(ctx.possession_anchor_px[1])
        H, W = frame.shape[:2]
        if 0 <= ax < W and 0 <= ay < H:
            size = 8
            pts  = np.array([[ax, ay-size], [ax+size, ay],
                              [ax, ay+size], [ax-size, ay]], np.int32)
            cv2.polylines(frame, [pts], True, ANCHOR_CLR, 2, cv2.LINE_AA)
            cv2.putText(frame, "carrier", (ax + 10, ay - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, ANCHOR_CLR, 1, cv2.LINE_AA)

    # Trajectory target — white cross
    if ctx.expected_ball_m is not None and mgr is not None and mgr.is_valid():
        px = _pitch_to_image(ctx.expected_ball_m, mgr)
        if px is not None:
            ex, ey = int(px[0]), int(px[1])
            H, W = frame.shape[:2]
            if 0 <= ex < W and 0 <= ey < H:
                s = 6
                cv2.line(frame, (ex-s, ey), (ex+s, ey), EXPECT_CLR, 1, cv2.LINE_AA)
                cv2.line(frame, (ex, ey-s), (ex, ey+s), EXPECT_CLR, 1, cv2.LINE_AA)


def _draw_event_badge(frame: np.ndarray, wm: BallWorldModel) -> None:
    """Show active world-model event context as a badge at the bottom-centre."""
    ctx = wm.context
    parts = []
    if ctx.in_counter:
        parts.append("COUNTER")
    if ctx.in_high_press:
        parts.append("PRESS")
    if ctx.line_break_x is not None:
        parts.append("LINE BRK")
    if not parts:
        return

    badge = "  WM: " + " | ".join(parts) + "  "
    H, W  = frame.shape[:2]
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    (tw, th), _ = cv2.getTextSize(badge, font, scale, thick)
    bx = (W - tw) // 2
    by = H - 30
    overlay = frame.copy()
    cv2.rectangle(overlay, (bx - 6, by - th - 6), (bx + tw + 6, by + 6), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, badge, (bx, by), font, scale, EVENT_CLR, thick, cv2.LINE_AA)


def _draw_hud(frame, frame_idx, fps, n_det, n_ext, n_lost, ball_result,
              is_live, rej: dict, ball_state: str = "UNKNOWN",
              approach_voters: int = 0, wm: BallWorldModel | None = None,
              wm_score: float | None = None) -> None:
    total = n_det + n_ext + n_lost
    ts    = frame_idx / fps
    status = ("LIVE" if (ball_result and is_live)
               else "PRED" if ball_result
               else "LOST")

    lines: list[tuple[str, tuple]] = []
    def t(s, clr=None): lines.append((s, clr or (230, 230, 230)))
    def h(s): lines.append((s, HUD_CLR))
    def sep(): lines.append(("", (0,0,0)))

    h(f"t={ts:.1f}s  Ball: {status}")
    t(f"State: {ball_state}  Voters: {approach_voters}")
    sep()
    t(f"Detected  : {n_det:4d} ({100*n_det/max(total,1):.0f}%)")
    t(f"Predicted : {n_ext:4d} ({100*n_ext/max(total,1):.0f}%)")
    t(f"Lost      : {n_lost:4d} ({100*n_lost/max(total,1):.0f}%)")
    sep()
    t("-- World Model --", (160, 160, 160))
    if wm is not None:
        ctx = wm.context
        anchor_str = (f"({ctx.possession_anchor_m[0]:.0f},{ctx.possession_anchor_m[1]:.0f})m"
                      if ctx.possession_anchor_m else "none")
        t(f"Anchor: {anchor_str}")
        score_str = f"{wm_score:.2f}" if wm_score is not None else "--"
        t(f"WM score: {score_str}")
        evts = []
        if ctx.in_counter:    evts.append("COUNTER")
        if ctx.in_high_press: evts.append("PRESS")
        if ctx.line_break_x:  evts.append("LINEBK")
        t(f"Events: {' '.join(evts) or 'none'}")
    sep()
    t("-- Filters --", (160, 160, 160))
    t(f"Spatial   : {rej.get('spatial',0)} killed")
    t(f"Stationary: {rej.get('stationary',0)} killed")
    t(f"Off-pitch : {rej.get('off_pitch',0)} killed")
    t(f"Suspect   : {rej.get('suspect_discards',0)} dropped")
    t(f"Recovery  : {rej.get('recovery_accept',0)} reacquired")

    pad, line_h = 8, 19
    box_h = pad * 2 + line_h * len(lines) + 4
    box_w = 260
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + box_w, 8 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, (line, clr) in enumerate(lines):
        if not line:
            continue
        cv2.putText(frame, line,
                    (8 + pad, 8 + pad + (i + 1) * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, clr, 1, cv2.LINE_AA)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Gaffer v1.0 — ball tracking demo")
    ap.add_argument("clip")
    ap.add_argument("--start",    type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--calib",    default=None)
    ap.add_argument("--out",      default=None)
    ap.add_argument("--no-propagate", action="store_true",
                    help="Disable optical-flow H propagation (static H)")
    args = ap.parse_args()

    clip_path = Path(args.clip)
    out_path  = Path(args.out) if args.out else \
                config.OUTPUTS_DIR / "ball_tracking_demo.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Calibration — enables on-pitch gate, propagation, world model
    mgr = None
    calib_path = Path(args.calib) if args.calib else \
                 config.DATA_DIR / "calibration" / f"{clip_path.stem}.json"
    if calib_path.exists():
        mgr = HomographyManager.from_calibration(calib_path)
        print(f"Calibration : {calib_path}  (on-pitch gate + world model ACTIVE)")
    else:
        print(f"Calibration : not found  (on-pitch gate + world model DISABLED)")

    loader = VideoLoader(str(clip_path))
    start_frame = int(args.start * loader.fps)
    n_frames    = min(int(args.duration * loader.fps),
                      loader.total_frames - start_frame)

    detector       = FootballDetector(verbose=False)
    player_tracker = PlayerTracker(fps=loader.fps)
    ball_tracker   = BallTracker()
    ball_filter    = BallCandidateFilter()
    state_estimator = BallStateEstimator()

    # World model + dependencies (only if calibrated)
    propagator  = None
    engine      = None
    world_model = None
    assigner    = None

    if mgr is not None and not args.no_propagate:
        propagator = HomographyPropagator(mgr)

    if mgr is not None:
        assigner = TeamAssigner()
        print("Fitting TeamAssigner on sample frames …")
        fit_frames = loader.sample_frames(12, start=start_frame, count=n_frames)
        fit_dets   = [detector.detect_raw(f) for f in fit_frames]
        assigner.fit(fit_frames, fit_dets)

        engine      = PitchAnalyticsEngine(mgr, fps=loader.fps)
        world_model = BallWorldModel(fps=loader.fps)

    print(f"Clip    : {clip_path.name}")
    print(f"Window  : {args.start:.0f}s – {args.start + args.duration:.0f}s  ({n_frames} frames)")
    print(f"Output  : {out_path}")
    print(f"H mode  : {'PROPAGATED' if propagator else 'STATIC' if mgr else 'NONE'}")
    print(f"WM      : {'ON' if world_model else 'OFF (no calibration)'}")
    print("Processing …")

    trail: deque[TrailPoint] = deque(maxlen=TRAIL_LENGTH)
    n_detected = n_extrap = n_lost = 0
    timings: list[float] = []

    with VideoWriter(out_path, fps=loader.fps,
                     width=loader.width, height=loader.height) as writer:

        for frame_idx, frame in loader.frames(start=start_frame, count=n_frames):
            t0 = time.perf_counter()

            # Advance H to current camera pose before projecting anything
            dets = detector.detect(frame, frame_idx)
            if propagator is not None:
                propagator.update(frame, exclude_dets=dets)

            ball_dets  = [d for d in dets if d.class_name == "ball"]
            field_dets = [d for d in dets if d.class_name != "ball"]

            is_detect_frame = (frame_idx == detector._last_detect_idx)
            wm_score: float | None = None

            if is_detect_frame:
                if assigner is not None:
                    field_dets = assigner.assign(frame, field_dets)
                field_dets = player_tracker.update(field_dets, frame)

                ball_dets = ball_filter.filter(
                    ball_dets, field_dets, frame_idx,
                    homography_manager=mgr,
                    last_ball_pos=ball_tracker.last_position_px(),
                    last_ball_vel=ball_tracker.last_velocity_vector(),
                    last_detection_frame=ball_tracker.last_detection_frame(),
                    state_estimator=state_estimator,
                    world_model=world_model,
                )
                ball_result = ball_tracker.update(ball_dets, frame_idx)

                state_det = (ball_result
                             if ball_result is not None and ball_result.confidence > 0
                             else None)
                state_estimator.update(state_det, ball_tracker.last_velocity_vector(),
                                       field_dets, frame_idx)

                # Analytics + world model update
                if engine is not None:
                    all_dets = field_dets + ([ball_result] if ball_result else [])
                    snap = engine.update(frame_idx, all_dets)
                    if snap is not None and world_model is not None and mgr is not None:
                        world_model.update(snap, mgr)

                # WM score of the accepted detection (for HUD display)
                if world_model is not None and ball_result is not None and mgr is not None:
                    if ball_result.confidence > 0:
                        wm_score = world_model.score_candidate_px(ball_result.center, mgr)

                is_live = bool(ball_dets)
            else:
                field_dets  = player_tracker.carry_forward()
                ball_result = ball_tracker.update([], frame_idx)
                is_live     = False

            # Track counters + trail
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

            if world_model is not None and mgr is not None:
                _draw_wm_markers(out, world_model, mgr)

            if ball_result is not None:
                _draw_ball(out, ball_result, is_live, wm_score)

            if world_model is not None:
                _draw_event_badge(out, world_model)

            rej    = ball_filter.rejection_summary()
            voters = (state_estimator.approach_voters(ball_result.center, field_dets)
                      if ball_result and field_dets else 0)
            _draw_hud(out, frame_idx, loader.fps,
                      n_detected, n_extrap, n_lost, ball_result, is_live, rej,
                      ball_state=state_estimator.state.value,
                      approach_voters=voters,
                      wm=world_model, wm_score=wm_score)

            writer.write(out)
            timings.append(time.perf_counter() - t0)

            done = n_detected + n_extrap + n_lost
            if done % 150 == 0 and done > 0:
                pct = 100 * done / n_frames
                fps_p = 1.0 / max(float(np.mean(timings[-50:])), 1e-6)
                print(f"  {pct:5.1f}%  frame {frame_idx}  {fps_p:.1f} fps")

    loader.close()

    total   = n_detected + n_extrap + n_lost
    rej     = ball_filter.rejection_summary()
    mean_ms = float(np.mean(timings)) * 1000
    size_mb = out_path.stat().st_size / 1024 ** 2

    print()
    print("=" * 54)
    print("  Ball Tracking Demo — Results")
    print("=" * 54)
    print(f"  Frames processed   : {total}")
    print(f"  Processing speed   : {mean_ms:.1f} ms/frame  ({1000/mean_ms:.1f} fps)")
    print()
    print(f"  Live detections    : {n_detected:4d} ({100*n_detected/max(total,1):.1f}%)")
    print(f"  Extrapolated       : {n_extrap:4d} ({100*n_extrap/max(total,1):.1f}%)")
    print(f"  Ball lost          : {n_lost:4d} ({100*n_lost/max(total,1):.1f}%)")
    print(f"  Ball visible total : {n_detected+n_extrap:4d} "
          f"({100*(n_detected+n_extrap)/max(total,1):.1f}%)")
    print()
    print("  -- Filter rejections --")
    print(f"  Spatial gate       : {rej['spatial']:4d} killed")
    print(f"  Stationary gate    : {rej['stationary']:4d} killed")
    print(f"  Off-pitch gate     : {rej['off_pitch']:4d} killed")
    print(f"  Suspect discards   : {rej['suspect_discards']:4d} dropped")
    print(f"  Recovery accept    : {rej['recovery_accept']:4d} reacquired")
    print(f"  Scene cuts         : {rej['scene_cuts']:4d} detected")
    if propagator is not None:
        s = propagator.summary()
        print()
        print(f"  H propagation      : {s['updates']} updates, "
              f"{s['holds']} holds, {s['cuts']} cuts")
    print()
    print(f"  Output : {out_path}  ({size_mb:.1f} MB)")
    print("=" * 54)


if __name__ == "__main__":
    main()
