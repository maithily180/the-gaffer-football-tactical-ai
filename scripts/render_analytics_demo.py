"""
scripts/render_analytics_demo.py
─────────────────────────────────
Football-intelligence demo (Gaffer v0.7).

Runs the full perception stack (detect → team-assign → track → ball-reason),
projects everyone to pitch metres via a calibration, then overlays geometric
football facts:

  Stats panel (top-left)
    Players / Width / Depth / Hull area / Defensive line / Space control %
    Possession % bar + pressing intensity

  Voronoi inset (top-right)
    2D pitch filled with each team's controlled space, players + ball dotted

Requires a calibration JSON (same stem as the clip, or pass --calib) — without
it there are no pitch coordinates and analytics cannot run.

Usage:
    uv run python scripts/render_analytics_demo.py data/test_clips/arsenal_newcastle_highlights.mp4
    uv run python scripts/render_analytics_demo.py clip.mp4 --start 0 --duration 30 --calib data/calibration/clip.json
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from gaffer import config
from gaffer.analytics.engine import PitchAnalyticsEngine
from gaffer.calibration.homography_manager import HomographyManager
from gaffer.calibration.homography_propagator import HomographyPropagator
from gaffer.detection.detector import FootballDetector
from gaffer.detection.team_assigner import TeamAssigner
from gaffer.output.analytics_overlay import AnalyticsOverlay
from gaffer.tracking.ball_candidate_filter import BallCandidateFilter
from gaffer.tracking.ball_state_estimator import BallStateEstimator
from gaffer.tracking.ball_tracker import BallTracker
from gaffer.tracking.tracker import PlayerTracker
from gaffer.tracking.world_model_v2 import WorldModelV2
from gaffer.video.loader import VideoLoader
from gaffer.video.writer import VideoWriter


def main() -> None:
    ap = argparse.ArgumentParser(description="Gaffer v0.7 — football analytics demo")
    ap.add_argument("clip")
    ap.add_argument("--start",    type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--calib",    default=None,
                    help="Calibration JSON (auto-detected from clip stem if omitted)")
    ap.add_argument("--out",      default=None)
    ap.add_argument("--no-propagate", action="store_true",
                    help="Disable optical-flow homography propagation (use static H)")
    args = ap.parse_args()

    clip_path = Path(args.clip)
    out_path  = Path(args.out) if args.out else \
                config.OUTPUTS_DIR / "analytics_demo.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    calib_path = Path(args.calib) if args.calib else \
                 config.DATA_DIR / "calibration" / f"{clip_path.stem}.json"
    if not calib_path.exists():
        print(f"ERROR: calibration not found at {calib_path}")
        print("Analytics need pitch coordinates — run scripts/collect_calibration.py first.")
        sys.exit(1)

    mgr = HomographyManager.from_calibration(calib_path)
    print(f"Calibration : {calib_path}")

    loader   = VideoLoader(str(clip_path))
    detector = FootballDetector(verbose=False)

    start_frame = int(args.start * loader.fps)
    n_frames    = min(int(args.duration * loader.fps),
                      loader.total_frames - start_frame)

    # ── Fit team assigner on sampled frames ───────────────────────────────────
    print("Fitting TeamAssigner …")
    assigner   = TeamAssigner()
    fit_frames = loader.sample_frames(12, start=start_frame, count=n_frames)
    fit_dets   = [detector.detect_raw(f) for f in fit_frames]
    assigner.fit(fit_frames, fit_dets)

    player_tracker  = PlayerTracker(fps=loader.fps)
    ball_tracker    = BallTracker()
    ball_filter     = BallCandidateFilter()
    ball_state      = BallStateEstimator()
    world_model     = WorldModelV2(fps=loader.fps)
    engine          = PitchAnalyticsEngine(mgr, fps=loader.fps,
                                           image_size=(loader.width, loader.height))
    overlay         = AnalyticsOverlay()
    propagator      = None if args.no_propagate else HomographyPropagator(mgr)

    print(f"Clip    : {clip_path.name}")
    print(f"Window  : {args.start:.0f}s – {args.start + args.duration:.0f}s  ({n_frames} frames)")
    print(f"Output  : {out_path}")
    print(f"Homography : {'STATIC' if args.no_propagate else 'PROPAGATED (optical flow)'}")
    print("Processing …")

    timings: list[float] = []
    # Running means for the end-of-run summary
    sum_area_a = sum_area_b = sum_press = 0.0
    n_area = n_press = 0

    with VideoWriter(out_path, fps=loader.fps,
                     width=loader.width, height=loader.height) as writer:

        for frame_idx, frame in loader.frames(start=start_frame, count=n_frames):
            t0 = time.perf_counter()

            dets       = detector.detect(frame, frame_idx)
            dets       = assigner.assign(frame, dets)
            ball_dets  = [d for d in dets if d.class_name == "ball"]
            field_dets = [d for d in dets if d.class_name != "ball"]

            # Problem B: advance H to this frame's camera pose BEFORE projecting.
            # Mask out players/ball (independent motion) — keep pitch + crowd.
            if propagator is not None:
                propagator.update(frame, exclude_dets=dets)

            is_detect_frame = (frame_idx == detector._last_detect_idx)

            if is_detect_frame:
                field_dets = player_tracker.update(field_dets, frame)
                ball_dets  = ball_filter.filter(
                    ball_dets, field_dets, frame_idx,
                    homography_manager=mgr,
                    last_ball_pos=ball_tracker.last_position_px(),
                    last_ball_vel=ball_tracker.last_velocity_vector(),
                    last_detection_frame=ball_tracker.last_detection_frame(),
                    state_estimator=ball_state,
                    world_model=world_model,
                )
                ball_result = ball_tracker.update(ball_dets, frame_idx)
                state_det = (ball_result if ball_result is not None
                             and ball_result.confidence > 0 else None)
                ball_state.update(state_det, ball_tracker.last_velocity_vector(),
                                  field_dets, frame_idx)
                # Player positions only change on detection frames, so run the
                # (Voronoi/hull) analytics here and reuse the snapshot on skip
                # frames — recomputing on carry-forward positions is pure waste.
                all_dets = field_dets + ([ball_result] if ball_result is not None else [])
                snap = engine.update(frame_idx, all_dets)
                # World model updates AFTER analytics so it has the full snapshot
                # (including events) — its context is used on the NEXT frame's filter.
                if snap is not None:
                    world_model.update(snap, mgr)
            else:
                snap = engine.last

            out = overlay.render(frame, snap)
            writer.write(out)
            timings.append(time.perf_counter() - t0)

            if snap is not None:
                sum_area_a += snap.team_a.compactness.hull_area_m2
                sum_area_b += snap.team_b.compactness.hull_area_m2
                n_area += 1
                if snap.pressing is not None:
                    sum_press += snap.pressing["intensity"]
                    n_press += 1

            done = len(timings)
            if done % 150 == 0:
                pct = 100 * done / n_frames
                fps_p = 1.0 / max(float(np.mean(timings[-50:])), 1e-6)
                print(f"  {pct:5.1f}%  frame {frame_idx}  {fps_p:.1f} fps")

    loader.close()

    poss    = engine.possession_summary()
    mean_ms = float(np.mean(timings)) * 1000
    size_mb = out_path.stat().st_size / 1024 ** 2

    print()
    print("=" * 54)
    print("  Football Analytics Demo — Results")
    print("=" * 54)
    print(f"  Frames processed   : {len(timings)}")
    print(f"  Processing speed   : {mean_ms:.1f} ms/frame  ({1000/mean_ms:.1f} fps)")
    print()
    print(f"  Possession         : Team A {poss['teamA_pct']:.1f}%  |  "
          f"Team B {poss['teamB_pct']:.1f}%   ({poss['frames']} owned frames)")
    if n_area:
        print(f"  Mean hull area     : Team A {sum_area_a/n_area:.0f} m2  |  "
              f"Team B {sum_area_b/n_area:.0f} m2")
    if n_press:
        print(f"  Mean press on ball : {sum_press/n_press:.1f} opponents within "
              f"{config.PRESSING_RADIUS_M:.0f}m")
    if propagator is not None:
        s = propagator.summary()
        print()
        print(f"  H propagation      : {s['updates']} updates, {s['holds']} holds, "
              f"{s['cuts']} scene cuts")

    report = engine.pass_network_report()
    print()
    print("  Pass network & player influence")
    if report.most_frequent is not None:
        mf = report.most_frequent
        print(f"    Most frequent conn : {mf.sender} -> {mf.receiver}  ({mf.count}x)")
    if report.progressive_leaders:
        leaders = "  ".join(f"{c.sender}->{c.receiver} ({c.count}x)"
                            for c in report.progressive_leaders)
        print(f"    Progressive leaders: {leaders}")
    if report.hub_players:
        hubs = "  ".join(f"{h.label} ({h.centrality:.2f}, {h.involvement} touches)"
                         for h in report.hub_players)
        print(f"    Hub players        : {hubs}")
    if report.longest_buildup:
        print(f"    Longest build-up   : {' -> '.join(report.longest_buildup)}")

    episodes = engine.episodes_so_far()
    print()
    print(f"  Tactical episodes  : {len(episodes)} completed")
    if episodes:
        outcome_counts = Counter(ep.outcome for ep in episodes)
        print("    Outcomes           : " +
              ", ".join(f"{k} x{v}" for k, v in outcome_counts.most_common()))
        notable = sorted(episodes, key=lambda ep: len(ep.events), reverse=True)[:3]
        for ep in notable:
            print(f"    #{ep.episode_id} {ep.team}: {ep.narrative()}  "
                  f"({ep.duration_s}s, outcome: {ep.outcome})")

    print()
    print(engine.match_report().render())

    print()
    print(f"  Output : {out_path}  ({size_mb:.1f} MB)")
    print("=" * 54)


if __name__ == "__main__":
    main()
