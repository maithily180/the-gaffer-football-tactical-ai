"""
evaluation/clip_runner.py
───────────────────────────
Shared instrumented pipeline runner for the evaluation suite.

Re-runs the exact perception loop scripts/render_analytics_demo.py uses
(detect -> team-assign -> track -> ball-filter -> ball-track -> ball-state
-> engine.update() -> world_model.update()), headless -- no overlay, no
video output, because this is analysis, not a demo. Parametrized by which
world-model class drives the ball filter, so the same function produces
both the v1.0 and v2.0 runs evaluate_ball_tracking.py compares.

Every metric here is either pulled straight from an existing public method
(engine.episodes_so_far(), engine.pass_network(), world_model.score_candidate_px(),
ball_filter.rejection_summary()) or, where nothing upstream tracks it yet,
accumulated inline:
  - ball live/extrapolated/lost classification (from ball_tracker's own
    confidence convention -- not new, just counted)
  - attacking-third entries per team (snap.ball_region + attack_dir, the
    same mapping analytics_overlay.py's _attacking_third_pct() uses)
  - the full per-frame event log (AnalyticsSnapshot.events only ever holds
    the current frame's events -- nothing upstream keeps the whole match)
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path

from gaffer.analytics.engine import PitchAnalyticsEngine
from gaffer.analytics.episodes import Episode
from gaffer.calibration.homography_manager import HomographyManager
from gaffer.calibration.homography_propagator import HomographyPropagator
from gaffer.detection.detector import FootballDetector
from gaffer.detection.team_assigner import TeamAssigner
from gaffer.events.base import FootballEvent
from gaffer.tracking.ball_candidate_filter import BallCandidateFilter
from gaffer.tracking.ball_state_estimator import BallStateEstimator
from gaffer.tracking.ball_tracker import BallTracker
from gaffer.tracking.tracker import PlayerTracker
from gaffer.tracking.world_model import BallWorldModel
from gaffer.tracking.world_model_v2 import WorldModelV2
from gaffer.video.loader import VideoLoader

_ATK_THIRD_FOR_DIR = {+1: "right_third", -1: "left_third"}


@dataclass
class ClipRunResult:
    clip_name:                str
    world_model_name:         str                                   # "v1.0" | "v2.0"
    n_frames:                 int                                    # detection frames processed
    duration_s:               float                                  # wall-clock span of the run
    fps:                      float
    ball_metrics:             dict
    events:                   list[FootballEvent] = field(default_factory=list)
    episodes:                 list[Episode] = field(default_factory=list)
    pass_net_checkpoints:     dict[float, dict[tuple[int, int], int]] = field(default_factory=dict)
    attacking_third_entries:  dict[str, list[float]] = field(default_factory=dict)
    roles:                    dict = field(default_factory=dict)      # track_id -> PlayerRole, best-known by run end


def run_clip(
    clip_path: Path,
    calib_path: Path,
    *,
    world_model_cls: type = WorldModelV2,
    start: float = 0.0,
    duration: float | None = None,
    checkpoint_fracs: tuple[float, ...] = (1.0,),
) -> ClipRunResult:
    """Run the full perception + analytics pipeline once over [start, start+duration)
    of clip_path, with world_model_cls driving the ball candidate filter."""
    clip_path = Path(clip_path)
    calib_path = Path(calib_path)

    mgr = HomographyManager.from_calibration(calib_path)
    loader = VideoLoader(str(clip_path))
    detector = FootballDetector(verbose=False)

    start_frame = int(start * loader.fps)
    if duration is None:
        n_total = loader.total_frames - start_frame
    else:
        n_total = min(int(duration * loader.fps), loader.total_frames - start_frame)

    assigner = TeamAssigner()
    fit_frames = loader.sample_frames(12, start=start_frame, count=n_total)
    fit_dets = [detector.detect_raw(f) for f in fit_frames]
    assigner.fit(fit_frames, fit_dets)

    player_tracker = PlayerTracker(fps=loader.fps)
    ball_tracker = BallTracker()
    ball_filter = BallCandidateFilter()
    ball_state = BallStateEstimator()
    world_model = world_model_cls(fps=loader.fps)
    engine = PitchAnalyticsEngine(mgr, fps=loader.fps,
                                   image_size=(loader.width, loader.height))
    propagator = HomographyPropagator(mgr)

    all_events: list[FootballEvent] = []
    attacking_third_entries: dict[str, list[float]] = {"teamA": [], "teamB": []}
    prev_in_atk_third = {"teamA": False, "teamB": False}

    n_live = n_extrapolated = n_lost = 0
    n_lost_streaks = 0
    in_lost_streak = False
    cur_run = 0
    continuity_runs: list[int] = []
    wm_scores: list[float] = []

    checkpoints_sorted = sorted(checkpoint_fracs)
    next_ckpt_i = 0
    pass_net_checkpoints: dict[float, dict] = {}

    for frame_idx, frame in loader.frames(start=start_frame, count=n_total):
        dets = detector.detect(frame, frame_idx)
        dets = assigner.assign(frame, dets)
        ball_dets = [d for d in dets if d.class_name == "ball"]
        field_dets = [d for d in dets if d.class_name != "ball"]

        propagator.update(frame, exclude_dets=dets)

        if frame_idx != detector._last_detect_idx:
            continue  # skip frame -- nothing new to instrument

        field_dets = player_tracker.update(field_dets, frame)
        ball_dets = ball_filter.filter(
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

        all_dets = field_dets + ([ball_result] if ball_result is not None else [])
        snap = engine.update(frame_idx, all_dets)
        if snap is not None:
            world_model.update(snap, mgr)

        # ── ball-state classification ────────────────────────────────────
        if ball_result is None:
            n_lost += 1
            if cur_run > 0:
                continuity_runs.append(cur_run)
            cur_run = 0
            in_lost_streak = True
        else:
            if ball_result.confidence > 0:
                n_live += 1
            else:
                n_extrapolated += 1
            cur_run += 1
            if in_lost_streak:
                n_lost_streaks += 1
                in_lost_streak = False
            if mgr.is_valid():
                wm_scores.append(world_model.score_candidate_px(ball_result.center, mgr))

        # ── events + attacking-third entries ─────────────────────────────
        if snap is not None:
            all_events.extend(snap.events)
            time_s = frame_idx / loader.fps
            for team, shape in (("teamA", snap.team_a), ("teamB", snap.team_b)):
                target = _ATK_THIRD_FOR_DIR.get(shape.attack_dir)
                in_atk = target is not None and snap.ball_region == target
                if in_atk and not prev_in_atk_third[team]:
                    attacking_third_entries[team].append(time_s)
                prev_in_atk_third[team] = in_atk

        # ── pass-network checkpoints ──────────────────────────────────────
        frac_done = (frame_idx - start_frame + 1) / n_total
        while next_ckpt_i < len(checkpoints_sorted) and frac_done >= checkpoints_sorted[next_ckpt_i]:
            pass_net_checkpoints[checkpoints_sorted[next_ckpt_i]] = dict(engine.pass_network())
            next_ckpt_i += 1

    loader.close()
    if cur_run > 0:
        continuity_runs.append(cur_run)
    for frac in checkpoints_sorted:
        pass_net_checkpoints.setdefault(frac, dict(engine.pass_network()))

    n_detection_frames = n_live + n_extrapolated + n_lost
    ball_metrics = {
        "n_detection_frames":       n_detection_frames,
        "n_live":                   n_live,
        "n_extrapolated":           n_extrapolated,
        "n_lost":                   n_lost,
        "lost_pct":                 round(100 * n_lost / n_detection_frames, 1) if n_detection_frames else 0.0,
        "mean_continuity_frames":   round(statistics.mean(continuity_runs), 1) if continuity_runs else 0.0,
        "median_continuity_frames": round(statistics.median(continuity_runs), 1) if continuity_runs else 0.0,
        "n_lost_streaks":           n_lost_streaks,
        "mean_wm_score":            round(statistics.mean(wm_scores), 3) if wm_scores else 0.0,
        "rejection_summary":        ball_filter.rejection_summary(),
    }

    return ClipRunResult(
        clip_name=clip_path.stem,
        world_model_name="v1.0" if world_model_cls is BallWorldModel else "v2.0",
        n_frames=n_detection_frames,
        duration_s=round(n_total / loader.fps, 1),
        fps=loader.fps,
        ball_metrics=ball_metrics,
        events=all_events,
        episodes=engine.episodes_so_far(),
        pass_net_checkpoints=pass_net_checkpoints,
        attacking_third_entries=attacking_third_entries,
        roles=engine.roles_known(),
    )
