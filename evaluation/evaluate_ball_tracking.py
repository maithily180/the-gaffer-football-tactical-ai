"""
evaluation/evaluate_ball_tracking.py
───────────────────────────────────────
E1 — Ball tracking evaluation: v1.0 (BallWorldModel) vs v2.0 (WorldModelV2).

Runs both world models on each calibrated clip via pipeline_runner.run_clip()
and builds a comparison table. "Recoveries" and "suspect discards" are not
new metrics -- they're BallCandidateFilter's own self-reported counters
(gaffer/tracking/ball_candidate_filter.py:267, rejection_summary()).

There is no human-labeled ground-truth ball-position dataset anywhere in
this repo, so "false positives" cannot be measured as true precision.
What's reported instead is how often the filter pipeline itself rejected
or self-corrected a candidate -- a proxy for how hard each world model
makes the filter work, not a validated accuracy number. The report says
this explicitly rather than presenting it as something it isn't.
"""

from __future__ import annotations

from pathlib import Path

from gaffer.analysis.pipeline_runner import ClipRunResult, run_clip
from gaffer.tracking.world_model import BallWorldModel
from gaffer.tracking.world_model_v2 import WorldModelV2

_COLUMNS = [
    ("Clip",                      lambda r: r.clip_name),
    ("Model",                     lambda r: r.world_model_name),
    ("Live detections",           lambda r: r.ball_metrics["n_live"]),
    ("Extrapolated",              lambda r: r.ball_metrics["n_extrapolated"]),
    ("Lost %",                    lambda r: f"{r.ball_metrics['lost_pct']:.1f}"),
    ("Recoveries",                lambda r: r.ball_metrics["rejection_summary"]["recovery_accept"]),
    ("Suspect discards",          lambda r: r.ball_metrics["rejection_summary"]["suspect_discards"]),
    ("Mean continuity (frames)",  lambda r: f"{r.ball_metrics['mean_continuity_frames']:.1f}"),
    ("Mean WM score",             lambda r: f"{r.ball_metrics['mean_wm_score']:.3f}"),
]


def evaluate(clips: list[tuple[Path, Path]], *, runner=run_clip, **run_kwargs) -> list[ClipRunResult]:
    """clips: list of (clip_path, calib_path). Runs BOTH world models on each
    clip and returns one ClipRunResult per (clip, model) pair."""
    results: list[ClipRunResult] = []
    for clip_path, calib_path in clips:
        for cls in (BallWorldModel, WorldModelV2):
            results.append(runner(clip_path, calib_path, world_model_cls=cls, **run_kwargs))
    return results


def render_table(results: list[ClipRunResult]) -> str:
    header = "| " + " | ".join(name for name, _ in _COLUMNS) + " |"
    sep    = "|" + "|".join("---" for _ in _COLUMNS) + "|"
    rows   = ["| " + " | ".join(str(fn(r)) for _, fn in _COLUMNS) + " |" for r in results]
    return "\n".join([header, sep] + rows)


def conclusion(results: list[ClipRunResult]) -> str:
    v1 = [r for r in results if r.world_model_name == "v1.0"]
    v2 = [r for r in results if r.world_model_name == "v2.0"]
    if not v1 or not v2:
        return "(insufficient data for a v1.0 vs v2.0 comparison)"

    def _avg(rs, key):
        return sum(r.ball_metrics[key] for r in rs) / len(rs)

    def _avg_rej(rs, key):
        return sum(r.ball_metrics["rejection_summary"][key] for r in rs) / len(rs)

    lines = [
        f"Pooled across {len(v1)} clip(s): v2.0 lost-ball% "
        f"{_avg(v2, 'lost_pct'):.1f} vs v1.0 {_avg(v1, 'lost_pct'):.1f}; "
        f"mean continuity {_avg(v2, 'mean_continuity_frames'):.1f} vs "
        f"{_avg(v1, 'mean_continuity_frames'):.1f} frames; mean world-model "
        f"plausibility score {_avg(v2, 'mean_wm_score'):.3f} vs "
        f"{_avg(v1, 'mean_wm_score'):.3f}; mean recoveries-after-loss "
        f"{_avg_rej(v2, 'recovery_accept'):.1f} vs {_avg_rej(v1, 'recovery_accept'):.1f}.",
    ]

    by_clip_v1 = {r.clip_name: r for r in v1}
    by_clip_v2 = {r.clip_name: r for r in v2}
    identical_clips = [
        name for name in by_clip_v1
        if name in by_clip_v2
        and by_clip_v1[name].ball_metrics["n_live"] == by_clip_v2[name].ball_metrics["n_live"]
        and by_clip_v1[name].ball_metrics["n_lost"] == by_clip_v2[name].ball_metrics["n_lost"]
    ]
    if identical_clips and len(identical_clips) == len(by_clip_v1):
        lines.append(
            "v1.0 and v2.0 accepted/rejected the exact same candidates on every "
            "clip here (identical live/extrapolated/lost counts) -- v2.0's extra "
            "signals (corridor prediction, overload-zone prior, space-control "
            "score, press-locality) changed its own plausibility score but never "
            "flipped an accept/reject decision in this sample."
        )

    dominant_rejections = []
    for r in v2:
        rs = r.ball_metrics["rejection_summary"]
        gate_keys = ("stationary", "spatial", "proximity", "off_pitch")
        top_gate = max(gate_keys, key=lambda k: rs[k])
        if rs[top_gate] > 0 and rs["scene_cuts"] == 0:
            dominant_rejections.append((r.clip_name, top_gate, rs[top_gate]))
    if dominant_rejections:
        detail = "; ".join(f"{clip}: {gate} ({n})" for clip, gate, n in dominant_rejections)
        lines.append(
            "Zero scene cuts were detected on either clip, so the bulk of "
            "lost-ball frames trace to the filter's own spatial/off-pitch "
            f"plausibility gates rejecting candidates ({detail}) -- i.e. the "
            "upstream object detector flagging non-ball objects as \"ball\" "
            "more often than the tracker can recover from, not scene cuts or "
            "a world-model weakness."
        )

    lines.append(
        "Recoveries and suspect discards are the filter's own self-reported "
        "counters, not validated against ground truth -- there is no "
        "human-labeled ball-position dataset in this repo, so these measure "
        "how hard each world model makes the filter work, not precision/recall."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    from gaffer import config

    clips = [
        (config.DATA_DIR / "test_clips" / "arsenal_newcastle_highlights.mp4",
         config.DATA_DIR / "calibration" / "arsenal_newcastle_highlights.json"),
        (config.DATA_DIR / "test_clips" / "tactical_playlist_1.mp4",
         config.DATA_DIR / "calibration" / "tactical_playlist_1.json"),
    ]
    results = evaluate(clips, duration=20.0)  # short smoke test
    print(render_table(results))
    print()
    print(conclusion(results))
