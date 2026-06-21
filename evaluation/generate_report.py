"""
evaluation/generate_report.py
─────────────────────────────────
Orchestrates E1-E4 across the calibrated clips and writes
evaluation/evaluation_report.md.

Runs v1.0 and v2.0 once per calibrated clip (4 runs total -- v2.0 runs also
capture pass-network checkpoints at the 50% and 100% marks for E3), and
caches each run's derived numbers under evaluation/_cache/<clip_stem>__
<model>.json so re-running the report after a wording or windowing change
doesn't repeat a several-minute-per-clip detection pass. Use --force to
bypass the cache, or --duration to smoke-test on a short clip prefix.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from evaluation import evaluate_ball_tracking, evaluate_episodes, evaluate_events, evaluate_networks
from evaluation.clip_runner import ClipRunResult, run_clip
from gaffer import config
from gaffer.analytics.roles import PlayerRole
from gaffer.tracking.world_model import BallWorldModel
from gaffer.tracking.world_model_v2 import WorldModelV2

CACHE_DIR = Path(__file__).parent / "_cache"
REPORT_PATH = Path(__file__).parent / "evaluation_report.md"

CALIBRATED_CLIPS = [
    (config.DATA_DIR / "test_clips" / "arsenal_newcastle_highlights.mp4",
     config.DATA_DIR / "calibration" / "arsenal_newcastle_highlights.json"),
    (config.DATA_DIR / "test_clips" / "tactical_playlist_1.mp4",
     config.DATA_DIR / "calibration" / "tactical_playlist_1.json"),
]


@dataclass
class _CachedEvent:
    event_type: str
    team: str | None
    time_s: float


@dataclass
class _CachedEpisode:
    outcome: str
    duration_s: float
    events: list  # placeholders -- only len() is used downstream


def _serialize(result: ClipRunResult) -> dict:
    return {
        "clip_name":         result.clip_name,
        "world_model_name":  result.world_model_name,
        "n_frames":          result.n_frames,
        "duration_s":        result.duration_s,
        "fps":               result.fps,
        "ball_metrics":      result.ball_metrics,
        "events": [
            {"event_type": e.event_type, "team": e.team, "time_s": e.time_s}
            for e in result.events
        ],
        "episodes": [
            {"outcome": ep.outcome, "duration_s": ep.duration_s, "n_events": len(ep.events)}
            for ep in result.episodes
        ],
        "pass_net_checkpoints": {
            str(frac): [[list(edge), count] for edge, count in net.items()]
            for frac, net in result.pass_net_checkpoints.items()
        },
        "attacking_third_entries": result.attacking_third_entries,
        "roles": {
            str(tid): {"role": r.role, "line": r.line}
            for tid, r in result.roles.items()
        },
    }


def _deserialize(data: dict) -> ClipRunResult:
    return ClipRunResult(
        clip_name=data["clip_name"],
        world_model_name=data["world_model_name"],
        n_frames=data["n_frames"],
        duration_s=data["duration_s"],
        fps=data["fps"],
        ball_metrics=data["ball_metrics"],
        events=[_CachedEvent(**e) for e in data["events"]],
        episodes=[
            _CachedEpisode(outcome=ep["outcome"], duration_s=ep["duration_s"],
                            events=[None] * ep["n_events"])
            for ep in data["episodes"]
        ],
        pass_net_checkpoints={
            float(frac): {tuple(edge): count for edge, count in edges}
            for frac, edges in data["pass_net_checkpoints"].items()
        },
        attacking_third_entries=data["attacking_third_entries"],
        roles={
            int(tid): PlayerRole(track_id=int(tid), role=r["role"], line=r["line"])
            for tid, r in data["roles"].items()
        },
    )


def get_or_run(
    clip_path: Path, calib_path: Path, world_model_cls: type, *,
    checkpoint_fracs: tuple[float, ...] = (1.0,), force: bool = False, **run_kwargs,
) -> ClipRunResult:
    model_name = "v1.0" if world_model_cls is BallWorldModel else "v2.0"
    cache_path = CACHE_DIR / f"{clip_path.stem}__{model_name}.json"
    cache_key = {"checkpoint_fracs": list(checkpoint_fracs), **run_kwargs}

    if cache_path.exists() and not force:
        cached = json.loads(cache_path.read_text())
        if cached.get("_cache_key") == cache_key:
            return _deserialize(cached)

    result = run_clip(clip_path, calib_path, world_model_cls=world_model_cls,
                       checkpoint_fracs=checkpoint_fracs, **run_kwargs)
    CACHE_DIR.mkdir(exist_ok=True)
    payload = _serialize(result)
    payload["_cache_key"] = cache_key
    cache_path.write_text(json.dumps(payload, indent=2))
    return result


def run_all(*, force: bool = False, **run_kwargs) -> dict:
    all_results: list[ClipRunResult] = []
    v2_results: list[ClipRunResult] = []
    for clip_path, calib_path in CALIBRATED_CLIPS:
        v1 = get_or_run(clip_path, calib_path, BallWorldModel, force=force, **run_kwargs)
        v2 = get_or_run(clip_path, calib_path, WorldModelV2,
                         checkpoint_fracs=(0.5, 1.0), force=force, **run_kwargs)
        all_results.extend([v1, v2])
        v2_results.append(v2)
    return {"all": all_results, "v2": v2_results}


def render_report(results: dict) -> str:
    all_results, v2_results = results["all"], results["v2"]

    e2_summary = evaluate_events.evaluate(v2_results)
    e3_summary = evaluate_networks.evaluate(v2_results)
    e4_summary = evaluate_episodes.evaluate(v2_results)

    clip_list = ", ".join(r.clip_name for r in v2_results)
    sections = [
        "# Gaffer Evaluation Report (v1.E)",
        "",
        "## Methodology & Limitations",
        "",
        f"- **Sample size: 2 calibrated clips** ({clip_list}). Calibration is an "
        "interactive point-and-click step (`scripts/collect_calibration.py`); "
        "`psg_newcastle_tactical`, `tactical_playlist_2`, and `tactical_playlist_3` "
        "have none yet, so every result below is drawn from just these two matches.",
        "- **No ground-truth ball-position labels exist in this repo.** E1's "
        "\"recoveries\" / \"suspect discards\" are `BallCandidateFilter`'s own "
        "self-reported counters, not validated precision/recall against labeled data.",
        "- **E3 (pass-network stability) is within-match only** (first half vs "
        "second half of the same clip), not cross-broadcast -- the only team that "
        "recurs across two different clips here (Newcastle) has its second "
        "appearance (`psg_newcastle_tactical`) uncalibrated.",
        "- **E2's conditional probabilities use 5-second time windows** -- a "
        "documented methodology choice, not a hidden constant "
        "(see `evaluation/evaluate_events.py`).",
        "",
        "## E1 — Ball Tracking: v1.0 vs v2.0",
        "",
        evaluate_ball_tracking.render_table(all_results),
        "",
        evaluate_ball_tracking.conclusion(all_results),
        "",
        "## E2 — Tactical Event Validation",
        "",
        evaluate_events.render_section(e2_summary),
        "",
        evaluate_events.conclusion(e2_summary),
        "",
        "## E3 — Pass Network Stability (within-match)",
        "",
        evaluate_networks.render_section(e3_summary),
        "",
        evaluate_networks.conclusion(e3_summary),
        "",
        "## E4 — Episode Validation",
        "",
        evaluate_episodes.render_section(e4_summary),
        "",
        evaluate_episodes.conclusion(e4_summary),
        "",
    ]
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="bypass cache, re-run detection")
    parser.add_argument("--duration", type=float, default=None, help="seconds per clip (smoke test)")
    args = parser.parse_args()

    run_kwargs = {}
    if args.duration is not None:
        run_kwargs["duration"] = args.duration

    results = run_all(force=args.force, **run_kwargs)
    report = render_report(results)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
