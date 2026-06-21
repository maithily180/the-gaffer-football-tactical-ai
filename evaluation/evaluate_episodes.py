"""
evaluation/evaluate_episodes.py
───────────────────────────────────
E4 — Episode validation: what do completed tactical episodes look like in
aggregate, broken down by outcome type?

Pools engine.episodes_so_far() across all v2.0 runs (one per calibrated
clip -- reuses the runs already made for E1/E2, no new detection pass
needed). Episode.outcome and Episode.duration_s already exist on the
dataclass (gaffer/analytics/episodes.py); this just groups and averages.
"""

from __future__ import annotations

import statistics
from collections import defaultdict

from gaffer.analysis.pipeline_runner import ClipRunResult


def evaluate(results: list[ClipRunResult]) -> dict:
    by_outcome: dict[str, list] = defaultdict(list)
    all_episodes = []
    for r in results:
        all_episodes.extend(r.episodes)
    for ep in all_episodes:
        by_outcome[ep.outcome].append(ep)

    per_outcome = {}
    for outcome, eps in by_outcome.items():
        n_events = [len(ep.events) for ep in eps]
        durations = [ep.duration_s for ep in eps]
        per_outcome[outcome] = {
            "n_episodes":      len(eps),
            "mean_events":     round(statistics.mean(n_events), 1),
            "mean_duration_s": round(statistics.mean(durations), 1),
        }

    overall = {
        "n_episodes":      len(all_episodes),
        "mean_events":      round(statistics.mean([len(ep.events) for ep in all_episodes]), 1) if all_episodes else 0.0,
        "mean_duration_s":  round(statistics.mean([ep.duration_s for ep in all_episodes]), 1) if all_episodes else 0.0,
        "n_per_clip":       {r.clip_name: len(r.episodes) for r in results},
    }

    return {"per_outcome": per_outcome, "overall": overall}


def render_section(summary: dict) -> str:
    lines = [
        "| Outcome | Episodes | Mean events/episode | Mean duration (s) |",
        "|---|---|---|---|",
    ]
    for outcome, stats in sorted(summary["per_outcome"].items(), key=lambda kv: -kv[1]["n_episodes"]):
        lines.append(f"| {outcome} | {stats['n_episodes']} | {stats['mean_events']:.1f} | {stats['mean_duration_s']:.1f} |")
    overall = summary["overall"]
    lines.append(
        f"| **All outcomes** | **{overall['n_episodes']}** | **{overall['mean_events']:.1f}** | "
        f"**{overall['mean_duration_s']:.1f}** |"
    )
    return "\n".join(lines)


def conclusion(summary: dict) -> str:
    overall = summary["overall"]
    if overall["n_episodes"] == 0:
        return "No completed tactical episodes were detected across the calibrated clips."
    per_clip = ", ".join(f"{clip}: {n}" for clip, n in overall["n_per_clip"].items())
    most_common = max(summary["per_outcome"].items(), key=lambda kv: kv[1]["n_episodes"])[0]
    return (
        f"{overall['n_episodes']} completed episodes pooled across calibrated clips ({per_clip}), "
        f"averaging {overall['mean_events']:.1f} events and {overall['mean_duration_s']:.1f}s each. "
        f"Most common outcome: {most_common}."
    )


if __name__ == "__main__":
    from gaffer import config
    from gaffer.analysis.pipeline_runner import run_clip
    from gaffer.tracking.world_model_v2 import WorldModelV2

    clips = [
        (config.DATA_DIR / "test_clips" / "arsenal_newcastle_highlights.mp4",
         config.DATA_DIR / "calibration" / "arsenal_newcastle_highlights.json"),
        (config.DATA_DIR / "test_clips" / "tactical_playlist_1.mp4",
         config.DATA_DIR / "calibration" / "tactical_playlist_1.json"),
    ]
    results = [run_clip(clip, calib, world_model_cls=WorldModelV2, duration=20.0) for clip, calib in clips]
    summary = evaluate(results)
    print(render_section(summary))
    print()
    print(conclusion(summary))
