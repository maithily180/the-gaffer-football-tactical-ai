"""
scripts/ab_compare_detectors.py
───────────────────────────────
The success gate for Day 5.

Runs the SAME tracking pipeline (detect → ByteTrack → PositionStore) twice over
one clip — once with the base COCO model, once with the fine-tuned football
model — and prints a side-by-side comparison. This is what actually proves the
fine-tune was worth it; the Colab mAP numbers only prove it learned the dataset,
not that it helps Gaffer's real pipeline.

Metrics compared (per requirement #8):
  • detections per frame, broken down by class (player/gk/referee/ball)
  • unique track IDs           (lower is better — fewer ID switches)
  • tracks ≥ 10 s              (higher is better — stable long tracks)
  • mean track persistence     (higher is better — less fragmentation)

Run locally on the Intel machine — inference only, no training:

    uv run python scripts/ab_compare_detectors.py \
        data/test_clips/tactical_playlist_1.mp4 \
        --fine-tuned weights/yolov11_football.pt \
        --start 30 --duration 60
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Windows consoles default to cp1252 and choke on the box-drawing glyphs below.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from gaffer import config
from gaffer.detection.detector import FootballDetector
from gaffer.tracking.position_store import PositionStore
from gaffer.tracking.tracker import PlayerTracker
from gaffer.video.loader import VideoLoader


@dataclass
class RunStats:
    label:             str
    model_path:        str
    model_type:        str
    frames:            int                = 0
    dets_per_frame:    float              = 0.0
    class_per_frame:   dict[str, float]   = field(default_factory=dict)
    unique_track_ids:  int                = 0
    long_tracks:       int                = 0     # tracks ≥ 10 s
    mean_persistence:  float              = 0.0
    mean_active_tracks:float              = 0.0


def _run(label: str, model_path: str | Path | None,
         clip: str | Path, start_s: float, duration_s: float) -> RunStats:
    loader = VideoLoader(clip)
    start_frame = int(start_s * loader.fps)
    n_frames    = min(int(duration_s * loader.fps), loader.total_frames - start_frame)
    ten_s       = int(10 * loader.fps)

    detector = FootballDetector(model_path=model_path, verbose=False)
    tracker  = PlayerTracker(fps=loader.fps)
    store    = PositionStore(fps=loader.fps)

    print(f"\n▶ {label}  ({detector.model_type} model: {Path(detector.model_path).name})")

    class_counts: dict[str, int] = {}
    detect_frames = 0          # frames where inference actually ran
    active_per_frame: list[int] = []

    for frame_idx, frame in loader.frames(start=start_frame, count=n_frames):
        dets = detector.detect(frame, frame_idx)

        if frame_idx == detector._last_detect_idx:        # real inference this frame
            dets = tracker.update(dets, frame)            # frame → BoT-SORT CMC
            detect_frames += 1
            for d in dets:
                class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1
        else:
            dets = tracker.carry_forward()

        store.update(frame_idx, dets)
        active_per_frame.append(sum(1 for d in dets if d.track_id >= 0))

    loader.close()

    persistences = [p for t in store.track_ids
                    if (p := store.persistence(t)) is not None]

    return RunStats(
        label             = label,
        model_path        = str(detector.model_path),
        model_type        = detector.model_type,
        frames            = n_frames,
        dets_per_frame    = sum(class_counts.values()) / max(detect_frames, 1),
        class_per_frame   = {c: n / max(detect_frames, 1) for c, n in sorted(class_counts.items())},
        unique_track_ids  = len(store.track_ids),
        long_tracks       = sum(1 for t in store.track_ids if store.track_length(t) >= ten_s),
        mean_persistence  = float(np.mean(persistences)) if persistences else 0.0,
        mean_active_tracks= float(np.mean(active_per_frame)) if active_per_frame else 0.0,
    )


def _print_comparison(base: RunStats, ft: RunStats) -> None:
    def row(name: str, b, f, better: str, fmt: str = "{:.2f}") -> None:
        bs, fs = fmt.format(b), fmt.format(f)
        if better == "higher":
            win = "FT ✓" if f > b else ("base ✓" if b > f else "tie")
        elif better == "lower":
            win = "FT ✓" if f < b else ("base ✓" if b < f else "tie")
        else:
            win = ""
        print(f"  {name:<26}{bs:>12}{fs:>12}   {win}")

    print("\n" + "═" * 66)
    print("  A/B COMPARISON — base COCO  vs  fine-tuned football")
    print("═" * 66)
    print(f"  {'metric':<26}{'base':>12}{'fine-tuned':>12}   winner")
    print("  " + "─" * 62)
    row("detections / frame",      base.dets_per_frame,     ft.dets_per_frame,     "higher")
    row("mean active tracks",      base.mean_active_tracks, ft.mean_active_tracks, "higher")
    row("unique track IDs",        base.unique_track_ids,   ft.unique_track_ids,   "lower", "{:.0f}")
    row("tracks >= 10s",           base.long_tracks,        ft.long_tracks,        "higher", "{:.0f}")
    row("mean persistence",        base.mean_persistence,   ft.mean_persistence,   "higher")

    print("\n  per-class detections / frame")
    classes = sorted(set(base.class_per_frame) | set(ft.class_per_frame))
    for c in classes:
        b = base.class_per_frame.get(c, 0.0)
        f = ft.class_per_frame.get(c, 0.0)
        print(f"    {c:<24}{b:>12.2f}{f:>12.2f}")

    # Verdict — weight the metrics that matter most for Gaffer.
    print("\n" + "─" * 66)
    ball_gain = ft.class_per_frame.get("ball", 0.0) - base.class_per_frame.get("ball", 0.0)
    id_better = ft.unique_track_ids < base.unique_track_ids
    long_better = ft.long_tracks >= base.long_tracks
    persist_better = ft.mean_persistence >= base.mean_persistence

    wins = sum([id_better, long_better, persist_better, ball_gain > 0])
    print(f"  Fine-tuned wins {wins}/4 headline checks "
          f"(fewer IDs, more long tracks, persistence, ball detection).")
    if "ball" in ft.class_per_frame and "ball" not in base.class_per_frame:
        print("  → Fine-tuned detects the BALL, which the base model never does.")
    verdict = "PASS — ship the fine-tuned model" if wins >= 3 else \
              "INCONCLUSIVE — inspect per-class numbers before swapping"
    print(f"  VERDICT: {verdict}")
    print("═" * 66)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A/B compare base vs fine-tuned detector")
    p.add_argument("clip", help="Path to input .mp4")
    p.add_argument("--base", default=None,
                   help="Base model path (default: repo yolo11n.pt)")
    p.add_argument("--fine-tuned", default=str(config.DETECTION_MODEL_PATH),
                   help=f"Fine-tuned model (default: {config.DETECTION_MODEL_PATH})")
    p.add_argument("--start", type=float, default=30.0)
    p.add_argument("--duration", type=float, default=60.0)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    base_model = args.base or (config.ROOT / "yolo11n.pt")

    ft_path = Path(args.fine_tuned)
    if not ft_path.exists():
        raise SystemExit(
            f"Fine-tuned model not found: {ft_path}\n"
            f"Train it on Colab first (notebooks/04_finetune_yolo.ipynb), then "
            f"drop best.pt at {config.DETECTION_MODEL_PATH}."
        )

    base = _run("BASE  (COCO yolo11n)", base_model, args.clip, args.start, args.duration)
    ft   = _run("FINE-TUNED (football)", ft_path,   args.clip, args.start, args.duration)
    _print_comparison(base, ft)
