"""
gaffer/pipeline.py
──────────────────
Top-level integration: video → detection → team assignment → tracking → annotation → MP4.

CLI usage
---------
    uv run python gaffer/pipeline.py data/test_clips/tactical_playlist_1.mp4

Python usage
------------
    from gaffer.pipeline import GafferPipeline
    pipe = GafferPipeline()
    metrics = pipe.run("clip.mp4", "outputs/v0_1_demo.mp4", start_s=30, duration_s=60)
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gaffer import config
from gaffer.detection.detector import FootballDetector
from gaffer.detection.team_assigner import TeamAssigner
from gaffer.output.annotator import Annotator
from gaffer.tracking.position_store import PositionStore
from gaffer.tracking.tracker import PlayerTracker
from gaffer.video.loader import VideoLoader
from gaffer.video.writer import VideoWriter


@dataclass
class PipelineMetrics:
    """Collected at the end of a run."""
    clip_name:        str      = ""
    start_s:          float    = 0.0
    duration_s:       float    = 0.0
    frames_processed: int      = 0
    mean_ms_per_frame:float    = 0.0
    mean_active_tracks:float   = 0.0
    unique_track_ids: int      = 0
    long_tracks:      int      = 0      # tracks ≥ 10 seconds
    output_path:      str      = ""
    output_size_mb:   float    = 0.0

    def print(self) -> None:
        print("=== Gaffer v0.1 — Pipeline Metrics ===")
        print(f"  Clip            : {self.clip_name}")
        print(f"  Window          : {self.start_s:.0f}s – {self.start_s+self.duration_s:.0f}s"
              f"  ({self.frames_processed} frames)")
        print(f"  Processing speed: {self.mean_ms_per_frame:.1f} ms/frame")
        print(f"  Mean active tracks/frame : {self.mean_active_tracks:.1f}")
        print(f"  Unique track IDs         : {self.unique_track_ids}")
        print(f"  Tracks ≥ 10 seconds      : {self.long_tracks}")
        print(f"  Output          : {self.output_path}  ({self.output_size_mb:.1f} MB)")


class GafferPipeline:
    """
    Connects FootballDetector → TeamAssigner → PlayerTracker → Annotator.

    Parameters
    ----------
    n_fit_frames : how many sample frames to use for TeamAssigner calibration
    """

    def __init__(self, n_fit_frames: int = 12):
        self._n_fit = n_fit_frames

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        clip_path:   str | Path,
        output_path: str | Path = "outputs/v0_1_demo.mp4",
        start_s:     float = 30.0,
        duration_s:  float = 60.0,
    ) -> PipelineMetrics:
        """
        Process `duration_s` seconds of `clip_path` starting at `start_s` and
        write an annotated MP4 to `output_path`.

        Returns a PipelineMetrics with summary statistics.
        """
        loader = VideoLoader(clip_path)
        print(f"Clip   : {loader}")
        print(f"Window : {start_s:.0f}s – {start_s+duration_s:.0f}s")

        start_frame = int(start_s   * loader.fps)
        n_frames    = int(duration_s * loader.fps)
        n_frames    = min(n_frames, loader.total_frames - start_frame)

        # ── 1. Initialise components ──────────────────────────────────────────
        print("Initialising detector …")
        detector = FootballDetector(verbose=False)

        print("Fitting TeamAssigner …")
        assigner = TeamAssigner()
        fit_frames = loader.sample_frames(
            self._n_fit, start=start_frame, count=n_frames
        )
        # Detect on each sample frame (raw — bypasses cache)
        fit_dets = [detector.detect_raw(f) for f in fit_frames]
        assigner.fit(fit_frames, fit_dets)
        print(f"  TeamAssigner fitted: {assigner.n_fit_samples} jersey crops  "
              f"cluster→team {assigner.cluster_to_team}")

        tracker  = PlayerTracker(fps=loader.fps)
        store    = PositionStore(fps=loader.fps)
        annotator = Annotator()

        # ── 2. Main loop ──────────────────────────────────────────────────────
        print(f"Processing {n_frames} frames …")
        active_per_frame: list[int] = []
        timings: list[float] = []
        total_processed = 0

        with VideoWriter(
            output_path,
            fps    = loader.fps,
            width  = loader.width,
            height = loader.height,
        ) as writer:
            for frame_idx, frame in loader.frames(start=start_frame, count=n_frames):
                t0 = time.perf_counter()

                dets = detector.detect(frame, frame_idx)
                dets = assigner.assign(frame, dets)

                if frame_idx == detector._last_detect_idx:
                    dets = tracker.update(dets)
                else:
                    dets = tracker.carry_forward()

                store.update(frame_idx, dets)
                elapsed = time.perf_counter() - t0

                total_processed += 1
                timings.append(elapsed)
                active_per_frame.append(sum(1 for d in dets if d.track_id >= 0))

                out = annotator.annotate(
                    frame, dets, frame_idx,
                    wall_time_s     = elapsed,
                    total_processed = total_processed,
                    fps_hint        = loader.fps,
                )
                writer.write(out)

                if total_processed % 250 == 0:
                    pct = 100 * total_processed / n_frames
                    fps = 1.0 / max(np.mean(timings[-50:]), 1e-6)
                    print(f"  {pct:5.1f}%  frame {frame_idx}  {fps:.1f} fps")

        loader.close()

        # ── 3. Collect metrics ────────────────────────────────────────────────
        ten_s = int(10 * loader.fps)
        long  = sum(1 for t in store.track_ids if store.track_length(t) >= ten_s)

        metrics = PipelineMetrics(
            clip_name         = Path(clip_path).name,
            start_s           = start_s,
            duration_s        = duration_s,
            frames_processed  = total_processed,
            mean_ms_per_frame = float(np.mean(timings)) * 1000,
            mean_active_tracks= float(np.mean(active_per_frame)),
            unique_track_ids  = len(store.track_ids),
            long_tracks       = long,
            output_path       = str(output_path),
            output_size_mb    = Path(output_path).stat().st_size / 1024 ** 2,
        )
        metrics.print()
        return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gaffer v0.1 — annotate a football clip")
    p.add_argument("clip",           help="Path to input .mp4")
    p.add_argument("--output", "-o", default="outputs/v0_1_demo.mp4",
                   help="Output path (default: outputs/v0_1_demo.mp4)")
    p.add_argument("--start",        type=float, default=30.0,
                   help="Start offset in seconds (default: 30)")
    p.add_argument("--duration",     type=float, default=60.0,
                   help="Duration in seconds (default: 60)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    pipe = GafferPipeline()
    pipe.run(args.clip, args.output, start_s=args.start, duration_s=args.duration)
