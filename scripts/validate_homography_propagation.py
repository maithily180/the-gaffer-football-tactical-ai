"""
scripts/validate_homography_propagation.py
────────────────────────────────────────────
Proves optical-flow homography propagation (v0.8) keeps H locked to the pitch
through a camera pan, where a static H drifts.

Runs two homographies in parallel over the same window:
  STATIC      — frozen calibration H
  PROPAGATED  — H advanced every frame by HomographyPropagator

Metric: the visible-pitch-region centroid (metres) from PitchVisibilityEstimator.
A correct dynamic H should make this centroid MOVE as the camera pans; the static
H holds it frozen (the v0.7.5 finding).  We also report how a fixed pitch
landmark (centre spot, 52.5, 34) projects back into the image under each H — the
propagated projection should stay glued to the real centre spot, the static one
should slide off as the camera moves.

Usage:
    uv run python scripts/validate_homography_propagation.py \
        data/test_clips/tactical_playlist_1.mp4 --start 65 --duration 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from gaffer import config
from gaffer.calibration.homography import HomographyEstimator
from gaffer.calibration.homography_manager import HomographyManager
from gaffer.calibration.homography_propagator import HomographyPropagator
from gaffer.calibration.pitch_visibility import PitchVisibilityEstimator
from gaffer.detection.detector import FootballDetector
from gaffer.video.loader import VideoLoader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--start",    type=float, default=65.0)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--calib",    default=None)
    args = ap.parse_args()

    clip_path  = Path(args.clip)
    calib_path = Path(args.calib) if args.calib else \
                 config.DATA_DIR / "calibration" / f"{clip_path.stem}.json"

    mgr_static = HomographyManager.from_calibration(calib_path)
    mgr_prop   = HomographyManager.from_calibration(calib_path)
    propagator = HomographyPropagator(mgr_prop)

    loader   = VideoLoader(str(clip_path))
    detector = FootballDetector(verbose=False)
    est      = PitchVisibilityEstimator(loader.width, loader.height)
    himg     = HomographyEstimator()

    start_frame = int(args.start * loader.fps)
    n_frames    = min(int(args.duration * loader.fps),
                      loader.total_frames - start_frame)

    centre_spot = (52.5, 34.0)   # known pitch landmark

    print(f"Clip   : {clip_path.name}")
    print(f"Window : {args.start:.0f}s – {args.start + args.duration:.0f}s")
    print(f"Metric : visible-region centroid (m) + centre-spot image projection (px)\n")
    print(f"{'time':>6} | {'STATIC centroid':>16} {'spot_px':>14} | "
          f"{'PROP centroid':>16} {'spot_px':>14} | {'inliers':>7}")
    print("-" * 92)

    ref_static = est.estimate(mgr_static)
    last = None
    for i, (fidx, frame) in enumerate(loader.frames(start=start_frame, count=n_frames)):
        dets = detector.detect(frame, fidx)
        res  = propagator.update(frame, exclude_dets=dets)
        last = res

        if i % 25 == 0:    # ~once per second
            vs = est.estimate(mgr_static)
            vp = est.estimate(mgr_prop)
            sp_s = himg.project_to_image(centre_spot, mgr_static.H)
            sp_p = himg.project_to_image(centre_spot, mgr_prop.H)
            t = args.start + i / loader.fps
            cs = f"({vs.centroid_m[0]:.0f},{vs.centroid_m[1]:.0f})" if vs else "--"
            cp = f"({vp.centroid_m[0]:.0f},{vp.centroid_m[1]:.0f})" if vp else "--"
            ss = f"({sp_s[0]:.0f},{sp_s[1]:.0f})" if sp_s else "--"
            sps = f"({sp_p[0]:.0f},{sp_p[1]:.0f})" if sp_p else "--"
            print(f"{t:>5.0f}s | {cs:>16} {ss:>14} | {cp:>16} {sps:>14} | {res.n_inliers:>7}")

    loader.close()
    s = propagator.summary()
    print("-" * 92)
    print(f"Propagation: {s['updates']} updates, {s['holds']} holds, {s['cuts']} cuts")
    print("\nRead: STATIC centroid stays frozen (H can't see the pan); PROP centroid")
    print("moves with the camera. The centre-spot px is the visual check — PROP keeps")
    print("the spot near its true on-screen location, STATIC slides away as it pans.")


if __name__ == "__main__":
    main()
