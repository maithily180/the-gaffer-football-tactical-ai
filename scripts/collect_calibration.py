"""
scripts/collect_calibration.py
──────────────────────────────
Interactive 4–8 point pitch calibration collector (run by a human, once).

Homography is extremely sensitive to point placement, so we click landmarks
rather than read coordinates off a gridded image. This opens one video frame in
an OpenCV window and walks you through the named pitch landmarks; click each one
that is visible, skip the rest. Output is a JSON of {landmark_name: [px, py]}.
World coordinates are looked up from config.PITCH_KEYPOINTS later (in the
notebook), so only pixel positions are stored here.

Usage:
    uv run python scripts/collect_calibration.py data/test_clips/tactical_playlist_1.mp4 --time 65

Controls:
    left-click : place the current landmark (shown top-left) and advance
    s          : skip current landmark (not visible in this frame)
    u          : undo last placed point
    Enter / q  : finish and save
    Esc        : abort without saving

Pick a frame where well-spread landmarks are visible (centre circle + a touchline
+ a penalty area). Aim for >= 6 points, NOT all clustered in one area.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from gaffer import config
from gaffer.calibration.homography import HomographyEstimator
from gaffer.video.loader import VideoLoader

GATE_PX = 10.0   # reprojection-error gate (image pixels)

# Order landmarks so widely-separated, non-collinear, easy-to-identify points
# come FIRST. The default 55s frame of tactical_playlist_1 is the LEFT penalty
# area, so left-box landmarks lead. CRITICAL: do not click only goal-line points
# (x=0) — they are collinear and give a degenerate H. The box FRONT corners
# (x=16.5), six-yard front corners (x=5.5) and penalty spot (x=11) give the
# x-spread that makes H well-conditioned.
LANDMARK_ORDER = [
    "left_penalty_spot",                                    # x=11  (breaks collinearity)
    "left_penalty_box_tr", "left_penalty_box_br",           # x=16.5 front corners
    "left_six_yard_tr", "left_six_yard_br",                 # x=5.5  front corners
    "left_arc_apex",                                        # x=20.15 (furthest spread)
    "left_penalty_box_tl", "left_penalty_box_bl",           # x=0 goal-line box corners
    "left_six_yard_tl", "left_six_yard_bl",                 # x=0 goal-line six-yard
    "left_corner_top", "left_corner_bottom",
    "left_goal_center",
    # midfield / right side (for differently-framed clips)
    "center_spot", "halfway_line_top", "halfway_line_bottom",
    "right_penalty_spot", "right_penalty_box_tl", "right_penalty_box_tr",
    "right_penalty_box_br", "right_penalty_box_bl",
    "right_six_yard_tl", "right_six_yard_tr", "right_six_yard_br", "right_six_yard_bl",
    "right_arc_apex", "right_corner_top", "right_corner_bottom", "right_goal_center",
]


def _grab_frame(clip: str, time_s: float):
    with VideoLoader(clip) as v:
        frame_idx = int(time_s * v.fps)
        v.seek(frame_idx)
        ok, frame = v.read()
        if not ok:
            raise RuntimeError(f"Could not read frame at {time_s}s (idx {frame_idx})")
        return frame, frame_idx, v.fps


def main():
    ap = argparse.ArgumentParser(description="Collect manual pitch calibration points")
    ap.add_argument("clip")
    ap.add_argument("--time", type=float, default=65.0, help="Frame time in seconds")
    ap.add_argument("--out", default=None, help="Output JSON (default data/calibration/<clip>.json)")
    args = ap.parse_args()

    landmarks = [n for n in LANDMARK_ORDER if n in config.PITCH_KEYPOINTS]
    frame, frame_idx, fps = _grab_frame(args.clip, args.time)

    out_path = Path(args.out) if args.out else \
        config.DATA_DIR / "calibration" / f"{Path(args.clip).stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    points: dict[str, list[int]] = {}
    order: list[str] = []        # landmarks placed, for undo
    idx = {"i": 0}               # index into `landmarks`

    win = "calibration  |  click landmark  ·  s=skip  u=undo  Enter/q=save  Esc=abort"

    def redraw():
        disp = frame.copy()
        for name, (px, py) in points.items():
            cv2.circle(disp, (px, py), 5, (0, 255, 255), -1)
            cv2.putText(disp, name, (px + 6, py - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        target = landmarks[idx["i"]] if idx["i"] < len(landmarks) else "(all done — Enter to save)"
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(disp, f"CLICK: {target}    [{len(points)} placed]", (8, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(win, disp)

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and idx["i"] < len(landmarks):
            name = landmarks[idx["i"]]
            points[name] = [int(x), int(y)]
            order.append(name)
            idx["i"] += 1
            redraw()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, ord("q")):          # Enter / q → save
            break
        if key == 27:                       # Esc → abort
            print("Aborted — nothing saved.")
            cv2.destroyAllWindows()
            return
        if key == ord("s") and idx["i"] < len(landmarks):
            idx["i"] += 1
            redraw()
        if key == ord("u") and order:
            last = order.pop()
            points.pop(last, None)
            # step back to the earliest unplaced landmark
            idx["i"] = min(idx["i"], landmarks.index(last))
            redraw()

    cv2.destroyAllWindows()

    if len(points) < 4:
        print(f"Only {len(points)} points — need >= 4 for a homography. Not saved.")
        return

    # ── Immediate verification: compute H + reprojection gate ────────────────
    names_used = list(points.keys())
    image_pts = np.array([points[n] for n in names_used], dtype=np.float32)
    world_pts = np.array([config.PITCH_KEYPOINTS[n] for n in names_used], dtype=np.float32)

    est = HomographyEstimator()
    H, valid = est.compute(image_pts, world_pts)

    print(f"\n{'='*60}")
    print(f"  Calibration check — {len(points)} points, frame {frame_idx}")
    print(f"{'='*60}")
    if H is None:
        print("  Homography FAILED — points are collinear or degenerate.")
        print("  Re-run and click points spread across the pitch (include the")
        print("  box FRONT corners + penalty spot, not only goal-line points).")
        mean_e = max_e = None
    else:
        mean_e, max_e, errs = est.reprojection_error(image_pts, world_pts, H)
        print(f"  Homography valid : {valid}")
        print(f"  Mean reprojection: {mean_e:.2f} px")
        print(f"  Max  reprojection: {max_e:.2f} px")
        for n, e in zip(names_used, errs):
            tag = "  <-- worst (consider re-clicking)" if e == max_e else ""
            print(f"    {n:<22} {e:6.2f} px{tag}")
        verdict = "PASS — geometry layer is trustworthy" if mean_e < GATE_PX \
            else f"FAIL — mean >= {GATE_PX:.0f}px; re-click the worst offenders"
        print(f"  GATE (<{GATE_PX:.0f}px): {verdict}")
    print(f"{'='*60}")

    payload = {
        "clip": Path(args.clip).name,
        "frame_idx": frame_idx,
        "fps": fps,
        "image_points": points,                 # {landmark_name: [px, py]} — source of truth
        "homography": H.tolist() if H is not None else None,
        "reprojection_mean_px": mean_e,
        "reprojection_max_px": max_e,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved → {out_path}")
    if H is not None and mean_e is not None and mean_e < GATE_PX:
        print("Next: run notebooks/05_homography.ipynb for the visual checks (Views A/B/C),")
        print("then tell me and I'll build the minimap.")
    else:
        print("Re-run with better-spread / more careful clicks before proceeding.")


if __name__ == "__main__":
    main()
