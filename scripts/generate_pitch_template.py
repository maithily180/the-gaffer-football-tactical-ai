"""
Generate data/pitch_template.png
=================================
Draws a standard 105x68m football pitch as a 1050x680 pixel PNG.
White lines on green background, proportional to real dimensions.

Run once:
    uv run python scripts/generate_pitch_template.py
"""

import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUTPUT_PATH = ROOT / "data" / "pitch_template.png"

# Canvas size — 1050x680 pixels = 10 pixels per metre
SCALE = 10  # px per metre
W = int(105 * SCALE)   # 1050
H = int(68 * SCALE)    # 680
MARGIN = 40            # pixels of padding around pitch

CANVAS_W = W + 2 * MARGIN
CANVAS_H = H + 2 * MARGIN

LINE_COLOR = (255, 255, 255)
PITCH_GREEN = (34, 139, 34)   # BGR forest green
LINE_THICKNESS = 2


def m(metres: float) -> int:
    """Convert metres to pixels."""
    return int(metres * SCALE)


def px(x_m: float, y_m: float):
    """Convert pitch coordinates (metres) to canvas pixel coordinates."""
    return (MARGIN + m(x_m), MARGIN + m(y_m))


def draw_pitch(canvas: np.ndarray) -> np.ndarray:
    # Outer boundary
    cv2.rectangle(canvas, px(0, 0), px(105, 68), LINE_COLOR, LINE_THICKNESS)

    # Halfway line
    cv2.line(canvas, px(52.5, 0), px(52.5, 68), LINE_COLOR, LINE_THICKNESS)

    # Centre circle (radius 9.15m)
    centre = px(52.5, 34)
    cv2.circle(canvas, centre, m(9.15), LINE_COLOR, LINE_THICKNESS)

    # Centre spot
    cv2.circle(canvas, centre, 3, LINE_COLOR, -1)

    # ── Left penalty area (16.5m deep, 40.32m wide centred) ───────────────
    cv2.rectangle(canvas, px(0, 13.84), px(16.5, 54.16), LINE_COLOR, LINE_THICKNESS)

    # Left six-yard box (5.5m deep, 18.32m wide centred)
    cv2.rectangle(canvas, px(0, 24.84), px(5.5, 43.16), LINE_COLOR, LINE_THICKNESS)

    # Left penalty spot (11m from goal line)
    cv2.circle(canvas, px(11, 34), 3, LINE_COLOR, -1)

    # Left penalty arc (radius 9.15m from penalty spot, outside penalty area)
    cv2.ellipse(canvas, px(11, 34), (m(9.15), m(9.15)),
                0, -53, 53, LINE_COLOR, LINE_THICKNESS)

    # Left goal (7.32m wide, 2.44m deep — shown as rectangle beyond line)
    cv2.rectangle(canvas, px(-2, 30.34), px(0, 37.66), LINE_COLOR, LINE_THICKNESS)

    # ── Right penalty area ─────────────────────────────────────────────────
    cv2.rectangle(canvas, px(88.5, 13.84), px(105, 54.16), LINE_COLOR, LINE_THICKNESS)

    # Right six-yard box
    cv2.rectangle(canvas, px(99.5, 24.84), px(105, 43.16), LINE_COLOR, LINE_THICKNESS)

    # Right penalty spot
    cv2.circle(canvas, px(94, 34), 3, LINE_COLOR, -1)

    # Right penalty arc
    cv2.ellipse(canvas, px(94, 34), (m(9.15), m(9.15)),
                0, 127, 233, LINE_COLOR, LINE_THICKNESS)

    # Right goal
    cv2.rectangle(canvas, px(105, 30.34), px(107, 37.66), LINE_COLOR, LINE_THICKNESS)

    # ── Corner arcs (radius 1m) ────────────────────────────────────────────
    cv2.ellipse(canvas, px(0, 0),   (m(1), m(1)), 0,   0,  90, LINE_COLOR, LINE_THICKNESS)
    cv2.ellipse(canvas, px(105, 0), (m(1), m(1)), 0,  90, 180, LINE_COLOR, LINE_THICKNESS)
    cv2.ellipse(canvas, px(0, 68),  (m(1), m(1)), 0, 270, 360, LINE_COLOR, LINE_THICKNESS)
    cv2.ellipse(canvas, px(105, 68),(m(1), m(1)), 0, 180, 270, LINE_COLOR, LINE_THICKNESS)

    return canvas


def main():
    canvas = np.full((CANVAS_H, CANVAS_W, 3), PITCH_GREEN, dtype=np.uint8)
    canvas = draw_pitch(canvas)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_PATH), canvas)
    print(f"Pitch template saved: {OUTPUT_PATH}")
    print(f"Size: {CANVAS_W}x{CANVAS_H} pixels  ({W}x{H} pitch + {MARGIN}px margin)")
    print(f"Scale: {SCALE} px/m")


if __name__ == "__main__":
    main()
