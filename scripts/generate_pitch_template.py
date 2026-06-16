"""
Generate data/pitch_template.png
=================================
Thin wrapper around PitchModel.draw_pitch() — the drawing lives in
gaffer/calibration/pitch_model.py so the template, minimap and coordinate
conversions all share one definition.

Run once (commit the PNG):
    uv run python scripts/generate_pitch_template.py
"""

import cv2

from gaffer import config
from gaffer.calibration.pitch_model import PitchModel


def main():
    model = PitchModel()
    canvas = model.draw_pitch()

    out = config.PITCH_TEMPLATE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)

    w, h = model.canvas_size
    print(f"Pitch template saved: {out}")
    print(f"Size: {w}x{h} px   scale: {model.scale} px/m   margin: {model.margin} px")


if __name__ == "__main__":
    main()
