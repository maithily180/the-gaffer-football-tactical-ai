"""
gaffer/output/analytics_overlay.py
────────────────────────────────────
Renders an AnalyticsSnapshot onto the video frame:

  • Stats panel (top-left)  — per-team compactness, defensive line, pressing,
    plus a possession bar.
  • Voronoi pitch inset     — the 2D pitch filled with each team's controlled
    cells (space control), players dotted on top.

Pure rendering; consumes the dataclasses produced by PitchAnalyticsEngine.
"""

from __future__ import annotations

import cv2
import numpy as np

from gaffer import config
from gaffer.analytics.engine import AnalyticsSnapshot
from gaffer.calibration.pitch_model import PitchModel

_A_CLR = config.TEAM_A_COLOR_BGR     # red
_B_CLR = config.TEAM_B_COLOR_BGR     # blue
_HUD   = (0, 220, 60)                # green
_GREY  = (160, 160, 160)


class AnalyticsOverlay:
    def __init__(self, pitch_model: PitchModel | None = None):
        self.pm = pitch_model or PitchModel()
        self._base_pitch = self.pm.draw_pitch()

    # ── Public API ────────────────────────────────────────────────────────────

    def render(
        self,
        frame: np.ndarray,
        snap: AnalyticsSnapshot | None,
        *,
        voronoi_width: int = 360,
    ) -> np.ndarray:
        """Return a copy of `frame` with the stats panel and Voronoi inset drawn."""
        out = frame.copy()
        if snap is None:
            cv2.putText(out, "ANALYTICS: no calibration", (16, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 220), 2, cv2.LINE_AA)
            return out
        self._draw_panel(out, snap)
        self._draw_voronoi_inset(out, snap, voronoi_width)
        return out

    # ── Stats panel ───────────────────────────────────────────────────────────

    def _draw_panel(self, frame: np.ndarray, snap: AnalyticsSnapshot) -> None:
        a, b = snap.team_a, snap.team_b
        ca, cb = a.compactness, b.compactness
        poss = snap.possession

        def _line(label: str, va, vb) -> str:
            return f"{label:<11}{va:>9}{vb:>9}"

        lines = [
            ("HDR", "                 TEAM A   TEAM B"),
            ("SEP", ""),
            ("TXT", _line("Players",   ca.n_players,                cb.n_players)),
            ("TXT", _line("Width m",   f"{ca.width_m:.0f}",         f"{cb.width_m:.0f}")),
            ("TXT", _line("Depth m",   f"{ca.length_m:.0f}",        f"{cb.length_m:.0f}")),
            ("TXT", _line("Area m2",   f"{ca.hull_area_m2:.0f}",    f"{cb.hull_area_m2:.0f}")),
            ("TXT", _line("Def line",  _fmt(a.def_line_m),          _fmt(b.def_line_m))),
            ("TXT", _line("Control%",  f"{snap.voronoi['teamA_pct']:.0f}",
                                       f"{snap.voronoi['teamB_pct']:.0f}")),
            ("SEP", ""),
            ("TXT", _line("Possess%",  f"{poss.pct_a:.0f}",         f"{poss.pct_b:.0f}")),
        ]
        if snap.pressing is not None:
            owner_lbl = "A" if poss.owner == "teamA" else "B"
            lines.append(("TXT",
                f"Press: {snap.pressing['intensity']} on {owner_lbl}  "
                f"(near {snap.pressing['nearest_dist_m']}m)"))

        pad, line_h = 10, 20
        box_w = 290
        box_h = pad * 2 + line_h * len(lines)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for i, (kind, text) in enumerate(lines):
            y = 10 + pad + (i + 1) * line_h - 4
            if kind == "SEP":
                continue
            clr = _HUD if kind == "HDR" else (230, 230, 230)
            cv2.putText(frame, text, (10 + pad, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, clr, 1, cv2.LINE_AA)

        # Possession bar under the panel
        self._draw_possession_bar(frame, 10 + pad, 10 + box_h + 6,
                                  box_w - 2 * pad, poss.pct_a, poss.pct_b)

    def _draw_possession_bar(self, frame, x, y, w, pct_a, pct_b) -> None:
        h = 14
        wa = int(w * pct_a / 100.0)
        cv2.rectangle(frame, (x, y), (x + wa, y + h), _A_CLR, -1)
        cv2.rectangle(frame, (x + wa, y), (x + w, y + h), _B_CLR, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
        cv2.putText(frame, f"{pct_a:.0f}%", (x + 4, y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{pct_b:.0f}%", (x + w - 34, y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Voronoi inset ─────────────────────────────────────────────────────────

    def _draw_voronoi_inset(self, frame: np.ndarray, snap: AnalyticsSnapshot,
                            width: int) -> None:
        pitch = self._render_voronoi_pitch(snap)
        h0, w0 = pitch.shape[:2]
        height = int(width * h0 / w0)
        pitch = cv2.resize(pitch, (width, height), interpolation=cv2.INTER_AREA)

        H, W = frame.shape[:2]
        margin = 12
        x2, y2 = W - margin, margin + height
        x1, y1 = x2 - width, margin
        if x1 < 0 or y2 > H:
            return
        roi = frame[y1:y2, x1:x2]
        cv2.addWeighted(pitch, 0.85, roi, 0.15, 0, roi)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    def _render_voronoi_pitch(self, snap: AnalyticsSnapshot) -> np.ndarray:
        img = self._base_pitch.copy()
        cells = snap.voronoi.get("cells", [])

        if cells:
            fill = img.copy()
            for team, poly_m in cells:
                pts = np.array(
                    [self.pm.pitch_to_pixels(x, y) for (x, y) in poly_m],
                    dtype=np.int32,
                )
                clr = config.VORONOI_A_COLOR if team == "teamA" else config.VORONOI_B_COLOR
                cv2.fillPoly(fill, [pts], clr)
            cv2.addWeighted(fill, 0.5, img, 0.5, 0, img)

        # Player dots on top
        for team, positions in snap.positions.items():
            clr = _A_CLR if team == "teamA" else _B_CLR
            for (x, y) in positions:
                cx, cy = self.pm.pitch_to_pixels(x, y)
                cv2.circle(img, (cx, cy), 6, clr, -1)
                cv2.circle(img, (cx, cy), 6, (255, 255, 255), 1)

        if snap.ball_xy is not None:
            bx, by = self.pm.pitch_to_pixels(*snap.ball_xy)
            cv2.circle(img, (bx, by), 5, config.BALL_COLOR_BGR, -1)
            cv2.circle(img, (bx, by), 5, (0, 0, 0), 1)

        return img


def _fmt(v: float | None) -> str:
    return "--" if v is None else f"{v:.0f}"
