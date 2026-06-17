"""
gaffer/calibration/pitch_visibility.py
───────────────────────────────────────
Problem A — "What part of the pitch am I looking at?"

Given the current homography and the frame size, work out which region of the
pitch is on screen, how much of it, and how far the camera has moved since
calibration.  This is deliberately separate from Problem B (keeping the minimap
accurate as the camera moves) — different objective, different evaluation.

Method (robust to the horizon)
──────────────────────────────
Projecting the four image corners is fragile: corners above the horizon line
project to garbage (near-infinite or sign-flipped coordinates).  Instead we
project a dense grid of image points through H, keep only those that land on the
pitch, and take the convex hull of the survivors.  Points above the horizon
simply fall off-pitch and are dropped — no special-casing needed.

Outputs (PitchVisibility)
─────────────────────────
    visible_polygon_m : convex hull of the on-screen pitch, in metres
    visible_area_m2   : area of that hull
    coverage_pct      : visible_area / full-pitch area
    centroid_m        : centroid of the visible region
    region            : dominant third the camera is centred on
    regions_visible   : every third the view touches
    n_points          : how many grid points landed on-pitch (confidence proxy)

Drift (bridge to Problem B)
───────────────────────────
Call set_reference() once on the calibration frame, then drift() on later
frames returns the centroid shift (metres) and area ratio.  A large shift means
the static H no longer matches the camera — the signal that triggers Step 2
(optical-flow homography propagation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from gaffer import config
from gaffer.calibration.homography_manager import HomographyManager


@dataclass
class PitchVisibility:
    visible_polygon_m: list[tuple[float, float]]
    visible_area_m2:   float
    coverage_pct:      float
    centroid_m:        tuple[float, float]
    region:            str
    regions_visible:   list[str]
    n_points:          int

    def as_dict(self) -> dict:
        return {
            "visible_area_m2": round(self.visible_area_m2, 0),
            "coverage_pct":    round(self.coverage_pct, 1),
            "centroid_m":      (round(self.centroid_m[0], 1), round(self.centroid_m[1], 1)),
            "region":          self.region,
            "regions_visible": self.regions_visible,
            "n_points":        self.n_points,
        }


class PitchVisibilityEstimator:
    def __init__(
        self,
        image_w: int,
        image_h: int,
        grid: int          = config.PITCH_VIS_GRID,
        margin_m: float    = config.PITCH_VIS_MARGIN_M,
        min_points: int    = config.PITCH_VIS_MIN_POINTS,
        pitch_l: float     = config.PITCH_LENGTH_M,
        pitch_w: float     = config.PITCH_WIDTH_M,
    ):
        self._w, self._h = image_w, image_h
        self._margin = margin_m
        self._min_pts = min_points
        self._pitch_l, self._pitch_w = pitch_l, pitch_w
        self._full_area = pitch_l * pitch_w

        # Pre-compute the grid of image sample points once.
        xs = np.linspace(0, image_w - 1, grid)
        ys = np.linspace(0, image_h - 1, grid)
        self._grid_px = [(float(x), float(y)) for y in ys for x in xs]

        self._ref: PitchVisibility | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def estimate(self, mgr: HomographyManager) -> PitchVisibility | None:
        """Project the image grid through H and summarise the visible pitch."""
        if not mgr.is_valid():
            return None

        on_pitch: list[tuple[float, float]] = []
        for px in self._grid_px:
            world = mgr.project(px)
            if world is None:
                continue
            x, y = world
            if (-self._margin <= x <= self._pitch_l + self._margin and
                    -self._margin <= y <= self._pitch_w + self._margin):
                on_pitch.append((x, y))

        if len(on_pitch) < self._min_pts:
            return None

        pts = np.array(on_pitch, dtype=float)
        # Clamp to the true pitch for area/region (margin was only for keeping pts)
        pts[:, 0] = np.clip(pts[:, 0], 0, self._pitch_l)
        pts[:, 1] = np.clip(pts[:, 1], 0, self._pitch_w)

        hull = self._convex_hull(pts)
        area = self._polygon_area(hull)
        cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        regions = self._regions_spanned(pts[:, 0].min(), pts[:, 0].max())
        dom = self._third_label(cx)

        return PitchVisibility(
            visible_polygon_m = [(float(x), float(y)) for x, y in hull],
            visible_area_m2   = area,
            coverage_pct      = 100.0 * area / self._full_area,
            centroid_m        = (cx, cy),
            region            = dom,
            regions_visible   = regions,
            n_points          = len(on_pitch),
        )

    def set_reference(self, vis: PitchVisibility) -> None:
        """Record the calibration-frame visibility for drift comparison."""
        self._ref = vis

    def drift(self, vis: PitchVisibility | None) -> dict | None:
        """
        Camera movement since the reference frame.
            centroid_shift_m : how far the view centre has moved (metres)
            area_ratio       : visible_area / reference_area (>1 zoom-out, <1 zoom-in)
        Returns None if no reference set or vis is None.
        """
        if self._ref is None or vis is None:
            return None
        dx = vis.centroid_m[0] - self._ref.centroid_m[0]
        dy = vis.centroid_m[1] - self._ref.centroid_m[1]
        return {
            "centroid_shift_m": round(math.hypot(dx, dy), 2),
            "area_ratio":       round(vis.visible_area_m2 / max(self._ref.visible_area_m2, 1e-6), 2),
        }

    def contains_pitch_point(self, vis: PitchVisibility, x_m: float, y_m: float) -> bool:
        """True if a pitch point falls inside the visible region (convex hull)."""
        return self._point_in_polygon((x_m, y_m), vis.visible_polygon_m)

    def region_of(self, x_m: float) -> str:
        """Third label for a pitch x-coordinate (left/middle/right)."""
        return self._third_label(x_m)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _third_label(self, x_m: float) -> str:
        third = self._pitch_l / 3.0
        if x_m < third:
            return "left_third"
        if x_m < 2 * third:
            return "middle_third"
        return "right_third"

    def _regions_spanned(self, x_min: float, x_max: float) -> list[str]:
        labels = []
        for lbl in ("left_third", "middle_third", "right_third"):
            if lbl == self._third_label(x_min) or lbl == self._third_label(x_max):
                labels.append(lbl)
        # Fill any middle thirds the span passes through
        order = ["left_third", "middle_third", "right_third"]
        i0, i1 = order.index(self._third_label(x_min)), order.index(self._third_label(x_max))
        return order[min(i0, i1): max(i0, i1) + 1]

    @staticmethod
    def _convex_hull(pts: np.ndarray) -> np.ndarray:
        """Andrew's monotone chain — no scipy dependency, handles small N."""
        uniq = np.unique(pts, axis=0)
        if len(uniq) < 3:
            return uniq
        p = uniq[np.lexsort((uniq[:, 1], uniq[:, 0]))]

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for q in p:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
                lower.pop()
            lower.append(q)
        upper = []
        for q in reversed(p):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
                upper.pop()
            upper.append(q)
        return np.array(lower[:-1] + upper[:-1])

    @staticmethod
    def _polygon_area(poly: np.ndarray) -> float:
        if len(poly) < 3:
            return 0.0
        x, y = poly[:, 0], poly[:, 1]
        return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2.0)

    @staticmethod
    def _point_in_polygon(pt: tuple[float, float], poly: list[tuple[float, float]]) -> bool:
        if len(poly) < 3:
            return False
        x, y = pt
        inside = False
        n = len(poly)
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and \
               (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
                inside = not inside
            j = i
        return inside
