"""
gaffer/analytics/compactness.py
────────────────────────────────
Team shape geometry — pure maths on pitch-coordinate (metre) positions.

All inputs are lists of (x_m, y_m) where:
    x : 0 = left goal line .. 105 = right goal line   (length axis)
    y : 0 = top touchline   ..  68 = bottom touchline  (width axis)

Outputs answer "how is this team arranged in space?" with no football opinion
attached — interpretation (compact block vs stretched possession shape) is left
to higher layers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    from scipy.spatial import ConvexHull
    from scipy.spatial.qhull import QhullError
except Exception:                       # pragma: no cover
    ConvexHull = None
    class QhullError(Exception):
        pass


@dataclass
class Compactness:
    n_players:    int
    centroid:     tuple[float, float]    # (x_m, y_m), (0,0) if no players
    length_m:     float                  # x-span (goal-to-goal stretch)
    width_m:      float                  # y-span (lateral stretch across pitch)
    hull_area_m2: float                  # convex-hull area of outfield shape
    spread_m:     float                  # mean distance of players from centroid

    def as_dict(self) -> dict:
        return {
            "n_players":    self.n_players,
            "centroid":     (round(self.centroid[0], 1), round(self.centroid[1], 1)),
            "length_m":     round(self.length_m, 1),
            "width_m":      round(self.width_m, 1),
            "hull_area_m2": round(self.hull_area_m2, 0),
            "spread_m":     round(self.spread_m, 1),
        }


def team_compactness(positions: list[tuple[float, float]]) -> Compactness:
    """
    Compute team shape metrics from outfield positions in pitch metres.

    Returns a zeroed Compactness when fewer than 1 position is supplied.
    Hull area falls back to the bounding-box area when there are < 3 points
    or the points are collinear (degenerate hull).
    """
    pts = np.asarray(positions, dtype=float)
    if pts.ndim != 2 or len(pts) == 0:
        return Compactness(0, (0.0, 0.0), 0.0, 0.0, 0.0, 0.0)

    cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    length = float(pts[:, 0].max() - pts[:, 0].min())
    width  = float(pts[:, 1].max() - pts[:, 1].min())
    spread = float(np.mean(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)))

    hull_area = _hull_area(pts, length, width)

    return Compactness(
        n_players    = len(pts),
        centroid     = (cx, cy),
        length_m     = length,
        width_m      = width,
        hull_area_m2 = hull_area,
        spread_m     = spread,
    )


def _hull_area(pts: np.ndarray, length: float, width: float) -> float:
    """Convex-hull area; bounding-box fallback for degenerate inputs."""
    if ConvexHull is not None and len(pts) >= 3:
        try:
            return float(ConvexHull(pts).volume)   # 2-D hull "volume" == area
        except QhullError:
            pass
        except Exception:
            pass
    return length * width
