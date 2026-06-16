"""Voronoi pitch control computation."""

import numpy as np
from scipy.spatial import Voronoi

from gaffer.config import PITCH_LENGTH_M, PITCH_WIDTH_M


def compute_voronoi_control(
    team_a_positions: np.ndarray,
    team_b_positions: np.ndarray,
    pitch_w: float = PITCH_LENGTH_M,
    pitch_h: float = PITCH_WIDTH_M,
) -> dict:
    """
    Compute Voronoi-based pitch control.

    Returns:
        teamA_pct: float — percentage of pitch controlled by Team A
        teamB_pct: float — percentage of pitch controlled by Team B
        cells: list of (team_label, polygon_vertices) tuples
    """
    team_a_positions = np.asarray(team_a_positions, dtype=float)
    team_b_positions = np.asarray(team_b_positions, dtype=float)

    if len(team_a_positions) == 0 or len(team_b_positions) == 0:
        return {"teamA_pct": 50.0, "teamB_pct": 50.0, "cells": []}

    n_a = len(team_a_positions)
    all_positions = np.vstack([team_a_positions, team_b_positions])

    # Mirror boundary points to prevent infinite cells at pitch edges
    boundary = _boundary_mirror_points(pitch_w, pitch_h)
    extended = np.vstack([all_positions, boundary])

    vor = Voronoi(extended)

    team_a_area = 0.0
    team_b_area = 0.0
    cells = []

    for i, (pt, region_idx) in enumerate(zip(all_positions, vor.point_region)):
        if i >= len(all_positions):
            break
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            continue
        vertices = vor.vertices[region]
        # Clamp vertices to pitch boundaries (numpy, no shapely dependency)
        clipped = _clip_polygon_to_pitch(vertices, pitch_w, pitch_h)
        if clipped is None or len(clipped) < 3:
            continue
        area = _polygon_area(clipped)
        team = "teamA" if i < n_a else "teamB"
        if team == "teamA":
            team_a_area += area
        else:
            team_b_area += area
        cells.append((team, clipped))

    total = team_a_area + team_b_area
    if total < 1e-6:
        return {"teamA_pct": 50.0, "teamB_pct": 50.0, "cells": cells}

    return {
        "teamA_pct": round(100.0 * team_a_area / total, 2),
        "teamB_pct": round(100.0 * team_b_area / total, 2),
        "cells": cells,
    }


def _boundary_mirror_points(pitch_w: float, pitch_h: float) -> np.ndarray:
    """Generate mirror points just outside the pitch boundary."""
    margin = 10.0
    pts = []
    for x in np.linspace(0, pitch_w, 10):
        pts.append([x, -margin])
        pts.append([x, pitch_h + margin])
    for y in np.linspace(0, pitch_h, 7):
        pts.append([-margin, y])
        pts.append([pitch_w + margin, y])
    return np.array(pts)


def _clip_polygon_to_pitch(vertices: np.ndarray, pw: float, ph: float) -> np.ndarray | None:
    """Clip polygon vertices to [0,pw] x [0,ph] bounding box (Sutherland-Hodgman)."""
    poly = vertices.tolist()
    for boundary in [
        ("left",   lambda p: p[0] >= 0,  lambda a, b: _intersect_x(a, b, 0.0)),
        ("right",  lambda p: p[0] <= pw, lambda a, b: _intersect_x(a, b, pw)),
        ("bottom", lambda p: p[1] >= 0,  lambda a, b: _intersect_y(a, b, 0.0)),
        ("top",    lambda p: p[1] <= ph, lambda a, b: _intersect_y(a, b, ph)),
    ]:
        _, inside, intersect = boundary
        output = []
        if not poly:
            return None
        for i in range(len(poly)):
            curr = poly[i]
            prev = poly[i - 1]
            if inside(curr):
                if not inside(prev):
                    output.append(intersect(prev, curr))
                output.append(curr)
            elif inside(prev):
                output.append(intersect(prev, curr))
        poly = output

    if len(poly) < 3:
        return None
    return np.array(poly)


def _intersect_x(a, b, x):
    t = (x - a[0]) / (b[0] - a[0] + 1e-12)
    return [x, a[1] + t * (b[1] - a[1])]


def _intersect_y(a, b, y):
    t = (y - a[1]) / (b[1] - a[1] + 1e-12)
    return [a[0] + t * (b[0] - a[0]), y]


def _polygon_area(vertices: np.ndarray) -> float:
    """Shoelace formula for polygon area."""
    v = np.asarray(vertices)
    n = len(v)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += v[i, 0] * v[j, 1]
        area -= v[j, 0] * v[i, 1]
    return abs(area) / 2.0
