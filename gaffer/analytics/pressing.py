"""Pressing intensity computation."""

import numpy as np
from scipy.spatial import cKDTree

from gaffer.config import PRESSING_RADIUS_M


def compute_pressing_intensity(
    ball_pos: tuple[float, float],
    ball_team: str,
    all_positions: dict[str, list[tuple[float, float]]],
    radius_m: float = PRESSING_RADIUS_M,
) -> dict:
    """
    Count opponents within radius_m of the ball carrier.

    Args:
        ball_pos: (x, y) in metres
        ball_team: "teamA" or "teamB"
        all_positions: {"teamA": [(x,y), ...], "teamB": [(x,y), ...]}
        radius_m: pressing radius in metres (default 10m)

    Returns:
        intensity: int — opponents within radius, capped at 5
        nearest_dist_m: float | None — distance to nearest opponent
    """
    opposing = "teamB" if ball_team == "teamA" else "teamA"
    pressing_positions = all_positions.get(opposing, [])

    if not pressing_positions:
        return {"intensity": 0, "nearest_dist_m": None}

    tree = cKDTree(pressing_positions)
    in_radius = tree.query_ball_point(ball_pos, r=radius_m)
    intensity = min(len(in_radius), 5)

    nearest_dist, _ = tree.query(ball_pos, k=1)

    return {
        "intensity": intensity,
        "nearest_dist_m": round(float(nearest_dist), 2),
    }
