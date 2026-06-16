"""Geometry helpers used across the pipeline."""

import numpy as np
from typing import Tuple

Point = Tuple[float, float]


def dist_m(a: Point, b: Point) -> float:
    """Euclidean distance between two pitch coordinates in metres."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b))


def angle_deg(v1: Point, v2: Point) -> float:
    """Angle in degrees between two 2D vectors."""
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))


def cosine_similarity(v1: Point, v2: Point) -> float:
    """Cosine similarity between two 2D vectors. Range [-1, 1]."""
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))


def pitch_zone(x_m: float, pitch_length: float = 105.0) -> str:
    """Classify pitch x-coordinate into a third."""
    third = pitch_length / 3.0
    if x_m < third:
        return "defensive_third"
    elif x_m < 2 * third:
        return "middle_third"
    else:
        return "attacking_third"
