"""
gaffer/analytics/space_control.py
────────────────────────────────────
Refines raw Voronoi cells into actionable territorial control: overall %,
control by third, and control by zone (the same 3x5 third x lane grid as
overload.py).

overload.py answers "who outnumbers whom, where" with a cheap nearest-player
headcount per zone — good enough to fire every detection frame.  This module
answers "who actually controls the space, where" by clipping the real
Voronoi cell polygons (from voronoi.compute_voronoi_control) against each
zone rectangle and summing the clipped area per team.  That's the difference
between "5 attackers near a corner" and "5 attackers controlling 40% of the
attacking third" — the latter is what actually matters tactically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gaffer import config
from gaffer.analytics.overload import LANE_NAMES
from gaffer.analytics.voronoi import clip_polygon_to_rect, polygon_area

_N_THIRDS_DEFAULT = 3
_N_LANES_DEFAULT  = 5


@dataclass
class ZoneControl:
    third_idx:     int
    lane_idx:      int
    lane_name:     str
    teamA_area_m2: float
    teamB_area_m2: float

    @property
    def teamA_pct(self) -> float:
        total = self.teamA_area_m2 + self.teamB_area_m2
        return round(100.0 * self.teamA_area_m2 / total, 1) if total > 1e-6 else 50.0

    @property
    def teamB_pct(self) -> float:
        return round(100.0 - self.teamA_pct, 1)


@dataclass
class SpaceControl:
    teamA_pct: float                          # overall pitch control %
    teamB_pct: float
    zones:     list[ZoneControl] = field(default_factory=list)
    by_third:  dict[int, tuple[float, float]] = field(default_factory=dict)  # third_idx -> (A%, B%)
    dominant_zone: ZoneControl | None = None  # most lopsided zone, for narrative


def compute_space_control(
    cells: list[tuple[str, np.ndarray]],
    n_thirds: int = _N_THIRDS_DEFAULT,
    n_lanes: int = _N_LANES_DEFAULT,
) -> SpaceControl | None:
    """
    cells: voronoi.compute_voronoi_control()['cells'] — list of (team, polygon_m).
    Clips every cell against each zone rectangle and sums clipped area.
    """
    if not cells:
        return None

    third_w = config.PITCH_LENGTH_M / n_thirds
    lane_w  = config.PITCH_WIDTH_M / n_lanes

    zone_areas: dict[tuple[int, int], list[float]] = {}
    total_a = total_b = 0.0

    for team, poly in cells:
        poly = np.asarray(poly, dtype=float)
        if len(poly) < 3:
            continue
        team_idx  = 0 if team == "teamA" else 1
        cell_area = polygon_area(poly)
        if team_idx == 0:
            total_a += cell_area
        else:
            total_b += cell_area

        xs, ys = poly[:, 0], poly[:, 1]
        ti_lo = max(int(xs.min() // third_w), 0)
        ti_hi = min(int(xs.max() // third_w), n_thirds - 1)
        li_lo = max(int(ys.min() // lane_w), 0)
        li_hi = min(int(ys.max() // lane_w), n_lanes - 1)

        for ti in range(ti_lo, ti_hi + 1):
            for li in range(li_lo, li_hi + 1):
                clipped = clip_polygon_to_rect(
                    poly, ti * third_w, (ti + 1) * third_w, li * lane_w, (li + 1) * lane_w
                )
                if clipped is None:
                    continue
                a = polygon_area(clipped)
                key = (ti, li)
                zone_areas.setdefault(key, [0.0, 0.0])[team_idx] += a

    total = total_a + total_b
    overall_a_pct = 100.0 * total_a / total if total > 1e-6 else 50.0

    zones: list[ZoneControl] = []
    for (ti, li), (a, b) in zone_areas.items():
        if a + b < 1e-6:
            continue
        lane_name = LANE_NAMES[li] if n_lanes == len(LANE_NAMES) else f"lane{li}"
        zones.append(ZoneControl(
            third_idx=ti, lane_idx=li, lane_name=lane_name,
            teamA_area_m2=a, teamB_area_m2=b,
        ))

    by_third = _control_by_third(zones, n_thirds)
    dominant = max(zones, key=lambda z: abs(z.teamA_pct - 50.0), default=None)

    return SpaceControl(
        teamA_pct=round(overall_a_pct, 1),
        teamB_pct=round(100.0 - overall_a_pct, 1),
        zones=zones,
        by_third=by_third,
        dominant_zone=dominant,
    )


def _control_by_third(
    zones: list[ZoneControl], n_thirds: int = _N_THIRDS_DEFAULT
) -> dict[int, tuple[float, float]]:
    """Aggregate zone areas across all lanes within each third."""
    sums: dict[int, list[float]] = {ti: [0.0, 0.0] for ti in range(n_thirds)}
    for z in zones:
        sums[z.third_idx][0] += z.teamA_area_m2
        sums[z.third_idx][1] += z.teamB_area_m2

    result: dict[int, tuple[float, float]] = {}
    for ti, (a, b) in sums.items():
        total = a + b
        if total < 1e-6:
            continue
        result[ti] = (round(100.0 * a / total, 1), round(100.0 * b / total, 1))
    return result
