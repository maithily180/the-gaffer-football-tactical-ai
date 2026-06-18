"""
gaffer/analytics/overload.py
───────────────────────────────
Numerical superiority by pitch zone — "4 vs 2 on the right wing."

Splits the pitch into a 3x5 grid: 3 lengthwise THIRDS (absolute x) x 5 width
LANES (left wing / left half-space / centre / right half-space / right wing).
For each occupied zone, counts how many of each team are currently standing
in it.

Deliberately does NOT use player roles or formation lines.  formation.py
needs sustained track continuity (a rolling window, a recency filter) that
breaks down under ByteTrack ID churn — useful for a 30-60s shape average,
not for catching a transient 3-second overload as it happens.  This module
only needs the CURRENT frame's positions, so it's robust to exactly the kind
of track instability that makes formation.py fragile, and it fires the
instant a numerical advantage appears rather than only after a player has
been tracked long enough to "count."
"""

from __future__ import annotations

from dataclasses import dataclass

from gaffer import config

LANE_NAMES = ["left_wing", "left_half_space", "centre", "right_half_space", "right_wing"]

_N_THIRDS_DEFAULT = 3
_N_LANES_DEFAULT  = 5


@dataclass
class ZoneOccupancy:
    third_idx:   int     # 0..n_thirds-1, absolute pitch x=0..105
    lane_idx:    int      # 0..n_lanes-1, absolute pitch y=0..68
    lane_name:   str
    teamA_count: int
    teamB_count: int

    @property
    def diff(self) -> int:
        """Positive => teamA advantage, negative => teamB advantage."""
        return self.teamA_count - self.teamB_count


def compute_overloads(
    team_a_pos: list[tuple[float, float]],
    team_b_pos: list[tuple[float, float]],
    n_thirds: int = _N_THIRDS_DEFAULT,
    n_lanes: int = _N_LANES_DEFAULT,
) -> list[ZoneOccupancy]:
    """Bucket both teams' positions into the third x lane grid."""
    third_w = config.PITCH_LENGTH_M / n_thirds
    lane_w  = config.PITCH_WIDTH_M / n_lanes

    counts: dict[tuple[int, int], list[int]] = {}
    for team_idx, positions in ((0, team_a_pos), (1, team_b_pos)):
        for (x, y) in positions:
            ti = min(max(int(x // third_w), 0), n_thirds - 1)
            li = min(max(int(y // lane_w), 0), n_lanes - 1)
            key = (ti, li)
            counts.setdefault(key, [0, 0])[team_idx] += 1

    zones: list[ZoneOccupancy] = []
    for ti in range(n_thirds):
        for li in range(n_lanes):
            a, b = counts.get((ti, li), [0, 0])
            if a == 0 and b == 0:
                continue
            lane_name = LANE_NAMES[li] if n_lanes == len(LANE_NAMES) else f"lane{li}"
            zones.append(ZoneOccupancy(
                third_idx=ti, lane_idx=li, lane_name=lane_name,
                teamA_count=a, teamB_count=b,
            ))
    return zones


def significant_overloads(
    zones: list[ZoneOccupancy], threshold: int = 2
) -> list[ZoneOccupancy]:
    """Zones where one team outnumbers the other by >= threshold players."""
    return [z for z in zones if abs(z.diff) >= threshold]


def third_label(third_idx: int, attack_dir: int, n_thirds: int = _N_THIRDS_DEFAULT) -> str:
    """
    Translate an absolute third index into "defensive"/"middle"/"attacking"
    relative to a team's attack direction.  Mirrors the same heuristic used
    by PitchAnalyticsEngine._defensive_line.
    """
    if attack_dir == 0:
        return "unknown"
    if attack_dir == +1:
        if third_idx == 0:
            return "defensive"
        if third_idx == n_thirds - 1:
            return "attacking"
        return "middle"
    else:
        if third_idx == 0:
            return "attacking"
        if third_idx == n_thirds - 1:
            return "defensive"
        return "middle"


def zone_centre_m(
    third_idx: int, lane_idx: int,
    n_thirds: int = _N_THIRDS_DEFAULT, n_lanes: int = _N_LANES_DEFAULT,
) -> tuple[float, float]:
    """Pitch-metre centre of a zone, for display/event location purposes."""
    third_w = config.PITCH_LENGTH_M / n_thirds
    lane_w  = config.PITCH_WIDTH_M / n_lanes
    return ((third_idx + 0.5) * third_w, (lane_idx + 0.5) * lane_w)
