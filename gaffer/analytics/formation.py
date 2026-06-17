"""
gaffer/analytics/formation.py
────────────────────────────────
Turns player dots into football roles.

Until now every player was just a coordinate in `positions`/`player_positions_m`.
This module maintains a rolling window of per-player pitch positions and
clusters outfield players into back/mid/forward LINES by depth from their own
goal — the same geometric idea a coach uses when looking at a formation
graphic: count how many players sit roughly level with each other.

Line clustering
────────────────
Depths are sorted, then split at the (n_lines - 1) largest gaps between
consecutive depths.  This is a simple, deterministic 1-D clustering — no
iterative algorithm needed, and it matches how formations actually look:
players bunch into rows with visible gaps between them, not a uniform spread.

Goalkeepers are excluded from line clustering (they sit deepest, near their
own goal, and would otherwise corrupt the back line count) using the
`player_is_gk` flag computed by PitchAnalyticsEngine.

Output is a TeamFormation: a formation string ("4-3-3"), per-line player
counts / average depth, and the average position per player over the window
— the latter is the data source for a future formation heatmap.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from gaffer import config

if TYPE_CHECKING:
    from gaffer.analytics.engine import AnalyticsSnapshot

_DEFAULT_WINDOW_S   = 30.0   # rolling window for averaging player positions
_MIN_SAMPLES        = 12     # detection frames a player must appear in to count (~1.4s
                              # at skip-3/25fps — ByteTrack ID lifetimes here are often
                              # shorter than the 3.6s a 30-sample threshold would need)
_MIN_OUTFIELD       = 6      # below this many sampled outfield players, no formation
_N_LINES            = 3      # defense / midfield / attack
_ROLE_NAMES         = ["defense", "midfield", "attack"]
_RECENCY_S          = 6.0    # a track must have a sample within this long of "now" to
                              # count — otherwise it's a ghost replaced by a newer ID
                              # (ByteTrack reassigns IDs faster than the window ages out)


@dataclass
class FormationLine:
    role:        str               # "defense" | "midfield" | "attack"
    n_players:   int
    avg_depth_m: float             # metres from this team's own goal
    track_ids:   list[int] = field(default_factory=list)


@dataclass
class TeamFormation:
    team:           str
    formation_str:  str                                  # e.g. "4-3-3"
    lines:          list[FormationLine]                   # ordered defense -> attack
    avg_positions:  dict[int, tuple[float, float]]        # track_id -> avg (x, y) in window
    n_outfield:     int                                    # sampled outfield players used


class FormationAnalyzer:
    """
    Stateful.  Call update() every detection frame with the latest
    AnalyticsSnapshot, then formation(team, attack_dir) at any point to get
    the current best estimate of that team's shape.
    """

    def __init__(
        self,
        window_s: float = _DEFAULT_WINDOW_S,
        fps: float = config.DEFAULT_FPS,
        min_samples: int = _MIN_SAMPLES,
        recency_s: float = _RECENCY_S,
    ):
        self._window_frames   = int(window_s * fps)
        self._min_samples     = min_samples
        self._recency_frames  = int(recency_s * fps)
        # track_id -> deque[(frame_idx, (x, y))], pruned to the rolling window
        self._hist:    dict[int, deque[tuple[int, tuple[float, float]]]] = {}
        self._team_of: dict[int, str]  = {}
        self._gk_of:   dict[int, bool] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, snap: "AnalyticsSnapshot") -> None:
        frame_idx = snap.frame_idx
        cutoff    = frame_idx - self._window_frames

        for tid, pos in snap.player_positions_m.items():
            hist = self._hist.setdefault(tid, deque())
            hist.append((frame_idx, pos))
            self._team_of[tid] = snap.player_teams.get(tid)
            self._gk_of[tid]   = snap.player_is_gk.get(tid, False)

        # Prune ALL histories to the rolling window every frame — not just the
        # ones updated this frame.  A track_id that drops out of view (ID switch,
        # occlusion) must age out of the window instead of lingering forever as
        # a ghost player; otherwise every ByteTrack ID switch permanently
        # inflates the line counts.
        stale: list[int] = []
        for tid, hist in self._hist.items():
            while hist and hist[0][0] < cutoff:
                hist.popleft()
            if not hist:
                stale.append(tid)
        for tid in stale:
            del self._hist[tid]
            self._team_of.pop(tid, None)
            self._gk_of.pop(tid, None)

    def formation(self, team: str, attack_dir: int) -> TeamFormation | None:
        """
        Best-estimate formation for `team`, given its current attack direction
        (+1 toward x=105, -1 toward x=0 — see PitchAnalyticsEngine).  Returns
        None if attack direction is unknown or too few outfield players have
        enough samples in the rolling window.
        """
        if attack_dir == 0:
            return None

        latest_frame = max((h[-1][0] for h in self._hist.values() if h), default=None)
        if latest_frame is None:
            return None
        recency_cutoff = latest_frame - self._recency_frames

        avg_pos: dict[int, tuple[float, float]] = {}
        outfield: list[tuple[int, float]] = []   # (track_id, depth_from_own_goal)

        for tid, hist in self._hist.items():
            if self._team_of.get(tid) != team or len(hist) < self._min_samples:
                continue
            if hist[-1][0] < recency_cutoff:
                continue   # stale — this track has already been replaced by a newer ID
            n  = len(hist)
            ax = sum(p[1][0] for p in hist) / n
            ay = sum(p[1][1] for p in hist) / n
            avg_pos[tid] = (ax, ay)
            if self._gk_of.get(tid):
                continue
            depth = ax if attack_dir == +1 else config.PITCH_LENGTH_M - ax
            outfield.append((tid, depth))

        if len(outfield) < _MIN_OUTFIELD:
            return None

        groups = _split_into_lines(outfield, _N_LINES)
        lines: list[FormationLine] = []
        for role, group in zip(_ROLE_NAMES, groups):
            if not group:
                continue
            depths = [d for _, d in group]
            lines.append(FormationLine(
                role        = role,
                n_players   = len(group),
                avg_depth_m = round(sum(depths) / len(depths), 1),
                track_ids   = [tid for tid, _ in group],
            ))

        formation_str = "-".join(str(l.n_players) for l in lines)

        return TeamFormation(
            team          = team,
            formation_str = formation_str,
            lines         = lines,
            avg_positions = avg_pos,
            n_outfield    = len(outfield),
        )


# ── Internal ──────────────────────────────────────────────────────────────────

def _split_into_lines(
    items: list[tuple[int, float]], n_lines: int
) -> list[list[tuple[int, float]]]:
    """
    Split (track_id, depth) pairs into n_lines groups by cutting at the
    (n_lines - 1) largest gaps in sorted depth.  Deterministic, no iteration.
    """
    pairs = sorted(items, key=lambda p: p[1])
    if len(pairs) <= n_lines:
        return [[p] for p in pairs]

    gaps = [(pairs[i + 1][1] - pairs[i][1], i) for i in range(len(pairs) - 1)]
    split_idxs = sorted(idx for _, idx in sorted(gaps, reverse=True)[: n_lines - 1])

    groups: list[list[tuple[int, float]]] = []
    start = 0
    for idx in split_idxs:
        groups.append(pairs[start: idx + 1])
        start = idx + 1
    groups.append(pairs[start:])
    return groups
