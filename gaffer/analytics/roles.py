"""
gaffer/analytics/roles.py
────────────────────────────
Role identification — turns formation.py's depth-based lines (defense /
midfield / attack, grouped purely by how far up the pitch a player sits)
into specific positional labels: LB, LCB, RCB, RB, DM, LW, ST, ...

Track 168, Track 198, Track 230 mean nothing to a human. LB, DM, RW do.
That's the entire point of this module — everything downstream (pass
network, HUD, future match reports) should read in roles, not track_ids.

How a line becomes labels
──────────────────────────
Each FormationLine already has the right COUNT of players (that's the
formation string, e.g. the "4" in "4-3-3"). This module orders those
players left-to-right and looks up the name for that slot count in
_ROLE_TABLES (a 4-player defense line is LB/LCB/RCB/RB; a 3-player one is
LCB/CB/RCB). Counts with no table entry (5+ in a line is almost always
ID-tracking noise, not a real back five split across two depth lines) fall
back to numbered generic labels so this never just throws an unlabeled gap.

Left/right convention (heuristic — same caveat as engine.py's attack_dir)
───────────────────────────────────────────────────────────────────────────
There's no ground-truth signal (e.g. which touchline the broadcast camera
favours) to anchor "left" against, so this picks a self-consistent
convention rather than a verified one: a team's own left is the +y side of
the pitch when it attacks toward x=105, and the -y side when it attacks
toward x=0. Equivalently, sort by `y * attack_dir` descending. Treat left/
right labels as approximate, like def_line_m.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from gaffer.analytics.formation import TeamFormation

_ROLE_TABLES: dict[str, dict[int, list[str]]] = {
    "defense": {
        1: ["CB"],
        2: ["LCB", "RCB"],
        3: ["LCB", "CB", "RCB"],
        4: ["LB", "LCB", "RCB", "RB"],
        5: ["LB", "LCB", "CB", "RCB", "RB"],
    },
    "midfield": {
        1: ["DM"],
        2: ["LCM", "RCM"],
        3: ["LCM", "CM", "RCM"],
        4: ["LM", "LCM", "RCM", "RM"],
        5: ["LM", "LCM", "CM", "RCM", "RM"],
    },
    "attack": {
        1: ["ST"],
        2: ["LS", "RS"],
        3: ["LW", "ST", "RW"],
        4: ["LW", "LS", "RS", "RW"],
    },
}
_FALLBACK_PREFIX = {"defense": "DEF", "midfield": "MID", "attack": "FWD"}
_KNOWN_LABELS = {label for table in _ROLE_TABLES.values()
                       for labels in table.values() for label in labels} | {"GK"}


@dataclass
class PlayerRole:
    track_id: int
    role:     str   # "GK" | "LB" | "DM" | "RW" | ... or a numbered fallback
    line:     str   # "goalkeeper" | "defense" | "midfield" | "attack"


def assign_roles(formation: TeamFormation, attack_dir: int) -> dict[int, PlayerRole]:
    """
    track_id -> PlayerRole for every player in `formation`, ordered
    left-to-right within each line via `formation.avg_positions` (the same
    rolling-window average formation.py used to build the lines — using it
    here too keeps role assignment as stable as the line split itself,
    rather than reacting to single-frame jitter).

    The goalkeeper isn't in any line (formation.py excludes it from line
    clustering on purpose), but avg_positions still has its track_id — it's
    whichever sampled team track_id never made it into a line.
    """
    roles: dict[int, PlayerRole] = {}

    outfield_ids = {tid for line in formation.lines for tid in line.track_ids}
    for tid in formation.avg_positions:
        if tid not in outfield_ids:
            roles[tid] = PlayerRole(tid, "GK", "goalkeeper")

    for line in formation.lines:
        ordered = sorted(
            line.track_ids,
            key=lambda tid: _lateral_signed(formation.avg_positions.get(tid), attack_dir),
            reverse=True,   # most-left first, matching how the role tables read
        )
        for tid, label in zip(ordered, _role_labels(line.role, len(ordered))):
            roles[tid] = PlayerRole(tid, label, line.role)

    return roles


class RoleTracker:
    """
    Smooths assign_roles() output over each track's own lifetime.

    A line's player COUNT (and therefore which role table applies, and a
    player's rank within it) depends on which OTHER track_ids currently
    clear formation.py's recency/min-samples bar — and those churn fast
    (ByteTrack reassigns IDs every few seconds in this footage). So a
    player who hasn't moved at all can still flip between LCB/CB/DEF4 every
    frame purely because a teammate's track flickered in or out of the
    line. Majority-voting each track's own raw-role history irons that out:
    a label only changes once a different one is the outright majority for
    THAT track_id, not merely tied or momentarily ahead.

    Roles are remembered for the track's entire lifetime, even once it
    stops appearing — a match-long pass network needs to label senders and
    receivers from minutes ago, long after their tracks have churned away.
    """

    def __init__(self):
        self._hist:  dict[int, Counter] = {}
        self._line:  dict[int, str] = {}
        self._stable: dict[int, str] = {}

    def update(self, raw_roles: dict[int, PlayerRole]) -> dict[int, PlayerRole]:
        """Feed one frame's raw assign_roles() output; returns the smoothed
        roles for exactly the track_ids present this frame (for live HUD use)."""
        for tid, role in raw_roles.items():
            counts = self._hist.setdefault(tid, Counter())
            counts[role.role] += 1
            self._line[tid] = role.line
            top_label, top_n = counts.most_common(1)[0]
            if tid not in self._stable or top_n > sum(counts.values()) / 2:
                self._stable[tid] = top_label
        return {tid: PlayerRole(tid, self._stable[tid], self._line[tid]) for tid in raw_roles}

    def all_known(self) -> dict[int, PlayerRole]:
        """Every track_id ever seen and its best-known role — for relabeling
        a match-long pass network whose senders/receivers may be long gone."""
        return {tid: PlayerRole(tid, label, self._line.get(tid, ""))
                for tid, label in self._stable.items()}


def label_pass_network(
    pass_network: dict[tuple[int, int], int],
    roles: dict[int, PlayerRole],
) -> dict[tuple[str, str], int]:
    """
    Re-key a (sender_id, receiver_id) -> count pass network by role label,
    e.g. {(168, 198): 3} -> {("LB", "DM"): 3}. Track_ids with no known role,
    OR whose only known role is a numbered fallback (DEF7, MID3, ...), fall
    back to "#track_id" instead — fallback labels are a frame-local rank
    index, not a stable identity, so two unrelated tracks can each end up
    "winning" e.g. MID3 at different points in the match. Trusting them here
    would silently merge two different players into one node. "#track_id"
    is an honest "don't know" rather than a wrong-but-confident-looking one.
    Track_ids that share a REAL role label (one player's ID churned
    mid-match) still collapse into one edge — that's the readable behaviour
    for a human-facing network, not a bug.
    """
    out: dict[tuple[str, str], int] = {}
    for (sender, receiver), count in pass_network.items():
        s = _label_or_unknown(sender, roles)
        r = _label_or_unknown(receiver, roles)
        out[(s, r)] = out.get((s, r), 0) + count
    return out


def label_for(track_id: int, roles: dict[int, PlayerRole]) -> str:
    """Public single-id version of the honest-fallback rule label_pass_network
    uses: a curated position name, or "#track_id" if only a frame-local
    fallback (or nothing) is known."""
    return _label_or_unknown(track_id, roles)


# ── Internal ──────────────────────────────────────────────────────────────────

def _label_or_unknown(track_id: int, roles: dict[int, PlayerRole]) -> str:
    role = roles.get(track_id)
    if role is None or role.role not in _KNOWN_LABELS:
        return f"#{track_id}"
    return role.role


def _lateral_signed(pos: tuple[float, float] | None, attack_dir: int) -> float:
    if pos is None:
        return 0.0
    return pos[1] * attack_dir


def _role_labels(line_role: str, n: int) -> list[str]:
    table = _ROLE_TABLES.get(line_role, {})
    if n in table:
        return table[n]
    prefix = _FALLBACK_PREFIX.get(line_role, "PLR")
    return [f"{prefix}{i + 1}" for i in range(n)]
