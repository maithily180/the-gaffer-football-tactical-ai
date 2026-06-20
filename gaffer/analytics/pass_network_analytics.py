"""
gaffer/analytics/pass_network_analytics.py
─────────────────────────────────────────────
v1.7 — Pass network analytics + player influence.

passing.py answers "did a pass happen, and what were its raw properties."
This module answers questions about the SHAPE of the passing network
accumulated over the match so far:

    most_frequent       — which two roles exchange the ball most
    progressive_leaders — which connections actually advance play, not just
                           recycle it sideways/backwards
    hub_players         — who the team's structure depends on most, by
                           degree centrality (distinct teammates connected
                           to, normalised by the size of that player's own
                           passing network component)
    longest_buildup      — the longest uninterrupted possession chain so far,
                           in role labels

Stateless by design — everything it needs is already accumulated by
PassDetector and RoleTracker; this module only reads and ranks it. Labels
go through roles.label_pass_network()/label_for() so the same "honest
fallback" rule applies here as everywhere else: a numbered fallback role
(DEF7, MID3, ...) is a frame-local rank index, not a stable identity, so it
is never trusted as a network node — it renders as "#track_id" instead.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from gaffer.analytics.passing import PassEvent
from gaffer.analytics.roles import PlayerRole, label_for, label_pass_network


@dataclass
class Connection:
    sender:   str
    receiver: str
    count:    int


@dataclass
class HubPlayer:
    label:       str
    degree:      int      # distinct teammates exchanged a pass with
    involvement: int       # total passes sent + received
    centrality:  float     # degree / (own component size - 1), 0..1


@dataclass
class PassNetworkReport:
    most_frequent:       Connection | None
    progressive_leaders: list[Connection] = field(default_factory=list)
    hub_players:         list[HubPlayer]  = field(default_factory=list)
    longest_buildup:     list[str]        = field(default_factory=list)


def analyze(
    pass_network: dict[tuple[int, int], int],
    passes: list[PassEvent],
    sequences: list[list[int]],
    roles: dict[int, PlayerRole],
    top_n: int = 3,
) -> PassNetworkReport:
    labeled_network = label_pass_network(pass_network, roles)
    frequent = _top_connections(labeled_network, top_n)

    progressive_network = _progressive_network(passes)
    progressive_leaders = _top_connections(label_pass_network(progressive_network, roles), top_n)

    return PassNetworkReport(
        most_frequent       = frequent[0] if frequent else None,
        progressive_leaders = progressive_leaders,
        hub_players         = _hub_players(labeled_network, top_n),
        longest_buildup     = _longest_buildup(sequences, roles),
    )


# ── Internal ──────────────────────────────────────────────────────────────────

def _progressive_network(passes: list[PassEvent]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for p in passes:
        if not p.progressive:
            continue
        key = (p.sender_id, p.receiver_id)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _top_connections(labeled_network: dict[tuple[str, str], int], top_n: int) -> list[Connection]:
    ranked = sorted(labeled_network.items(), key=lambda kv: kv[1], reverse=True)
    return [Connection(s, r, count) for (s, r), count in ranked[:top_n]]


def _hub_players(labeled_network: dict[tuple[str, str], int], top_n: int) -> list[HubPlayer]:
    """Degree centrality over the passing graph (nodes = role labels, edges =
    distinct sender/receiver pairs).  Self-loops (track-id churn within one
    role) carry no network information and are skipped."""
    neighbors:   dict[str, set[str]] = {}
    involvement: Counter = Counter()
    for (s, r), count in labeled_network.items():
        if s == r:
            continue
        neighbors.setdefault(s, set()).add(r)
        neighbors.setdefault(r, set()).add(s)
        involvement[s] += count
        involvement[r] += count

    comp_size = {node: len(comp) for comp in _connected_components(neighbors) for node in comp}

    hubs = [
        HubPlayer(
            label       = node,
            degree      = len(team),
            involvement = involvement[node],
            centrality  = (len(team) / (comp_size[node] - 1) if comp_size[node] > 1 else 0.0),
        )
        for node, team in neighbors.items()
    ]
    hubs.sort(key=lambda h: (h.centrality, h.involvement), reverse=True)
    return hubs[:top_n]


def _connected_components(neighbors: dict[str, set[str]]) -> list[set[str]]:
    seen: set[str] = set()
    components: list[set[str]] = []
    for start in neighbors:
        if start in seen:
            continue
        comp, stack = set(), [start]
        while stack:
            node = stack.pop()
            if node in comp:
                continue
            comp.add(node)
            stack.extend(neighbors.get(node, ()) - comp)
        seen |= comp
        components.append(comp)
    return components


def _longest_buildup(sequences: list[list[int]], roles: dict[int, PlayerRole]) -> list[str]:
    if not sequences:
        return []
    longest = max(sequences, key=len)
    return [label_for(tid, roles) for tid in longest]
