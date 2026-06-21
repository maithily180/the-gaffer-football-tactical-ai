"""
evaluation/evaluate_networks.py
───────────────────────────────────
E3 — Pass network stability, WITHIN a single match (first half vs second
half of the same clip).

This is explicitly NOT a cross-broadcast check. The only team that recurs
across two different clips here is Newcastle (arsenal_newcastle_highlights /
psg_newcastle_tactical), and the second clip has no calibration yet --
calibrating it would unlock a true cross-match comparison as a follow-up.
What this DOES check: does the same match produce a recognizably similar
pass network in its second half as its first?

Diffing happens in raw track_id space (end-checkpoint minus mid-checkpoint),
because track_id is a stable key that never changes meaning even if a
track itself drops and a player gets re-identified with a new id later --
unlike role labels, which can be reassigned mid-match as RoleTracker learns
more. Only after the subtraction is done are both halves labeled, using the
SAME final (end-of-match) roles_known() mapping for both -- so any
relabeling that happened over the course of the match can't make the two
halves look more different than they actually are.
"""

from __future__ import annotations

from evaluation.clip_runner import ClipRunResult
from gaffer.analytics.roles import label_pass_network


def _diff_network(end: dict[tuple[int, int], int], mid: dict[tuple[int, int], int]) -> dict[tuple[int, int], int]:
    """Edges that happened strictly after the midpoint checkpoint."""
    out = dict(end)
    for edge, mid_count in mid.items():
        remaining = out.get(edge, 0) - mid_count
        if remaining > 0:
            out[edge] = remaining
        else:
            out.pop(edge, None)
    return out


def half_split(result: ClipRunResult) -> dict[str, dict[tuple[int, int], int]]:
    """First-half / second-half raw (track_id-keyed) pass networks, derived
    from the 0.5 and 1.0 checkpoints captured during a single run."""
    mid = result.pass_net_checkpoints.get(0.5, {})
    end = result.pass_net_checkpoints.get(1.0, {})
    return {"first_half": mid, "second_half": _diff_network(end, mid)}


def _hub_players(network: dict, top_n: int = 3) -> list[tuple]:
    """Touch count per player (sender + receiver), a cheap proxy for degree
    centrality -- enough to compare "who's central" between halves without
    needing the full pass_network_report() machinery."""
    touches: dict = {}
    for (sender, receiver), n in network.items():
        touches[sender] = touches.get(sender, 0) + n
        touches[receiver] = touches.get(receiver, 0) + n
    return sorted(touches.items(), key=lambda kv: kv[1], reverse=True)[:top_n]


def _top_edge(network: dict) -> tuple | None:
    return max(network.items(), key=lambda kv: kv[1])[0] if network else None


def stability(result: ClipRunResult) -> dict:
    halves = half_split(result)
    first_raw, second_raw = halves["first_half"], halves["second_half"]

    first = label_pass_network(first_raw, result.roles)
    second = label_pass_network(second_raw, result.roles)

    first_hubs, second_hubs = _hub_players(first), _hub_players(second)
    first_hub_set = {p for p, _ in first_hubs}
    second_hub_set = {p for p, _ in second_hubs}
    hub_overlap = (len(first_hub_set & second_hub_set) / len(first_hub_set)
                   if first_hub_set else None)

    first_edges, second_edges = set(first), set(second)
    union = first_edges | second_edges
    edge_jaccard = (len(first_edges & second_edges) / len(union)) if union else None

    return {
        "clip":                 result.clip_name,
        "first_half_top_edge":  _top_edge(first),
        "second_half_top_edge": _top_edge(second),
        "first_half_hubs":      first_hubs,
        "second_half_hubs":     second_hubs,
        "hub_overlap_pct":      round(100 * hub_overlap) if hub_overlap is not None else None,
        "edge_jaccard_pct":     round(100 * edge_jaccard) if edge_jaccard is not None else None,
        "first_half_n_edges":   len(first_edges),
        "second_half_n_edges":  len(second_edges),
    }


def evaluate(results: list[ClipRunResult]) -> list[dict]:
    """results: one v2.0 ClipRunResult per calibrated clip (reuses E1's runs --
    no new detection pass needed)."""
    return [stability(r) for r in results]


def _fmt_pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.0f}%"


def _fmt_edge(edge: tuple | None) -> str:
    return "n/a" if edge is None else f"{edge[0]} -> {edge[1]}"


def render_section(stabilities: list[dict]) -> str:
    lines = [
        "| Clip | Top edge (1st half) | Top edge (2nd half) | Hub overlap | Edge overlap (Jaccard) |",
        "|---|---|---|---|---|",
    ]
    for s in stabilities:
        lines.append(
            f"| {s['clip']} | {_fmt_edge(s['first_half_top_edge'])} | "
            f"{_fmt_edge(s['second_half_top_edge'])} | {_fmt_pct(s['hub_overlap_pct'])} | "
            f"{_fmt_pct(s['edge_jaccard_pct'])} |"
        )
    return "\n".join(lines)


def conclusion(stabilities: list[dict]) -> str:
    overlaps = [s["hub_overlap_pct"] for s in stabilities if s["hub_overlap_pct"] is not None]
    mean_overlap = sum(overlaps) / len(overlaps) if overlaps else None
    base = (
        f"Mean hub-player overlap between first and second half: {_fmt_pct(mean_overlap)}."
        if mean_overlap is not None else
        "Not enough pass-network data in either half to compare hubs."
    )
    return (
        base + " This is a within-match stability check (same broadcast, same two "
        "teams, split in time at the midpoint) -- NOT a cross-match check. The only "
        "team appearing in two different clips in this repo is Newcastle "
        "(arsenal_newcastle_highlights / psg_newcastle_tactical), and the second "
        "clip has no calibration yet; calibrating it would unlock a true "
        "cross-broadcast comparison as a follow-up."
    )


if __name__ == "__main__":
    from gaffer import config
    from evaluation.clip_runner import run_clip
    from gaffer.tracking.world_model_v2 import WorldModelV2

    clips = [
        (config.DATA_DIR / "test_clips" / "arsenal_newcastle_highlights.mp4",
         config.DATA_DIR / "calibration" / "arsenal_newcastle_highlights.json"),
        (config.DATA_DIR / "test_clips" / "tactical_playlist_1.mp4",
         config.DATA_DIR / "calibration" / "tactical_playlist_1.json"),
    ]
    results = [
        run_clip(clip, calib, world_model_cls=WorldModelV2, duration=20.0, checkpoint_fracs=(0.5, 1.0))
        for clip, calib in clips
    ]
    stabilities = evaluate(results)
    print(render_section(stabilities))
    print()
    print(conclusion(stabilities))
