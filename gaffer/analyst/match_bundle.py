"""
gaffer/analyst/match_bundle.py
─────────────────────────────────
A single cached snapshot of everything the query engine needs to answer
questions about one match: the full event log, episodes, match report,
pass network report, and possession summary. All five come from ONE run
of gaffer.analysis.pipeline_runner.run_clip() driven by WorldModelV2 --
unlike the evaluation suite, the analyst doesn't compare world models, it
just wants the best available one.

Cached to gaffer/analyst/_cache/<clip_stem>.json using the same
serialize/cache-key pattern evaluation/generate_report.py already
validated, so re-asking questions about a clip already analyzed doesn't
repeat a multi-minute detection pass. Only the fields retrieval.py actually
reads are persisted (events, episodes, match_report, pass_network_report,
possession) -- the same "cache what's used, not everything" rule
generate_report.py's _CachedEvent/_CachedEpisode already follow.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gaffer.analysis.pipeline_runner import ClipRunResult, run_clip
from gaffer.analytics.episodes import Episode
from gaffer.analytics.match_report import MatchReport
from gaffer.analytics.pass_network_analytics import Connection, HubPlayer, PassNetworkReport
from gaffer.events.base import FootballEvent
from gaffer.tracking.world_model_v2 import WorldModelV2

CACHE_DIR = Path(__file__).parent / "_cache"


@dataclass
class MatchBundle:
    clip_name:           str
    run:                 ClipRunResult
    match_report:        MatchReport
    pass_network_report: PassNetworkReport
    possession:          dict
    formation_a:         str | None = None
    formation_b:         str | None = None


def build_bundle(clip_path: Path, calib_path: Path, *, force: bool = False) -> MatchBundle:
    clip_path = Path(clip_path)
    calib_path = Path(calib_path)
    cache_path = CACHE_DIR / f"{clip_path.stem}.json"

    if cache_path.exists() and not force:
        return _deserialize(json.loads(cache_path.read_text()))

    run = run_clip(clip_path, calib_path, world_model_cls=WorldModelV2)
    bundle = MatchBundle(
        clip_name=clip_path.stem,
        run=run,
        match_report=run.match_report,
        pass_network_report=run.pass_network_report,
        possession=run.possession,
        formation_a=run.formation_a,
        formation_b=run.formation_b,
    )
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path.write_text(json.dumps(_serialize(bundle), indent=2, default=_json_default))
    return bundle


def _json_default(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(f"not JSON serializable: {type(obj)}")


def _serialize_event(e: FootballEvent) -> dict:
    return {"event_type": e.event_type, "team": e.team, "time_s": e.time_s, "data": e.data}


def _deserialize_event(d: dict) -> FootballEvent:
    return FootballEvent(frame_idx=0, time_s=d["time_s"], event_type=d["event_type"],
                          team=d["team"], location_m=None, data=d["data"])


def _serialize(bundle: MatchBundle) -> dict:
    run, mr, pnr = bundle.run, bundle.match_report, bundle.pass_network_report
    return {
        "clip_name":   bundle.clip_name,
        "duration_s":  run.duration_s,
        "events":      [_serialize_event(e) for e in run.events],
        "episodes": [
            {
                "episode_id":          ep.episode_id,
                "team":                ep.team,
                "start_time_s":        ep.start_time_s,
                "end_time_s":          ep.end_time_s,
                "outcome":             ep.outcome,
                "distance_advanced_m": ep.distance_advanced_m,
                "possession_chain":    ep.possession_chain,
                "players":             sorted(ep.players),
                "events":              [_serialize_event(e) for e in ep.events],
            }
            for ep in run.episodes
        ],
        "match_report": {
            "possession_pct":    list(mr.possession_pct),
            "space_control_pct": list(mr.space_control_pct),
            "total_passes":      mr.total_passes,
            "progressive_passes": mr.progressive_passes,
            "line_breaks":       mr.line_breaks,
            "overloads":         mr.overloads,
            "dominance_periods": mr.dominance_periods,
            "top_episode_ids":   [ep.episode_id for ep in mr.top_episodes],
        },
        "pass_network_report": {
            "most_frequent": (
                {"sender": pnr.most_frequent.sender, "receiver": pnr.most_frequent.receiver,
                 "count": pnr.most_frequent.count} if pnr.most_frequent else None
            ),
            "progressive_leaders": [
                {"sender": c.sender, "receiver": c.receiver, "count": c.count}
                for c in pnr.progressive_leaders
            ],
            "hub_players": [
                {"label": h.label, "degree": h.degree, "involvement": h.involvement,
                 "centrality": h.centrality}
                for h in pnr.hub_players
            ],
            "longest_buildup": pnr.longest_buildup,
        },
        "possession": bundle.possession,
        "formation_a": bundle.formation_a,
        "formation_b": bundle.formation_b,
    }


def _deserialize(data: dict) -> MatchBundle:
    events = [_deserialize_event(e) for e in data["events"]]

    episodes = [
        Episode(
            episode_id=ep["episode_id"], team=ep["team"],
            start_time_s=ep["start_time_s"], end_time_s=ep["end_time_s"],
            events=[_deserialize_event(e) for e in ep["events"]],
            possession_chain=ep["possession_chain"], players=set(ep["players"]),
            outcome=ep["outcome"], distance_advanced_m=ep["distance_advanced_m"],
        )
        for ep in data["episodes"]
    ]
    episodes_by_id = {ep.episode_id: ep for ep in episodes}

    mr_d = data["match_report"]
    match_report = MatchReport(
        possession_pct=tuple(mr_d["possession_pct"]),
        space_control_pct=tuple(mr_d["space_control_pct"]),
        total_passes=mr_d["total_passes"],
        progressive_passes=mr_d["progressive_passes"],
        line_breaks=mr_d["line_breaks"],
        overloads=mr_d["overloads"],
        dominance_periods=mr_d["dominance_periods"],
        top_episodes=[episodes_by_id[eid] for eid in mr_d["top_episode_ids"] if eid in episodes_by_id],
    )

    pnr_d = data["pass_network_report"]
    mf = pnr_d["most_frequent"]
    pass_network_report = PassNetworkReport(
        most_frequent=Connection(mf["sender"], mf["receiver"], mf["count"]) if mf else None,
        progressive_leaders=[Connection(c["sender"], c["receiver"], c["count"])
                              for c in pnr_d["progressive_leaders"]],
        hub_players=[HubPlayer(h["label"], h["degree"], h["involvement"], h["centrality"])
                     for h in pnr_d["hub_players"]],
        longest_buildup=pnr_d["longest_buildup"],
    )

    formation_a = data.get("formation_a")
    formation_b = data.get("formation_b")
    run = ClipRunResult(
        clip_name=data["clip_name"], world_model_name="v2.0",
        n_frames=0, duration_s=data["duration_s"], fps=0.0, ball_metrics={},
        events=events, episodes=episodes,
        match_report=match_report, pass_network_report=pass_network_report,
        possession=data["possession"],
        formation_a=formation_a, formation_b=formation_b,
    )
    return MatchBundle(
        clip_name=data["clip_name"], run=run,
        match_report=match_report, pass_network_report=pass_network_report,
        possession=data["possession"],
        formation_a=formation_a, formation_b=formation_b,
    )
