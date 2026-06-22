"""
gaffer/analyst/retrieval.py
─────────────────────────────
Step 2: given a classified question, pull ONLY the relevant slice of a
MatchBundle. Nothing here computes new analytics -- it filters and counts
what gaffer/analytics/ already produced (match_report's event counts are
match-wide only, so per-team breakdowns go through the small
_count_by_team() helper below rather than inventing a new tracker).

If a filtered slice comes back empty -- e.g. an event type that simply
never fired in this match, which v1.E found true of COUNTER_ATTACK,
PROGRESSIVE_PASS and DOMINANCE on both calibrated clips -- retrieval
returns an explicit {"empty": True, "reason": ...} marker instead of a
silently empty list, so evidence/answer generation can treat "this didn't
happen" as a real, expected answer rather than a retrieval bug.
"""

from __future__ import annotations

from gaffer.analyst.match_bundle import MatchBundle
from gaffer.analyst.question_types import QuestionType
from gaffer.analytics.episodes import Episode
from gaffer.events.base import DOMINANCE, LINE_BREAK, OVERLOAD, PASS, FootballEvent

_SUPPORTING_OUTCOMES = {"Counter", "Line Break", "Attacking Third Entry"}


def retrieve(bundle: MatchBundle, qtype: QuestionType, *,
             team: str | None = None, event_type: str | None = None) -> dict:
    if qtype is QuestionType.UNSUPPORTED:
        # classify() passes the human-readable reason through the event_type
        # slot for this one question type -- see question_types.py docstring.
        return {"empty": True, "reason": event_type or "Gaffer doesn't track this"}
    if qtype is QuestionType.MATCH_SUMMARY:
        return _retrieve_match_summary(bundle)
    if qtype is QuestionType.DOMINANCE_EXPLANATION:
        return _retrieve_dominance(bundle, team)
    if qtype is QuestionType.EVENT_SEARCH:
        return _retrieve_event_search(bundle, team, event_type)
    if qtype is QuestionType.PLAYER_INFLUENCE:
        return _retrieve_player_influence(bundle)
    return _retrieve_passing(bundle, team)


def _count_by_team(events: list[FootballEvent], event_type: str) -> dict[str, int]:
    counts = {"teamA": 0, "teamB": 0}
    for ev in events:
        if ev.event_type == event_type and ev.team in counts:
            counts[ev.team] += 1
    return counts


def _retrieve_match_summary(bundle: MatchBundle) -> dict:
    return {
        "match_report": bundle.match_report,
        "formation_a": bundle.formation_a,
        "formation_b": bundle.formation_b,
    }


def _retrieve_dominance(bundle: MatchBundle, team: str | None) -> dict:
    events = bundle.run.events
    overload = _count_by_team(events, OVERLOAD)
    line_break = _count_by_team(events, LINE_BREAK)
    dominance = _count_by_team(events, DOMINANCE)

    if sum(overload.values()) == 0 and sum(line_break.values()) == 0 and sum(dominance.values()) == 0:
        return {"empty": True, "reason": "no overload, line-break, or dominance events were recorded in this match"}

    teams = [team] if team else ["teamA", "teamB"]
    supporting: list[Episode] = [
        ep for ep in bundle.run.episodes
        if ep.team in teams and ep.outcome in _SUPPORTING_OUTCOMES
    ]
    supporting.sort(key=lambda ep: len(ep.events), reverse=True)
    sc = bundle.match_report.space_control_pct

    result = {
        "overload_counts":   overload,
        "line_break_counts": line_break,
        "dominance_counts":  dominance,
        "space_control_pct": sc,
        "supporting_episodes": supporting[:3],
    }

    # The question may presuppose a team "dominated" that the comparison
    # metrics don't actually back -- e.g. "why did Team B dominate?" asked
    # about a clip where Team A leads overloads, line breaks, AND space
    # control. Surface that mismatch explicitly instead of letting the LLM
    # answer a premise the evidence contradicts.
    if team is not None:
        other = "teamB" if team == "teamA" else "teamA"
        team_idx, other_idx = (0, 1) if team == "teamA" else (1, 0)
        team_leads = sum([
            overload[team] > overload[other],
            line_break[team] > line_break[other],
            sc[team_idx] > sc[other_idx],
        ])
        if team_leads < 2:
            label = {"teamA": "Team A", "teamB": "Team B"}
            result["premise_mismatch"] = (
                f"the evidence does not support that {label[team]} dominated -- "
                f"{label[other]} led on {3 - team_leads} of 3 comparison metrics "
                f"(overloads {overload[other]} vs {overload[team]}, "
                f"line breaks {line_break[other]} vs {line_break[team]}, "
                f"space control {sc[other_idx]:.0f}% vs {sc[team_idx]:.0f}%)"
            )

    return result


def _retrieve_event_search(bundle: MatchBundle, team: str | None, event_type: str | None) -> dict:
    if event_type is None:
        return {"empty": True, "reason": "couldn't tell which event type the question is asking about"}

    matches = [ev for ev in bundle.run.events
               if ev.event_type == event_type and (team is None or ev.team == team)]
    if not matches:
        suffix = f" for {team}" if team else ""
        return {"empty": True, "reason": f"{event_type} never occurred in this match{suffix}"}

    context_episodes = [
        ep for ep in bundle.run.episodes
        if (team is None or ep.team == team)
        and any(e.event_type == event_type for e in ep.events)
    ]

    return {
        "event_type": event_type,
        "team": team,
        "events": matches,
        "context_episodes": context_episodes[:3],
    }


def _retrieve_player_influence(bundle: MatchBundle) -> dict:
    pnr = bundle.pass_network_report
    if not pnr.hub_players:
        return {"empty": True, "reason": "no pass network data was recorded in this match"}
    return {
        "hub_players": pnr.hub_players,
        "progressive_leaders": pnr.progressive_leaders,
    }


def _retrieve_passing(bundle: MatchBundle, team: str | None) -> dict:
    pnr = bundle.pass_network_report
    if pnr.most_frequent is None and not pnr.longest_buildup:
        return {"empty": True, "reason": "no completed passes were recorded in this match"}
    return {
        "most_frequent": pnr.most_frequent,
        "longest_buildup": pnr.longest_buildup,
        "pass_counts": _count_by_team(bundle.run.events, PASS),
    }
