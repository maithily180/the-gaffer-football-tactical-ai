"""
gaffer/analyst/evidence.py
─────────────────────────────
Step 3: turn a retrieval dict into a small, deterministic, pre-formatted
EvidencePack the LLM can read without doing any arithmetic or inference of
its own.

Deliberately no `confidence` field on Evidence -- there is no real
uncertainty model anywhere in Gaffer that would back a number like that,
and inventing one would be exactly the kind of fabricated precision the
project has avoided everywhere else (E1's evaluation report explicitly
refuses to call filter counters "precision/recall" for the same reason).
`source` gives traceability back to the analytics module instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gaffer.analyst.question_types import QuestionType


@dataclass
class Evidence:
    title:  str
    value:  str    # pre-formatted -- the LLM never does arithmetic
    source: str    # e.g. "match_report.space_control_pct" -- audit trail


@dataclass
class EvidencePack:
    question:     str
    evidence:     list[Evidence] = field(default_factory=list)
    empty:        bool = False
    empty_reason: str | None = None

    def render_for_prompt(self) -> str:
        if self.empty:
            return f"(no evidence -- {self.empty_reason})"
        return "\n".join(f"{i}. {e.title}: {e.value}" for i, e in enumerate(self.evidence, start=1))


def build_evidence(question: str, qtype: QuestionType, retrieved: dict) -> EvidencePack:
    if retrieved.get("empty"):
        return EvidencePack(question=question, empty=True,
                             empty_reason=retrieved.get("reason", "no matching data"))
    # A 3B local model loses track of which fact answers the question once a
    # full match-summary evidence list (11+ items) is on the page -- seen in
    # practice answering "what formation" from unrelated Overload-episode
    # text instead of the Formation line that was right there. For a
    # question this narrow, hand it only the one relevant fact.
    if qtype is QuestionType.MATCH_SUMMARY and "formation" in question.lower():
        ev = _build_formation_only(retrieved)
        if ev is None:
            return EvidencePack(question=question, empty=True,
                                 empty_reason="not enough players were tracked to estimate a formation in this match")
        return EvidencePack(question=question, evidence=ev)
    return EvidencePack(question=question, evidence=_BUILDERS[qtype](retrieved))


def _build_formation_only(r: dict) -> list[Evidence] | None:
    fa, fb = r.get("formation_a"), r.get("formation_b")
    if not fa and not fb:
        return None
    return [Evidence(
        "Formation (rough estimate)",
        f"Team A {fa or 'unknown'} - Team B {fb or 'unknown'} -- based on a 30s rolling "
        "window of tracked players, may be incomplete or not sum to 11",
        "run.formation_a/formation_b",
    )]


def _build_match_summary(r: dict) -> list[Evidence]:
    mr = r["match_report"]
    ev = []
    fa, fb = r.get("formation_a"), r.get("formation_b")
    if fa or fb:
        ev.append(Evidence(
            "Formation (rough estimate)",
            f"Team A {fa or 'unknown'} - Team B {fb or 'unknown'} -- based on a 30s rolling "
            "window of tracked players, may be incomplete or not sum to 11",
            "run.formation_a/formation_b",
        ))
    ev += [
        Evidence("Possession", f"Team A {mr.possession_pct[0]:.0f}% - Team B {mr.possession_pct[1]:.0f}%",
                  "match_report.possession_pct"),
        Evidence("Space control", f"Team A {mr.space_control_pct[0]:.0f}% - Team B {mr.space_control_pct[1]:.0f}%",
                  "match_report.space_control_pct"),
        Evidence("Passes", str(mr.total_passes), "match_report.total_passes"),
        Evidence("Progressive passes", str(mr.progressive_passes), "match_report.progressive_passes"),
        Evidence("Line breaks", str(mr.line_breaks), "match_report.line_breaks"),
        Evidence("Overloads", str(mr.overloads), "match_report.overloads"),
        Evidence("Dominance periods", str(mr.dominance_periods), "match_report.dominance_periods"),
    ]
    for i, ep in enumerate(mr.top_episodes, start=1):
        ev.append(Evidence(f"Top episode #{i}",
                            f"{ep.team}: {ep.narrative()} ({ep.outcome}, {ep.duration_s}s)",
                            "match_report.top_episodes"))
    return ev


def _build_dominance(r: dict) -> list[Evidence]:
    oc, lb, dc = r["overload_counts"], r["line_break_counts"], r["dominance_counts"]
    sc = r["space_control_pct"]
    ev = []
    if r.get("premise_mismatch"):
        ev.append(Evidence("Premise check", r["premise_mismatch"], "retrieval.premise_check"))
    ev += [
        Evidence("Overloads",         f"Team A {oc['teamA']} - Team B {oc['teamB']}", "events.overload (by team)"),
        Evidence("Line breaks",       f"Team A {lb['teamA']} - Team B {lb['teamB']}", "events.line_break (by team)"),
        Evidence("Dominance periods", f"Team A {dc['teamA']} - Team B {dc['teamB']}", "events.dominance (by team)"),
        Evidence("Space control",     f"Team A {sc[0]:.0f}% - Team B {sc[1]:.0f}%", "match_report.space_control_pct"),
    ]
    for ep in r["supporting_episodes"]:
        ev.append(Evidence("Supporting episode", f"{ep.team}: {ep.narrative()} ({ep.outcome}, {ep.duration_s}s)",
                            "episodes"))
    return ev


def _build_event_search(r: dict) -> list[Evidence]:
    ev = [Evidence("Count", str(len(r["events"])), "events (filtered by type)")]
    for e in r["events"][:10]:
        ev.append(Evidence(f"{e.event_type} @ {e.time_s:.1f}s", e.label(), "events"))
    for ep in r["context_episodes"]:
        ev.append(Evidence("Context episode", f"{ep.team}: {ep.narrative()} ({ep.outcome})", "episodes"))
    return ev


def _build_time_window(r: dict) -> list[Evidence]:
    start_s, end_s = r["window"]
    ev = [Evidence("Time window", f"{start_s:.1f}s - {end_s:.1f}s", "retrieval.window")]
    events = r["events"]
    for e in events[:15]:
        ev.append(Evidence(f"{e.event_type} @ {e.time_s:.1f}s", e.label(), "events"))
    if len(events) > 15:
        ev.append(Evidence("Note", f"showing the first 15 of {len(events)} events in this window, sorted by time",
                            "evidence.truncation"))
    for ep in r["episodes"][:5]:
        ev.append(Evidence("Episode",
                            f"{ep.team}: {ep.narrative()} ({ep.outcome}, "
                            f"{ep.start_time_s:.1f}s-{ep.end_time_s:.1f}s)",
                            "episodes"))
    return ev


def _build_player_influence(r: dict) -> list[Evidence]:
    ev = []
    for h in r["hub_players"]:
        ev.append(Evidence(f"Hub player {h.label}",
                            f"degree {h.degree}, involvement {h.involvement}, centrality {h.centrality:.2f}",
                            "pass_network_report.hub_players"))
    for c in r["progressive_leaders"]:
        ev.append(Evidence(f"Progressive link {c.sender} -> {c.receiver}",
                            f"{c.count} progressive passes", "pass_network_report.progressive_leaders"))
    return ev


def _build_passing(r: dict) -> list[Evidence]:
    ev = []
    mf = r["most_frequent"]
    if mf is not None:
        ev.append(Evidence("Most frequent connection", f"{mf.sender} -> {mf.receiver} ({mf.count} passes)",
                            "pass_network_report.most_frequent"))
    if r["longest_buildup"]:
        ev.append(Evidence("Longest possession chain", " -> ".join(r["longest_buildup"]),
                            "pass_network_report.longest_buildup"))
    pc = r["pass_counts"]
    ev.append(Evidence("Passes", f"Team A {pc['teamA']} - Team B {pc['teamB']}", "events.pass (by team)"))
    return ev


_BUILDERS = {
    QuestionType.MATCH_SUMMARY:          _build_match_summary,
    QuestionType.DOMINANCE_EXPLANATION:  _build_dominance,
    QuestionType.EVENT_SEARCH:           _build_event_search,
    QuestionType.PLAYER_INFLUENCE:       _build_player_influence,
    QuestionType.PASSING_ANALYSIS:       _build_passing,
    QuestionType.TIME_WINDOW:            _build_time_window,
}
