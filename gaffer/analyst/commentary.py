"""
gaffer/analyst/commentary.py
─────────────────────────────────────
v3.1 — Commentary: turn an Episode into analyst-style prose grounded in the
events Gaffer actually detected. This is the project's original goal ("watch
football -> understand football -> talk about football") and it reuses the
whole analytics stack rather than adding new detection.

Two layers, by design:

  episode_facts(ep)            deterministic, grounded English clauses built
                               straight off ep.events / ep.outcome -- the
                               anti-hallucination guardrail. Nothing here is
                               invented; every clause traces to a real event's
                               .data (overload counts, line-break opponent,
                               counter distance, ...).

  commentate_episode(ep)       weaves those exact clauses into flowing prose.
                               If the local LLM (qwen2.5:3b via Ollama) is
                               reachable it does the connecting; if not, a
                               deterministic join is returned instead -- so
                               commentary still works with zero LLM and can
                               never narrate a fact that isn't in the clause
                               list. Same "LLM connects pre-filtered facts,
                               never sees raw data" contract as query_engine.

No new detection, analytics, or persisted state.
"""

from __future__ import annotations

import re
from pathlib import Path

from gaffer.analyst import answer_generator
from gaffer.analyst.match_bundle import MatchBundle
from gaffer.analytics.episodes import (
    OUTCOME_ATTACKING_THIRD_ENTRY,
    OUTCOME_COUNTER,
    OUTCOME_LINE_BREAK,
    OUTCOME_LOST_POSSESSION,
    OUTCOME_PRESS_SUCCESS,
    OUTCOME_SUSTAINED_POSSESSION,
    Episode,
)
from gaffer.events.base import (
    COUNTER_ATTACK,
    DOMINANCE,
    HIGH_PRESS,
    LINE_BREAK,
    OVERLOAD,
    POSSESSION_RECOVERY,
    PROGRESSIVE_PASS,
)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_STYLE_PROMPTS = {
    "broadcast": _PROMPTS_DIR / "commentary_broadcast.txt",
    "tactical": _PROMPTS_DIR / "commentary_tactical.txt",
    "casual": _PROMPTS_DIR / "commentary_casual.txt",
}

# Below this many grounded facts there's no sequence worth an LLM connecting,
# and a single clause ("win the ball back") gives a small model far too much
# room to invent a cause or aftermath -- the deterministic line is both safer
# and already complete, so the LLM only earns its keep at >= this many facts.
_LLM_MIN_FACTS = 2

# Mirrors episodes.py's own _classify() priority order (Counter > Line Break >
# Attacking Third Entry > Press Success > Sustained Possession > Lost
# Possession) -- the same ranking already used to pick an outcome is a
# reasonable proxy for how much it's worth interrupting the match to narrate.
_OUTCOME_BASE_SCORE = {
    OUTCOME_COUNTER:               5.0,
    OUTCOME_LINE_BREAK:            4.0,
    OUTCOME_ATTACKING_THIRD_ENTRY: 3.0,
    OUTCOME_PRESS_SUCCESS:         2.5,
    OUTCOME_SUSTAINED_POSSESSION:  2.0,
    OUTCOME_LOST_POSSESSION:       0.5,
}


def importance_score(ep: Episode) -> float:
    """How much this episode is worth interrupting the match to narrate --
    a derived value, not a persisted field, so it can change (e.g. retuned
    weights) without touching match_bundle.py's cache schema or forcing a
    pipeline rebuild on any clip. Built entirely from fields Episode already
    carries: the outcome's existing priority rank, how many distinct
    highlight beats happened (episode_facts(), already computed elsewhere),
    and how far the ball actually moved."""
    score = _OUTCOME_BASE_SCORE.get(ep.outcome, 1.0)
    score += 0.5 * len(episode_facts(ep))
    if ep.distance_advanced_m is not None and ep.distance_advanced_m > 0:
        score += min(ep.distance_advanced_m, 50.0) / 10.0
    return round(score, 2)


def _team(t: str | None) -> str:
    return {"teamA": "Team A", "teamB": "Team B"}.get(t or "", "the team")


def _clause(ev) -> str | None:
    """One grounded verb-phrase for a single highlight event, built from its
    real .data. Returns None for event types we don't narrate (so callers can
    just filter Nones out)."""
    d = ev.data or {}
    if ev.event_type == HIGH_PRESS:
        return "press high up the pitch"
    if ev.event_type == POSSESSION_RECOVERY:
        return "win the ball back"
    if ev.event_type == COUNTER_ATTACK:
        fwd = d.get("fwd_m")
        return (f"break on the counter, the ball surging {fwd:.0f}m toward goal"
                if fwd is not None else "break on the counter")
    if ev.event_type == OVERLOAD:
        lane = str(d.get("lane", "")).replace("_", " ")
        third = d.get("third", "")
        cf, ca = d.get("count_for"), d.get("count_against")
        sc = d.get("space_control_pct")
        where = f"in the {lane} of the {third} third" if lane and third else "in a key zone"
        base = f"work a {cf}v{ca} overload {where}" if cf is not None else f"overload {where}"
        return base + (f", controlling {sc:.0f}% of the space there" if sc is not None else "")
    if ev.event_type == LINE_BREAK:
        opp = _team(d.get("def_team"))
        return f"break through {opp}'s defensive line"
    if ev.event_type == PROGRESSIVE_PASS:
        dist = d.get("distance_m") or d.get("fwd_m")
        return f"drive forward with a progressive pass of {dist:.0f}m" if dist else "drive forward with a progressive pass"
    if ev.event_type == DOMINANCE:
        pct, dur = d.get("control_pct"), d.get("duration_s")
        if pct is not None and dur is not None:
            return f"establish sustained dominance, controlling {pct:.0f}% of the attacking third for {dur:.0f}s"
        return "establish sustained dominance"
    return None


def episode_facts(ep: Episode) -> list[str]:
    """Ordered, grounded clauses describing what happened in this episode --
    the exact, only material commentate_episode() is allowed to use. Repeated
    overloads (the same advantage logged frame after frame) collapse to the
    single most significant one so the story reads cleanly without inventing
    or dropping anything that matters."""
    overloads = [ev for ev in ep.events if ev.is_highlight and ev.event_type == OVERLOAD]
    best_overload = None
    if overloads:
        def _margin(ev):
            d = ev.data or {}
            return ((d.get("count_for") or 0) - (d.get("count_against") or 0),
                    d.get("space_control_pct") or 0)
        best_overload = max(overloads, key=_margin)

    facts: list[str] = []
    overload_emitted = False
    for ev in ep.events:
        if not ev.is_highlight:
            continue
        if ev.event_type == OVERLOAD:
            if overload_emitted:
                continue
            c = _clause(best_overload)
            overload_emitted = True
        else:
            c = _clause(ev)
        # Collapse consecutive duplicates -- e.g. a repeated HIGH_PRESS logged
        # over several frames is one story beat, not three.
        if c and (not facts or facts[-1] != c):
            facts.append(c)
    return facts


def _deterministic(ep: Episode, facts: list[str]) -> str:
    """Grounded prose with no LLM -- the always-available baseline and the
    fallback when Ollama is unreachable. Honest about quiet possessions
    rather than inventing drama."""
    subject = _team(ep.team)
    if not facts:
        return (f"{subject} keep possession for {ep.duration_s:.0f}s "
                f"without creating anything notable.")

    if len(facts) == 1:
        body = f"{subject} {facts[0]}."
    else:
        body = f"{subject} {facts[0]}, " + ", ".join(facts[1:-1] + [f"and {facts[-1]}"]) + "."

    tail = ""
    if ep.distance_advanced_m is not None and ep.distance_advanced_m >= 5:
        tail = f" The move carries the ball {ep.distance_advanced_m:.0f}m upfield over {ep.duration_s:.0f}s."
    return body + tail


# Event types Gaffer has no detector for at all -- their presence in LLM
# output is always fabrication, never a legitimate paraphrase of a real fact.
# Deliberately excludes "goal"/"goals": the COUNTER_ATTACK clause itself says
# "...toward goal" (a direction, not a scored goal), so banning it outright
# rejected correct, already-grounded text -- confirmed by hand the first time
# this check ran for real (the "tactical" style failed on a clean sentence
# purely because it echoed "toward goal" from its own source fact).
_BANNED_WORDS = re.compile(
    r"\b(shot|shots|score|scores|scored|scoring|foul|fouls|tackle|tackles|"
    r"save|saves|card|cards|penalty|penalties|offside|own half)\b",
    re.IGNORECASE,
)


def _passes_fact_check(text: str, facts: list[str], ep: Episode) -> bool:
    """Cheap, explicit guardrail on LLM output, run after generation --
    same spirit as v2.1's UNSUPPORTED keyword checks, not a second LLM call.
    Catches the two embellishment classes actually observed in v3.1/v3.2
    (invented events, loosened/invented numbers); does NOT catch a wrong
    number spelled out in words ("four" vs "5") -- a known gap in a
    deliberately cheap check, not an exhaustive validator."""
    if _BANNED_WORDS.search(text):
        return False
    allowed_numbers = set(re.findall(r"\d+", " ".join(facts)))
    allowed_numbers.add(str(round(ep.duration_s)))
    text_numbers = set(re.findall(r"\d+", text))
    return text_numbers.issubset(allowed_numbers)


def commentate_episode(ep: Episode, *, use_llm: bool = True, style: str = "broadcast",
                       context_line: str | None = None) -> str:
    """Analyst commentary for one episode. With the LLM reachable it connects
    the grounded clauses into natural prose in the requested register
    ("broadcast" | "tactical" | "casual"); otherwise it returns the
    deterministic join. Either way the output is built only from
    episode_facts(ep) -- the LLM never sees raw events or invents play.
    context_line, if given (see narrative_memory.py), is continuity flavor
    only -- a failed fact-check still falls back to the deterministic line,
    so a bad inference there can taint phrasing but never a stated fact."""
    facts = episode_facts(ep)
    if not use_llm or len(facts) < _LLM_MIN_FACTS:
        return _deterministic(ep, facts)

    fact_block = "\n".join(f"- {_team(ep.team)} {f}" for f in facts)
    # Deliberately NOT feeding ep.outcome here: a small model treats an outcome
    # label like "Lost Possession" as license to invent HOW it was lost (a
    # misplaced pass, a tackle) -- none of which Gaffer detected. The climax
    # events (counter, line break, ...) are already in the facts; the duration
    # is the only safe extra context.
    meta = f"Possession by {_team(ep.team)}, lasting {ep.duration_s:.0f}s"
    context_block = (
        f"\nContext (continuity/flavor only -- do not add detail from this, "
        f"state only what's in Facts): {context_line}\n" if context_line else "\n"
    )
    template_path = _STYLE_PROMPTS.get(style, _STYLE_PROMPTS["broadcast"])
    prompt = template_path.read_text(encoding="utf-8").format(
        meta=meta, facts=fact_block, context_block=context_block)
    try:
        text = answer_generator.chat(prompt)
        if text and _passes_fact_check(text, facts, ep):
            return text
        return _deterministic(ep, facts)
    except Exception:
        # Ollama not running / unreachable -- grounded deterministic fallback,
        # never a hard failure or a hallucinated fill-in.
        return _deterministic(ep, facts)


def commentate_match(bundle: MatchBundle, *, use_llm: bool = True,
                     notable_only: bool = False, min_importance: float = 0.0,
                     style: str = "broadcast", use_narrative_memory: bool = True
                     ) -> list[tuple[Episode, str]]:
    """Per-episode commentary across the whole match in chronological order.
    notable_only drops quiet (no-highlight) possessions; min_importance is
    the finer-grained version (see importance_score()) -- both apply if set.
    use_narrative_memory threads continuity context (narrative_memory.py)
    through episodes in order; only meaningful for a full chronological
    pass, so it's a flag here rather than always-on."""
    from gaffer.analyst.narrative_memory import NarrativeMemory  # local: avoid a module-level cycle

    episodes = sorted(bundle.run.episodes, key=lambda e: e.start_time_s)
    memory = NarrativeMemory() if use_narrative_memory else None
    out: list[tuple[Episode, str]] = []
    for ep in episodes:
        facts = episode_facts(ep)
        if notable_only and not facts:
            continue
        if importance_score(ep) < min_importance:
            continue
        context_line = memory.context_line() if memory else None
        text = commentate_episode(ep, use_llm=use_llm, style=style, context_line=context_line)
        if memory:
            memory.update(ep, facts)
        out.append((ep, text))
    return out
