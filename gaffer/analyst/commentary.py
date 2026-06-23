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

from pathlib import Path

from gaffer.analyst import answer_generator
from gaffer.analyst.match_bundle import MatchBundle
from gaffer.analytics.episodes import Episode
from gaffer.events.base import (
    COUNTER_ATTACK,
    DOMINANCE,
    HIGH_PRESS,
    LINE_BREAK,
    OVERLOAD,
    POSSESSION_RECOVERY,
    PROGRESSIVE_PASS,
)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "commentary.txt"

# Below this many grounded facts there's no sequence worth an LLM connecting,
# and a single clause ("win the ball back") gives a small model far too much
# room to invent a cause or aftermath -- the deterministic line is both safer
# and already complete, so the LLM only earns its keep at >= this many facts.
_LLM_MIN_FACTS = 2


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


def commentate_episode(ep: Episode, *, use_llm: bool = True) -> str:
    """Analyst commentary for one episode. With the LLM reachable it connects
    the grounded clauses into natural prose; otherwise it returns the
    deterministic join. Either way the output is built only from
    episode_facts(ep) -- the LLM never sees raw events or invents play."""
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
    prompt = _PROMPT_PATH.read_text(encoding="utf-8").format(meta=meta, facts=fact_block)
    try:
        text = answer_generator.chat(prompt)
        return text or _deterministic(ep, facts)
    except Exception:
        # Ollama not running / unreachable -- grounded deterministic fallback,
        # never a hard failure or a hallucinated fill-in.
        return _deterministic(ep, facts)


def commentate_match(bundle: MatchBundle, *, use_llm: bool = True,
                     notable_only: bool = False) -> list[tuple[Episode, str]]:
    """Per-episode commentary across the whole match in chronological order.
    notable_only drops quiet (no-highlight) possessions, for a tighter
    highlight-style narration."""
    episodes = sorted(bundle.run.episodes, key=lambda e: e.start_time_s)
    out: list[tuple[Episode, str]] = []
    for ep in episodes:
        if notable_only and not episode_facts(ep):
            continue
        out.append((ep, commentate_episode(ep, use_llm=use_llm)))
    return out
