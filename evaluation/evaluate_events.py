"""
evaluation/evaluate_events.py
────────────────────────────────
E2 — Tactical event validation: do OVERLOAD / DOMINANCE events actually
predict the outcomes their names claim, or are they just noise?

Two questions, one method:
    P(progressive pass | overload)        vs  P(progressive pass | no overload)
    P(attacking-third entry | dominance)   vs  P(...| no dominance)

Time is binned into fixed windows (default 5s) per team. A window is
"trigger-active" if the relevant event fired in it; "outcome-hit" if the
outcome happened in that window OR the next one (covers triggers that land
near a window boundary). This is a deliberate methodology choice, not a
hidden constant -- window_s is a parameter, and the report states it.
"""

from __future__ import annotations

import math

from gaffer.analysis.pipeline_runner import ClipRunResult
from gaffer.events.base import DOMINANCE, OVERLOAD, PROGRESSIVE_PASS

_DEFAULT_WINDOW_S = 5.0


def windowed_conditional_probability(
    trigger_times: dict[str, list[float]],
    outcome_times: dict[str, list[float]],
    duration_s: float,
    window_s: float = _DEFAULT_WINDOW_S,
) -> dict:
    """Bins [0, duration_s) into window_s windows per team. Returns
    P(outcome window | trigger window) vs P(outcome window | no trigger),
    per team and pooled across teams."""
    n_windows = max(1, math.ceil(duration_s / window_s))

    def _bin(times: list[float]) -> list[bool]:
        bins = [False] * n_windows
        for t in times:
            idx = int(t // window_s)
            if 0 <= idx < n_windows:
                bins[idx] = True
        return bins

    per_team: dict[str, dict] = {}
    pooled = {"trig_hit": 0, "trig_n": 0, "notrig_hit": 0, "notrig_n": 0}

    for team in sorted(set(trigger_times) | set(outcome_times)):
        trig = _bin(trigger_times.get(team, []))
        out  = _bin(outcome_times.get(team, []))
        trig_n = trig_hit = notrig_n = notrig_hit = 0
        for i in range(n_windows):
            hit = out[i] or (i + 1 < n_windows and out[i + 1])
            if trig[i]:
                trig_n += 1
                trig_hit += int(hit)
            else:
                notrig_n += 1
                notrig_hit += int(hit)
        per_team[team] = {
            "n_trigger_windows":          trig_n,
            "n_no_trigger_windows":       notrig_n,
            "n_trigger_hits":             trig_hit,
            "n_no_trigger_hits":          notrig_hit,
            "p_outcome_given_trigger":    (trig_hit / trig_n) if trig_n else None,
            "p_outcome_given_no_trigger": (notrig_hit / notrig_n) if notrig_n else None,
        }
        pooled["trig_hit"]   += trig_hit
        pooled["trig_n"]     += trig_n
        pooled["notrig_hit"] += notrig_hit
        pooled["notrig_n"]   += notrig_n

    return {
        "window_s": window_s,
        "per_team": per_team,
        "pooled": {
            "n_trigger_windows":          pooled["trig_n"],
            "n_no_trigger_windows":       pooled["notrig_n"],
            "n_trigger_hits":             pooled["trig_hit"],
            "n_no_trigger_hits":          pooled["notrig_hit"],
            "p_outcome_given_trigger":    (pooled["trig_hit"] / pooled["trig_n"]) if pooled["trig_n"] else None,
            "p_outcome_given_no_trigger": (pooled["notrig_hit"] / pooled["notrig_n"]) if pooled["notrig_n"] else None,
        },
    }


def _event_times_by_team(events, event_type: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {"teamA": [], "teamB": []}
    for ev in events:
        if ev.event_type == event_type and ev.team in out:
            out[ev.team].append(ev.time_s)
    return out


def evaluate(results: list[ClipRunResult], *, window_s: float = _DEFAULT_WINDOW_S) -> dict:
    """results: the v2.0 ClipRunResult for each clip (one per clip -- reuses
    the run already made for E1, no new run needed). Pools across clips by
    summing trigger/outcome window counts before computing probabilities."""
    overload_pool = {"trig_hit": 0, "trig_n": 0, "notrig_hit": 0, "notrig_n": 0}
    dominance_pool = {"trig_hit": 0, "trig_n": 0, "notrig_hit": 0, "notrig_n": 0}
    per_clip: list[dict] = []

    for r in results:
        overload_times = _event_times_by_team(r.events, OVERLOAD)
        prog_pass_times = _event_times_by_team(r.events, PROGRESSIVE_PASS)
        dominance_times = _event_times_by_team(r.events, DOMINANCE)

        overload_result = windowed_conditional_probability(
            overload_times, prog_pass_times, r.duration_s, window_s)
        dominance_result = windowed_conditional_probability(
            dominance_times, r.attacking_third_entries, r.duration_s, window_s)

        per_clip.append({
            "clip": r.clip_name,
            "overload_to_progressive_pass": overload_result,
            "dominance_to_attacking_third":  dominance_result,
        })
        for pool, res in ((overload_pool, overload_result), (dominance_pool, dominance_result)):
            pool["trig_n"]     += res["pooled"]["n_trigger_windows"]
            pool["notrig_n"]   += res["pooled"]["n_no_trigger_windows"]
            pool["trig_hit"]   += res["pooled"]["n_trigger_hits"]
            pool["notrig_hit"] += res["pooled"]["n_no_trigger_hits"]

    def _finalize(pool):
        return {
            "n_trigger_windows":          pool["trig_n"],
            "n_no_trigger_windows":       pool["notrig_n"],
            "n_trigger_hits":             pool["trig_hit"],
            "n_no_trigger_hits":          pool["notrig_hit"],
            "p_outcome_given_trigger":    (pool["trig_hit"] / pool["trig_n"]) if pool["trig_n"] else None,
            "p_outcome_given_no_trigger": (pool["notrig_hit"] / pool["notrig_n"]) if pool["notrig_n"] else None,
        }

    return {
        "window_s":   window_s,
        "per_clip":   per_clip,
        "overload_to_progressive_pass_pooled": _finalize(overload_pool),
        "dominance_to_attacking_third_pooled": _finalize(dominance_pool),
    }


def _fmt_p(p: float | None) -> str:
    return "n/a" if p is None else f"{100 * p:.0f}%"


def render_section(summary: dict) -> str:
    ov = summary["overload_to_progressive_pass_pooled"]
    do = summary["dominance_to_attacking_third_pooled"]
    lines = [
        f"Window size: {summary['window_s']:.0f}s.",
        "",
        "| Trigger -> Outcome | P(outcome \\| trigger) | P(outcome \\| no trigger) | Trigger windows | No-trigger windows |",
        "|---|---|---|---|---|",
        f"| Overload -> Progressive Pass | {_fmt_p(ov['p_outcome_given_trigger'])} | "
        f"{_fmt_p(ov['p_outcome_given_no_trigger'])} | {ov['n_trigger_windows']} | {ov['n_no_trigger_windows']} |",
        f"| Dominance -> Attacking-Third Entry | {_fmt_p(do['p_outcome_given_trigger'])} | "
        f"{_fmt_p(do['p_outcome_given_no_trigger'])} | {do['n_trigger_windows']} | {do['n_no_trigger_windows']} |",
    ]
    return "\n".join(lines)


def conclusion(summary: dict) -> str:
    ov = summary["overload_to_progressive_pass_pooled"]
    do = summary["dominance_to_attacking_third_pooled"]
    parts = []
    ov_outcomes_seen = ov["n_trigger_hits"] + ov["n_no_trigger_hits"]
    if ov_outcomes_seen == 0 and ov["n_trigger_windows"] + ov["n_no_trigger_windows"] > 0:
        parts.append(
            "PROGRESSIVE_PASS never fired at all across either calibrated clip, "
            "so the 0% vs 0% below reflects an outcome that never happened in "
            "this sample, not evidence that overload has no effect on it."
        )
    elif ov["p_outcome_given_trigger"] is not None and ov["p_outcome_given_no_trigger"] is not None:
        lift = ov["p_outcome_given_trigger"] - ov["p_outcome_given_no_trigger"]
        parts.append(
            f"Overload windows are {'more' if lift > 0 else 'no more (or less)'} likely to "
            f"contain a progressive pass than non-overload windows "
            f"({_fmt_p(ov['p_outcome_given_trigger'])} vs {_fmt_p(ov['p_outcome_given_no_trigger'])})."
        )
    if do["p_outcome_given_trigger"] is not None and do["p_outcome_given_no_trigger"] is not None:
        lift = do["p_outcome_given_trigger"] - do["p_outcome_given_no_trigger"]
        parts.append(
            f"Dominance windows are {'more' if lift > 0 else 'no more (or less)'} likely to "
            f"see the dominant team enter their attacking third than non-dominance windows "
            f"({_fmt_p(do['p_outcome_given_trigger'])} vs {_fmt_p(do['p_outcome_given_no_trigger'])})."
        )
    elif do["n_trigger_windows"] == 0:
        parts.append(
            "DOMINANCE never fired at all across either calibrated clip, so "
            "P(attacking-third entry | dominance) can't be measured here -- the "
            "event's trigger condition (sustained space-control imbalance) "
            "apparently never held for a full window in this sample, which says "
            "as much about how rare/strict the DOMINANCE trigger is as it does "
            "about its predictive value."
        )
    parts.append(
        f"Based on {ov['n_trigger_windows'] + ov['n_no_trigger_windows']} {summary['window_s']:.0f}s "
        "windows pooled across all calibrated clips -- a small sample; treat the direction of "
        "the effect as more meaningful than the exact percentages."
    )
    return " ".join(parts)
