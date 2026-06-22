# Gaffer V2 Product Review

*Written 2026-06-22, immediately after v2.2 (Temporal Evidence Retrieval), while the
v1.E evaluation results, the 28-question analyst stress test, and the raw-video/report
review are still fresh. This is not a status report -- it's the design input for
whatever comes after v2.2.*

## Project State (Post v2.2)

```
Vision
  -> Football Analytics
    -> Events
      -> Episodes
        -> Reports
          -> Analyst Interface
            -> Temporal Evidence Retrieval
```

The core pipeline is feature-complete for the original vision: a video goes in, and
Gaffer can detect, track, understand, summarize, and answer questions about it. Every
later layer was deliberately built on top of validated lower layers (v1.E gated v2.0;
v2.1/v2.1.1 fixed hallucinations found by using the system, not by guessing).

---

## What Gaffer Can Do Today

### Vision
- Detect players, goalkeepers, referees, and the ball
- Track players across frames (ByteTrack)
- Assign players to teams
- Project pixel positions onto a metric pitch model (homography + minimap)
- Maintain calibration under camera pan/zoom (optical-flow homography propagation)

### Football Analytics
- Possession (team-level, hysteresis-debounced)
- Compactness, defensive line height
- Space control (Voronoi-clipped territorial %)
- Overloads (numerical superiority by pitch zone)
- Dominance (sustained territorial control)
- Pass detection, pass networks, progressive passes
- Formation estimation (back-4 vs back-3)
- Role identification (track IDs become LB / DM / RW, not just numbers)
- Player influence (hub players via degree centrality, build-up chains)

### Events
- Possession change / recovery
- Counter attack
- High press (onset + end)
- Line break
- Sprint start / end
- Compact block
- Progressive pass
- Overload
- Dominance
- Pass (sender -> receiver, with progressive flag)

### Higher-Level Understanding
- Tactical episodes (possession-level stories built from event sequences)
- Structured match reports (fully factual, no LLM)
- Natural-language analyst (LLM reasons over evidence, never raw detections)
- Temporal evidence retrieval ("what happened around 94s?")
- Event/episode-to-clip lookup and export ("the counter attack" -> a real .mp4)

---

## What Gaffer Cannot Do

- Goals, shots, xG
- Cards
- Offside detection
- Real player identity (jersey numbers, names -- only anonymous track IDs / roles)
- Full-match tactical analysis from multiple cameras
- Reliable ball tracking under long occlusion

These are unsupported and should remain unsupported -- declined honestly, not
guessed -- until measured capability exists to back them.

---

## Findings From Evaluation (v1.E)

### Ball tracking
World Model v2's extra signals (pass-corridor prediction, overload-zone prior, real
space-control score, press-locality) shifted the model's own internal confidence score
but produced **identical accept/reject outcomes** to v1.0 on both evaluated clips.
World Model v2 does not currently change tracking behavior, only self-scoring.

The dominant rejection gate was `spatial` (531/605 of 1000 detection frames on the two
evaluated clips), not scene cuts (`scene_cuts: 0` on both) and not a tracking/world-model
weakness -- the upstream detector is flagging non-ball objects as "ball" more often than
the gate is mis-handling real ball motion.

### Events
PROGRESSIVE_PASS and DOMINANCE fired **zero times** across two full 120s evaluation
clips. This isn't evidence the logic is wrong (the v1.8/v1.9 synthetic suites confirm
the mechanics work), but it does mean these two event types are rare/strict in practice
-- any report or analyst answer that leans on them will look sparse on a typical clip.

### Episodes
29 completed episodes pooled across both clips. Episodes meaningfully compress large
event streams into interpretable sequences -- the most common outcome was "Lost
Possession" (17 of 29), mean 3.1 events/episode, mean 4.2s duration.

---

## Findings From Self-Test

### Raw video
**Strengths:** possession visible immediately, team structure visible, minimap useful.
**Weaknesses:** event ticker noisy, abbreviated labels unclear, no narrative synthesis
of what's happening beyond single events.

### Match report
**Strengths:** tactical episodes highly informative, surfaces non-obvious patterns.
**Weaknesses:** sparse metrics reduce trust (e.g. "Passes: 2" reads as broken, not as
an honest small sample), no team-level breakdowns, no temporal grounding (a report
entry has no way to point back at a moment in the video).

### Analyst interface
**Strengths:** honest decline path works (no fabricated goals/shots/win-probabilities),
retrieval-based explanations are generally useful and grounded.
**Weaknesses (pre-v2.2):** temporal retrieval was missing entirely -- "what happened
around 94s?" pulled the whole match and the LLM fabricated a timestamp. Premise
checking is still limited. Sparse-data answers can be technically correct
("0 progressive passes") but practically unhelpful.

v2.2 closed the temporal-retrieval weakness directly (time-window questions, event/
episode lookup, clip export). The other two analyst weaknesses are still open.

---

## Confirmed Product Insights

1. **Users do not primarily complain about ball tracking.** Every weak spot found in
   the self-test was about retrieval, grounding, or trust -- not detection quality.
2. **Users repeatedly ask "when did this happen?" and "show me."** Both raw-video and
   analyst review converged on the same underlying need from different angles.
3. **Temporal grounding creates more perceived value than additional analytics
   complexity.** v2.2 (no new detection, no new analytics) fixed the single
   highest-leverage gap found across three review angles.
4. **Trust and explainability are becoming more important than new football metrics.**
   The report's sparse-metric problem and the analyst's premise-checking gap are both
   trust problems, not capability gaps.

---

## What Feels Unfinished?

Looking at the three review angles together, one thread runs underneath nearly every
weakness, ahead of any of them individually: **Gaffer has four separate output
surfaces that don't point at each other.**

- `render_analytics_demo.py` produces an annotated video with a live event ticker.
- `MatchReport.render()` produces plain text with a "Top Tactical Episodes" list.
- `ask_gaffer.py` produces prose answers to natural-language questions.
- `find_clip.py` (v2.2) produces an actual playable clip -- but only if you already
  know the right query string to type.

None of these link to each other. A report's "#1 Recovery -> Counter -> Line Break"
episode has `start_time_s`/`end_time_s` sitting right there on the `Episode` object,
but reading the report gives no way to actually watch it -- you'd have to copy the
narrative back out, guess a `find_clip.py` query that resolves to the same event, and
run it from a second terminal. An analyst answer about "the counter attack in the
second half" is grounded in real evidence (v2.2 made sure of that), but it ends in
prose, not a place to look. Temporal retrieval closed the *worst* version of this gap
-- Gaffer no longer fabricates timestamps -- but it closed it as a fourth disconnected
surface, not as a bridge between the other three.

This also reframes "go deeper on analytics" as lower-priority right now, not just
lower-urgency. v1.E already showed a real cost to that direction before any payoff:
PROGRESSIVE_PASS and DOMINANCE fired zero times across two full evaluation clips, and
World Model v2's extra signals didn't change a single ball-tracking decision. Adding
more metrics on top of that foundation risks producing reports that are *differently*
sparse, not more trustworthy.

**Conclusion:** the unfinished piece isn't a missing capability, it's the connective
tissue between capabilities that already exist -- report <-> episode <-> event <->
video. That points directly at **v2.3 (Highlight Reels)**: turning a report's
top-episode list into a folder of playable clips, generated automatically from
`Episode.start_time_s`/`end_time_s` with no new detection and no new LLM call, is the
most direct way to build that bridge. **v2.4 (sparse-metric honesty, e.g. "Passes: 2
(small sample)")** is the second, smaller, parallel thread -- a trust fix, not a
navigation fix -- and is cheap enough to fold in alongside v2.3 rather than sequenced
strictly after it.

After both: stop feature development and actually use Gaffer on a handful of different
matches before deciding what v2.5 even is. Every real bug and every real insight so far
(role flicker, episode same-frame ordering, the 94s hallucination) came from using the
system on real footage, not from reasoning about it in the abstract -- there's no
reason to expect that to change.

---

## Non-goals (Current)

- Additional football analytics metrics
- New world-model priors / ball-tracking redesign
- More event types
- Goals / shots / xG / cards / offside detection
- Real player identity, multi-camera analysis

Not because these are solved forever, or impossible, or uninteresting -- but because
current evidence (this document) does not indicate they are the highest-leverage next
improvement. Re-opening any of these should require new evidence, not just renewed
interest: a new self-test finding, a new evaluation result, or a user repeatedly
hitting the same wall v2.3/v2.4 don't cover. Re-litigating this list from scratch every
few weeks is exactly what this document exists to prevent.
