# Gaffer Evaluation Report (v1.E)

## Methodology & Limitations

- **Sample size: 2 calibrated clips** (arsenal_newcastle_highlights, tactical_playlist_1). Calibration is an interactive point-and-click step (`scripts/collect_calibration.py`); `psg_newcastle_tactical`, `tactical_playlist_2`, and `tactical_playlist_3` have none yet, so every result below is drawn from just these two matches.
- **No ground-truth ball-position labels exist in this repo.** E1's "recoveries" / "suspect discards" are `BallCandidateFilter`'s own self-reported counters, not validated precision/recall against labeled data.
- **E3 (pass-network stability) is within-match only** (first half vs second half of the same clip), not cross-broadcast -- the only team that recurs across two different clips here (Newcastle) has its second appearance (`psg_newcastle_tactical`) uncalibrated.
- **E2's conditional probabilities use 5-second time windows** -- a documented methodology choice, not a hidden constant (see `evaluation/evaluate_events.py`).

## E1 — Ball Tracking: v1.0 vs v2.0

| Clip | Model | Live detections | Extrapolated | Lost % | Recoveries | Suspect discards | Mean continuity (frames) | Mean WM score |
|---|---|---|---|---|---|---|---|---|
| arsenal_newcastle_highlights | v1.0 | 54 | 89 | 85.7 | 14 | 0 | 11.0 | 0.661 |
| arsenal_newcastle_highlights | v2.0 | 54 | 89 | 85.7 | 14 | 0 | 11.0 | 0.639 |
| tactical_playlist_1 | v1.0 | 219 | 195 | 58.6 | 20 | 2 | 19.7 | 0.775 |
| tactical_playlist_1 | v2.0 | 219 | 195 | 58.6 | 20 | 2 | 19.7 | 0.743 |

Pooled across 2 clip(s): v2.0 lost-ball% 72.2 vs v1.0 72.2; mean continuity 15.3 vs 15.3 frames; mean world-model plausibility score 0.691 vs 0.718; mean recoveries-after-loss 17.0 vs 17.0.
v1.0 and v2.0 accepted/rejected the exact same candidates on every clip here (identical live/extrapolated/lost counts) -- v2.0's extra signals (corridor prediction, overload-zone prior, space-control score, press-locality) changed its own plausibility score but never flipped an accept/reject decision in this sample.
Zero scene cuts were detected on either clip, so the bulk of lost-ball frames trace to the filter's own spatial/off-pitch plausibility gates rejecting candidates (arsenal_newcastle_highlights: spatial (531); tactical_playlist_1: spatial (605)) -- i.e. the upstream object detector flagging non-ball objects as "ball" more often than the tracker can recover from, not scene cuts or a world-model weakness.
Recoveries and suspect discards are the filter's own self-reported counters, not validated against ground truth -- there is no human-labeled ball-position dataset in this repo, so these measure how hard each world model makes the filter work, not precision/recall.

## E2 — Tactical Event Validation

Window size: 5s.

| Trigger -> Outcome | P(outcome \| trigger) | P(outcome \| no trigger) | Trigger windows | No-trigger windows |
|---|---|---|---|---|
| Overload -> Progressive Pass | 0% | 0% | 14 | 82 |
| Dominance -> Attacking-Third Entry | n/a | 36% | 0 | 96 |

PROGRESSIVE_PASS never fired at all across either calibrated clip, so the 0% vs 0% below reflects an outcome that never happened in this sample, not evidence that overload has no effect on it. DOMINANCE never fired at all across either calibrated clip, so P(attacking-third entry | dominance) can't be measured here -- the event's trigger condition (sustained space-control imbalance) apparently never held for a full window in this sample, which says as much about how rare/strict the DOMINANCE trigger is as it does about its predictive value. Based on 96 5s windows pooled across all calibrated clips -- a small sample; treat the direction of the effect as more meaningful than the exact percentages.

## E3 — Pass Network Stability (within-match)

| Clip | Top edge (1st half) | Top edge (2nd half) | Hub overlap | Edge overlap (Jaccard) |
|---|---|---|---|---|
| arsenal_newcastle_highlights | #69 -> #98 | n/a | 0% | 0% |
| tactical_playlist_1 | #139 -> #159 | #798 -> LM | 0% | 0% |

Mean hub-player overlap between first and second half: 0%. This is a within-match stability check (same broadcast, same two teams, split in time at the midpoint) -- NOT a cross-match check. The only team appearing in two different clips in this repo is Newcastle (arsenal_newcastle_highlights / psg_newcastle_tactical), and the second clip has no calibration yet; calibrating it would unlock a true cross-broadcast comparison as a follow-up.

## E4 — Episode Validation

| Outcome | Episodes | Mean events/episode | Mean duration (s) |
|---|---|---|---|
| Lost Possession | 17 | 2.8 | 2.5 |
| Attacking Third Entry | 8 | 2.4 | 4.6 |
| Counter | 2 | 9.0 | 1.1 |
| Line Break | 1 | 4.0 | 2.3 |
| Sustained Possession | 1 | 2.0 | 36.2 |
| **All outcomes** | **29** | **3.1** | **4.2** |

29 completed episodes pooled across calibrated clips (arsenal_newcastle_highlights: 5, tactical_playlist_1: 24), averaging 3.1 events and 4.2s each. Most common outcome: Lost Possession.
