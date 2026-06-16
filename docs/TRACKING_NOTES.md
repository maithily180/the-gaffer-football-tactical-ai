# Tracking Engineering Notes

Lessons learned the hard way. Read before touching any tracker parameter.

---

## 0. Day 5 audit: measure ID SWITCHES, not unique-ID count

After fine-tuning, a 60s clip still produced ~392 unique IDs for ~20 players, so
I ran `scripts/tracking_audit.py` to reduce it. **My first pass optimised the
wrong metric (unique-ID count) and shipped a regression.** The visible problem
is ID *switches* — how often the number on a player flips — and that is what the
audit now ranks by. Final numbers (tactical_playlist_1, 30–90s):

| config | **switches** | uniqIDs | ≥10s | speed |
|---|---|---|---|---|
| **B  ByteTrack skip3 conf.35  (SHIPPED)** | **47** | 374 | 28 | fast |
| A  ByteTrack skip3 conf.25  (old prod ≈ v0.2) | 60 | 395 | 29 | fast |
| D  BoT-SORT+CMC skip3 | 148 | 355 | 26 | slow |
| F  BoT-SORT+CMC every-frame conf.35 | 176 | 159 | 33 | 3× slow |
| C  ByteTrack every-frame | 177 | 278 | 34 | slow |
| E  BoT-SORT+CMC every-frame conf.25 | 294 | 242 | 35 | slow |

**Hard lessons:**
1. **Unique-ID count is a misleading target.** A tracker can keep the count low
   while constantly swapping IDs between players. Config F had the *fewest* IDs
   (159) but ~3× the switching of the simple baseline. Most of the ~390 IDs in
   the good configs are players **re-entering frame after a camera cut** (a new
   ID on return) — not mid-track instability. Low switches = IDs stick to players.
2. **BoT-SORT+CMC was worse than plain ByteTrack at every setting.** Its looser
   association (first-assoc IoU 0.2) swaps IDs in crowds, and CMC mis-warps on
   textureless grass with independently-moving players. The fancy tracker lost.
3. **Every-frame detection increased switching**, not decreased it — more
   frame-to-frame association wobble, whereas skip-3's carry-forward holds IDs
   steady for the 2 cached frames. (It does make box *motion* smoother, though.)
4. **Confidence is the one real win:** `conf .35` → 47 switches vs `.25` → 60.
   Shipped as a tracker-level filter (`config.TRACK_MIN_CONF`) on players/GKs
   only; the detector floor stays 0.25 so the ball (~0.10/frame) and referees
   still get detected/drawn.
5. **Team-assignment instability is a non-cause** — `team_id` never feeds the
   tracker.

**Verdict: kept supervision ByteTrack + skip-3, raised the track-entry conf to
0.35.** Lowering the unique-ID count further needs cross-gap re-identification,
which is hard for football (teammates wear identical kits) and doesn't improve
what you see on screen — switches do.

---

## 0b. Why BoT-SORT (roboflow `trackers`) was tried and reverted

BoT-SORT adds Camera-Motion Compensation (frame-to-frame affine warp via sparse
optical flow). It *sounds* ideal for broadcast pans, and reduced unique-ID count,
but **raised ID switching** (§0) so it was reverted. Notes if revisited:
- **The mapping trap that cost a day:** roboflow trackers RE-ORDER their output
  (confirmed tracks first, unconfirmed `-1`s last) while returning the input box
  coords unchanged. Mapping the returned `tracker_id` array **by index** assigns
  IDs to the wrong players — looks fine on unique-count, catastrophic on
  switches. Map by **bbox key**. (Verify with a ≥3-box mixed-confidence probe;
  a 2-box clean probe misleadingly shows order preserved.)
- **Convention trap:** roboflow BoT-SORT uses a DIRECT IoU floor
  (`minimum_iou_threshold_first_assoc=0.2`, higher = stricter) — opposite of
  supervision ByteTrack's IoU-*cost* ceiling (§1).
- Its pedestrian defaults (`track_activation_threshold=0.7`) activate **zero**
  tracks on football detections at conf ~0.3 — set activation to `TRACK_MIN_CONF`.
- If retried: disable CMC (grass is featureless), tighten first-assoc IoU, and
  judge on switches — not count.

---

## 1. supervision ByteTrack — threshold semantics are inverted

**The trap:**

```python
# WRONG — looks like "accept IoU ≥ 0.3", actually means "accept IoU ≥ 0.7"
sv.ByteTrack(minimum_matching_threshold=0.3)

# CORRECT — accepts matches with IoU ≥ 0.3
sv.ByteTrack(minimum_matching_threshold=0.7)
```

**Why:**
supervision ByteTrack's `minimum_matching_threshold` is an **IoU-cost ceiling**, not an IoU floor.
IoU cost = `1 - IoU`. The match is accepted if `cost ≤ threshold`.

```
threshold = 0.3  →  cost ≤ 0.3  →  IoU ≥ 0.7   (strict — kills football tracking)
threshold = 0.7  →  cost ≤ 0.7  →  IoU ≥ 0.3   (appropriate for football)
threshold = 0.8  →  cost ≤ 0.8  →  IoU ≥ 0.2   (supervision default, permissive)
```

**The trackers package (`ByteTrackTracker`) uses the opposite convention:**

```python
# trackers package: minimum_iou_threshold IS a direct IoU floor
ByteTrackTracker(minimum_iou_threshold=0.3)   # accepts IoU ≥ 0.3  ✓
```

**Mapping between APIs:**
```
trackers.minimum_iou_threshold = X
  ↔
sv.ByteTrack.minimum_matching_threshold = 1 - X
```

**Current config:** `MATCH_THRESH = 0.7` (for supervision ByteTrack)

---

## 2. Why 467 unique IDs is expected (not a bug)

Over 60s at 25fps with skip-3, ~500 detection events × ~14 detections = ~7000 detection calls.
Even with good matching, every time the detector **misses** a player for a few frames:

```
player visible frames 0–90  →  track #3
player missing frames 91–96 (detector missed)
player visible frames 97+   →  track #41  (new ID)
```

**Root cause:** YOLO11n is a COCO model with no football-specific training.
At conf=0.25, false negatives on partially-occluded or small players are frequent.

**Expected improvement after Day 5 (Roboflow fine-tuning):**
Unique IDs should drop to 20–60 over 60s (matching the actual number of player entry/exit events).

**The metric that matters now:** `Tracks ≥ 10s` — only long-lived tracks feed analytics.

---

## 3. Skip-3 detection requires explicit carry_forward()

```
Frame 750: detector runs  → tracker.update(dets)    → track_ids populated
Frame 751: detector cache → tracker.carry_forward()  → same track_ids reused
Frame 752: detector cache → tracker.carry_forward()  → same track_ids reused
Frame 753: detector runs  → tracker.update(dets)    → tracker sees new positions
```

**If you forget carry_forward():** frames 751 and 752 have zero active tracks,
which collapses mean-active metrics and causes spurious gap detection in PositionStore.

**If you call tracker.update() on every frame:** ByteTrack sees 3× the expected update
rate and loses tracks because positions barely change between adjacent frames — the
Kalman filter interprets it as very slow movement and raises IoU cost.

---

## 4. minimum_consecutive_frames — set to 1 for skip-3 detection

With `minimum_consecutive_frames=2` and skip-3:
- Frame 750: track created (tentative)
- Frame 753: match required to confirm
- Frame 756: now confirmed — but track might already be lost

At 25fps/skip-3, 120ms between detection events means a player running at 5 m/s moves
~10px per detection cycle. IoU at that spacing is fine for matching, but requiring 2
consecutive cycles before confirmation means 6 real frames of latency before a track
gets an ID. This is too slow.

**Use:** `minimum_consecutive_frames=1` with supervision ByteTrack + skip-3 strategy.

---

## 5. PositionStore records detection-frame positions only

`store.update(gidx, dets)` should be called on **every rendered frame** (not just
detection frames), because carry_forward() re-emits the same detections for cache frames.
The `track_id` is already set; positions are the last detected position (bboxes don't
update on cache frames, so entries for the same track in adjacent frames will have
identical centers during skip windows — this is acceptable at 25fps).

---

## 6. Team assignment imbalance with COCO model

KMeans k=3 on jersey HSV separates referee + two teams. With COCO model at conf=0.25,
false positives (referees/coaches on the sideline) distort cluster membership.
Expect T0:T1 imbalance up to 3:1 until fine-tuned model removes non-player detections.

---

## Config reference (as of Day 3)

```python
DETECTION_CONF_THRESHOLD = 0.25  # Day 1 finding: more players than 0.35
DETECT_EVERY_N_FRAMES    = 3     # Day 1 finding: frame delta ≈ 0 → safe
TRACK_THRESH             = 0.25  # match detection floor
MATCH_THRESH             = 0.7   # supervision cost ceiling → IoU ≥ 0.3
TRACK_BUFFER_FRAMES      = 45    # 1.8s at 25fps
```
