# Tracking Engineering Notes

Lessons learned the hard way. Read before touching any tracker parameter.

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
