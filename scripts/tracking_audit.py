"""
scripts/tracking_audit.py
─────────────────────────
Why are there still ~392 unique track IDs for ~20 players in 60s?

This audit isolates the cause by sweeping the cheap knobs (confidence,
detection cadence, buffer) AND testing the structural fix the literature points
to for broadcast sport — **camera-motion compensation (CMC)** via BoT-SORT.

Background (researched):
  A broadcast camera pans/zooms constantly. ByteTrack predicts each box's next
  position with a Kalman filter in IMAGE space; a camera pan shifts every box by
  the same vector, so predictions stop overlapping the new detections, IoU drops
  below the match floor, tracks go "lost", and new IDs spawn — for ALL tracks at
  once. Supervision's ByteTrack has no CMC. BoT-SORT adds it: it estimates the
  frame-to-frame affine warp (sparse optical flow) and warps Kalman predictions
  into the current frame before matching. (roboflow `trackers` package.)

Design (efficient):
  Detection is the expensive part, so we run YOLO on EVERY frame ONCE at a low
  conf floor and cache per-frame detections. Every (cadence, conf) config is
  then derived in memory. CMC configs re-read frames (cheap decode) to feed the
  optical-flow estimator. Metrics mirror the production pipeline exactly:
  positions are recorded on every video frame (track ids carried forward between
  detection frames), so "tracks >= 10s" is measured in real video-frame spans.

Run:
    uv run python scripts/tracking_audit.py \
        data/test_clips/tactical_playlist_1.mp4 --start 30 --duration 60
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import supervision as sv

from gaffer import config
from gaffer.detection.detector import Detection, FootballDetector
from gaffer.tracking.position_store import PositionStore
from gaffer.video.loader import VideoLoader

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

_TRACKABLE = frozenset({"player", "goalkeeper", "person"})
CACHE_CONF = 0.20            # detect at a low floor; configs filter upward in memory


def _iou(a, b) -> float:
    """IoU of two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


# ─── Tracker adapters — uniform .assign(dets, frame) -> list[(det, track_id)] ───

class _SvByteTrack:
    """Production tracker: supervision.ByteTrack (no CMC, cost-based IoU match)."""
    def __init__(self, fps, conf, buffer_steps, match_cost):
        self.t = sv.ByteTrack(
            track_activation_threshold=conf,
            lost_track_buffer=buffer_steps,
            minimum_matching_threshold=match_cost,   # COST ceiling: 0.7 -> IoU>=0.3
            frame_rate=fps,
            minimum_consecutive_frames=1,
        )

    def assign(self, dets, frame):
        if not dets:
            return []
        sv_in = sv.Detections(
            xyxy=np.array([d.bbox for d in dets], dtype=float),
            confidence=np.array([d.confidence for d in dets], dtype=float),
            class_id=np.array([d.class_id for d in dets], dtype=int),
        )
        out = self.t.update_with_detections(sv_in)
        # supervision returns only confirmed tracks, reordered -> map by bbox key
        tid = {}
        if out.tracker_id is not None:
            for i in range(len(out)):
                tid[tuple(out.xyxy[i].astype(int))] = int(out.tracker_id[i])
        return [(d, tid.get(tuple(int(v) for v in d.bbox), -1)) for d in dets]


class _RfTracker:
    """roboflow `trackers` BoT-SORT / ByteTrack. Returns input boxes in order."""
    def __init__(self, kind, fps, conf, buffer_steps, enable_cmc):
        from trackers import BoTSORTTracker, ByteTrackTracker
        if kind == "botsort":
            self.t = BoTSORTTracker(
                lost_track_buffer=buffer_steps,
                frame_rate=fps,
                track_activation_threshold=conf,      # DIRECT, not cost
                minimum_consecutive_frames=1,
                high_conf_det_threshold=max(conf + 0.10, 0.30),
                enable_cmc=enable_cmc,
                cmc_method="sparseOptFlow",
            )
        else:
            self.t = ByteTrackTracker(
                lost_track_buffer=buffer_steps,
                frame_rate=fps,
                track_activation_threshold=conf,
                minimum_consecutive_frames=1,
                high_conf_det_threshold=max(conf + 0.10, 0.30),
            )

    def assign(self, dets, frame):
        if not dets:
            return []
        sv_in = sv.Detections(
            xyxy=np.array([d.bbox for d in dets], dtype=float),
            confidence=np.array([d.confidence for d in dets], dtype=float),
            class_id=np.array([d.class_id for d in dets], dtype=int),
        )
        out = self.t.update(sv_in, frame)
        # roboflow RE-ORDERS output (confirmed first, -1s last) but keeps box
        # coords -> map by bbox key, NOT by index.
        tid = {}
        if out.tracker_id is not None:
            for i in range(len(out)):
                tid[tuple(np.rint(out.xyxy[i]).astype(int))] = int(out.tracker_id[i])
        return [(d, tid.get(tuple(int(v) for v in d.bbox), -1)) for d in dets]


# ─── Config matrix ──────────────────────────────────────────────────────────

@dataclass
class Cfg:
    name:        str
    backend:     str               # "sv-bytetrack" | "rf-botsort" | "rf-bytetrack"
    skip:        int               # detection cadence (1 = every frame)
    conf:        float
    cmc:         bool   = False
    buffer_wall: int    = 45       # lost-track buffer in *video* frames (~1.8s @25)
    match_cost:  float  = 0.7      # supervision ByteTrack IoU-cost ceiling (legacy prod value)


CONFIGS: list[Cfg] = [
    Cfg("A  sv-ByteTrack  skip3 conf.25  (PROD baseline)", "sv-bytetrack", 3, 0.25),
    Cfg("B  sv-ByteTrack  skip3 conf.35  (conf sweep)",    "sv-bytetrack", 3, 0.35),
    Cfg("C  sv-ByteTrack  skip1 conf.25  (every frame)",   "sv-bytetrack", 1, 0.25),
    Cfg("D  BoT-SORT+CMC  skip3 conf.25  (CMC only)",      "rf-botsort",   3, 0.25, cmc=True),
    Cfg("E  BoT-SORT+CMC  skip1 conf.25  (CMC+everyframe)","rf-botsort",   1, 0.25, cmc=True),
    Cfg("F  BoT-SORT+CMC  skip1 conf.35  (CMC+ef+conf)",   "rf-botsort",   1, 0.35, cmc=True),
]


@dataclass
class Result:
    name:           str
    unique_ids:     int   = 0
    long_tracks:    int   = 0      # >= 10s
    mean_active:    float = 0.0
    mean_persist:   float = 0.0
    frag:           float = 0.0    # unique_ids / mean_active (lower = better)
    id_switches:    int   = 0      # IoU-continuity: # times a box's id changed
    sw_per_100f:    float = 0.0    # id switches per 100 video frames (the visible churn)
    seconds:        float = 0.0


def _make_tracker(cfg: Cfg, fps: float):
    buffer_steps = max(1, round(cfg.buffer_wall / cfg.skip))   # buffer in update-steps
    if cfg.backend == "sv-bytetrack":
        return _SvByteTrack(fps, cfg.conf, buffer_steps, cfg.match_cost)
    if cfg.backend == "rf-botsort":
        return _RfTracker("botsort", fps, cfg.conf, buffer_steps, cfg.cmc)
    return _RfTracker("bytetrack", fps, cfg.conf, buffer_steps, False)


def _run_config(cfg, cache, frames_dir_loader, start, n_frames, fps) -> Result:
    tracker = _make_tracker(cfg, fps)
    store = PositionStore(fps=fps)
    ten_s = int(10 * fps)
    active = []
    last_assign: list[Detection] = []
    t0 = time.perf_counter()

    # CMC needs the actual image; re-read frames only when required.
    frame_iter = (frames_dir_loader.frames(start=start, count=n_frames)
                  if cfg.cmc else None)

    prev_boxes: list[tuple] = []     # (xyxy float tuple, track_id) from previous frame
    switches = 0

    for k in range(n_frames):
        fidx = start + k
        if cfg.cmc:
            _, frame = next(frame_iter)
        else:
            frame = None

        if k % cfg.skip == 0:                      # detection / update frame
            dets = [d for d in cache[fidx]
                    if d.confidence >= cfg.conf and d.class_name in _TRACKABLE]
            assigned = tracker.assign(dets, frame)
            last_assign = [_with_id(d, tid) for d, tid in assigned]

        store.update(fidx, last_assign)
        active.append(sum(1 for d in last_assign if d.track_id >= 0))

        # ID-switch via IoU continuity: a box overlapping a previous box (IoU>0.5)
        # but carrying a different id = the number on that player changed.
        cur = [(tuple(map(float, d.bbox)), d.track_id)
               for d in last_assign if d.track_id >= 0]
        if prev_boxes:
            for cb, cid in cur:
                best_iou, best_pid = 0.0, None
                for pb, pid in prev_boxes:
                    iou = _iou(cb, pb)
                    if iou > best_iou:
                        best_iou, best_pid = iou, pid
                if best_iou > 0.5 and best_pid is not None and best_pid != cid:
                    switches += 1
        prev_boxes = cur

    persist = [p for t in store.track_ids if (p := store.persistence(t)) is not None]
    mean_active = float(np.mean(active)) if active else 0.0
    uids = len(store.track_ids)
    return Result(
        name=cfg.name,
        unique_ids=uids,
        long_tracks=sum(1 for t in store.track_ids if store.track_length(t) >= ten_s),
        mean_active=mean_active,
        id_switches=switches,
        sw_per_100f=100.0 * switches / max(n_frames, 1),
        mean_persist=float(np.mean(persist)) if persist else 0.0,
        frag=uids / mean_active if mean_active else 0.0,
        seconds=time.perf_counter() - t0,
    )


def _with_id(det: Detection, tid: int) -> Detection:
    from dataclasses import replace
    return replace(det, track_id=tid)


def main():
    ap = argparse.ArgumentParser(description="Gaffer tracking audit")
    ap.add_argument("clip")
    ap.add_argument("--start", type=float, default=30.0)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--model", default=str(config.DETECTION_MODEL_PATH))
    args = ap.parse_args()

    loader = VideoLoader(args.clip)
    fps = loader.fps
    start = int(args.start * fps)
    n_frames = min(int(args.duration * fps), loader.total_frames - start)

    # ── Phase 1: cache detections on EVERY frame, once ──────────────────────
    print(f"Clip {loader}  window {args.start:.0f}-{args.start+args.duration:.0f}s "
          f"({n_frames} frames)")
    print(f"Phase 1: caching detections every frame @ conf>={CACHE_CONF} ...")
    detector = FootballDetector(model_path=args.model, conf=CACHE_CONF,
                                detect_every_n=1, verbose=False)
    cache: dict[int, list[Detection]] = {}
    t0 = time.perf_counter()
    for fidx, frame in loader.frames(start=start, count=n_frames):
        cache[fidx] = detector.detect_raw(frame)
        if (fidx - start) % 300 == 0 and fidx > start:
            print(f"  cached {fidx-start}/{n_frames} frames")
    loader.close()
    print(f"  done in {time.perf_counter()-t0:.0f}s  "
          f"(mean {np.mean([len(v) for v in cache.values()]):.1f} raw dets/frame)")

    # ── Phase 2: sweep configs ──────────────────────────────────────────────
    print("\nPhase 2: evaluating configs ...")
    results = []
    for cfg in CONFIGS:
        # fresh loader for CMC configs (needs sequential frame reads)
        fl = VideoLoader(args.clip)
        r = _run_config(cfg, cache, fl, start, n_frames, fps)
        fl.close()
        results.append(r)
        print(f"  done: {cfg.name}  ->  {r.unique_ids} ids  "
              f"{r.id_switches} switches  {r.long_tracks} long  ({r.seconds:.0f}s)")

    # ── Report: ranked table ────────────────────────────────────────────────
    # PRIMARY metric is ID SWITCHES (what you actually see on screen): how often
    # the number on a player flips. Unique-ID count alone is misleading — a
    # tracker can keep the count low while constantly swapping IDs between players.
    base_long = next(r.long_tracks for r in results if r.name.startswith("A"))
    long_floor = 0.85 * base_long
    ranked = sorted(
        results,
        key=lambda r: (r.long_tracks < long_floor, r.id_switches))

    print("\n" + "═" * 100)
    print("  TRACKING AUDIT — ranked by ID SWITCHES (visible churn), keeping long tracks")
    print("═" * 100)
    print(f"  {'config':<46}{'switches':>9}{'sw/100f':>9}{'uniqIDs':>8}"
          f"{'>=10s':>7}{'active':>8}{'persist':>9}")
    print("  " + "─" * 96)
    for i, r in enumerate(ranked):
        star = " <- best" if i == 0 else ""
        flag = "  (lost long tracks!)" if r.long_tracks < long_floor else ""
        print(f"  {r.name:<46}{r.id_switches:>9}{r.sw_per_100f:>9.1f}{r.unique_ids:>8}"
              f"{r.long_tracks:>7}{r.mean_active:>8.1f}{r.mean_persist:>9.2f}{star}{flag}")
    print("═" * 100)
    print("  switches = # times a box overlapping its previous-frame position (IoU>0.5) "
          "carried a different id")
    best = ranked[0]
    base = next(r for r in results if r.name.startswith("A"))
    print(f"\n  Baseline A: {base.id_switches} switches, {base.unique_ids} IDs, "
          f"{base.long_tracks} long tracks.")
    print(f"  Best     : {best.name.strip()}")
    print(f"             {best.id_switches} switches "
          f"({100*(base.id_switches-best.id_switches)/max(base.id_switches,1):+.0f}% vs baseline), "
          f"{best.unique_ids} IDs, {best.long_tracks} long tracks.")


if __name__ == "__main__":
    main()
