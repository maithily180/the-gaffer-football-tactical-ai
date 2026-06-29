"""
gaffer/calibration/homography_propagator.py
─────────────────────────────────────────────
Problem B — keep the homography locked to the pitch as the camera moves.

A single calibration gives one H, valid only while the camera matches the
calibration framing.  As the broadcast pans / tilts / zooms, that static H
drifts (measured: ~1131px of camera translation over 40s on tactical_playlist_1,
while the static-H "visible region" stayed frozen).

This propagator keeps H current WITHOUT re-detecting pitch keypoints:

    frame t            frame t+1
      │  Lucas-Kanade optical flow on STATIC-WORLD features
      ▼  (pitch + crowd; players/ball masked out — they move independently)
    inter-frame image homography  A   (p_{t+1} = A · p_t), RANSAC
      ▼
    H_{t+1} = H_t · A⁻¹

Derivation: a fixed pitch point X projects to p_t in frame t and p_{t+1} in
frame t+1.  X = H_t·p_t and p_t = A⁻¹·p_{t+1}, so X = H_t·A⁻¹·p_{t+1}, i.e.
H_{t+1} = H_t·A⁻¹.

Why a homography (not just affine) for A: a camera that only rotates and zooms
about its optical centre induces an EXACT homography between frames for all
scene points regardless of depth — so pitch and crowd features are mutually
consistent.  Broadcast main cameras are mast-mounted with little translation,
so this holds well.  Falls back to a similarity transform when the homography
solve is unstable.

Guards
──────
- Players/ball masked out of feature selection (independent motion).
- Scene-cut detection (Bhattacharyya histogram): can't propagate across a cut —
  hold H and flag it stale.
- Extreme inter-frame scale / too few inliers → hold H, reseed features.

Limitation: errors accumulate frame-to-frame (no global re-anchoring).  Good for
keeping a shot locked; the real fix for long sequences is a per-frame keypoint
model (v0.9).  This mutates HomographyManager.H in place so all downstream
consumers (minimap, visibility, analytics) stay current automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from gaffer import config
from gaffer.calibration.homography_manager import HomographyManager
from gaffer.detection.detector import Detection


@dataclass
class PropagationResult:
    updated:    bool          # was H advanced this frame?
    held:       bool          # was H held (cut / too few features)?
    scene_cut:  bool
    n_inliers:  int
    note:       str = ""


class HomographyPropagator:
    def __init__(
        self,
        mgr: HomographyManager,
        max_corners:   int   = config.HPROP_MAX_CORNERS,
        quality:       float = config.HPROP_QUALITY,
        min_distance:  int   = config.HPROP_MIN_DISTANCE,
        min_features:  int   = config.HPROP_MIN_FEATURES,
        win:           int   = config.HPROP_LK_WINSIZE,
        max_level:     int   = config.HPROP_LK_MAXLEVEL,
        mask_dilate:   int   = config.HPROP_MASK_DILATE_PX,
        max_scale:     float = config.HPROP_MAX_SCALE_STEP,
        cut_threshold: float = config.HPROP_SCENE_CUT_THRESHOLD,
    ):
        self._mgr = mgr
        self._feat = dict(maxCorners=max_corners, qualityLevel=quality,
                          minDistance=min_distance, blockSize=7)
        self._lk = dict(winSize=(win, win), maxLevel=max_level,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self._min_feat   = min_features
        self._mask_dilate = mask_dilate
        self._max_scale  = max_scale
        self._cut_thr    = cut_threshold

        self._prev_gray: np.ndarray | None = None
        self._prev_pts:  np.ndarray | None = None
        self.stale = False          # True once we've held across a cut without recalibration

        # cumulative stats
        self.n_updates = 0
        self.n_holds   = 0
        self.n_cuts    = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Forget tracked features so the next update() re-seeds from scratch
        via the existing first-call path, instead of trying to chain optical
        flow from a now-irrelevant previous frame. Call this whenever the
        caller snaps self._mgr.H to a different anchor (a new shot, or a
        scene cut with a closer anchor available) -- propagation should
        measure distance from THAT anchor, not silently continue accumulating
        from wherever it was."""
        self._prev_gray = None
        self._prev_pts = None

    def update(
        self,
        frame_bgr: np.ndarray,
        exclude_dets: list[Detection] | None = None,
    ) -> PropagationResult:
        """
        Advance H to match this frame's camera pose.  Call once per frame (every
        frame, not just detection frames — the camera moves continuously).
        `exclude_dets` are masked out of feature selection (players + ball).
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._reseed(gray, exclude_dets)
            return PropagationResult(False, False, False, 0, "seed")

        # Scene cut → cannot propagate; hold H, flag stale, reseed for next shot
        if self._is_scene_cut(self._prev_gray, gray):
            self.n_cuts += 1
            self.n_holds += 1
            self.stale = True
            self._reseed(gray, exclude_dets)
            return PropagationResult(False, True, True, 0, "scene_cut")

        if self._prev_pts is None or len(self._prev_pts) < self._min_feat:
            self._reseed(gray, exclude_dets)
            self.n_holds += 1
            return PropagationResult(False, True, False, 0, "few_features")

        nxt, st, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_pts, None, **self._lk)
        if nxt is None or st is None:
            self._reseed(gray, exclude_dets)
            self.n_holds += 1
            return PropagationResult(False, True, False, 0, "lk_failed")

        st = st.reshape(-1)
        good_old = self._prev_pts[st == 1]
        good_new = nxt[st == 1]

        if len(good_new) < self._min_feat:
            self._reseed(gray, exclude_dets)
            self.n_holds += 1
            return PropagationResult(False, True, False, len(good_new), "few_tracked")

        # Inter-frame image homography A: p_new = A · p_old
        A, inl = cv2.findHomography(good_old, good_new, cv2.RANSAC, 3.0)
        n_inl = int(inl.sum()) if inl is not None else 0

        if A is None or n_inl < self._min_feat or not self._sane(A):
            # Fall back to a similarity transform (translation + rotation + scale)
            M, inl2 = cv2.estimateAffinePartial2D(
                good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            if M is None:
                self._reseed(gray, exclude_dets)
                self.n_holds += 1
                return PropagationResult(False, True, False, n_inl, "no_transform")
            A = np.vstack([M, [0.0, 0.0, 1.0]])
            n_inl = int(inl2.sum()) if inl2 is not None else 0
            if not self._sane(A):
                self._reseed(gray, exclude_dets)
                self.n_holds += 1
                return PropagationResult(False, True, False, n_inl, "insane_transform")

        # H_new = H_old · A⁻¹
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            self._reseed(gray, exclude_dets)
            self.n_holds += 1
            return PropagationResult(False, True, False, n_inl, "singular")

        H_new = self._mgr.H @ A_inv
        if abs(H_new[2, 2]) > 1e-9:
            H_new = H_new / H_new[2, 2]          # renormalise homogeneous scale
        self._mgr.H = H_new

        self._reseed(gray, exclude_dets)         # detect-then-track-one-frame (robust)
        self.n_updates += 1
        return PropagationResult(True, False, False, n_inl, "ok")

    def summary(self) -> dict:
        return {
            "updates": self.n_updates,
            "holds":   self.n_holds,
            "cuts":    self.n_cuts,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _reseed(self, gray: np.ndarray, exclude_dets: list[Detection] | None) -> None:
        mask = self._build_mask(gray.shape, exclude_dets)
        self._prev_pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self._feat)
        self._prev_gray = gray

    def _build_mask(
        self, shape: tuple[int, int], exclude_dets: list[Detection] | None
    ) -> np.ndarray:
        """255 where features may be picked; 0 over players/ball (independent motion)."""
        h, w = shape[:2]
        mask = np.full((h, w), 255, dtype=np.uint8)
        if not exclude_dets:
            return mask
        d = self._mask_dilate
        for det in exclude_dets:
            x1, y1, x2, y2 = det.bbox
            x1 = max(0, int(x1) - d); y1 = max(0, int(y1) - d)
            x2 = min(w, int(x2) + d); y2 = min(h, int(y2) + d)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 0
        return mask

    def _sane(self, A: np.ndarray) -> bool:
        """Reject inter-frame transforms with implausible scale / perspective."""
        # Scale from the 2×2 linear part
        scale = np.sqrt(abs(np.linalg.det(A[:2, :2])))
        if not (1.0 / self._max_scale <= scale <= self._max_scale):
            return False
        # Perspective row should be near zero for a stable mast camera
        if abs(A[2, 0]) > 1e-2 or abs(A[2, 1]) > 1e-2:
            return False
        return True

    def _is_scene_cut(self, prev: np.ndarray, curr: np.ndarray) -> bool:
        try:
            ps = cv2.resize(prev, (160, 90)); cs = cv2.resize(curr, (160, 90))
            ph = cv2.calcHist([ps], [0], None, [32], [0, 256])
            ch = cv2.calcHist([cs], [0], None, [32], [0, 256])
            cv2.normalize(ph, ph); cv2.normalize(ch, ch)
            return cv2.compareHist(ph, ch, cv2.HISTCMP_BHATTACHARYYA) > self._cut_thr
        except Exception:
            return False
