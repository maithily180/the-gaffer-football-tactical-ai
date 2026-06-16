from __future__ import annotations

from typing import List, Sequence

import cv2
import numpy as np
from sklearn.cluster import KMeans

from gaffer import config
from gaffer.detection.detector import Detection


class TeamAssigner:
    """
    Assigns players to two teams using K-Means clustering on jersey colour.

    Pipeline per player:
      1. Crop the jersey region (avoids head and shorts which contaminate colour)
      2. Convert crop to HSV
      3. Filter out pitch-green and near-black pixels so they don't skew the mean
      4. Compute a single 3-element HSV feature vector (mean of remaining pixels)

    Global pipeline:
      - fit()   : collect features from multiple frames, run KMeans k=n_clusters
      - assign(): predict cluster for each player in a frame, map to team label

    Cluster → team mapping:
      KMeans with n_clusters=3 separates teamA, teamB, and referee (who wears a
      distinct colour). The smallest cluster by member count is assumed to be the
      referee cluster and gets team_id=-1. The two larger clusters become team 0
      and team 1.
    """

    # Classes that can receive a team label
    _PLAYER_CLASSES: frozenset[str] = frozenset({"player", "goalkeeper", "person"})

    def __init__(
        self,
        n_clusters: int    = config.TEAM_KMEANS_CLUSTERS,
        crop_top: float    = config.JERSEY_CROP_TOP,
        crop_bottom: float = config.JERSEY_CROP_BOTTOM,
        min_saturation: int = config.JERSEY_MIN_SATURATION,
        min_value: int     = config.JERSEY_MIN_VALUE,
    ):
        """
        n_clusters:     how many K-Means clusters (3 = teamA + teamB + referee)
        crop_top:       fraction of bbox height to skip at top (head region)
        crop_bottom:    fraction of bbox height to skip at bottom (shorts region)
        min_saturation: HSV S floor — pixels below this are near-grey, excluded
        min_value:      HSV V floor — pixels below this are shadows, excluded
        """
        self.n_clusters     = n_clusters
        self.crop_top       = crop_top
        self.crop_bottom    = crop_bottom
        self.min_saturation = min_saturation
        self.min_value      = min_value

        self._kmeans: KMeans | None                  = None
        self._cluster_to_team: dict[int, int]        = {}   # cluster_id → team_id or -1
        self._team_colors_bgr: dict[int, np.ndarray] = {}   # team_id → BGR display color
        self._n_fit_samples: int                     = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        frames: Sequence[np.ndarray],
        all_detections: Sequence[List[Detection]],
    ) -> None:
        """
        Collect jersey colour features from all provided (frame, detections) pairs
        and fit the K-Means model.

        Call this once on a handful of representative frames (e.g. 5–10 sampled
        from the action section of a clip). More frames = more robust clusters.
        """
        features: list[np.ndarray] = []

        for frame, detections in zip(frames, all_detections):
            for det in detections:
                if det.class_name not in self._PLAYER_CLASSES:
                    continue
                feat = self._jersey_feature(frame, det.bbox)
                if feat is not None:
                    features.append(feat)

        if len(features) < self.n_clusters:
            raise ValueError(
                f"Only {len(features)} valid jersey crops from {len(frames)} frame(s). "
                f"Need at least {self.n_clusters}. "
                "Try sampling frames from the action section of the clip."
            )

        X = np.array(features, dtype=np.float32)
        self._kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        self._kmeans.fit(X)
        self._n_fit_samples = len(features)

        # Build cluster → team_id mapping
        labels  = self._kmeans.labels_
        centers = self._kmeans.cluster_centers_
        counts  = np.bincount(labels, minlength=self.n_clusters)

        # Smallest cluster = referee (distinct kit, fewer people)
        sorted_by_count = np.argsort(counts)           # ascending
        referee_cluster = int(sorted_by_count[0])
        team_clusters   = [int(c) for c in sorted_by_count[1:]]  # two largest

        self._cluster_to_team = {referee_cluster: -1}
        for team_id, cluster_id in enumerate(team_clusters):
            self._cluster_to_team[cluster_id] = team_id

        # Store representative BGR colour for each team (for annotation)
        for team_id, cluster_id in enumerate(team_clusters):
            hsv_vec = centers[cluster_id].astype(np.uint8)
            bgr     = cv2.cvtColor(np.uint8([[hsv_vec]]), cv2.COLOR_HSV2BGR)[0, 0]
            self._team_colors_bgr[team_id] = bgr

    def assign(
        self,
        frame: np.ndarray,
        detections: List[Detection],
    ) -> List[Detection]:
        """
        Return a new list of Detections with team_id filled in.
        Non-player detections (ball, referee) pass through unchanged.
        Requires fit() to have been called first.
        """
        if self._kmeans is None:
            raise RuntimeError("Call fit() before assign().")

        out: List[Detection] = []
        for det in detections:
            if det.class_name not in self._PLAYER_CLASSES:
                out.append(det)
                continue

            feat = self._jersey_feature(frame, det.bbox)
            if feat is None:
                out.append(det)
                continue

            cluster = int(self._kmeans.predict(feat.reshape(1, -1))[0])
            out.append(det.with_team(self._cluster_to_team.get(cluster, -1)))

        return out

    def get_jersey_crop(self, frame: np.ndarray, bbox: tuple) -> np.ndarray | None:
        """
        Extract the jersey region from a bounding box.
        Public so the notebook can visualise what we're looking at.
        """
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        if h < 15:
            return None
        cy1 = y1 + int(h * self.crop_top)
        cy2 = y2 - int(h * self.crop_bottom)
        if cy2 <= cy1:
            return None
        crop = frame[cy1:cy2, x1:x2]
        return crop if crop.size > 0 else None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        return self._kmeans is not None

    @property
    def team_colors_bgr(self) -> dict[int, np.ndarray]:
        """BGR color representative of each team cluster center."""
        return dict(self._team_colors_bgr)

    @property
    def n_fit_samples(self) -> int:
        return self._n_fit_samples

    @property
    def cluster_to_team(self) -> dict[int, int]:
        return dict(self._cluster_to_team)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _jersey_feature(self, frame: np.ndarray, bbox: tuple) -> np.ndarray | None:
        """
        Extract a 3-element HSV feature vector from the jersey region.

        Filtering steps (in order):
          1. Discard pixels that are too dark (shadows don't tell us the kit colour)
          2. Discard pixels that look like pitch grass (H ≈ 35–85, high saturation)
          3. If fewer than 10 pixels survive, fall back to the unfiltered crop mean
             (handles white kits where saturation is low)
        """
        crop = self.get_jersey_crop(frame, bbox)
        if crop is None:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)

        # Step 1: remove dark pixels
        mask = hsv[:, 2] >= self.min_value

        # Step 2: remove grass-green pixels
        not_grass = ~(
            (hsv[:, 0] >= 35) & (hsv[:, 0] <= 85) & (hsv[:, 1] >= 50)
        )

        filtered = hsv[mask & not_grass]

        if len(filtered) < 10:
            # Fall back: use all pixels (needed for white/light-coloured kits)
            filtered = hsv

        return filtered.mean(axis=0)  # shape (3,): [H_mean, S_mean, V_mean]
