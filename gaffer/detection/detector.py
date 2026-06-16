from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import List

import numpy as np
from ultralytics import YOLO

from gaffer import config


@dataclass
class Detection:
    """
    One detected entity in a frame.
    Uses the football class IDs from config (CLASS_PLAYER=0, CLASS_BALL=3, etc.)
    regardless of whether the underlying model is COCO or fine-tuned.
    team_id is -1 until TeamAssigner fills it in.
    """
    bbox: tuple         # (x1, y1, x2, y2) pixel ints
    confidence: float
    class_id: int       # gaffer CLASS_* constant
    class_name: str     # "player" | "goalkeeper" | "referee" | "ball"
    team_id: int = -1   # 0=teamA  1=teamB  -1=unassigned
    track_id: int = -1  # stable ByteTrack ID; -1 until tracker assigns one

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return self.width * self.height

    def with_team(self, team_id: int) -> "Detection":
        return replace(self, team_id=team_id)


class FootballDetector:
    """
    Wraps YOLOv11 for football detection with per-frame caching.

    Inference runs only every detect_every_n frames; the last result is
    returned unchanged for intervening frames. This trades a small amount of
    temporal accuracy for ~3× compute savings on 25fps video.

    Works with both the base COCO model (yolo11n.pt) and a fine-tuned
    football model. Auto-detects which one is loaded and maps classes
    accordingly.
    """

    # COCO class-id → (football class_id, name)
    _COCO_MAP: dict[int, tuple[int, str]] = {
        0:  (config.CLASS_PLAYER, "player"),
        32: (config.CLASS_BALL,   "ball"),
    }

    def __init__(
        self,
        model_path: str | Path | None = None,
        conf: float = config.DETECTION_CONF_THRESHOLD,
        imgsz: int = config.DETECTION_IMG_SIZE,
        detect_every_n: int = config.DETECT_EVERY_N_FRAMES,
        verbose: bool = False,
    ):
        """
        model_path: path to .pt file. If None, prefers the fine-tuned football
                    model from config; falls back to yolo11n.pt in the repo root.
        """
        if model_path is None:
            fine_tuned = config.DETECTION_MODEL_PATH
            model_path = fine_tuned if fine_tuned.exists() else config.ROOT / "yolo11n.pt"

        self.model_path   = Path(model_path)
        self.conf         = conf
        self.imgsz        = imgsz
        self.detect_every_n = detect_every_n
        self._verbose     = verbose

        self._model = YOLO(str(self.model_path))

        self._is_football_model: bool = bool(
            {"player", "goalkeeper", "referee", "ball"}
            & set(self._model.names.values())
        )

        self._cache: List[Detection] = []
        self._last_detect_idx: int   = -(detect_every_n + 1)  # force run on first call

        # Warm-up so first real call isn't slow
        _dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self._model(_dummy, conf=self.conf, imgsz=self.imgsz, verbose=False)

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray, frame_idx: int) -> List[Detection]:
        """
        Return detections for this frame.
        Runs inference only when frame_idx is at least detect_every_n ahead of
        the last inference; otherwise returns the cached result from that run.
        """
        if frame_idx - self._last_detect_idx >= self.detect_every_n:
            results = self._model(
                frame, conf=self.conf, imgsz=self.imgsz, verbose=False
            )[0]
            self._cache = self._parse(results)
            self._last_detect_idx = frame_idx
        return self._cache

    def detect_raw(self, frame: np.ndarray) -> List[Detection]:
        """Force inference regardless of frame index. Useful for single-frame analysis."""
        results = self._model(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        return self._parse(results)

    @property
    def model_type(self) -> str:
        return "football" if self._is_football_model else "coco"

    @property
    def cache(self) -> List[Detection]:
        return list(self._cache)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _parse(self, results) -> List[Detection]:
        detections: List[Detection] = []

        if results.boxes is None or len(results.boxes) == 0:
            return detections

        for i in range(len(results.boxes)):
            raw_cls = int(results.boxes.cls[i].item())
            conf    = float(results.boxes.conf[i].item())
            x1, y1, x2, y2 = results.boxes.xyxy[i].numpy().astype(int)

            if self._is_football_model:
                if raw_cls not in config.CLASS_NAMES:
                    continue
                cls_id   = raw_cls
                cls_name = config.CLASS_NAMES[cls_id]
            else:
                # Base COCO model — only keep person and sports ball
                if raw_cls not in self._COCO_MAP:
                    continue
                cls_id, cls_name = self._COCO_MAP[raw_cls]

            detections.append(Detection(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                confidence=conf,
                class_id=cls_id,
                class_name=cls_name,
            ))

        return detections
