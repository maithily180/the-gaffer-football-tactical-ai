from __future__ import annotations

from pathlib import Path
from typing import Generator, Iterator

import cv2
import numpy as np

from gaffer import config


class VideoLoader:
    """
    Thin cv2.VideoCapture wrapper that yields (frame_idx, frame) tuples.

    Usage
    -----
    with VideoLoader("clip.mp4") as v:
        print(v.fps, v.width, v.height)
        for frame_idx, frame in v.frames(start=750, count=300):
            ...
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Clip not found: {self.path}")
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"cv2 could not open: {self.path}")

    # ── Metadata ──────────────────────────────────────────────────────────────

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) or config.DEFAULT_FPS

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def total_frames(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def duration_s(self) -> float:
        return self.total_frames / self.fps

    # ── Reading ───────────────────────────────────────────────────────────────

    def seek(self, frame_idx: int) -> None:
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    def read(self) -> tuple[bool, np.ndarray | None]:
        return self._cap.read()

    def frames(
        self,
        start: int = 0,
        count: int | None = None,
    ) -> Generator[tuple[int, np.ndarray], None, None]:
        """
        Yield (frame_idx, frame) starting at `start`, up to `count` frames.
        If count is None, reads to the end of the file.
        """
        self.seek(start)
        n = 0
        frame_idx = start
        while True:
            if count is not None and n >= count:
                break
            ret, frame = self._cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1
            n += 1

    def sample_frames(self, n: int, start: int = 0, count: int | None = None) -> list[np.ndarray]:
        """
        Read n evenly-spaced frames from [start, start+count).
        Used for fitting TeamAssigner.
        """
        end   = start + (count or (self.total_frames - start))
        idxs  = np.linspace(start, end - 1, n, dtype=int)
        frames: list[np.ndarray] = []
        for idx in idxs:
            self.seek(idx)
            ret, frame = self._cap.read()
            if ret:
                frames.append(frame)
        return frames

    # ── Context manager ───────────────────────────────────────────────────────

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> "VideoLoader":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __repr__(self) -> str:
        return (f"VideoLoader({self.path.name}  "
                f"{self.width}x{self.height}@{self.fps:.1f}fps  "
                f"{self.duration_s:.1f}s)")
