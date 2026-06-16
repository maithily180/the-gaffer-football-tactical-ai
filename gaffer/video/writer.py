from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from gaffer import config


class VideoWriter:
    """
    Thin cv2.VideoWriter wrapper.

    Usage
    -----
    with VideoWriter("out.mp4", fps=25.0, width=1280, height=720) as w:
        for frame in ...:
            w.write(frame)
    print(w.frames_written)
    """

    def __init__(
        self,
        path: str | Path,
        fps: float,
        width: int,
        height: int,
        codec: str = config.OUTPUT_VIDEO_CODEC,
    ):
        self.path   = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fps   = fps
        self._size  = (width, height)
        self._fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(
            str(self.path), self._fourcc, fps, (width, height)
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter could not open: {self.path}")
        self._n = 0

    # ── Write ─────────────────────────────────────────────────────────────────

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)
        self._n += 1

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def frames_written(self) -> int:
        return self._n

    @property
    def size_mb(self) -> float:
        if self.path.exists():
            return self.path.stat().st_size / 1024 ** 2
        return 0.0

    # ── Context manager ───────────────────────────────────────────────────────

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()
