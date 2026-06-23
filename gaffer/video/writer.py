from __future__ import annotations

import struct
from pathlib import Path

import cv2
import numpy as np

from gaffer import config

# mp4 box types that can contain other boxes -- the only ones worth
# recursing into when hunting for stco/co64 inside moov.
_CONTAINER_BOX_TYPES = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"mvex", b"udta", b"dinf"}


def _patch_chunk_offsets(buf: bytearray, start: int, end: int, delta: int) -> None:
    """Recursively walk mp4 boxes in buf[start:end], adding `delta` to every
    chunk-offset entry in any stco (32-bit) / co64 (64-bit) box found."""
    pos = start
    while pos + 8 <= end:
        size = struct.unpack(">I", buf[pos:pos + 4])[0]
        typ = bytes(buf[pos + 4:pos + 8])
        if size < 8:
            break
        if typ in _CONTAINER_BOX_TYPES:
            _patch_chunk_offsets(buf, pos + 8, pos + size, delta)
        elif typ == b"stco":
            n = struct.unpack(">I", buf[pos + 12:pos + 16])[0]
            for i in range(n):
                off = pos + 16 + i * 4
                val = struct.unpack(">I", buf[off:off + 4])[0]
                struct.pack_into(">I", buf, off, val + delta)
        elif typ == b"co64":
            n = struct.unpack(">I", buf[pos + 12:pos + 16])[0]
            for i in range(n):
                off = pos + 16 + i * 8
                val = struct.unpack(">Q", buf[off:off + 8])[0]
                struct.pack_into(">Q", buf, off, val + delta)
        pos += size


def _faststart(path: Path) -> None:
    """Rewrite an mp4 so the `moov` box (duration/seek metadata) sits before
    `mdat` (frame data) instead of after it. cv2's ffmpeg-backed VideoWriter
    always writes moov last -- fine for a CLI artifact opened in a desktop
    player, but browsers can't report duration or seek until the whole file
    is fetched, which is exactly the "0:00 / NaN:NaN, won't play until fully
    downloaded" symptom Gradio's inline <video> hit. No ffmpeg binary is
    available in this environment to do this the normal way (`-movflags
    +faststart`), so this reimplements the same moov-relocation algorithm
    directly: move moov in front of mdat, then patch every absolute
    chunk-offset in its stco/co64 tables by however far mdat just shifted."""
    data = bytearray(path.read_bytes())

    boxes = []
    pos = 0
    while pos + 8 <= len(data):
        size = struct.unpack(">I", data[pos:pos + 4])[0]
        typ = data[pos + 4:pos + 8]
        if size == 0:
            size = len(data) - pos
        if size < 8:
            break
        boxes.append((typ, pos, size))
        pos += size

    moov = next((b for b in boxes if b[0] == b"moov"), None)
    mdat = next((b for b in boxes if b[0] == b"mdat"), None)
    if moov is None or mdat is None or moov[1] < mdat[1]:
        return  # unexpected layout, or already faststart -- leave untouched

    _, moov_start, moov_size = moov
    _, mdat_start, _ = mdat

    moov_bytes = bytearray(data[moov_start:moov_start + moov_size])
    _patch_chunk_offsets(moov_bytes, 8, moov_size, delta=moov_size)

    new_data = (bytes(data[:mdat_start]) + bytes(moov_bytes)
                + bytes(data[mdat_start:moov_start]) + bytes(data[moov_start + moov_size:]))
    path.write_bytes(new_data)


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
        if self._n > 0 and self.path.suffix.lower() == ".mp4":
            _faststart(self.path)

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()
