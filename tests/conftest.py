"""
Shared pytest fixtures.

Unit tests use only synthetic data — no real clips or weights required.
Integration tests are skipped automatically if weights or clips are missing.
"""

import numpy as np
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent


# ─── Synthetic frames ─────────────────────────────────────────────────────────

@pytest.fixture
def green_frame():
    """720p green pitch frame with white centre line. No real players."""
    import cv2
    frame = np.full((720, 1280, 3), (34, 139, 34), dtype=np.uint8)
    cv2.line(frame, (640, 0), (640, 720), (255, 255, 255), 3)
    return frame


@pytest.fixture
def blank_frame():
    """Plain black frame — use when content doesn't matter."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


# ─── Synthetic player positions ───────────────────────────────────────────────

@pytest.fixture
def symmetric_positions():
    """
    22 players in symmetric layout on a 105×68m pitch.
    Team A occupies left half, Team B right half.
    Voronoi should split 50/50.
    """
    team_a = np.array([
        [10, 34], [25, 10], [25, 28], [25, 40], [25, 58],
        [40, 20], [40, 34], [40, 48],
        [60, 14], [60, 34], [60, 54],
    ], dtype=float)
    team_b = 105 - team_a.copy()
    team_b[:, 1] = team_a[:, 1]  # mirror x only
    return team_a, team_b


@pytest.fixture
def compact_defense():
    """Team B in a tight defensive block near their own goal."""
    return np.array([
        [85, 34],   # GK
        [78, 15], [78, 30], [78, 38], [78, 53],   # 4 defenders
        [72, 20], [72, 34], [72, 48],              # 3 midfielders
        [65, 15], [65, 34], [65, 53],              # 3 forwards
    ], dtype=float)


@pytest.fixture
def ball_position():
    """Ball in centre of pitch."""
    return np.array([52.5, 34.0])


# ─── Mock detections ──────────────────────────────────────────────────────────

@pytest.fixture
def mock_player_detections():
    """
    sv.Detections with 6 player bounding boxes (3 per team visually).
    No real image needed.
    """
    import supervision as sv
    return sv.Detections(
        xyxy=np.array([
            [100, 200, 160, 280],   # player 1
            [300, 150, 360, 230],   # player 2
            [500, 300, 560, 380],   # player 3
            [700, 200, 760, 280],   # player 4
            [900, 150, 960, 230],   # player 5
            [1100, 300, 1160, 380], # player 6
        ], dtype=np.float32),
        confidence=np.array([0.9, 0.85, 0.8, 0.88, 0.82, 0.79]),
        class_id=np.array([0, 0, 0, 0, 0, 0]),
    )


# ─── Skip guards for integration tests ───────────────────────────────────────

def weights_available() -> bool:
    return (ROOT / "weights" / "yolov11_football.pt").exists()


def clip_available() -> bool:
    clips = list((ROOT / "data" / "test_clips").glob("*.mp4"))
    return len(clips) > 0


def ollama_available() -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2):
            return True
    except Exception:
        return False


requires_weights = pytest.mark.skipif(
    not weights_available(),
    reason="weights/yolov11_football.pt not found — run Colab training first",
)

requires_clip = pytest.mark.skipif(
    not clip_available(),
    reason="No .mp4 found in data/test_clips/ — download one with scripts/download_clip.py",
)

requires_ollama = pytest.mark.skipif(
    not ollama_available(),
    reason="Ollama not running — start with: ollama serve",
)
