"""
Integration tests — require weights/yolov11_football.pt and a test clip.
Automatically skipped if either is missing.
"""
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

# Import skip guards from conftest
from tests.conftest import requires_weights, requires_clip


@requires_weights
@requires_clip
class TestDetectionPipeline:
    def test_detects_at_least_ten_players(self):
        import cv2
        from gaffer.detection.detector import FootballDetector
        from gaffer import config

        detector = FootballDetector(config.DETECTION_MODEL_PATH)

        clips = list((ROOT / "data" / "test_clips").glob("*.mp4"))
        clip = clips[0]

        cap = cv2.VideoCapture(str(clip))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
        ret, frame = cap.read()
        cap.release()

        assert ret, f"Could not read frame from {clip}"

        detections = detector.detect(frame, frame_idx=0, detect_every=1)
        players = (detections.class_id == 0).sum()
        assert players >= 10, (
            f"Expected ≥10 players, got {players}. "
            "Check model quality or conf threshold."
        )

    @requires_weights
    def test_frame_skip_cache_works(self):
        import cv2
        import numpy as np
        from gaffer.detection.detector import FootballDetector
        from gaffer import config

        detector = FootballDetector(config.DETECTION_MODEL_PATH)

        clips = list((ROOT / "data" / "test_clips").glob("*.mp4"))
        cap = cv2.VideoCapture(str(clips[0]))

        results = []
        for frame_idx in range(9):
            ret, frame = cap.read()
            if not ret:
                break
            det = detector.detect(frame, frame_idx, detect_every=3)
            results.append(len(det))

        cap.release()

        # Frames 0, 3, 6 trigger detection; 1,2,4,5,7,8 use cache
        # All frames should have the same count within a window
        assert results[0] == results[1] == results[2], (
            "Cached detections should equal previous detected frame"
        )
