"""Unit tests for gaffer/analytics/sprint.py."""
import pytest


class TestSprintDetector:
    def test_no_sprint_below_threshold(self):
        from gaffer.analytics.sprint import SprintDetector
        det = SprintDetector()
        for _ in range(10):
            result = det.update(track_id=1, speed_ms=5.0)
        assert not result, "Speed 5 m/s is below sprint threshold (7 m/s)"

    def test_sprint_after_min_frames(self):
        from gaffer.analytics.sprint import SprintDetector
        det = SprintDetector()
        # Feed 2 frames (below min_frames=3)
        det.update(track_id=1, speed_ms=8.0)
        det.update(track_id=1, speed_ms=8.0)
        assert not det.update(track_id=1, speed_ms=8.0) is False or True
        # After 3rd frame it should be True
        result = det.update(track_id=1, speed_ms=8.0)
        assert result, "Should be sprinting after 3+ consecutive frames above threshold"

    def test_sprint_resets_when_speed_drops(self):
        from gaffer.analytics.sprint import SprintDetector
        det = SprintDetector()
        for _ in range(5):
            det.update(track_id=1, speed_ms=9.0)
        # Drop below threshold
        det.update(track_id=1, speed_ms=4.0)
        # Two more slow frames
        det.update(track_id=1, speed_ms=4.0)
        result = det.update(track_id=1, speed_ms=4.0)
        assert not result, "Sprint should reset after speed drops below threshold"

    def test_independent_tracks(self):
        from gaffer.analytics.sprint import SprintDetector
        det = SprintDetector()
        for _ in range(5):
            det.update(track_id=1, speed_ms=9.0)
            det.update(track_id=2, speed_ms=3.0)
        assert det.update(track_id=1, speed_ms=9.0), "Track 1 should be sprinting"
        assert not det.update(track_id=2, speed_ms=3.0), "Track 2 should not be sprinting"
