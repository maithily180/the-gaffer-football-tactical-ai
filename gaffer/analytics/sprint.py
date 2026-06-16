"""Sprint detection."""

from gaffer.config import SPRINT_THRESHOLD_MS, SPRINT_MIN_FRAMES


class SprintDetector:
    """
    Stateful per-track sprint detector.
    A player is sprinting when their speed exceeds SPRINT_THRESHOLD_MS
    for at least SPRINT_MIN_FRAMES consecutive frames.
    """

    def __init__(
        self,
        threshold_ms: float = SPRINT_THRESHOLD_MS,
        min_frames: int = SPRINT_MIN_FRAMES,
    ):
        self.threshold_ms = threshold_ms
        self.min_frames = min_frames
        self._consecutive: dict[int, int] = {}

    def update(self, track_id: int, speed_ms: float) -> bool:
        """
        Update sprint state for a track.

        Returns:
            True if the player is currently in a sprint, False otherwise.
        """
        if speed_ms >= self.threshold_ms:
            self._consecutive[track_id] = self._consecutive.get(track_id, 0) + 1
        else:
            self._consecutive[track_id] = 0

        return self._consecutive.get(track_id, 0) >= self.min_frames

    def reset_track(self, track_id: int) -> None:
        """Remove a track that has been lost."""
        self._consecutive.pop(track_id, None)

    def sprinting_tracks(self) -> list[int]:
        """Return list of track_ids currently in sprint state."""
        return [tid for tid, n in self._consecutive.items() if n >= self.min_frames]
