"""Defensive line tracker."""


def compute_defensive_line(
    team_positions_y: list[float],
    ball_y: float,
    pitch_h: float = 68.0,
) -> float | None:
    """
    Return the y-coordinate of the defensive line (second-to-last defender).

    The defensive line is conventionally the 2nd-deepest outfield player
    (last being the goalkeeper). Lower y = closer to left goal.

    Args:
        team_positions_y: list of y-coordinates for all defending team players
        ball_y: current ball y-coordinate (unused in simple version, kept for API)
        pitch_h: pitch width in metres

    Returns:
        float y-coordinate of defensive line, or None if < 2 players
    """
    if len(team_positions_y) < 2:
        return None
    sorted_y = sorted(team_positions_y)
    return sorted_y[-2]
