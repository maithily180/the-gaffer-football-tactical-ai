"""
gaffer/analytics/engine.py
───────────────────────────
The football-intelligence orchestrator.

Takes per-frame Detection lists (with team_id + track_id) and a calibrated
HomographyManager, and turns them into geometric football facts:

    raw detections (pixels)
        │  project foot points through H
        ▼
    player positions (pitch metres), grouped by team
        │
        ├── team compactness   (width / depth / hull area / centroid)
        ├── attack direction   (inferred from relative centroids)
        ├── defensive line     (backline height from own goal)
        ├── possession         (nearest player to ball, running %)
        ├── pressing intensity (opponents within radius of ball carrier)
        └── Voronoi control    (space % per team)
        ▼
    AnalyticsSnapshot  (one per detection frame)

No LLM, no opinions — pure geometry.  Interpretation happens in later layers.

Attack-direction heuristic
───────────────────────────
Within one broadcast clip the two teams occupy opposite halves on average, so
the team whose centroid sits at smaller x is attacking toward x=105 (+1) and
the other toward x=0 (-1).  This is a heuristic: it can be wrong during heavy
one-sided pressure, so consumers should treat def_line_m as approximate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from gaffer import config
from gaffer.analytics.compactness import Compactness, team_compactness
from gaffer.analytics.defensive_line import compute_defensive_line
from gaffer.analytics.possession import PossessionState, PossessionTracker
from gaffer.analytics.pressing import compute_pressing_intensity
from gaffer.analytics.voronoi import compute_voronoi_control
from gaffer.calibration.homography_manager import HomographyManager
from gaffer.calibration.pitch_visibility import PitchVisibility, PitchVisibilityEstimator
from gaffer.detection.detector import Detection
from gaffer.events.base import FootballEvent
from gaffer.events.detector import EventDetector


@dataclass
class TeamShape:
    team:        str                       # "teamA" | "teamB"
    compactness: Compactness
    attack_dir:  int                       # +1 toward x=105, -1 toward x=0, 0 unknown
    def_line_m:  float | None              # backline height measured from own goal

    def as_dict(self) -> dict:
        return {
            "team":       self.team,
            "attack_dir": self.attack_dir,
            "def_line_m": (round(self.def_line_m, 1) if self.def_line_m is not None else None),
            **self.compactness.as_dict(),
        }


@dataclass
class AnalyticsSnapshot:
    frame_idx:          int
    ball_xy:            tuple[float, float] | None
    team_a:             TeamShape
    team_b:             TeamShape
    possession:         PossessionState
    voronoi:            dict
    pressing:           dict | None = None
    positions:          dict = field(default_factory=dict)       # {"teamA":[(x,y)..], "teamB":[..]}
    player_positions_m: dict = field(default_factory=dict)       # track_id -> (x, y)
    player_teams:       dict = field(default_factory=dict)       # track_id -> "teamA"|"teamB"
    visibility:         PitchVisibility | None = None
    ball_region:        str | None = None
    events:             List[FootballEvent] = field(default_factory=list)


class PitchAnalyticsEngine:
    """
    Per-frame football analytics.  Construct once with a calibrated
    HomographyManager, then call update() on every detection frame.
    """

    def __init__(
        self,
        homography_manager: HomographyManager,
        fps: float = config.DEFAULT_FPS,
        pressing_radius_m: float = config.PRESSING_RADIUS_M,
        image_size: tuple[int, int] | None = None,    # (w, h) → enables visibility
    ):
        self.mgr  = homography_manager
        self._fps = fps
        self._press_r = pressing_radius_m
        self._possession = PossessionTracker()
        self._event_detector = EventDetector(fps=fps)
        self._last: AnalyticsSnapshot | None = None
        self._vis_est = (
            PitchVisibilityEstimator(image_size[0], image_size[1])
            if image_size is not None else None
        )
        self._vis_cache: PitchVisibility | None = None
        self._vis_cache_key: int | None = None      # id of the H it was computed for

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, frame_idx: int, detections: List[Detection]) -> AnalyticsSnapshot | None:
        """
        Compute an AnalyticsSnapshot for this frame.  Returns None if the
        homography is invalid (no pitch coordinates available).
        """
        if not self.mgr.is_valid():
            return None

        team_a_pos, team_b_pos, player_pos_m, player_teams = self._project_players(detections)
        ball_xy = self._project_ball(detections)

        # Attack directions from relative centroids (heuristic, see module doc)
        dir_a, dir_b = self._attack_directions(team_a_pos, team_b_pos)

        shape_a = self._team_shape("teamA", team_a_pos, dir_a)
        shape_b = self._team_shape("teamB", team_b_pos, dir_b)

        poss = self._possession.update(ball_xy, team_a_pos, team_b_pos)

        voronoi = compute_voronoi_control(team_a_pos, team_b_pos)

        pressing = None
        if ball_xy is not None and poss.owner is not None:
            pressing = compute_pressing_intensity(
                ball_xy, poss.owner,
                {"teamA": team_a_pos, "teamB": team_b_pos},
                radius_m=self._press_r,
            )

        # Problem A — what part of the pitch is on screen + which third the ball is in.
        # Visibility is a pure function of H; cache it and recompute only when H
        # changes (i.e. once dynamic-H / Step 2 lands). Static H → computed once.
        visibility = self._visibility()
        ball_region = (
            self._vis_est.region_of(ball_xy[0])
            if (self._vis_est and ball_xy is not None) else None
        )

        snap = AnalyticsSnapshot(
            frame_idx          = frame_idx,
            ball_xy            = ball_xy,
            team_a             = shape_a,
            team_b             = shape_b,
            possession         = poss,
            voronoi            = voronoi,
            pressing           = pressing,
            positions          = {"teamA": team_a_pos, "teamB": team_b_pos},
            player_positions_m = player_pos_m,
            player_teams       = player_teams,
            visibility         = visibility,
            ball_region        = ball_region,
        )
        snap.events = self._event_detector.update(snap)
        self._last = snap
        return snap

    @property
    def last(self) -> AnalyticsSnapshot | None:
        return self._last

    def _visibility(self) -> PitchVisibility | None:
        if self._vis_est is None or not self.mgr.is_valid():
            return None
        key = id(self.mgr.H)
        if key != self._vis_cache_key:
            self._vis_cache = self._vis_est.estimate(self.mgr)
            self._vis_cache_key = key
        return self._vis_cache

    def possession_summary(self) -> dict:
        return self._possession.summary()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _project_players(
        self, detections: List[Detection]
    ) -> tuple[
        list[tuple[float, float]],
        list[tuple[float, float]],
        dict[int, tuple[float, float]],
        dict[int, str],
    ]:
        """
        Project player/GK foot points to pitch metres.

        Returns (team_a_list, team_b_list, per_player_positions, per_player_teams).
        per_player_positions maps track_id → (x_m, y_m) for tracked players.
        """
        team_a: list[tuple[float, float]] = []
        team_b: list[tuple[float, float]] = []
        player_pos:   dict[int, tuple[float, float]] = {}
        player_teams: dict[int, str] = {}

        for det in detections:
            if det.class_name not in ("player", "goalkeeper"):
                continue
            if det.team_id not in (0, 1):
                continue
            world = self.mgr.project(det.foot_point)
            if world is None or not self.mgr.on_pitch(*world, margin_m=2.0):
                continue
            team_name = "teamA" if det.team_id == 0 else "teamB"
            (team_a if det.team_id == 0 else team_b).append(world)
            if det.track_id is not None and det.track_id >= 0:
                player_pos[det.track_id]   = world
                player_teams[det.track_id] = team_name

        return team_a, team_b, player_pos, player_teams

    def _project_ball(self, detections: List[Detection]) -> tuple[float, float] | None:
        for det in detections:
            if det.class_name == "ball":
                world = self.mgr.project(det.center)
                if world is not None and self.mgr.on_pitch(*world, margin_m=3.0):
                    return world
        return None

    @staticmethod
    def _attack_directions(
        team_a_pos: list[tuple[float, float]],
        team_b_pos: list[tuple[float, float]],
    ) -> tuple[int, int]:
        """Team with smaller mean-x attacks toward +x. 0/0 if undecidable."""
        if not team_a_pos or not team_b_pos:
            return 0, 0
        mean_ax = sum(p[0] for p in team_a_pos) / len(team_a_pos)
        mean_bx = sum(p[0] for p in team_b_pos) / len(team_b_pos)
        if abs(mean_ax - mean_bx) < 1.0:           # too close to call
            return 0, 0
        if mean_ax < mean_bx:
            return +1, -1
        return -1, +1

    def _team_shape(
        self,
        team: str,
        positions: list[tuple[float, float]],
        attack_dir: int,
    ) -> TeamShape:
        comp = team_compactness(positions)
        def_line = self._defensive_line(positions, attack_dir)
        return TeamShape(team=team, compactness=comp, attack_dir=attack_dir,
                         def_line_m=def_line)

    def _defensive_line(
        self,
        positions: list[tuple[float, float]],
        attack_dir: int,
    ) -> float | None:
        """
        Backline height measured from the team's OWN goal (metres up-pitch).

        Attacking +x → own goal at x=0   → backline = 2nd-smallest x → height = x
        Attacking -x → own goal at x=105 → backline = 2nd-largest  x → height = 105 - x
        """
        if attack_dir == 0 or len(positions) < 2:
            return None
        xs = sorted(p[0] for p in positions)
        if attack_dir == +1:
            backline_x = xs[1]                      # 2nd-deepest toward x=0
            return backline_x
        backline_x = xs[-2]                         # 2nd-deepest toward x=105
        return config.PITCH_LENGTH_M - backline_x
