"""
All configurable constants in one place.
Change values here; nothing else needs editing.
"""

from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
WEIGHTS_DIR = ROOT / "weights"
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"

DETECTION_MODEL_PATH = WEIGHTS_DIR / "yolov11_football.pt"
DETECTION_MODEL_OPENVINO_PATH = WEIGHTS_DIR / "yolov11_football_openvino"
PITCH_TEMPLATE_PATH = DATA_DIR / "pitch_template.png"
FORMATION_TEMPLATES_PATH = DATA_DIR / "formation_templates.json"

# ─── Pitch Dimensions ─────────────────────────────────────────────────────────
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

# Pitch-template / minimap drawing scale. Single source of truth shared by
# PitchModel, generate_pitch_template.py and the minimap so they never drift.
# A pitch point (x_m, y_m) maps to template pixel:
#   (PITCH_MARGIN_PX + x_m * PITCH_SCALE_PX_PER_M, PITCH_MARGIN_PX + y_m * PITCH_SCALE_PX_PER_M)
PITCH_SCALE_PX_PER_M = 10       # 1050x680 pitch at 10 px/m
PITCH_MARGIN_PX = 40            # padding around the pitch (room for goals at x<0 / x>105)

PITCH_KEYPOINTS = {
    "left_corner_top":          (0.0,    0.0),
    "left_corner_bottom":       (0.0,   68.0),
    "right_corner_top":         (105.0,  0.0),
    "right_corner_bottom":      (105.0, 68.0),
    "left_penalty_spot":        (11.0,  34.0),
    "right_penalty_spot":       (94.0,  34.0),
    "center_spot":              (52.5,  34.0),
    "left_goal_center":         (0.0,   34.0),
    "right_goal_center":        (105.0, 34.0),
    "halfway_line_top":         (52.5,   0.0),
    "halfway_line_bottom":      (52.5,  68.0),
    "left_penalty_box_tl":      (0.0,  13.85),
    "left_penalty_box_tr":      (16.5, 13.85),
    "left_penalty_box_br":      (16.5, 54.15),
    "left_penalty_box_bl":      (0.0,  54.15),
    "right_penalty_box_tl":     (88.5, 13.85),
    "right_penalty_box_tr":     (105.0,13.85),
    "right_penalty_box_br":     (105.0,54.15),
    "right_penalty_box_bl":     (88.5, 54.15),
    # Six-yard boxes (5.5m deep, 18.32m wide). Naming follows the pitch-diagram
    # convention: l/r = smaller/larger x, t/b = smaller/larger y.
    "left_six_yard_tl":         (0.0,  24.84),
    "left_six_yard_tr":         (5.5,  24.84),
    "left_six_yard_br":         (5.5,  43.16),
    "left_six_yard_bl":         (0.0,  43.16),
    "right_six_yard_tl":        (99.5, 24.84),
    "right_six_yard_tr":        (105.0,24.84),
    "right_six_yard_br":        (105.0,43.16),
    "right_six_yard_bl":        (99.5, 43.16),
    # Penalty-arc apex (furthest point of the "D" from goal, on y=34)
    "left_arc_apex":            (20.15, 34.0),
    "right_arc_apex":           (84.85, 34.0),
}

# ─── Detection ────────────────────────────────────────────────────────────────
DETECTION_CONF_THRESHOLD = 0.25  # player / GK / referee floor
BALL_CONF_THRESHOLD = 0.10       # ball is tiny (5-15px); needs lower floor than players
DETECTION_IOU_THRESHOLD = 0.45
DETECTION_IMG_SIZE = 640       # 640 for OpenVINO on Intel Arc; 1280 on NVIDIA
# Day 5 audit: every-frame detection INCREASED visible ID switching (carry-forward
# at skip-3 holds IDs steadier) and cost 3×. Kept skip-3.
DETECT_EVERY_N_FRAMES = 3

# ─── Ball tracking ────────────────────────────────────────────────────────────
BALL_MAX_INTERP_FRAMES = 15     # extrapolate/interpolate ball gaps up to 15 frames (~0.6s @ 25fps)
BALL_SMOOTH_ALPHA = 0.45        # EMA weight for new ball position (higher = more responsive)

# ─── Ball candidate filtering ─────────────────────────────────────────────────
# Stationary-object filter: penalty spots / white lines appear at same location
# every detection frame. A real ball is always moving.
BALL_STATIONARY_MOVE_PX = 8     # displacement below this counts as "not moved"
BALL_STATIONARY_MIN_FRAMES = 10 # reject if stationary for this many detection frames
BALL_STATIONARY_WINDOW = 30     # detection-frame window to check
# Player-proximity filter: a lone ball near the stands is almost certainly a false positive.
# Relaxed when the ball was moving fast (long kick / in-air).
BALL_PROXIMITY_THRESHOLD_PX = 450   # max pixel distance from nearest player
BALL_IN_AIR_SPEED_PX = 10           # px/detection-frame above which proximity is relaxed

# Spatial gate: reject any candidate that cannot be reached from the last known
# ball position at a physically plausible speed.
# Gate radius = BASE + prev_speed * SPEED_MULT + missed_dframes * MISS_GROWTH
# capped at MAX.  After a scene cut the gate is reset (no prior position assumed).
BALL_SPATIAL_GATE_BASE_PX    = 60   # minimum gate even for a stationary ball (jitter + slow touch)
BALL_SPATIAL_GATE_SPEED_MULT = 1.4  # gate grows with measured ball speed
BALL_SPATIAL_GATE_MISS_GROWTH = 30  # px added per missed detection frame (uncertainty grows)
BALL_SPATIAL_GATE_MAX_PX     = 320  # hard ceiling — no teleportation across the pitch
# Scene-cut detection: large histogram difference between consecutive frames
# means a camera cut; spatial gate is reset so we don't reject the first
# valid detection in the new shot.
BALL_SCENE_CUT_THRESHOLD    = 0.40  # normalised histogram diff above this → cut detected
# Recovery mode: after losing the ball for this many detection frames, relax
# the spatial gate and reacquire near the tightest player cluster.
BALL_RECOVERY_FRAMES        = 5     # detection frames of loss before recovery mode
BALL_CLUSTER_RADIUS_PX      = 150   # in recovery mode, candidate must be within this of cluster

# Ball state machine + approach voter scoring
# Players are tracked via ByteTrack IDs; their frame-to-frame velocity vectors
# vote for whichever ball candidate they are converging toward.
BALL_POSSESSION_DIST_PX        = 65    # px — nearest player foot → in possession
BALL_PASS_SPEED_PX_FRAME       = 12.0  # px/det-frame above which ball is in a pass
BALL_APPROACH_DIST_PX          = 300   # px — search radius for approach voters
BALL_APPROACH_DOT_THRESH       = 0.35  # dot product floor: "player moving toward ball"
BALL_SUSPECT_NO_APPROACH_FRAMES = 8    # consecutive det-frames with 0 voters → suspect
BALL_SUSPECT_MIN_TRACK_FRAMES  = 5     # don't declare suspect until tracked this long

# ─── Homography propagation (Problem B: keep H locked to pitch under motion) ───
# Lucas-Kanade tracks static-world features (pitch + crowd) frame-to-frame;
# players/ball are masked out (they move independently). Inter-frame image
# homography A is estimated by RANSAC, then H_new = H_old @ inv(A).
HPROP_MAX_CORNERS    = 300
HPROP_QUALITY        = 0.01
HPROP_MIN_DISTANCE   = 12
HPROP_MIN_FEATURES   = 25      # below this many tracked inliers → hold H, reseed
HPROP_LK_WINSIZE     = 21
HPROP_LK_MAXLEVEL    = 3
HPROP_MASK_DILATE_PX = 14      # dilate player/ball boxes by this when masking out
HPROP_MAX_SCALE_STEP = 1.25    # reject inter-frame H with |scale| outside [1/x, x]
HPROP_SCENE_CUT_THRESHOLD = 0.40   # Bhattacharyya hist diff above this → cut → hold H

# ─── Pitch visibility (Problem A: what part of the pitch is on screen) ─────────
# Project a grid of image points through H; keep those landing on-pitch; the
# convex hull of those points is the visible-pitch region. Robust to the horizon
# (image points above it project off-pitch and are simply dropped).
PITCH_VIS_GRID = 13                 # NxN grid of image points to project (169 pts)
PITCH_VIS_MARGIN_M = 2.0            # keep projections within pitch + this margin
PITCH_VIS_MIN_POINTS = 6           # below this many on-pitch points → view unreliable

CLASS_PLAYER = 0
CLASS_GOALKEEPER = 1
CLASS_REFEREE = 2
CLASS_BALL = 3

CLASS_NAMES = {
    CLASS_PLAYER: "player",
    CLASS_GOALKEEPER: "goalkeeper",
    CLASS_REFEREE: "referee",
    CLASS_BALL: "ball",
}

# ─── Tracking ─────────────────────────────────────────────────────────────────
# Day 5 audit verdict: supervision ByteTrack + skip-3 minimises ID *switches*
# (47 at conf .35) — BoT-SORT+CMC and every-frame detection both made switching
# worse (see docs/TRACKING_NOTES.md §0/§0b). conf .35 is the one tuning win.
TRACK_MIN_CONF = 0.35          # only players/GKs >= this enter tracking (47 vs 60 switches @ .25)
TRACK_THRESH = 0.25            # legacy alias; tracking activation now uses TRACK_MIN_CONF
TRACK_BUFFER_FRAMES = 45       # wall-clock frames a lost track survives (~1.8s @ 25fps)
# supervision ByteTrack uses IoU COST (1-IoU) as the matching threshold.
# cost <= MATCH_THRESH  ↔  IoU >= (1 - MATCH_THRESH);  0.7 → accepts IoU >= 0.3.
MATCH_THRESH = 0.7
DEFAULT_FPS = 25

# ─── Team Assignment ──────────────────────────────────────────────────────────
TEAM_KMEANS_CLUSTERS = 3
JERSEY_CROP_TOP = 0.10         # exclude head (top 10% of bbox)
JERSEY_CROP_BOTTOM = 0.20      # exclude shorts (bottom 20% of bbox)
JERSEY_MIN_SATURATION = 40     # HSV saturation floor — filters shadows/whites
JERSEY_MIN_VALUE = 40          # HSV value floor

# ─── Camera Motion ────────────────────────────────────────────────────────────
OPTICAL_FLOW_MAX_CORNERS = 200
OPTICAL_FLOW_QUALITY = 0.01
OPTICAL_FLOW_MIN_DISTANCE = 30
SCENE_CHANGE_HIST_THRESHOLD = 0.4   # histogram diff above this → scene cut

# ─── Spatial Analytics ────────────────────────────────────────────────────────
PRESSING_RADIUS_M = 10.0
SPRINT_THRESHOLD_MS = 7.0          # m/s ≈ 25 km/h
SPRINT_MIN_FRAMES = 3
FORMATION_WINDOW_FRAMES = 125      # classify formation every 5s at 25fps
VELOCITY_WINDOW_FRAMES = 5         # frames used to compute rolling velocity
MINIMAP_RENDER_EVERY_N = 5         # only re-render minimap every 5th frame

# ─── Event Detection ──────────────────────────────────────────────────────────
SHOT_SPEED_THRESHOLD_MS = 15.0
SHOT_GOAL_ANGLE_THRESHOLD = 0.7    # cosine similarity to goal direction
HIGH_PRESS_INTENSITY_THRESHOLD = 3
FORMATION_SHIFT_COOLDOWN_FRAMES = 125

# ─── Commentary ───────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:3b"        # use 3b on Intel Arc; 7b on NVIDIA
COMMENTARY_TEMPERATURE = 0.7
COMMENTARY_MAX_TOKENS = 120
REPORT_TEMPERATURE = 0.5
REPORT_MAX_TOKENS = 400

# ─── Output / Rendering ───────────────────────────────────────────────────────
OUTPUT_VIDEO_WIDTH = 1280          # write at 720p-equivalent width
OUTPUT_VIDEO_HEIGHT = 720
OUTPUT_VIDEO_FPS = 25
OUTPUT_VIDEO_CODEC = "avc1"        # H.264 -- mp4v isn't decodable by browsers' inline <video>

MINIMAP_WIDTH = 300
MINIMAP_HEIGHT = 200
MINIMAP_ALPHA = 0.85               # transparency when compositing onto frame

# BGR color palette
TEAM_A_COLOR_BGR = (0, 0, 220)     # red
TEAM_B_COLOR_BGR = (220, 0, 0)     # blue
BALL_COLOR_BGR = (0, 255, 255)     # yellow
REFEREE_COLOR_BGR = (30, 30, 30)   # dark grey
SPRINT_COLOR_BGR = (0, 165, 255)   # orange
DEF_LINE_COLOR_BGR = (0, 255, 0)   # green
VORONOI_A_COLOR = (0, 0, 80)       # dark red (semi-transparent overlay)
VORONOI_B_COLOR = (80, 0, 0)       # dark blue (semi-transparent overlay)
