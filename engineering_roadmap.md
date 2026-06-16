THE GAFFER — Engineering Roadmap
Lead Technical Architect Review | June 2026
SECTION 1 — Feasibility Analysis
Is the project technically feasible on your hardware?
Yes, with one significant caveat: your GPU is Intel Arc integrated graphics with 128 MB of dedicated VRAM, not a discrete card. This changes almost every performance estimate in the blueprint. The blueprint already acknowledges this in Section 3.2.1 and recommends mitigations. Those mitigations are correct and sufficient. You will build a slower but fully functional system.

Realistic end-to-end time on your machine:

Stage	Blueprint estimate	Realistic on your machine
Detection (OpenVINO, 640px, every 3rd frame)	40–80s	60–120s
ByteTrack + camera motion	8s	8–12s
Spatial analytics	5s	5–8s
Qwen2.5:3b commentary (~8 events)	55s	90–150s (CPU inference likely)
Video annotation + write	30s	30–45s
Total for 60-second clip	~3 min	~4–6 min
This is within the 5-minute objective stated in OBJ-05. You are fine.

What runs locally vs cloud?
Component	Runs Locally	Notes
YOLOv11 inference (OpenVINO)	Yes	Export to OpenVINO format first
ByteTrack	Yes	Pure CPU, no GPU needed
K-Means team assignment	Yes	Trivial
OpenCV optical flow	Yes	CPU, fast
Homography computation	Yes	CPU, fast
scipy spatial analytics	Yes	CPU, fast
Qwen2.5:3b via Ollama	Yes	CPU or Intel GPU backend
YOLOv11 fine-tuning	No — use Google Colab	Your machine has no CUDA. Fine-tuning on CPU would take 8–12 hours for 100 epochs. Colab T4 does it in 2 hours for free.
Gradio UI	Yes	localhost
Bottlenecks (ranked by severity)
1. YOLOv11 inference speed — This is your hardest constraint. Without OpenVINO export and the every-3rd-frame trick, detection alone takes 5–8 minutes per clip on CPU. With both, you get under 2 minutes. Do this on Day 1, not later.

2. Qwen2.5 LLM inference — Ollama may not successfully route to your Intel Arc GPU (the OLLAMA_INTEL_GPU=1 flag works on some Arc machines but not all). Be prepared for CPU inference at 4–8 tokens/second, meaning a 100-token response takes 12–25 seconds per event. With 8 events, that is 2–3 minutes just for commentary. Use qwen2.5:3b, not 7B. Consider batching all commentary at the end rather than inline.

3. Homography — This is the hardest engineering problem (not hardware). The manual 4-point method in Phase 1 is the right starting point. Do not attempt automatic keypoint detection until the rest of the system works.

4. Video write speed — OpenCV writing annotated 1080p frames at 25fps is I/O heavy. Write at 720p instead. Cuts output time by ~40%.

What is genuinely difficult vs just engineering work?
Genuinely difficult (requires iteration and tuning):

Homography stability during camera pan/zoom — even with optical flow compensation, fast pans will break it
Ball detection reliability — you will get ~55–65% recall at 640px on your device. Accept this.
Team assignment with similar-coloured kits — the K-Means approach fails on roughly 1 in 10 match combinations
Getting Qwen commentary to be specific rather than generic — prompt engineering takes real time
Just engineering work (do it, no surprises):

ByteTrack integration
Voronoi / pressing / compactness computation
Sprint detection
Gradio UI
Session output structure
All testing infrastructure
SECTION 2 — MVP Decomposition
Version 0.1 — Smallest Working Demo
Target: Day 4

Features included:

YOLOv11 detection (base pretrained weights, no fine-tuning yet)
ByteTrack tracking with consistent IDs
K-Means team assignment (two colored bounding boxes)
Annotated video output (boxes + team colors + IDs)
No minimap, no analytics, no commentary
Features excluded: Everything else

Implementation time: 3–4 days

Main risks: Getting OpenVINO export working on Windows. Use WSL2 to avoid Windows-specific path issues.

Success definition: Upload any football clip, get back a video with red/blue boxes and player IDs that stay consistent. Demo-able in 60 seconds to anyone.

Version 0.5 — Usable Prototype
Target: Day 10

Features included:

Everything in 0.1
Manual 4-point homography + minimap overlay
Voronoi space control (visual on minimap)
Defensive line visualization
Pressing intensity meter (live on video)
Basic Gradio UI (upload → annotated video out)
Features excluded: Commentary, sprint detection, formation classification, event log, match report

Implementation time: 6 more days (Days 5–10)

Main risks: Homography accuracy on clips with moving cameras. The minimap will look wrong when the camera pans. Accept this for 0.5.

Success definition: The minimap shows approximately correct player positions for the portions of the video where the camera is roughly static.

Version 1.0 — College-Project-Grade Product
Target: Day 21

Features included:

Everything in 0.5
YOLOv11 fine-tuned on Roboflow football dataset
Event detection (shot, high press, line break, sprint, formation shift)
Qwen2.5:3b commentary in 3 modes (tactical, excited, analytical)
Match report generation
Sprint detection with speed labels
Formation classification (4-back vs 3-back only)
Commentary subtitle overlay on video
Event log JSON output
Match report text output
Gradio UI with mode selector + download buttons
Features excluded: Automatic keypoint-based homography, fine-tuned commentary model, multi-clip analysis, Docker deployment, real-time processing, StatsBomb validation

Implementation time: 21 days total

Main risks: Commentary quality plateau — you will get good-not-great output from zero-shot Qwen:3b. Manage this expectation. The demo still looks impressive.

Success definition: A recruiter or professor can upload a 60-second clip and receive back an annotated video, a 3-mode commentary track, and a match report, without touching any code.

Version 2.0 — Startup-Grade Product
Target: 6–8 weeks beyond v1.0

Features included:

Automatic pitch keypoint detection (YOLOv11-pose fine-tuned on SoccerNet calibration data)
Robust homography with camera pan handling
Fine-tuned commentary model (QLoRA on Qwen2.5:7b using football commentary transcripts)
Pass network graph
xG (expected goals) estimation model
Multi-clip season analysis
StatsBomb open data validation
Docker deployment
Real web hosting option
Features excluded: Real-time processing, mobile deployment, multi-camera sync, player identification by name/face

Implementation time: 6–8 weeks of dedicated work post v1.0

Main risks: Access to training data for commentary fine-tuning, QLoRA fine-tuning requires a GPU with 16GB+ VRAM (use Colab Pro or a cloud instance — ~$5–10 for the fine-tuning run)

SECTION 3 — Architecture Validation
What the blueprint gets right
Sequential pipeline with no microservices — correct for v1.0. Adding a message queue or async workers at this stage would be overengineering.
JSON files instead of a database — correct. You process one clip at a time. SQLite would add zero value.
Ollama as a sidecar — correct. Running it as a local HTTP server is exactly right.
ByteTrack via supervision — correct. Do not implement ByteTrack from scratch.
K-Means team assignment — pragmatic and correct for v1.0.
Separation of event detection from commentary — correct. Rule-based event detection + LLM narration is the right architecture. Do not ask the LLM to detect events.
Overengineering to cut
1. decord video reader — The blueprint calls for decord for hardware-accelerated video decoding. On your machine (Intel Arc integrated, Windows/WSL2), decord's GPU acceleration will not activate, and its Windows support is inconsistent. Use cv2.VideoCapture instead. It is slower but reliable, and for 60-second clips the difference is ~3 seconds. Add decord only if you hit a real I/O bottleneck on longer clips.

2. Phase 2 automatic keypoint homography in the initial plan — The blueprint schedules this for Day 7. Cut it from the first pass entirely. Manual 4-point homography gives you a working minimap in Day 6. Return to automatic keypoints only after v1.0 ships.

3. Docker + docker-compose — The blueprint includes a full Dockerfile. Postpone until v2.0. Docker on Windows with Intel GPU passthrough is a known pain point and adds zero value to a local demo.

4. TrackEval HOTA evaluation — Good for a paper, not necessary for a demo. Eyeball your tracking quality. Run HOTA only if you have time in week 4.

5. Shapely for Voronoi clipping — The blueprint imports shapely for polygon clipping. This is the correct mathematical approach but adds a dependency. For v1.0, clip Voronoi cells to pitch boundary using simple numpy rectangle intersection. Only add shapely if you need pixel-perfect polygon rendering.

Underengineering to fix
1. No frame skipping strategy for the minimap — The blueprint runs spatial analytics every frame but does not address that Voronoi computation on 22 points is called 25 times/second. This is fine (scipy Voronoi on 22 points takes ~0.5ms), but the rendering should be throttled to 5fps (every 5th frame). Don't rerender the minimap every frame.

2. Missing: inference frame cache — If you detect every 3rd frame, you need to explicitly carry forward the last detection result for the 2 skipped frames before passing to ByteTrack. The blueprint shows the pattern in code but does not make it a named component. Make it explicit in detector.py.

3. Missing: graceful Ollama startup check — The blueprint has a check_ollama() function but it is in Section 7.6 as reference. Make it a blocking startup gate in pipeline.py. If Ollama is not running, fail immediately with a clear message rather than silently failing 20 minutes into processing.

4. Commentary should be async or batched — In the current design, commentary generation blocks the video processing pipeline. For every event detected, the system stops and waits 10–25 seconds for Qwen. This means a 60-second clip with 8 events could spend 3+ minutes blocked on LLM calls. Better: collect all events during video processing, then generate all commentary in a post-processing batch step. This separates the CV pipeline (fast) from the LLM pipeline (slow) cleanly.

Missing components
1. A FrameCache class — You need a rolling buffer of the last N frames for: (a) ball interpolation needs frames N-2 and N+2, (b) optical flow needs previous frame, (c) scene detection needs previous frame. One centralized rolling deque, not three separate prev_frame variables.

2. A HomographyManager class — The blueprint has HomographyEstimator and FlowCompensator as separate classes. You need a managing class that holds the current H matrix, decides when to recompute it, and provides a single project(pixel_pt) → pitch_pt interface to the rest of the system. Without this, you will have H matrix state scattered across the pipeline.

3. Confidence annotation on minimap — When homography is invalid (tight zoom), the minimap should show a visual indicator rather than showing stale positions. A simple "CALIBRATION LOST" text overlay is enough.

Better alternatives to note
Blueprint choice	Better for v1.0	Why
decord for video reading	cv2.VideoCapture	More reliable on Windows, no GPU needed for 60s clips
shapely for Voronoi clipping	numpy boundary clamp	One fewer dependency, 90% of the visual result
yolo11s everywhere	yolo11n for speed testing	Start with nano to validate the pipeline, switch to small after
Qwen2.5:7b	Qwen2.5:3b	1.9 GB vs 4.7 GB, 2× faster on CPU, quality is sufficient for structured JSON input
SECTION 4 — Repository Planning

gaffer/
│
├── pyproject.toml              ← uv dependencies + project metadata
├── .python-version             ← "3.11"
├── .env.example
├── .gitignore
├── Makefile
├── README.md
│
├── gaffer/                     ← main Python package
│   ├── __init__.py
│   ├── config.py               ← all constants, thresholds, paths, model names
│   ├── pipeline.py             ← GafferPipeline orchestrator
│   │
│   ├── video/
│   │   ├── loader.py           ← VideoLoader: cv2.VideoCapture wrapper, yields (idx, frame)
│   │   ├── scene_detector.py   ← SceneDetector: histogram diff → scene_changed bool
│   │   ├── frame_cache.py      ← FrameCache: rolling deque of last N frames
│   │   └── writer.py           ← VideoWriter: writes annotated frames to MP4 at target fps
│   │
│   ├── detection/
│   │   ├── detector.py         ← FootballDetector: YOLOv11/OpenVINO wrapper
│   │   │                          detect(frame) → sv.Detections
│   │   │                          skips frames per DETECT_EVERY_N, carries forward cache
│   │   ├── team_assigner.py    ← TeamAssigner: K-Means on HSV jersey crops
│   │   └── ball_tracker.py     ← BallTracker: Bezier interpolation for missed ball frames
│   │
│   ├── tracking/
│   │   ├── tracker.py          ← PlayerTracker: supervision ByteTracker wrapper
│   │   ├── camera_motion.py    ← CameraMotionEstimator: LK optical flow → 2×3 affine matrix
│   │   └── position_store.py   ← PositionStore: {track_id: deque of (frame_idx, x_m, y_m)}
│   │                              get_velocity(track_id) → m/s
│   │
│   ├── calibration/
│   │   ├── pitch_model.py      ← PITCH_KEYPOINTS dict: name → (x_m, y_m)
│   │   ├── homography.py       ← HomographyEstimator: cv2.findHomography wrapper
│   │   ├── homography_manager.py ← HomographyManager: owns H matrix, decides recompute/update
│   │   │                           project(pixel_pt) → (x_m, y_m) | None
│   │   │                           update_with_flow(affine_matrix) → None
│   │   │                           recompute(frame) → bool (success)
│   │   └── flow_compensator.py ← FlowCompensator: updates H from optical flow delta
│   │
│   ├── analytics/
│   │   ├── space_analyzer.py   ← SpaceAnalyzer: runs all sub-analyzers, returns AnalyticsResult
│   │   ├── voronoi.py          ← VoronoiComputer: scipy Voronoi + numpy boundary clamp
│   │   ├── defensive_line.py   ← DefensiveLineTracker
│   │   ├── pressing.py         ← PressingIntensityComputer: cKDTree radius query
│   │   ├── compactness.py      ← CompactnessScorer: ConvexHull.volume
│   │   ├── sprint.py           ← SprintDetector: stateful per-track frame counter
│   │   └── formation.py        ← FormationClassifier: Hausdorff distance to templates
│   │
│   ├── events/
│   │   ├── models.py           ← Pydantic: Event, ShotEvent, HighPressEvent, etc.
│   │   ├── detector.py         ← EventDetector: stateful, compares curr vs prev analytics
│   │   └── log.py              ← EventLog: ordered list + to_json()
│   │
│   ├── commentary/
│   │   ├── agent.py            ← CommentaryAgent: Ollama client + retry + fallback templates
│   │   ├── prompts.py          ← MODE_INSTRUCTIONS + system prompt + per-event user prompts
│   │   ├── serializer.py       ← one serialize_*_event() function per event type
│   │   └── report.py           ← ReportGenerator: batch Qwen call over full EventLog
│   │
│   ├── output/
│   │   ├── annotator.py        ← VideoAnnotator: draws all overlays on a single frame
│   │   ├── minimap.py          ← MinimapRenderer: pitch PNG + Voronoi + player dots
│   │   │                          renders every 5th frame, caches result for others
│   │   └── exporter.py         ← Exporter: writes MP4 + JSON + TXT to session dir
│   │
│   └── utils/
│       ├── geometry.py         ← dist_m(), angle_deg(), line_intersection()
│       ├── colors.py           ← BGR color constants, team color lookup
│       ├── video_utils.py      ← get_fps(), get_resolution(), check_codec()
│       └── logger.py           ← structured logging: %(asctime)s %(levelname)s %(name)s
│
├── app/
│   └── gradio_app.py           ← Gradio UI: video upload, mode selector, progress, download
│
├── scripts/
│   ├── download_clip.py        ← yt-dlp wrapper (Python, not shell — works on Windows)
│   ├── download_weights.py     ← downloads YOLOv11 weights, checks checksums
│   ├── setup_ollama.py         ← checks Ollama running, pulls qwen2.5:3b if missing
│   ├── export_openvino.py      ← exports YOLO model to OpenVINO format
│   └── benchmark.py            ← times each pipeline stage, prints report
│
├── notebooks/
│   ├── 01_yolo_test.ipynb      ← run YOLOv11 on a frame, visualize detections
│   ├── 02_tracking_test.ipynb  ← ByteTrack on 30s clip, inspect track IDs
│   ├── 03_team_assign.ipynb    ← K-Means clustering, tune HSV thresholds
│   ├── 04_homography.ipynb     ← manual 4-point calibration debug
│   ├── 05_analytics.ipynb      ← Voronoi, pressing, defensive line on real data
│   ├── 06_events.ipynb         ← event detection rules, tune thresholds
│   └── 07_commentary.ipynb     ← prompt engineering, compare 3 modes side-by-side
│
├── weights/                    ← gitignored
│   ├── yolov11_football.pt     ← fine-tuned weights (downloaded or trained on Colab)
│   └── yolov11_football_openvino/  ← exported OpenVINO model directory
│
├── data/
│   ├── test_clips/             ← gitignored: 2–3 short MP4s for development
│   ├── pitch_template.png      ← 2D pitch diagram, white lines on green
│   └── formation_templates.json
│
├── outputs/                    ← gitignored: session output directories
│   └── session_YYYYMMDD_HHMMSS_clipname/
│       ├── config.json
│       ├── event_log.json
│       ├── analytics_log.json
│       ├── commentary_log.json
│       ├── match_report.txt
│       └── annotated_output.mp4
│
└── tests/
    ├── conftest.py             ← fixtures: sample_frame, mock_detections, synthetic_positions
    ├── unit/
    │   ├── test_geometry.py
    │   ├── test_team_assigner.py
    │   ├── test_homography.py
    │   ├── test_voronoi.py
    │   ├── test_defensive_line.py
    │   ├── test_pressing.py
    │   ├── test_sprint.py
    │   ├── test_event_detector.py
    │   └── test_serializer.py
    └── integration/
        ├── test_detection_pipeline.py  ← needs weights/ to exist
        └── test_full_pipeline.py       ← skipped in CI if no clip present
Why each top-level folder exists:

gaffer/ — importable package; keeps all business logic importable from scripts and notebooks
app/ — separated from core package because the UI layer should not be imported by tests
scripts/ — one-off operations (setup, download, export, benchmark) that are not part of the library
notebooks/ — exploration and debugging; not production code; explicitly excluded from linting
weights/ — gitignored binary files; kept local only
data/ — input data; test clips gitignored (too large), pitch template and formation templates committed
outputs/ — gitignored generated artifacts
tests/ — standard pytest layout; unit tests have no external dependencies, integration tests may need weights
SECTION 5 — Dependency Audit
Core Runtime
Dependency	Purpose	Free?	RAM/Storage	Install complexity	Alternative
ultralytics	YOLOv11 detection, training, export	Yes (AGPL)	~1.5 GB env	Low	YOLOv8 (same API)
supervision	ByteTrack + annotation utilities	Yes (MIT)	~50 MB	Low	Raw ByteTrack repo (harder)
opencv-python	Video I/O, drawing, optical flow, homography	Yes (Apache)	~100 MB	Low	None practical
numpy	All array operations	Yes	~50 MB	Auto	None
scipy	Voronoi, ConvexHull, cKDTree	Yes	~50 MB	Low	Pure numpy (much harder)
scikit-learn	K-Means team assignment	Yes	~80 MB	Low	Custom K-Means (not worth it)
ollama (Python client)	Qwen2.5 LLM inference	Yes (MIT)	~5 MB	Low	httpx + raw REST calls
gradio	Demo UI	Yes (Apache)	~150 MB	Low	Streamlit (heavier)
pydantic	Event data models, validation	Yes (MIT)	~30 MB	Low	dataclasses (less validation)
tqdm	Progress bars in CLI	Yes (MIT)	~1 MB	Trivial	None
shapely	Voronoi polygon clipping (optional)	Yes (BSD)	~20 MB	Low	numpy clamp (simpler, less accurate)
openvino	Intel GPU/CPU accelerated inference	Yes (Apache)	~500 MB	Medium	CPU-only PyTorch (2–5× slower)
torch	PyTorch (YOLO dependency)	Yes (BSD)	~2 GB	Medium (CUDA vs CPU build)	None
torchvision	Vision ops (YOLO dependency)	Yes (BSD)	~300 MB	Medium	None
Dev Tools
Dependency	Purpose	Free?	Notes
pytest	Unit and integration testing	Yes	Standard
ruff	Linting + formatting (replaces flake8 + black + isort)	Yes	Fastest Python linter available
uv	Package manager, virtual env	Yes	Must-use on this project; 10–100× faster than pip
ipykernel	Jupyter notebooks	Yes	Only needed if using notebooks
System Dependencies
Dependency	Purpose	Free?	Install complexity
ffmpeg	Video codec support (needed by OpenCV for MP4 write)	Yes	Medium on Windows; use winget install ffmpeg
Ollama (application)	Runs Qwen2.5 as local HTTP server	Yes	Low — single installer
Python 3.11	Runtime	Yes	Use uv python install 3.11
Models
Model	Size	License	Download command	Notes
yolo11s.pt (base)	21 MB	AGPL	Auto via ultralytics	Starting point for fine-tuning
yolov11_football.pt (fine-tuned)	~21 MB	AGPL + CC BY	Train on Colab	Your fine-tuned weights
qwen2.5:3b via Ollama	1.9 GB	Apache 2.0	ollama pull qwen2.5:3b	Use 3b, not 7b, on your machine
Datasets
Dataset	Size to download	License	Purpose
Roboflow football-players-detection	~120 MB	CC BY 4.0	YOLO fine-tuning
DFL Bundesliga (Kaggle, 2–3 clips)	~500 MB	Non-commercial	Testing pipeline
3–5 YouTube clips via yt-dlp	~300 MB	Fair use (testing)	Development testing
SoccerNet 224p (optional, 2 clips)	~400 MB	Research-only	Higher quality test footage
Do not download the full SoccerNet dataset (4 TB). You have ~302 GB free but this would consume a third of it for no benefit over YouTube clips during development.

Total storage budget

Python .venv:                ~5 GB
Ollama + qwen2.5:3b:         ~2 GB
PyTorch (CPU build):         ~2 GB
OpenVINO runtime:            ~0.5 GB
Model weights (all):         ~0.2 GB
Test clips (5 × 60s):        ~1.5 GB
Training dataset:            ~0.2 GB
Outputs (5 runs):            ~2 GB
─────────────────────────────────────
Total:                      ~13.5 GB
Available:                  ~302 GB
Comfortable margin: yes
SECTION 6 — Football Intelligence Layer
This section explains from first principles how Gaffer understands football. You know ML and software engineering but not football analytics, so I will build from the ground up.

6.1 How Football Works (Minimum Viable Football Knowledge)
A football match has two teams of 11 players on a 105m × 68m pitch. One team tries to put the ball in the opponent's goal; the other tries to stop them. The match lasts 90 minutes.

At any moment, one team "has possession" — their player controls the ball. The other team is defending. Play is continuous (unlike American football or cricket) which means spatial arrangements change fluidly and rapidly.

Key spatial concepts:

Pitch thirds: The pitch is conceptually divided into defensive third (near your own goal), middle third, and attacking third (near opponent's goal).
Defensive line: The horizontal line formed by a team's defensive players. Attackers try to get behind it. A high defensive line (far from your own goal) means aggressive defending.
Width and depth: Teams can be narrow (compact) or wide (stretched). Deep means close to own goal.
6.2 How Football Knowledge Is Represented in Gaffer
Gaffer represents football knowledge at three levels:

Level 1 — Geometric facts (computed from positions)
These are pure math applied to player coordinates. No football knowledge is required to compute them, but they encode tactically meaningful quantities:

Geometric fact	What it means tactically
Voronoi cell area per team	Which team "owns" which zones — this is pitch control / space dominance
Convex hull area of a team	How compact or spread out they are — a small hull means a tight defensive block
Distance from player to defensive line	How far ahead of or behind the defensive shape a player is
Count of opponents within 10m of ball	How much pressure the ball carrier is under
Ball velocity vector direction	Whether the ball is moving toward goal (shot) vs sideways (pass)
Player speed (m/s)	Whether a player is jogging, running, or sprinting
Level 2 — Rule-based events (applied to geometric facts)
These are football concepts encoded as threshold rules:


high_press:      count(opponents within 10m of ball) >= 3
shot:            ball_speed >= 15 m/s AND ball_direction_toward_goal
line_break:      attacker_y > defensive_line_y (in direction of attack)
sprint:          player_speed >= 7 m/s for >= 3 consecutive frames
formation_shift: detected_shape differs from previous 5-second window
These rules encode what human analysts actually look for. They are not ML — they are domain knowledge written as code.

Level 3 — Natural language narrative (generated by Qwen from levels 1+2)
Qwen2.5 was pre-trained on vast football text — match reports, tactical analyses, coaching literature. When you give it structured facts (level 1) and a named event (level 2), it can generate fluent football commentary because it already has football language and tactical concepts from pre-training. You are not teaching it football; you are giving it structured facts to narrate.

6.3 How Formations Are Represented
A formation like "4-3-3" means: 4 defenders, 3 midfielders, 3 forwards. It describes the typical spatial arrangement when the team is in shape.

In Gaffer, formations are represented as normalized position templates:


"4-3-3": np.array([
    [0.1, 0.5],   # goalkeeper (near own goal, center)
    [0.25, 0.1], [0.25, 0.37], [0.25, 0.63], [0.25, 0.9],  # 4 defenders
    [0.45, 0.2], [0.45, 0.5], [0.45, 0.8],                  # 3 midfielders
    [0.7, 0.15], [0.7, 0.5], [0.7, 0.85],                   # 3 forwards
])
X-axis = depth (0 = own goal, 1 = opponent goal). Y-axis = width (0 = left, 1 = right).

Classification works by computing the Hausdorff distance between actual player positions (normalized, sorted by depth) and each template. Closest template wins.

Important honesty: This is a rough approximation. Real formation analysis requires knowing which player has which role, and roles change dynamically during a match. For a college project, "approximately 4-3-3" is sufficient and honest to state as approximate.

6.4 How Match Events Are Represented
Each event is a Pydantic model:


class HighPressEvent(BaseModel):
    event_type: str = "high_press"
    frame_idx: int
    timestamp: str               # "00:00:23"
    pressing_team: str           # "teamA" or "teamB"
    ball_carrier_team: str
    intensity: int               # count of pressers within radius
    nearest_dist_m: float        # meters to nearest presser
    zone: str                    # "defensive_third" | "middle_third" | "attacking_third"
These are serialized to JSON and passed to Qwen. The structure is the football knowledge — Qwen just narrativizes it.

6.5 How Player Roles Are Represented
Gaffer in v1.0 does not identify individual player roles (goalkeeper aside). It knows:

Class: player, goalkeeper, referee, ball
Team: teamA, teamB
Track ID: arbitrary integer
It does not know "that is the left winger" or "that is the defensive midfielder." This is acceptable — the analytics are position-based, not role-based. A player pressing the ball is pressing regardless of their nominal role.

6.6 How the System Reasons About Football
Gaffer does not reason in the AI sense. It computes:

Where is everyone (detection + homography → real-world coordinates)
How are they organized (analytics on those coordinates)
What just changed (event detection: rule comparison between current and previous state)
What does this mean (Qwen: trained football language → sentence)
The "reasoning" is entirely in step 3 (threshold rules written by you, the engineer) and step 4 (Qwen's pre-trained football knowledge, activated by your structured prompt).

This is not a limitation — it is the correct architecture. Do not attempt to make Qwen detect events. Keep event detection in deterministic code and narration in Qwen.

SECTION 7 — Implementation Order
Step 1: Environment Setup
Goal: Every tool confirmed running before writing any project code.

Deliverables:

Python 3.11 environment via uv
All packages installed
Ollama running with qwen2.5:3b pulled
YOLOv11 base weights auto-downloaded
ffmpeg installed on system
One test clip downloaded via yt-dlp
Dependencies: Internet connection, Windows 11 with WSL2 recommended

Estimated time: 2–3 hours

Success criteria:


# This must run without error:
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
print(model.info())

import ollama
r = ollama.generate(model='qwen2.5:3b', prompt='Say "hello"')
print(r['response'])
Step 2: Detection Pipeline (Notebook first, then code)
Goal: Get player bounding boxes out of a real football frame.

Deliverables:

notebooks/01_yolo_test.ipynb: load model, run on sample frame, visualize boxes
Confirm player, goalkeeper, referee, ball classes are detected
gaffer/detection/detector.py: FootballDetector class wrapping inference
Dependencies: Step 1 complete

Estimated time: 1 day

Success criteria:

On a 720p football frame: detect ≥ 10 players, ≥ 0 ball with base weights
No crashes, clean class interface
Step 3: YOLOv11 Fine-Tuning on Google Colab
Goal: Get better weights for football-specific detection.

Deliverables:

Roboflow dataset downloaded (free account required)
Training notebook run on Colab T4 (100 epochs, ~2 hours)
weights/yolov11_football.pt downloaded to local machine
Dependencies: Step 2 complete, Roboflow free account, Google account

Estimated time: 1 day (mostly waiting for Colab)

Success criteria:

yolo detect val reports player mAP@0.5 ≥ 0.85
Ball mAP@0.5 ≥ 0.60 (lower is acceptable)
Important: Do this in parallel with Step 4. While Colab trains, work on team assignment locally using the base model.

Step 4: Team Assignment
Goal: Label each player bounding box as teamA or teamB.

Deliverables:

notebooks/03_team_assign.ipynb: develop and tune HSV K-Means
gaffer/detection/team_assigner.py: TeamAssigner.fit() and .assign()
Test on 3 clips with visually different jersey colors
Dependencies: Step 2 complete

Estimated time: 1 day

Success criteria:

Both teams consistently labeled with correct colors on 2 test clips
Goalkeeper handled (third cluster or excluded)
Step 5: OpenVINO Export
Goal: Make inference fast enough on your machine for practical use.

Deliverables:

scripts/export_openvino.py: exports fine-tuned weights to OpenVINO format
weights/yolov11_football_openvino/ directory created
Speed benchmark: confirm inference under 100ms/frame at 640px
Dependencies: Step 3 complete (fine-tuned weights)

Estimated time: 2 hours

Success criteria:


python scripts/export_openvino.py
# Reports: inference time X ms/frame
# Target: < 100ms. If > 200ms, reduce to imgsz=480.
Step 6: ByteTrack Integration + Camera Motion
Goal: Assign consistent track IDs to players across frames.

Deliverables:

notebooks/02_tracking_test.ipynb: track players through 30s clip, verify IDs are stable
gaffer/tracking/tracker.py: PlayerTracker class
gaffer/tracking/camera_motion.py: CameraMotionEstimator using LK optical flow
gaffer/tracking/position_store.py: PositionStore with velocity computation
Dependencies: Steps 4 and 5 complete

Estimated time: 1.5 days

Success criteria:

A player visible for 10+ seconds maintains the same track ID
Track ID count equals approximately the number of visible players (no duplicates per player)
Camera motion matrix is computed per frame in < 5ms
Step 7: Annotated Video Output (v0.1 checkpoint)
Goal: Produce the first shippable output.

Deliverables:

gaffer/output/annotator.py: draws team-colored boxes, IDs, ball marker
gaffer/output/writer.py: writes annotated frames to MP4
gaffer/video/loader.py: VideoLoader using cv2.VideoCapture
End-to-end: python -c "from gaffer.pipeline import GafferPipeline; GafferPipeline().process_clip('clip.mp4')"
Dependencies: Steps 4, 5, 6 complete

Estimated time: 1 day

Success criteria:

Upload a 60-second clip → receive annotated MP4
Boxes are team-colored, track IDs are visible and stable
Processing time < 3 minutes
This is v0.1. Commit it. Push it. It is already a working computer vision system.

Step 8: Homography + Minimap
Goal: Project players onto a 2D pitch minimap.

Deliverables:

gaffer/calibration/pitch_model.py: PITCH_KEYPOINTS dict
gaffer/calibration/homography.py: HomographyEstimator
gaffer/calibration/homography_manager.py: HomographyManager managing H state
gaffer/output/minimap.py: MinimapRenderer (load data/pitch_template.png, draw dots)
notebooks/04_homography.ipynb: manual 4-point calibration on 3 clips
Dependencies: Steps 6 and 7 complete

Estimated time: 2.5 days

Success criteria:

For a static-camera clip, player dots on minimap are in approximately correct pitch positions
Minimap updates at 5fps (every 5th frame)
When H is invalid, minimap shows "CALIBRATION LOST" text
Do not attempt automatic keypoint detection at this step.

Step 9: Spatial Analytics
Goal: Compute all tactical metrics from player positions.

Deliverables:

gaffer/analytics/voronoi.py
gaffer/analytics/defensive_line.py
gaffer/analytics/pressing.py
gaffer/analytics/compactness.py
gaffer/analytics/sprint.py
gaffer/analytics/formation.py
gaffer/analytics/space_analyzer.py: SpaceAnalyzer.analyze() → AnalyticsResult
notebooks/05_analytics.ipynb: visualize Voronoi + defensive line on minimap
Unit tests for all analytics components
Dependencies: Step 8 complete

Estimated time: 2 days

Success criteria:

space_analyzer.analyze() runs in < 5ms per frame
Voronoi cells sum to 100% pitch control
Defensive line draws at approximately correct y-position when visually inspected on minimap
All unit tests pass
Step 10: Event Detection
Goal: Detect tactical events from analytics stream.

Deliverables:

gaffer/events/models.py: Pydantic event classes
gaffer/events/detector.py: EventDetector with all 5 detectors
gaffer/events/log.py: EventLog
notebooks/06_events.ipynb: run detector on full clip, inspect detected events visually
Dependencies: Step 9 complete

Estimated time: 1.5 days

Success criteria:

High press events detected on clips where visual inspection confirms pressing
Shot events fire on obvious shooting actions (not random ball movement)
Event timestamps are within 2 seconds of actual moment
Step 11: Commentary Generation
Goal: Narrate each detected event using Qwen2.5:3b.

Deliverables:

gaffer/commentary/serializer.py: one serialize function per event type
gaffer/commentary/prompts.py: system prompt + 3 mode instructions
gaffer/commentary/agent.py: CommentaryAgent with fallback templates
gaffer/commentary/report.py: ReportGenerator
notebooks/07_commentary.ipynb: prompt engineering — compare 3 modes side-by-side, iterate until quality is satisfactory
Dependencies: Step 10 complete, Ollama running with qwen2.5:3b

Estimated time: 2 days (prompt engineering takes real time)

Success criteria:

Commentary is specific to the numbers in the event (not generic)
3 modes are stylistically distinct
All events have commentary within 30 seconds of pipeline running
Fallback templates fire if Ollama is unavailable
Step 12: Full Integration + Gradio UI (v1.0)
Goal: Connect everything into a single working demo.

Deliverables:

gaffer/pipeline.py: GafferPipeline.process_clip() orchestrates all steps
app/gradio_app.py: file upload → mode selector → process → video player + report + download
Session output directory structure
Run full pipeline on 5 different test clips
Fix all crashes
Dependencies: Steps 7–11 complete

Estimated time: 2 days

Success criteria:

Any broadcast football clip (60–120 seconds) processes end-to-end without crashes
Output: annotated MP4 + event_log.json + match_report.txt
Gradio UI reachable at localhost:7860
Step 13: Testing + Benchmark + Documentation
Goal: Make it a real project, not just code.

Deliverables:

All unit tests written and passing (pytest tests/unit/ -v)
scripts/benchmark.py timing report
README with demo GIF, installation steps, example output
docs/COMMENTARY_PROMPTS.md documenting all prompts
Dependencies: Step 12 complete

Estimated time: 2 days

SECTION 8 — Resource Planning
Local Development
Resource	Requirement	Your machine	Status
RAM for inference	4–6 GB	32 GB	Comfortable
RAM for Qwen2.5:3b	~3 GB	32 GB	Comfortable
RAM during full pipeline	~10–12 GB peak	31.4 GB usable	Comfortable
GPU VRAM for OpenVINO	Shared system RAM	128 MB dedicated + shared	Marginal — model may run on CPU
CPU cores	Benefits from 8+	16 cores	Excellent
Storage for dev environment	~14 GB	~302 GB free	Comfortable
Dataset Sizes
Dataset	Download size	Extracted size
Roboflow football detection	~80 MB	~120 MB
5 × 60s YouTube clips (720p)	~300 MB	~300 MB
DFL Bundesliga test clips (3)	~500 MB	~500 MB
YOLOv11 base weights	21 MB	21 MB
Fine-tuned weights + OpenVINO	120 MB	120 MB
Qwen2.5:3b	1.9 GB	1.9 GB
Training Requirements
Task	Hardware	Time	Cost
Fine-tune YOLOv11s, 100 epochs	Google Colab T4	~2 hours	Free (Colab)
Fine-tune YOLOv11s on your machine	Intel Core Ultra 9 (CPU)	~8–12 hours	Free (electricity)
QLoRA commentary fine-tune (v2.0)	Colab Pro A100	~3–5 hours	~$5–10
Recommendation: Always use Colab for YOLO fine-tuning. Your CPU is fast (16 cores) but without CUDA the training loop is 10× slower.

Inference Costs (per 60-second clip)
Stage	Time on your machine	% of total
Video decode + frame cache	~5s	5%
YOLOv11 detection (OpenVINO, every 3rd frame)	~60–120s	45%
ByteTrack + camera motion	~10s	8%
Homography management	~5s	4%
Spatial analytics	~5s	4%
Event detection	~2s	2%
Commentary (8 events × ~15s)	~120s	33%
Video annotation + write	~30s	20%
Total	~4–6 minutes	
The two bottlenecks are detection and commentary. Both are already addressed by the every-3rd-frame trick and qwen2.5:3b respectively.

Production Deployment (hypothetical)
For a web deployment serving multiple users, you would need:

NVIDIA RTX 4090 or A100 for detection (real-time capable)
Qwen2.5:7b with vLLM for concurrent commentary requests
Estimated cloud cost: ~$0.50–1.00 per clip processed on spot instances
This is v2.0 territory. Do not design for it now.
SECTION 9 — Risk Register
ID	Risk	Category	Probability	Impact	Mitigation
R-01	OpenVINO export fails or does not accelerate on Intel Arc integrated	Technical	Medium	High	Fall back to CPU-only PyTorch inference with every-3rd-frame trick. Still workable, just 2× slower.
R-02	Ollama does not use Intel GPU backend, runs on CPU only	Technical	High	Medium	Use qwen2.5:3b (1.9 GB, 2× faster on CPU than 7b). Batch all commentary at end of pipeline instead of inline.
R-03	Homography breaks on panning shots	Technical	High	Medium	Flag H as invalid, skip spatial analytics for those frames. Minimap shows stale positions. Acceptable for v1.0.
R-04	Ball detection too unreliable at 640px	Technical	High	Low	Bezier interpolation for missing frames. Shot detection requires 3 consecutive ball positions so single-frame misses are tolerated.
R-05	ByteTrack ID switches in dense clusters	Technical	Medium	Low	Increase track_buffer to 45 frames. Accept occasional switches, note in documentation.
R-06	K-Means team assignment fails on similar kits	Technical	Medium	High	Position-based initial assignment fallback. Manual override via config file.
R-07	YOLOv11 fine-tuning fails on Colab (session timeout)	Dataset	Low	Medium	Split training: 50 epochs, save checkpoint, resume. Colab Pro ($10/month) eliminates this risk.
R-08	Roboflow dataset access revoked or changed	Dataset	Low	Low	204-image dataset is small enough to mirror locally once downloaded.
R-09	Qwen2.5 commentary is too generic despite prompt engineering	Model	Medium	Medium	Add 2 few-shot examples directly in the system prompt. Explicit prohibition of vague phrases. This consistently improves specificity.
R-10	Formation classifier is always wrong	Model	High	Low	Report "4-back" vs "3-back" only (2-class, much easier). Never claim "4-3-3" with false precision.
R-11	yt-dlp download breaks due to YouTube changes	Dataset	Medium	Low	Manual download fallback. Any MP4 file works. DFL Kaggle dataset as backup.
R-12	WSL2 / Windows file path issues	Technical	Medium	Medium	Use Python pathlib.Path everywhere, never hardcoded slashes. Test on both native Windows Python and WSL2.
R-13	Single-person student team, scope too large	Team	High	High	Strictly follow the MVP decomposition. v0.1 by Day 4, v0.5 by Day 10, v1.0 by Day 21. Cut features ruthlessly at each checkpoint.
R-14	60s processing time exceeds 8 minutes	Technical	Low	Low	Reduce annotation quality (draw fewer overlays). Skip commentary for minor events. Run benchmark early (Day 5) to catch this.
SECTION 10 — 10-Day Plan
Goal: Reach a working v0.5 prototype (annotated video + minimap + spatial overlays) by Day 10. Commentary is added in week 2.

Day 0 (Today) — Environment Setup
Time budget: 3–4 hours

Tasks:

Install uv: winget install astral.uv or the PowerShell installer
Create project structure: uv init gaffer && cd gaffer && uv venv --python 3.11
Install ffmpeg: winget install ffmpeg
Install Ollama (Windows native installer from ollama.ai)
Pull model: ollama pull qwen2.5:3b
Install core packages: uv add ultralytics supervision opencv-python numpy scipy scikit-learn ollama gradio pydantic tqdm openvino
Install dev packages: uv add --dev pytest ruff ipykernel
Sanity check:

from ultralytics import YOLO
YOLO('yolo11n.pt')  # auto-downloads base weights

import ollama
print(ollama.generate(model='qwen2.5:3b', prompt='Say hello')['response'])
Download one 30-second test clip via yt-dlp
Initialize git repo, push empty structure to GitHub
Deliverable: Running environment. Every import works. Ollama responds.

Expected output: Console printout of Qwen saying hello. YOLO model info printed.

Day 1 — Detection Notebook + Detector Class
Time budget: 5–6 hours

Tasks:

Open notebooks/01_yolo_test.ipynb
Load yolo11n.pt, run on one frame from your test clip
Visualize: draw bounding boxes on the frame, display inline
Confirm class labels: player (0), goalkeeper (1), referee (2), ball (3)
Tune conf threshold — try 0.25, 0.35, 0.45 — pick the value with fewest false positives
Write gaffer/detection/detector.py:

class FootballDetector:
    def __init__(self, model_path, device='cpu', imgsz=640, conf=0.35):
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf
        self._last_detections = None  # frame skip cache

    def detect(self, frame, frame_idx, detect_every=3):
        if frame_idx % detect_every == 0:
            results = self.model(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)
            self._last_detections = sv.Detections.from_ultralytics(results[0])
        return self._last_detections
Write a quick standalone test that runs detect() on 10 consecutive frames and prints detection counts
Deliverable: FootballDetector class with frame-skip caching. Notebook showing working detections on your test clip.

Expected output: Notebook cell showing a football frame with red bounding boxes around detected players.

Day 2 — Team Assignment
Time budget: 5–6 hours

Tasks:

Open notebooks/03_team_assign.ipynb
Load 5 frames from your test clip
Crop each player bounding box, convert to HSV
Visualize the HSV crops — understand what hue values correspond to each team's jersey
Implement extract_jersey_color() — try mean hue first, add saturation filtering
Fit K-Means (K=3 for teamA, teamB, goalkeepers)
Visualize: color each player crop by its cluster assignment
Write gaffer/detection/team_assigner.py with fit() and assign() methods
Test on a second clip with different jersey colors — does it still work?
Deliverable: TeamAssigner class. Notebook showing colored player clusters.

Expected output: Players in one team's colors clustered together, other team separately. Visual confirmation.

Day 3 — Fine-Tuning on Google Colab (remote) + ByteTrack (local)
Time budget: 8 hours (split: 3 hours setup/monitoring, 5 hours ByteTrack)

Tasks (Colab, runs in background):

Sign in to Google Colab (T4 GPU)
Mount Google Drive or use Colab storage
Install Roboflow Python SDK, download dataset
Write training cell:

from ultralytics import YOLO
model = YOLO('yolo11s.pt')
model.train(
    data='/content/football_dataset/data.yaml',
    epochs=100,
    imgsz=1280,
    batch=8,
    project='/content/runs',
    name='football_v1'
)
Start training, download best.pt when done (happens overnight if needed)
Tasks (local, concurrent):

Open notebooks/02_tracking_test.ipynb
Run detection on every frame of a 30s clip (use base model for now)
Pass detections to sv.ByteTracker, visualize track IDs
Observe ID stability — note when IDs switch
Write gaffer/tracking/tracker.py with PlayerTracker wrapping sv.ByteTracker
Write gaffer/tracking/camera_motion.py with CameraMotionEstimator using cv2.calcOpticalFlowPyrLK
Deliverable: ByteTrack running on a clip. Colab training started.

Expected output: Tracked video with stable integer IDs above each player's head.

Day 4 — v0.1: First Working Annotated Video
Time budget: 6–7 hours

Tasks:

Write gaffer/video/loader.py — VideoLoader yields (frame_idx, frame) tuples via cv2.VideoCapture
Write gaffer/video/writer.py — VideoWriter takes annotated frames and writes MP4
Write gaffer/output/annotator.py — draws team-colored boxes, track IDs, ball marker
Write gaffer/pipeline.py — minimal version:

class GafferPipeline:
    def process_clip(self, video_path, output_dir):
        for frame_idx, frame in self.loader.load(video_path):
            detections = self.detector.detect(frame, frame_idx)
            detections = self.team_assigner.assign(frame, detections)
            tracked = self.tracker.update(detections)
            annotated = self.annotator.annotate(frame, tracked)
            self.writer.write(annotated)
        self.writer.close()
Run on your test clip end-to-end
Fix any crashes
Time it: record total processing time
Deliverable: v0.1 — annotated MP4 with team-colored boxes and track IDs.

Success gate: If this doesn't produce a real annotated video today, stop and debug before proceeding. Everything downstream depends on this working.

Day 5 — OpenVINO Export + Position Store + Speed Benchmark
Time budget: 5–6 hours

Tasks:

Download fine-tuned weights from Colab (if done) OR continue with base model
Write scripts/export_openvino.py:

from ultralytics import YOLO
model = YOLO('weights/yolov11_football.pt')
model.export(format='openvino', imgsz=640)
Update FootballDetector to accept OpenVINO model path and use task='detect' with OpenVINO runtime
Benchmark: time inference before and after OpenVINO export on 100 frames
Write gaffer/tracking/position_store.py — PositionStore with rolling deque, get_velocity() method
Write scripts/benchmark.py — times each pipeline stage, prints summary table
Run benchmark on test clip, record numbers
Deliverable: OpenVINO model working. Benchmark numbers documented. PositionStore accumulating track histories.

Expected output: benchmark.py output showing per-stage timing. Inference should be faster than Day 4 run.

Day 6 — Homography + Minimap
Time budget: 7–8 hours (this is the hardest day)

Tasks:

Create data/pitch_template.png — draw a simple 2D pitch in OpenCV (green background, white lines for touchlines, halfway line, penalty boxes, center circle). Save once.
Write gaffer/calibration/pitch_model.py with PITCH_KEYPOINTS dict
Open notebooks/04_homography.ipynb
Manually identify 4 visible pitch landmarks in your test clip's first frame (e.g., corner flags, penalty spot, center circle edge)
Record their pixel coordinates
Write the corresponding world coordinates from PITCH_KEYPOINTS
Compute H with cv2.findHomography
Project all player bounding box centers through H → pitch coordinates
Visualize: draw player dots on pitch_template.png
Does it look approximately right?
Write gaffer/calibration/homography_manager.py
Write gaffer/output/minimap.py — load pitch template, draw team-colored dots, return minimap image
Composite minimap onto annotated frame (bottom-right corner, 300×200 pixels)
Run full pipeline with minimap on test clip
Deliverable: Annotated video now shows minimap overlay in the corner.

Expected output: Minimap showing approximately correct player positions relative to the pitch. It will not be perfect — that is fine.

Day 7 — Spatial Analytics: Voronoi + Defensive Line + Pressing
Time budget: 6–7 hours

Tasks:

Write gaffer/analytics/voronoi.py — VoronoiComputer using scipy.spatial.Voronoi. Clamp cells to pitch boundary with numpy (not shapely for now).
Write gaffer/analytics/defensive_line.py — DefensiveLineTracker
Write gaffer/analytics/pressing.py — PressingIntensityComputer using cKDTree
Open notebooks/05_analytics.ipynb:
Feed synthetic positions (known layout) to Voronoi — verify areas sum to 100%
Feed real tracked positions from your clip — visualize Voronoi on pitch template
Check: do Voronoi regions look correct?
Update minimap.py to color Voronoi regions (teamA = light red, teamB = light blue) behind player dots
Add defensive line as a horizontal line on the minimap
Add pressing intensity meter as a bar overlay on the video frame (0–5 scale, live)
Write unit tests for all three analytics functions
Deliverable: v0.5 minimap showing Voronoi zones + defensive line + pressing meter.

Day 8 — Compactness, Sprint Detection, Formation Classifier
Time budget: 5–6 hours

Tasks:

Write gaffer/analytics/compactness.py
Write gaffer/analytics/sprint.py — SprintDetector with per-track state
Write gaffer/analytics/formation.py — Hausdorff distance to templates. Start with 4-back vs 3-back only.
Write gaffer/analytics/space_analyzer.py — SpaceAnalyzer.analyze() calls all of the above, returns AnalyticsResult dataclass
Add sprint highlight to video annotator: orange bounding box + speed label when sprinting
Write unit tests for compactness and sprint
Run space_analyzer.analyze() on every 5th frame of your test clip, confirm < 5ms per call
Deliverable: Complete spatial analytics suite. Sprint highlights visible in annotated video.

Day 9 — Event Detection
Time budget: 6 hours

Tasks:

Write gaffer/events/models.py — all Pydantic event models
Write gaffer/events/detector.py — EventDetector with all 5 detectors
Write gaffer/events/log.py — EventLog
Open notebooks/06_events.ipynb:
Run detector on a clip
Print all detected events with timestamps
Watch the clip manually and verify: did the events fire at the right moments?
Tune thresholds if events fire too frequently or too rarely
Wire EventDetector into the pipeline — events accumulated during clip processing
Write unit tests for event detection rules
Deliverable: event_log.json produced after processing a clip. Events timestamped and verified visually.

Day 10 — Commentary + v0.5 End-to-End Demo
Time budget: 7–8 hours

Tasks:

Verify Ollama is running: ollama list should show qwen2.5:3b
Write gaffer/commentary/serializer.py — serialize_high_press_event() etc.
Write gaffer/commentary/prompts.py — system prompt + 3 mode instructions
Open notebooks/07_commentary.ipynb:
Send one serialized event to Qwen, get commentary back
Compare 3 modes side-by-side
Iterate prompts until quality is satisfying
Key question: is the commentary specific (mentions actual numbers) or generic ("players pressed hard")?
Write gaffer/commentary/agent.py — CommentaryAgent with fallback templates
Wire commentary into pipeline: batch mode — process full clip, collect all events, generate all commentary at the end
Write gaffer/commentary/report.py — ReportGenerator
Run complete pipeline end-to-end:
Input: 60s football clip
Output: annotated MP4 + event_log.json + commentary_log.json + match_report.txt
Record total processing time
Commit. Push. Tag as v0.5.
Deliverable: v0.5 — complete working prototype demonstrating the full pipeline from video in to annotated video + tactical commentary + match report out.

What happens after Day 10
Days 11–14: Gradio UI, bug fixing across 5 test clips, unit tests for all components
Days 15–18: Polish annotation overlays, handle edge cases (short clips, zoomed footage, replay cuts)
Days 19–21: README, demo GIF, documentation → v1.0

What to cut if you fall behind
If you are behind on Day 6: Skip automatic homography. Do manual 4-point calibration only. Continue.

If you are behind on Day 8: Skip formation classification entirely. It adds little to the demo and is unreliable anyway.

If you are behind on Day 9: Skip shot detection (hardest event — requires reliable ball tracking). Keep high press, line break, sprint.

If you are behind on Day 10: Skip the match report. Commentary-on-events is sufficient for the demo.

Never cut: Detection, tracking, team assignment, minimap, event detection, commentary. These are the demo. Everything else is polish.

Summary
Your hardware is sufficient. Your blueprint is 90% correct. The three things you need to do differently from the blueprint:

Use OpenVINO from Day 1, not as an afterthought. It is the difference between a usable demo and a frustrating wait.
Batch commentary at the end of clip processing, not inline. Qwen on CPU is too slow to block the CV pipeline.
Follow the 10-day plan strictly. The blueprint's Phase structure is sensible but it does not give you daily granularity. The plan above does.
The project is genuinely impressive as a portfolio piece. Ship v0.1 on Day 4, v0.5 on Day 10, and v1.0 on Day 21. Each of those is a demo-able checkpoint.