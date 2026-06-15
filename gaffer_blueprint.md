# THE GAFFER — Football Tactical AI Analyst
## Complete Technical Project Blueprint
### Version 1.0 | IIIT Hyderabad | June 2026

---

> **Reading this document**: This is the single source of truth for the entire project. It is intended to be read cover to cover once at the start, then used as a reference during development. Every design decision is recorded here with its rationale. Every resource, model, dataset, and library has a direct link. Every hard problem is identified honestly with proposed solutions.

---

## Table of Contents

1. Project Vision and Problem Definition
2. Complete System Architecture
3. Technology Stack Justification
4. Complete Repository Structure
5. Feature-by-Feature Breakdown
6. Data Sources and Resources
7. Open-Source Repository Analysis
8. AI and Machine Learning Components
9. Commentary Agent Design
10. Hard Problems and Research Challenges
11. Database Design
12. API Design
13. Infrastructure and Deployment
14. Testing and Evaluation
15. Development Roadmap
16. Risks and Contingency Planning
17. Complete Reference Section

---

---

# SECTION 1 — Project Vision and Problem Definition

---

## 1.1 Project Title

**The Gaffer** — Local Football Tactical AI Analyst

*"Gaffer"* is British slang for a football manager. The system functions as an AI tactical mind watching footage and narrating what it sees — exactly what a gaffer does from the dugout.

---

## 1.2 Problem Statement

Watching a football match is easy. Understanding what is actually happening — tactically, spatially, and strategically — is hard. Professional football clubs employ teams of analysts who annotate footage, compute positional data, and generate tactical reports. This process is expensive, slow, and requires human expertise. For fans, students, coaches at lower levels, or anyone who just watched a highlight clip, there is no accessible way to get this kind of analysis.

At the same time, the open-source computer vision ecosystem in 2025–2026 has reached a point where a motivated student with a single consumer GPU can run state-of-the-art object detection, multi-object tracking, and local language models — all completely free. The tools exist. The project is to connect them intelligently into a system that actually does something useful.

**The gap this fills**: There are several football analysis projects on GitHub. Almost all of them stop at tracking players and computing speed. None combine the full pipeline — detection → tracking → spatial analytics → event detection → natural language commentary — into a single, locally-runnable, well-engineered system.

---

## 1.3 Why This Project Matters

### For the builder (you)

- Forces you to understand video object detection and multi-object tracking at a production level, not just in tutorials
- Requires solving a genuinely hard computer vision problem: homography calibration from a moving broadcast camera
- Demonstrates end-to-end system building: from raw video bytes to natural language narrative
- Results in something immediately demoable to anyone, technical or not
- All skills learned here — YOLO fine-tuning, tracking, homography, spatial analytics, local LLM inference — are directly transferable to real computer vision roles

### For the recruiter

This project shows that you can:

1. Take a real-world messy signal (broadcast video) and extract structured information from it
2. Combine multiple CV models purposefully (not randomly)
3. Design a system where each component exists for a reason
4. Ship something that actually works and can be demonstrated

### For the field

Football analytics is a legitimate industry. Companies like StatsBomb, Opta, Hudl, Sportlight, and Wyscout sell exactly this kind of analysis to Premier League clubs for millions of dollars per year. This project is a local, open-source approximation of that pipeline — academically and practically relevant.

---

## 1.4 Target Users

**Primary user (for this project version)**: The developer/student using it to demonstrate technical skill. You are building this for your GitHub portfolio and interview demonstrations.

**Secondary users (for documentation completeness)**:

| User | Description | What They Want |
|------|-------------|----------------|
| Football coach (amateur) | Coaches local club, no budget for Opta | Tactical breakdown of match footage they shot on a phone |
| Football analytics student | Learning sports analytics, no access to expensive tools | A working pipeline to study and extend |
| Sports journalist | Covering non-elite football | Quick tactical summary of a clip |
| Recruiter at a CV company | Reviewing your GitHub | Evidence of serious engineering |

---

## 1.5 User Stories

For the primary demo user (you, or anyone with access to the Gradio interface):

**US-01**: As a user, I want to upload a YouTube football highlight clip and receive an annotated video showing player tracking with team colors, so I can see who is where at every moment.

**US-02**: As a user, I want to see a bird's-eye view minimap updating in real time alongside the broadcast view, so I can understand the spatial positioning of players on the pitch.

**US-03**: As a user, I want to see which team controls which areas of the pitch via a Voronoi diagram overlay on the minimap, so I understand territorial dominance.

**US-04**: As a user, I want the defensive line of the defending team to be visualized on the minimap, so I can see when attackers break behind it.

**US-05**: As a user, I want to see a pressing intensity meter showing how many opponents are pressing the ball carrier, so I understand the defensive pressure in each moment.

**US-06**: As a user, I want the AI to detect key events (shot, high press, line break, sprint, formation change) and display a commentary line at each event, so I get a running narrative of the match.

**US-07**: As a user, I want to choose between three commentary styles — tactical, excited, and analytical — to see how the same event is described differently.

**US-08**: As a user, I want to download a match report at the end summarizing possession, pressing stats, key events, and formation analysis.

---

## 1.6 Core Objectives

| ID | Objective | Measurable Criterion |
|----|-----------|---------------------|
| OBJ-01 | Detect players, ball, and referee reliably | Player mAP@0.5 ≥ 0.85 on test clips |
| OBJ-02 | Track players with consistent identity | HOTA tracking score ≥ 0.6 on standard clips |
| OBJ-03 | Compute bird's-eye view accurately | Homography reprojection error ≤ 5% of pitch width |
| OBJ-04 | Generate event-triggered commentary | Commentary fires within 1 second of detected event |
| OBJ-05 | Process a 60-second clip end-to-end | Total processing time ≤ 5 minutes on consumer hardware |
| OBJ-06 | Ship a working Gradio demo | Any user can upload a clip and receive output without code changes |

---

## 1.7 Scope

**In scope for v1.0**:
- Broadcast camera football footage (full-pitch wide shots preferred)
- Detection of players, ball, referee
- Team assignment by jersey color
- Multi-object tracking
- Camera motion compensation
- Bird's-eye view homography
- Spatial analytics: Voronoi, defensive line, pressing intensity, compactness, formation
- Event detection: shot attempt, high press, line break, sprint, formation shift
- Commentary generation using local Qwen2.5 7B text model
- Annotated video output
- Match report text output
- Gradio demo UI

**Out of scope for v1.0**:
- Audio processing or existing commentary as input
- Real-time processing (system works on pre-recorded clips)
- Individual player identification by face or name
- 3D pose estimation
- Offside detection (requires millimeter-level precision)
- Multi-camera synchronization
- Mobile deployment
- Web hosting / cloud service
- Non-football sports (cricket, F1 — separate projects entirely)

---

## 1.8 Success Metrics

| Metric | Target | How Measured |
|--------|--------|--------------|
| Player detection mAP@0.5 | ≥ 0.85 | YOLOv11 validation on Roboflow football test set |
| Tracking HOTA | ≥ 0.60 | ByteTrack evaluation on DFL Bundesliga clips |
| Homography error | ≤ 5% pitch width | Manual keypoint vs computed position |
| Commentary coherence | ≥ 4/5 human rating | Rate 10 generated events on a 1-5 scale |
| End-to-end clip time | ≤ 5 min / 60 sec clip | Time the full pipeline |
| Demo uptime | Works on any broadcast full-pitch clip | Test on 20 different clips |

---

## 1.9 Future Expansion Opportunities

- **Fine-tuned commentary model**: Fine-tune a small LLM specifically on football commentary transcripts from real broadcasts — creates a more authentic voice
- **Offside detection**: With sufficiently accurate homography, detect when attackers break the line
- **Pass network graph**: Track who passes to whom and visualize it as a network
- **Shot quality model**: Fine-tune a classifier on shot positions and situations to estimate xG (expected goals)
- **Multi-clip season analysis**: Process multiple clips from the same team and compare tactical patterns
- **Integration with StatsBomb open data**: Align with publicly available event data for validation

---

---

# SECTION 2 — Complete System Architecture

---

## 2.1 High-Level Architecture

The system is a sequential processing pipeline. Video enters at one end; annotated video, event log, and narrative report exit at the other.

```
┌─────────────────────────────────────────────────────────────────┐
│                        THE GAFFER                               │
│                  Football AI Analyst v1.0                       │
└─────────────────────────────────────────────────────────────────┘

INPUT LAYER
──────────────────────────────────────────────────────
  MP4/MOV video file (user upload or yt-dlp download)
  ↓
  VideoLoader: decodes frames using OpenCV/decord
  ↓
  SceneDetector: detects shot cuts via histogram diff
  ↓
  FrameSampler: selects keyframes + full frame stream

DETECTION LAYER
──────────────────────────────────────────────────────
  ↓ (every frame)
  YOLOv11 Detector → bounding boxes: {player, ball, referee}
  ↓
  TeamAssigner (K-Means) → label each player bbox: {teamA, teamB}
  ↓
  CameraMotionEstimator (LK optical flow) → motion vector per frame

TRACKING LAYER
──────────────────────────────────────────────────────
  ↓
  ByteTrack → consistent track_id per object across frames
  ↓
  TrackFilter → separate track lists: {players, ball, referee}
  ↓
  PositionStore → running dict: {track_id: [frame_positions]}

CALIBRATION LAYER
──────────────────────────────────────────────────────
  ↓ (on scene change or every N frames)
  PitchKeypointDetector → detect visible pitch markings
  ↓
  HomographyEstimator → 3x3 H matrix: pixel → pitch coords (meters)
  ↓
  HomographyValidator → check reprojection error < threshold
  ↓ (between scene changes)
  FlowCompensator → update H matrix using optical flow delta

SPATIAL ANALYTICS LAYER
──────────────────────────────────────────────────────
  ↓ (per frame, using real-world coordinates from H)
  SpaceAnalyzer:
    ├── VoronoiComputer → pitch control polygons per team
    ├── DefensiveLineTracker → y-coordinate of 2nd-to-last defender
    ├── PressingIntensityComputer → count opponents within radius
    ├── ClosingSpeedComputer → velocity of nearest defender
    ├── CompactnessScorer → convex hull area of each team
    ├── SprintDetector → flag players exceeding speed threshold
    └── FormationClassifier → estimate team shape every 5 seconds

EVENT DETECTION LAYER
──────────────────────────────────────────────────────
  ↓
  EventDetector:
    ├── ShotDetector → ball velocity spike toward goal
    ├── HighPressDetector → intensity ≥ 3 in ball zone
    ├── LineBreakDetector → attacker y > defensive line y
    ├── SprintBurstDetector → sustained speed > threshold
    └── FormationShiftDetector → shape change between windows
  ↓
  EventLog → timestamped list of {event_type, frame, data}

COMMENTARY LAYER
──────────────────────────────────────────────────────
  ↓ (on each detected event)
  EventSerializer → convert event + context to JSON string
  ↓
  CommentaryAgent (Qwen2.5 7B via Ollama):
    Input: structured JSON event data
    Output: natural language commentary line
    Mode: tactical | excited | analytical
  ↓
  CommentaryLog → {timestamp, event_type, commentary_text}

OUTPUT LAYER
──────────────────────────────────────────────────────
  ↓
  VideoAnnotator:
    ├── Draw bounding boxes (team-colored)
    ├── Draw player IDs
    ├── Draw ball trail
    ├── Overlay defensive line
    ├── Overlay pressing intensity meter
    ├── Highlight sprinting players
    └── Render minimap with Voronoi + player dots (corner overlay)
  ↓
  CommentaryRenderer → embed commentary text as subtitles
  ↓
  ReportGenerator → Qwen generates match summary from full EventLog
  ↓
  OUTPUT FILES:
    ├── annotated_output.mp4
    ├── event_log.json
    └── match_report.txt
```

---

## 2.2 Data Flow Diagram

```
RAW VIDEO
    │ (decord: frame-by-frame bytes)
    ▼
FRAME BUFFER ──────────────────────────────────────────────────┐
    │ RGB numpy array [H, W, 3]                                │
    │                                                          │
    ├──► SceneDetector                                        │
    │       [histogram diff > threshold] → scene_changed: bool│
    │                                                          │
    ├──► YOLOv11                                              │
    │       input: [1, 3, 640, 640] (resized, normalized)     │
    │       output: [{bbox, conf, class_id}]                  │
    │                                                          │
    ├──► CameraMotionEstimator                                │
    │       input: prev_frame, curr_frame (grayscale)         │
    │       output: motion_matrix [2, 3] affine transform     │
    │                                                          │
    ▼
DETECTION RESULTS
    │ [{track_id, bbox, class, team, frame_idx}]
    │
    ├──► ByteTrack
    │       input: detections + motion_compensation
    │       output: same + consistent track_ids
    │
    ├──► HomographyEstimator (if scene_changed or every 30 frames)
    │       input: frame + keypoint model output
    │       output: H matrix [3, 3] + validity flag
    │
    ▼
TRACKED DETECTIONS + H MATRIX
    │
    ├──► ProjectionEngine
    │       input: bbox centers + H matrix
    │       output: {track_id: (x_meters, y_meters)}
    │
    ▼
REAL-WORLD POSITIONS (meters on 105×68m pitch)
    │
    ├──► SpaceAnalyzer → {voronoi_cells, def_line_y, intensity, ...}
    ├──► EventDetector → [{event_type, frame, payload}]
    │
    ▼
EVENTS
    │
    ├──► EventSerializer → JSON string
    ├──► Ollama Client → POST /api/generate → commentary: str
    │
    ▼
COMMENTARY LOG
    │
    ├──► VideoAnnotator → writes annotated frames to buffer
    ├──► ReportGenerator → end-of-clip Qwen call → match_report.txt
    │
    ▼
OUTPUT FILES
```

---

## 2.3 Component Interaction Summary

| Component | Consumes | Produces | Critical Dependency |
|-----------|----------|----------|---------------------|
| VideoLoader | file path | frames (numpy) | decord, OpenCV |
| SceneDetector | frames | scene_changed bool | numpy |
| YOLOv11 | frames | raw detections | ultralytics |
| TeamAssigner | player crop arrays | team labels | scikit-learn |
| CameraMotionEstimator | frame pairs | affine matrix | OpenCV |
| ByteTrack | detections + motion | tracked detections | supervision |
| PitchKeypointDetector | frames | keypoint coords | ultralytics (separate model) |
| HomographyEstimator | keypoints | H matrix | OpenCV |
| ProjectionEngine | bboxes + H | real-world coords | numpy |
| SpaceAnalyzer | positions | analytics dict | scipy, numpy |
| EventDetector | analytics + positions | event list | rule-based logic |
| CommentaryAgent | event JSON | commentary str | Ollama (running) |
| VideoAnnotator | frames + all data | annotated frames | OpenCV |
| ReportGenerator | full event log | match_report.txt | Ollama (running) |

---

## 2.4 End-to-End Workflow (User Perspective)

1. User opens Gradio UI at `localhost:7860`
2. User uploads an MP4 clip or pastes a yt-dlp command to download one
3. User selects commentary mode: tactical / excited / analytical
4. User clicks "Analyse"
5. System processes clip (progress bar shows current stage)
6. System shows:
   - Annotated video player (in browser)
   - Scrollable event log with timestamps
   - Match report text panel
   - Download buttons for video + report
7. User can adjust commentary mode and re-run commentary only (no re-running CV) — fast iteration on the text output

---

## 2.5 Service Boundaries

The system runs as a single Python process in v1.0. No microservices. The only external process is Ollama (which runs as a local HTTP server on port 11434). Everything else is in-process.

```
┌─────────────────────────────────┐   HTTP   ┌───────────────┐
│   Python process (gaffer/)      │ ◄──────► │  Ollama       │
│                                 │          │  localhost:    │
│   Gradio UI                     │          │  11434        │
│   Video pipeline                │          │               │
│   Analytics engine              │          │  qwen2.5:7b   │
│   Report generator              │          └───────────────┘
└─────────────────────────────────┘
```

---

---

# SECTION 3 — Technology Stack Justification

---

## 3.1 Core Programming Language: Python 3.11+

**Why Python**: The entire CV and ML ecosystem is Python-native. PyTorch, Ultralytics, OpenCV Python bindings, scikit-learn, scipy, and the Ollama Python client are all first-class Python. Using any other language for this project would add complexity with no benefit.

**Version**: 3.11 specifically because PyTorch 2.x and Ultralytics are tested against 3.11. Avoid 3.12 for now — some Ultralytics internals have intermittent 3.12 issues.

**Package manager**: `uv` (Astral). Dramatically faster than pip. Creates virtual environments correctly. Handles CUDA-specific torch installs cleanly.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create project
uv init gaffer
cd gaffer
uv venv --python 3.11
source .venv/bin/activate
```

---

## 3.2 Object Detection: Ultralytics YOLOv11

**Link**: https://github.com/ultralytics/ultralytics

**Why YOLOv11 over alternatives**:

| Model | Speed | Accuracy | Football-specific weights available | Ease of fine-tuning |
|-------|-------|----------|-------------------------------------|---------------------|
| YOLOv11 | ★★★★★ | ★★★★ | ✅ (Roboflow football dataset) | ★★★★★ (Ultralytics ecosystem) |
| YOLOv9 | ★★★★ | ★★★★ | Partial | ★★★★ |
| YOLOv8 | ★★★★★ | ★★★★ | ✅ (older) | ★★★★★ |
| RT-DETR | ★★★ | ★★★★★ | ❌ | ★★★ |
| DINO | ★★ | ★★★★★ | ❌ | ★★ |
| GroundingDINO | ★ | ★★★★ | Partial | ★ |

Decision: YOLOv11 (or YOLOv8 as fallback). Both use the identical Ultralytics API. The Roboflow football-players-detection dataset has pre-trained YOLO weights that are directly loadable.

**License**: AGPL-3.0 (open source, fine for a student project)

**Hardware requirement**: Runs on CPU (slow) or GPU. On NVIDIA GPU, inference takes ~5ms/frame at 640×640. On CPU, ~100–200ms/frame at 640×640. On Intel Arc integrated graphics without OpenVINO, YOLOv11 at 1280px can be ~300ms/frame. For a 60-second clip at 25fps = 1500 frames, this is too slow without adaptation.

### 3.2.1 Intel-specific adaptation: Core Ultra 9 185H + Arc integrated graphics

This project is being built on a machine with:
- Intel Core Ultra 9 185H: 16 cores (6P + 8E + 2 LPE)
- 32GB RAM
- Intel Arc Graphics integrated GPU (128MB shared)
- ~302GB free storage
- Windows host OS

The key hardware fact: this is not an NVIDIA CUDA machine. Every PyTorch model runs on CPU by default unless an Intel runtime is used.

**Recommended adaptation**:
- Use Ultralytics OpenVINO export for YOLOv11.
- Run detection at `imgsz=640` instead of `1280`.
- Detect every 3rd frame.
- Update ByteTrack every frame.

This gives the best tradeoff for your device:
- Player detection remains very strong.
- Ball detection drops modestly from ~70% to ~55% recall, but frame interpolation recovers most missed ball positions.
- Detection workload drops from 1500 frames to 500 model calls.
- OpenVINO on Arc can reduce inference to ~30–80ms/frame at 640px, instead of ~300ms/frame on CPU.

**Exact adaptation commands**:
```bash
# Export your fine-tuned YOLO model to OpenVINO
yolo export model=weights/yolov11_football.pt format=openvino imgsz=640
```

```python
for frame_idx, frame in enumerate(video_frames):
    if frame_idx % 3 == 0:
        detections = model(frame)           # run detection
    tracked = tracker.update(detections)    # tracking runs every frame
```

**Intel GPU commentary note**:
- Prefer `qwen2.5:3b` on this machine because it is smaller (1.9GB) and fits the Intel Arc performance profile.
- If Ollama does not auto-detect the Intel GPU, start it with `OLLAMA_INTEL_GPU=1 ollama serve`.

**Realistic timing on this device**:
- Detection (OpenVINO, 640px, every 3rd frame = 500 calls): ~40–80 seconds
- Tracking + camera motion: ~8 seconds
- Spatial analytics: ~5 seconds
- Commentary (~8 events × 7s): ~55 seconds
- Video annotation + write: ~30 seconds
- Total: ~2.5–3 minutes for a 60-second clip

**Storage check**:
- Python `.venv`: 3 GB
- Qwen2.5 3B (Ollama): 1.9 GB
- YOLOv11s weights: 45 MB
- OpenVINO export: 90 MB
- Test clips: ~2 GB
- Training dataset: 120 MB
- Output videos: ~500 MB each
- Total needed: ~10 GB

With ~302GB free, this is comfortable. Do not download SoccerNet 4TB; use YouTube clips instead.

**Windows/WSL2 guidance**:
- Use WSL2 on Windows for the Python/CV pipeline.
- Install `Ubuntu-22.04` in WSL2 and run the core Python workflow there.
- Ollama can stay on native Windows and be accessed from WSL2 at `http://localhost:11434`.

---

## 3.3 Multi-Object Tracking: ByteTrack via supervision

**ByteTrack paper**: https://arxiv.org/abs/2110.06864

**ByteTrack GitHub (original)**: https://github.com/ifzhang/ByteTrack

**supervision (Roboflow)**: https://github.com/roboflow/supervision

**Why ByteTrack**: ByteTrack assigns track IDs to every detection — not just high-confidence ones. It handles occlusion better than SORT and is simpler to integrate than DeepSORT (which requires a re-identification model). For football, where players frequently occlude each other, ByteTrack's approach of using low-confidence detections during the matching step helps maintain track continuity.

**Why supervision over raw ByteTrack**: The `supervision` library wraps ByteTrack (and SORT, BotSORT) with a clean Python interface. It also provides annotation utilities, which eliminates a lot of video annotation boilerplate.

```python
# Example integration
import supervision as sv

tracker = sv.ByteTracker()
annotator = sv.BoundingBoxAnnotator()

for frame in frames:
    detections = model(frame)
    detections = tracker.update_with_detections(detections)
    annotated = annotator.annotate(frame, detections)
```

**License**: MIT (supervision), Apache 2.0 (ByteTrack)

---

## 3.4 Team Assignment: K-Means Clustering (scikit-learn)

**Library**: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

**Algorithm**: 
1. For each detected player, crop their bounding box from the frame
2. Convert the crop to HSV color space (hue is jersey color, saturation filters out white/black)
3. Exclude the top 10% of the crop (head/hair) and bottom 15% (shorts often differ from jersey)
4. Sample 50 pixel HSV values from the remaining region
5. Fit K-Means with K=2 across all player crops in the first 100 frames
6. Label each player by their cluster assignment: Team A or Team B

**Why K-Means and not a classifier**: No labeled data needed. The two teams have two distinct jersey colors by definition. K-Means finds them automatically. This is zero-shot team detection.

**Failure modes**: 
- White vs light grey kits (common in La Liga): clusters overlap in HSV space. Mitigation: use multiple color features (hue + saturation histogram, not just mean hue).
- Goalkeeper jerseys differ from outfield players: handle by treating goalkeeper as a third cluster, then manually merging or using position (goalkeeper stays near goal).

---

## 3.5 Camera Motion Compensation: OpenCV Optical Flow

**Library**: https://opencv.org/ — `cv2.calcOpticalFlowPyrLK`

**Why Lucas-Kanade sparse optical flow**: The camera in broadcast football footage moves significantly — panning horizontally, zooming in and out, following the ball. Without compensating for this, player position deltas between frames will be dominated by camera motion, not player motion. LK optical flow tracks a set of "good features to track" (pitch line corners, texture patterns) from frame to frame and estimates the affine transform that describes camera movement.

**Alternative**: RAFT (deep learning optical flow) gives much better results on large motions but is 100x slower. Not necessary here since camera motion is fairly smooth.

**Implementation**:
```python
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, 
                                    qualityLevel=0.01, minDistance=30)
curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, 
                                                 prev_pts, None)
valid_prev = prev_pts[status == 1]
valid_curr = curr_pts[status == 1]
M, _ = cv2.estimateAffinePartial2D(valid_prev, valid_curr)
```

The resulting affine matrix `M` is used to adjust player positions from frame to frame before passing to ByteTrack.

---

## 3.6 Bird's-Eye View: OpenCV Homography

**Library**: https://opencv.org/ — `cv2.findHomography`, `cv2.perspectiveTransform`

**What it does**: Computes a 3×3 projective transformation matrix H such that any pixel coordinate (u, v) in the broadcast frame maps to a real-world coordinate (x_meters, y_meters) on the 105×68m pitch.

**How to get pitch keypoints for calibration**:

Option A (simpler, slightly less accurate): Manually define the four corner regions of the visible pitch portion in each clip and map them to known pitch coordinates. Works for static/slow-moving cameras.

Option B (better): Train a YOLOv11 keypoint detection model on the SoccerNet camera calibration dataset to detect pitch line intersections (penalty spot, corner arc intersection, centre circle tangent points, etc.). These give 10–20 points per frame for a much more stable H matrix.

**Recommended approach for v1.0**: Start with Option A using OpenCV's `getPerspectiveTransform` (4 point pairs) and `findHomography` (N point pairs). Move to Option B when the base system is working.

**Pitch coordinate system**:
```
(0, 0)────────────────────────(105, 0)
  │                               │
  │     [pitch, 105m × 68m]       │
  │                               │
(0, 68)───────────────────────(105, 68)
```

Centre circle: (52.5, 34). Left penalty spot: (11, 34). Right penalty spot: (94, 34).

---

## 3.7 Spatial Analytics: NumPy + SciPy

**numpy**: https://numpy.org/ — array operations, distance calculations, position deltas
**scipy**: https://scipy.org/ — `scipy.spatial.Voronoi`, `scipy.spatial.ConvexHull`, `scipy.spatial.cKDTree`

**Why not a dedicated sports analytics library**: Libraries like `mplsoccer` are great for visualization but add unnecessary dependencies. The underlying math is straightforward numpy/scipy. Understanding the math yourself is the learning value.

**Key computations**:
- Voronoi: `scipy.spatial.Voronoi(all_player_positions)` — instant
- Convex hull: `scipy.spatial.ConvexHull(team_positions).volume` — `volume` is area in 2D
- Distance queries: `scipy.spatial.cKDTree(defender_positions).query(ball_position)` for nearest defender

---

## 3.8 Commentary Generation: Qwen2.5 7B via Ollama

**Ollama**: https://ollama.ai/ | https://github.com/ollama/ollama

**Model**: Qwen2.5 7B Instruct — https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

**Why text-only (not vision)**: Qwen2.5-VL watching every frame would be:
- Too slow (5–10 seconds per frame)
- Too unreliable for dense structured analysis
- Wasteful — the CV pipeline already extracts the information

Instead, Qwen2.5 **text** model receives structured JSON produced by the CV pipeline and generates natural language from it. This is the right separation of concerns. Qwen2.5 7B is excellent at this task because it has strong sports knowledge from pre-training and handles JSON → narrative well.

**Hardware requirement**: 
- 8GB VRAM GPU (NVIDIA): Qwen2.5 7B Q4_K_M quantized via Ollama, ~5GB model, ~1–2s per commentary line
- Intel Arc GPU / OpenVINO: prefer `qwen2.5:3b` (1.9GB). Expect 8–15 tokens/second on Intel GPU, so an 80-token commentary line takes ~5–10 seconds.
- CPU only: ~15–20s per commentary line for Qwen2.5 7B; `qwen2.5:3b` is faster and better suited to low-memory or CPU-first devices.
- Apple Silicon M-series: runs efficiently via Metal, comparable to GPU.

**Intel-specific note**:
This device uses Intel Arc integrated graphics, not an NVIDIA card. Ollama can use the Intel backend if detected automatically or when started with `OLLAMA_INTEL_GPU=1 ollama serve`.
Use `ollama pull qwen2.5:3b` rather than 7B for the current laptop, and keep 7B as a future option on stronger hardware or cloud.

**Installation**:
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the model
ollama pull qwen2.5:3b  # 1.9GB instead of 4.7GB

# Verify
ollama run qwen2.5:3b "Describe a football press in one sentence."
```

**API usage** (Python):
```python
import ollama

response = ollama.generate(
    model='qwen2.5:7b',
    prompt=system_prompt + user_prompt,
    options={'temperature': 0.7, 'num_predict': 100}
)
commentary = response['response']
```

---

## 3.9 Demo UI: Gradio

**Link**: https://gradio.app/ | https://github.com/gradio-app/gradio

**Why Gradio over Flask/FastAPI/Streamlit**: 
- Video upload + video playback is built in (no custom HTML/JS)
- Progress bars for long-running tasks work out of the box
- File download buttons are built in
- Runs in browser immediately without frontend code
- One dependency, not a whole web stack

**For this project**, Gradio is not the "interesting" part — the CV pipeline is. Gradio is chosen to minimize the time spent on UI and maximize time spent on the actual system.

---

## 3.10 Video I/O: decord + OpenCV

**decord**: https://github.com/dmlc/decord — fast video decoding, GPU-accelerated if available

**OpenCV**: https://opencv.org/ — frame processing, drawing, video writing

**Why decord for reading**: OpenCV's VideoCapture is fine but slow for long videos because it reads frames sequentially without hardware acceleration. Decord uses FFMPEG under the hood with proper buffering and can decode much faster, especially for H.264/H.265 encoded football clips.

**ffmpeg** (system dependency): https://ffmpeg.org/ — needed by decord and for video writing

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

---

## 3.11 Package Management Summary

| Package | Version | Purpose | Install |
|---------|---------|---------|---------|
| ultralytics | ≥8.3.0 | YOLOv11 detection | `uv add ultralytics` |
| supervision | ≥0.24.0 | ByteTrack + annotations | `uv add supervision` |
| opencv-python | ≥4.10.0 | Video I/O, drawing, optical flow | `uv add opencv-python` |
| decord | ≥0.6.0 | Fast video decoding | `uv add decord` |
| numpy | ≥1.26.0 | Array operations | auto |
| scipy | ≥1.13.0 | Voronoi, convex hull, KD-tree | `uv add scipy` |
| scikit-learn | ≥1.5.0 | K-Means clustering | `uv add scikit-learn` |
| ollama | ≥0.2.0 | Qwen2.5 7B client | `uv add ollama` |
| openvino | optional | Intel OpenVINO runtime for Arc/CPU inference | `uv add openvino` |
| gradio | ≥4.40.0 | Demo UI | `uv add gradio` |
| torch | ≥2.3.0 | PyTorch (YOLO dependency) | `uv add torch` |
| torchvision | ≥0.18.0 | Vision utilities | `uv add torchvision` |
| yt-dlp | ≥2024.9 | YouTube clip download | `uv add yt-dlp` |
| tqdm | ≥4.66.0 | Progress bars | `uv add tqdm` |
| pydantic | ≥2.0.0 | Data models + validation | `uv add pydantic` |
| pytest | ≥8.0.0 | Testing | `uv add --dev pytest` |
| ruff | ≥0.6.0 | Linting + formatting | `uv add --dev ruff` |

---

---

# SECTION 4 — Complete Repository Structure

---

```
gaffer/
│
├── README.md                        ← Project overview, demo GIF, quickstart
├── pyproject.toml                   ← Dependencies (uv), project metadata
├── .python-version                  ← "3.11"
├── .env.example                     ← Environment variable template
├── .gitignore                       ← Excludes: .venv, __pycache__, *.mp4, weights/
├── Makefile                         ← Shortcuts: make setup, make demo, make test
│
├── gaffer/                          ← Main Python package
│   ├── __init__.py
│   │
│   ├── config.py                    ← All configuration constants (pitch dimensions,
│   │                                   thresholds, model paths, Ollama URL)
│   │
│   ├── pipeline.py                  ← GafferPipeline class: orchestrates full run
│   │                                   process_clip(video_path, mode) → OutputBundle
│   │
│   ├── video/
│   │   ├── __init__.py
│   │   ├── loader.py                ← VideoLoader: decord-based frame reader
│   │   │                               load(path) → generator of (frame_idx, np.ndarray)
│   │   ├── scene_detector.py        ← SceneDetector: histogram diff cut detection
│   │   │                               is_scene_change(prev_frame, curr_frame) → bool
│   │   └── writer.py                ← VideoWriter: writes annotated frames to MP4
│   │                                   write(frames_generator, output_path)
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── detector.py              ← FootballDetector: YOLOv11 wrapper
│   │   │                               detect(frame) → sv.Detections
│   │   ├── team_assigner.py         ← TeamAssigner: K-Means on jersey crops
│   │   │                               fit(frames, detections) → fitted
│   │   │                               assign(detections, frame) → team_labels
│   │   └── ball_tracker.py          ← BallTracker: interpolates missing ball detections
│   │                                   smooth(ball_positions) → interpolated positions
│   │
│   ├── tracking/
│   │   ├── __init__.py
│   │   ├── tracker.py               ← PlayerTracker: ByteTrack wrapper via supervision
│   │   │                               update(detections, motion_matrix) → tracked
│   │   ├── camera_motion.py         ← CameraMotionEstimator: LK optical flow
│   │   │                               estimate(prev_frame, curr_frame) → affine matrix
│   │   └── position_store.py        ← PositionStore: rolling dict of track positions
│   │                                   update(track_id, frame_idx, pos) → None
│   │                                   get_velocity(track_id, window=5) → m/s
│   │
│   ├── calibration/
│   │   ├── __init__.py
│   │   ├── keypoint_detector.py     ← PitchKeypointDetector: detects pitch markings
│   │   │                               detect(frame) → [(u,v), (u,v), ...]
│   │   ├── homography.py            ← HomographyEstimator: computes H matrix
│   │   │                               compute(image_pts, world_pts) → H, valid
│   │   │                               project(bbox_center, H) → (x_m, y_m)
│   │   └── pitch_model.py           ← PitchModel: known keypoint world coords
│   │                                   KEYPOINTS: dict of marking_name → (x_m, y_m)
│   │
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── space.py                 ← SpaceAnalyzer: runs all spatial analytics
│   │   │                               analyze(frame_data) → AnalyticsResult
│   │   ├── voronoi.py               ← VoronoiComputer: pitch control polygons
│   │   │                               compute(positions_by_team) → VoronoiResult
│   │   ├── defensive_line.py        ← DefensiveLineTracker: finds defensive line y
│   │   │                               compute(team_positions, ball_half) → float
│   │   ├── pressing.py              ← PressingIntensityComputer
│   │   │                               compute(positions, ball_pos) → IntensityResult
│   │   ├── compactness.py           ← CompactnessScorer: convex hull areas
│   │   │                               compute(team_positions) → {teamA: float, teamB: float}
│   │   ├── sprint.py                ← SprintDetector: flags fast-moving players
│   │   │                               detect(velocity_store) → [sprint_events]
│   │   └── formation.py             ← FormationClassifier: estimates team shape
│   │                                   classify(team_positions) → "4-3-3" | "4-4-2" | ...
│   │
│   ├── events/
│   │   ├── __init__.py
│   │   ├── detector.py              ← EventDetector: coordinates all event detection
│   │   │                               detect(analytics, prev_analytics) → [Event]
│   │   ├── models.py                ← Pydantic models: Event, ShotEvent, PressEvent, etc.
│   │   └── log.py                   ← EventLog: timestamped event storage
│   │                                   add(event) → None
│   │                                   to_json() → str
│   │
│   ├── commentary/
│   │   ├── __init__.py
│   │   ├── agent.py                 ← CommentaryAgent: Ollama Qwen2.5 wrapper
│   │   │                               generate(event, mode) → str
│   │   ├── prompts.py               ← All prompt templates for each event × mode
│   │   ├── serializer.py            ← EventSerializer: Event → structured JSON string
│   │   │                               serialize(event, analytics_context) → str
│   │   └── report.py                ← ReportGenerator: full match report from EventLog
│   │                                   generate(event_log, stats) → str
│   │
│   ├── output/
│   │   ├── __init__.py
│   │   ├── annotator.py             ← VideoAnnotator: draws all overlays onto frames
│   │   │                               annotate(frame, tracked, analytics, commentary) → frame
│   │   ├── minimap.py               ← MinimapRenderer: generates the bird's-eye view
│   │   │                               render(positions, voronoi, def_line) → minimap_img
│   │   └── exporter.py              ← Exporter: writes final MP4 + text files
│   │                                   export(frames, commentary_log, report) → OutputBundle
│   │
│   └── utils/
│       ├── __init__.py
│       ├── geometry.py              ← Geometry helpers: dist, angle, line intersection
│       ├── colors.py                ← Color palette: team colors, overlay colors
│       ├── video_utils.py           ← FPS detection, codec info, resolution check
│       └── logging.py               ← Structured logging setup
│
├── app/
│   ├── __init__.py
│   └── gradio_app.py                ← Gradio interface: upload → process → display
│
├── scripts/
│   ├── download_clip.sh             ← yt-dlp wrapper: usage: ./scripts/download_clip.sh URL
│   ├── download_weights.sh          ← Downloads all model weights to weights/
│   ├── setup_ollama.sh              ← Installs Ollama + pulls qwen2.5:7b
│   └── run_benchmark.py             ← Runs evaluation on DFL Bundesliga test clips
│
├── weights/                         ← (gitignored) Model weight files
│   ├── yolov11_football.pt          ← Fine-tuned YOLOv11 (download from Roboflow)
│   └── keypoint_model.pt            ← Pitch keypoint model (fine-tuned on SoccerNet)
│
├── data/
│   ├── test_clips/                  ← (gitignored) Short test video clips
│   │   └── sample_30s.mp4           ← One bundled clip (≤25MB for GitHub)
│   ├── pitch_template.png           ← 2D pitch diagram for minimap rendering
│   └── formation_templates.json     ← Known formation position patterns
│
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_team_assigner.py
│   │   ├── test_homography.py
│   │   ├── test_voronoi.py
│   │   ├── test_defensive_line.py
│   │   ├── test_pressing.py
│   │   ├── test_event_detector.py
│   │   └── test_serializer.py
│   ├── integration/
│   │   ├── test_detection_pipeline.py
│   │   └── test_full_pipeline_sample.py
│   └── conftest.py                  ← Test fixtures (sample frame, mock detections)
│
├── notebooks/
│   ├── 01_yolo_exploration.ipynb    ← Load YOLOv11, test on sample frame
│   ├── 02_bytetrack_test.ipynb      ← Track players through a short clip
│   ├── 03_kmeans_team_assign.ipynb  ← Develop + tune team assignment
│   ├── 04_homography_debug.ipynb    ← Step through homography calibration
│   ├── 05_voronoi_analytics.ipynb   ← Visualize Voronoi on pitch
│   ├── 06_event_detection.ipynb     ← Design + test event detection rules
│   └── 07_commentary_prompts.ipynb  ← Develop Qwen prompt templates
│
├── docs/
│   ├── BLUEPRINT.md                 ← This document
│   ├── ARCHITECTURE.md              ← Architecture diagram (exported from code)
│   └── COMMENTARY_PROMPTS.md        ← All prompt templates documented
│
└── deployment/
    ├── Dockerfile                   ← Container for running The Gaffer
    ├── docker-compose.yml           ← App + Ollama together
    └── .env.production              ← Production environment variables
```

---

---

# SECTION 5 — Feature-by-Feature Breakdown

---

## Feature 1: Player, Ball, and Referee Detection

### What It Does
Identifies all objects of interest in each frame: players (two teams + goalkeepers), the ball, and referee(s). Returns bounding boxes with confidence scores and class labels.

### Why It Exists
Every other feature depends on knowing where players and the ball are. This is the foundation of the entire system.

### Technical Design

**Model**: YOLOv11n or YOLOv11s (nano or small) fine-tuned on Roboflow football dataset

**Input**: RGB frame, shape [H, W, 3], any resolution

**Output**: `sv.Detections` object containing:
- `xyxy`: bounding box coordinates [N, 4]
- `confidence`: confidence scores [N]
- `class_id`: class labels [N] where 0=player, 1=goalkeeper, 2=referee, 3=ball

**Class definitions**:
| Class ID | Class Name | Notes |
|----------|-----------|-------|
| 0 | player | Outfield players both teams |
| 1 | goalkeeper | Often different jersey color |
| 2 | referee | Black/yellow jersey |
| 3 | ball | Hardest to detect reliably |

**Inference parameters**:
```python
model = YOLO('weights/yolov11_football.pt')
results = model(frame, conf=0.35, iou=0.45, imgsz=1280, verbose=False)
```
Note: `imgsz=1280` instead of default 640 is critical for ball detection on a powerful GPU. On this Intel Arc device, the practical default is `imgsz=640` with OpenVINO export and every-3rd-frame detection; ball gaps are filled by interpolation.

**Training approach** (fine-tuning):
- Base: `yolo11s.pt` (Ultralytics pretrained on COCO)
- Dataset: Roboflow football-players-detection (204 images + augmentation → ~800 effective)
- 100 epochs on Google Colab free tier (Tesla T4) — approximately 2 hours
- Augmentations: mosaic, flip, HSV shift (helps with different jersey colors and lighting)

### Implementation Plan

Step 1: Download base YOLOv11s weights
```bash
python -c "from ultralytics import YOLO; YOLO('yolo11s.pt')"
```

Step 2: Download Roboflow dataset in YOLO format
```bash
# Requires free Roboflow account
# Link: https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc
pip install roboflow
python scripts/download_roboflow_dataset.py
```

Step 3: Fine-tune
```bash
yolo detect train \
  data=data/football_dataset/data.yaml \
  model=yolo11s.pt \
  epochs=100 \
  imgsz=1280 \
  batch=8 \
  project=runs/detect \
  name=football_v1
```

Step 4: Validate and export
```bash
yolo detect val model=runs/detect/football_v1/weights/best.pt data=data/football_dataset/data.yaml
```

### Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Ball detection failure (fast motion, small size) | High | Medium | Use imgsz=1280, interpolate missing detections across frames |
| Two teams with similar jersey colors | Low | High | Extra HSV feature engineering in team assigner |
| Referee misclassified as player | Medium | Low | Filter referees by class_id in downstream steps |
| Crowded area detection failures | Medium | Medium | Reduce IOU threshold to 0.35 in dense areas |

---

## Feature 2: Multi-Object Tracking (ByteTrack)

### What It Does
Assigns a consistent integer track_id to each detected person/ball across frames. Player 7 detected in frame 100 is still Player 7 in frame 200, even if they went briefly behind another player.

### Why It Exists
Without tracking, you have a new set of detections per frame with no identity continuity. You can't compute who moved where, who's pressing whom, or how fast any player is running.

### Technical Design

**Algorithm**: ByteTrack

**Key innovation of ByteTrack**: Standard trackers (SORT) discard low-confidence detections. ByteTrack uses them for recovery — a low-confidence detection of a player coming out from behind another can be matched to their existing track, maintaining identity. This is crucial for football where players cluster constantly.

**Integration via supervision**:
```python
tracker = sv.ByteTracker(
    track_thresh=0.45,           # Min confidence to start a track
    track_buffer=30,              # Frames a track survives without detection (1.2s at 25fps)
    match_thresh=0.8,             # IoU threshold for matching
    frame_rate=25
)

def track(detections: sv.Detections, motion_matrix: np.ndarray) -> sv.Detections:
    # Compensate for camera motion before tracking
    compensated = compensate_motion(detections, motion_matrix)
    return tracker.update_with_detections(compensated)
```

**Camera motion compensation in tracking**:
Before passing detections to ByteTrack, adjust all bounding box centers by the inverse of the camera motion matrix. This ensures the tracker sees player motion relative to the pitch, not relative to the camera.

```python
def compensate_motion(detections, M):
    """M is the 2x3 affine matrix from LK optical flow."""
    centers = detections.get_anchors_coordinates(sv.Position.CENTER)
    # Apply inverse motion to centers
    inv_M = cv2.invertAffineTransform(M)
    ones = np.ones((len(centers), 1))
    centers_h = np.hstack([centers, ones])
    compensated_centers = (inv_M @ centers_h.T).T
    # Reconstruct detections with compensated positions
    return rebuild_detections(detections, compensated_centers)
```

### Risks

| Risk | Mitigation |
|------|-----------|
| ID switches when players cross paths | Increase `match_thresh` slightly, or use appearance features if needed |
| Tracks dying during replays (camera cuts to replay view) | SceneDetector catches cuts, reset tracker on new scene |
| Ball loses track (occluded by player) | BallTracker interpolates position using Bezier curve between last seen and reacquired positions |

---

## Feature 3: Team Assignment via Jersey Color

### What It Does
Labels each detected player as belonging to Team A or Team B based on the dominant color of their jersey.

### Technical Design

**Algorithm**: K-Means clustering in HSV color space

**Full implementation**:

```python
class TeamAssigner:
    def __init__(self, n_clusters=3):
        # 3 clusters: teamA, teamB, goalkeeper/referee
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        self.team_colors = {}  # {0: 'teamA', 1: 'teamB'}
    
    def extract_jersey_color(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]
        h, w = crop.shape[:2]
        
        # Exclude top 10% (head) and bottom 20% (shorts/legs)
        torso = crop[int(0.1*h):int(0.8*h), int(0.1*w):int(0.9*w)]
        
        # Convert to HSV
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        
        # Filter out very dark (shadow) and very light (white line on ground)
        mask = (hsv[:,:,1] > 40) & (hsv[:,:,2] > 40)
        pixels = hsv[mask].reshape(-1, 3)
        
        if len(pixels) < 10:
            return np.array([0, 0, 128])  # fallback: gray
        
        # Return mean hue-saturation-value (could also use histogram)
        return pixels.mean(axis=0)
    
    def fit(self, frames, all_detections):
        """Call on first 5 frames to fit the color clusters."""
        color_samples = []
        for frame, detections in zip(frames, all_detections):
            player_det = detections[detections.class_id == 0]
            for bbox in player_det.xyxy:
                color = self.extract_jersey_color(frame, bbox)
                color_samples.append(color)
        
        self.kmeans.fit(np.array(color_samples))
        
        # Assign cluster labels to Team A / Team B based on cluster centers
        # Lower hue = Team A (arbitrary, consistent within a clip)
        centers = self.kmeans.cluster_centers_
        sorted_clusters = np.argsort(centers[:, 0])  # sort by hue
        self.team_map = {sorted_clusters[0]: 'teamA', sorted_clusters[1]: 'teamB'}
    
    def assign(self, frame, detection_bbox):
        color = self.extract_jersey_color(frame, detection_bbox)
        cluster = self.kmeans.predict([color])[0]
        return self.team_map.get(cluster, 'unknown')
```

---

## Feature 4: Bird's-Eye View (Homography Transform)

### What It Does
Converts every player's pixel position in the broadcast view into a real-world (x, y) coordinate on the 105×68m pitch. Renders a top-down minimap showing these positions.

### Why This Is The Hard Feature
See Section 10 for the full discussion. In brief: broadcast cameras pan, zoom, and cut constantly. The homography matrix must be recomputed on every scene cut and updated between cuts using optical flow. This is the core engineering challenge of the project.

### Technical Design

**Phase 1 (v1.0): Simple 4-point homography**

The user manually specifies 4 visible pitch landmarks in the first frame of each clip (or a config file stores these for known clip types). The system computes H from these 4 points and uses it until the next scene cut.

```python
# Known pitch points in pixel coordinates (manual annotation for v1.0)
image_pts = np.array([
    [320, 480],  # left penalty spot (as seen in frame)
    [640, 480],  # center circle left
    [320, 100],  # left corner flag
    [640, 100],  # right corner flag (partial)
], dtype=np.float32)

# Known pitch points in real-world meters
world_pts = np.array([
    [11.0, 34.0],   # left penalty spot
    [52.5, 34.0],   # center circle
    [0.0,  0.0],    # left corner
    [52.5, 0.0],    # halfway line left
], dtype=np.float32)

H, mask = cv2.findHomography(image_pts, world_pts, cv2.RANSAC, 5.0)

# Project any pixel to pitch
def project_to_pitch(pixel_pt, H):
    pt = np.array([[pixel_pt]], dtype=np.float32)
    world = cv2.perspectiveTransform(pt, H)
    return world[0][0]  # (x_meters, y_meters)
```

**Phase 2 (v1.1+): Automatic keypoint detection**

Fine-tune YOLOv11-pose on SoccerNet camera calibration data to detect ~30 pitch marking keypoints. Use RANSAC-based `cv2.findHomography` with these N points for a much more stable H matrix.

---

## Feature 5: Voronoi Space Control Map

### What It Does
Divides the pitch into regions. Each region "belongs" to the team whose player is closest to it (geometrically). Shows which team dominates which areas of the pitch.

### Technical Design

```python
from scipy.spatial import Voronoi
import numpy as np

def compute_voronoi_control(team_a_positions, team_b_positions, 
                             pitch_w=105.0, pitch_h=68.0):
    """
    Returns:
        team_a_area: float, percentage of pitch area controlled by Team A
        team_b_area: float, percentage of pitch area controlled by Team B
        cells: list of polygons for rendering
    """
    all_positions = np.vstack([team_a_positions, team_b_positions])
    n_a = len(team_a_positions)
    
    # Add pitch boundary mirror points (prevents infinite cells at edges)
    boundary_pts = create_boundary_points(pitch_w, pitch_h)
    extended_pts = np.vstack([all_positions, boundary_pts])
    
    vor = Voronoi(extended_pts)
    
    # Clip each Voronoi cell to pitch boundaries
    from shapely.geometry import Polygon, box
    pitch_box = box(0, 0, pitch_w, pitch_h)
    
    team_a_area = 0.0
    team_b_area = 0.0
    cells = []
    
    for i, (pos, region_idx) in enumerate(zip(all_positions, vor.point_region)):
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            continue
        vertices = vor.vertices[region]
        poly = Polygon(vertices).intersection(pitch_box)
        area = poly.area
        
        if i < n_a:
            team_a_area += area
            cells.append(('teamA', poly))
        else:
            team_b_area += area
            cells.append(('teamB', poly))
    
    total = team_a_area + team_b_area
    return {
        'teamA_pct': 100 * team_a_area / total,
        'teamB_pct': 100 * team_b_area / total,
        'cells': cells
    }
```

**Dependency note**: This uses `shapely` for polygon clipping. Add it: `uv add shapely`.

---

## Feature 6: Defensive Line Tracking

### What It Does
Finds the y-coordinate of the defensive line on the pitch (the "offside line" in spirit, though exact offside detection requires more precision). Visualizes it as a horizontal bar on the minimap.

### Why It Matters
The defensive line position reveals how high or low a team is defending. A line at y=65 (near their own goal) means deep defending. A line at y=40 (near halfway) means aggressive high press with a high defensive line.

### Technical Design

```python
def compute_defensive_line(team_positions_y_sorted: list[float], 
                             ball_y: float,
                             pitch_h: float = 68.0) -> float:
    """
    The defensive line is the y-position of the second-to-last defender
    (last defender being the goalkeeper).
    
    team_positions_y_sorted: y-coords of defending team, sorted ascending
    ball_y: current ball y-position
    """
    if len(team_positions_y_sorted) < 2:
        return None
    
    # Sort by distance from own goal (lower y = closer to goal A)
    sorted_y = sorted(team_positions_y_sorted)
    
    # The defensive line is the second-from-last in the sorted order
    # (last = goalkeeper who is behind the line)
    defensive_line_y = sorted_y[-2]
    
    return defensive_line_y
```

**Line break detection**: An attacker from the opposing team is considered to have broken the line when their y-coordinate (adjusted for direction of play) exceeds the defensive line y.

---

## Feature 7: Pressing Intensity Meter

### What It Does
Counts how many opposition players are within a configurable radius (default: 10 meters) of the player currently in possession of the ball. Shows as a live meter (0–5 scale) on the video overlay.

### Technical Design

```python
from scipy.spatial import cKDTree

def compute_pressing_intensity(
    ball_pos: tuple[float, float],
    ball_team: str,
    all_positions: dict[str, list[tuple[float, float]]],
    radius_m: float = 10.0
) -> dict:
    """
    Returns pressing intensity and closing speed of nearest presser.
    """
    opposing_team = 'teamB' if ball_team == 'teamA' else 'teamA'
    pressing_positions = all_positions.get(opposing_team, [])
    
    if not pressing_positions:
        return {'intensity': 0, 'nearest_dist': None, 'nearest_speed': None}
    
    tree = cKDTree(pressing_positions)
    
    # Count opponents within radius
    indices_in_radius = tree.query_ball_point(ball_pos, r=radius_m)
    intensity = len(indices_in_radius)
    
    # Distance to nearest opponent
    dist, idx = tree.query(ball_pos, k=1)
    
    return {
        'intensity': min(intensity, 5),  # cap at 5 for display
        'nearest_dist_m': round(dist, 1),
    }
```

**Closing speed**: Computed in `PositionStore.get_velocity(track_id)` using the position delta over the last 3 frames, scaled by the homography pixel-to-meter ratio.

---

## Feature 8: Compactness Score

### What It Does
Measures how "compact" each team is by computing the area of the convex hull of all their players' positions on the pitch. A smaller area means a tightly-organized defensive block.

### Technical Design

```python
from scipy.spatial import ConvexHull

def compute_compactness(team_positions: list[tuple[float, float]]) -> float | None:
    """
    Returns area of convex hull in square meters.
    Lower = more compact.
    """
    if len(team_positions) < 3:
        return None
    
    pts = np.array(team_positions)
    try:
        hull = ConvexHull(pts)
        return round(hull.volume, 1)  # In 2D, ConvexHull.volume is area
    except Exception:
        return None
```

**Typical values**: 
- Compact defensive block: 300–500 m²
- Average in-game: 600–900 m²
- Very spread out (attacking): 900–1300 m²

---

## Feature 9: Sprint Detection

### What It Does
Flags any player whose calculated ground speed exceeds a threshold (default: 7 m/s = ~25 km/h) for at least 3 consecutive frames. Highlights them with a different color bounding box and a speed label.

### Technical Design

```python
SPRINT_THRESHOLD_MS = 7.0   # m/s
SPRINT_MIN_FRAMES = 3

class SprintDetector:
    def __init__(self):
        self.sprint_states = {}  # {track_id: consecutive_sprint_frames}
    
    def update(self, track_id: int, speed_ms: float) -> bool:
        """Returns True if this player is currently sprinting."""
        if speed_ms >= SPRINT_THRESHOLD_MS:
            self.sprint_states[track_id] = \
                self.sprint_states.get(track_id, 0) + 1
        else:
            self.sprint_states[track_id] = 0
        
        return self.sprint_states.get(track_id, 0) >= SPRINT_MIN_FRAMES
```

---

## Feature 10: Formation Classifier

### What It Does
Every 5 seconds of video (or every 125 frames at 25fps), estimates the tactical formation of each team based on the average player positions during that window.

### Technical Design

**Method**: Template matching against known formation position distributions

```python
# Formation templates in normalized pitch coordinates (0-1)
# Positions represent typical player position clusters for a given formation
FORMATION_TEMPLATES = {
    "4-3-3": np.array([
        [0.1, 0.5],   # GK
        [0.25, 0.1], [0.25, 0.37], [0.25, 0.63], [0.25, 0.9],  # DEF
        [0.45, 0.2], [0.45, 0.5], [0.45, 0.8],  # MID
        [0.7, 0.15], [0.7, 0.5], [0.7, 0.85],   # FWD
    ]),
    "4-4-2": np.array([
        [0.1, 0.5],
        [0.25, 0.1], [0.25, 0.37], [0.25, 0.63], [0.25, 0.9],
        [0.5, 0.1], [0.5, 0.37], [0.5, 0.63], [0.5, 0.9],
        [0.72, 0.3], [0.72, 0.7],
    ]),
    "3-5-2": np.array([
        [0.1, 0.5],
        [0.28, 0.2], [0.28, 0.5], [0.28, 0.8],
        [0.5, 0.1], [0.5, 0.3], [0.5, 0.5], [0.5, 0.7], [0.5, 0.9],
        [0.72, 0.3], [0.72, 0.7],
    ]),
    "4-2-3-1": np.array([
        [0.1, 0.5],
        [0.25, 0.1], [0.25, 0.37], [0.25, 0.63], [0.25, 0.9],
        [0.45, 0.3], [0.45, 0.7],
        [0.6, 0.15], [0.6, 0.5], [0.6, 0.85],
        [0.75, 0.5],
    ]),
}

def classify_formation(team_positions_normalized: np.ndarray) -> str:
    """Match team positions to closest formation template using Hausdorff distance."""
    from scipy.spatial.distance import directed_hausdorff
    
    best_match = "unknown"
    best_dist = float('inf')
    
    for name, template in FORMATION_TEMPLATES.items():
        if len(team_positions_normalized) != len(template):
            # Resize to match — use KMeans to reduce to 11 clusters
            continue
        
        dist = max(
            directed_hausdorff(team_positions_normalized, template)[0],
            directed_hausdorff(template, team_positions_normalized)[0]
        )
        
        if dist < best_dist:
            best_dist = dist
            best_match = name
    
    return best_match
```

**Reliability note**: Formation classification is approximate. It reliably distinguishes 4-back vs 3-back formations. Distinguishing 4-3-3 from 4-2-3-1 requires higher position accuracy. Report confidence level alongside formation name.

---

## Feature 11: Event Detection

### What It Does
Identifies discrete tactical events from the continuous analytics stream. These events trigger commentary generation.

### Technical Design

**Event types and detection logic**:

```python
class EventDetector:
    
    def detect_shot(self, ball_positions: list, prev_ball_pos, 
                    goal_positions) -> ShotEvent | None:
        """Ball velocity vector pointing at goal, above threshold."""
        if len(ball_positions) < 3:
            return None
        
        velocity = np.array(ball_positions[-1]) - np.array(ball_positions[-3])
        speed = np.linalg.norm(velocity)
        
        if speed < 15.0:  # m/s, shots are fast
            return None
        
        # Check if direction points toward goal (simplified)
        goal_center = np.array([105.0, 34.0])  # Right goal
        to_goal = goal_center - np.array(ball_positions[-1])
        angle = np.dot(velocity / (speed + 1e-9), 
                       to_goal / (np.linalg.norm(to_goal) + 1e-9))
        
        if angle > 0.7:  # cos(45°) threshold
            return ShotEvent(frame=self.frame_idx, ball_speed_ms=speed)
        return None
    
    def detect_high_press(self, pressing_result: dict) -> HighPressEvent | None:
        """3+ opponents within pressing radius."""
        if pressing_result['intensity'] >= 3:
            return HighPressEvent(
                frame=self.frame_idx,
                intensity=pressing_result['intensity'],
                nearest_dist=pressing_result['nearest_dist_m']
            )
        return None
    
    def detect_line_break(self, analytics: AnalyticsResult, 
                           prev_analytics: AnalyticsResult) -> LineBreakEvent | None:
        """Attacker's y crossed defensive line y between prev and current frame."""
        ...
    
    def detect_formation_shift(self, curr_formation: str, 
                                prev_formation: str) -> FormationShiftEvent | None:
        """Formation changed between 5-second windows."""
        if curr_formation != prev_formation and curr_formation != 'unknown':
            return FormationShiftEvent(
                frame=self.frame_idx,
                old_formation=prev_formation,
                new_formation=curr_formation
            )
        return None
```

---

## Feature 12: Commentary Generation (The Gaffer's Voice)

### What It Does
Converts each detected event into a natural language commentary line using Qwen2.5 7B via Ollama. Three modes: tactical, excited, analytical.

### Why Structured JSON Input Is Critical

**Bad approach** (too slow, too generic):
```
Prompt: "Describe what you see in this football frame: [IMAGE]"
Result: "A football match is being played on a grass pitch. Several players in red and blue uniforms can be seen running."
```

**Correct approach** (fast, specific, reliable):
```json
{
  "event": "high_press",
  "frame_time": "00:00:23",
  "ball_carrier_team": "teamA",
  "ball_pos_m": [67.2, 41.1],
  "pressing_team": "teamB",
  "intensity": 4,
  "nearest_presser_dist_m": 3.2,
  "nearest_presser_speed_ms": 5.8,
  "zone": "attacking_third",
  "teamA_formation": "4-3-3",
  "teamB_formation": "4-4-2"
}
```

### Prompt Templates

**System prompt (all modes)**:
```
You are The Gaffer, an expert football analyst. You receive structured data about a 
tactical event in a football match and generate a commentary line. Be specific to the 
data provided. Do not hallucinate. Keep responses under 2 sentences.
```

**Tactical mode example**:
```
USER: Event data: {JSON}
Style: Tactical analysis. Sound like Pep Guardiola explaining to his team. 
Focus on positional structure and tactical implication.

EXPECTED: "Four red players pressing simultaneously in the attacking third — 
that kind of coordinated gegenpressing from 3.2 meters collapses any 
build-up. The blue team need to play quicker or switch the field."
```

**Excited mode example**:
```
USER: Event data: {JSON}
Style: Excited TV commentator. High energy, emotional, speaks to the moment.

EXPECTED: "FOUR men swarming the ball in the final third! The red team 
is absolutely relentless right now — can the blues handle this pressure?!"
```

**Analytical mode example**:
```
USER: Event data: {JSON}
Style: Data analyst. Include specific numbers. Cold, precise, informative.

EXPECTED: "High press event at 00:23 — 4 opposing players within 10m radius, 
nearest at 3.2m closing at 5.8 m/s. Attacking third press. 
Team B compactness: 420 m² (compact block)."
```

### Full Ollama Integration

```python
import ollama
import json

class CommentaryAgent:
    def __init__(self, model: str = "qwen2.5:7b", ollama_url: str = "http://localhost:11434"):
        self.model = model
        self.client = ollama.Client(host=ollama_url)
        self._check_model_available()
    
    def _check_model_available(self):
        models = self.client.list()
        names = [m['name'] for m in models['models']]
        if self.model not in names:
            raise RuntimeError(f"Model {self.model} not available. Run: ollama pull {self.model}")
    
    def generate(self, event: dict, context: dict, mode: str = "tactical") -> str:
        serialized = json.dumps({**event, **context}, indent=2)
        
        system = self._system_prompt()
        user = self._user_prompt(serialized, mode)
        
        response = self.client.generate(
            model=self.model,
            system=system,
            prompt=user,
            options={
                'temperature': 0.7,
                'num_predict': 120,
                'stop': ['\n\n']
            }
        )
        return response['response'].strip()
    
    def generate_match_report(self, event_log: list[dict], stats: dict) -> str:
        summary_data = {
            'total_events': len(event_log),
            'key_events': event_log[:10],
            'stats': stats
        }
        prompt = f"""
Generate a football match tactical report based on this data:
{json.dumps(summary_data, indent=2)}

Format as:
TACTICAL OVERVIEW: (2 sentences)
KEY MOMENTS: (3-5 bullet points)
STATISTICS: (list key numbers)
VERDICT: (1 sentence summary)
        """
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            options={'temperature': 0.5, 'num_predict': 400}
        )
        return response['response'].strip()
```

---

---

# SECTION 6 — Data Sources and Resources

---

## 6.1 Training Data

### Roboflow Football Players Detection Dataset

| Field | Detail |
|-------|--------|
| URL | https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc |
| Purpose | Fine-tune YOLOv11 for football-specific player, ball, referee detection |
| Size | 204 raw images → ~800 with augmentation |
| Format | YOLO format (class_id cx cy w h normalized) |
| Classes | player, goalkeeper, referee, ball |
| License | CC BY 4.0 |
| Download | Via Roboflow Python SDK (requires free account) |
| Storage | ~120MB with images |

```python
# Download script
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_FREE_API_KEY")  # Get at roboflow.com
project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
dataset = project.version(12).download("yolov11")
```

---

### SoccerNet — Match Video Dataset

| Field | Detail |
|-------|--------|
| URL | https://www.soccer-net.org/ |
| GitHub SDK | https://github.com/SoccerNet/soccernet |
| PyPI | https://pypi.org/project/SoccerNet/ |
| Purpose | Actual broadcast match footage for testing and evaluation |
| Size | 550 complete matches, 764 hours total |
| Format | MP4 (720p), 25fps |
| License | Research-only (non-commercial) |
| Registration | Free at soccer-net.org (takes ~1 day for access) |
| Storage | Full dataset: ~4TB. Single match: ~7GB. Short clips: ~200MB |

```python
# Download a specific game (after registration)
from SoccerNet.Downloader import SoccerNetDownloader
downloader = SoccerNetDownloader(LocalDirectory="data/soccernet")
downloader.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], 
                          split="valid")
```

**What to download for this project**: Only the 224p validation clips (~30GB for full validation set). For initial development, 2–3 individual match clips is sufficient.

---

### SoccerNet Camera Calibration Dataset

| Field | Detail |
|-------|--------|
| URL | https://github.com/SoccerNet/sn-calibration |
| Purpose | Training data for the pitch keypoint detection model (Phase 2 homography) |
| Size | 1,200 annotated frames with pitch line keypoints |
| License | Research-only |

---

### DFL Bundesliga Data Shootout (Kaggle)

| Field | Detail |
|-------|--------|
| URL | https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout |
| Purpose | Short broadcast clips with diverse camera angles for testing |
| Size | ~2GB of video clips |
| Format | MP4 |
| License | Non-commercial research |
| Download | Via Kaggle CLI: `kaggle competitions download -c dfl-bundesliga-data-shootout` |

```bash
pip install kaggle
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
kaggle competitions download -c dfl-bundesliga-data-shootout -p data/
```

---

## 6.2 Pretrained Models

### YOLOv11 Base Weights

| Field | Detail |
|-------|--------|
| URL | https://github.com/ultralytics/assets/releases/tag/v8.3.0 |
| File | yolo11s.pt (21MB) |
| Purpose | Base model for fine-tuning on football dataset |
| Download | `python -c "from ultralytics import YOLO; YOLO('yolo11s.pt')"` (auto-downloads) |

### Qwen2.5 7B Instruct (via Ollama)

| Field | Detail |
|-------|--------|
| HuggingFace | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct |
| Ollama tag | qwen2.5:7b |
| Size | 4.7GB (Q4_K_M quantization via Ollama) |
| License | Apache 2.0 (free for commercial use) |
| Download | `ollama pull qwen2.5:7b` |
| VRAM needed | 6GB (Q4) or 14GB (FP16) |

---

## 6.3 Video Download Tool

### yt-dlp

| Field | Detail |
|-------|--------|
| GitHub | https://github.com/yt-dlp/yt-dlp |
| Purpose | Download football highlight clips from YouTube for testing |
| License | Unlicense (public domain) |
| Install | `pip install yt-dlp` |

```bash
# Download best quality up to 1080p
yt-dlp -f "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]" \
  --merge-output-format mp4 \
  -o "data/test_clips/%(title)s.mp4" \
  "https://www.youtube.com/watch?v=VIDEO_ID"

# Download a shorter clip (30 seconds starting at 2 minutes)
yt-dlp --download-sections "*02:00-02:30" -f "bestvideo[height<=720]" \
  -o "data/test_clips/sample.mp4" "URL"
```

**Legal note**: Downloading YouTube videos for personal testing and research is generally accepted for non-commercial, educational purposes. Do not redistribute downloaded clips.

---

---

# SECTION 7 — Open-Source Repository Analysis

---

## 7.1 Ultralytics (YOLOv11)

**URL**: https://github.com/ultralytics/ultralytics

**Stars**: 34,000+ | **License**: AGPL-3.0

**Key files for this project**:
- `ultralytics/models/yolo/detect/predict.py` — inference pipeline
- `ultralytics/data/augment.py` — understand augmentation for fine-tuning
- `ultralytics/utils/plotting.py` — drawing utilities (can borrow)

**Key API we use**:
```python
from ultralytics import YOLO
model = YOLO('yolo11s.pt')
results = model('frame.jpg', conf=0.35, iou=0.45)
# results[0].boxes.xyxy → tensor of [N, 4] boxes
# results[0].boxes.conf → tensor of [N] confidences
# results[0].boxes.cls  → tensor of [N] class ids
```

**Integration**: Used as a dependency, not modified. We wrap it in `gaffer/detection/detector.py`.

**Risk**: Major version updates sometimes break the API. Pin to specific version in `pyproject.toml`: `ultralytics>=8.3.0,<9.0.0`.

---

## 7.2 Roboflow supervision

**URL**: https://github.com/roboflow/supervision

**Stars**: 24,000+ | **License**: MIT

**Why we use this**: Provides ByteTrack implementation, clean detection data structures (`sv.Detections`), and annotation utilities. Eliminates ~500 lines of boilerplate.

**Key classes**:
- `sv.Detections` — unified detection format with `.xyxy`, `.confidence`, `.class_id`, `.tracker_id`
- `sv.ByteTracker` — ByteTrack wrapper
- `sv.BoundingBoxAnnotator` — draws boxes
- `sv.LabelAnnotator` — draws labels
- `sv.TraceAnnotator` — draws trajectory lines

**Key files**:
- `supervision/tracker/byte_tracker/` — ByteTrack implementation
- `supervision/annotators/` — annotation utilities
- `supervision/detection/core.py` — Detections data class

**Important**: `sv.ByteTracker` parameters (especially `track_buffer`) are frame-rate sensitive. Multiply by fps/25 if your clip isn't 25fps.

---

## 7.3 ByteTrack (original paper and repo)

**URL**: https://github.com/ifzhang/ByteTrack

**Paper**: https://arxiv.org/abs/2110.06864

**We use supervision's implementation** rather than the original repo directly. Read the original repo to understand the algorithm — specifically the two-step matching in `byte_tracker.py` that makes ByteTrack better than SORT.

**Key insight from code**: In `update()`, ByteTrack runs IoU matching for high-confidence detections first, then a second pass that matches remaining tracks to low-confidence detections. This second pass is what prevents track loss during occlusion.

---

## 7.4 SoccerNet SDK

**URL**: https://github.com/SoccerNet/soccernet

**PyPI**: https://pypi.org/project/SoccerNet/

**Purpose for this project**: Download match videos for testing. We do not use any SoccerNet models or pipelines — only the data.

**Key function**:
```python
from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="data/soccernet")
mySoccerNetDownloader.downloadGames(files=["1_224p.mkv"], split="valid")
```

---

## 7.5 SoccerNet Camera Calibration (for Phase 2)

**URL**: https://github.com/SoccerNet/sn-calibration

**Paper**: https://arxiv.org/abs/2104.09403

**Purpose**: Training data for the pitch keypoint detection model. The repo contains annotated pitch line intersection coordinates for ~1200 frames from SoccerNet matches.

**Key file**: `sn-calibration/src/` contains the homography evaluation code. Study this to understand how homography accuracy is measured.

**Integration**: We train a YOLOv11-pose model on the keypoint annotations from this dataset. The annotations are in a custom JSON format that needs conversion to YOLO keypoint format.

---

## 7.6 Ollama

**URL**: https://github.com/ollama/ollama | https://ollama.ai/

**Stars**: 85,000+ | **License**: MIT

**Purpose**: Runs Qwen2.5 7B as a local HTTP server. Handles model loading, quantization, memory management, and inference API.

**Python client**: https://github.com/ollama/ollama-python

**Key endpoints we use**:
- `POST /api/generate` — single completion
- `GET /api/tags` — list available models (for startup check)

**Important**: Ollama must be running as a background process before the Python pipeline starts. Add a startup check:
```python
import requests
def check_ollama():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False
```

---

## 7.7 mplsoccer (visualization reference)

**URL**: https://github.com/andrewRowlinson/mplsoccer

**Stars**: 2,000+ | **License**: MIT

**We don't use this directly** — it requires matplotlib and adds overhead. But study its pitch drawing code to understand how to render the pitch template for the minimap:

`mplsoccer/pitch.py` — `draw()` method shows how to draw pitch lines in normalized coordinates. We replicate this in OpenCV in `gaffer/output/minimap.py`.

---

## 7.8 StatsBomb Open Data

**URL**: https://github.com/statsbomb/open-data

**Purpose**: Reference data for testing and validation. StatsBomb provides free event data (passes, shots, tackles) for certain matches. We can use this to validate our event detection against ground truth.

**Relevant data**: `data/events/*.json` — each file is a match with timestamped events. Cross-reference our detected events against ground truth.

---

---

# SECTION 8 — AI and Machine Learning Components

---

## 8.1 YOLOv11 (Detection)

**Architecture**: One-stage anchor-free object detector. Based on a CSP (Cross Stage Partial) backbone with C3k2 blocks, a PANet FPN neck, and a decoupled head for classification and regression.

**Input format**: 
- Shape: `[1, 3, H, W]` where H=W=1280 for our use
- Values: Normalized to [0, 1] (Ultralytics handles this internally)
- Color: BGR (OpenCV default) — Ultralytics converts internally to RGB

**Output format**:
- `results[0].boxes.xyxy`: `[N, 4]` tensor — absolute pixel coordinates
- `results[0].boxes.conf`: `[N,]` tensor — confidence scores
- `results[0].boxes.cls`: `[N,]` tensor — class IDs

**Training requirements**:
- Google Colab free tier (T4 GPU): ~2 hours for 100 epochs on 800 images
- Local NVIDIA GPU (RTX 3060+): ~45 minutes for 100 epochs
- Memory: ~4GB VRAM for batch size 8 at 1280×1280

**Inference requirements**:
- NVIDIA GPU: ~5ms per frame at 1280×1280 (200 FPS theoretical)
- Apple M-series: ~30ms per frame via MPS backend
- CPU only: ~200ms per frame (5 FPS)

**Expected performance after fine-tuning**:
- Player detection: mAP@0.5 ~0.91
- Referee detection: mAP@0.5 ~0.88
- Ball detection: mAP@0.5 ~0.65–0.75 (hard, highly variable)

**Failure cases**:
- Ball in motion blur at >1/60s shutter: very low detection confidence
- Two players in identical jerseys very close together: may merge into one detection
- Players at the edge of the pitch near advertising boards: false positives

**Fine-tuning decisions**:
- Use `yolo11s` (small), not `yolo11n` (nano) — the extra capacity helps with ball detection
- Use `imgsz=1280` — critical for detecting the small ball
- Freeze backbone for first 20 epochs, then unfreeze all — faster convergence

---

## 8.2 ByteTrack (Tracking)

**Architecture**: Kalman filter + two-step IoU matching + byte association

**Input format**: 
```python
# sv.Detections object with fields:
detections.xyxy          # [N, 4] bounding boxes
detections.confidence    # [N,] confidence scores
detections.class_id      # [N,] class labels
```

**Output format**: Same sv.Detections with additional `tracker_id` field assigned

**Key parameters**:
```python
sv.ByteTracker(
    track_thresh=0.45,     # Lower = more tracks started, more noise
    track_buffer=30,        # 30 frames = 1.2s at 25fps before track dies
    match_thresh=0.8,       # Higher = stricter IoU matching
    frame_rate=fps          # Must match actual video FPS
)
```

**GPU requirements**: None — ByteTrack is CPU-only, pure Python/numpy

**Limitations**:
- Tracking accuracy degrades in very dense clusters (5+ players overlapping)
- Does not use appearance features (no re-ID model) — can swap IDs in long occlusions
- Track IDs reset after a scene cut (must manually handle this)

---

## 8.3 Qwen2.5 7B (Commentary)

**Architecture**: Transformer decoder (LLM), 7 billion parameters

**Input format**:
- Tokenized text via Ollama API (we send raw strings, Ollama tokenizes)
- Context window: 128K tokens (more than enough)
- We send: system prompt (~100 tokens) + event JSON (~200 tokens) + instruction (~50 tokens)

**Output format**: Generated text string, maximum 120 tokens (controlled via `num_predict`)

**Quantization**: Ollama serves Q4_K_M by default — 4-bit quantization reduces 14GB FP16 model to ~4.7GB with minimal quality degradation

**Inference latency**:
- NVIDIA GPU (8GB): 1–3 seconds per call (60–80 tok/s)
- Apple M2 Pro: 3–5 seconds per call
- CPU only: 15–30 seconds per call

**Fine-tuning possibility (Phase 2)**:
The commentary can be improved significantly by fine-tuning on real football commentary transcripts. Dataset sources:
- SoccerNet-Echoes: https://github.com/SoccerNet/sn-echoes (football commentary audio)
- BBC Sport match reports (scrape and clean)
- Use QLoRA for efficient fine-tuning (16GB GPU required)

This is a Phase 2 research extension, not v1.0.

**Expected quality assessment**:

| Commentary Mode | Quality with Zero-Shot Qwen2.5 7B |
|----------------|-----------------------------------|
| Tactical | Good — Qwen has strong football knowledge from pre-training |
| Excited | Decent — sometimes over-enthusiastic or under-specific |
| Analytical | Very Good — converts numbers to prose reliably |

The key to quality is the structured JSON input. The more specific and clean the data, the better the commentary.

---

---

# SECTION 9 — Commentary Agent Design

---

## 9.1 Agent Overview

The CommentaryAgent is not an agent in the multi-step autonomous sense — it does not plan, search, or call tools. It is a single-call LLM component that converts structured tactical data into natural language. The word "agent" is used in the loose sense of "a component that reasons and generates text."

**Responsibility**: For each detected event, take the event data + current analytics context → produce one commentary line.

**Inputs**:
1. `event: Event` — Pydantic model with event type, frame, and event-specific data
2. `analytics_context: dict` — current state of all analytics (formation, compactness, possession, etc.)
3. `mode: str` — "tactical" | "excited" | "analytical"

**Output**: `str` — 1–2 sentence commentary

---

## 9.2 Prompt Design

All prompts follow a consistent structure:

```
[SYSTEM PROMPT]
You are The Gaffer, an expert football analyst with deep tactical knowledge. 
You receive structured JSON data describing a single tactical event in a football match.
Generate exactly 1-2 sentences of commentary. Be specific to the numbers provided.
Do not invent information not present in the data. Use football terminology correctly.

[MODE INSTRUCTION]
{mode_specific_instruction}

[EVENT DATA]
{serialized_json}

[INSTRUCTION]
Generate commentary now. Do not explain your reasoning. Just the commentary.
```

**Mode instructions**:

```python
MODE_INSTRUCTIONS = {
    "tactical": (
        "Style: Expert tactical analyst. Like Johan Cruyff or Pep Guardiola explaining "
        "to his coaching staff. Use tactical terms: pressing, compactness, space, "
        "lines, shape. Focus on WHY the moment is tactically significant."
    ),
    "excited": (
        "Style: Passionate TV commentator (think Martin Tyler or Peter Drury). "
        "High energy. Convey the drama and excitement. Use vivid language. "
        "Make the moment feel significant even if it's a minor event."
    ),
    "analytical": (
        "Style: Sports data analyst. Include specific numbers from the data. "
        "Be precise and measured. Format: describe what happened, then give the key metric."
    )
}
```

---

## 9.3 Event Serialization

Each event type has a dedicated serializer that produces clean JSON:

```python
def serialize_high_press_event(event: HighPressEvent, context: AnalyticsResult) -> str:
    return json.dumps({
        "event_type": "high_press",
        "timestamp": event.timestamp_str,
        "pressing_team": event.pressing_team,
        "ball_carrier_team": event.ball_carrier_team,
        "press_intensity": event.intensity,
        "opponents_within_10m": event.intensity,
        "nearest_presser_distance_m": round(event.nearest_dist_m, 1),
        "ball_zone": event.zone,
        "pressing_team_formation": context.formation.get(event.pressing_team),
        "pressing_team_compactness_sqm": round(
            context.compactness.get(event.pressing_team, 0), 0
        ),
        "game_context": {
            "possession_teamA_pct": context.possession_pct.get("teamA", 50),
            "space_control_teamA_pct": round(context.voronoi.get("teamA_pct", 50), 0)
        }
    }, indent=2)
```

---

## 9.4 Error Handling and Fallbacks

```python
class CommentaryAgent:
    def generate(self, event, context, mode="tactical") -> str:
        try:
            response = self._call_ollama(event, context, mode)
            if len(response) < 10:
                raise ValueError("Response too short")
            return response
        except Exception as e:
            # Fallback: template-based commentary (no LLM)
            return self._fallback_template(event)
    
    def _fallback_template(self, event) -> str:
        """Simple string templates when Ollama is unavailable."""
        templates = {
            "high_press": "High press detected — {intensity} opponents within pressing range.",
            "shot": "Shot attempt at goal!",
            "line_break": "Line break — attacker behind the defensive line.",
            "sprint": "Sprint burst detected.",
            "formation_shift": "Formation change: {old} to {new}."
        }
        template = templates.get(event.event_type, "Tactical event detected.")
        return template.format(**event.dict())
```

---

---

# SECTION 10 — Hard Problems and Research Challenges

---

## Hard Problem 1: Homography from a Moving Broadcast Camera

**Why it's difficult**: 

A football broadcast camera is constantly moving — panning left/right following the ball, zooming in on goal action, cutting between cameras. The homography matrix H (which maps pixels to pitch coordinates) is only valid for one camera position. Every pan changes H. Every cut completely invalidates it.

Professional solutions (used by Opta, StatsBomb) involve multi-view camera arrays with fixed positions and calibration targets. We have a single broadcast camera with no calibration reference and significant motion.

**Existing solutions**:
- TVCalib (Theiner & Ewerth, 2023): https://arxiv.org/abs/2303.11128 — deep learning calibration trained on large sports datasets
- SoccerNet calibration challenge submissions: Various methods, top ones use transformer-based keypoint detection
- Eagle tracker: Real-time HRNet-based keypoint detection with Kalman filter smoothing

**Our approach (pragmatic for v1.0)**:

1. Detect pitch line intersections using a fine-tuned YOLOv11-pose model (train on SoccerNet calibration data)
2. Match detected points to known pitch coordinates using RANSAC
3. On each cut (detected by histogram difference), recompute H from scratch
4. Between cuts, use Lucas-Kanade optical flow on 20 background feature points to estimate camera motion and update H with an incremental warp

**Failure mode**: When the camera is zoomed into a close-up (player face, bench shot), no pitch lines are visible and H cannot be computed. Handle by:
- Flagging frames as "homography unavailable"
- Skipping spatial analytics for those frames
- Interpolating player positions from the last valid H if the zoom is brief (<3 seconds)

**Evaluation**: Measure reprojection error. Take 10 manually annotated frame pairs (pixel coordinate → known pitch coordinate), project using estimated H, measure Euclidean distance in meters. Target: < 3 meters average error.

---

## Hard Problem 2: Ball Detection and Tracking

**Why it's difficult**:

The ball is small (30–40 pixels at wide shot), fast (can move 50+ pixels between frames at 25fps), and frequently occluded by players. YOLOv11 at standard resolution misses it frequently. At 1280×1280 resolution, detection improves significantly but still drops to 30–50% when the ball is in motion with blur.

**Existing solutions**:
- TrackNetV2: https://github.com/yastrebksv/TrackNet — specialized small ball tracker using temporal frames, achieves high recall for tennis/badminton balls
- Ball detection papers use optical flow + temporal windows to find the ball in consecutive frames

**Our approach**:

1. Use YOLOv11 with `imgsz=1280` for base detection (~65% recall)
2. When ball is not detected in frame N but was detected in N-2 and N+2, interpolate position using Bezier curve
3. Apply additional post-processing: use optical flow to predict ball position from last known position, then run a localized high-resolution crop search in that predicted region

**Realistic expectation**: Ball tracking will be imperfect. Accept this and handle gracefully — shot detection requires 3 consecutive ball positions, so single-frame misses are acceptable.

---

## Hard Problem 3: Team Assignment with Similar Jerseys

**Why it's difficult**:

K-Means on jersey color works well for red vs blue, or yellow vs green. But Arsenal white away vs Real Madrid white at home would fail completely.

**Existing solutions**:
- Use multiple color features: mean hue, saturation histogram, value histogram
- Use a small CNN trained on jersey patches (requires labeled data)
- Use position heuristics: at kickoff, teams start in known halves, allowing initial labeling

**Our approach**:

1. Primary: HSV clustering as described
2. Fallback for ambiguous cases: Position-based initial assignment. In the first 30 seconds of a clip, assume the team defending the left goal is Team A and the other is Team B. Compute average x-position of all players in each cluster — the cluster with lower average x is Team A.
3. Track consistency: Once a track_id has a team label, keep it for the entire clip unless a major color change is detected (jersey change after substitution).

---

## Hard Problem 4: Formation Classification Reliability

**Why it's difficult**:

Real football formations are fluid. A 4-3-3 in attack looks like a 4-5-1 in defense when the wingers drop. Formation identification from positions alone requires understanding the ball position, phase of play, and team shape conventions. Even human analysts disagree on formation names.

**Our approach**:

Be intentionally conservative:
- Classify only the "defensive shape" (when team is defending)
- Use simple 2-class detection: 4-back vs 3-back (easier and more reliable than 4-3-3 vs 4-4-2)
- For 11 players, use K-Means with K=4 to find positional clusters (GK + 3 outfield lines), then count clusters
- Report confidence percentage alongside formation name

**Evaluation**: There is no clean ground truth for formation detection without manual annotation. Visual inspection of the minimap against the classified formation is the primary validation method.

---

## Hard Problem 5: Commentary That Doesn't Sound Generic

**Why it's difficult**:

Without player names, without score context, and without knowing the team names, Qwen2.5 7B can sound generic: "The attacking team has players near the goal." We want specificity.

**Mitigation strategies**:

1. **Rich context JSON**: Include as much specific data as possible — not just "high press" but the exact intensity, nearest distance, zone, formation of pressing team, compactness score
2. **Few-shot examples in system prompt**: Include 2 example event/commentary pairs to calibrate style
3. **Prohibit generic language**: Explicit instruction: "Do not use vague phrases like 'near the goal' or 'a lot of pressure'. Use specific meters and player counts from the data."
4. **Temperature tuning**: 0.7 for tactical and excited modes, 0.4 for analytical mode (more deterministic for data-heavy commentary)

---

---

# SECTION 11 — Database Design

---

## 11.1 Philosophy

This is a single-user, local, batch-processing system. There is no persistent multi-user state, no authentication, no concurrent users. Therefore: **no database server required for v1.0**. 

All data is stored as JSON files in a session directory and loaded into Python dictionaries during processing.

---

## 11.2 Session Data Structure

Each processed clip creates a session directory:

```
outputs/
└── session_20260615_143022_clip_name/
    ├── config.json          ← Processing parameters used
    ├── event_log.json       ← All detected events with timestamps
    ├── analytics_log.json   ← Per-frame analytics results
    ├── track_log.json       ← Player track positions and velocities
    ├── commentary_log.json  ← Generated commentary lines
    ├── match_report.txt     ← Final match report
    └── annotated_output.mp4 ← Annotated video
```

---

## 11.3 Event Log Schema

```json
{
  "session_id": "20260615_143022",
  "clip_name": "arsenal_vs_chelsea_highlight.mp4",
  "total_frames": 1500,
  "fps": 25,
  "events": [
    {
      "id": "evt_001",
      "event_type": "high_press",
      "frame_idx": 342,
      "timestamp": "00:00:13.68",
      "data": {
        "pressing_team": "teamB",
        "ball_carrier_team": "teamA",
        "intensity": 4,
        "nearest_dist_m": 3.2,
        "zone": "attacking_third"
      },
      "commentary": {
        "tactical": "Four red players converging in the attacking third...",
        "excited": "FOUR on the ball! This is relentless pressing!",
        "analytical": "High press: 4 opponents within 10m, nearest at 3.2m."
      }
    }
  ]
}
```

---

## 11.4 Analytics Log Schema (per-frame, sampled)

Storing analytics for every frame at 25fps × 60sec = 1500 entries would be ~15MB of JSON. We sample every 5th frame (5fps equivalent) and store the full data.

```json
{
  "frame_samples": [
    {
      "frame_idx": 0,
      "timestamp": "00:00:00.00",
      "homography_valid": true,
      "teamA": {
        "positions_m": [[11.2, 34.1], [23.4, 12.3], "..."],
        "formation": "4-3-3",
        "compactness_sqm": 720.5,
        "space_control_pct": 48.2,
        "avg_depth_m": 34.2
      },
      "teamB": {
        "positions_m": [[92.1, 34.0], "..."],
        "formation": "4-4-2",
        "compactness_sqm": 580.0,
        "space_control_pct": 51.8,
        "avg_depth_m": 70.1
      },
      "ball_pos_m": [52.3, 34.1],
      "defensive_line_teamB_y": 65.2,
      "pressing_intensity": 2
    }
  ]
}
```

---

---

# SECTION 12 — API Design

---

## 12.1 Internal Python API

The system is a Python library, not a web service. The public API is the `GafferPipeline` class:

```python
from gaffer import GafferPipeline, CommentaryMode

pipeline = GafferPipeline(
    detection_model_path="weights/yolov11_football.pt",
    commentary_mode=CommentaryMode.TACTICAL,
    ollama_url="http://localhost:11434"
)

result = pipeline.process_clip(
    video_path="data/test_clips/sample.mp4",
    output_dir="outputs/",
    on_progress=lambda pct, msg: print(f"{pct:.0%} — {msg}")
)

print(result.match_report)
result.save_video("outputs/annotated.mp4")
```

---

## 12.2 Gradio HTTP Interface

Gradio exposes an internal HTTP API automatically. For programmatic access:

**Base URL**: `http://localhost:7860`

**Upload and process**:
```
POST /run/predict
Content-Type: application/json

{
  "data": [
    {"path": "/tmp/clip.mp4"},  // file upload
    "tactical"                   // mode
  ]
}
```

**Response**:
```json
{
  "data": [
    {"path": "/tmp/annotated.mp4"},  // video file
    "TACTICAL OVERVIEW: ...",         // match report text
    [{"time": "00:13", "text": "..."}, ...]  // event log
  ]
}
```

---

## 12.3 Ollama Internal API (reference)

**Check model availability**:
```
GET http://localhost:11434/api/tags
→ {"models": [{"name": "qwen2.5:7b", ...}]}
```

**Generate commentary**:
```
POST http://localhost:11434/api/generate
Content-Type: application/json

{
  "model": "qwen2.5:7b",
  "system": "You are The Gaffer...",
  "prompt": "{event_json}\n\nGenerate commentary:",
  "stream": false,
  "options": {
    "temperature": 0.7,
    "num_predict": 120
  }
}

→ {
  "response": "Four opponents pressing relentlessly in the final third...",
  "done": true,
  "total_duration": 1834122000
}
```

---

---

# SECTION 13 — Infrastructure and Deployment

---

## 13.1 Local Development Setup

**Prerequisites**:
- Python 3.11
- NVIDIA GPU with CUDA 12.1+ (or Apple M-series or CPU)
- Ollama installed
- ffmpeg installed

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/gaffer.git
cd gaffer

# 2. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create virtual environment and install dependencies
uv sync

# 4. Install Ollama and pull model
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5:7b

# 5. Download model weights
chmod +x scripts/download_weights.sh
./scripts/download_weights.sh

# 6. Run the demo
uv run python app/gradio_app.py
```

---

## 13.2 Dockerfile

```dockerfile
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Copy source code
COPY gaffer/ gaffer/
COPY app/ app/
COPY data/pitch_template.png data/
COPY data/formation_templates.json data/

# Copy weights (or mount as volume)
COPY weights/ weights/

# Expose Gradio port
EXPOSE 7860

# Note: Ollama runs as separate container
ENV OLLAMA_URL=http://ollama:11434

CMD ["uv", "run", "python", "app/gradio_app.py", "--server-name", "0.0.0.0"]
```

---

## 13.3 docker-compose.yml

```yaml
version: '3.8'

services:
  gaffer:
    build: .
    ports:
      - "7860:7860"
    volumes:
      - ./data:/app/data
      - ./weights:/app/weights
      - ./outputs:/app/outputs
    environment:
      - OLLAMA_URL=http://ollama:11434
    depends_on:
      ollama:
        condition: service_healthy
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  ollama_models:
```

---

## 13.4 Hardware Requirements

| Configuration | GPU | RAM | Storage | Processing Speed |
|--------------|-----|-----|---------|-----------------|
| Minimum | None (CPU) | 8GB | 20GB | ~10 min/60s clip |
| Recommended | NVIDIA RTX 3060 (12GB) | 16GB | 30GB | ~3 min/60s clip |
| Optimal | NVIDIA RTX 4070 (12GB) or 3090 (24GB) | 32GB | 30GB | ~1 min/60s clip |
| Apple Silicon | M2 Pro/Max (18GB+ unified) | 18GB+ | 30GB | ~4 min/60s clip |

**Storage breakdown**:
- Ollama + Qwen2.5 7B: 4.7GB
- YOLOv11 weights: 45MB
- Python environment (.venv): 3GB
- Test clip library: 2–5GB
- Processing outputs: 1–3GB per clip (annotated video at 1080p)

---

---

# SECTION 14 — Testing and Evaluation

---

## 14.1 Unit Tests

Located in `tests/unit/`. Run with `pytest tests/unit/ -v`.

### Key test cases:

**test_homography.py**:
```python
def test_project_known_point():
    """Penalty spot (11m, 34m) should project correctly."""
    image_pts = get_sample_image_pts()
    world_pts = get_known_world_pts()
    H, _ = cv2.findHomography(image_pts, world_pts)
    
    test_pixel = np.array([[320, 480]], dtype=np.float32)
    result = cv2.perspectiveTransform(test_pixel.reshape(-1,1,2), H)
    x, y = result[0][0]
    
    assert abs(x - 11.0) < 2.0, f"X error: {abs(x - 11.0):.1f}m"
    assert abs(y - 34.0) < 2.0, f"Y error: {abs(y - 34.0):.1f}m"

def test_homography_invalid_when_collinear():
    """Should return invalid=True when points are collinear."""
    ...
```

**test_voronoi.py**:
```python
def test_voronoi_areas_sum_to_100():
    """Team A + Team B control % should equal 100."""
    result = compute_voronoi_control(team_a_pos, team_b_pos)
    total = result['teamA_pct'] + result['teamB_pct']
    assert abs(total - 100.0) < 0.1

def test_voronoi_symmetric():
    """Equal number of players at symmetric positions → 50/50 control."""
    ...
```

**test_defensive_line.py**:
```python
def test_defensive_line_second_to_last():
    """Defensive line should be the 2nd-to-last defender's y position."""
    positions = [5.0, 20.0, 35.0, 50.0, 60.0]  # sorted y-coords
    result = compute_defensive_line(positions, ball_y=60.0)
    assert result == 50.0  # second to last
```

---

## 14.2 Integration Tests

**test_detection_pipeline.py**:
```python
def test_full_detection_on_sample_frame(sample_frame):
    """End-to-end detection on a known frame should find ≥10 players."""
    detector = FootballDetector('weights/yolov11_football.pt')
    detections = detector.detect(sample_frame)
    
    player_mask = detections.class_id == 0
    assert player_mask.sum() >= 10, "Should find at least 10 players"

def test_team_assignment_consistent():
    """Same player across 5 frames should always get same team label."""
    ...
```

---

## 14.3 Model Evaluation

**Detection evaluation**:
```bash
# Run YOLOv11 validation on Roboflow test split
yolo detect val \
  model=weights/yolov11_football.pt \
  data=data/football_dataset/data.yaml \
  split=test \
  imgsz=1280
# Reports: mAP50, mAP50-95 per class
```

**Tracking evaluation** (HOTA metric):
```bash
# Use TrackEval on DFL Bundesliga annotated sequences
# Repository: https://github.com/JonathonLuiten/TrackEval
pip install trackeval
python scripts/run_tracking_eval.py --clips data/dfl_eval/ --model weights/yolov11_football.pt
```

**Homography evaluation**:
```python
# Manual: annotate 10 frame-keypoint pairs in data/homography_eval/
# Compare projected vs ground truth pitch coordinates
python scripts/eval_homography.py
# Reports: mean error in meters, max error
```

**Commentary evaluation**:
- Manual: Rate 20 generated commentary lines on 1–5 scale for:
  - Accuracy (does it describe what happened?)
  - Specificity (does it use the actual numbers?)
  - Fluency (is it natural football language?)
- Target: ≥ 4.0 average across all three dimensions

---

## 14.4 End-to-End Benchmark

```bash
# Time the full pipeline on the standard test clip
python scripts/run_benchmark.py \
  --clip data/test_clips/sample_60s.mp4 \
  --output outputs/benchmark_run/ \
  --mode tactical

# Reports:
# - Detection FPS
# - Tracking overhead (ms per frame)
# - Homography time per scene change
# - Commentary generation latency (seconds per event)
# - Total wall-clock time
```

**Target benchmarks**:

| Stage | Target | Acceptable |
|-------|--------|-----------|
| Detection (GPU) | 15ms/frame | 30ms/frame |
| Tracking | 5ms/frame | 10ms/frame |
| Spatial analytics | 3ms/frame | 8ms/frame |
| Event detection | 1ms/frame | 2ms/frame |
| Commentary (per event) | 2s/event | 10s/event |
| Total (60s clip, GPU) | 3 minutes | 8 minutes |

---

---

# SECTION 15 — Development Roadmap

---

## Phase 0: Environment Setup (Day 1)

**Tasks**:
- Set up Python 3.11 environment with uv
- Install all dependencies
- Install Ollama, pull qwen2.5:7b
- Run a sanity check: YOLOv11 on a sample image, Qwen2.5 generating text
- Download one DFL Bundesliga clip for testing

**Deliverable**: Running development environment, all tools confirmed working.

**Dependencies**: Internet connection, Python 3.11, ffmpeg

---

## Phase 1: Detection and Tracking (Week 1, Days 2–5)

**Day 2–3: YOLOv11 fine-tuning**
- Download Roboflow football dataset
- Fine-tune YOLOv11s for 100 epochs on Colab
- Validate model — check mAP, visualize detections on test frames
- Move weights to `weights/yolov11_football.pt`

**Day 3–4: Team assignment**
- Implement `TeamAssigner` with K-Means on HSV crops
- Test on 3 different match clips with different jersey combinations
- Handle the goalkeeper edge case

**Day 4–5: ByteTrack integration**
- Implement `PlayerTracker` wrapper
- Test on one full 30-second clip
- Verify consistent track IDs across 300 frames
- Add camera motion compensation

**Deliverable at end of Phase 1**: Given any football clip, produce an annotated video with colored bounding boxes, team labels, and consistent player IDs.

**Notebook checkpoints**: `01_yolo_exploration.ipynb`, `02_bytetrack_test.ipynb`, `03_kmeans_team_assign.ipynb`

---

## Phase 2: Homography and Bird's-Eye View (Week 2, Days 6–10)

**Day 6: Pitch model and manual calibration**
- Define `PitchModel` with all keypoint world coordinates
- Implement manual 4-point H estimation
- Test on 3 clips with manual point annotation

**Day 7: Keypoint model (Phase 1.5 option)**
- Try the SoccerNet calibration keypoint detector
- If available pre-trained: integrate it
- If not: fall back to manual 4-point for v1.0

**Day 8: Minimap renderer**
- Implement `MinimapRenderer` — load pitch template PNG, draw player dots
- Test: do player positions on minimap look correct for known positions?

**Day 9: Camera motion compensation for H**
- Implement optical flow-based H update between scene cuts
- Test on a clip with significant camera pan

**Day 10: Integration + debugging**
- Connect detection → tracking → homography → minimap
- Process a full 60-second clip
- Visual inspection: do minimap positions look right?

**Deliverable**: Annotated video with minimap overlay showing live player positions on a 2D pitch.

**Notebook**: `04_homography_debug.ipynb`

---

## Phase 3: Spatial Analytics (Week 2–3, Days 11–14)

**Day 11: Voronoi + compactness**
- Implement `VoronoiComputer` with shapely boundary clipping
- Test on synthetic position data first
- Visualize colored Voronoi cells on minimap

**Day 12: Defensive line + pressing intensity**
- Implement `DefensiveLineTracker`
- Implement `PressingIntensityComputer`
- Test both on real tracking data

**Day 13: Sprint detection + formation classifier**
- Implement `SprintDetector`
- Implement basic formation classifier (4-back vs 3-back first)

**Day 14: Integration of all analytics**
- `SpaceAnalyzer.analyze()` calls all above components
- Test on full 60-second clip
- Check timing: all analytics must complete in < 10ms per frame

**Deliverable**: For any frame, produce a complete analytics dictionary with all spatial metrics.

**Notebook**: `05_voronoi_analytics.ipynb`

---

## Phase 4: Event Detection + Commentary (Week 3, Days 15–18)

**Day 15: Event detector**
- Implement all 5 event detectors
- Test on clips where you can visually confirm events are correct
- Log all detected events with timestamps

**Day 16: Commentary prompts**
- Write system prompt and mode instructions
- Test all three modes in `07_commentary_prompts.ipynb`
- Iterate on prompts until quality is satisfactory

**Day 17: Commentary agent integration**
- Implement `CommentaryAgent` with error handling
- Connect `EventDetector` → `EventSerializer` → `CommentaryAgent`
- Verify commentary fires for each detected event

**Day 18: Match report generator**
- Implement `ReportGenerator`
- Test: does the match report accurately summarize the detected events?

**Deliverable**: System detects events and generates commentary automatically.

---

## Phase 5: Output and Demo (Week 3–4, Days 19–21)

**Day 19: Video annotator**
- Draw all overlays onto frames: boxes, IDs, minimap, pressing meter, commentary subtitles, sprint highlights, defensive line
- Write annotated frames to output MP4

**Day 20: Gradio UI**
- Implement `gradio_app.py`
- Test: upload clip → process → view results → download
- Add commentary mode selector

**Day 21: Polish and testing**
- Run full pipeline on 5 different test clips
- Fix crashes and edge cases
- Write README with demo GIF

**Deliverable**: Complete working demo. Any football clip in → annotated video + commentary + report out.

---

## Phase 6: Evaluation and Documentation (Week 4, Days 22–25)

**Day 22–23: Evaluation**
- Run official YOLOv11 evaluation
- Benchmark full pipeline timing
- Manual evaluation of 20 commentary lines

**Day 24: Documentation**
- Complete all docstrings
- Write developer guide in docs/
- Record demo video for GitHub README

**Day 25: GitHub launch**
- Push to GitHub
- Write good README with GIFs, features, installation guide
- Add badges: Python version, license, last commit

---

---

# SECTION 16 — Risks and Contingency Planning

---

| Risk ID | Risk | Probability | Impact | Mitigation | Backup Plan |
|---------|------|------------|--------|-----------|------------|
| R-01 | Homography fails on tight zoom shots | High | Medium | Flag as "unavailable", skip spatial analytics for those frames | Implement zone-based fallback (broad left/right/center estimation from pixel x) |
| R-02 | Ball detection too unreliable | High | Medium | Interpolate missing ball positions using Bezier curves | Fall back to player-only analytics; skip ball-dependent events (shots) |
| R-03 | Ollama too slow on CPU | Medium | Low | Commentary only fires on events (not every frame) — CPU is acceptable | Pre-generate commentary in batch at end of processing instead of real-time |
| R-04 | K-Means team assignment fails on similar jerseys | Medium | High | Position-based initial assignment as fallback | Require user to manually specify team colors in config |
| R-05 | ByteTrack loses track IDs in dense clusters | High | Medium | Increase `track_buffer` to 45 frames; use IoU threshold tuning | Accept occasional ID switches, display warning in output |
| R-06 | Formation classifier is inaccurate | High | Low | Report low confidence; only classify 4-back vs 3-back in v1 | Remove formation feature from v1.0, keep as a stretch goal |
| R-07 | SoccerNet dataset access delays | Low | Low | Use DFL Bundesliga Kaggle dataset instead + YouTube clips | Entire project testable on YouTube highlights alone |
| R-08 | Fine-tuning YOLOv11 requires GPU access | Low | Medium | Google Colab free tier has T4 GPU (enough for 100 epochs) | Use pre-trained Roboflow weights directly if Colab unavailable |
| R-09 | Qwen2.5 commentary is too generic | Medium | Medium | Rich structured JSON input + few-shot examples in prompt | Fine-tune Qwen on football commentary data (Phase 2) |
| R-10 | Processing time exceeds 10 min/clip | Low | Low | Optimize: cache H matrix, run analytics every 3rd frame | Run in batch, present result when ready (not "real-time") |

---

---

# SECTION 17 — Complete Reference Section

---

## 17.1 Primary Libraries and Frameworks

| Resource | URL | Purpose |
|----------|-----|---------|
| Ultralytics YOLOv11 | https://github.com/ultralytics/ultralytics | Object detection |
| Ultralytics Docs | https://docs.ultralytics.com/ | Training guide, API reference |
| Roboflow supervision | https://github.com/roboflow/supervision | ByteTrack, annotation utilities |
| supervision Docs | https://supervision.roboflow.com/ | Complete API reference |
| Ollama | https://ollama.ai/ | Local LLM serving |
| Ollama Python | https://github.com/ollama/ollama-python | Python client |
| Gradio | https://gradio.app/ | Demo UI |
| OpenCV Python | https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html | Computer vision |
| scipy.spatial | https://docs.scipy.org/doc/scipy/reference/spatial.html | Voronoi, KD-tree |
| scikit-learn KMeans | https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html | Team assignment |
| shapely | https://shapely.readthedocs.io/ | Polygon clipping for Voronoi |
| decord | https://github.com/dmlc/decord | Fast video decoding |
| yt-dlp | https://github.com/yt-dlp/yt-dlp | YouTube clip download |
| uv (package manager) | https://github.com/astral-sh/uv | Python environment management |

---

## 17.2 Datasets and Data Sources

| Resource | URL | Purpose |
|----------|-----|---------|
| Roboflow Football Detection | https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc | YOLOv11 fine-tuning data |
| SoccerNet | https://www.soccer-net.org/ | Match videos for testing |
| SoccerNet SDK GitHub | https://github.com/SoccerNet/soccernet | Video downloader SDK |
| SoccerNet Camera Calibration | https://github.com/SoccerNet/sn-calibration | Keypoint training data |
| DFL Bundesliga Kaggle | https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout | Short broadcast clips |
| StatsBomb Open Data | https://github.com/statsbomb/open-data | Ground truth event data |
| TrackEval | https://github.com/JonathonLuiten/TrackEval | Tracking evaluation toolkit |
| SoccerNet-Echoes | https://github.com/SoccerNet/sn-echoes | Football commentary audio (Phase 2) |

---

## 17.3 Model Weights

| Resource | URL | Purpose |
|----------|-----|---------|
| Qwen2.5 7B (HuggingFace) | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct | Commentary model |
| Qwen2.5 Ollama page | https://ollama.ai/library/qwen2.5 | Easy local serving |
| YOLOv11 release assets | https://github.com/ultralytics/assets/releases/tag/v8.3.0 | Base model weights |

---

## 17.4 Research Papers

| Paper | URL | Relevance |
|-------|-----|-----------|
| ByteTrack (Zhang et al., 2022) | https://arxiv.org/abs/2110.06864 | Tracking algorithm |
| Segment Anything 2 | https://arxiv.org/abs/2408.00714 | Reference for video segmentation (not used but relevant) |
| TVCalib: Camera Calibration for Sports | https://arxiv.org/abs/2303.11128 | Homography reference |
| SoccerNet Camera Calibration Challenge | https://arxiv.org/abs/2104.09403 | Pitch keypoint detection |
| LLM-Commentator | https://www.sciencedirect.com/science/article/pii/S0950705124008530 | Football commentary with LLMs |
| Qwen2.5 Technical Report | https://arxiv.org/abs/2412.15115 | Model capabilities |

---

## 17.5 Key Tutorials and Guides

| Resource | URL | Purpose |
|----------|-----|---------|
| YOLOv11 custom training guide | https://docs.ultralytics.com/modes/train/ | Fine-tuning instructions |
| Camera calibration (Roboflow blog) | https://blog.roboflow.com/camera-calibration-sports-computer-vision/ | Pitch homography guide |
| ByteTrack paper walkthrough | https://paperswithcode.com/paper/bytetrack-multi-object-tracking-by | Understanding the algorithm |
| supervision tutorial | https://supervision.roboflow.com/develop/how-to/track-objects/ | ByteTrack integration |
| Ollama model library | https://ollama.ai/library | Available models |
| Gradio quickstart | https://www.gradio.app/guides/quickstart | UI implementation |
| mplsoccer (visualization reference) | https://mplsoccer.readthedocs.io/ | Pitch drawing reference |

---

## 17.6 Related Projects to Study

| Project | URL | What to Learn |
|---------|-----|--------------|
| roboflow/sports (official) | https://github.com/roboflow/sports | Reference football analysis pipeline |
| Darkmyter/Football-Players-Tracking | https://github.com/Darkmyter/Football-Players-Tracking | YOLOv8 + ByteTrack integration |
| football_analysis_yolo | https://github.com/TrishamBP/football_analysis_yolo | Camera motion + speed estimation |
| qassem0x/computer-vision-football-analysis | https://github.com/qassem0x/computer-vision-football-analysis | Homography + view transformation |
| andrewRowlinson/mplsoccer | https://github.com/andrewRowlinson/mplsoccer | Pitch visualization (reference) |

---

## 17.7 Community and Support

| Resource | URL | Purpose |
|----------|-----|---------|
| Ultralytics Discord | https://ultralytics.com/discord | YOLOv11 questions |
| Roboflow Forum | https://discuss.roboflow.com/ | Dataset and supervision questions |
| Ollama Discord | https://discord.gg/ollama | Ollama model serving questions |
| SoccerNet Discord | https://discord.gg/soccer-net | Dataset access, CV sports questions |
| Papers With Code | https://paperswithcode.com/task/multi-object-tracking | Tracking state of the art |

---

---

# APPENDIX A — Configuration Reference

```python
# gaffer/config.py — all configurable constants in one place

# ─── Pitch Dimensions ─────────────────────────────────────────────────
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

# Standard pitch marking positions (meters)
PITCH_KEYPOINTS = {
    "left_corner_top":         (0.0,   0.0),
    "left_corner_bottom":      (0.0,  68.0),
    "right_corner_top":        (105.0,  0.0),
    "right_corner_bottom":     (105.0, 68.0),
    "left_penalty_spot":       (11.0, 34.0),
    "right_penalty_spot":      (94.0, 34.0),
    "center_spot":             (52.5, 34.0),
    "left_goal_center":        (0.0,  34.0),
    "right_goal_center":       (105.0, 34.0),
    "halfway_line_top":        (52.5,  0.0),
    "halfway_line_bottom":     (52.5, 68.0),
    "left_penalty_box_tl":     (0.0,  13.85),
    "left_penalty_box_tr":     (16.5, 13.85),
    "left_penalty_box_br":     (16.5, 54.15),
    "left_penalty_box_bl":     (0.0,  54.15),
    "right_penalty_box_tl":    (88.5, 13.85),
    "right_penalty_box_tr":    (105.0, 13.85),
    "right_penalty_box_br":    (105.0, 54.15),
    "right_penalty_box_bl":    (88.5, 54.15),
}

# ─── Detection ────────────────────────────────────────────────────────
DETECTION_CONF_THRESHOLD = 0.35
DETECTION_IOU_THRESHOLD = 0.45
DETECTION_IMG_SIZE = 1280
CLASS_PLAYER = 0
CLASS_GOALKEEPER = 1
CLASS_REFEREE = 2
CLASS_BALL = 3

# ─── Tracking ─────────────────────────────────────────────────────────
TRACK_THRESH = 0.45
TRACK_BUFFER_FRAMES = 30       # frames a lost track survives
MATCH_THRESH = 0.8
DEFAULT_FPS = 25

# ─── Team Assignment ──────────────────────────────────────────────────
TEAM_KMEANS_CLUSTERS = 3       # teamA, teamB, goalkeeper cluster
JERSEY_CROP_TOP_MARGIN = 0.10  # exclude head
JERSEY_CROP_BOTTOM_MARGIN = 0.20  # exclude shorts
JERSEY_MIN_SATURATION = 40     # filter grayed-out pixels

# ─── Camera Motion ────────────────────────────────────────────────────
OPTICAL_FLOW_MAX_CORNERS = 200
OPTICAL_FLOW_QUALITY = 0.01
OPTICAL_FLOW_MIN_DISTANCE = 30
SCENE_CHANGE_HIST_THRESHOLD = 0.4

# ─── Spatial Analytics ────────────────────────────────────────────────
PRESSING_RADIUS_M = 10.0
SPRINT_THRESHOLD_MS = 7.0      # m/s (≈ 25 km/h)
SPRINT_MIN_FRAMES = 3
FORMATION_WINDOW_FRAMES = 125  # analyze formation every 5 seconds at 25fps
VELOCITY_WINDOW_FRAMES = 5     # frames to average for velocity calculation

# ─── Event Detection ──────────────────────────────────────────────────
SHOT_SPEED_THRESHOLD_MS = 15.0
SHOT_GOAL_ANGLE_THRESHOLD = 0.7  # cosine similarity
HIGH_PRESS_INTENSITY_THRESHOLD = 3
FORMATION_SHIFT_COOLDOWN_FRAMES = 125  # don't fire again for 5 sec

# ─── Commentary ───────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"
COMMENTARY_TEMPERATURE = 0.7
COMMENTARY_MAX_TOKENS = 120
ANALYTICS_TEMPERATURE = 0.4

# ─── Output ───────────────────────────────────────────────────────────
MINIMAP_WIDTH = 300
MINIMAP_HEIGHT = 200
MINIMAP_ALPHA = 0.85
OUTPUT_VIDEO_CODEC = "mp4v"
OUTPUT_VIDEO_FPS = 25
TEAM_A_COLOR_BGR = (0, 0, 220)    # Red (BGR)
TEAM_B_COLOR_BGR = (220, 0, 0)    # Blue (BGR)
BALL_COLOR_BGR = (0, 255, 255)    # Yellow
REFEREE_COLOR_BGR = (0, 0, 0)    # Black
SPRINT_COLOR_BGR = (0, 165, 255)  # Orange
DEF_LINE_COLOR_BGR = (0, 255, 0) # Green
```

---

# APPENDIX B — .env.example

```bash
# Ollama settings
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b

# Model paths
DETECTION_MODEL_PATH=weights/yolov11_football.pt
KEYPOINT_MODEL_PATH=weights/keypoint_model.pt

# Processing
DEFAULT_COMMENTARY_MODE=tactical  # tactical | excited | analytical
DEFAULT_FPS=25
GPU_DEVICE=cuda  # cuda | mps | cpu

# Output
OUTPUT_DIR=outputs/
LOG_LEVEL=INFO

# Development
ROBOFLOW_API_KEY=your_key_here  # for dataset download only
```

---

# APPENDIX C — Makefile

```makefile
.PHONY: setup demo test lint clean

setup:
	uv sync
	chmod +x scripts/*.sh
	./scripts/setup_ollama.sh
	./scripts/download_weights.sh
	@echo "✅ Setup complete. Run 'make demo' to start."

demo:
	uv run python app/gradio_app.py

test:
	uv run pytest tests/ -v --tb=short

test-unit:
	uv run pytest tests/unit/ -v

test-integration:
	uv run pytest tests/integration/ -v

lint:
	uv run ruff check gaffer/ app/ tests/
	uv run ruff format --check gaffer/ app/ tests/

format:
	uv run ruff format gaffer/ app/ tests/

benchmark:
	uv run python scripts/run_benchmark.py \
		--clip data/test_clips/sample_60s.mp4 \
		--output outputs/benchmark/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf outputs/
```

---

*End of Blueprint — The Gaffer v1.0*
*Total sections: 17 + 3 appendices*
*Built for IIIT Hyderabad CS Student Portfolio*
*All models: local and free. No paid APIs.*
