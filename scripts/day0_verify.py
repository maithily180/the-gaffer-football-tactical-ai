"""
Day 0 Bottleneck Verification
==============================
Run this BEFORE writing any project code.
Every major risk in the engineering roadmap is tested here.

Usage:
    uv run python scripts/day0_verify.py

Each check prints PASS / WARN / FAIL with details.
Fix all FAILs before proceeding to Day 1.
A WARN is a known limitation that has a documented mitigation.
"""

import sys

# Force UTF-8 output on Windows (avoids cp1252 errors with box-drawing chars)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import time
import subprocess
import urllib.request
import urllib.error
import json
import tempfile
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ─── Terminal colours (Windows-safe via ANSI) ─────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def enable_ansi():
    """Enable ANSI escape codes on Windows."""
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

def header(text):
    print(f"\n{BOLD}{CYAN}{'─' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * 60}{RESET}")

def passed(msg):
    print(f"  {GREEN}✓ PASS{RESET}  {msg}")

def warned(msg):
    print(f"  {YELLOW}⚠ WARN{RESET}  {msg}")

def failed(msg):
    print(f"  {RED}✗ FAIL{RESET}  {msg}")

def info(msg):
    print(f"         {msg}")


results = {"pass": 0, "warn": 0, "fail": 0}

def record(status, msg, detail=None):
    results[status] += 1
    fn = {"pass": passed, "warn": warned, "fail": failed}[status]
    fn(msg)
    if detail:
        info(detail)


# ──────────────────────────────────────────────────────────────────────────────
# CHECK 1: Python version
# ──────────────────────────────────────────────────────────────────────────────
header("CHECK 1 — Python Version")

major, minor = sys.version_info.major, sys.version_info.minor
version_str = f"{major}.{minor}.{sys.version_info.micro}"

if major == 3 and minor == 11:
    record("pass", f"Python {version_str}")
elif major == 3 and minor in (10, 12):
    record("warn", f"Python {version_str} — recommended is 3.11",
           "Some Ultralytics internals have intermittent issues on 3.12. "
           "3.10 is fine but install with: uv venv --python 3.11")
else:
    record("fail", f"Python {version_str} — need Python 3.11",
           "Run: uv python install 3.11 && uv venv --python 3.11")


# ──────────────────────────────────────────────────────────────────────────────
# CHECK 2: Core package imports
# ──────────────────────────────────────────────────────────────────────────────
header("CHECK 2 — Core Package Imports")

packages = [
    ("numpy",       "numpy"),
    ("opencv",      "cv2"),
    ("ultralytics", "ultralytics"),
    ("supervision", "supervision"),
    ("scipy",       "scipy"),
    ("sklearn",     "sklearn"),
    ("pydantic",    "pydantic"),
    ("tqdm",        "tqdm"),
    ("gradio",      "gradio"),
    ("ollama",      "ollama"),
]

missing = []
for name, module in packages:
    try:
        __import__(module)
        record("pass", f"{name} imports OK")
    except ImportError as e:
        record("fail", f"{name} not found — run: uv sync",
               str(e))
        missing.append(name)

if missing:
    info(f"Missing packages: {', '.join(missing)}")
    info("Fix: uv sync  (or: uv add " + " ".join(missing) + ")")


# ──────────────────────────────────────────────────────────────────────────────
# CHECK 3: PyTorch device detection
# ──────────────────────────────────────────────────────────────────────────────
header("CHECK 3 — PyTorch & Device")

try:
    import torch
    record("pass", f"PyTorch {torch.__version__} imported")

    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        record("pass", f"CUDA available — {device}")
    else:
        record("warn", "CUDA not available — running on CPU",
               "Expected on Intel Arc machine. OpenVINO will be used instead.")

    # Check if MPS (Apple Silicon) is available — won't be on Windows
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        record("pass", "MPS (Apple Silicon) available")

    # Test a basic tensor op
    t = torch.zeros(1, 3, 64, 64)
    _ = t + 1
    record("pass", "PyTorch tensor operations work")

except ImportError:
    record("fail", "PyTorch not installed — run: uv sync")
except Exception as e:
    record("fail", f"PyTorch error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# CHECK 4: YOLOv11 model load + inference timing
# ──────────────────────────────────────────────────────────────────────────────
header("CHECK 4 — YOLOv11 Load & Inference Speed")

try:
    from ultralytics import YOLO
    import numpy as np

    print("  Downloading yolo11n.pt (21 MB) if not cached...")
    t0 = time.perf_counter()
    model = YOLO("yolo11n.pt")
    load_time = time.perf_counter() - t0
    record("pass", f"YOLOv11n loaded in {load_time:.1f}s")

    # Warm-up run
    dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    _ = model(dummy_frame, imgsz=640, verbose=False)

    # Timed inference — 10 frames
    N = 10
    t0 = time.perf_counter()
    for _ in range(N):
        results_yolo = model(dummy_frame, imgsz=640, verbose=False)
    elapsed = time.perf_counter() - t0
    ms_per_frame = (elapsed / N) * 1000

    if ms_per_frame < 100:
        record("pass", f"Inference: {ms_per_frame:.0f}ms/frame at imgsz=640 (CPU) — fast enough")
    elif ms_per_frame < 250:
        record("warn", f"Inference: {ms_per_frame:.0f}ms/frame at imgsz=640 (CPU)",
               "Acceptable. OpenVINO export should bring this under 100ms.")
    else:
        record("warn", f"Inference: {ms_per_frame:.0f}ms/frame at imgsz=640 (CPU)",
               "Slow — but OpenVINO export is expected to improve this 2–5×. "
               "Run Check 5 (OpenVINO) before worrying.")

    # Classes check
    class_names = model.names
    info(f"Model classes: {class_names}")

except Exception as e:
    record("fail", f"YOLOv11 error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# CHECK 5: OpenVINO export + inference speed
# ──────────────────────────────────────────────────────────────────────────────
header("CHECK 5 — OpenVINO Export & Accelerated Inference")

try:
    import openvino
    record("pass", f"OpenVINO {openvino.__version__} imported")

    from ultralytics import YOLO
    import numpy as np

    ov_export_dir = ROOT / "weights" / "yolo11n_openvino_verify"

    if not ov_export_dir.exists():
        print("  Exporting yolo11n to OpenVINO format (one-time, ~30s)...")
        t0 = time.perf_counter()
        model_base = YOLO("yolo11n.pt")
        model_base.export(format="openvino", imgsz=640)

        # Ultralytics exports to a sibling directory next to the weights
        # (name depends on YOLO version: yolo11n_openvino_model or yolo11n_openvino)
        for candidate in ["yolo11n_openvino_model", "yolo11n_openvino"]:
            default_export = Path(candidate)
            if default_export.exists():
                import shutil
                shutil.move(str(default_export), str(ov_export_dir))
                break

        export_time = time.perf_counter() - t0
        record("pass", f"OpenVINO export completed in {export_time:.1f}s")
    else:
        record("pass", f"OpenVINO export already exists at {ov_export_dir.name}/")

    # Load and benchmark OpenVINO model
    ov_model_path = ov_export_dir / "yolo11n.xml"
    if not ov_model_path.exists():
        # Try alternate naming
        xml_files = list(ov_export_dir.glob("*.xml"))
        if xml_files:
            ov_model_path = xml_files[0]

    if ov_model_path.exists():
        print(f"  Loading OpenVINO model from {ov_model_path.name}...")
        ov_model = YOLO(str(ov_model_path), task="detect")

        dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

        # Warm-up
        _ = ov_model(dummy_frame, imgsz=640, verbose=False)

        # Timed
        N = 10
        t0 = time.perf_counter()
        for _ in range(N):
            _ = ov_model(dummy_frame, imgsz=640, verbose=False)
        elapsed = time.perf_counter() - t0
        ov_ms = (elapsed / N) * 1000

        if ov_ms < 100:
            record("pass", f"OpenVINO inference: {ov_ms:.0f}ms/frame — GREEN LIGHT for Day 1")
        elif ov_ms < 200:
            record("warn", f"OpenVINO inference: {ov_ms:.0f}ms/frame",
                   "For 500 detection calls (60s clip, every 3rd frame): "
                   f"~{ov_ms*500/1000:.0f}s. Pipeline stays under 6min total.")
        else:
            record("warn", f"OpenVINO inference: {ov_ms:.0f}ms/frame — slower than expected",
                   "Mitigation: use imgsz=480 instead of 640. "
                   "Run: model.export(format='openvino', imgsz=480)")
    else:
        record("warn", "OpenVINO XML model file not found after export",
               f"Expected: {ov_export_dir}/. Check the export directory manually.")

except ImportError:
    record("warn", "OpenVINO not installed",
           "Run: uv add openvino   — then re-run this script.")
except Exception as e:
    record("warn", f"OpenVINO check error: {e}",
           "OpenVINO may still work — check manually with scripts/export_openvino.py")


# ──────────────────────────────────────────────────────────────────────────────
# CHECK 6: OpenCV video I/O
# ──────────────────────────────────────────────────────────────────────────────
header("CHECK 6 — OpenCV Video I/O")

try:
    import cv2
    import numpy as np

    record("pass", f"OpenCV {cv2.__version__} imported")

    # Write a tiny test video and read it back
    with tempfile.TemporaryDirectory() as tmpdir:
        test_video = os.path.join(tmpdir, "test.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(test_video, fourcc, 25, (320, 240))
        if not writer.isOpened():
            record("fail", "cv2.VideoWriter failed to open — ffmpeg/codec issue",
                   "Install ffmpeg: winget install ffmpeg  then restart your terminal.")
        else:
            for _ in range(10):
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                writer.write(frame)
            writer.release()

            cap = cv2.VideoCapture(test_video)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret:
                    record("pass", "VideoWriter + VideoCapture round-trip works")
                else:
                    record("warn", "VideoCapture opened but could not read frame",
                           "ffmpeg may not be installed. Run: winget install ffmpeg")
            else:
                record("warn", "VideoCapture could not open test video",
                       "ffmpeg may not be installed. Run: winget install ffmpeg")

    # Test optical flow (needed for camera motion)
    prev = np.random.randint(0, 255, (240, 320), dtype=np.uint8)
    curr = np.random.randint(0, 255, (240, 320), dtype=np.uint8)
    pts = cv2.goodFeaturesToTrack(prev, maxCorners=50, qualityLevel=0.01, minDistance=10)
    if pts is not None:
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev, curr, pts, None)
        record("pass", "Lucas-Kanade optical flow works")
    else:
        record("warn", "goodFeaturesToTrack returned None on random frame",
               "This is expected on random noise — will work on real footage.")

    # Test homography
    src = np.float32([[0,0],[1,0],[1,1],[0,1]]) * 100
    dst = np.float32([[10,10],[110,10],[110,110],[10,110]])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC)
    if H is not None:
        record("pass", "cv2.findHomography works")
    else:
        record("fail", "cv2.findHomography returned None")

except ImportError:
    record("fail", "OpenCV (cv2) not installed — run: uv add opencv-python")
except Exception as e:
    record("fail", f"OpenCV error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# CHECK 7: ffmpeg system binary
# ──────────────────────────────────────────────────────────────────────────────
header("CHECK 7 — ffmpeg System Binary")

try:
    result = subprocess.run(
        ["ffmpeg", "-version"],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode == 0:
        first_line = result.stdout.split("\n")[0]
        record("pass", f"ffmpeg found — {first_line}")
    else:
        record("warn", "ffmpeg returned non-zero exit code",
               result.stderr[:200])
except FileNotFoundError:
    record("fail", "ffmpeg not found in PATH",
           "Install: winget install ffmpeg\n"
           "         Then open a NEW terminal (PATH needs refresh)")
except subprocess.TimeoutExpired:
    record("warn", "ffmpeg timed out — likely installed but slow to start")
except Exception as e:
    record("warn", f"ffmpeg check error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# CHECK 8: Ollama process running
# ──────────────────────────────────────────────────────────────────────────────
header("CHECK 8 — Ollama Process")

ollama_running = False
try:
    req = urllib.request.Request(
        "http://localhost:11434/api/tags",
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=3) as resp:
        data = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
        record("pass", f"Ollama is running — {len(models)} model(s) available")
        if models:
            info(f"Available: {', '.join(models)}")
        ollama_running = True
except urllib.error.URLError:
    record("fail", "Ollama is NOT running",
           "Start it: open a new terminal and run: ollama serve\n"
           "          Or install from: https://ollama.ai/download")
except Exception as e:
    record("fail", f"Ollama connection error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# CHECK 9: qwen2.5:3b available
# ──────────────────────────────────────────────────────────────────────────────
header("CHECK 9 — Qwen2.5:3b Model Available")

if ollama_running:
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/tags",
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]

        target = "qwen2.5:3b"
        if any(target in m for m in models):
            record("pass", f"{target} is pulled and ready")
        else:
            record("warn", f"{target} not found in Ollama",
                   f"Pull it: ollama pull {target}  (downloads 1.9 GB)")
    except Exception as e:
        record("warn", f"Could not check model list: {e}")
else:
    record("warn", "Skipped — Ollama not running (fix Check 8 first)")


# ──────────────────────────────────────────────────────────────────────────────
# CHECK 10: Ollama inference speed
# ──────────────────────────────────────────────────────────────────────────────
header("CHECK 10 — Ollama Inference Speed")

if ollama_running:
    try:
        import ollama as ollama_client

        # Use explicit Client — module-level functions may fail on some Windows setups
        client = ollama_client.Client(host="http://localhost:11434")

        target_model = "qwen2.5:3b"
        models_resp = client.list()
        # SDK returns Pydantic models — access via .models and .model attribute
        available = [m.model for m in getattr(models_resp, "models", [])]

        if not any(target_model in m for m in available):
            record("warn", f"{target_model} not pulled — skipping speed test",
                   f"Run: ollama pull {target_model}")
        else:
            prompt = (
                "You are a football analyst. A high press event occurred: "
                "4 opponents within 10 meters, nearest at 3.2m. "
                "Generate one sentence of tactical commentary."
            )

            print(f"  Testing inference with {target_model}...")
            t0 = time.perf_counter()
            response = client.generate(
                model=target_model,
                prompt=prompt,
                options={"num_predict": 80, "temperature": 0.7},
            )
            elapsed = time.perf_counter() - t0
            text = getattr(response, "response", "").strip()
            tokens = getattr(response, "eval_count", 0) or 0
            tok_s = tokens / elapsed if elapsed > 0 else 0

            info(f'Response: "{text[:120]}..."' if len(text) > 120 else f'Response: "{text}"')
            info(f"Tokens: {tokens} | Time: {elapsed:.1f}s | Speed: {tok_s:.1f} tok/s")

            if elapsed < 10:
                record("pass", f"Commentary in {elapsed:.1f}s ({tok_s:.0f} tok/s) — fast enough")
            elif elapsed < 25:
                record("warn", f"Commentary takes {elapsed:.1f}s per event",
                       f"With 8 events: ~{elapsed*8/60:.1f} min for commentary batch. "
                       "Acceptable if run as post-processing step after CV pipeline.")
            else:
                record("warn", f"Commentary takes {elapsed:.1f}s per event — very slow",
                       "Check if Ollama is using Intel GPU: set OLLAMA_INTEL_GPU=1 before 'ollama serve'\n"
                       "Alternatively reduce num_predict to 60 tokens.")

    except ImportError:
        record("warn", "ollama Python client not installed — run: uv add ollama")
    except Exception as e:
        record("warn", f"Ollama inference test error: {e}")
else:
    record("warn", "Skipped — Ollama not running (fix Check 8 first)")


# ──────────────────────────────────────────────────────────────────────────────
# CHECK 11: scipy spatial
# ──────────────────────────────────────────────────────────────────────────────
header("CHECK 11 — SciPy Spatial (Voronoi, ConvexHull, cKDTree)")

try:
    import numpy as np
    from scipy.spatial import Voronoi, ConvexHull, cKDTree

    # Voronoi on 22 player-like positions
    positions = np.random.rand(22, 2) * np.array([105, 68])
    vor = Voronoi(positions)
    record("pass", f"scipy.spatial.Voronoi works — {len(vor.regions)} regions")

    # ConvexHull
    team_a = positions[:11]
    hull = ConvexHull(team_a)
    record("pass", f"scipy.spatial.ConvexHull works — area: {hull.volume:.1f} m²")

    # cKDTree
    tree = cKDTree(positions[:11])
    dist, idx = tree.query([52.5, 34.0])
    record("pass", f"scipy.spatial.cKDTree works — nearest to centre: {dist:.1f}m away")

    # Timing
    N = 1000
    t0 = time.perf_counter()
    for _ in range(N):
        Voronoi(positions)
    elapsed = time.perf_counter() - t0
    ms = elapsed / N * 1000
    if ms < 2.0:
        record("pass", f"Voronoi compute time: {ms:.2f}ms per call — well within budget")
    else:
        record("warn", f"Voronoi compute time: {ms:.2f}ms per call",
               "Still fine for every-5th-frame rendering throttle.")

except ImportError:
    record("fail", "scipy not installed — run: uv add scipy")
except Exception as e:
    record("fail", f"scipy error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# CHECK 12: ByteTrack via supervision
# ──────────────────────────────────────────────────────────────────────────────
header("CHECK 12 — ByteTrack via supervision")

try:
    import supervision as sv
    import numpy as np

    record("pass", f"supervision {sv.__version__} imported")

    # supervision 0.28+ renamed ByteTracker → ByteTrack
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        tracker = sv.ByteTrack()

    # Feed mock detections for 5 frames
    for i in range(5):
        mock_detections = sv.Detections(
            xyxy=np.array([
                [100 + i, 200, 160 + i, 280],
                [300 + i, 150, 360 + i, 230],
                [500 + i, 300, 560 + i, 380],
            ], dtype=np.float32),
            confidence=np.array([0.9, 0.85, 0.8]),
            class_id=np.array([0, 0, 0]),
        )
        tracked = tracker.update_with_detections(mock_detections)

    if tracked.tracker_id is not None:
        record("pass", f"ByteTrack works — {len(tracked.tracker_id)} tracks after 5 frames")
    else:
        record("warn", "ByteTrack returned no tracker_id — check supervision version")

except ImportError:
    record("fail", "supervision not installed — run: uv add supervision")
except Exception as e:
    record("fail", f"ByteTracker error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# CHECK 13: yt-dlp
# ──────────────────────────────────────────────────────────────────────────────
header("CHECK 13 — yt-dlp (clip downloader)")

try:
    result = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "--version"],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode == 0:
        record("pass", f"yt-dlp {result.stdout.strip()} available")
    else:
        record("warn", "yt-dlp not responding correctly",
               "Install: uv add yt-dlp")
except FileNotFoundError:
    record("warn", "yt-dlp not installed",
           "Install: uv add yt-dlp")
except Exception as e:
    record("warn", f"yt-dlp check error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# CHECK 14: Project structure
# ──────────────────────────────────────────────────────────────────────────────
header("CHECK 14 — Project Structure")

expected_dirs = [
    "gaffer", "gaffer/video", "gaffer/detection", "gaffer/tracking",
    "gaffer/calibration", "gaffer/analytics", "gaffer/events",
    "gaffer/commentary", "gaffer/output", "gaffer/utils",
    "app", "scripts", "notebooks", "data", "weights", "outputs",
    "tests", "tests/unit", "tests/integration",
]

all_present = True
for d in expected_dirs:
    path = ROOT / d
    if not path.exists():
        record("fail", f"Missing directory: {d}")
        all_present = False

if all_present:
    record("pass", f"All {len(expected_dirs)} expected directories present")

expected_files = [
    "pyproject.toml", ".python-version", ".env.example",
    "Makefile", "README.md", ".gitignore",
    "gaffer/__init__.py", "gaffer/config.py",
]
for f in expected_files:
    path = ROOT / f
    if not path.exists():
        record("warn", f"Missing file: {f}")


# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
header("SUMMARY")

total = results["pass"] + results["warn"] + results["fail"]
print(f"  {GREEN}PASS: {results['pass']}{RESET}  "
      f"{YELLOW}WARN: {results['warn']}{RESET}  "
      f"{RED}FAIL: {results['fail']}{RESET}  "
      f"(of {total} checks)")
print()

if results["fail"] == 0 and results["warn"] == 0:
    print(f"  {GREEN}{BOLD}ALL CHECKS PASSED — you are ready for Day 1.{RESET}")
elif results["fail"] == 0:
    print(f"  {YELLOW}{BOLD}No failures. Review WARNs above — most have documented mitigations.{RESET}")
    print(f"  {YELLOW}Proceed to Day 1 but keep the WARNs in mind.{RESET}")
else:
    print(f"  {RED}{BOLD}Fix all FAILs before proceeding.{RESET}")
    print(f"  {RED}Each FAIL has a fix printed above.{RESET}")

print()
print("  Next step after all checks pass:")
print("    → Download a test clip: uv run python scripts/download_clip.py <YouTube-URL>")
print("    → Open notebooks/01_yolo_test.ipynb and run detections")
print()
