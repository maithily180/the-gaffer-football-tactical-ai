# The Gaffer — Football Tactical AI Analyst

> Local, open-source football video analysis. Upload a clip, get back an annotated video with tactical overlays, event-triggered commentary, and a match report. No cloud required.

---

## Quick Start

```bash
# 1. Install uv (Windows)
winget install astral.uv

# 2. Create environment
uv sync --extra dev

# 3. Verify all tools work BEFORE building anything
uv run python scripts/day0_verify.py

# 4. Launch demo (after setup is complete)
make demo
```

## Day 0 Verification

Before writing any code, run the bottleneck check:

```bash
uv run python scripts/day0_verify.py
```

This checks:
- Python 3.11 is active
- YOLOv11 loads and runs inference
- OpenVINO export works and measures speed
- Ollama is running with qwen2.5:3b
- ffmpeg is available
- OpenCV can read/write video

## Project Structure

```
gaffer/             Main Python package
app/                Gradio demo UI
scripts/            Setup, export, benchmark utilities
notebooks/          Exploration notebooks (start here)
data/               Pitch template, formation templates
weights/            Model weights (gitignored)
outputs/            Processing results (gitignored)
tests/              Unit and integration tests
```

## Development Milestones

| Version | Target | What works |
|---------|--------|-----------|
| v0.1 | Day 4 | Detection + tracking + team colors → annotated video |
| v0.5 | Day 10 | + Minimap + Voronoi + pressing meter + commentary |
| v1.0 | Day 21 | + Full Gradio UI + match report + fine-tuned model |

## Tech Stack

- **Detection**: YOLOv11 (Ultralytics) + OpenVINO for Intel GPU/CPU
- **Tracking**: ByteTrack via supervision
- **Analytics**: NumPy + SciPy (Voronoi, ConvexHull, cKDTree)
- **Commentary**: Qwen2.5:3b via Ollama (local LLM)
- **UI**: Gradio

## Hardware

Built for: Intel Core Ultra 9 185H · 32 GB RAM · Intel Arc integrated graphics · Windows 11

All inference runs locally. No paid APIs.
