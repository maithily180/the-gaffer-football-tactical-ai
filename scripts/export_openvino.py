"""
Export YOLOv11 model to OpenVINO format for fast Intel CPU/GPU inference.

Run after fine-tuning on Colab and downloading weights/yolov11_football.pt.

Usage:
    uv run python scripts/export_openvino.py
    uv run python scripts/export_openvino.py --model weights/yolov11_football.pt --imgsz 640
"""

import argparse
import shutil
import time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent


def export(model_path: Path, imgsz: int = 640):
    from ultralytics import YOLO

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Options:")
        print("  1. Download fine-tuned weights from Colab")
        print("  2. Use base model: python scripts/export_openvino.py --model yolo11n.pt")
        return

    print(f"Exporting: {model_path}")
    print(f"Image size: {imgsz}x{imgsz}")
    print()

    model = YOLO(str(model_path))

    t0 = time.perf_counter()
    model.export(format="openvino", imgsz=imgsz, half=False)
    export_time = time.perf_counter() - t0

    # Ultralytics exports to a folder next to the model file
    stem = model_path.stem
    exported_dir = model_path.parent / f"{stem}_openvino"

    if not exported_dir.exists():
        # May have exported relative to cwd
        exported_dir_cwd = Path(f"{stem}_openvino")
        if exported_dir_cwd.exists():
            target = ROOT / "weights" / f"{stem}_openvino"
            shutil.move(str(exported_dir_cwd), str(target))
            exported_dir = target

    print(f"Export completed in {export_time:.1f}s")
    print(f"Output directory: {exported_dir}")

    # Benchmark the exported model
    print()
    print("Benchmarking OpenVINO model...")
    xml_files = list(exported_dir.glob("*.xml")) if exported_dir.exists() else []

    if not xml_files:
        print("No .xml file found — check export output manually")
        return

    ov_model = YOLO(str(xml_files[0]), task="detect")
    dummy = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    # Warm-up
    for _ in range(3):
        ov_model(dummy, imgsz=imgsz, verbose=False)

    N = 20
    t0 = time.perf_counter()
    for _ in range(N):
        ov_model(dummy, imgsz=imgsz, verbose=False)
    ms = (time.perf_counter() - t0) / N * 1000

    print(f"OpenVINO inference: {ms:.0f}ms/frame at imgsz={imgsz}")

    # Estimate 60s clip time
    clip_frames = 1500
    detect_every = 3
    calls = clip_frames // detect_every
    est = ms * calls / 1000
    print(f"Estimated detection time for 60s clip: {est:.0f}s ({calls} calls, every {detect_every}rd frame)")

    if ms < 100:
        print("\n✓ GREEN — fast enough for the 5-minute pipeline target")
    elif ms < 250:
        print("\n⚠ WARN — acceptable; stays within 6-minute target with every-3rd-frame trick")
    else:
        print(f"\n✗ SLOW — try imgsz=480: python scripts/export_openvino.py --imgsz 480")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO to OpenVINO")
    parser.add_argument("--model", default="weights/yolov11_football.pt",
                        help="Path to .pt weights file")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image size (640 recommended for Intel Arc)")
    args = parser.parse_args()
    export(ROOT / args.model, args.imgsz)
