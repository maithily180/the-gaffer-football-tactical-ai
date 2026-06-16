"""
scripts/train_detector.py
─────────────────────────
Fine-tune YOLOv11 into a football-specific detector for Gaffer.

Designed to run on a CUDA GPU (Google Colab free T4 is the target). Local
training on the Intel Arc machine is NOT viable — use Colab.

THE LOAD-BEARING DETAIL — class order
-------------------------------------
Gaffer's canonical class order (gaffer/config.py) is:

    player=0  goalkeeper=1  referee=2  ball=3

The Roboflow `football-players-detection` dataset ships in *alphabetical*
order:

    ball=0  goalkeeper=1  player=2  referee=3

gaffer/detection/detector.py trusts the model's raw class index directly when
a football model is loaded (`cls_id = raw_cls`). If we trained on the raw
dataset, the model would emit 0 for a ball and Gaffer would label it a player —
a silent, catastrophic mislabel. So before training we REMAP every label file
to Gaffer's order and rewrite data.yaml accordingly.

Usage
-----
    # 1. download dataset from Roboflow (gives ROBOFLOW alphabetical order)
    # 2. remap it to Gaffer's order, in place:
    python scripts/train_detector.py remap --dataset path/to/football-players-detection-12

    # 3. train:
    python scripts/train_detector.py train \
        --data path/to/football-players-detection-12/data.yaml \
        --model yolo11n.pt --epochs 50 --imgsz 1280 --batch 16

You can also import remap_dataset() / train() from the Colab notebook.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Windows consoles default to cp1252 and choke on the Unicode glyphs below.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ─── Canonical Gaffer class scheme ─────────────────────────────────────────────
# index == Gaffer class id.  Mirrors gaffer/config.CLASS_NAMES — keep in sync.
GAFFER_NAMES: list[str] = ["player", "goalkeeper", "referee", "ball"]

# Roboflow alphabetical id  →  Gaffer id
#   ball(0)→3   goalkeeper(1)→1   player(2)→0   referee(3)→2
ROBOFLOW_TO_GAFFER: dict[int, int] = {0: 3, 1: 1, 2: 0, 3: 2}

# Roboflow's class names in *their* index order — used to verify we are
# remapping a dataset that actually matches the assumed alphabetical layout.
_EXPECTED_ROBOFLOW_NAMES = ["ball", "goalkeeper", "player", "referee"]


# ─── Remap ─────────────────────────────────────────────────────────────────────

def _remap_label_file(txt: Path, mapping: dict[int, int]) -> int:
    """Rewrite one YOLO label .txt in place. Returns number of boxes remapped."""
    lines_out: list[str] = []
    n = 0
    for line in txt.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        old_cls = int(parts[0])
        if old_cls not in mapping:
            # Unknown class id — drop it rather than silently mislabel.
            continue
        parts[0] = str(mapping[old_cls])
        lines_out.append(" ".join(parts))
        n += 1
    txt.write_text("\n".join(lines_out) + ("\n" if lines_out else ""))
    return n


def remap_dataset(dataset_dir: str | Path,
                  mapping: dict[int, int] = ROBOFLOW_TO_GAFFER) -> None:
    """
    Remap every label file under `dataset_dir` from Roboflow order to Gaffer
    order, and rewrite data.yaml `names` to Gaffer's order. Idempotency is
    guarded by a sentinel key written into data.yaml.
    """
    dataset_dir = Path(dataset_dir)
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found in {dataset_dir}")

    cfg = yaml.safe_load(yaml_path.read_text())

    if cfg.get("gaffer_remapped"):
        print(f"  [remap] {dataset_dir.name} already remapped — skipping.")
        return

    names = cfg.get("names")
    # Roboflow exports names either as a list or an index→name dict.
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    if names != _EXPECTED_ROBOFLOW_NAMES:
        print(f"  [remap] WARNING: dataset names are {names}, expected "
              f"{_EXPECTED_ROBOFLOW_NAMES}. Verify the mapping before training!",
              file=sys.stderr)

    total_files = total_boxes = 0
    for label_txt in dataset_dir.rglob("labels/*.txt"):
        total_boxes += _remap_label_file(label_txt, mapping)
        total_files += 1

    cfg["names"] = GAFFER_NAMES
    cfg["nc"] = len(GAFFER_NAMES)
    cfg["gaffer_remapped"] = True
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    print(f"  [remap] {total_files} label files, {total_boxes} boxes → Gaffer "
          f"order {GAFFER_NAMES}")
    print(f"  [remap] data.yaml names rewritten and tagged gaffer_remapped: true")


# ─── Train ─────────────────────────────────────────────────────────────────────

def train(
    data_yaml: str | Path,
    model: str = "yolo11n.pt",
    epochs: int = 50,
    imgsz: int = 1280,
    batch: int = 16,
    device: str | int = 0,
    project: str = "runs/gaffer_detector",
    name: str | None = None,
    patience: int = 15,
):
    """
    Fine-tune `model` on `data_yaml`. Returns the ultralytics results object.

    ultralytics natively writes best.pt, results.csv, confusion_matrix.png,
    and PR/F1 curves into  {project}/{name}/ . We surface the key numbers after.
    """
    from ultralytics import YOLO  # imported here so `remap` runs without torch

    name = name or f"{Path(model).stem}_football"
    print(f"╭─ Training {model} on {data_yaml}")
    print(f"│  epochs={epochs}  imgsz={imgsz}  batch={batch}  device={device}")
    print(f"╰─ output → {project}/{name}/weights/best.pt")

    yolo = YOLO(model)
    results = yolo.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        patience=patience,
        # Football-tuned augmentation: players are small & dense, ball is tiny.
        # No vertical flip (football has a clear up/down), modest mosaic.
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        close_mosaic=10,      # disable mosaic for last 10 epochs → cleaner boxes
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        plots=True,
    )
    summarize_metrics(yolo, data_yaml, device)
    return results


def summarize_metrics(yolo, data_yaml: str | Path, device: str | int = 0) -> None:
    """Print a per-class mAP table — the numbers requirement #5/#7 ask for."""
    print("\n── Per-class validation metrics ──")
    metrics = yolo.val(data=str(data_yaml), device=device, plots=True)
    box = metrics.box
    names = yolo.names
    print(f"{'class':<12}{'mAP50':>8}{'mAP50-95':>10}{'precision':>11}{'recall':>9}")
    print(f"{'all':<12}{box.map50:>8.3f}{box.map:>10.3f}"
          f"{box.mp:>11.3f}{box.mr:>9.3f}")
    # per-class: ap_class_index maps row → class id
    for row, cls_id in enumerate(box.ap_class_index):
        cname = names[int(cls_id)]
        print(f"{cname:<12}{box.ap50[row]:>8.3f}{box.ap[row]:>10.3f}"
              f"{box.p[row]:>11.3f}{box.r[row]:>9.3f}")
    print("\nArtifacts (best.pt, confusion_matrix.png, results.csv) saved under "
          "the run directory printed above.")


# ─── CLI ───────────────────────────────────────────────────────────────────────

def _main() -> None:
    p = argparse.ArgumentParser(description="Gaffer detector fine-tuning")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("remap", help="Remap Roboflow dataset to Gaffer class order")
    pr.add_argument("--dataset", required=True, help="Dataset root (contains data.yaml)")

    pt = sub.add_parser("train", help="Fine-tune YOLOv11")
    pt.add_argument("--data", required=True, help="Path to data.yaml")
    pt.add_argument("--model", default="yolo11n.pt", help="Base model (yolo11n.pt / yolo11s.pt)")
    pt.add_argument("--epochs", type=int, default=50)
    pt.add_argument("--imgsz", type=int, default=1280)
    pt.add_argument("--batch", type=int, default=16)
    pt.add_argument("--device", default="0")
    pt.add_argument("--project", default="runs/gaffer_detector")
    pt.add_argument("--name", default=None)
    pt.add_argument("--patience", type=int, default=15)

    args = p.parse_args()
    if args.cmd == "remap":
        remap_dataset(args.dataset)
    elif args.cmd == "train":
        dev = int(args.device) if str(args.device).isdigit() else args.device
        train(args.data, args.model, args.epochs, args.imgsz, args.batch,
              dev, args.project, args.name, args.patience)


if __name__ == "__main__":
    _main()
