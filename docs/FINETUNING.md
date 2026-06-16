# Day 5 — Football detector fine-tuning

Replace the base `yolo11n.pt` (COCO: only `person` + `sports ball`) with a
football-aware `yolov11_football.pt` that detects **player / goalkeeper /
referee / ball**. This is the highest-leverage step in the project: every
downstream module — tracking, team assignment, analytics, events, commentary —
improves when detection improves.

## Dataset

**Roboflow `football-players-detection`** (`roboflow-jvuqo/football-players-detection-3zvbc`)

| | |
|---|---|
| Classes | `ball`, `goalkeeper`, `player`, `referee` (4) |
| Images | ~600 annotated broadcast frames (train/valid/test) |
| Labels | YOLO-format bounding boxes |
| License | CC BY 4.0 (free; needs a free Roboflow API key to download) |

**Why this dataset:** its four classes map *exactly* onto Gaffer's class scheme.
No other free dataset lines up this cleanly with the existing pipeline.

**Limitations (design around these):**
- **Ball is heavily under-represented** — one tiny box per frame vs ~20 players.
  Expect lower ball mAP; watch ball false-negatives in the A/B.
- **Small (~600 imgs)** — real overfitting risk. Mitigated with `patience` early
  stopping and `close_mosaic`.
- **Single broadcast style** — domain gap vs the tactical-cam test clips. The
  **local A/B test is the real proof**, not the validation mAP.
- Scale-up option for later: **SoccerNet** (far larger, more preprocessing).

## ⚠️ The class-order trap (most important thing on this page)

Roboflow ships classes in **alphabetical** order:

```
ball=0  goalkeeper=1  player=2  referee=3
```

Gaffer's canonical order (`gaffer/config.py`) is:

```
player=0  goalkeeper=1  referee=2  ball=3
```

`gaffer/detection/detector.py` trusts the model's raw class index directly when a
football model is loaded (`cls_id = raw_cls`). **Train on the raw dataset and the
model emits `0` for a ball, which Gaffer draws as a player** — a silent,
catastrophic mislabel with no error.

Fix: **remap every label file and `data.yaml` to Gaffer's order before training.**
`scripts/train_detector.py remap` (and the notebook's remap cell) do this and tag
`data.yaml` with `gaffer_remapped: true` so it is idempotent.

Mapping: `ball 0→3 · goalkeeper 1→1 · player 2→0 · referee 3→2`.

## Workflow

### 1. Train (Google Colab T4 — local Intel Arc can't train)

Open `notebooks/04_finetune_yolo.ipynb` in Colab, set runtime to **T4 GPU**, then:
1. Install + download dataset (needs free Roboflow API key).
2. **Run the remap cell** — do not skip.
3. Train `yolo11n` (`imgsz=1280`, 50 epochs, `patience=15`, `close_mosaic=10`).
4. Inspect per-class mAP + confusion matrix.
5. Decision gate: train `yolo11s` only if `n` is weak — `s` is ~2× slower at
   inference, which matters because Gaffer runs on an Intel Arc iGPU, not a GPU.
6. Download `best.pt` of the winner.

The reusable trainer is also a CLI (runs on any CUDA box):

```bash
python scripts/train_detector.py remap --dataset path/to/football-players-detection-12
python scripts/train_detector.py train  --data path/to/.../data.yaml \
    --model yolo11n.pt --epochs 50 --imgsz 1280 --batch 16
```

### 2. Install the model

Drop the downloaded `best.pt` at the exact path config expects:

```
weights/yolov11_football.pt
```

`config.DETECTION_MODEL_PATH` points here and `FootballDetector` auto-detects it
(no code change needed — base COCO is used only as a fallback when this is absent).

### 3. Prove it helps (the success gate)

Validation mAP proves the model learned the dataset. It does **not** prove it
helps Gaffer on *your* clips. Run the local A/B:

```bash
uv run python scripts/ab_compare_detectors.py \
    data/test_clips/tactical_playlist_1.mp4 --start 30 --duration 60
```

It runs the identical detect→ByteTrack→PositionStore pipeline with base vs
fine-tuned and prints a side-by-side table.

**Success criterion:** fine-tuned wins ≥3/4 headline checks —
- fewer **unique track IDs** (less ID-switching; base was ~467),
- more **tracks ≥10s**,
- higher **mean persistence**,
- it actually **detects the ball** (base never does).

Then regenerate the demo to *see* it:

```bash
uv run python gaffer/pipeline.py data/test_clips/tactical_playlist_1.mp4 \
    -o outputs/v0_2_finetuned_demo.mp4
```

## Known follow-up

The fine-tuned model introduces `goalkeeper` and `referee` classes the base model
never produced. The tracker already treats `goalkeeper` as trackable and lets
`referee` pass through untracked. **TeamAssigner should be revisited** so referees
are excluded from team colour clustering and goalkeepers (distinct kit) aren't
miscoloured — handle when the A/B surfaces it.
