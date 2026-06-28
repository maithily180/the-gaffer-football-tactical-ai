"""
gaffer/analyst/commentary_video.py
─────────────────────────────────────
v3.2 — Commentary Video: one downloadable mp4 of a match with the bird's-eye
minimap composited in, the episode commentary burned in as timed subtitles,
and a spoken narration track.

Pulls together pieces that already exist rather than inventing new ones:
  - commentary.commentate_match()      -> qwen-grounded text per episode
  - MinimapRenderer.composite()        -> the bird's-eye inset (same path
                                          render_minimap.py uses)
  - pyttsx3 (offline SAPI5)            -> narration wav per episode
  - imageio-ffmpeg's bundled ffmpeg    -> mux the narration onto the video at
                                          the right timestamps (cv2 can't do
                                          audio)

The render needs a perception pass (the minimap needs per-frame positions),
so it is minutes-long for a full clip -- a `duration` cap is provided for
quick tests.
"""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pyttsx3

from gaffer import config
from gaffer.analyst.commentary import commentate_match
from gaffer.analyst.match_bundle import build_bundle
from gaffer.calibration.homography_manager import HomographyManager
from gaffer.calibration.homography_propagator import HomographyPropagator
from gaffer.detection.detector import FootballDetector
from gaffer.detection.team_assigner import TeamAssigner
from gaffer.output.minimap import MinimapRenderer
from gaffer.video.loader import VideoLoader
from gaffer.video.writer import VideoWriter

_TTS_RATE = 165          # words-per-minute; SAPI default ~200 is too fast for commentary
_GAP_S = 0.3             # silence between back-to-back narrations
_FONT = cv2.FONT_HERSHEY_SIMPLEX


# ── Text-to-speech ────────────────────────────────────────────────────────────

def _tts(text: str, out_wav: Path) -> float:
    """Render `text` to a wav with offline SAPI5 and return its duration.
    A fresh engine per call -- pyttsx3 reuse-across-runAndWait only reliably
    writes the first file on Windows."""
    eng = pyttsx3.init()
    eng.setProperty("rate", _TTS_RATE)
    eng.save_to_file(text, str(out_wav))
    eng.runAndWait()
    eng.stop()
    with wave.open(str(out_wav), "rb") as w:
        return w.getnframes() / float(w.getframerate())


# ── Subtitle drawing ──────────────────────────────────────────────────────────

def _draw_subtitle(frame: np.ndarray, text: str, *, scale: float = 0.7, thick: int = 2) -> None:
    """Burn word-wrapped commentary into a translucent band at the bottom."""
    H, W = frame.shape[:2]
    max_w = int(W * 0.9)
    words, lines, cur = text.split(), [], ""
    for word in words:
        trial = (cur + " " + word).strip()
        (tw, _), _ = cv2.getTextSize(trial, _FONT, scale, thick)
        if tw > max_w and cur:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)

    line_h = int(cv2.getTextSize("Ag", _FONT, scale, thick)[0][1] * 1.9)
    box_h = line_h * len(lines) + 14
    y0 = H - box_h - 16
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (W, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    y = y0 + line_h
    for ln in lines:
        (tw, _), _ = cv2.getTextSize(ln, _FONT, scale, thick)
        cv2.putText(frame, ln, ((W - tw) // 2, y), _FONT, scale, (255, 255, 255), thick, cv2.LINE_AA)
        y += line_h


# ── Audio muxing (ffmpeg) ─────────────────────────────────────────────────────

def _mux_audio(video_path: Path, audio_segs: list[tuple[Path, float]], out_path: Path,
               video_duration_s: float) -> Path:
    """Overlay each narration wav onto the (silent) video at its start time,
    via the bundled ffmpeg -- adelay shifts each clip to its timestamp, amix
    blends them, -c:v copy keeps the already-encoded H.264 untouched. Output
    is capped to the video length (`-t`) so a narration that runs past the
    final whistle doesn't stretch the file with a frozen tail."""
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    inputs: list[str] = ["-i", str(video_path)]
    filters: list[str] = []
    labels: list[str] = []
    for i, (wav, start_s) in enumerate(audio_segs):
        inputs += ["-i", str(wav)]
        ms = max(0, int(start_s * 1000))
        filters.append(f"[{i + 1}:a]adelay={ms}|{ms}[a{i}]")
        labels.append(f"[a{i}]")
    fc = ";".join(filters) + ";" + "".join(labels) + f"amix=inputs={len(audio_segs)}:normalize=0[aout]"
    cmd = [ff, "-y", "-v", "error", *inputs, "-filter_complex", fc,
           "-map", "0:v", "-map", "[aout]", "-t", f"{video_duration_s:.3f}",
           "-c:v", "copy", "-c:a", "aac", "-movflags", "+faststart", str(out_path)]
    subprocess.run(cmd, check=True)
    return out_path


# ── Orchestration ─────────────────────────────────────────────────────────────

def _fit_assigner(detector: FootballDetector, loader: VideoLoader, start: int, count: int) -> TeamAssigner:
    assigner = TeamAssigner()
    frames = loader.sample_frames(12, start=start, count=count)
    assigner.fit(frames, [detector.detect_raw(f) for f in frames])
    return assigner


def build_commentary_video(clip_path, calib_path, *, use_llm: bool = True,
                           notable_only: bool = True, duration: float | None = None,
                           out_path: Path | None = None, log=print) -> Path:
    """Render <clip> as one mp4 with bird's-eye minimap, timed commentary
    subtitles, and a spoken narration track. Returns the output path."""
    clip_path, calib_path = Path(clip_path), Path(calib_path)
    bundle = build_bundle(clip_path, calib_path)

    work = config.OUTPUTS_DIR / "commentary_video" / clip_path.stem
    work.mkdir(parents=True, exist_ok=True)

    # 1. commentary + TTS, scheduled so narrations never overlap
    log("Generating commentary + narration...")
    segments: list[tuple[float, float, str]] = []   # (audio_start, audio_end, text)
    audio_segs: list[tuple[Path, float]] = []
    cursor = 0.0
    for ep, text in commentate_match(bundle, use_llm=use_llm, notable_only=notable_only):
        if duration is not None and ep.start_time_s >= duration:
            break
        wav = work / f"seg_{ep.episode_id}.wav"
        dur = _tts(text, wav)
        start = max(ep.start_time_s, cursor)
        cursor = start + dur + _GAP_S
        segments.append((start, start + dur, text))
        audio_segs.append((wav, start))
    log(f"  {len(segments)} narrated episodes")

    # 2. render the base (silent) video: footage + bird's-eye minimap + subtitle
    loader = VideoLoader(str(clip_path))
    fps = loader.fps
    n_total = loader.total_frames if duration is None else min(int(duration * fps), loader.total_frames)

    detector = FootballDetector(verbose=False)
    assigner = _fit_assigner(detector, loader, 0, n_total)
    mgr = HomographyManager.from_calibration(calib_path)
    minimap = MinimapRenderer(mgr)
    propagator = HomographyPropagator(mgr)

    # Size the minimap to the frame, not a fixed pixel width -- the default
    # 300px is a small inset on a 1280px clip but eats half a 640px one.
    mini_w = max(160, int(loader.width * 0.24))

    # Anchor the homography at the calibration frame. Propagating naively from
    # frame 0 anchors H to the wrong camera pose, and any early scene cut then
    # corrupts it for the rest of the clip (exactly what broke the bird's-eye on
    # the multi-shot ronaldo clip). Instead: hold the static calibrated H up to
    # the calibration frame, then propagate FORWARD from that correct anchor.
    calib_H = mgr.H.copy() if mgr.H is not None else None
    calib_frame = mgr.calibration_frame or 0
    seeded = False

    base = work / "base.mp4"
    log(f"Rendering {n_total} frames with minimap + subtitles...")
    with VideoWriter(base, fps=fps, width=loader.width, height=loader.height) as writer:
        for fidx, frame in loader.frames(start=0, count=n_total):
            dets = assigner.assign(frame, detector.detect(frame, fidx))
            if calib_H is not None and fidx < calib_frame:
                mgr.H = calib_H.copy()                       # static, pre-calibration
            elif calib_H is not None and not seeded:
                mgr.H = calib_H.copy()                       # re-anchor exactly at calib frame
                propagator.update(frame, exclude_dets=dets)  # seed flow here
                seeded = True
            else:
                propagator.update(frame, exclude_dets=dets)  # propagate forward from anchor
            out = minimap.composite(frame, dets, width=mini_w, corner="top_right")
            t = fidx / fps
            active = next((txt for (s, e, txt) in segments if s <= t < e), None)
            if active:
                _draw_subtitle(out, active)
            writer.write(out)
            if fidx and fidx % 250 == 0:
                log(f"  {fidx}/{n_total} frames")
    loader.close()

    # 3. mux narration onto the video
    out_path = Path(out_path) if out_path else (config.OUTPUTS_DIR / "commentary_video" / f"{clip_path.stem}_commentary.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if audio_segs:
        log("Muxing narration audio...")
        _mux_audio(base, audio_segs, out_path, video_duration_s=n_total / fps)
    else:
        out_path.write_bytes(base.read_bytes())

    # Drop the large silent base render + narration wavs once muxed -- the
    # final mp4 is self-contained and the base is a ~full-size duplicate.
    shutil.rmtree(work, ignore_errors=True)
    log(f"Done: {out_path}")
    return out_path
