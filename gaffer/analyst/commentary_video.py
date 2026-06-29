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

import re
import shutil
import subprocess
import wave
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pyttsx3

from gaffer import config
from gaffer.analyst.commentary import commentate_match, importance_score
from gaffer.analyst.match_bundle import build_bundle
from gaffer.calibration.homography_manager import HomographyManager
from gaffer.calibration.homography_propagator import HomographyPropagator
from gaffer.detection.detector import FootballDetector
from gaffer.detection.team_assigner import TeamAssigner
from gaffer.output.minimap import MinimapRenderer
from gaffer.video.loader import VideoLoader
from gaffer.video.writer import VideoWriter

_TTS_RATE = 165          # words-per-minute; SAPI default ~200 is too fast for commentary
_GAP_S = 0.3             # baseline silence between back-to-back narrations
_FONT = cv2.FONT_HERSHEY_SIMPLEX


# ── Text-to-speech ────────────────────────────────────────────────────────────

def _tts(text: str, out_wav: Path, rate: int = _TTS_RATE) -> float:
    """Render `text` to a wav with offline SAPI5 and return its duration.
    A fresh engine per call -- pyttsx3 reuse-across-runAndWait only reliably
    writes the first file on Windows."""
    eng = pyttsx3.init()
    eng.setProperty("rate", rate)
    eng.save_to_file(text, str(out_wav))
    eng.runAndWait()
    eng.stop()
    with wave.open(str(out_wav), "rb") as w:
        return w.getnframes() / float(w.getframerate())


_PAUSE_S = 0.25  # real silence spliced between sentences -- see _tts_with_pauses


def _rate_for_importance(importance: float) -> int:
    """Faster, more energetic delivery for high-importance episodes, calmer
    for routine ones -- pyttsx3/SAPI5 has no working pause/prosody markup
    (verified directly: embedding a <silence msec="..."/>-style tag produces
    ZERO actual silent gap in the output's energy envelope -- the engine
    reads the tag's literal characters aloud instead of executing it), but
    `rate` via setProperty is a real, already-working lever."""
    return int(150 + min(importance, 8.0) * 7.5)  # ~154 (quiet) .. 210 (big moment)


def _tts_with_pauses(text: str, out_wav: Path, rate: int) -> float:
    """Same contract as _tts() but splices a real silence buffer between
    sentences instead of relying on markup that doesn't work with this
    engine (see _rate_for_importance) -- synthesizes each sentence
    separately and concatenates the raw PCM with actual zero-amplitude
    frames in between, so the pause is real audio, not a hopeful tag."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if len(sentences) <= 1:
        return _tts(text, out_wav, rate)

    frames_list: list[bytes] = []
    params = None
    for i, sent in enumerate(sentences):
        part = out_wav.parent / f"{out_wav.stem}_part{i}.wav"
        _tts(sent, part, rate)
        with wave.open(str(part), "rb") as w:
            params = params or w.getparams()
            frames_list.append(w.readframes(w.getnframes()))
        part.unlink()

    # Byte count must be a whole multiple of the sample frame size (sampwidth
    # * nchannels) or every sample after the splice point shifts by a byte --
    # silent in isolation, but corrupts the PCM stream once ffmpeg muxes it
    # (confirmed: this exact bug produced a real "Invalid PCM packet" decode
    # warning the first time this ran for real). Round the FRAME count, not
    # the byte count, then convert.
    n_silence_frames = int(params.framerate * _PAUSE_S)
    silence = b"\x00" * (n_silence_frames * params.sampwidth * params.nchannels)
    combined = silence.join(frames_list)
    with wave.open(str(out_wav), "wb") as out:
        out.setparams(params)
        out.writeframes(combined)
    return len(combined) / (params.framerate * params.sampwidth * params.nchannels)


# ── Subtitle drawing ──────────────────────────────────────────────────────────

# cv2.putText's Hershey fonts are ASCII-only -- qwen's output routinely
# includes curly quotes/dashes, which render as "???" boxes. Map the common
# ones to plain ASCII before drawing; nothing else (anything not listed
# passes through, accepting the rare odd glyph rather than aggressively
# stripping characters that might still render fine).
_ASCII_FOLD = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...",
})


def _draw_subtitle(frame: np.ndarray, text: str, *, scale: float = 0.7, thick: int = 2) -> None:
    """Burn word-wrapped commentary into a translucent band at the bottom."""
    text = text.translate(_ASCII_FOLD)
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


# ── Calibration confidence ────────────────────────────────────────────────────
#
# The propagator's own update() already knows when tracking has degraded
# (`held`) or a cut just happened (`scene_cut`) -- today's render loop just
# silently keeps the old H either way and never tells anyone. A second,
# independent signal catches what the propagator's internal flow-tracking
# CAN'T see: slow drift that never trips a cut or a hold, only visible by
# checking whether the resulting projection is even plausible (this is
# exactly the diagnostic that caught the original arsenal 60s bug -- `held`
# and `scene_cut` were both False there the whole time). Surfacing both as a
# state, rather than rendering every frame with the same visual confidence,
# is the fix: don't pretend to know the geometry when the evidence says
# otherwise.

_UNCERTAIN_RATIO = 0.65   # on-pitch fraction below this (with enough players to judge) -> at least uncertain
_LOST_RATIO = 0.35        # below this -> lost
_MIN_PLAYERS_TO_JUDGE = 4 # fewer detections than this and the ratio is too noisy to trust


def _on_pitch_ratio(mgr: HomographyManager, dets) -> tuple[int, float] | None:
    """(n_player_dets, on_pitch_fraction), or None if too few player-class
    detections this frame to judge plausibility from the ratio alone."""
    players = [d for d in dets if d.class_name != "ball"]
    if len(players) < _MIN_PLAYERS_TO_JUDGE:
        return None
    n_on = sum(1 for d in players if (w := mgr.project(d.foot_point)) is not None and mgr.on_pitch(*w))
    return len(players), n_on / len(players)


def _confidence_state(just_anchored: bool, prop_result, mgr: HomographyManager, dets) -> str:
    """CALIBRATED | PROPAGATING | UNCERTAIN | LOST for this frame."""
    if just_anchored:
        return "CALIBRATED"
    if prop_result is not None and prop_result.scene_cut:
        return "LOST"           # propagator itself just detected a camera cut
    ratio_info = _on_pitch_ratio(mgr, dets)
    if ratio_info is not None:
        _, ratio = ratio_info
        if ratio < _LOST_RATIO:
            return "LOST"        # current H projects most players off-pitch -- implausible
        if ratio < _UNCERTAIN_RATIO:
            return "UNCERTAIN"
    if prop_result is not None and prop_result.held:
        return "UNCERTAIN"      # flow tracking degraded this frame (old H kept, not yet disproven)
    return "PROPAGATING"


def _scan_confidence_runs(clip_path: Path, calib_path: Path,
                          duration: float | None = None) -> list[tuple[str, float, float]]:
    """[(state, start_s, end_s), ...] for the whole clip, computed BEFORE
    the real render loop so commentary generation (which has to happen
    before that loop, to know narration timings for the subtitle burn-in)
    can know which episodes overlap a LOST stretch. Deliberately cheap: no
    object detection, no minimap, no video output -- just the propagator's
    own scene_cut/held signals, which is why this can run as a quick extra
    pass instead of doubling render time by running detection twice.

    Honest tradeoff: without real detections this can't run the on-pitch-
    ratio check the full render loop also does, so it won't catch slow
    undetected drift (only cuts and flow-tracking failures) -- the visual
    minimap banner in the main loop still catches that case fully. A
    second full detection pass just to make the spoken caveat as sharp as
    the visual one isn't worth doubling render time for."""
    mgr = HomographyManager.from_calibration(calib_path)
    loader = VideoLoader(str(clip_path))
    fps = loader.fps
    n_total = loader.total_frames if duration is None else min(int(duration * fps), loader.total_frames)
    propagator = HomographyPropagator(mgr)
    anchors = mgr.anchors
    anchor_idx = 0

    runs: list[tuple[str, float, float]] = []
    cur_state, cur_start = None, 0.0
    for fidx, frame in loader.frames(start=0, count=n_total):
        just_anchored = False
        result = None
        if fidx < anchors[0][0]:
            just_anchored = True
        else:
            while anchor_idx + 1 < len(anchors) and fidx >= anchors[anchor_idx + 1][0]:
                anchor_idx += 1
                propagator.reset()
                just_anchored = True
            if fidx == anchors[anchor_idx][0]:
                propagator.reset()
                just_anchored = True
            result = propagator.update(frame, exclude_dets=None)

        if just_anchored:
            state = "CALIBRATED"
        elif result is not None and result.scene_cut:
            state = "LOST"
        elif result is not None and result.held:
            state = "UNCERTAIN"
        else:
            state = "PROPAGATING"

        t = fidx / fps
        if state != cur_state:
            if cur_state is not None:
                runs.append((cur_state, cur_start, t))
            cur_state, cur_start = state, t
    if cur_state is not None:
        runs.append((cur_state, cur_start, n_total / fps))
    loader.close()
    return runs


def _overlaps_lost(ep_start: float, ep_end: float, runs: list[tuple[str, float, float]]) -> bool:
    return any(state == "LOST" and ep_start < end and ep_end > start for state, start, end in runs)


# ── Orchestration ─────────────────────────────────────────────────────────────

def _fit_assigner(detector: FootballDetector, loader: VideoLoader, start: int, count: int) -> TeamAssigner:
    assigner = TeamAssigner()
    frames = loader.sample_frames(12, start=start, count=count)
    assigner.fit(frames, [detector.detect_raw(f) for f in frames])
    return assigner


def _fit_assigner_near_calibration(detector: FootballDetector, loader: VideoLoader,
                                   calib_frame: int, n_total: int) -> TeamAssigner:
    """Fit team-color clustering on a window around the calibration frame, not
    the whole clip -- same window render_minimap.py already uses (+-5s). A
    highlight clip can cut to a celebration close-up or a replay with no clean
    two-team structure; sampling evenly across the WHOLE clip duration (as a
    long continuous match recording allows) pulls jersey-color features from
    those cutaways too and corrupts the K-means fit for every frame. The
    calibration frame is guaranteed to sit inside the real tactical shot, so a
    window around it can't land in a cutaway."""
    win = int(5 * loader.fps)
    start = max(0, min(calib_frame - win, n_total - 1))
    end = min(n_total, calib_frame + win)
    return _fit_assigner(detector, loader, start, max(1, end - start))


def build_commentary_video(clip_path, calib_path, *, use_llm: bool = True,
                           notable_only: bool = True, min_importance: float = 0.0,
                           style: str = "broadcast", duration: float | None = None,
                           out_path: Path | None = None, log=print) -> Path:
    """Render <clip> as one mp4 with bird's-eye minimap, timed commentary
    subtitles, and a spoken narration track. Returns the output path."""
    clip_path, calib_path = Path(clip_path), Path(calib_path)
    bundle = build_bundle(clip_path, calib_path)

    work = config.OUTPUTS_DIR / "commentary_video" / clip_path.stem
    work.mkdir(parents=True, exist_ok=True)

    log("Scanning tracking confidence...")
    confidence_runs = _scan_confidence_runs(clip_path, calib_path, duration=duration)

    # 1. commentary + TTS, scheduled so narrations never overlap. Gap between
    # back-to-back narrations grows with how much has already been said in
    # the preceding window -- "let the match breathe" rather than a flat gap.
    log("Generating commentary + narration...")
    segments: list[tuple[float, float, str]] = []   # (audio_start, audio_end, text)
    audio_segs: list[tuple[Path, float]] = []
    cursor = 0.0
    for ep, text in commentate_match(bundle, use_llm=use_llm, notable_only=notable_only,
                                     min_importance=min_importance, style=style):
        if duration is not None and ep.start_time_s >= duration:
            break
        if _overlaps_lost(ep.start_time_s, ep.end_time_s, confidence_runs):
            text = "Tracking was uncertain through this passage. " + text
        wav = work / f"seg_{ep.episode_id}.wav"
        rate = _rate_for_importance(importance_score(ep))
        dur = _tts_with_pauses(text, wav, rate)
        recent_talk = sum(min(e, cursor) - s for s, e, _ in segments
                          if e > cursor - 20.0 and s < cursor)
        gap = _GAP_S + min(recent_talk / 20.0, 1.0) * 1.5
        start = max(ep.start_time_s, cursor)
        cursor = start + dur + gap
        segments.append((start, start + dur, text))
        audio_segs.append((wav, start))
    log(f"  {len(segments)} narrated episodes")

    # 2. render the base (silent) video: footage + bird's-eye minimap + subtitle
    loader = VideoLoader(str(clip_path))
    fps = loader.fps
    n_total = loader.total_frames if duration is None else min(int(duration * fps), loader.total_frames)

    detector = FootballDetector(verbose=False)
    mgr = HomographyManager.from_calibration(calib_path)
    assigner = _fit_assigner_near_calibration(detector, loader, mgr.calibration_frame or 0, n_total)
    minimap = MinimapRenderer(mgr)
    propagator = HomographyPropagator(mgr)

    # Size the minimap to the frame, not a fixed pixel width -- the default
    # 300px is a small inset on a 1280px clip but eats half a 640px one.
    mini_w = max(160, int(loader.width * 0.24))

    # Anchor the homography at the NEAREST calibrated frame, not just the
    # first one. Propagating from a single anchor across a whole multi-shot
    # clip accumulates drift with no way to correct itself (confirmed: correct
    # near the anchor, visibly wrong tens of seconds away on cut footage) --
    # the propagator's own docstring says as much ("good for ONE shot, no
    # global re-anchoring"). With multiple anchors (one per distinct camera
    # shot, see scripts/collect_calibration.py --append), propagation never
    # has to travel farther than the gap between two anchors -- the same
    # short-range regime already proven correct.
    anchors = mgr.anchors
    anchor_idx = 0   # index into anchors of the currently-active one

    base = work / "base.mp4"
    state_counts = {"CALIBRATED": 0, "PROPAGATING": 0, "UNCERTAIN": 0, "LOST": 0}
    log(f"Rendering {n_total} frames with minimap + subtitles ({len(anchors)} anchor(s))...")
    with VideoWriter(base, fps=fps, width=loader.width, height=loader.height) as writer:
        for fidx, frame in loader.frames(start=0, count=n_total):
            dets = assigner.assign(frame, detector.detect(frame, fidx))

            just_anchored = False
            result = None
            if fidx < anchors[0][0]:
                mgr.H = anchors[0][1].copy()                 # before the first anchor: hold it
                just_anchored = True
            else:
                while anchor_idx + 1 < len(anchors) and fidx >= anchors[anchor_idx + 1][0]:
                    anchor_idx += 1
                    mgr.H = anchors[anchor_idx][1].copy()     # advance to the next anchor
                    propagator.reset()
                    just_anchored = True
                if fidx == anchors[anchor_idx][0]:
                    mgr.H = anchors[anchor_idx][1].copy()     # land exactly on it -- re-anchor precisely
                    propagator.reset()
                    just_anchored = True
                result = propagator.update(frame, exclude_dets=dets)   # seeds (just reset) or propagates forward

            state = _confidence_state(just_anchored, result, mgr, dets)
            state_counts[state] += 1

            # LOST: don't draw player positions we have positive reason to
            # distrust -- an empty pitch + banner is honest, wrong dots aren't.
            minimap_dets = [] if state == "LOST" else dets
            banner = {"UNCERTAIN": "TRACKING UNCERTAIN", "LOST": "TRACKING LOST"}.get(state)
            banner_color = {"UNCERTAIN": (0, 210, 230), "LOST": (60, 60, 230)}.get(state, (60, 220, 60))
            out = minimap.composite(frame, minimap_dets, width=mini_w, corner="top_right",
                                    label=banner, label_color=banner_color)
            t = fidx / fps
            active = next((txt for (s, e, txt) in segments if s <= t < e), None)
            if active:
                _draw_subtitle(out, active)
            writer.write(out)
            if fidx and fidx % 250 == 0:
                log(f"  {fidx}/{n_total} frames")
    log(f"  confidence: {state_counts}")
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
