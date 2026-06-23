"""
app/gradio_app.py
─────────────────────
Gaffer v3.0 — Match Explorer.

Connects the four previously-disconnected output surfaces named in
docs/V2_PRODUCT_REVIEW.md (demo video, MatchReport text, ask_gaffer.py
prose, find_clip.py/highlight reels) into one interactive app: pick an
already-analyzed match, browse every episode, watch the clip behind any of
them, and ask the analyst free-text questions -- all backed by the exact
same functions the CLI scripts already call. No new detection, analytics,
or LLM call.

Only lists matches with a cached MatchBundle (gaffer/analyst/_cache/) --
building a fresh bundle takes ~10+ minutes and that stays a CLI job, not a
button in this app.

Usage:
    uv run python app/gradio_app.py
"""

from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr

from gaffer import config
from gaffer.analyst import explorer_data as ed
from gaffer.analyst.clip_finder import export_clip
from gaffer.analyst.commentary import commentate_episode
from gaffer.analyst.highlight_reel import render_highlight_reel
from gaffer.analyst.match_bundle import build_bundle
from gaffer.analyst.query_engine import ask

_EXPLORER_DIR = config.OUTPUTS_DIR / "explorer"
_PLAYBACK_DIR = _EXPLORER_DIR / "playback"
_EPISODE_HEADERS = ["ID", "Team", "Start", "End", "Duration", "Outcome", "Narrative"]


def _fresh_url(path) -> str | None:
    """Copy a generated video to a uniquely-named file before handing it to
    gr.Video. clip_finder/highlight_reel name their outputs deterministically
    (same episode -> same filename), so without this the browser re-requests an
    identical URL on every click and serves the *cached* bytes from a previous
    click -- which is exactly how a freshly-fixed video kept showing the old
    broken one ("0:00 / NaN:NaN"). A unique name per click guarantees a fresh
    URL, so the browser can never show a stale copy."""
    if path is None:
        return None
    src = Path(path)
    _PLAYBACK_DIR.mkdir(parents=True, exist_ok=True)
    dst = _PLAYBACK_DIR / f"{src.stem}_{uuid.uuid4().hex[:8]}{src.suffix}"
    shutil.copyfile(src, dst)
    return str(dst)


def _no_match_loaded():
    return None, None, [], "Pick a match and click Load Match.", [], None


def on_load(clip_stem: str | None):
    if not clip_stem:
        return _no_match_loaded()
    clip_path = config.DATA_DIR / "test_clips" / f"{clip_stem}.mp4"
    calib_path = config.DATA_DIR / "calibration" / f"{clip_stem}.json"
    bundle = build_bundle(clip_path, calib_path)
    rows, ids = ed.episode_rows(bundle)
    timeline_path = ed.render_timeline_png(bundle, _EXPLORER_DIR / f"{clip_stem}_timeline.png")
    return bundle, str(clip_path), ids, bundle.match_report.render(), rows, str(timeline_path)


def on_generate_reel(bundle, clip_path: str | None):
    if bundle is None or clip_path is None:
        return None
    return _fresh_url(render_highlight_reel(bundle, clip_path))


def on_select_episode(evt: gr.SelectData, bundle, episode_ids):
    if bundle is None or not episode_ids:
        return None, "Load a match first.", "", "", "", ""
    episode_id = episode_ids[evt.index[0]]
    ep = ed.find_episode(bundle, episode_id)
    tlabel = "Team A" if ep.team == "teamA" else "Team B"
    detail = (f"**Episode #{ep.episode_id} -- {tlabel}**\n\n"
              f"{ep.narrative()}\n\n"
              f"Duration: {ep.duration_s}s | Outcome: {ep.outcome}\n\n"
              f"{ed.explain_outcome(ep)}")
    evidence = "\n".join(f"- {line}" for line in ed.evidence_lines(ep)) or "*(no highlight events recorded on this episode)*"
    preceding = ed.preceding_episode_summary(bundle, episode_id)
    # Deterministic commentary is instant and strictly grounded -- show it
    # inline. The flowing LLM version is opt-in (button below) since it costs
    # an Ollama call and takes phrasing liberties.
    grounded = commentate_episode(ep, use_llm=False)
    return episode_id, detail, evidence, preceding, grounded, ""


def on_flowing_commentary(bundle, episode_id: int | None):
    if bundle is None or episode_id is None:
        return "Select an episode first."
    ep = ed.find_episode(bundle, episode_id)
    return commentate_episode(ep, use_llm=True)


def on_watch_clip(bundle, clip_path: str | None, episode_id: int | None):
    if bundle is None or clip_path is None or episode_id is None:
        return None
    ep = ed.find_episode(bundle, episode_id)
    return _fresh_url(export_clip(clip_path, ep.start_time_s, ep.end_time_s))


def on_ask(bundle, question: str):
    if bundle is None:
        return "Load a match first."
    if not question or not question.strip():
        return "Type a question first."
    return ask(bundle, question)


with gr.Blocks(title="Gaffer -- Match Explorer") as demo:
    gr.Markdown("# Gaffer -- Match Explorer")

    bundle_state = gr.State()
    clip_path_state = gr.State()
    episode_ids_state = gr.State([])
    selected_episode_state = gr.State()

    with gr.Row():
        clip_dd = gr.Dropdown(choices=ed.list_available_clips(), label="Match clip")
        load_btn = gr.Button("Load Match", variant="primary")

    with gr.Tabs():
        with gr.Tab("Overview"):
            report_tb = gr.Textbox(label="Match Report", lines=18, interactive=False)
            timeline_img = gr.Image(label="Timeline", interactive=False)
            reel_btn = gr.Button("Generate Highlight Reel")
            reel_video = gr.Video(label="Highlight Reel")

        with gr.Tab("Episode Explorer"):
            with gr.Row():
                with gr.Column(scale=2):
                    episodes_df = gr.Dataframe(headers=_EPISODE_HEADERS, interactive=False, label="Episodes (click a row)")
                with gr.Column(scale=1):
                    detail_md = gr.Markdown("Select an episode above.")
                    evidence_md = gr.Markdown()
                    gr.Markdown("**Commentary (grounded):**")
                    commentary_md = gr.Markdown()
                    flowing_btn = gr.Button("Flowing commentary (AI)")
                    flowing_md = gr.Markdown()
                    gr.Markdown("**Preceding possession:**")
                    preceding_md = gr.Markdown()
                    watch_btn = gr.Button("Watch Clip")
                    episode_video = gr.Video(label="Episode Clip")

        with gr.Tab("Ask the Analyst"):
            question_tb = gr.Textbox(label="Question", placeholder="e.g. Why did Team A lose control?")
            ask_btn = gr.Button("Ask", variant="primary")
            answer_md = gr.Markdown()

    load_btn.click(
        fn=on_load, inputs=[clip_dd],
        outputs=[bundle_state, clip_path_state, episode_ids_state, report_tb, episodes_df, timeline_img],
    )
    reel_btn.click(fn=on_generate_reel, inputs=[bundle_state, clip_path_state], outputs=[reel_video])
    episodes_df.select(
        fn=on_select_episode, inputs=[bundle_state, episode_ids_state],
        outputs=[selected_episode_state, detail_md, evidence_md, preceding_md, commentary_md, flowing_md],
    )
    flowing_btn.click(fn=on_flowing_commentary, inputs=[bundle_state, selected_episode_state], outputs=[flowing_md])
    watch_btn.click(fn=on_watch_clip, inputs=[bundle_state, clip_path_state, selected_episode_state], outputs=[episode_video])
    ask_btn.click(fn=on_ask, inputs=[bundle_state, question_tb], outputs=[answer_md])


if __name__ == "__main__":
    demo.launch(allowed_paths=[str(config.OUTPUTS_DIR.resolve()), str(config.DATA_DIR.resolve())])
