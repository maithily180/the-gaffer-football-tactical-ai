"""
gaffer/analyst/answer_generator.py
─────────────────────────────────────
Step 4: the only part of the query engine that touches an LLM. Reads a
small evidence pack -- never raw events, tracking data, or video -- and
asks qwen2.5:3b (already pulled locally via Ollama, see scripts/setup_ollama.py)
to phrase it as prose. Each prompt template tells the model to use ONLY the
listed facts and to say so plainly when the facts show something never
happened, rather than speculating.

If retrieval already came back empty, there is exactly one correct
sentence to write and no reason to spend an LLM call manufacturing it --
generate_answer() returns a canned sentence instead and skips ollama.chat()
entirely.
"""

from __future__ import annotations

from pathlib import Path

import ollama

from gaffer.analyst.evidence import EvidencePack
from gaffer.analyst.question_types import QuestionType

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_MODEL = "qwen2.5:3b"
# The local OLLAMA_HOST env var is set to 0.0.0.0 for the server to bind all
# interfaces; the Python client otherwise reads the same var as its connect
# target, which 0.0.0.0 isn't valid for. Connect to the loopback explicitly.
_CLIENT = ollama.Client(host="http://127.0.0.1:11434")

_PROMPT_FILES = {
    QuestionType.MATCH_SUMMARY:         "match_summary.txt",
    QuestionType.DOMINANCE_EXPLANATION: "dominance_explanation.txt",
    QuestionType.EVENT_SEARCH:          "event_search.txt",
    QuestionType.PLAYER_INFLUENCE:      "player_influence.txt",
    QuestionType.PASSING_ANALYSIS:      "passing_analysis.txt",
    QuestionType.TIME_WINDOW:           "time_window.txt",
}


def generate_answer(question: str, qtype: QuestionType, pack: EvidencePack) -> str:
    if pack.empty:
        return f"No evidence found for this question -- {pack.empty_reason}."

    template = (_PROMPTS_DIR / _PROMPT_FILES[qtype]).read_text(encoding="utf-8")
    prompt = template.format(question=question, evidence=pack.render_for_prompt())
    response = _CLIENT.chat(model=_MODEL, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip()
