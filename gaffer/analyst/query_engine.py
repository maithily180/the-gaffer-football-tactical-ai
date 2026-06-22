"""
gaffer/analyst/query_engine.py
─────────────────────────────────
The orchestrator: Intent Detection -> Retrieval -> Structured Evidence ->
LLM Explanation, exactly the four steps implemented in question_types.py,
retrieval.py, evidence.py, and answer_generator.py. ask() just wires them
together in order -- no analysis happens here.
"""

from __future__ import annotations

from gaffer.analyst.answer_generator import generate_answer
from gaffer.analyst.evidence import build_evidence
from gaffer.analyst.match_bundle import MatchBundle
from gaffer.analyst.question_types import classify
from gaffer.analyst.retrieval import retrieve


def ask(bundle: MatchBundle, question: str) -> str:
    qtype, team, event_type = classify(question)
    retrieved = retrieve(bundle, qtype, team=team, event_type=event_type, question=question)
    pack = build_evidence(question, qtype, retrieved)
    return generate_answer(question, qtype, pack)
