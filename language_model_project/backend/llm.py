"""
Thin backward-compatibility wrapper around generators.py (Phase 5, decision
#4 in derma_guide_plan.md). main.py still calls `llm.answer(...)`; this
routes to ClaudeGenerator, which reproduces this module's pre-Phase-5
behaviour exactly. New code (eval scripts, the pluggable-generator demo)
should import generators.py directly instead.
"""

from generators import ClaudeGenerator

_default = ClaudeGenerator()


def answer(question: str, chunks: list[dict], history: list[dict]) -> str:
    return _default.answer(question, chunks, history)
