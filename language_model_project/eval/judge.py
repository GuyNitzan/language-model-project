"""
Phase 5 LLM-as-judge (derma_guide_plan.md's "ROUGE-L / BLEU / citation
accuracy / LLM-judge groundedness on the synthetic open-ended slice").

ROUGE-L/BLEU catch lexical overlap but miss a common failure mode here: an
answer can be clinically correct and well-grounded while using different
wording than the reference (or vice versa — high n-gram overlap with a
subtly wrong claim). A single Claude judge call scores what those metrics
can't: factual correctness against the reference answer, and groundedness
against the retrieved context — independent of phrasing.

Deliberately a DIFFERENT check than backend/verifier.py: verifier.py checks
"does the cited excerpt actually support this claim" (citation-level,
per-claim). This judges "is the answer as a whole correct and useful
compared to the reference" (answer-level, holistic) — the two together are
what the plan's Analysis section needs (citation accuracy vs. overall
answer quality are genuinely different questions).
"""

from __future__ import annotations

import json
import os
import re

from dotenv import load_dotenv
from pathlib import Path

EVAL_DIR = Path(__file__).parent
JUDGE_MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """You are grading a dermatology RAG system's answer against a reference answer,
written by an expert model with access to the same source material. Score the CANDIDATE answer on:

- "correctness" (1-5): does it convey the same key facts as the reference, with no invented or
  contradictory claims? 5 = fully correct and complete, 1 = wrong or contradicts the reference.
- "groundedness" (1-5): are the candidate's claims plausibly supported by the RETRIEVED EXCERPTS
  shown (not outside knowledge)? 5 = every claim traces to the excerpts, 1 = unsupported/fabricated.
  If the candidate explicitly abstains (says the excerpts are insufficient) AND the excerpts genuinely
  don't support the reference answer, treat that as groundedness=5 (correct behaviour), not a penalty.

Respond with ONLY a single JSON object, no other text, no markdown fences:
{"correctness": 1-5, "groundedness": 1-5, "reasoning": "one sentence"}"""


def _client():
    load_dotenv(EVAL_DIR.parent / "backend" / ".env")
    import anthropic
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def _parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text.strip(), re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if "correctness" not in obj or "groundedness" not in obj:
        return None
    return obj


_CLIENT = None


def judge(question: str, reference_answer: str, candidate_answer: str, chunks: list[dict],
          model: str = JUDGE_MODEL) -> dict:
    """Returns {"correctness": 1-5, "groundedness": 1-5, "reasoning": str}
    or {"error": str} if the judge call/parse failed (never silently
    scored — a failed judge call must not look like a real low score)."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = _client()

    excerpts = "\n\n---\n\n".join(
        f"[Excerpt {i}] {c['text']}" for i, c in enumerate(chunks, 1)
    ) if chunks else "(no excerpts retrieved)"

    user_message = (
        f"Question: {question}\n\n"
        f"Retrieved excerpts:\n{excerpts}\n\n"
        f"Reference answer: {reference_answer}\n\n"
        f"Candidate answer to grade: {candidate_answer}"
    )
    try:
        resp = _CLIENT.messages.create(
            model=model, max_tokens=400,
            thinking={"type": "disabled"},  # see translate_heldout_mcq.py — avoids the thinking-budget bug
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
    except Exception as e:  # noqa: BLE001
        return {"error": f"judge API call failed: {e}"}

    text = next((b.text for b in resp.content if b.type == "text"), "")
    parsed = _parse(text)
    if parsed is None:
        return {"error": "judge response was not parseable JSON"}
    return parsed
