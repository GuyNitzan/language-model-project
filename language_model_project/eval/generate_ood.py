"""
Phase 5 abstention test set (derma_guide_plan.md, "Three further gifts"
item 1): a small set of Claude-generated questions on topics genuinely
outside Bolognia's Dermatology scope.

This reverts to Revision 1's design after Revision 2's plan (turn
Lever-referenced exam questions into a labelled abstention set) was
dropped per instruction — once Lever references are excluded outright
(not repurposed), there is no in-domain "known unanswerable" bucket left in
the real exam data, so this is the only source of abstention ground truth.

Correct behaviour on every one of these is refusal — the retriever will
still return SOME chunks (top-k retrieval always returns k results, however
irrelevant), so this tests whether the generator recognises irrelevant
context and says so, not whether retrieval returns nothing.

One direct (non-batch) Claude call — 24 questions is small enough that
there's no reason to round-trip through the Batch API — with extended
thinking explicitly disabled (see translate_heldout_mcq.py's comment: this
model sometimes burns the whole max_tokens budget on an invisible thinking
block otherwise).

Usage:
    /home/guynitz/venvs/derma_eval/bin/python3 generate_ood.py

Output: eval/data/ood_questions.json — [{"question": "...", "domain": "..."}]
"""

import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

EVAL_DIR = Path(__file__).parent
OUT_PATH = EVAL_DIR / "data" / "ood_questions.json"

MODEL = "claude-sonnet-5"
N_QUESTIONS = 24

SYSTEM_PROMPT = f"""Generate exactly {N_QUESTIONS} short factual questions that are CLEARLY outside \
the scope of a dermatology textbook (Bolognia's Dermatology). Spread them across clearly distant \
domains so there's no ambiguity: cardiology, orthopedic surgery, pediatric endocrinology (non-skin), \
nephrology, general nutrition science, astronomy, and world history. None should touch skin, hair, \
nails, or anything a dermatology textbook might plausibly discuss (e.g. avoid "skin manifestations \
of X" framing even for a non-dermatology disease). Each question should be answerable by someone \
knowledgeable in its field but have NO correct answer derivable from a dermatology reference.

Respond with ONLY a JSON array, no other text, no markdown fences:
[{{"question": "...", "domain": "..."}}, ...]"""


def parse_response(text: str) -> list[dict] | None:
    m = re.search(r"\[.*\]", text.strip(), re.DOTALL)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(arr, list) or not all("question" in q for q in arr):
        return None
    return arr


def main():
    load_dotenv(EVAL_DIR.parent / "backend" / ".env")
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    resp = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        thinking={"type": "disabled"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Generate the {N_QUESTIONS} questions now."}],
    )
    text = next((b.text for b in resp.content if b.type == "text"), "")
    parsed = parse_response(text)
    if parsed is None:
        sys.exit(f"Could not parse response as a JSON array:\n{text}")

    print(f"Generated {len(parsed)} out-of-corpus questions:")
    for q in parsed:
        print(f"  [{q.get('domain', '?')}] {q['question']}")

    OUT_PATH.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWritten -> {OUT_PATH}")


if __name__ == "__main__":
    main()
