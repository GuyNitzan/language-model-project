"""
Phase 5: structured English translation of the held-out 2025+2026 exam MCQs
(text_answerable + image_dependent buckets — 231 + 55 = 286 questions).

translate_queries.py (Phase 3) already translates the *develop*-years
text_answerable set, but as one merged "stem + A: ... B: ... " string — fine
for a retrieval query, useless for MCQ likelihood scoring, which needs the
stem and each option translated as SEPARATE strings (eval_mcq.py scores each
option's own text as a continuation of the stem). 2025/2026 were never
translated at all (Phase 3 held them out of the query-language experiment on
purpose). Hence a second, small, synchronous (not Batch — 286 calls finishes
in a few minutes, not worth the async round-trip) translator here.

image_dependent questions are included because Phase 5 reports MCQ accuracy
on that bucket too (the measured ceiling) — same translation need.

Usage:
    /home/guynitz/venvs/derma_eval/bin/python3 translate_heldout_mcq.py [--limit N]

Output: eval/data/exam_heldout_translations.json
    {"<year>::<q_num>": {"stem_en": "...", "options_en": {"א": "...", ...}}}
Safe to re-run: skips questions already present in the output file.
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

EVAL_DIR = Path(__file__).parent
ALL_PATH = EVAL_DIR / "data" / "exam_parsed" / "all.jsonl"
OUT_PATH = EVAL_DIR / "data" / "exam_heldout_translations.json"

HELD_OUT_YEARS = {"2025", "2026"}
BUCKETS = {"text_answerable", "image_dependent"}

MODEL = "claude-sonnet-5"
MAX_TOKENS = 700
WORKERS = 6  # small, synchronous fan-out — polite to the API, fast enough for 286 items

SYSTEM_PROMPT = """You translate a single Hebrew dermatology board-exam MCQ into English for a \
model-evaluation pipeline. You will get a JSON object with a "stem" and four "options" (א/ב/ג/ד; \
an option may be an empty string or garbled OCR noise — translate garbage as an empty string, \
don't invent content). Translate the medically meaningful content into natural, precise English. \
Keep each option translation self-contained (a full clinical statement), since it will be scored \
independently of the others — do not write "same as above" or similar shorthand.

Respond with ONLY a single JSON object, no other text, no markdown fences:
{"stem_en": "...", "options_en": {"א": "...", "ב": "...", "ג": "...", "ד": "..."}}"""


def load_heldout() -> list[dict]:
    recs = []
    with ALL_PATH.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["year"] in HELD_OUT_YEARS and r["bucket"] in BUCKETS:
                recs.append(r)
    return recs


def source_json(record: dict) -> str:
    return json.dumps({"stem": record["stem"], "options": record["options"]}, ensure_ascii=False)


def parse_response(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text.strip(), re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if "stem_en" not in obj or "options_en" not in obj:
        return None
    if not all(letter in obj["options_en"] for letter in "אבגד"):
        return None
    return obj


def get_client():
    load_dotenv(EVAL_DIR.parent / "backend" / ".env")
    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        sys.exit("ANTHROPIC_API_KEY not found (checked backend/.env).")
    return anthropic.Anthropic(api_key=key)


def translate_one(client, record: dict, retries: int = 2) -> dict | None:
    for attempt in range(retries + 1):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                # Without this, claude-sonnet-5 sometimes burns the entire
                # max_tokens budget on an invisible extended-thinking block
                # before emitting any text — confirmed directly: stop_reason
                # "max_tokens" with a ThinkingBlock consuming 699/700 tokens
                # and zero text output. Not needed for a translation task.
                thinking={"type": "disabled"},
                messages=[{"role": "user", "content": source_json(record)}],
            )
            text = next((b.text for b in resp.content if b.type == "text"), "")
            parsed = parse_response(text)
            if parsed is not None:
                return parsed
        except Exception as e:  # noqa: BLE001 — retry on any transient API error
            if attempt == retries:
                print(f"  {record['year']} Q{record['q_num']}: FAILED after retries ({e})")
                return None
            time.sleep(2 * (attempt + 1))
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="translate only the first N (smoke test)")
    args = ap.parse_args()

    heldout = load_heldout()
    if args.limit:
        heldout = heldout[: args.limit]
    print(f"{len(heldout)} held-out (2025+2026, text_answerable+image_dependent) questions")

    existing = json.loads(OUT_PATH.read_text(encoding="utf-8")) if OUT_PATH.exists() else {}
    todo = [r for r in heldout if f"{r['year']}::{r['q_num']}" not in existing]
    print(f"{len(existing)} already translated, {len(todo)} to do")
    if not todo:
        return

    client = get_client()
    n_ok = n_fail = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(translate_one, client, r): r for r in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            r = futures[fut]
            key = f"{r['year']}::{r['q_num']}"
            parsed = fut.result()
            if parsed is None:
                n_fail += 1
            else:
                existing[key] = parsed
                n_ok += 1
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} done ({n_ok} ok, {n_fail} failed)")
                OUT_PATH.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")

    OUT_PATH.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(existing)} total translations → {OUT_PATH}")
    print(f"This run: {n_ok} ok, {n_fail} failed out of {len(todo)}")


if __name__ == "__main__":
    main()
