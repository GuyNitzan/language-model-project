"""
Phase 3 query-language axis: translate the Hebrew exam questions (develop set
only: 2021, 2022, 2023, 2024_05, 2024_09 — 2025/2026 stay held out per the
plan) into English, plus a short list of key medical terms, via Claude's
Batch API. Caches results so eval_retrieval.py can build all three query
variants (hebrew_raw / english_claude / english_glossary) without
re-translating per (config, embedder) run.

    hebrew_raw        stem + options, untouched Hebrew
    english_claude    Claude's English translation of stem + options
    english_glossary  english_claude + a "Key terms: ..." suffix of the
                       standard English names for the entities Claude
                       identifies in the question (disease/drug/procedure
                       names) — a cheap proxy for query expansion

Two phases (Batch API is async):
    /home/guynitz/venvs/derma_eval/bin/python3 translate_queries.py test
    /home/guynitz/venvs/derma_eval/bin/python3 translate_queries.py submit
    /home/guynitz/venvs/derma_eval/bin/python3 translate_queries.py status
    /home/guynitz/venvs/derma_eval/bin/python3 translate_queries.py collect

Output: eval/data/exam_query_translations.json
    {"<year>::<q_num>": {"english": "...", "terms": ["...", ...]}}
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

EVAL_DIR      = Path(__file__).parent
ALL_PATH      = EVAL_DIR / "data" / "exam_parsed" / "all.jsonl"
OUT_PATH      = EVAL_DIR / "data" / "exam_query_translations.json"
MANIFEST_PATH = EVAL_DIR / "data" / "translate_batch_manifest.json"

DEV_YEARS = {"2021", "2022", "2023", "2024_05", "2024_09"}  # 2025/2026 held out

MODEL = "claude-sonnet-5"
MAX_TOKENS = 700  # Phase 2b's synthetic-QA batch truncated at 400 on ~6% of items; sized up generously here

SYSTEM_PROMPT = """You translate Hebrew dermatology board-exam MCQ questions into English for a \
retrieval-evaluation pipeline. You will get a question stem plus its answer options (some \
options may be garbled OCR noise — ignore fragments that are clearly page-footer artifacts, \
not real answer choices). Translate the medically meaningful content into natural, precise \
English. Then list up to 4 key medical entities named or implied (diseases, drugs, procedures, \
lab tests) using their standard English names.

Respond with ONLY a single JSON object, no other text, no markdown fences:
{"english": "...", "terms": ["...", "..."]}"""


def load_dev_questions() -> list[dict]:
    recs = []
    with ALL_PATH.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["year"] in DEV_YEARS and r["bucket"] == "text_answerable":
                recs.append(r)
    return recs


def source_text(record: dict) -> str:
    parts = [record["stem"].strip()]
    for letter, text in record["options"].items():
        text = text.strip()
        if text:
            parts.append(f"{letter}: {text}")
    return "\n".join(parts)


def build_request(custom_id: str, record: dict) -> dict:
    return {
        "custom_id": custom_id,
        "params": {
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": source_text(record)}],
        },
    }


def extract_text(content_blocks) -> str:
    for block in content_blocks:
        if block.type == "text":
            return block.text
    return ""


def parse_json_response(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text.strip(), re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if "english" not in obj or "terms" not in obj:
        return None
    return obj


def get_client():
    load_dotenv(EVAL_DIR.parent / "backend" / ".env")
    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        sys.exit("ANTHROPIC_API_KEY not found (checked backend/.env).")
    return anthropic.Anthropic(api_key=key)


def cmd_test(args):
    client = get_client()
    dev = load_dev_questions()
    for r in dev[:3]:
        req = build_request("test", r)["params"]
        resp = client.messages.create(**req)
        text = extract_text(resp.content)
        print(f"\n=== {r['year']} Q{r['q_num']} ===")
        print("SOURCE:", source_text(r)[:200])
        print("RESPONSE:", text)
        print("PARSED:", parse_json_response(text))


def cmd_submit(args):
    client = get_client()
    dev = load_dev_questions()
    print(f"{len(dev)} develop-set text_answerable questions to translate")

    manifest_requests = []
    batch_requests = []
    for i, r in enumerate(dev):
        custom_id = f"tr_{i:05d}"
        batch_requests.append(build_request(custom_id, r))
        manifest_requests.append({"custom_id": custom_id, "year": r["year"], "q_num": r["q_num"]})

    batch = client.messages.batches.create(requests=batch_requests)
    print(f"Submitted batch {batch.id} ({len(batch_requests)} requests), status={batch.processing_status}")

    EVAL_DIR.joinpath("data").mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps({"batch_id": batch.id, "requests": manifest_requests}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Manifest written → {MANIFEST_PATH}")


def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        sys.exit(f"No manifest at {MANIFEST_PATH} — run `submit` first.")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def cmd_status(args):
    manifest = _load_manifest()
    client = get_client()
    batch = client.messages.batches.retrieve(manifest["batch_id"])
    print(f"batch {batch.id}: status={batch.processing_status}")
    print(f"  counts: {batch.request_counts}")


def cmd_collect(args):
    manifest = _load_manifest()
    client = get_client()
    batch = client.messages.batches.retrieve(manifest["batch_id"])
    print(f"batch {batch.id}: status={batch.processing_status}, counts={batch.request_counts}")
    if batch.processing_status != "ended":
        print("Not finished yet — re-run this command later.")
        return

    by_custom_id = {r["custom_id"]: r for r in manifest["requests"]}
    out = {}
    n_failed = n_parse_failed = 0
    for result in client.messages.batches.results(batch.id):
        meta = by_custom_id.get(result.custom_id)
        if meta is None:
            continue
        if result.result.type != "succeeded":
            n_failed += 1
            print(f"WARNING: {result.custom_id} ({meta['year']} Q{meta['q_num']}) failed: {result.result.type}")
            continue
        text = extract_text(result.result.message.content)
        parsed = parse_json_response(text)
        if parsed is None:
            n_parse_failed += 1
            print(f"WARNING: {result.custom_id} ({meta['year']} Q{meta['q_num']}) unparseable")
            continue
        out[f"{meta['year']}::{meta['q_num']}"] = parsed

    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(out)} translations written → {OUT_PATH}")
    print(f"{n_failed} failed, {n_parse_failed} unparseable out of {len(by_custom_id)} requested.")


REPAIR_MAX_TOKENS = 1200
REPAIR_MANIFEST_PATH = EVAL_DIR / "data" / "translate_repair_manifest.json"


def cmd_repair_submit(args):
    if not OUT_PATH.exists():
        sys.exit("No collected translations yet — run `collect` first.")
    have = set(json.loads(OUT_PATH.read_text(encoding="utf-8")).keys())
    manifest = _load_manifest()
    missing = [r for r in manifest["requests"] if f"{r['year']}::{r['q_num']}" not in have]
    if not missing:
        print("Nothing to repair.")
        return

    dev_by_key = {f"{r['year']}::{r['q_num']}": r for r in load_dev_questions()}
    client = get_client()
    batch_requests = []
    for meta in missing:
        r = dev_by_key[f"{meta['year']}::{meta['q_num']}"]
        req = build_request(meta["custom_id"], r)
        req["params"]["max_tokens"] = REPAIR_MAX_TOKENS
        batch_requests.append(req)

    batch = client.messages.batches.create(requests=batch_requests)
    print(f"Submitted repair batch {batch.id} ({len(batch_requests)} requests, "
          f"max_tokens={REPAIR_MAX_TOKENS})")
    REPAIR_MANIFEST_PATH.write_text(
        json.dumps({"batch_id": batch.id, "requests": missing}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Repair manifest written → {REPAIR_MANIFEST_PATH}")


def cmd_repair_collect(args):
    if not REPAIR_MANIFEST_PATH.exists():
        sys.exit(f"No repair manifest at {REPAIR_MANIFEST_PATH} — run `repair-submit` first.")
    manifest = json.loads(REPAIR_MANIFEST_PATH.read_text(encoding="utf-8"))
    client = get_client()
    batch = client.messages.batches.retrieve(manifest["batch_id"])
    print(f"repair batch {batch.id}: status={batch.processing_status}, counts={batch.request_counts}")
    if batch.processing_status != "ended":
        print("Not finished yet — re-run this command later.")
        return

    by_custom_id = {r["custom_id"]: r for r in manifest["requests"]}
    out = json.loads(OUT_PATH.read_text(encoding="utf-8")) if OUT_PATH.exists() else {}
    n_failed = n_parse_failed = 0
    for result in client.messages.batches.results(batch.id):
        meta = by_custom_id.get(result.custom_id)
        if meta is None:
            continue
        if result.result.type != "succeeded":
            n_failed += 1
            continue
        parsed = parse_json_response(extract_text(result.result.message.content))
        if parsed is None:
            n_parse_failed += 1
            print(f"WARNING: {meta['year']} Q{meta['q_num']} still unparseable at max_tokens={REPAIR_MAX_TOKENS}")
            continue
        out[f"{meta['year']}::{meta['q_num']}"] = parsed

    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(out)} total translations now cached → {OUT_PATH}")
    print(f"Repair: {n_failed} failed, {n_parse_failed} still unparseable out of {len(by_custom_id)} retried.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("test")
    sub.add_parser("submit")
    sub.add_parser("status")
    sub.add_parser("collect")
    sub.add_parser("repair-submit")
    sub.add_parser("repair-collect")
    args = parser.parse_args()
    {
        "test": cmd_test,
        "submit": cmd_submit,
        "status": cmd_status,
        "collect": cmd_collect,
        "repair-submit": cmd_repair_submit,
        "repair-collect": cmd_repair_collect,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
