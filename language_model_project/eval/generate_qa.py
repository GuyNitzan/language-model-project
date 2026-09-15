"""
Phase 2b: generate the synthetic training QA set via Claude's Batch API.

Samples chunks stratified by chapter from pipeline/data/chunks/chunks.jsonl
(the fixed chunker's output — see 02_chunk.py), asks Claude to write one
question answerable *only* from that chunk plus a citation-grounded
reference answer in the same "[Chapter: <name>, p.<page>]" format the
product already uses (backend/llm.py), and splits the result chunk-disjoint
into train/val/test. This set is for QLoRA training only — the headline
retrieval/generation metrics come from the real exam questions parsed in
eval/parse_exam.py, not from this synthetic set (see derma_guide_plan.md
Phase 2b).

Two phases, since the Batch API is async (results can take up to 24h, but
usually much less):

    python eval/generate_qa.py test                 # 3 sync calls, sanity-check the prompt
    python eval/generate_qa.py submit                # sample chunks, submit the batch job
    python eval/generate_qa.py status                # poll batch status
    python eval/generate_qa.py collect               # once ended: download, parse, write splits

Run from the eval/ directory with backend/.venv's python (has anthropic +
python-dotenv installed):
    ../backend/.venv/bin/python3 generate_qa.py <subcommand>
"""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

EVAL_DIR       = Path(__file__).parent
CHUNKS_PATH    = EVAL_DIR.parent / "pipeline" / "data" / "chunks" / "chunks.jsonl"
DATA_DIR       = EVAL_DIR / "data"
MANIFEST_PATH  = DATA_DIR / "qa_batch_manifest.json"

MODEL = "claude-sonnet-5"
MAX_TOKENS = 400
REPAIR_MAX_TOKENS = 800  # some answers run long; 400 truncated ~6% of responses mid-JSON (stop_reason=max_tokens)
REPAIR_MANIFEST_PATH = DATA_DIR / "qa_repair_manifest.json"
MIN_CHUNK_WORDS = 150   # skip trailing/short chunks — not enough content for a good question

SPLIT_SIZES = {"train": 1000, "val": 120, "test": 100}
SEED = 42

SYSTEM_PROMPT = """You write study questions for a dermatology reference (Bolognia's Dermatology). \
Given one excerpt, write ONE question that a reader could answer using ONLY the information \
in that excerpt — do not require outside knowledge, and do not write a question the excerpt \
can't fully answer. Then write a concise, correct reference answer (2-4 sentences) grounded \
strictly in the excerpt, ending with an inline citation in exactly this format: \
[Chapter: <chapter name>, p.<page>]. Use the chapter and page given to you for the citation, \
not ones you infer.

Respond with ONLY a single JSON object, no other text, no markdown fences:
{"question": "...", "answer": "..."}"""


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if len(c["text"].split()) >= MIN_CHUNK_WORDS:
                chunks.append(c)
    return chunks


def stratified_sample(chunks: list[dict], n_total: int, seed: int = SEED) -> list[dict]:
    """Sample n_total chunks, proportionally by chapter, largest-remainder
    method so the total lands exactly on n_total."""
    rng = random.Random(seed)
    by_chapter: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        by_chapter[c["chapter"] or "__front_matter__"].append(c)

    n_available = len(chunks)
    quotas = {}
    remainders = []
    allocated = 0
    for chapter, cs in by_chapter.items():
        exact = n_total * len(cs) / n_available
        base = int(exact)
        quotas[chapter] = min(base, len(cs))
        allocated += quotas[chapter]
        remainders.append((exact - base, chapter))

    # distribute the remainder (n_total - allocated) to the chapters with the
    # largest fractional part, respecting each chapter's available count
    remainders.sort(reverse=True)
    i = 0
    while allocated < n_total and i < len(remainders):
        _, chapter = remainders[i]
        if quotas[chapter] < len(by_chapter[chapter]):
            quotas[chapter] += 1
            allocated += 1
        i += 1

    sampled = []
    for chapter, cs in by_chapter.items():
        k = quotas[chapter]
        if k:
            sampled.extend(rng.sample(cs, k))
    rng.shuffle(sampled)
    return sampled[:n_total]


def build_request(custom_id: str, chunk: dict, max_tokens: int = MAX_TOKENS) -> dict:
    page_ref = (f"{chunk['page_start']}"
                if chunk["page_start"] == chunk["page_end"]
                else f"{chunk['page_start']}-{chunk['page_end']}")
    user_msg = (f"Chapter: {chunk['chapter']}\n"
                f"Page: {page_ref}\n\n"
                f"Excerpt:\n{chunk['text']}")
    return {
        "custom_id": custom_id,
        "params": {
            "model": MODEL,
            "max_tokens": max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_msg}],
        },
    }


def extract_text(content_blocks) -> str:
    """Claude-sonnet-5 responses can include a ThinkingBlock before the
    TextBlock; find the actual text block rather than assuming index 0."""
    for block in content_blocks:
        if block.type == "text":
            return block.text
    return ""


def parse_json_response(text: str) -> dict | None:
    text = text.strip()
    # tolerate stray markdown fences or leading/trailing prose
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if "question" not in obj or "answer" not in obj:
        return None
    return obj


def get_client():
    load_dotenv(EVAL_DIR.parent / "backend" / ".env")
    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        sys.exit("ANTHROPIC_API_KEY not found (checked backend/.env).")
    return anthropic.Anthropic(api_key=key)


# ---------------------------------------------------------------- test ----

def cmd_test(args):
    """Sanity-check the prompt synchronously on a few chunks before paying
    for the full batch."""
    client = get_client()
    chunks = load_chunks(CHUNKS_PATH)
    sample = stratified_sample(chunks, 3)
    for c in sample:
        req = build_request("test", c)["params"]
        resp = client.messages.create(**req)
        text = extract_text(resp.content)
        print(f"\n=== {c['chunk_id']} (chapter={c['chapter']!r}, "
              f"p.{c['page_start']}-{c['page_end']}) ===")
        print(text)
        parsed = parse_json_response(text)
        print("PARSED OK" if parsed else "PARSE FAILED", parsed)


# -------------------------------------------------------------- submit ----

def cmd_submit(args):
    client = get_client()
    chunks = load_chunks(CHUNKS_PATH)
    print(f"{len(chunks)} chunks eligible (>= {MIN_CHUNK_WORDS} words) out of "
          f"total in {CHUNKS_PATH}")

    n_total = args.train + args.val + args.test
    sampled = stratified_sample(chunks, n_total, seed=args.seed)
    assert len(sampled) == n_total, f"sampled {len(sampled)}, wanted {n_total}"

    # split by chunk, disjoint by construction (each chunk sampled once,
    # assigned to exactly one split below)
    splits = (["train"] * args.train + ["val"] * args.val + ["test"] * args.test)
    assert len(splits) == len(sampled)

    manifest_requests = []
    batch_requests = []
    for i, (chunk, split) in enumerate(zip(sampled, splits)):
        custom_id = f"qa_{i:05d}"
        batch_requests.append(build_request(custom_id, chunk))
        manifest_requests.append({
            "custom_id":    custom_id,
            "chunk_id":     chunk["chunk_id"],
            "chapter":      chunk["chapter"],
            "page_start":   chunk["page_start"],
            "page_end":     chunk["page_end"],
            "split":        split,
        })

    if args.dry_run:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(
            json.dumps({"batch_id": None, "requests": manifest_requests}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[dry-run] Would submit {len(batch_requests)} requests "
              f"(train={args.train}, val={args.val}, test={args.test}). "
              f"Manifest written without a batch_id → {MANIFEST_PATH}")
        return

    batch = client.messages.batches.create(requests=batch_requests)
    print(f"Submitted batch {batch.id} ({len(batch_requests)} requests), "
          f"status={batch.processing_status}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps({"batch_id": batch.id, "requests": manifest_requests}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Manifest written → {MANIFEST_PATH}")
    print("Check progress with: python eval/generate_qa.py status")


# -------------------------------------------------------------- status ----

def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        sys.exit(f"No manifest at {MANIFEST_PATH} — run `submit` first.")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def cmd_status(args):
    manifest = _load_manifest()
    if not manifest.get("batch_id"):
        sys.exit("Manifest has no batch_id (was written with --dry-run). Run `submit` for real.")
    client = get_client()
    batch = client.messages.batches.retrieve(manifest["batch_id"])
    print(f"batch {batch.id}: status={batch.processing_status}")
    print(f"  counts: {batch.request_counts}")


# ------------------------------------------------------------- collect ----

def cmd_collect(args):
    manifest = _load_manifest()
    batch_id = args.batch_id or manifest.get("batch_id")
    if not batch_id:
        sys.exit("No batch_id in manifest and none passed via --batch-id.")

    client = get_client()
    batch = client.messages.batches.retrieve(batch_id)
    print(f"batch {batch.id}: status={batch.processing_status}, counts={batch.request_counts}")
    if batch.processing_status != "ended":
        print("Not finished yet — re-run this command later.")
        return

    by_custom_id = {r["custom_id"]: r for r in manifest["requests"]}
    records_by_split: dict[str, list[dict]] = defaultdict(list)
    n_failed = 0
    n_parse_failed = 0
    seen_custom_ids = set()

    for result in client.messages.batches.results(batch_id):
        cid = result.custom_id
        seen_custom_ids.add(cid)
        meta = by_custom_id.get(cid)
        if meta is None:
            print(f"WARNING: result for unknown custom_id {cid}, skipping")
            continue

        if result.result.type != "succeeded":
            n_failed += 1
            print(f"WARNING: {cid} ({meta['chunk_id']}) did not succeed: {result.result.type}")
            continue

        text = extract_text(result.result.message.content)
        parsed = parse_json_response(text)
        if parsed is None:
            n_parse_failed += 1
            print(f"WARNING: {cid} ({meta['chunk_id']}) — could not parse JSON from response")
            continue

        records_by_split[meta["split"]].append({
            "question":      parsed["question"],
            "answer":        parsed["answer"],
            "gold_chunk_id": meta["chunk_id"],
            "chapter":       meta["chapter"],
            "page_start":    meta["page_start"],
            "page_end":      meta["page_end"],
        })

    missing = set(by_custom_id) - seen_custom_ids
    if missing:
        print(f"WARNING: {len(missing)} manifest entries had no result at all "
              f"(batch may have been cancelled/expired for those)")

    # leakage check — must never trip, since each chunk was sampled once and
    # assigned to exactly one split; a failure here means the manifest/split
    # logic itself is broken, so fail loudly rather than write bad splits.
    ids_by_split = {s: {r["gold_chunk_id"] for r in rs} for s, rs in records_by_split.items()}
    for a in ids_by_split:
        for b in ids_by_split:
            if a < b:
                overlap = ids_by_split[a] & ids_by_split[b]
                assert not overlap, f"LEAKAGE: {len(overlap)} chunk_ids in both {a} and {b}: {list(overlap)[:5]}"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split, records in records_by_split.items():
        out_path = DATA_DIR / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split}: {len(records)} records → {out_path}")

    print(f"\nDone. {n_failed} failed generations, {n_parse_failed} unparseable responses "
          f"out of {len(by_custom_id)} requested.")


# --------------------------------------------------------------- repair ---

def _existing_chunk_ids_by_split() -> dict[str, set[str]]:
    have = {}
    for split in SPLIT_SIZES:
        path = DATA_DIR / f"{split}.jsonl"
        ids = set()
        if path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    ids.add(json.loads(line)["gold_chunk_id"])
        have[split] = ids
    return have


def cmd_repair_submit(args):
    """Re-run generation for manifest entries whose response failed to parse
    (typically stop_reason=max_tokens truncating the JSON) with a larger
    max_tokens budget."""
    manifest = _load_manifest()
    have = _existing_chunk_ids_by_split()
    already = set().union(*have.values()) if have else set()

    missing = [r for r in manifest["requests"] if r["chunk_id"] not in already]
    if not missing:
        print("Nothing to repair — every manifest entry is already collected.")
        return

    chunk_by_id = {c["chunk_id"]: c for c in load_chunks(CHUNKS_PATH)}
    client = get_client()
    batch_requests = []
    for meta in missing:
        chunk = chunk_by_id.get(meta["chunk_id"])
        if chunk is None:
            print(f"WARNING: {meta['chunk_id']} not found in current chunks.jsonl, skipping repair")
            continue
        batch_requests.append(build_request(meta["custom_id"], chunk, max_tokens=REPAIR_MAX_TOKENS))

    batch = client.messages.batches.create(requests=batch_requests)
    print(f"Submitted repair batch {batch.id} ({len(batch_requests)} requests, "
          f"max_tokens={REPAIR_MAX_TOKENS}), status={batch.processing_status}")

    REPAIR_MANIFEST_PATH.write_text(
        json.dumps({"batch_id": batch.id, "requests": missing}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Repair manifest written → {REPAIR_MANIFEST_PATH}")


def cmd_repair_collect(args):
    if not REPAIR_MANIFEST_PATH.exists():
        sys.exit(f"No repair manifest at {REPAIR_MANIFEST_PATH} — run `repair-submit` first.")
    manifest = json.loads(REPAIR_MANIFEST_PATH.read_text(encoding="utf-8"))
    batch_id = manifest["batch_id"]

    client = get_client()
    batch = client.messages.batches.retrieve(batch_id)
    print(f"repair batch {batch.id}: status={batch.processing_status}, counts={batch.request_counts}")
    if batch.processing_status != "ended":
        print("Not finished yet — re-run this command later.")
        return

    by_custom_id = {r["custom_id"]: r for r in manifest["requests"]}
    new_by_split: dict[str, list[dict]] = defaultdict(list)
    n_failed = 0
    n_parse_failed = 0

    for result in client.messages.batches.results(batch_id):
        meta = by_custom_id.get(result.custom_id)
        if meta is None:
            continue
        if result.result.type != "succeeded":
            n_failed += 1
            print(f"WARNING: {result.custom_id} ({meta['chunk_id']}) still did not succeed: {result.result.type}")
            continue
        text = extract_text(result.result.message.content)
        parsed = parse_json_response(text)
        if parsed is None:
            n_parse_failed += 1
            print(f"WARNING: {result.custom_id} ({meta['chunk_id']}) — still unparseable even at "
                  f"max_tokens={REPAIR_MAX_TOKENS}")
            continue
        new_by_split[meta["split"]].append({
            "question":      parsed["question"],
            "answer":        parsed["answer"],
            "gold_chunk_id": meta["chunk_id"],
            "chapter":       meta["chapter"],
            "page_start":    meta["page_start"],
            "page_end":      meta["page_end"],
        })

    have = _existing_chunk_ids_by_split()
    for split, records in new_by_split.items():
        # leakage guard: a repaired record must not already exist in a
        # *different* split (would indicate a manifest/bookkeeping bug)
        for other_split, ids in have.items():
            if other_split != split:
                dup = {r["gold_chunk_id"] for r in records} & ids
                assert not dup, f"LEAKAGE: repaired {split} record(s) already present in {other_split}: {dup}"

        out_path = DATA_DIR / f"{split}.jsonl"
        with out_path.open("a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split}: +{len(records)} repaired records appended → {out_path}")

    print(f"\nRepair done. {n_failed} still failed, {n_parse_failed} still unparseable "
          f"out of {len(by_custom_id)} retried.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("test", help="synchronously test the prompt on 3 sample chunks")

    p_submit = sub.add_parser("submit", help="sample chunks and submit the batch job")
    p_submit.add_argument("--train", type=int, default=SPLIT_SIZES["train"])
    p_submit.add_argument("--val", type=int, default=SPLIT_SIZES["val"])
    p_submit.add_argument("--test", type=int, default=SPLIT_SIZES["test"])
    p_submit.add_argument("--seed", type=int, default=SEED)
    p_submit.add_argument("--dry-run", action="store_true",
                           help="sample + write manifest but don't call the API")

    sub.add_parser("status", help="poll batch status")

    p_collect = sub.add_parser("collect", help="download results and write train/val/test.jsonl")
    p_collect.add_argument("--batch-id", default=None)

    sub.add_parser("repair-submit", help="resubmit unparseable manifest entries with a larger max_tokens")
    sub.add_parser("repair-collect", help="collect the repair batch and append into train/val/test.jsonl")

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
