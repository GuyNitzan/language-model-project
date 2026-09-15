"""
Phase 5, "still open" item #2 (derma_guide_plan.md): do the synthetic and
expert ground truths agree on which chunks matter? Nothing currently links a
synthetic train/val/test item to a specific exam question, so this builds the
practical version of that comparison: for each exam-dev question (2021-2024,
text_answerable, n=517), check whether the synthetic set (train+val+test,
1,219 items total) has *any* item whose source chunk's page range overlaps
the exam question's own resolved gold-chunk page range. That splits exam_dev
into "topically covered by the synthetic set too" vs. "book content the
synthetic set never sampled" -- then retrieval accuracy (config=400_80/bge,
non-reranked) is compared between the two subsets.

Answers: is the model's real-exam accuracy problem the same one the
synthetic-QA numbers already measure (i.e. retrieval is just worse on
whatever content the synthetic generator under-sampled), or a materially
different problem (retrieval is roughly as good/bad everywhere, and the
synthetic-vs-exam Hit@k gap documented in Phase 3 is about something else --
e.g. the ±2-page tolerance / approximate-citation noise already identified).

Usage:
    /home/guynitz/venvs/derma_eval/bin/python3 eval/analyze_synthetic_overlap.py
"""
import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

from eval_retrieval import (  # noqa: E402
    load_chunks, build_chapter_index, load_index, load_translations,
    load_exam_dev, load_synthetic, gold_for_exam, exam_query_text,
    embed_queries, hits_and_mrr, ranges_overlap, TOP_K,
)
from build_index import CHUNK_CONFIGS  # noqa: E402

CONFIG = "400_80"
EMBEDDER = "bge"
VARIANTS = ["hebrew_raw", "english_claude"]


def load_all_synthetic() -> list[dict]:
    items = []
    for split in ("train", "val", "test"):
        items.extend(load_synthetic(split))
    return items


def synthetic_ranges(synthetic_items: list[dict]) -> list[tuple[int, int]]:
    return [(it["page_start"], it["page_end"]) for it in synthetic_items]


def has_synthetic_overlap(gold_chunk_ids: set[str], chunks_by_id: dict,
                           syn_ranges: list[tuple[int, int]]) -> bool:
    for cid in gold_chunk_ids:
        c = chunks_by_id.get(cid)
        if not c:
            continue
        target = (c["page_start"], c["page_end"])
        if any(ranges_overlap(target, sr) for sr in syn_ranges):
            return True
    return False


def score_group(records, index, meta, chunks, by_chapter, page_map, translations, variant):
    query_texts, golds = [], []
    for r in records:
        text = exam_query_text(r, variant, translations)
        if text is None:
            continue
        gold = gold_for_exam(r, chunks, by_chapter, page_map)
        query_texts.append(text)
        golds.append(gold)
    if not query_texts:
        return None

    qvecs = embed_queries(query_texts, EMBEDDER)
    _, idxs = index.search(qvecs, TOP_K)

    hit1, hit10, mrrs = [], [], []
    for row, gold in zip(idxs, golds):
        ranked_ids = [meta[i]["chunk_id"] for i in row if i != -1]
        hits, mrr = hits_and_mrr(ranked_ids, gold)
        if mrr is None:
            continue
        hit1.append(hits[1])
        hit10.append(hits[10])
        mrrs.append(mrr)

    n = len(mrrs)
    if n == 0:
        return None
    return {
        "n": n,
        "hit_at_1": round(sum(hit1) / n, 4),
        "hit_at_10": round(sum(hit10) / n, 4),
        "mrr_at_10": round(sum(mrrs) / n, 4),
    }


def main():
    from parse_exam import _load_page_map

    page_map = _load_page_map()
    translations = load_translations()
    exam_dev = load_exam_dev()  # 2021-2024, text_answerable, n=517
    synthetic_items = load_all_synthetic()
    syn_ranges = synthetic_ranges(synthetic_items)
    print(f"exam_dev n={len(exam_dev)}; synthetic train+val+test n={len(synthetic_items)}")

    index, meta = load_index(CONFIG, EMBEDDER)
    if index is None:
        sys.exit(f"no index for {CONFIG}__{EMBEDDER} -- run build_index.py first")
    chunks = load_chunks(CHUNK_CONFIGS[CONFIG])
    chunks_by_id = {c["chunk_id"]: c for c in chunks}
    by_chapter = build_chapter_index(chunks)

    overlap, no_overlap, ungraded = [], [], 0
    for r in exam_dev:
        gold = gold_for_exam(r, chunks, by_chapter, page_map)
        if not gold:
            ungraded += 1
            continue
        (overlap if has_synthetic_overlap(gold, chunks_by_id, syn_ranges) else no_overlap).append(r)

    print(f"{len(overlap)} exam-dev questions have >=1 gold chunk whose page range overlaps "
          f"some synthetic item's source range (\"covered\"); {len(no_overlap)} do not "
          f"(\"uncovered\"); {ungraded} have no resolvable gold at all (excluded from this split)")

    results = {"n_covered": len(overlap), "n_uncovered": len(no_overlap), "n_ungraded_excluded": ungraded}
    print(f"\n{'group':10s} {'variant':16s} {'n':>4s} {'Hit@1':>7s} {'Hit@10':>7s} {'MRR@10':>7s}")
    for group_name, group in [("covered", overlap), ("uncovered", no_overlap)]:
        for variant in VARIANTS:
            m = score_group(group, index, meta, chunks, by_chapter, page_map, translations, variant)
            if m is None:
                print(f"{group_name:10s} {variant:16s}  [no usable queries]")
                continue
            results[f"{group_name}__{variant}"] = m
            print(f"{group_name:10s} {variant:16s} {m['n']:4d} {m['hit_at_1']:7.3f} "
                  f"{m['hit_at_10']:7.3f} {m['mrr_at_10']:7.3f}")

    out_path = EVAL_DIR / "data" / "synthetic_overlap_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWritten -> {out_path}")


if __name__ == "__main__":
    main()
