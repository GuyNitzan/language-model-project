"""
Phase 5, "still open" item #1 (derma_guide_plan.md): are multi-hop references
(a citation that points at two or more distinct book pages, e.g. "391, 397-398,
406") a harder retrieval target than an ordinary single-location reference?

Reuses eval_retrieval.py's exact loading/scoring machinery (config=400_80,
embedder=bge, non-reranked -- the same setup the page-range-bugfix
re-verification and the #6 table/figure split both used) and its exam_dev
population (2021-2024 develop years, text_answerable bucket, n=517), so this
is a like-for-like split of an already-computed eval, not a new experiment.

A reference is counted as real multi-hop only if:
  - its raw text has a comma AND
  - it has no Latin letters (excludes the one genuine parser artifact found in
    this set -- 2021 Q135's reference is a full journal citation
    ("Siegel, Jacob, et al. ... 79.6 (2018): 1081-1088"), not a page list; its
    comma is bibliographic punctuation, not a second citation) AND
  - it resolves to >=2 distinct pages after range-expansion.

This is stricter than "contains a comma", by design: the earlier page-range
abbreviation bug (`"357-60"` -> disjoint [60, 357]) is already fixed upstream
in parse_exam.py, so no bug-workaround is needed here -- this script only
needs to separate genuine multi-location citations from single-location ones
and from the one bibliographic outlier.

Usage:
    /home/guynitz/venvs/derma_eval/bin/python3 eval/analyze_multihop.py
"""
import json
import re
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

from eval_retrieval import (  # noqa: E402
    load_chunks, build_chapter_index, load_index, load_translations,
    load_exam_dev, gold_for_exam, exam_query_text, embed_queries, hits_and_mrr,
    TOP_K,
)
from build_index import CHUNK_CONFIGS  # noqa: E402

CONFIG = "400_80"
EMBEDDER = "bge"
VARIANTS = ["hebrew_raw", "english_claude"]


def is_real_multihop(ref: dict | None) -> bool:
    if not ref:
        return False
    raw = ref["raw"]
    if "," not in raw or re.search(r"[A-Za-z]", raw):
        return False
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return len(parts) >= 2 and len(ref["pages"]) >= 2


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

    hit1, mrrs, n_ungraded = [], [], 0
    for row, gold in zip(idxs, golds):
        ranked_ids = [meta[i]["chunk_id"] for i in row if i != -1]
        hits, mrr = hits_and_mrr(ranked_ids, gold)
        if mrr is None:
            n_ungraded += 1
            continue
        hit1.append(hits[1])
        mrrs.append(mrr)

    n = len(mrrs)
    if n == 0:
        return None
    return {
        "n": n,
        "n_ungraded": n_ungraded,
        "hit_at_1": round(sum(hit1) / n, 4),
        "mrr_at_10": round(sum(mrrs) / n, 4),
    }


def main():
    from parse_exam import _load_page_map

    page_map = _load_page_map()
    translations = load_translations()
    exam_dev = load_exam_dev()  # 2021-2024, text_answerable, n=517 -- same population as finding #6

    multihop = [r for r in exam_dev if is_real_multihop(r["reference"])]
    single = [r for r in exam_dev if not is_real_multihop(r["reference"])]
    print(f"exam_dev text_answerable n={len(exam_dev)}: "
          f"{len(multihop)} real multi-hop, {len(single)} single-location")
    by_year = {}
    for r in multihop:
        by_year[r["year"]] = by_year.get(r["year"], 0) + 1
    print(f"multi-hop by year: {by_year}")

    index, meta = load_index(CONFIG, EMBEDDER)
    if index is None:
        sys.exit(f"no index for {CONFIG}__{EMBEDDER} -- run build_index.py first")
    chunks = load_chunks(CHUNK_CONFIGS[CONFIG])
    by_chapter = build_chapter_index(chunks)

    results = {}
    print(f"\n{'group':12s} {'variant':16s} {'n':>4s} {'Hit@1':>7s} {'MRR@10':>7s}")
    for group_name, group in [("multi_hop", multihop), ("single_location", single)]:
        for variant in VARIANTS:
            m = score_group(group, index, meta, chunks, by_chapter, page_map, translations, variant)
            if m is None:
                print(f"{group_name:12s} {variant:16s}  [no usable queries]")
                continue
            results[f"{group_name}__{variant}"] = m
            print(f"{group_name:12s} {variant:16s} {m['n']:4d} {m['hit_at_1']:7.3f} {m['mrr_at_10']:7.3f}")

    out_path = EVAL_DIR / "data" / "multihop_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWritten -> {out_path}")


if __name__ == "__main__":
    main()
