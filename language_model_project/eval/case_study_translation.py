"""
Phase 5, "still open" item #2 (derma_guide_plan.md): finding #5 already shows
translating Hebrew -> English before retrieval buys +23-27% relative MRR in
aggregate (400_80/bge, exam_dev text_answerable). This script makes it
concrete at the record level: find individual questions where the raw-Hebrew
query missed entirely (no gold chunk in the top 10) but the Claude-translated
English query hit, and print the two query texts + what was actually
retrieved side by side, so a handful can be picked for the report.

Reuses the exact same population, config, and scoring as eval_retrieval.py's
exam_dev / hebrew_raw / english_claude run (config=400_80, embedder=bge,
non-reranked) -- this is a per-record breakdown of an already-computed
aggregate, not a new retrieval experiment.

Usage:
    /home/guynitz/venvs/derma_eval/bin/python3 eval/case_study_translation.py
"""
import json
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


def rank_and_top_batch(index, meta, chunk_by_id, query_texts, golds):
    """Embeds every query in one batch (embed_queries loads the SentenceTransformer
    fresh per call -- calling it once per record would reload the model hundreds
    of times) and searches each against the index individually."""
    qvecs = embed_queries(query_texts, EMBEDDER)
    _, idxs = index.search(qvecs, TOP_K)
    results = []
    for row, gold in zip(idxs, golds):
        ranked_ids = [meta[i]["chunk_id"] for i in row if i != -1]
        hits, mrr = hits_and_mrr(ranked_ids, gold)
        top = chunk_by_id.get(ranked_ids[0]) if ranked_ids else None
        results.append({
            "hit_at_10": bool(hits[10]) if hits[10] is not None else None,
            "mrr": mrr,
            "top1_chunk_id": ranked_ids[0] if ranked_ids else None,
            "top1_chapter": top["chapter"] if top else None,
            "top1_snippet": (top["text"][:220] + "...") if top else None,
        })
    return results


def main():
    from parse_exam import _load_page_map

    page_map = _load_page_map()
    translations = load_translations()
    exam_dev = load_exam_dev()

    index, meta = load_index(CONFIG, EMBEDDER)
    if index is None:
        sys.exit(f"no index for {CONFIG}__{EMBEDDER} -- run build_index.py first")
    chunks = load_chunks(CHUNK_CONFIGS[CONFIG])
    chunk_by_id = {c["chunk_id"]: c for c in chunks}
    by_chapter = build_chapter_index(chunks)

    usable, heb_texts, eng_texts, golds = [], [], [], []
    for r in exam_dev:
        heb_text = exam_query_text(r, "hebrew_raw", translations)
        eng_text = exam_query_text(r, "english_claude", translations)
        if heb_text is None or eng_text is None:
            continue  # no translation available for this record
        gold = gold_for_exam(r, chunks, by_chapter, page_map)
        if not gold:
            continue
        usable.append(r)
        heb_texts.append(heb_text)
        eng_texts.append(eng_text)
        golds.append(gold)

    heb_results = rank_and_top_batch(index, meta, chunk_by_id, heb_texts, golds)
    eng_results = rank_and_top_batch(index, meta, chunk_by_id, eng_texts, golds)

    cases = []
    for r, heb_text, eng_text, gold, heb, eng in zip(
            usable, heb_texts, eng_texts, golds, heb_results, eng_results):
        if heb["hit_at_10"] is False and eng["hit_at_10"] is True:
            cases.append({
                "year": r["year"], "q_num": r["q_num"],
                "reference": r["reference"],
                "hebrew_query": heb_text,
                "english_query": eng_text,
                "gold_chunk_ids": sorted(gold),
                "hebrew_result": heb,
                "english_result": eng,
            })

    print(f"exam_dev n={len(exam_dev)}; {len(cases)} records where hebrew_raw missed (Hit@10=0) "
          f"but english_claude hit (Hit@10=1)")

    out_path = EVAL_DIR / "data" / "translation_case_studies.json"
    out_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"All {len(cases)} cases written -> {out_path}")

    # Print a handful of the cleanest examples (english hit at rank 1) for the report.
    clean = [c for c in cases if c["english_result"]["mrr"] == 1.0]
    print(f"\n{len(clean)} of those have english_claude hitting at rank 1 (cleanest cases). "
          f"First 3:\n")
    for c in clean[:3]:
        print(f"--- {c['year']} Q{c['q_num']} (ref pages {c['reference']['pages']}, "
              f"chapter {c['reference']['chapter']}) ---")
        print(f"Hebrew query:  {c['hebrew_query'][:200]}")
        print(f"English query: {c['english_query'][:200]}")
        print(f"Hebrew top-1 retrieved:  [{c['hebrew_result']['top1_chapter']}] "
              f"{c['hebrew_result']['top1_snippet']}")
        print(f"English top-1 retrieved: [{c['english_result']['top1_chapter']}] "
              f"{c['english_result']['top1_snippet']}")
        print()


if __name__ == "__main__":
    main()
