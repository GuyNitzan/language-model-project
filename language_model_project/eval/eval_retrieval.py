"""
Phase 3: evaluate a built FAISS index (see build_index.py) against two
independent gold-label sources:

    synthetic_val / synthetic_test   eval/data/{val,test}.jsonl — gold is an
                                      exact chunk_id when it exists in the
                                      config being tested (it always does for
                                      "400_80", the config the synthetic set
                                      was sampled from), else a page-range
                                      overlap against the record's stored
                                      page_start/page_end (the same
                                      near-duplicate-chunk tolerance the plan
                                      documents for the single-config case,
                                      generalised here to cross-config eval).

    exam_dev                         eval/data/exam_parsed/all.jsonl,
                                      restricted to the develop years
                                      (2021-2024; 2025/2026 stay held out) and
                                      the text_answerable bucket. Gold is
                                      recomputed fresh per chunk config via
                                      parse_exam.resolve_reference — the
                                      stored `reference` field (book/chapter/
                                      pages) is config-independent, but the
                                      resolved chunk_ids baked into
                                      exam_parsed/*.jsonl are NOT (they were
                                      computed against whichever chunks.jsonl
                                      existed when parse_exam.py last ran),
                                      so those stored chunk_ids must never be
                                      reused directly across configs.

Query-language axis (exam_dev only — the synthetic set is already English):
    hebrew_raw        stem + options, untouched
    english_claude     Claude's translation (eval/translate_queries.py)
    english_glossary   translation + "Key terms: ..." suffix

Metrics: Hit@1/3/5/10 and MRR@10, retrieved once at k=10 and sliced post-hoc.

Usage:
    /home/guynitz/venvs/derma_eval/bin/python3 eval_retrieval.py \\
        --configs baseline_buggy 400_80 250_50 --embedder bge

Output: prints a results table and merges into eval/data/retrieval_results.json
(merges by (config, embedder, query_set, query_variant) key, so partial runs —
e.g. before exam-question translations exist — compose across invocations).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import faiss
import numpy as np

EVAL_DIR     = Path(__file__).parent
PIPELINE_DIR = EVAL_DIR.parent / "pipeline"
CHUNKS_DIR   = PIPELINE_DIR / "data" / "chunks"
INDICES_DIR  = EVAL_DIR / "data" / "indices"
RESULTS_PATH = EVAL_DIR / "data" / "retrieval_results.json"
TRANSLATIONS_PATH = EVAL_DIR / "data" / "exam_query_translations.json"

sys.path.insert(0, str(EVAL_DIR))
# parse_exam is intentionally NOT imported at module level: it hard-requires
# pymupdf (`import fitz`, no try/except fallback) purely for its PDF-extraction
# code path, which gold_for_exam()/main() below never touch (they only need
# _load_page_map/resolve_reference, which operate on already-loaded dicts).
# Importing it eagerly here would force every caller of this module —
# including eval_mcq.py and eval_generation.py, which use only
# embed_queries/load_chunks/load_index and have nothing to do with exam
# parsing — to have pymupdf installed and parse_exam.py/audit_exams.py on
# disk. Concretely: this broke notebooks/04_generation_eval.ipynb on Colab,
# which never uploads the PDF-parsing stack (see that notebook's file list).
# Deferred into the two functions that actually need it instead.
from build_index import CHUNK_CONFIGS, BGE_MODEL  # noqa: E402

DEV_YEARS = {"2021", "2022", "2023", "2024_05", "2024_09"}
TOP_K = 10
K_VALUES = [1, 3, 5, 10]
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
RERANK_CANDIDATES = 20  # retrieve this many, then rerank down to TOP_K
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker = None  # lazy singleton — loaded once per process, not once per query set


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(query_text: str, candidate_ids: list[str], chunks_by_id: dict) -> list[str]:
    pairs = [(query_text, chunks_by_id[cid]["text"]) for cid in candidate_ids if cid in chunks_by_id]
    ids = [cid for cid in candidate_ids if cid in chunks_by_id]
    if not pairs:
        return candidate_ids
    scores = get_reranker().predict(pairs)
    order = np.argsort(-scores)
    return [ids[i] for i in order]


# --------------------------------------------------------------- loading ---

def load_chunks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_chapter_index(chunks: list[dict]) -> dict:
    by_chapter = {}
    for c in chunks:
        m = re.match(r"\s*(\d+)\s", c["chapter"])
        if m:
            by_chapter.setdefault(int(m.group(1)), []).append(c)
    return by_chapter


def load_index(config: str, embedder: str):
    index_path = INDICES_DIR / f"{config}__{embedder}.faiss"
    meta_path  = INDICES_DIR / f"{config}__{embedder}.meta.jsonl"
    if not index_path.exists():
        return None, None
    index = faiss.read_index(str(index_path))
    meta = load_chunks(meta_path)
    return index, meta


def load_synthetic(split: str) -> list[dict]:
    path = EVAL_DIR / "data" / f"{split}.jsonl"
    return load_chunks(path) if path.exists() else []


def load_exam_dev() -> list[dict]:
    path = EVAL_DIR / "data" / "exam_parsed" / "all.jsonl"
    recs = load_chunks(path)
    return [r for r in recs if r["year"] in DEV_YEARS and r["bucket"] == "text_answerable"]


def load_translations() -> dict:
    if not TRANSLATIONS_PATH.exists():
        return {}
    return json.loads(TRANSLATIONS_PATH.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ gold ---

def ranges_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


SYNTHETIC_SOURCE_CONFIG = "400_80"  # eval/generate_qa.py sampled from pipeline/data/chunks/chunks.jsonl == this config


def gold_for_synthetic(record: dict, chunks: list[dict], chunk_ids: set[str], config: str) -> set[str]:
    # chunk_id naming ("<chapter>_<index:04d>") is only unique WITHIN one
    # config's own file, not across configs — a 400_80 chunk_id like
    # "..._0053" can coincidentally also exist in 250_50's namespace,
    # pointing at a totally unrelated chunk. Only trust an exact chunk_id
    # match against the exact config the synthetic set was generated from;
    # every other config must use the page-overlap fallback even if the
    # string happens to collide.
    if config == SYNTHETIC_SOURCE_CONFIG and record["gold_chunk_id"] in chunk_ids:
        return {record["gold_chunk_id"]}
    target = (record["page_start"], record["page_end"])
    return {c["chunk_id"] for c in chunks if ranges_overlap((c["page_start"], c["page_end"]), target)}


def gold_for_exam(record: dict, chunks: list[dict], by_chapter: dict, page_map: dict) -> set[str]:
    from parse_exam import resolve_reference
    return set(resolve_reference(chunks, by_chapter, page_map, record["reference"]))


# --------------------------------------------------------------- queries ---

def exam_query_text(record: dict, variant: str, translations: dict) -> str | None:
    if variant == "hebrew_raw":
        parts = [record["stem"].strip()]
        for letter, text in record["options"].items():
            if text.strip():
                parts.append(f"{letter}: {text.strip()}")
        return "\n".join(parts)

    key = f"{record['year']}::{record['q_num']}"
    tr = translations.get(key)
    if tr is None:
        return None
    if variant == "english_claude":
        return tr["english"]
    if variant == "english_glossary":
        terms = tr.get("terms") or []
        suffix = f" Key terms: {', '.join(terms)}." if terms else ""
        return tr["english"] + suffix
    raise ValueError(variant)


# -------------------------------------------------------------- metrics ---

def embed_queries(texts: list[str], embedder: str):
    if embedder == "bge":
        import torch
        torch.set_num_threads(8)
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(BGE_MODEL)
        prefixed = [BGE_QUERY_INSTRUCTION + t for t in texts]
        return model.encode(prefixed, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    elif embedder == "openai":
        import os
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv(EVAL_DIR.parent / "backend" / ".env")
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.embeddings.create(model="text-embedding-3-large", input=texts)
        arr = np.array([d.embedding for d in resp.data], dtype="float32")
        faiss.normalize_L2(arr)
        return arr
    raise ValueError(embedder)


def hits_and_mrr(ranked_chunk_ids: list[str], gold: set[str]) -> tuple[dict, float]:
    """ranked_chunk_ids: top-10 chunk_ids in rank order. Returns (hit@k for
    each k in K_VALUES, reciprocal rank capped at rank 10, else 0)."""
    if not gold:
        return {k: None for k in K_VALUES}, None  # ungraded query — no usable gold at all
    first_hit_rank = None
    for rank, cid in enumerate(ranked_chunk_ids, start=1):
        if cid in gold:
            first_hit_rank = rank
            break
    hits = {k: int(first_hit_rank is not None and first_hit_rank <= k) for k in K_VALUES}
    mrr = (1.0 / first_hit_rank) if first_hit_rank else 0.0
    return hits, mrr


def evaluate_query_set(index, meta, chunks, by_chapter, page_map, chunk_ids: set[str],
                        records: list[dict], is_synthetic: bool, embedder: str,
                        variant: str | None, translations: dict, config: str,
                        use_reranker: bool = False, chunks_by_id: dict | None = None) -> dict | None:
    query_texts, golds = [], []
    for r in records:
        if is_synthetic:
            text = r["question"]
            gold = gold_for_synthetic(r, chunks, chunk_ids, config)
        else:
            text = exam_query_text(r, variant, translations)
            if text is None:
                continue
            gold = gold_for_exam(r, chunks, by_chapter, page_map)
        query_texts.append(text)
        golds.append(gold)

    if not query_texts:
        return None

    qvecs = embed_queries(query_texts, embedder)
    search_k = RERANK_CANDIDATES if use_reranker else TOP_K
    _, idxs = index.search(qvecs, search_k)

    per_k_hits = {k: [] for k in K_VALUES}
    mrrs = []
    n_ungraded = 0
    for row, gold, text in zip(idxs, golds, query_texts):
        ranked_ids = [meta[i]["chunk_id"] for i in row if i != -1]
        if use_reranker:
            ranked_ids = rerank(text, ranked_ids, chunks_by_id)[:TOP_K]
        hits, mrr = hits_and_mrr(ranked_ids, gold)
        if mrr is None:
            n_ungraded += 1
            continue
        for k in K_VALUES:
            per_k_hits[k].append(hits[k])
        mrrs.append(mrr)

    n = len(mrrs)
    if n == 0:
        return None
    return {
        "n_queries": n,
        "n_ungraded": n_ungraded,
        "hit_at_k": {str(k): round(sum(v) / n, 4) for k, v in per_k_hits.items()},
        "mrr_at_10": round(sum(mrrs) / n, 4),
    }


# ---------------------------------------------------------------- driver ---

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", default=list(CHUNK_CONFIGS), choices=list(CHUNK_CONFIGS))
    parser.add_argument("--embedder", default="bge", choices=["bge", "openai"])
    parser.add_argument("--rerank", action="store_true",
                         help=f"retrieve top-{RERANK_CANDIDATES} then rerank to top-{TOP_K} with {RERANKER_MODEL} "
                              f"(intended for the winning config only, per the plan)")
    args = parser.parse_args()

    from parse_exam import _load_page_map
    page_map = _load_page_map()
    translations = load_translations()
    synthetic_val = load_synthetic("val")
    synthetic_test = load_synthetic("test")
    exam_dev = load_exam_dev()
    print(f"Loaded {len(synthetic_val)} synthetic val, {len(synthetic_test)} synthetic test, "
          f"{len(exam_dev)} exam-dev text_answerable questions ({len(translations)} translated)")

    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else {}

    for config in args.configs:
        index, meta = load_index(config, args.embedder)
        if index is None:
            print(f"[skip] no index for {config}__{args.embedder} — run build_index.py first")
            continue

        chunks = load_chunks(CHUNK_CONFIGS[config])
        chunk_ids = {c["chunk_id"] for c in chunks}
        chunks_by_id = {c["chunk_id"]: c for c in chunks} if args.rerank else None
        by_chapter = build_chapter_index(chunks)
        tag = " +rerank" if args.rerank else ""
        print(f"\n=== {config} / {args.embedder}{tag} ({index.ntotal} vectors) ===")

        runs = [("synthetic_val", synthetic_val, True, None),
                ("synthetic_test", synthetic_test, True, None)]
        for variant in ["hebrew_raw", "english_claude", "english_glossary"]:
            runs.append((f"exam_dev__{variant}", exam_dev, False, variant))

        for set_name, records, is_synthetic, variant in runs:
            metrics = evaluate_query_set(index, meta, chunks, by_chapter, page_map, chunk_ids,
                                          records, is_synthetic, args.embedder, variant, translations, config,
                                          use_reranker=args.rerank, chunks_by_id=chunks_by_id)
            if metrics is None:
                print(f"  [skip] {set_name}: no usable queries yet")
                continue
            key = f"{config}__{args.embedder}{'__rerank' if args.rerank else ''}__{set_name}"
            results[key] = {"config": config, "embedder": args.embedder, "reranked": args.rerank,
                             "query_set": set_name, **metrics}
            hit_str = " ".join(f"Hit@{k}={metrics['hit_at_k'][str(k)]:.3f}" for k in K_VALUES)
            print(f"  {set_name:28s} n={metrics['n_queries']:4d}  {hit_str}  MRR@10={metrics['mrr_at_10']:.3f}")

    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults merged → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
