"""
Phase 3: build a FAISS retrieval index for one (chunk-config, embedder) pair.

Chunk configs (see derma_guide_plan.md Phase 3's grid):
    baseline_buggy  pipeline/data/chunks/chunks_baseline_buggy.jsonl  (~770w/0 overlap — the pre-fix chunker's output, kept for comparison)
    400_80          pipeline/data/chunks/chunks.jsonl                 (the fixed chunker's default output)
    250_50          pipeline/data/chunks/chunks_250_50.jsonl          (fixed chunker, smaller/tighter window)

Embedders:
    bge     BAAI/bge-small-en-v1.5, local, free — used across the whole config grid
    openai  text-embedding-3-large, paid — used only for the winning bge config,
            to compare open vs. closed embeddings on equal footing. For
            baseline_buggy this reuses the already-embedded
            chunks_baseline_buggy_embedded.jsonl (real product cost already spent)
            rather than re-calling the API.

Output, per (config, embedder):
    eval/data/indices/<config>__<embedder>.faiss       — a cosine-similarity
                                                          (inner-product over
                                                          L2-normalised vectors) index
    eval/data/indices/<config>__<embedder>.meta.jsonl  — one line per FAISS row,
                                                          same order as the index:
                                                          {chunk_id, chapter, page_start, page_end}

Usage:
    /home/guynitz/venvs/derma_eval/bin/python3 build_index.py --config 400_80 --embedder bge
    /home/guynitz/venvs/derma_eval/bin/python3 build_index.py --config baseline_buggy --embedder openai
"""

import argparse
import json
import os
import time
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv

EVAL_DIR     = Path(__file__).parent
PIPELINE_DIR = EVAL_DIR.parent / "pipeline"
CHUNKS_DIR   = PIPELINE_DIR / "data" / "chunks"
INDICES_DIR  = EVAL_DIR / "data" / "indices"

CHUNK_CONFIGS = {
    "baseline_buggy": CHUNKS_DIR / "chunks_baseline_buggy.jsonl",
    "400_80":         CHUNKS_DIR / "chunks.jsonl",
    "250_50":         CHUNKS_DIR / "chunks_250_50.jsonl",
}
# pre-computed OpenAI embeddings from the product pipeline, reused rather than re-billed
CACHED_OPENAI_EMBEDDED = {
    "baseline_buggy": CHUNKS_DIR / "chunks_baseline_buggy_embedded.jsonl",
}

BGE_MODEL = "BAAI/bge-small-en-v1.5"
OPENAI_MODEL = "text-embedding-3-large"
BGE_BATCH = 64
OPENAI_BATCH = 64


def load_chunks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def embed_bge(chunks: list[dict]) -> np.ndarray:
    """BGE encodes the corpus side with no instruction prefix — the query
    instruction ('Represent this sentence for searching relevant passages: ')
    is added only at query time, per the BGE model card's asymmetric-retrieval
    convention. Encoding here, not query time."""
    import torch
    torch.set_num_threads(os.cpu_count() or 4)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(BGE_MODEL)
    texts = [c["text"] for c in chunks]
    t0 = time.time()
    vecs = model.encode(texts, batch_size=BGE_BATCH, normalize_embeddings=True,
                         show_progress_bar=True, convert_to_numpy=True)
    print(f"bge-encoded {len(texts)} chunks in {time.time()-t0:.1f}s")
    return vecs.astype("float32")


def embed_openai_fresh(chunks: list[dict]) -> np.ndarray:
    from openai import OpenAI, RateLimitError
    load_dotenv(EVAL_DIR.parent / "backend" / ".env")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    vecs = []
    for i in range(0, len(chunks), OPENAI_BATCH):
        batch = chunks[i:i + OPENAI_BATCH]
        while True:
            try:
                resp = client.embeddings.create(model=OPENAI_MODEL, input=[c["text"] for c in batch])
                break
            except RateLimitError:
                print("\n  rate limited — waiting 20s...")
                time.sleep(20)
        vecs.extend(item.embedding for item in resp.data)
        print(f"  embedded {min(i+OPENAI_BATCH, len(chunks))}/{len(chunks)}", end="\r")
    print()
    arr = np.array(vecs, dtype="float32")
    faiss.normalize_L2(arr)
    return arr


def embed_openai_cached(config: str, chunks: list[dict]) -> np.ndarray:
    cached_path = CACHED_OPENAI_EMBEDDED[config]
    by_id = {}
    with cached_path.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            by_id[obj["chunk_id"]] = obj["embedding"]
    missing = [c["chunk_id"] for c in chunks if c["chunk_id"] not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} chunk_ids missing from cached embeddings "
                          f"(e.g. {missing[:3]}) — the chunk file and its cached "
                          f"embeddings are out of sync.")
    arr = np.array([by_id[c["chunk_id"]] for c in chunks], dtype="float32")
    faiss.normalize_L2(arr)
    return arr


def build_faiss_index(vecs: np.ndarray) -> faiss.Index:
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity via inner product on normalised vectors
    index.add(vecs)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, choices=CHUNK_CONFIGS)
    parser.add_argument("--embedder", required=True, choices=["bge", "openai"])
    args = parser.parse_args()

    chunks_path = CHUNK_CONFIGS[args.config]
    chunks = load_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    if args.embedder == "bge":
        vecs = embed_bge(chunks)
    else:
        if args.config in CACHED_OPENAI_EMBEDDED:
            print(f"Reusing cached OpenAI embeddings for '{args.config}' (no API cost)")
            vecs = embed_openai_cached(args.config, chunks)
        else:
            print(f"Calling OpenAI {OPENAI_MODEL} fresh for {len(chunks)} chunks "
                  f"(this costs real money — should only run for the winning config)")
            vecs = embed_openai_fresh(chunks)

    index = build_faiss_index(vecs)

    INDICES_DIR.mkdir(parents=True, exist_ok=True)
    index_path = INDICES_DIR / f"{args.config}__{args.embedder}.faiss"
    meta_path  = INDICES_DIR / f"{args.config}__{args.embedder}.meta.jsonl"

    faiss.write_index(index, str(index_path))
    with meta_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({
                "chunk_id":   c["chunk_id"],
                "chapter":    c["chapter"],
                "page_start": c["page_start"],
                "page_end":   c["page_end"],
            }, ensure_ascii=False) + "\n")

    print(f"Wrote index ({index.ntotal} vectors, dim={index.d}) → {index_path}")
    print(f"Wrote metadata → {meta_path}")


if __name__ == "__main__":
    main()
