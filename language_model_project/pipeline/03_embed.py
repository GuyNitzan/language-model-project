"""
Step 3: Generate embeddings for every chunk.

Reads  data/chunks/chunks.jsonl
Writes data/chunks/chunks_embedded.jsonl  (same records + "embedding" field)

Uses OpenAI text-embedding-3-large (3072 dims).
Batches requests to stay within API rate limits.
Skips chunks that already have embeddings (resumable).
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

IN_PATH  = Path("data/chunks/chunks.jsonl")
OUT_PATH = Path("data/chunks/chunks_embedded.jsonl")

EMBED_MODEL = "text-embedding-3-large"
BATCH_SIZE  = 64      # chunks per API call
SLEEP_ON_RATE = 10    # seconds to wait on rate-limit error


def load_done_ids(path: Path) -> set[str]:
    """Return chunk_ids already written (for resuming)."""
    done = set()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if "chunk_id" in obj and "embedding" in obj:
                    done.add(obj["chunk_id"])
    return done


def embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    while True:
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
            return [item.embedding for item in resp.data]
        except Exception as e:
            if "rate" in str(e).lower():
                print(f"\nRate limited — waiting {SLEEP_ON_RATE}s...")
                time.sleep(SLEEP_ON_RATE)
            else:
                raise


def main():
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    chunks: list[dict] = []
    with IN_PATH.open(encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"Loaded {len(chunks)} chunks.")

    done_ids = load_done_ids(OUT_PATH)
    pending  = [c for c in chunks if c["chunk_id"] not in done_ids]
    print(f"Already embedded: {len(done_ids)}  |  Pending: {len(pending)}")

    with OUT_PATH.open("a", encoding="utf-8") as f:
        for i in tqdm(range(0, len(pending), BATCH_SIZE), desc="Embedding"):
            batch = pending[i : i + BATCH_SIZE]
            texts = [c["text"] for c in batch]
            embeddings = embed_batch(client, texts)
            for chunk, emb in zip(batch, embeddings):
                chunk["embedding"] = emb
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nDone. Embeddings written → {OUT_PATH}")


if __name__ == "__main__":
    main()
