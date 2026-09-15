"""
Step 4: Ingest embedded chunks into Qdrant vector database.

Reads  data/chunks/chunks_embedded.jsonl
Upserts every chunk as a Qdrant point with:
  - vector:  the 3072-dim OpenAI embedding
  - payload: all metadata fields (chapter, section, page range, text, etc.)

Collection is created if it doesn't exist.
Safe to re-run: uses chunk_id as the point ID (upsert semantics).
"""

import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from tqdm import tqdm

load_dotenv()

IN_PATH         = Path("data/chunks/chunks_embedded.jsonl")
COLLECTION_NAME = "dermatology_book"
VECTOR_SIZE     = 3072   # text-embedding-3-large
BATCH_SIZE      = 100


def chunk_id_to_int(chunk_id: str) -> int:
    """Convert string chunk_id to a stable integer for Qdrant point ID."""
    return int(hashlib.md5(chunk_id.encode()).hexdigest()[:15], 16)


def main():
    qdrant_url    = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")  # None for local

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    # Create collection if needed
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"Created collection '{COLLECTION_NAME}'")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists — upserting.")

    chunks: list[dict] = []
    with IN_PATH.open(encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"Loaded {len(chunks)} embedded chunks.")

    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Upserting"):
        batch = chunks[i : i + BATCH_SIZE]
        points = []
        for c in batch:
            embedding = c.pop("embedding")
            points.append(
                PointStruct(
                    id=chunk_id_to_int(c["chunk_id"]),
                    vector=embedding,
                    payload=c,   # everything else becomes searchable metadata
                )
            )
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"\nDone. {len(chunks)} chunks ingested into '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
