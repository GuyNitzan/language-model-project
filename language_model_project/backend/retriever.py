"""Semantic search against the Qdrant vector store."""

from openai import OpenAI
from qdrant_client import QdrantClient

import config

_openai  = OpenAI(api_key=config.OPENAI_API_KEY)
_qdrant  = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)


def embed_query(text: str) -> list[float]:
    resp = _openai.embeddings.create(model=config.EMBED_MODEL, input=text)
    return resp.data[0].embedding


def retrieve(query: str, top_k: int = config.TOP_K) -> list[dict]:
    """Return top_k chunks most relevant to `query`."""
    vector = embed_query(query)
    results = _qdrant.query_points(
        collection_name=config.COLLECTION_NAME,
        query=vector,
        limit=top_k,
        with_payload=True,
    ).points
    chunks = []
    for hit in results:
        p = hit.payload
        chunks.append({
            "score":         round(hit.score, 4),
            "chunk_id":      p.get("chunk_id", ""),
            "chapter":       p.get("chapter", ""),
            "section":       p.get("section", ""),
            "subsection":    p.get("subsection", ""),
            "page_start":    p.get("page_start"),
            "page_end":      p.get("page_end"),
            "toc_path":      p.get("toc_path", []),
            "text":          p.get("text", ""),
        })
    return chunks
