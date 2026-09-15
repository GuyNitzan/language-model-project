#!/usr/bin/env bash
# Run the full data pipeline end-to-end.
# Steps 3 and 4 require .env to be populated (see .env.example).
set -e

echo "=== Step 1: Extract text from PDF ==="
uv run python3 01_extract.py

echo ""
echo "=== Step 2: Chunk pages ==="
uv run python3 02_chunk.py

echo ""
echo "=== Step 3: Generate embeddings ==="
uv run python3 03_embed.py

echo ""
echo "=== Step 4: Ingest into Qdrant ==="
uv run python3 04_ingest.py

echo ""
echo "Pipeline complete."
