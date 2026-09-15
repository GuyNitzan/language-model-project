"""
Step 2: Chunk extracted pages into overlapping text segments.

Reads data/extracted/pages.jsonl (one page per line) and produces
data/chunks/chunks.jsonl (one chunk per line), each with full metadata.

Strategy (fixed — see derma_guide_plan.md Phase 2a for the bug this replaces):
  - Per chapter, flatten all pages into a single word stream, tagging each
    word with the page it came from.
  - Slide a fixed-size window of TARGET_WORDS words over that stream,
    advancing by (TARGET_WORDS - OVERLAP_WORDS) words each step. This is
    the part the old page-accumulation version got wrong: a Bolognia page
    averages ~700 words, well above the 400-word target, so the old loop's
    very first page already overshot the target and closed the window —
    every chunk ended up ~1 page long with ~0 words of true overlap between
    consecutive chunks. Chunking at word granularity instead of page
    granularity fixes both problems at once.
  - Never split a chunk across different chapters (chapter boundaries reset
    the stream).
  - Each chunk carries: chapter, section, subsection, page range, text.
    section/subsection/subsubsection/toc_path are taken from whichever page
    the chunk's first word came from — a chunk that happens to straddle a
    section boundary is labelled with the earlier section. This is the same
    approximation the original chunker made and is a known, acceptable
    caveat (documented in the plan) rather than something this fix changes.

(target, overlap) are CLI-configurable so the chunk-size sweep in Phase 3
is one command per variant, not a code change.
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

IN_PATH  = Path("data/extracted/pages.jsonl")
OUT_PATH = Path("data/chunks/chunks.jsonl")

# Defaults — ~500 tokens per chunk with ~100 tokens of overlap for most
# English medical text.
TARGET_WORDS = 400
OVERLAP_WORDS = 80


def word_count(text: str) -> int:
    return len(text.split())


def build_word_stream(pages: list[dict]) -> list[tuple[str, dict]]:
    """Flatten a sorted list of same-chapter pages into (word, page) pairs."""
    stream = []
    for page in pages:
        for word in page["text"].split():
            stream.append((word, page))
    return stream


def stream_to_chunks(stream: list[tuple[str, dict]], target: int, overlap: int) -> list[dict]:
    """Slide a fixed-size window of `target` words over the stream,
    advancing by (target - overlap) words each step."""
    if not stream:
        return []

    step = max(1, target - overlap)
    chunks = []
    i = 0
    n = len(stream)
    while i < n:
        window = stream[i : i + target]
        words = [w for w, _ in window]
        pages_in_window = [p for _, p in window]
        first_page = pages_in_window[0]
        page_nums = [p["page"] for p in pages_in_window]

        chunks.append({
            "chapter":        first_page["chapter"],
            "section":        first_page["section"],
            "subsection":     first_page["subsection"],
            "subsubsection":  first_page["subsubsection"],
            "toc_path":       first_page["toc_path"],
            "page_start":     min(page_nums),
            "page_end":       max(page_nums),
            "text":           " ".join(words),
        })

        if i + target >= n:
            break
        i += step

    return chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=TARGET_WORDS,
                         help=f"target words per chunk (default {TARGET_WORDS})")
    parser.add_argument("--overlap", type=int, default=OVERLAP_WORDS,
                         help=f"overlap words between consecutive chunks (default {OVERLAP_WORDS})")
    parser.add_argument("--in", dest="in_path", type=Path, default=IN_PATH)
    parser.add_argument("--out", dest="out_path", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    if args.overlap >= args.target:
        parser.error("--overlap must be smaller than --target")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    pages: list[dict] = []
    with args.in_path.open(encoding="utf-8") as f:
        for line in f:
            pages.append(json.loads(line))

    print(f"Loaded {len(pages)} pages.")

    # Group pages by chapter to avoid cross-chapter chunks
    chapters: dict[str, list[dict]] = {}
    for p in pages:
        key = p["chapter"] or "__front_matter__"
        chapters.setdefault(key, []).append(p)

    total_chunks = 0
    with args.out_path.open("w", encoding="utf-8") as f:
        for chapter_title, chapter_pages in tqdm(chapters.items(), desc="Chunking"):
            chapter_pages.sort(key=lambda p: p["page"])
            stream = build_word_stream(chapter_pages)
            for idx, chunk in enumerate(stream_to_chunks(stream, args.target, args.overlap)):
                chunk["chunk_id"] = f"{chapter_title[:40].replace(' ', '_')}_{idx:04d}"
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"\nDone. Produced {total_chunks} chunks (target={args.target}, "
          f"overlap={args.overlap}) → {args.out_path}")


if __name__ == "__main__":
    main()
