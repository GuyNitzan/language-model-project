"""
Step 1: Extract structured text from the Dermatology PDF.

For each page, extracts body text and associates it with the TOC hierarchy
(chapter, section, subsection, etc.) derived from the embedded TOC.

Output: data/extracted/pages.jsonl  — one JSON object per page.
"""

import json
import re
from pathlib import Path

import fitz  # pymupdf
from tqdm import tqdm

PDF_PATH = Path("../Bolognia Dermatology 5th 2024.pdf")
OUT_PATH = Path("data/extracted/pages.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Font size thresholds (observed from PDF inspection)
MIN_BODY_SIZE = 7.5       # below this → diagram labels, skip
CAPTION_FONTS = {"MyriadPro-Bold", "MyriadPro-Semibold", "MyriadPro-Regular",
                 "MyriadPro-It"}
CAPTION_MAX_SIZE = 9.0    # captions use ~8.5pt Myriad fonts
BODY_FONT = "Kuenstler480BT-Roman"
BODY_SIZE = 9.0


def build_toc_index(toc: list) -> list[dict]:
    """
    Convert the flat TOC list into a sorted list of context snapshots.
    Each entry records the page where a TOC node starts and the full
    hierarchy path at each level at that point.
    """
    # current state: level -> title
    state: dict[int, str] = {}
    entries = []
    for level, title, page in toc:
        state[level] = title
        # clear deeper levels when a shallower heading starts
        for deeper in list(state.keys()):
            if deeper > level:
                del state[deeper]
        entries.append({
            "page": page,
            "level": level,
            "title": title,
            "path": [state.get(l, "") for l in range(1, max(state.keys()) + 1)],
        })
    return entries


def get_context_for_page(toc_entries: list[dict], page_num: int) -> dict:
    """Return the deepest TOC context active at `page_num` (1-indexed)."""
    context = {}
    for entry in toc_entries:
        if entry["page"] <= page_num:
            context = entry
        else:
            break
    return context


def is_figure_caption(text: str) -> bool:
    return bool(re.match(r"^Fig\.\s+\d", text) or re.match(r"^Table\s+\d", text))


def is_page_header(text: str, page_num: int) -> bool:
    """Skip running headers: chapter number, section labels at top of page."""
    stripped = text.strip()
    if re.fullmatch(r"\d{1,4}", stripped):
        return True
    if stripped in ("SECTION", "CHAPTER", "C HAPT ER", "SE C TI ON"):
        return True
    return False


def extract_page_text(page: fitz.Page) -> str:
    """Extract meaningful body text from a page, skipping noise."""
    blocks = page.get_text("dict")["blocks"]
    lines_out = []

    for block in blocks:
        if block["type"] != 0:  # skip image blocks
            continue

        block_lines = []
        for line in block["lines"]:
            spans = line["spans"]
            if not spans:
                continue

            # Representative span (first non-empty)
            rep = next((s for s in spans if s["text"].strip()), None)
            if rep is None:
                continue

            size = rep["size"]
            font = rep["font"]
            text = " ".join(s["text"] for s in spans).strip()

            if not text or text == "­":   # soft hyphen placeholders
                continue

            # Skip diagram labels and very small decorative text
            if size < MIN_BODY_SIZE:
                continue

            # Skip figure/table captions
            if is_figure_caption(text):
                continue

            # Skip running page headers (chapter number, section label)
            if is_page_header(text, page.number + 1):
                continue

            block_lines.append(text)

        if block_lines:
            lines_out.append(" ".join(block_lines))

    return "\n".join(lines_out)


def main():
    doc = fitz.open(PDF_PATH)
    toc = doc.get_toc()
    toc_entries = build_toc_index(toc)

    # Find first real content page (skip front matter before page 30)
    first_content_page = 30

    written = 0
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for page_num in tqdm(range(first_content_page, len(doc) + 1), desc="Extracting pages"):
            page = doc[page_num - 1]
            text = extract_page_text(page)

            if not text.strip():
                continue

            ctx = get_context_for_page(toc_entries, page_num)
            path = ctx.get("path", [])

            record = {
                "page": page_num,
                "toc_path": path,                          # full hierarchy list
                "chapter": path[0] if len(path) > 0 else "",
                "section": path[1] if len(path) > 1 else "",
                "subsection": path[2] if len(path) > 2 else "",
                "subsubsection": path[3] if len(path) > 3 else "",
                "text": text,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"\nDone. Extracted {written} pages → {OUT_PATH}")


if __name__ == "__main__":
    main()
