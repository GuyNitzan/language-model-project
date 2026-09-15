"""
Build a printed-book-page -> Bolognia-PDF-page lookup table, anchored on
chapter-start pages from the PDF's own table of contents.

Why not a constant offset: verified directly that the offset grows across the
volume (pdf 636 shows printed "501", not "676" as a from-the-front +40 would
give at that depth; pdf 3000 shows "2513"). A reference citing a page deep in
the book resolves to the WRONG chapter's chunks under a constant-offset model
-- confirmed on 2026 Q4 (chapter 29, printed p.506, which +40 sends into
chapter 24's page range instead).

Why not scan every page for a running header/footer number either: this book
mixes real body pages with table-of-contents pages (which list OTHER
chapters' page numbers, not the current page's own), chapter/section-number
labels ("38" next to "C HAPT ER"), and "eFigure/eTable" online-supplement
pages numbered "640.e1" etc. A naive full-book scan seeds on this noise and
diverges -- confirmed: a first attempt produced a single garbage entry from a
front-matter false start, and a second (with continuity-based seeding) drifted
just as badly starting from page 1 of a 3233-page, 9-minute scan.

The fix: anchor ONLY on chapter-start pages (~160 of them, from doc.get_toc()
-- the same authoritative source pipeline/01_extract.py already uses to tag
chunks.jsonl's chapter field). A chapter's opening pages are real body
content, never TOC/e-supplement noise, and probing up to 2 pages past each
chapter start for an UNAMBIGUOUS single bare-digit candidate reliably
recovers the great majority of them. Runs in ~30s, not ~9min, because it only
reads ~350 pages instead of 3233.

Usage: pipeline/.venv/bin/python3 eval/build_page_map.py
"""
import json, re, sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = ROOT / "Bolognia Dermatology 5th 2024.pdf"
OUT_PATH = ROOT / "data" / "bolognia_page_map.json"


def chapter_start_pages(doc) -> dict:
    """{chapter_num: first_pdf_page} from the embedded TOC -- same source
    pipeline/01_extract.py uses to tag chunks.jsonl's `chapter` field."""
    starts = {}
    for level, title, pdf_page in doc.get_toc():
        if level != 1:
            continue
        m = re.match(r"\s*(\d+)", title)
        if m:
            ch = int(m.group(1))
            starts.setdefault(ch, pdf_page)  # keep first occurrence
    return starts


def printed_number_on(doc, pdf_page_1indexed: int):
    """Bare 1-4 digit candidates among the first/last 3 non-empty lines --
    where a running header/footer number would appear. Only trusted when
    exactly one candidate is found (ambiguous pages are skipped, not guessed)."""
    lines = [l.strip() for l in doc[pdf_page_1indexed - 1].get_text().split("\n") if l.strip()]
    return [int(l) for l in (lines[:3] + lines[-3:]) if re.fullmatch(r"\d{1,4}", l)]


def main():
    doc = fitz.open(PDF_PATH)
    ch_starts = chapter_start_pages(doc)
    print(f"chapters found in TOC: {len(ch_starts)}")

    mapping, misses = {}, []
    for ch, pdf_pg in sorted(ch_starts.items()):
        got = None
        # a chapter's very first page often has no running header (title-page
        # layout); probe a couple of pages in and back-compute pdf_pg's own
        # printed number from whichever probe gives an unambiguous single hit.
        for offset in (0, 1, 2):
            cands = printed_number_on(doc, pdf_pg + offset)
            if len(cands) == 1:
                got = cands[0] - offset
                break
        if got:
            mapping[str(got)] = pdf_pg
        else:
            misses.append(ch)

    print(f"anchors: {len(mapping)}  unresolved chapters: {len(misses)} {misses[:15]}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(mapping, f)
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
