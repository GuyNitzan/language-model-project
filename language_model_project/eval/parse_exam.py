"""
Parse every downloaded exam year (data/exams/<year>/{questions,answers,references}.pdf)
into one structured dataset: per-question stem + options + correct answer(s) +
resolved chunk_id(s) + a bucket label (text_answerable / image_dependent /
excluded_lever / unresolvable).

Builds directly on eval/audit_exams.py, which already established (and this file
reuses rather than reimplements):
  - per-file glyph-bug autodetection (`ð` on 2025/2026, `` on 2023/2024_09)
  - dual numbering convention ("N." vs the reversed ".N" on 2024_09)
  - crop-box-intersected image visibility (get_images() alone is wrong on 2024_09,
    whose content stream is shared across all 44 pages)
See derma_guide_plan.md, Phase 1, for the full audit these choices are based on.

Usage:
    pipeline/.venv/bin/python3 eval/parse_exam.py                  # all years
    pipeline/.venv/bin/python3 eval/parse_exam.py --year 2026      # one year
    pipeline/.venv/bin/python3 eval/parse_exam.py --skip-2023      # skip the
                                                                    # alignment-table year

Output (eval/data/exam_parsed/):
    <year>.jsonl   -- one question per line for that year
    all.jsonl      -- concatenation with a `year` field added
    summary.json   -- per-year bucket counts and parser-health diagnostics
"""
import argparse
import json
import re
import sys
from pathlib import Path

try:
    import fitz  # pymupdf
except ImportError:
    sys.exit("pymupdf missing -- run with pipeline/.venv/bin/python3")

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Reuse audited primitives rather than reimplementing glyph/cue/image-visibility logic.
from audit_exams import (
    HEB_LETTERS, CUES, BIG_PX,
    detect_glyph_bug, fix, squash,
)

ROOT = Path(__file__).resolve().parent.parent
EXAMS_DIR = ROOT / "data" / "exams"
CHUNKS_PATH = ROOT / "pipeline" / "data" / "chunks" / "chunks.jsonl"
OUT_DIR = Path(__file__).resolve().parent / "data" / "exam_parsed"

# A CONSTANT printed-page -> PDF-page offset is wrong past the first few hundred
# pages: verified directly (pdf 636 shows printed "501", not "596" as +40 would
# predict; pdf 3000 shows "2513", offset 487 not 40). The offset grows because
# extra pages accumulate deeper into this 2-volume book. Confirmed to matter in
# practice: 2026 Q4 (Bolognia ch.29, printed p.506) resolved to the WRONG
# chapter's chunks under the old constant-offset assumption. Real mapping is
# built once by eval/build_page_map.py (see PAGE_MAP_PATH) and cached to JSON.
PAGE_MAP_PATH = ROOT / "data" / "bolognia_page_map.json"
PAGE_TOLERANCE = 2


def _load_page_map():
    if not PAGE_MAP_PATH.exists():
        sys.exit(
            f"Missing {PAGE_MAP_PATH}. Run:\n"
            f"    pipeline/.venv/bin/python3 eval/build_page_map.py\n"
            f"first (one-time, ~9 min) -- a constant offset silently misresolves "
            f"references past the first few hundred book pages."
        )
    with PAGE_MAP_PATH.open(encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def printed_to_pdf_page(page_map: dict, printed: int) -> int:
    """Nearest-known-entry lookup with linear interpolation between the two
    closest mapped points -- the map isn't necessarily dense at every printed
    page (some pages have no recoverable header/footer number)."""
    if printed in page_map:
        return page_map[printed]
    keys = sorted(page_map)
    if not keys:
        raise ValueError("empty page map")
    if printed <= keys[0]:
        lo_k, hi_k = keys[0], keys[min(1, len(keys) - 1)]
    elif printed >= keys[-1]:
        lo_k, hi_k = keys[max(0, len(keys) - 2)], keys[-1]
    else:
        i = 0
        while keys[i] < printed:
            i += 1
        lo_k, hi_k = keys[i - 1], keys[i]
    if hi_k == lo_k:
        return page_map[lo_k]
    frac = (printed - lo_k) / (hi_k - lo_k)
    return round(page_map[lo_k] + frac * (page_map[hi_k] - page_map[lo_k]))

LEVER_WORD = "לבר"
BOLOGNIA_WORD = "בולוניה"

NUMBERING_PATTERNS = {
    "N.": r"(?m)^\s*(\d{1,3})\s*[.,:?]",
    ".N": r"(?m)^\s*[.,:?]\s*(\d{1,3})",
}

# 2023's answer key is a cross-version alignment table, not a plain key -- see
# audit_exams.audit_answers() and derma_guide_plan.md Phase 1b-2023.
ALIGNMENT_TABLE_YEARS = {"2023"}


# ---------------------------------------------------------------------------
# Chunk index
# ---------------------------------------------------------------------------

def load_chunk_index():
    chunks = []
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    by_chapter = {}
    for c in chunks:
        m = re.match(r"\s*(\d+)\s", c["chapter"])
        if m:
            by_chapter.setdefault(int(m.group(1)), []).append(c)
    return chunks, by_chapter


def resolve_pages_to_chunks(chunks, by_chapter, page_map, pages, chapter=None):
    """pages: printed book page numbers. Returns sorted chunk_ids whose page
    range overlaps any of them (in PDF-page space, via the real page map --
    see the note above), optionally scoped to a chapter when the reference
    format supplies one (2/7 years do)."""
    if not pages:
        return []
    pdf_pages = {printed_to_pdf_page(page_map, p) for p in pages}
    pool = by_chapter.get(chapter, []) if chapter else []
    if not pool:
        pool = chunks  # no chapter, or chapter not found -> search everything
    hits = set()
    for c in pool:
        lo, hi = c["page_start"] - PAGE_TOLERANCE, c["page_end"] + PAGE_TOLERANCE
        if any(lo <= pp <= hi for pp in pdf_pages):
            hits.add(c["chunk_id"])
    return sorted(hits)


# ---------------------------------------------------------------------------
# Questions parser
# ---------------------------------------------------------------------------

def _visible_content_images_by_page(doc):
    """Per-page list of (area, xref) for content-sized images, crop-intersected
    (see audit_exams.audit_questions -- required because 2024_09's content stream
    is shared across all 44 pages; a naive get_images() scan reports every image
    on every page). Each xref is attributed to only the FIRST page it is
    genuinely visible on, in document order, so a repeated/shared declaration
    is never double-counted.
    """
    by_page = {}
    seen_xrefs = set()
    for i, p in enumerate(doc):
        crop = p.rect
        for info in p.get_image_info(xrefs=True):
            xref = info.get("xref")
            if not xref or xref in seen_xrefs:
                continue
            bbox = fitz.Rect(info["bbox"])
            if not bbox.intersects(crop):
                continue
            area = bbox.width * bbox.height
            if area >= BIG_PX:
                by_page.setdefault(i, []).append((area, xref))
                seen_xrefs.add(xref)
    return by_page


def _split_options(span_text: str) -> tuple[dict, bool]:
    """Split a question's body into up to 4 options keyed by Hebrew letter.

    Option markers appear as EITHER "letter." or (bidi-reversed) ".letter" --
    confirmed both directions occur even within a single file (e.g. 2021's
    Q33 has ".א HERMANSKY..." while its own numbering convention is the
    non-reversed "N."; the reversal is a per-line bidi artifact, not a
    file-level toggle, so both forms must be tried everywhere).

    To avoid false hits from ordinary Hebrew words that happen to start with
    א/ב/ג/ד followed by punctuation, only accept letters in the expected
    sequence א,ב,ג,ד (skipping non-sequential matches rather than accepting
    them) -- the same "expected-next" repair technique used for numbering.
    """
    pat = re.compile(r"(?:([אבגד])\s*[.,:]|[.,:]\s*([אבגד]))")
    expected = list(HEB_LETTERS)  # א ב ג ד
    marks = []  # (letter, end_of_marker_char_index)
    next_i = 0
    for m in pat.finditer(span_text):
        letter = m.group(1) or m.group(2)
        if next_i < len(expected) and letter == expected[next_i]:
            marks.append((letter, m.end()))
            next_i += 1
        if next_i == len(expected):
            break

    if not marks:
        return {}, False

    options = {}
    for idx, (letter, end) in enumerate(marks):
        start_text = end
        stop_text = marks[idx + 1][1] - 2 if idx + 1 < len(marks) else len(span_text)
        # subtract ~2 to trim the next marker's own letter+punct off this option's tail;
        # exact trim isn't critical since downstream consumers strip whitespace anyway.
        options[letter] = span_text[start_text:stop_text].strip()
    return options, len(options) == 4


def parse_questions(path: Path) -> dict:
    """Returns {"questions": [...], "numbering_style": "N."|".N", "glyph_bug": str|None}."""
    doc = fitz.open(path)
    per_page_raw = [p.get_text() for p in doc]
    bad = detect_glyph_bug("".join(per_page_raw))
    per_page = [fix(t, bad) for t in per_page_raw]

    # Map an absolute offset in the concatenated fixed text back to a page index.
    page_starts = []
    acc = 0
    for t in per_page:
        page_starts.append(acc)
        acc += len(t)
    full_text = "".join(per_page)

    def page_of(offset: int) -> int:
        lo, hi = 0, len(page_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if page_starts[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo

    # Pick whichever numbering convention recovers the most matches -- see
    # audit_exams.audit_questions for why both must be tried (2024_09 needs ".N").
    best_style, best_matches = None, []
    for style, pat in NUMBERING_PATTERNS.items():
        seen, matches = set(), []
        for m in re.finditer(pat, full_text):
            n = int(m.group(1))
            if 1 <= n <= 200 and n not in seen:
                seen.add(n)
                matches.append((n, m.start(), m.end()))
        if len(matches) > len(best_matches):
            best_style, best_matches = style, matches
    best_matches.sort(key=lambda t: t[1])

    content_images = _visible_content_images_by_page(doc)

    questions = []
    for idx, (qnum, start, end) in enumerate(best_matches):
        span_end = best_matches[idx + 1][1] if idx + 1 < len(best_matches) else len(full_text)
        body = full_text[end:span_end]
        pg = page_of(start)

        options, complete = _split_options(body)
        # Stem is the text up to the first option marker.
        m0 = re.search(r"(?:[אבגד]\s*[.,:]|[.,:]\s*[אבגד])", body)
        stem = body[: m0.start()].strip() if m0 else body.strip()

        cue_hit = any(c in squash(stem) for c in CUES)
        page_images = content_images.get(pg, [])
        # Confidence: unambiguous only when this page has exactly one question
        # starting on it and exactly one content image -- otherwise "usually
        # one owns the image" (per the plan) is a page-level approximation,
        # not a per-question fact. Computed below in a second pass once we
        # know how many questions start on each page.
        questions.append({
            "q_num": qnum,
            "page": pg,
            "stem": stem,
            "options": options,
            "options_complete": complete,
            "visual_cue_in_stem": cue_hit,
            "page_has_content_image": bool(page_images),
        })

    # second pass: image-ownership confidence now that per-page question counts are known
    q_per_page = {}
    for q in questions:
        q_per_page.setdefault(q["page"], []).append(q)
    for pg, qs in q_per_page.items():
        n_imgs = len(content_images.get(pg, []))
        unambiguous = len(qs) == 1 and n_imgs == 1
        for q in qs:
            q["image_ownership_confidence"] = "unambiguous" if unambiguous else (
                "page_level" if q["page_has_content_image"] else "no_image")
            q["has_image"] = q["page_has_content_image"]
            del q["page_has_content_image"]

    return {"questions": questions, "numbering_style": best_style, "glyph_bug": bad}


# ---------------------------------------------------------------------------
# Answers parser
# ---------------------------------------------------------------------------

def parse_answers_simple(path: Path) -> dict:
    """Plain 'qnum + one-or-more letters' key. Returns {q_num: {letters...}}."""
    doc = fitz.open(path)
    raw = "".join(p.get_text() for p in doc)
    bad = detect_glyph_bug(raw)
    txt = fix(raw, bad)
    pat = re.compile(rf"(\d{{1,3}})\s*([{HEB_LETTERS}])((?:\s*[{HEB_LETTERS}])*)")
    answers = {}
    for m in pat.finditer(txt):
        q = int(m.group(1))
        if not (1 <= q <= 200) or q in answers:
            continue
        letters = {m.group(2)} | set(re.findall(rf"[{HEB_LETTERS}]", m.group(3) or ""))
        answers[q] = letters
    return answers


def parse_answers_alignment_table(path: Path, version: int = 1) -> dict:
    """2023-style cross-version report: each row is one MASTER question giving,
    per exam version, that question's own number + correct letter (the last
    version's number is omitted -- it equals the master number by convention).
    We only have `questions.pdf` for one version (1, per the user's selection),
    so we pull column `version` and re-key by ITS question numbers -- that's
    what lines up with the downloaded questions.pdf. See derma_guide_plan.md
    Phase 1b-2023 for the full worked example this is based on.
    """
    doc = fitz.open(path)
    raw = "".join(p.get_text() for p in doc)
    bad = detect_glyph_bug(raw)
    txt = fix(raw, bad)
    lines = [l.strip() for l in txt.split("\n")]

    entries, current, buf = [], None, []
    for line in lines:
        if line.isdigit() and (current is None or int(line) == current + 1):
            if current is not None:
                entries.append((current, buf))
            current, buf = int(line), []
        elif current is not None:
            buf.append(line)
    if current is not None:
        entries.append((current, buf))

    answers = {}
    for _master_q, body_lines in entries:
        body = " ".join(x for x in body_lines if x)
        pairs = re.findall(rf"\(([{HEB_LETTERS}])\)\s*(\d*)", body)
        if len(pairs) < version:
            continue
        letter, num = pairs[version - 1]
        if not num:
            continue  # last column's number is implied = master; we don't need that column
        answers[int(num)] = {letter}
    return answers


# ---------------------------------------------------------------------------
# References parser
# ---------------------------------------------------------------------------

def _split_ref_entries(txt: str, expected_max: int) -> dict:
    """Boundary-detect per-question entries. A boundary is a line that, once
    Hebrew-stripped-and-fixed, EQUALS the expected next number OR starts with
    it immediately followed by a non-digit (handles glued cases like
    "56עמוד-738" for expected 56, or "150לבר עמ '7" for expected 150 -- a
    plain `^\\d+$` line match silently misses these; confirmed on 2026's
    references.pdf, where it stopped at Q55 of 150)."""
    lines = txt.split("\n")
    entries, expected, buf, first_line_remainder = {}, 1, [], None

    def is_boundary(line: str, n: int) -> str | None:
        s = line.strip()
        if s == str(n):
            return ""
        if s.startswith(str(n)):
            rest = s[len(str(n)):]
            # A glued boundary marker is followed by real content -- a Hebrew
            # letter, in practice ("56עמוד-738", "150לבר..."). A bare "-digits"
            # continuation (e.g. "132-133") is a PAGE RANGE belonging to the
            # PREVIOUS entry, not a new boundary -- confirmed on 2022 Q131,
            # where "131\n132-133" is Q131's page range, not a Q132 marker.
            if rest and ("א" <= rest[0] <= "ת"):
                return rest
        return None

    for line in lines:
        if expected > expected_max:
            buf.append(line)
            continue
        rem = is_boundary(line, expected)
        if rem is not None:
            if expected > 1:
                entries[expected - 1] = " ".join(x for x in buf if x.strip())
                if first_line_remainder:
                    entries[expected - 1] = first_line_remainder + " " + entries[expected - 1]
            buf = []
            first_line_remainder = rem if rem else None
            expected += 1
        else:
            buf.append(line)
    if expected - 1 >= 1:
        entries[expected - 1] = " ".join(x for x in buf if x.strip())
        if first_line_remainder:
            entries[expected - 1] = first_line_remainder + " " + entries[expected - 1]
    return entries


def parse_references(path: Path, expected_max: int) -> dict:
    """Returns {q_num: {"book": "bolognia"|"lever"|"unknown", "chapter": int|None,
    "pages": [int...], "table_or_figure": bool, "raw": str}}."""
    doc = fitz.open(path)
    raw = "".join(p.get_text() for p in doc)
    bad = detect_glyph_bug(raw)
    txt = fix(raw, bad)
    entries = _split_ref_entries(txt, expected_max)

    out = {}
    for q, body in entries.items():
        table_or_figure = bool(re.search(r"טבלה|איור", body))
        # Strip table/figure pointers (e.g. "טבלה74.1") before extracting pages,
        # so a table/figure index number is never mistaken for a page number.
        stripped = re.sub(r"(?:טבלה|איור)\s*\d+(?:\.\d+)?", " ", body)
        chapter_m = re.search(r"פרק\s*(\d{1,3})", stripped)
        chapter = int(chapter_m.group(1)) if chapter_m else None
        stripped_no_chapter = re.sub(r"פרק\s*\d{1,3}", " ", stripped)

        pages = []
        for m in re.finditer(r"(\d{1,4})\s*-\s*(\d{1,4})", stripped_no_chapter):
            lo_s, hi_s = m.group(1), m.group(2)
            lo, hi = int(lo_s), int(hi_s)
            if 0 < hi - lo <= 30:
                pages.extend(range(lo, hi + 1))
            elif hi < lo and len(hi_s) < len(lo_s):
                # Hebrew citation convention abbreviates a range's end to just
                # its trailing digits -- "357-60" means pages 357-360, "635-8"
                # means 635-638 -- which looks like hi < lo until reconstructed
                # from lo's leading digits. Confirmed on real data: ~5% of
                # multi-page references (19/517 in the exam-dev set alone)
                # were silently corrupted into two disjoint, often
                # nonsensically low, "pages" (e.g. "357-60" -> [60, 357])
                # before this fix -- polluting resolve_reference()'s gold
                # chunk sets with an unrelated early-book/front-matter page.
                hi_full = int(lo_s[: len(lo_s) - len(hi_s)] + hi_s)
                if hi_full > lo:
                    pages.extend(range(lo, hi_full + 1))
                else:
                    # Reconstruction still doesn't make sense (e.g. "939-0",
                    # "2069-0" -- likely a source typo/OCR artifact rather
                    # than a recoverable abbreviation). Keep the one page we
                    # trust (lo) instead of inventing a bogus low page.
                    pages.append(lo)
            else:
                pages.extend([lo, hi])
            stripped_no_chapter = stripped_no_chapter.replace(m.group(0), " ", 1)
        pages.extend(int(n) for n in re.findall(r"\d{1,4}", stripped_no_chapter))

        if LEVER_WORD in body:
            book = "lever"
        elif BOLOGNIA_WORD in body:
            book = "bolognia"
        elif pages:
            book = "bolognia"  # tabular years (2021/2022) name the book only in the header
        else:
            book = "unknown"

        out[q] = {
            "book": book,
            "chapter": chapter,
            "pages": sorted(set(pages)),
            "table_or_figure": table_or_figure,
            "raw": body.strip(),
        }
    return out


# ---------------------------------------------------------------------------
# Assembly: bucket every question and resolve references to chunks
# ---------------------------------------------------------------------------

def resolve_reference(chunks, by_chapter, page_map, ref: dict | None) -> list:
    """Chunk IDs for one reference entry. Falls back to every chunk in the
    named chapter when a page is missing but the chapter is known (seen on
    2026 Q33/Q34: "פרק52" with no page at all) -- coarser than a page match,
    but still a real, non-fabricated signal rather than an empty result."""
    if not ref or ref["book"] != "bolognia":
        return []
    if ref["pages"]:
        return resolve_pages_to_chunks(chunks, by_chapter, page_map, ref["pages"], ref["chapter"])
    if ref["chapter"] and ref["chapter"] in by_chapter:
        return sorted(c["chunk_id"] for c in by_chapter[ref["chapter"]])
    return []


def build_year(year_dir: Path, chunks, by_chapter, page_map, skip_2023: bool = False) -> tuple[list, dict]:
    year = year_dir.name
    q_result = parse_questions(year_dir / "questions.pdf")
    questions = q_result["questions"]
    expected_max = max((q["q_num"] for q in questions), default=0)

    if year in ALIGNMENT_TABLE_YEARS:
        if skip_2023:
            return [], {"year": year, "skipped": "alignment_table_year"}
        answers = parse_answers_alignment_table(year_dir / "answers.pdf", version=1)
    else:
        answers = parse_answers_simple(year_dir / "answers.pdf")

    refs = parse_references(year_dir / "references.pdf", expected_max=expected_max)

    records, buckets = [], {"text_answerable": 0, "image_dependent": 0,
                             "excluded_lever": 0, "unresolvable": 0}
    for q in questions:
        qn = q["q_num"]
        ref = refs.get(qn)
        correct = sorted(answers.get(qn, set()))

        chunk_ids = resolve_reference(chunks, by_chapter, page_map, ref)
        if ref and ref["book"] == "lever":
            bucket, chunk_ids = "excluded_lever", []
        elif q["has_image"] and q["visual_cue_in_stem"]:
            bucket = "image_dependent"
        elif chunk_ids:
            bucket = "text_answerable"
        else:
            bucket = "unresolvable"
        buckets[bucket] += 1

        records.append({
            "year": year,
            "q_num": qn,
            "stem": q["stem"],
            "options": q["options"],
            "options_complete": q["options_complete"],
            "correct": correct,
            "has_multi_answer": len(correct) > 1,
            "page": q["page"],
            "has_image": q["has_image"],
            "image_ownership_confidence": q["image_ownership_confidence"],
            "visual_cue_in_stem": q["visual_cue_in_stem"],
            "reference": ref,
            "chunk_ids": chunk_ids,
            "bucket": bucket,
        })

    diag = {
        "year": year,
        "n_questions": len(questions),
        "numbering_style": q_result["numbering_style"],
        "glyph_bug": q_result["glyph_bug"],
        "options_complete": sum(q["options_complete"] for q in questions),
        "n_answers_parsed": len(answers),
        "n_refs_parsed": len(refs),
        "buckets": buckets,
    }
    return records, diag


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", help="parse only this year (e.g. 2026)")
    ap.add_argument("--skip-2023", action="store_true",
                     help="skip 2023's cross-version alignment-table year")
    args = ap.parse_args()

    if not EXAMS_DIR.exists():
        sys.exit(f"No exam data found under {EXAMS_DIR}")
    years = [args.year] if args.year else sorted(p.name for p in EXAMS_DIR.iterdir() if p.is_dir())

    chunks, by_chapter = load_chunk_index()
    page_map = _load_page_map()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_records, summary = [], {}
    for year in years:
        year_dir = EXAMS_DIR / year
        if not year_dir.is_dir():
            print(f"skip {year}: not found", file=sys.stderr)
            continue
        records, diag = build_year(year_dir, chunks, by_chapter, page_map, skip_2023=args.skip_2023)
        summary[year] = diag
        if not records:
            print(f"{year}: {diag.get('skipped', 'no records')}")
            continue
        all_records.extend(records)
        with (OUT_DIR / f"{year}.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        b = diag["buckets"]
        print(f"{year:9} n={diag['n_questions']:3d}  "
              f"text={b['text_answerable']:3d}  image={b['image_dependent']:3d}  "
              f"lever={b['excluded_lever']:2d}  unresolved={b['unresolvable']:2d}  "
              f"options_complete={diag['options_complete']}/{diag['n_questions']}")

    with (OUT_DIR / "all.jsonl").open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nTotal: {len(all_records)} questions written to {OUT_DIR}/all.jsonl")


if __name__ == "__main__":
    main()
