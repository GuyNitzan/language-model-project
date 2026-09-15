"""
Audit every downloaded exam year for the format variation the parser must handle.

Usage:  pipeline/.venv/bin/python3 eval/audit_exams.py
Expects: data/exams/<year>/{questions,answers,references}.pdf
"""
import re, sys, json
from pathlib import Path

try:
    import fitz  # pymupdf
except ImportError:
    sys.exit("pymupdf missing. Run:  pipeline/.venv/bin/python3 eval/audit_exams.py")

ROOT = Path(__file__).resolve().parent.parent
EXAMS = ROOT / "data" / "exams"

HEB_LETTERS = "אבגד"                      # A B C D
# Different export pipelines corrupt a different single glyph (always onto "נ", nun,
# confirmed by word-context: "מהðכון"/"מה\uf8ffכון" both read as "מה נכון" = "what is correct").
# Detect per-file rather than hardcoding one bad codepoint.
GLYPH_CANDIDATES = ["ð", "\uf8ff"]
CUES = ["תמונ", "בתמו", "צילום", "איור", "המוצג", "המודגם", "היסטולוג"]
BIG_PX = 40_000                            # above this, an image is plausibly clinical


def detect_glyph_bug(raw: str) -> str | None:
    """Return whichever candidate codepoint is actually present and frequent."""
    counts = {c: raw.count(c) for c in GLYPH_CANDIDATES}
    bad = max(counts, key=counts.get)
    return bad if counts[bad] > 20 else None


def fix(s: str, bad: str | None = None) -> str:
    bad = bad if bad is not None else detect_glyph_bug(s)
    return s.replace(bad, "נ") if bad else s


def squash(s: str) -> str:
    """Strip ALL whitespace: RTL extraction splits words across line breaks."""
    return re.sub(r"\s+", "", s)


def odd_chars(s: str) -> dict:
    """Non-Hebrew, non-ASCII leftovers -> candidate glyph-map bugs."""
    keep = set("μϰαβγ–—’‘ְֱֲֳִֵֶַָֹֻּֽ")
    c = {}
    for ch in s:
        if ch.isascii() or "֐" <= ch <= "׿" or ch in keep or ch.isspace():
            continue
        c[ch] = c.get(ch, 0) + 1
    return dict(sorted(c.items(), key=lambda kv: -kv[1])[:6])


def audit_questions(path: Path) -> dict:
    """Single get_text() pass per page. Image dims read from the get_images tuple
    (idx 2,3) -- decoding every image costs ~15s and isn't needed for dims.

    Two format variants confirmed across years, tried in order, best (closest to
    150 recovered, ties by count) kept:
      - "N ." / "N ?." / "N ,." / "N :."  -- number then punctuation (2021/22/23/25/26)
      - ".N"                              -- punctuation then number, reversed (2024_09)
    Also: multi-digit numbers can render digit-split across a line break inside
    the number itself (e.g. "30" -> "0\n3"); the \s* between digit groups below
    absorbs that for the common 2-digit case.
    """
    d = fitz.open(path)
    per_page = []
    # xref -> list of (page_index, bbox) for placements that are actually VISIBLE
    # (i.e. the bbox intersects that page's own crop rect). Confirmed necessary on
    # 2024_09: get_images(full=True) reports every image on every page there because
    # its content stream/resource dict is shared across the whole document -- an
    # xref's raw bbox can sit at y=-14000 or y=+9000 on a normal 0..842 page, meaning
    # it is declared but never drawn there. Naively counting "repeated xref" as
    # decorative on that basis produced images_content=0 for a file that in fact has
    # 41 real, unique clinical images (each verified visible on 1-3 real pages).
    visible = {}
    for i, p in enumerate(d):
        per_page.append(p.get_text())
        crop = p.rect
        for info in p.get_image_info(xrefs=True):
            xref = info.get("xref")
            bbox = fitz.Rect(info["bbox"])
            if xref and bbox.intersects(crop):
                visible.setdefault(xref, []).append((i, bbox.width * bbox.height))

    raw = "".join(per_page)
    bad = detect_glyph_bug(raw)
    fixed = fix(raw, bad)

    # Bug history: an earlier version of this pattern capped every count at 99
    # because it only captured 2 digit-groups -- 3-digit numbers (100-150) were
    # silently truncated. \d{1,3} is the correct contiguous form; it misses the
    # rare case where a 2-digit number is split across a line break inside the
    # number itself (e.g. "30" -> "0\n3", seen 4x in 2021) -- that residual is
    # cheap enough to leave as a documented manual/Phase-1 concern rather than
    # a regex, since fixing it generically requires expected-next-number repair.
    patterns = {
        "N.":  r"(?m)^\s*(\d{1,3})\s*[.,:?]",   # digits then punctuation
        ".N":  r"(?m)^\s*[.,:?]\s*(\d{1,3})",   # punctuation then digits (2024_09 style)
    }
    best_name, best_nums = None, set()
    for name, pat in patterns.items():
        nums = {int(m.group(1)) for m in re.finditer(pat, fixed) if 1 <= int(m.group(1)) <= 200}
        if len(nums) > len(best_nums):
            best_name, best_nums = name, nums

    # A visible placement counts as "content" if it is plausibly a clinical image
    # (>=BIG_PX at its largest occurrence), not a small inline icon/bullet.
    content_pages = set()
    for xref, placements in visible.items():
        if max(area for _, area in placements) >= BIG_PX:
            content_pages.update(pg for pg, _ in placements)
    big_pages = content_pages

    cue_pages = {i for i, t in enumerate(per_page) if any(c in squash(fix(t, bad)) for c in CUES)}

    qnums = sorted(best_nums)
    return {
        "pages": d.page_count,
        "chars": len(raw),
        "glyph_bug_char": repr(bad) if bad else None,
        "glyph_hits": raw.count(bad) if bad else 0,
        "odd_chars_after_fix": odd_chars(fixed),
        "numbering_style": best_name,
        "images_distinct_xref_visible": len(visible),
        "images_content": sum(1 for xref, pl in visible.items()
                               if max(a for _, a in pl) >= BIG_PX),
        "pages_big_image": len(big_pages),
        "pages_visual_cue": len(cue_pages),
        "pages_both": len(big_pages & cue_pages),
        "q_numbers_found": len(qnums),
        "q_range": (min(qnums), max(qnums)) if qnums else None,
        "q_gaps": [n for n in range(1, (max(qnums) if qnums else 0) + 1)
                   if n not in set(qnums)][:12],
    }


def audit_answers(path: Path) -> dict:
    d = fitz.open(path)
    txt = "".join(p.get_text() for p in d)
    bad = detect_glyph_bug(txt)
    txt = fix(txt, bad)

    # A single mention is just exam-title metadata ("...118 - 1 גירסה - 2022...");
    # a real alignment table repeats the column header once per version (seen: 20x
    # for a real 4-version table in 2023, vs 1x for 2022's plain single-version key).
    n_version_word = len(re.findall(r"גירסה|גרסה", txt))
    if n_version_word >= 4:
        # Cross-version alignment table (seen in 2023): for each MASTER question
        # number, lists that question's number + correct letter *within each of
        # several exam versions* -- not a direct "qnum -> letter" key. Needs a
        # dedicated parser keyed on which version corresponds to the downloaded
        # questions.pdf. Flagged, not parsed here.
        n_versions = len(re.findall(r"גירסה", txt))
        return {
            "pages": d.page_count,
            "schema": "cross_version_alignment_table",
            "version_mentions": n_versions,
            "note": "needs per-version parser -- see plan Phase 1 risk note",
        }

    pat = re.compile(rf"(\d{{1,3}})\s*([{HEB_LETTERS}])((?:\s*[{HEB_LETTERS}])*)")
    keys, multi = {}, {}
    for m in pat.finditer(txt):
        q = int(m.group(1))
        if not (1 <= q <= 200) or q in keys:
            continue
        letters = [m.group(2)] + re.findall(rf"[{HEB_LETTERS}]", m.group(3) or "")
        keys[q] = letters
        if len(letters) > 1:
            multi[q] = letters
    return {
        "pages": d.page_count,
        "schema": "simple_key",
        "answers_parsed": len(keys),
        "multi_answer_qs": {k: "".join(v) for k, v in sorted(multi.items())},
        "letters_used": "".join(sorted({l for v in keys.values() for l in v})),
        "max_q": max(keys) if keys else None,
    }


def audit_references(path: Path) -> dict:
    d = fitz.open(path)
    txt = fix("".join(p.get_text() for p in d))
    flat = re.sub(r"[ \t]+", " ", txt)
    books = {
        "bolognia(בולוניה)": len(re.findall("בולוניה", flat)),
        "lever(לבר)": len(re.findall("לבר", flat)),
    }
    return {
        "pages": d.page_count,
        "book_mentions": books,
        "chapter_tokens(פרק)": len(re.findall(r"פרק\s*\d+", flat)),
        "page_tokens(עמ)": len(re.findall(r"עמ|עמוד", flat)),
        "table_or_figure(טבלה/איור)": len(re.findall(r"טבלה|איור", flat)),
    }


def main():
    years = sorted(p for p in EXAMS.iterdir() if p.is_dir()) if EXAMS.exists() else []
    if not years:
        sys.exit(f"No exam folders under {EXAMS}")

    report = {}
    for y in years:
        entry = {}
        for kind, fn in (("questions", audit_questions),
                         ("answers", audit_answers),
                         ("references", audit_references)):
            f = y / f"{kind}.pdf"
            entry[kind] = fn(f) if f.exists() else {"MISSING": True}
        report[y.name] = entry

    print(json.dumps(report, ensure_ascii=False, indent=2))

    print("\n" + "=" * 92)
    print(f"{'year':8} {'Qs':>4} {'style':>5} {'imgs':>5} {'both':>5} {'ans':>4} {'schema':>10} {'multi':>5} {'refs:ch':>8} {'glyph':>6}")
    print("=" * 92)
    for yr, e in report.items():
        q, a, r = e["questions"], e["answers"], e["references"]
        if q.get("MISSING"):
            print(f"{yr:8} -- questions.pdf missing --")
            continue
        ans_str = a.get("answers_parsed", "n/a") if a.get("schema") == "simple_key" else "ALIGN-TBL"
        multi = len(a.get("multi_answer_qs", {})) if a.get("schema") == "simple_key" else "-"
        print(f"{yr:8} {q['q_numbers_found']:>4} {str(q['numbering_style']):>5} {q['images_content']:>5} "
              f"{q['pages_both']:>5} {str(ans_str):>4} {a.get('schema','?'):>10} {str(multi):>5} "
              f"{r.get('chapter_tokens(פרק)', 0):>8} {q.get('glyph_hits',0):>6}")
    print("=" * 92)
    print("Qs=question numbers found · style=N. or .N numbering convention detected")
    print("imgs=content images after xref-dedup (decorative/repeated assets excluded)")
    print("both=pages with content image AND visual cue · ans=answer-key entries (simple_key schema only)")
    print("schema=answer-key format · multi=questions accepting >1 answer")
    print("refs:ch=chapter tokens in references · glyph=count of the detected broken-glyph char")
    print("\nRed flags: Qs != 150 (or != 100 for a known 100-Q exam) · q_gaps non-empty ·")
    print("           odd_chars_after_fix non-trivial · schema=ALIGN-TBL (needs bespoke parser)")


if __name__ == "__main__":
    main()
