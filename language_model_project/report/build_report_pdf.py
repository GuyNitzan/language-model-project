"""
Renders report/report.md to report/report.pdf with reportlab (pure Python,
no headless-browser dependency -- marp-cli/puppeteer needed a ~200MB Chrome
download that didn't survive a sandbox restart; this avoids that entirely).

report.md is the source of truth; this is a small, scoped Markdown subset
parser (H1/H2, tables, one fenced code block, **bold**/*italic*/`code`
inline spans) -- not a general Markdown renderer, just enough for this one
file's actual content.

Reportlab's default Helvetica renders em-dash/en-dash/arrows/math symbols
fine, but not Hebrew glyphs (renders as tofu boxes) -- checked directly.
report.md uses 4 bare Hebrew option-letters (א/ד) in one sentence about MCQ
position bias; those are transliterated to their Latin option-letter
equivalents (א->A, ד->D) for this PDF rendering only, not in report.md
itself.

Usage:
    uv run --with reportlab python3 report/build_report_pdf.py
"""
import re
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted,
)

REPORT_DIR = Path(__file__).resolve().parent
SRC = REPORT_DIR / "report.md"
OUT = REPORT_DIR / "report.pdf"

HEBREW_TO_LATIN = {"א": "A", "ב": "B", "ג": "C", "ד": "D"}
# Box-drawing arrows (U+2500 horizontal + U+25B6 triangle) in the ASCII
# architecture diagram render as solid boxes under reportlab's Courier
# (unlike the plain U+2192 arrow elsewhere, which renders fine) -- checked
# directly. The plain arrow "→" is intentionally left alone.
BOX_DRAWING_TO_ASCII = {"──▶": "-->", "──": "--", "▶": ">"}


def transliterate(text: str) -> str:
    for heb, lat in HEBREW_TO_LATIN.items():
        text = text.replace(heb, lat)
    for box, ascii_ in BOX_DRAWING_TO_ASCII.items():
        text = text.replace(box, ascii_)
    return text


def inline(text: str) -> str:
    """**bold** / *italic* / `code` -> reportlab's mini-XML tags. Escape
    real XML special chars first so e.g. a literal '&' in prose doesn't
    break the parser."""
    text = transliterate(text)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", text)
    text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", text)
    return text


def parse_table(lines: list[str]) -> list[list[str]]:
    rows = []
    for line in lines:
        if re.match(r"^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?$", line):
            continue  # header-separator row
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    return rows


def build_flowables(md_text: str, styles: dict) -> list:
    flow = []
    blocks = re.split(r"\n\s*\n", md_text.strip())
    i = 0
    while i < len(blocks):
        block = blocks[i]
        lines = block.split("\n")

        if block.startswith("```"):
            # a fenced code block may span multiple blank-line-separated
            # blocks if it happens to contain a blank line; merge forward
            # until the closing fence is found.
            code_lines = lines[1:]
            while not (code_lines and code_lines[-1].strip() == "```"):
                i += 1
                code_lines.append("")
                code_lines.extend(blocks[i].split("\n"))
            code_lines = code_lines[:-1]  # drop the closing ``` line
            flow.append(Preformatted(transliterate("\n".join(code_lines)), styles["code"]))
            flow.append(Spacer(1, 10))

        elif lines[0].startswith("# "):
            flow.append(Paragraph(inline(lines[0][2:]), styles["h1"]))
            flow.append(Spacer(1, 4))

        elif lines[0].startswith("## "):
            flow.append(Spacer(1, 8))
            flow.append(Paragraph(inline(lines[0][3:]), styles["h2"]))
            flow.append(Spacer(1, 4))

        elif lines[0].startswith("|"):
            rows = parse_table(lines)
            n_cols = max(len(r) for r in rows)
            for r in rows:
                while len(r) < n_cols:
                    r.append("")
            table_data = [[Paragraph(inline(c), styles["cell"]) for c in row] for row in rows]
            col_width = 6.6 * inch / n_cols
            t = Table(table_data, colWidths=[col_width] * n_cols)
            t.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c9c7c0")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0efec")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            flow.append(t)
            flow.append(Spacer(1, 10))

        elif lines[0].startswith("*") and lines[0].endswith("*") and len(lines) == 1:
            flow.append(Paragraph(inline(lines[0].strip("*")), styles["subtitle"]))
            flow.append(Spacer(1, 6))

        else:
            para = " ".join(l.strip() for l in lines)
            flow.append(Paragraph(inline(para), styles["body"]))
            flow.append(Spacer(1, 6))

        i += 1
    return flow


def make_styles():
    return {
        "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=17, leading=21,
                              spaceAfter=2, textColor=colors.HexColor("#0b0b0b")),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12.5, leading=15,
                              textColor=colors.HexColor("#0b0b0b")),
        "subtitle": ParagraphStyle("subtitle", fontName="Helvetica-Oblique", fontSize=10,
                                    textColor=colors.HexColor("#52514e")),
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.3, leading=12.6,
                                alignment=4, textColor=colors.HexColor("#0b0b0b")),  # 4 = justify
        "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=8.5, leading=10.5,
                                textColor=colors.HexColor("#0b0b0b")),
        "code": ParagraphStyle("code", fontName="Courier", fontSize=7.6, leading=9.2,
                                textColor=colors.HexColor("#0b0b0b"),
                                backColor=colors.HexColor("#f0efec")),
    }


def main():
    md_text = SRC.read_text(encoding="utf-8")
    styles = make_styles()
    flow = build_flowables(md_text, styles)
    doc = SimpleDocTemplate(
        str(OUT), pagesize=LETTER,
        topMargin=0.65 * inch, bottomMargin=0.65 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        title="DermaForce: RAG + QLoRA for Hebrew Dermatology Board-Exam QA",
    )
    doc.build(flow)
    n_pages = _count_pages(OUT)
    print(f"wrote {OUT} ({n_pages} pages)")


def _count_pages(path: Path) -> int:
    data = path.read_bytes()
    return data.count(b"/Type /Page") - data.count(b"/Type /Pages")


if __name__ == "__main__":
    main()
