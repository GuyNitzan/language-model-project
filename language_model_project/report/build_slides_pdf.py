"""
Renders report/slides.md to report/slides.pdf with matplotlib (one page per
slide, landscape) -- avoids marp-cli/puppeteer entirely, which needs a
~200MB headless-Chrome download that didn't survive a sandbox restart last
attempt. slides.md (Marp-formatted markdown) stays the source of truth for
anyone who later has a working marp-cli; this is a scoped renderer for this
project's actual slide content only (bullets, **bold**/*italic*/`code`
inline spans, one embedded image per slide, numbered lists) -- not a general
Marp implementation.

Usage:
    uv run --with matplotlib python3 report/build_slides_pdf.py
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

REPORT_DIR = Path(__file__).resolve().parent
SRC = REPORT_DIR / "slides.md"
OUT = REPORT_DIR / "slides.pdf"

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_SECONDARY, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
FIG_W, FIG_H = 11.0, 6.2  # 16:9-ish landscape

plt.rcParams.update({"font.size": 15, "figure.facecolor": SURFACE})

SEGMENT_RE = re.compile(r"\*\*(.+?)\*\*|`(.+?)`|\*(.+?)\*")
# matplotlib has no bidi text-shaping engine: a single bare Hebrew letter
# embedded in an LTR sentence (slides.md's position-bias slide: "option א
# correct...") renders with broken advance/positioning -- checked directly
# ("option אcorrect", the letter visually detached from its intended
# neighbour). Same fix as report/build_report_pdf.py: transliterate to the
# Latin option-letter equivalent for this rendering only, not in slides.md.
HEBREW_TO_LATIN = {"א": "A", "ב": "B", "ג": "C", "ד": "D"}


def split_segments(text: str) -> list[tuple[str, str]]:
    """Splits one line into (text, style) runs; style in normal/bold/code/italic."""
    for heb, lat in HEBREW_TO_LATIN.items():
        text = text.replace(heb, lat)
    segments, pos = [], 0
    for m in SEGMENT_RE.finditer(text):
        if m.start() > pos:
            segments.append((text[pos:m.start()], "normal"))
        if m.group(1) is not None:
            segments.append((m.group(1), "bold"))
        elif m.group(2) is not None:
            segments.append((m.group(2), "code"))
        else:
            segments.append((m.group(3), "italic"))
        pos = m.end()
    if pos < len(text):
        segments.append((text[pos:], "normal"))
    return segments or [("", "normal")]


def draw_rich_line(fig, renderer, x, y, segments, fontsize, color) -> float:
    """Draws left-aligned styled text runs starting at figure-fraction (x, y);
    returns the y (still in figure fraction) for the next line, using measured
    glyph widths so bold/code/italic runs don't overlap or leave gaps."""
    cur_x = x
    for text, style in segments:
        if not text:
            continue
        kwargs = dict(fontsize=fontsize, color=color, va="top", ha="left",
                       transform=fig.transFigure)
        if style == "bold":
            kwargs["fontweight"] = "bold"
        elif style == "italic":
            kwargs["style"] = "italic"
        elif style == "code":
            kwargs["family"] = "monospace"
            kwargs["fontsize"] = fontsize * 0.92
        t = fig.text(cur_x, y, text, **kwargs)
        fig.canvas.draw()  # forces layout so get_window_extent below is accurate
        bbox = t.get_window_extent(renderer=renderer)
        cur_x += bbox.width / fig.get_size_inches()[0] / fig.dpi
        if style == "code":
            # Checked directly: a `code`-styled run immediately followed by
            # another segment occasionally renders with its trailing space
            # collapsed in the final PDF (not reproducible in an isolated
            # PNG render of the same two strings -- a monospace-run-specific
            # bbox/advance mismatch between the Agg measurement pass and
            # PdfPages' actual vector text layout). A small fixed pad after
            # every code segment sidesteps it without stretching ordinary
            # normal-to-normal word gaps elsewhere on the line.
            cur_x += 0.006
    return cur_x


def wrap_segments(segments: list[tuple[str, str]], max_chars: int) -> list[list[tuple[str, str]]]:
    """Greedy word-wrap across styled segments, wrapping on whitespace inside
    a segment while keeping each word's own style tag. Each space is baked
    into the end of the preceding word rather than kept as its own token --
    a standalone whitespace-only Text object gets a near-zero-width bbox in
    matplotlib (checked directly: it silently ate the space between a
    `code` span and the following word), so spaces must always ride along
    with visible ink."""
    words = []
    for text, style in segments:
        parts = text.split(" ")
        for j, p in enumerate(parts):
            if p == "" and j == 0:
                if words:  # a leading space in this segment -> stick it onto the previous word
                    prev_text, prev_style = words[-1]
                    words[-1] = (prev_text + " ", prev_style)
                continue
            suffix = " " if j < len(parts) - 1 else ""
            if p or suffix:
                words.append((p + suffix, style))
    lines, cur, cur_len = [], [], 0
    for w, style in words:
        wl = len(w)
        if cur_len + wl > max_chars and cur:
            lines.append(cur)
            cur, cur_len = [], 0
        if not (w.isspace() and not cur):  # don't start a line with whitespace
            cur.append((w, style))
            cur_len += wl
    if cur:
        lines.append(cur)
    return lines


def render_slide(pdf, index, raw: str):
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(SURFACE)
    renderer = fig.canvas.get_renderer()

    lines = raw.strip("\n").split("\n")
    y = 0.90
    left = 0.07
    max_chars = 95

    # Title: first line starting with # or ##
    body_start = 0
    if lines and lines[0].startswith("#"):
        title_text = lines[0].lstrip("#").strip()
        is_h1 = lines[0].startswith("# ") and not lines[0].startswith("## ")
        title_fontsize = 30 if is_h1 else 24
        # A long H2 title can run past the right margin at the default size
        # (checked directly: "Two artifacts caught before being reported as
        # findings" touched the page edge) -- shrink it to fit rather than
        # let it clip or overflow the figure.
        max_title_w = 1.0 - left - 0.04
        for _ in range(6):
            t = fig.text(left, y, title_text, fontsize=title_fontsize, fontweight="bold",
                          color=INK, va="top", ha="left", transform=fig.transFigure)
            fig.canvas.draw()
            w = t.get_window_extent(renderer=renderer).width / fig.get_size_inches()[0] / fig.dpi
            if w <= max_title_w:
                break
            t.remove()
            title_fontsize -= 2
        y -= 0.16 if is_h1 else 0.13
        body_start = 1
        # a title-slide subtitle line (### ...) right under an H1
        if is_h1 and body_start < len(lines) and lines[body_start].startswith("###"):
            fig.text(left, y, lines[body_start].lstrip("#").strip(), fontsize=17,
                      color=INK_SECONDARY, va="top", ha="left", transform=fig.transFigure)
            y -= 0.09
            body_start += 1

    image_path = None
    body_lines = []
    in_code = False
    table_buf = []

    def flush_table():
        if table_buf:
            body_lines.append(("__TABLE__", list(table_buf)))
            table_buf.clear()

    for line in lines[body_start:]:
        if line.strip().startswith("```"):
            # Fenced blocks are rendered once, separately, by
            # extract_fenced_block() below -- skip their lines here entirely
            # rather than also emitting them as ordinary body text.
            in_code = not in_code
            continue
        if in_code:
            continue
        if line.strip().startswith("|"):
            table_buf.append(line.strip())
            continue
        flush_table()
        m = re.match(r"!\[.*?\]\((.+?)\)", line.strip())
        if m:
            image_path = (REPORT_DIR / m.group(1)).resolve()
            body_lines.append(None)  # placeholder marks image position
        else:
            body_lines.append(line)
    flush_table()

    # Pre-pass: an image slide's text below it can run long enough to overflow
    # the page (checked directly -- the table/figure slide's 4-line caption
    # under a fixed 0.46 image fraction ran off the bottom edge). Count the
    # non-image lines' wrapped height first and size the image to whatever
    # vertical space is actually left, rather than a fixed fraction.
    text_height = 0.0
    for line in body_lines:
        if line is None:
            continue
        if isinstance(line, tuple) and line[0] == "__TABLE__":
            text_height += 0.072 * len(parse_table_rows(line[1])) + 0.05
            continue
        stripped = line.strip()
        if stripped == "":
            text_height += 0.035
            continue
        content = stripped[2:] if stripped.startswith(("- ", "* ")) else stripped
        content = re.sub(r"^\d+\.\s", "", content)
        n_wrapped = len(wrap_segments(split_segments(content), max_chars))
        text_height += 0.075 * n_wrapped
    available = y - 0.06 - text_height - 0.05  # 0.06 bottom margin, 0.05 image/text gap
    default_img_h_frac = 0.46 if image_path else 0.0
    img_h_frac_budget = max(0.20, min(default_img_h_frac, available)) if image_path else 0.0

    # Second safety net: a text-only (no-image) slide can independently
    # overflow if it just has a lot of content (checked directly -- the
    # verifier-agent slide's last line ran under the page-number footer).
    # Shrinking the image above doesn't help here since there isn't one;
    # compress line/blank spacing instead, floored so it never gets
    # cramped enough to be unreadable.
    remaining_for_text = y - 0.04 - (img_h_frac_budget + 0.05 if image_path else 0.0)
    line_scale = max(0.72, min(1.0, remaining_for_text / text_height)) if text_height > 0 else 1.0
    line_h, blank_h = 0.075 * line_scale, 0.035 * line_scale

    for line in body_lines:
        if line is None:
            if image_path and image_path.exists():
                img = mpimg.imread(image_path)
                h, w = img.shape[0], img.shape[1]
                img_h_frac = img_h_frac_budget
                img_w_frac = img_h_frac * (w / h) * (FIG_H / FIG_W)
                ax = fig.add_axes([0.5 - img_w_frac / 2, y - img_h_frac, img_w_frac, img_h_frac])
                ax.imshow(img)
                ax.axis("off")
                y -= img_h_frac + 0.05
            continue
        if isinstance(line, tuple) and line[0] == "__TABLE__":
            y = draw_table(fig, line[1], left, y)
            continue
        stripped = line.strip()
        if stripped == "":
            y -= blank_h
            continue
        bullet = ""
        indent = left
        content = stripped
        if stripped.startswith("- ") or stripped.startswith("* "):
            bullet, content, indent = "•  ", stripped[2:], left + 0.02
        elif re.match(r"^\d+\.\s", stripped):
            num, rest = stripped.split(".", 1)
            bullet, content, indent = f"{num}.  ", rest.strip(), left + 0.02
        elif stripped.startswith("```") :
            continue  # fenced code handled as a block below (system-design slide)

        segments = split_segments(content)
        for wrapped in wrap_segments(segments, max_chars):
            if bullet:
                fig.text(indent - 0.025, y, bullet, fontsize=15, color=BLUE,
                          va="top", ha="left", transform=fig.transFigure)
            draw_rich_line(fig, renderer, indent, y, wrapped, 15, INK)
            y -= line_h
            bullet = ""  # only the first wrapped line gets the bullet glyph

    # fenced ASCII-diagram block (System design slide) rendered verbatim, monospace
    code_block = extract_fenced_block(raw)
    if code_block:
        fig.text(0.5, y - 0.02, code_block, fontsize=11.5, family="monospace",
                  color=INK_SECONDARY, va="top", ha="center", transform=fig.transFigure)

    fig.text(0.965, 0.03, str(index), fontsize=10, color=INK_SECONDARY,
              va="bottom", ha="right", transform=fig.transFigure)
    pdf.savefig(fig)
    plt.close(fig)


def parse_table_rows(raw_lines: list[str]) -> list[list[str]]:
    rows = []
    for line in raw_lines:
        if re.match(r"^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?$", line):
            continue  # the header-separator row
        rows.append([c.strip() for c in line.strip("|").split("|")])
    return rows


def draw_table(fig, raw_lines: list[str], left: float, y: float) -> float:
    """Renders one Markdown table as a matplotlib grid. Cell text with
    **bold** markers is stripped and re-applied as real font weight after
    the table is built (mpl table cells don't parse inline markup)."""
    rows = parse_table_rows(raw_lines)
    n_rows, n_cols = len(rows), max(len(r) for r in rows)
    for r in rows:
        while len(r) < n_cols:
            r.append("")
    bold_mask = [[bool(re.search(r"\*\*(.+?)\*\*", c)) for c in row] for row in rows]
    clean_rows = [[re.sub(r"\*\*(.+?)\*\*", r"\1", c) for c in row] for row in rows]

    row_h = 0.072
    table_h = row_h * n_rows
    table_w = 0.86
    ax = fig.add_axes([left, y - table_h, table_w, table_h])
    ax.axis("off")
    tbl = ax.table(cellText=clean_rows, loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_fontsize(13)
        cell.set_edgecolor("#c9c7c0")
        cell.PAD = 0.02
        if r == 0:
            cell.set_facecolor("#f0efec")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(SURFACE)
        if bold_mask[r][c]:
            cell.get_text().set_fontweight("bold")
        cell.get_text().set_color(INK)
    return y - table_h - 0.05


def extract_fenced_block(raw: str) -> str | None:
    m = re.search(r"```\n(.*?)```", raw, re.S)
    return m.group(1).rstrip("\n") if m else None


def main():
    text = SRC.read_text(encoding="utf-8")
    # drop the marp front-matter (--- ... ---) before splitting real slides
    text = re.sub(r"^---\n.*?\n---\n", "", text, count=1, flags=re.S)
    slides = [s for s in text.split("\n---\n")]
    with PdfPages(OUT) as pdf:
        for i, slide in enumerate(slides, start=1):
            if slide.strip():
                render_slide(pdf, i, slide)
    print(f"wrote {OUT} ({len(slides)} slides)")


if __name__ == "__main__":
    main()
