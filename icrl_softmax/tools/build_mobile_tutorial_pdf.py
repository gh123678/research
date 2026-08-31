"""Build the 9:16 Chinese mobile tutorial PDF with ReportLab.

The source stays editable as Markdown.  This builder intentionally supports only the
small Markdown subset used by the tutorial: headings, paragraphs, lists, display
math, page breaks, inline emphasis/code, and three labelled teaching callouts.
"""

from __future__ import annotations

import hashlib
import html
import re
import shutil
from pathlib import Path

from matplotlib import mathtext
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import portrait
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "论文_草稿" / "softmax_Q控制_手机教学版.md"
OUTPUT = ROOT / "output" / "pdf" / "softmax-q-control-mobile-tutorial.pdf"
SYNC_OUTPUT = ROOT / "论文_草稿" / "softmax_Q控制_手机教学版.pdf"
FORMULA_DIR = ROOT / "tmp" / "pdfs" / "mobile_formulae"

PAGE_SIZE = portrait((108 * mm, 192 * mm))
PAGE_WIDTH, PAGE_HEIGHT = PAGE_SIZE
LEFT = RIGHT = 9.5 * mm
TOP = 11.5 * mm
BOTTOM = 12 * mm
CONTENT_WIDTH = PAGE_WIDTH - LEFT - RIGHT

INK = colors.HexColor("#172033")
BLUE = colors.HexColor("#2457A7")
BLUE_DARK = colors.HexColor("#163D7A")
BLUE_LIGHT = colors.HexColor("#EAF2FF")
GREY_LIGHT = colors.HexColor("#F2F4F7")
AMBER_LIGHT = colors.HexColor("#FFF5DC")
WHITE = colors.white


def find_font(*names: str) -> Path:
    font_dir = Path("C:/Windows/Fonts")
    for name in names:
        path = font_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"None of the requested fonts exist: {names}")


def register_fonts() -> None:
    regular = find_font("msyh.ttc", "msyh.ttf", "simhei.ttf")
    bold = find_font("msyhbd.ttc", "msyhbd.ttf", "msyh.ttc", "simhei.ttf")
    pdfmetrics.registerFont(TTFont("MobileSans", str(regular), subfontIndex=0))
    pdfmetrics.registerFont(TTFont("MobileSansBold", str(bold), subfontIndex=0))
    pdfmetrics.registerFontFamily(
        "MobileSans",
        normal="MobileSans",
        bold="MobileSansBold",
        italic="MobileSans",
        boldItalic="MobileSansBold",
    )


def build_styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    body = ParagraphStyle(
        "MobileBody",
        parent=sample["BodyText"],
        fontName="MobileSans",
        fontSize=13,
        leading=18.6,
        textColor=INK,
        alignment=TA_LEFT,
        wordWrap="CJK",
        spaceAfter=6.2,
        allowWidows=0,
        allowOrphans=0,
    )
    return {
        "body": body,
        "cover_title": ParagraphStyle(
            "CoverTitle",
            parent=body,
            fontName="MobileSansBold",
            fontSize=25,
            leading=33,
            textColor=BLUE_DARK,
            alignment=TA_CENTER,
            spaceAfter=13,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle",
            parent=body,
            fontName="MobileSansBold",
            fontSize=16,
            leading=23,
            textColor=BLUE,
            alignment=TA_CENTER,
            spaceAfter=16,
        ),
        "h2": ParagraphStyle(
            "MobileH2",
            parent=body,
            fontName="MobileSansBold",
            fontSize=19,
            leading=25,
            textColor=BLUE_DARK,
            spaceBefore=2,
            spaceAfter=10,
            keepWithNext=1,
        ),
        "h3": ParagraphStyle(
            "MobileH3",
            parent=body,
            fontName="MobileSansBold",
            fontSize=15,
            leading=21,
            textColor=BLUE,
            spaceBefore=5,
            spaceAfter=6,
            keepWithNext=1,
        ),
        "callout": ParagraphStyle(
            "MobileCallout",
            parent=body,
            fontSize=12.1,
            leading=17.4,
            spaceAfter=0,
        ),
        "list": ParagraphStyle(
            "MobileList",
            parent=body,
            fontSize=12.6,
            leading=18,
            leftIndent=0,
            firstLineIndent=0,
            spaceAfter=2,
        ),
        "footer": ParagraphStyle(
            "MobileFooter",
            parent=body,
            fontSize=8.2,
            leading=9,
            textColor=colors.HexColor("#677083"),
        ),
    }


def inline_markup(text: str) -> str:
    escaped = html.escape(text, quote=False)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(
        r"`([^`]+)`",
        r'<font name="MobileSansBold" color="#2457A7">\1</font>',
        escaped,
    )
    return escaped


def normalise_formula(formula: str) -> str:
    formula = " ".join(line.strip() for line in formula.splitlines())
    formula = formula.replace(r"\operatorname{softmax}", r"\mathrm{softmax}")
    formula = formula.replace(r"\mathbb E", r"\mathbb{E}")
    return formula


def formula_image(formula: str) -> Image:
    FORMULA_DIR.mkdir(parents=True, exist_ok=True)
    formula = normalise_formula(formula)
    digest = hashlib.sha256(formula.encode("utf-8")).hexdigest()[:16]
    path = FORMULA_DIR / f"formula-{digest}.png"
    if not path.exists():
        mathtext.math_to_image(
            f"${formula}$",
            str(path),
            dpi=300,
            format="png",
            color="#172033",
        )
    with PILImage.open(path) as bitmap:
        width_px, height_px = bitmap.size
    width_pt = width_px / 300 * 72
    height_pt = height_px / 300 * 72
    target_width = min(width_pt, CONTENT_WIDTH)
    scale = target_width / width_pt
    target_height = height_pt * scale
    image = Image(str(path), width=target_width, height=target_height)
    image.hAlign = "CENTER"
    image.spaceBefore = 4
    image.spaceAfter = 7
    return image


def callout(text: str, marker: str, styles: dict[str, ParagraphStyle]) -> Table:
    labels = {
        "WHY": ("为什么这样做", BLUE_LIGHT, BLUE),
        "MISUNDERSTANDING": ("最容易误解", AMBER_LIGHT, colors.HexColor("#A86A00")),
        "DEFENSE": ("答辩表述", GREY_LIGHT, BLUE_DARK),
    }
    label, background, accent = labels[marker]
    paragraph = Paragraph(
        f'<font name="MobileSansBold" color="{accent.hexval()}">{label}</font><br/>'
        f"{inline_markup(text)}",
        styles["callout"],
    )
    table = Table([[paragraph]], colWidths=[CONTENT_WIDTH])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.7, accent),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    table.spaceBefore = 4
    table.spaceAfter = 8
    return table


class MobileDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, styles: dict[str, ParagraphStyle]) -> None:
        super().__init__(
            filename,
            pagesize=PAGE_SIZE,
            leftMargin=LEFT,
            rightMargin=RIGHT,
            topMargin=TOP,
            bottomMargin=BOTTOM,
            title="从 Softmax TD 评估到两阶段 Q 控制",
            author="Anonymous Authors",
            subject="9:16 手机科研教学讲义",
        )
        self._section = "核心问题"
        self._styles = styles
        frame = Frame(
            LEFT,
            BOTTOM,
            CONTENT_WIDTH,
            PAGE_HEIGHT - TOP - BOTTOM,
            id="mobile-frame",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates(
            PageTemplate(
                id="mobile",
                frames=[frame],
                onPage=self._draw_page_background,
                onPageEnd=self._draw_footer,
            )
        )

    def afterFlowable(self, flowable) -> None:  # noqa: N802 - ReportLab API
        section = getattr(flowable, "mobile_section", None)
        if section:
            self._section = section

    def _draw_page_background(self, canvas, _doc) -> None:
        canvas.saveState()
        canvas.setFillColor(BLUE)
        canvas.rect(0, PAGE_HEIGHT - 3.2 * mm, PAGE_WIDTH, 3.2 * mm, fill=1, stroke=0)
        canvas.restoreState()

    def _draw_footer(self, canvas, doc) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D6DBE5"))
        canvas.setLineWidth(0.4)
        canvas.line(LEFT, 8.6 * mm, PAGE_WIDTH - RIGHT, 8.6 * mm)
        canvas.setFont("MobileSans", 7.8)
        canvas.setFillColor(colors.HexColor("#677083"))
        section = self._section
        if len(section) > 18:
            section = section[:18] + "..."
        canvas.drawString(LEFT, 5.2 * mm, section)
        canvas.drawRightString(PAGE_WIDTH - RIGHT, 5.2 * mm, str(doc.page))
        canvas.restoreState()


def flush_paragraph(buffer: list[str], story: list, styles: dict[str, ParagraphStyle]) -> None:
    if not buffer:
        return
    text = " ".join(part.strip() for part in buffer if part.strip())
    if text:
        story.append(Paragraph(inline_markup(text), styles["body"]))
    buffer.clear()


def flush_list(
    items: list[str],
    numbered: bool,
    story: list,
    styles: dict[str, ParagraphStyle],
) -> None:
    if not items:
        return
    list_items = [
        ListItem(Paragraph(inline_markup(item), styles["list"]), leftIndent=6)
        for item in items
    ]
    list_kwargs = {
        "bulletType": "1" if numbered else "bullet",
        "leftIndent": 17,
        "bulletFontName": "MobileSansBold",
        "bulletFontSize": 10,
        "bulletColor": BLUE,
        "spaceAfter": 7,
    }
    if numbered:
        list_kwargs["start"] = "1"
    story.append(ListFlowable(list_items, **list_kwargs))
    items.clear()


def parse_markdown(source: str, styles: dict[str, ParagraphStyle]) -> list:
    lines = source.splitlines()
    story: list = [Spacer(1, 26 * mm)]
    paragraph_buffer: list[str] = []
    list_buffer: list[str] = []
    list_numbered = False
    first_h1 = True
    first_h2 = True
    index = 0

    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()

        if stripped == "$$":
            flush_paragraph(paragraph_buffer, story, styles)
            flush_list(list_buffer, list_numbered, story, styles)
            formula_lines: list[str] = []
            index += 1
            while index < len(lines) and lines[index].strip() != "$$":
                formula_lines.append(lines[index])
                index += 1
            story.append(formula_image("\n".join(formula_lines)))
        elif stripped == "<!-- PAGE -->":
            flush_paragraph(paragraph_buffer, story, styles)
            flush_list(list_buffer, list_numbered, story, styles)
            story.append(PageBreak())
        elif stripped.startswith("# "):
            flush_paragraph(paragraph_buffer, story, styles)
            flush_list(list_buffer, list_numbered, story, styles)
            heading = Paragraph(inline_markup(stripped[2:]), styles["cover_title"])
            story.append(heading)
            first_h1 = False
        elif stripped.startswith("## "):
            flush_paragraph(paragraph_buffer, story, styles)
            flush_list(list_buffer, list_numbered, story, styles)
            text = stripped[3:]
            if first_h2:
                heading = Paragraph(inline_markup(text), styles["cover_subtitle"])
                first_h2 = False
            else:
                heading = Paragraph(inline_markup(text), styles["h2"])
                heading.mobile_section = re.sub(r"^\d+\.\s*", "", text)
            story.append(heading)
        elif stripped.startswith("### "):
            flush_paragraph(paragraph_buffer, story, styles)
            flush_list(list_buffer, list_numbered, story, styles)
            story.append(Paragraph(inline_markup(stripped[4:]), styles["h3"]))
        elif stripped.startswith("> ["):
            flush_paragraph(paragraph_buffer, story, styles)
            flush_list(list_buffer, list_numbered, story, styles)
            match = re.match(r"> \[(WHY|MISUNDERSTANDING|DEFENSE)\]\s*(.*)", stripped)
            if not match:
                raise ValueError(f"Unsupported callout syntax: {stripped}")
            story.append(callout(match.group(2), match.group(1), styles))
        else:
            bullet_match = re.match(r"^-\s+(.*)", stripped)
            number_match = re.match(r"^\d+\.\s+(.*)", stripped)
            if bullet_match or number_match:
                flush_paragraph(paragraph_buffer, story, styles)
                current_numbered = bool(number_match)
                if list_buffer and current_numbered != list_numbered:
                    flush_list(list_buffer, list_numbered, story, styles)
                list_numbered = current_numbered
                list_buffer.append((number_match or bullet_match).group(1))
            elif not stripped:
                flush_paragraph(paragraph_buffer, story, styles)
                flush_list(list_buffer, list_numbered, story, styles)
            else:
                flush_list(list_buffer, list_numbered, story, styles)
                paragraph_buffer.append(stripped)
        index += 1

    flush_paragraph(paragraph_buffer, story, styles)
    flush_list(list_buffer, list_numbered, story, styles)
    return story


def main() -> None:
    register_fonts()
    styles = build_styles()
    source = SOURCE.read_text(encoding="utf-8")
    story = parse_markdown(source, styles)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = MobileDocTemplate(str(OUTPUT), styles)
    doc.build(story)
    shutil.copy2(OUTPUT, SYNC_OUTPUT)
    print(f"BUILT {OUTPUT}")
    print(f"SYNCED {SYNC_OUTPUT}")


if __name__ == "__main__":
    main()
