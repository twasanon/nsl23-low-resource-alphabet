#!/usr/bin/env python3
"""Build the reader-friendly NSL23 paper companion as a polished DOCX.

The document is intentionally derived from audited repository artifacts rather
than from manually copied screenshots or untraceable summaries.  It uses the
Documents skill's ``compact_reference_guide`` preset and an editorial-cover
first page.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_ALIGN_VERTICAL, WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "NSL23_Paper_Explained.docx"
FIGURES = ROOT / "artifacts" / "figures"
V2_PROTOCOL = json.loads((ROOT / "configs" / "v2_protocol.json").read_text(encoding="utf-8"))

# compact_reference_guide tokens
PAGE_WIDTH_IN = 8.5
PAGE_HEIGHT_IN = 11.0
MARGIN_IN = 1.0
CONTENT_WIDTH_IN = 6.5
TABLE_WIDTH_DXA = 9360
TABLE_INDENT_DXA = 120
CELL_MARGIN_TOP_DXA = 80
CELL_MARGIN_BOTTOM_DXA = 80
CELL_MARGIN_START_DXA = 120
CELL_MARGIN_END_DXA = 120

FONT_BODY = "Calibri"
FONT_MATH = "Cambria Math"
FONT_CODE = "Aptos Mono"
BLUE = "2E74B5"
DARK_BLUE = "1F4D78"
INK = "0B2545"
MUTED = "5D6875"
PALE_BLUE = "E8EEF5"
PALE_GRAY = "F2F4F7"
CALLOUT = "F4F6F9"
PALE_GOLD = "FFF8E8"
GOLD = "7A5A00"
PALE_GREEN = "EAF4EC"
GREEN = "245B36"
PALE_RED = "FBEDEE"
RED = "9B1C1C"
WHITE = "FFFFFF"
BLACK = "000000"


def set_run_font(run, name: str = FONT_BODY, size: float | None = None,
                 color: str | None = None, bold: bool | None = None,
                 italic: bool | None = None) -> None:
    run.font.name = name
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:hAnsi"), name)
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), name)
    if size is not None:
        run.font.size = Pt(size)
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_row_cant_split(row) -> None:
    """Keep a table row intact so callouts never break across pages."""
    tr_pr = row._tr.get_or_add_trPr()
    if tr_pr.find(qn("w:cantSplit")) is None:
        tr_pr.append(OxmlElement("w:cantSplit"))


def shade_cell(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, *, top: int = CELL_MARGIN_TOP_DXA,
                     start: int = CELL_MARGIN_START_DXA,
                     bottom: int = CELL_MARGIN_BOTTOM_DXA,
                     end: int = CELL_MARGIN_END_DXA) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for tag, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{tag}"))
        if node is None:
            node = OxmlElement(f"w:{tag}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_cell_width(cell, width_dxa: int) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(width_dxa))
    tc_w.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths_dxa: Sequence[int], *, indent_dxa: int = TABLE_INDENT_DXA) -> None:
    if sum(widths_dxa) != TABLE_WIDTH_DXA:
        raise ValueError(f"Table widths must total {TABLE_WIDTH_DXA}: {widths_dxa}")
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.autofit = False
    tbl_pr = table._tbl.tblPr

    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(TABLE_WIDTH_DXA))
    tbl_w.set(qn("w:type"), "dxa")

    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), str(indent_dxa))
    tbl_ind.set(qn("w:type"), "dxa")

    layout = tbl_pr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tbl_pr.append(layout)
    layout.set(qn("w:type"), "fixed")

    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths_dxa:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)

    for row in table.rows:
        set_row_cant_split(row)
        for idx, cell in enumerate(row.cells):
            set_cell_width(cell, widths_dxa[min(idx, len(widths_dxa) - 1)])
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_table_borders(table, color: str = "CBD3DC", size: str = "6") -> None:
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = borders.find(qn(f"w:{edge}"))
        if tag is None:
            tag = OxmlElement(f"w:{edge}")
            borders.append(tag)
        tag.set(qn("w:val"), "single")
        tag.set(qn("w:sz"), size)
        tag.set(qn("w:space"), "0")
        tag.set(qn("w:color"), color)


def set_paragraph_keep(paragraph, *, keep_next: bool = False, keep_lines: bool = True,
                       page_break_before: bool = False) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    if keep_next:
        node = OxmlElement("w:keepNext")
        p_pr.append(node)
    if keep_lines:
        node = OxmlElement("w:keepLines")
        p_pr.append(node)
    if page_break_before:
        node = OxmlElement("w:pageBreakBefore")
        p_pr.append(node)


def add_page_field(paragraph) -> None:
    run = paragraph.add_run()
    fld_char_1 = OxmlElement("w:fldChar")
    fld_char_1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_char_2 = OxmlElement("w:fldChar")
    fld_char_2.set(qn("w:fldCharType"), "end")
    run._r.extend((fld_char_1, instr, fld_char_2))


def paragraph_border_bottom(paragraph, color: str = "D7DEE6", size: int = 8, space: int = 4) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    p_bdr = p_pr.find(qn("w:pBdr"))
    if p_bdr is None:
        p_bdr = OxmlElement("w:pBdr")
        p_pr.append(p_bdr)
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), str(size))
    bottom.set(qn("w:space"), str(space))
    bottom.set(qn("w:color"), color)
    p_bdr.append(bottom)


def configure_styles(doc: Document) -> tuple[int, int]:
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = FONT_BODY
    normal._element.rPr.rFonts.set(qn("w:ascii"), FONT_BODY)
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), FONT_BODY)
    normal.font.size = Pt(11)
    pf = normal.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.space_before = Pt(0)
    pf.space_after = Pt(6)
    pf.line_spacing = 1.25

    for name, size, color, before, after in (
        ("Heading 1", 16, BLUE, 18, 10),
        ("Heading 2", 13, BLUE, 14, 7),
        ("Heading 3", 12, DARK_BLUE, 10, 5),
    ):
        style = styles[name]
        style.font.name = FONT_BODY
        style._element.rPr.rFonts.set(qn("w:ascii"), FONT_BODY)
        style._element.rPr.rFonts.set(qn("w:hAnsi"), FONT_BODY)
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.line_spacing = 1.0
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.keep_together = True

    caption = styles["Caption"]
    caption.font.name = FONT_BODY
    caption._element.rPr.rFonts.set(qn("w:ascii"), FONT_BODY)
    caption._element.rPr.rFonts.set(qn("w:hAnsi"), FONT_BODY)
    caption.font.size = Pt(9)
    caption.font.italic = True
    caption.font.color.rgb = RGBColor.from_string(MUTED)
    caption.paragraph_format.space_before = Pt(4)
    caption.paragraph_format.space_after = Pt(8)
    caption.paragraph_format.line_spacing = 1.0
    caption.paragraph_format.keep_with_next = False

    for name, font, size, color, before, after, spacing in (
        ("Guide Kicker", FONT_BODY, 10, GOLD, 0, 8, 1.0),
        ("Guide Lead", FONT_BODY, 12, INK, 0, 10, 1.25),
        ("Guide Equation", FONT_MATH, 10.5, INK, 4, 4, 1.0),
        ("Guide Code", FONT_CODE, 8.5, INK, 2, 2, 1.0),
        ("Table Citation", FONT_BODY, 8.5, MUTED, 4, 4, 1.0),
    ):
        if name not in styles:
            style = styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        else:
            style = styles[name]
        style.font.name = font
        style._element.rPr.rFonts.set(qn("w:ascii"), font)
        style._element.rPr.rFonts.set(qn("w:hAnsi"), font)
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.line_spacing = spacing

    bullet_num_id = add_numbering_definition(doc, "bullet")
    decimal_num_id = add_numbering_definition(doc, "decimal")
    return bullet_num_id, decimal_num_id


def add_numbering_definition(doc: Document, kind: str) -> int:
    numbering = doc.part.numbering_part.element
    abstract_ids = [int(x.get(qn("w:abstractNumId"))) for x in numbering.findall(qn("w:abstractNum"))]
    num_ids = [int(x.get(qn("w:numId"))) for x in numbering.findall(qn("w:num"))]
    abstract_id = max(abstract_ids, default=-1) + 1
    num_id = max(num_ids, default=0) + 1

    abstract = OxmlElement("w:abstractNum")
    abstract.set(qn("w:abstractNumId"), str(abstract_id))
    multi = OxmlElement("w:multiLevelType")
    multi.set(qn("w:val"), "singleLevel")
    abstract.append(multi)
    lvl = OxmlElement("w:lvl")
    lvl.set(qn("w:ilvl"), "0")
    start = OxmlElement("w:start")
    start.set(qn("w:val"), "1")
    lvl.append(start)
    num_fmt = OxmlElement("w:numFmt")
    num_fmt.set(qn("w:val"), "bullet" if kind == "bullet" else "decimal")
    lvl.append(num_fmt)
    lvl_text = OxmlElement("w:lvlText")
    lvl_text.set(qn("w:val"), "•" if kind == "bullet" else "%1.")
    lvl.append(lvl_text)
    lvl_jc = OxmlElement("w:lvlJc")
    lvl_jc.set(qn("w:val"), "left")
    lvl.append(lvl_jc)
    p_pr = OxmlElement("w:pPr")
    tabs = OxmlElement("w:tabs")
    tab = OxmlElement("w:tab")
    tab.set(qn("w:val"), "num")
    tab.set(qn("w:pos"), "540")
    tabs.append(tab)
    p_pr.append(tabs)
    ind = OxmlElement("w:ind")
    ind.set(qn("w:left"), "540")
    ind.set(qn("w:hanging"), "270")
    p_pr.append(ind)
    spacing = OxmlElement("w:spacing")
    spacing.set(qn("w:after"), "80")
    spacing.set(qn("w:line"), "300")
    spacing.set(qn("w:lineRule"), "auto")
    p_pr.append(spacing)
    lvl.append(p_pr)
    r_pr = OxmlElement("w:rPr")
    r_fonts = OxmlElement("w:rFonts")
    r_fonts.set(qn("w:ascii"), FONT_BODY)
    r_fonts.set(qn("w:hAnsi"), FONT_BODY)
    r_pr.append(r_fonts)
    lvl.append(r_pr)
    abstract.append(lvl)
    numbering.append(abstract)

    num = OxmlElement("w:num")
    num.set(qn("w:numId"), str(num_id))
    abstract_num_id = OxmlElement("w:abstractNumId")
    abstract_num_id.set(qn("w:val"), str(abstract_id))
    num.append(abstract_num_id)
    numbering.append(num)
    return num_id


def add_list_item(doc: Document, text: str, num_id: int, *, bold_prefix: str | None = None) -> None:
    p = doc.add_paragraph()
    p_pr = p._p.get_or_add_pPr()
    num_pr = OxmlElement("w:numPr")
    ilvl = OxmlElement("w:ilvl")
    ilvl.set(qn("w:val"), "0")
    num_id_node = OxmlElement("w:numId")
    num_id_node.set(qn("w:val"), str(num_id))
    num_pr.extend((ilvl, num_id_node))
    p_pr.append(num_pr)
    if bold_prefix and text.startswith(bold_prefix):
        lead = p.add_run(bold_prefix)
        set_run_font(lead, bold=True)
        rest = p.add_run(text[len(bold_prefix):])
        set_run_font(rest)
    else:
        run = p.add_run(text)
        set_run_font(run)
    set_paragraph_keep(p)


def add_heading(doc: Document, text: str, level: int = 1, *, page_break: bool = False) -> None:
    p = doc.add_paragraph(text, style=f"Heading {level}")
    if page_break:
        set_paragraph_keep(p, keep_next=True, page_break_before=True)


def add_body(doc: Document, text: str, *, bold_prefix: str | None = None,
             italic: bool = False) -> None:
    p = doc.add_paragraph()
    if bold_prefix and text.startswith(bold_prefix):
        lead = p.add_run(bold_prefix)
        set_run_font(lead, bold=True)
        rest = p.add_run(text[len(bold_prefix):])
        set_run_font(rest, italic=italic)
    else:
        run = p.add_run(text)
        set_run_font(run, italic=italic)
    set_paragraph_keep(p)


def add_lead(doc: Document, text: str) -> None:
    p = doc.add_paragraph(style="Guide Lead")
    r = p.add_run(text)
    set_run_font(r, size=12, color=INK)
    set_paragraph_keep(p)


def add_callout(doc: Document, title: str, text: str, *, fill: str = CALLOUT,
                title_color: str = DARK_BLUE) -> None:
    table = doc.add_table(rows=1, cols=1)
    set_table_geometry(table, [TABLE_WIDTH_DXA])
    set_table_borders(table, color=fill, size="2")
    cell = table.cell(0, 0)
    shade_cell(cell, fill)
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    r1 = p.add_run(f"{title}  ")
    set_run_font(r1, size=10.5, color=title_color, bold=True)
    r2 = p.add_run(text)
    set_run_font(r2, size=10.5, color=INK)
    set_paragraph_keep(p)
    after = doc.add_paragraph()
    after.paragraph_format.space_after = Pt(2)


def add_equation(doc: Document, equation: str, explanation: str | None = None) -> None:
    table = doc.add_table(rows=1, cols=1)
    set_table_geometry(table, [TABLE_WIDTH_DXA])
    set_table_borders(table, color="D9E2EC", size="4")
    cell = table.cell(0, 0)
    shade_cell(cell, "F7FAFC")
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(equation)
    set_run_font(r, name=FONT_MATH, size=10.5, color=INK)
    if explanation:
        p2 = cell.add_paragraph()
        p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p2.paragraph_format.space_before = Pt(0)
        p2.paragraph_format.space_after = Pt(2)
        r2 = p2.add_run(explanation)
        set_run_font(r2, size=9, color=MUTED, italic=True)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_code_block(doc: Document, lines: Iterable[str]) -> None:
    table = doc.add_table(rows=1, cols=1)
    set_table_geometry(table, [TABLE_WIDTH_DXA])
    set_table_borders(table, color="D8DEE6", size="4")
    cell = table.cell(0, 0)
    shade_cell(cell, PALE_GRAY)
    p = cell.paragraphs[0]
    p.style = doc.styles["Guide Code"]
    for idx, line in enumerate(lines):
        if idx:
            p.add_run().add_break()
        r = p.add_run(line)
        set_run_font(r, name=FONT_CODE, size=8.5, color=INK)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_table(doc: Document, headers: Sequence[str], rows: Sequence[Sequence[str]],
              widths_dxa: Sequence[int], *, font_size: float = 9.2,
              header_fill: str = PALE_BLUE, first_col_bold: bool = False,
              numeric_cols: set[int] | None = None) -> None:
    numeric_cols = numeric_cols or set()
    table = doc.add_table(rows=1, cols=len(headers))
    set_table_geometry(table, widths_dxa)
    set_table_borders(table)
    hdr = table.rows[0]
    set_repeat_table_header(hdr)
    for idx, value in enumerate(headers):
        cell = hdr.cells[idx]
        shade_cell(cell, header_fill)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if idx in numeric_cols else WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.line_spacing = 1.0
        run = p.add_run(value)
        set_run_font(run, size=font_size, color=INK, bold=True)
    for row_idx, values in enumerate(rows):
        row = table.add_row()
        set_row_cant_split(row)
        cells = row.cells
        for idx, value in enumerate(values):
            cell = cells[idx]
            if row_idx % 2 == 1:
                shade_cell(cell, "FAFBFC")
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if idx in numeric_cols else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.05
            run = p.add_run(str(value))
            set_run_font(run, size=font_size, color=INK, bold=(first_col_bold and idx == 0))
    # Newly added rows must carry the same fixed tcW/cell-margin geometry as
    # the header; running this a second time is intentional.
    set_table_geometry(table, widths_dxa)
    after = doc.add_paragraph()
    after.paragraph_format.space_before = Pt(0)
    after.paragraph_format.space_after = Pt(2)


def add_figure(doc: Document, filename: str, caption: str, alt_text: str,
               *, width: float = 6.25, page_break_before: bool = False) -> None:
    path = FIGURES / filename
    if not path.exists():
        raise FileNotFoundError(path)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(2)
    set_paragraph_keep(p, keep_next=True, page_break_before=page_break_before)
    run = p.add_run()
    shape = run.add_picture(str(path), width=Inches(width))
    doc_pr = shape._inline.docPr
    doc_pr.set("title", caption.split(".")[0])
    doc_pr.set("descr", alt_text)
    c = doc.add_paragraph(caption, style="Caption")
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_paragraph_keep(c)


def add_section_header_footer(section) -> None:
    section.different_first_page_header_footer = True
    def populate_header(header) -> None:
        p = header.paragraphs[0]
        p.text = ""
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.0
        left = p.add_run("NSL23 PAPER COMPANION")
        set_run_font(left, size=8.5, color=MUTED, bold=True)
        p.add_run("\t")
        right = p.add_run("Exploratory V1 · V2 annotation gate")
        set_run_font(right, size=8.5, color=MUTED)
        tabs = p.paragraph_format.tab_stops
        tabs.add_tab_stop(Inches(CONTENT_WIDTH_IN), alignment=2)

    def populate_footer(footer) -> None:
        fp = footer.paragraphs[0]
        fp.text = ""
        fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        fp.paragraph_format.space_before = Pt(3)
        r = fp.add_run("NSL23 reference guide  ·  ")
        set_run_font(r, size=8.5, color=MUTED)
        add_page_field(fp)

    populate_header(section.header)
    populate_header(section.even_page_header)
    populate_footer(section.footer)
    populate_footer(section.even_page_footer)

    first_footer = section.first_page_footer
    ffp = first_footer.paragraphs[0]
    ffp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rr = ffp.add_run("Updated after the final repository scan · 11 October 2026")
    set_run_font(rr, size=8.5, color=MUTED)


def configure_document(doc: Document) -> tuple[int, int]:
    # Explicit even-page furniture avoids renderer-dependent left/right-page
    # behavior and makes the guide consistent in both print and PDF views.
    doc.settings.odd_and_even_pages_header_footer = True
    section = doc.sections[0]
    section.page_width = Inches(PAGE_WIDTH_IN)
    section.page_height = Inches(PAGE_HEIGHT_IN)
    section.top_margin = Inches(MARGIN_IN)
    section.right_margin = Inches(MARGIN_IN)
    section.bottom_margin = Inches(MARGIN_IN)
    section.left_margin = Inches(MARGIN_IN)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)
    add_section_header_footer(section)
    return configure_styles(doc)


def add_cover(doc: Document) -> None:
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(46)

    kicker = doc.add_paragraph()
    kicker.alignment = WD_ALIGN_PARAGRAPH.CENTER
    kicker.paragraph_format.space_after = Pt(16)
    r = kicker.add_run("NSL23 PROJECT COMPANION")
    set_run_font(r, size=10, color=GOLD, bold=True)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(8)
    r = title.add_run("Understanding the NSL23\nLow-Resource Alphabet Study")
    set_run_font(r, size=29, color=INK, bold=True)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(24)
    r = subtitle.add_run(
        "A plain-language, detailed guide to the data, models, mathematics, results, "
        "limitations, and the Version 2 publication path"
    )
    set_run_font(r, size=14, color=DARK_BLUE)

    add_callout(
        doc,
        "The result to remember",
        "The strongest Version 1 model scored 59.4% ± 3.3% accuracy on a 49-way "
        "unseen-signer development exam. That is real evidence from these clips, but it is "
        "exploratory—not a final estimate for all future signers.",
        fill=PALE_BLUE,
        title_color=INK,
    )

    add_table(
        doc,
        ["49 classes", "375 primary clips", "2 complete test signers"],
        [["36 consonants + 13 vowels", "Audited original isolated videos", "S1 and S2, each in turn"]],
        [3120, 3120, 3120],
        font_size=9.5,
        numeric_cols={0, 1, 2},
    )

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run("CURRENT PROJECT STATUS")
    set_run_font(r, size=9, color=GOLD, bold=True)
    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p2.paragraph_format.space_after = Pt(18)
    r2 = p2.add_run(
        "Version 1 has audited results and remains an exploratory study.\n"
        "Version 2 is protocol-ready but must not be trained until independent human review is complete."
    )
    set_run_font(r2, size=11, color=INK, bold=True)

    note = doc.add_paragraph()
    note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rr = note.add_run(
        "This guide is for the project owner’s understanding. The manuscript remains the formal scientific record."
    )
    set_run_font(rr, size=9.5, color=MUTED, italic=True)
    doc.add_page_break()


def add_reading_map(doc: Document, bullet_id: int, decimal_id: int) -> None:
    add_heading(doc, "1. Start here: the paper in five minutes", 1)
    add_lead(
        doc,
        "The study asks a simple but demanding question: if the model learns from some people, "
        "can it recognize all 49 Nepali Sign Language alphabet classes when a different person signs them?"
    )
    add_body(
        doc,
        "Imagine studying with six classmates and then taking an exam written by a seventh person. "
        "A random clip split would let that seventh person appear in both study material and the exam; "
        "this paper instead removes the exam signer from training. That makes changes in hand shape, "
        "speed, camera angle, clothing, and background much harder to exploit accidentally."
    )
    add_callout(
        doc,
        "Plainest possible conclusion",
        "Full-frame ImageNet features generalized poorly. A compact description of the hand—21 "
        "landmarks, wrist normalization, and 20 joint angles—worked much better, especially when the "
        "network learned a distance-based prototype for each class.",
        fill=PALE_GREEN,
        title_color=GREEN,
    )

    add_heading(doc, "The five facts that organize everything", 2)
    facts = [
        "The release contains 630 videos, but not 630 equivalent independent training examples. Some are aggregate tapes and some are cropped copies.",
        "The defensible Version 1 sample is 375 original, individually labelled, isolated clips covering all 49 classes.",
        "S1 and S2 are the only people with a complete 49-class isolated set. Each contributes 98 test clips: 49 classes × two lighting conditions.",
        "The strongest exploratory model is a prototypical BiLSTM using wrist-relative coordinates plus 20 joint angles: 59.4% ± 3.3% accuracy.",
        "A prospective Version 2 result needs untouched test signers recovered from aggregate tapes, so human temporal annotation—not more model design—is now the gate.",
    ]
    facts_num_id = add_numbering_definition(doc, "decimal")
    for item in facts:
        add_list_item(doc, item, facts_num_id)

    add_heading(doc, "Version 1 and Version 2 are different scientific jobs", 2)
    add_table(
        doc,
        ["Question", "Version 1", "Version 2"],
        [
            ("Purpose", "Explore what works on the audited isolated sample", "Prospectively test a locked method on sealed signers"),
            ("Status", "Complete; reproducible; exploratory", "Protocol-ready; zero verified segments; do not train"),
            ("Evaluation people", "S1 and S2, reused during study-level development", "All complete signers among sealed S5/S14/S4/S6; at least two required"),
            ("Legitimate headline", "59.4% cross-signer development accuracy", "Only the frozen Stage 4 endpoint, if coverage gate passes"),
            ("Immediate next work", "Journal formatting and transparent positioning", "Two independent annotation passes, agreement, adjudication, freeze"),
        ],
        [1900, 3650, 3810],
        font_size=8.8,
        first_col_bold=True,
    )

    add_heading(doc, "How to read this guide", 2)
    reading = [
        "Sections 2–4 explain what data enters the experiment and why the split is credible.",
        "Sections 5–7 explain the features, neural networks, training objectives, and evaluation mathematics.",
        "Sections 8–10 interpret the results without turning a development result into a deployment claim.",
        "Sections 11–12 explain exactly why manual review is required and how Version 2 becomes publishable.",
        "Sections 13–15 give reproducibility, storage, submission, and glossary checklists.",
    ]
    for item in reading:
        add_list_item(doc, item, bullet_id)


def add_dataset_section(doc: Document, bullet_id: int) -> None:
    add_heading(doc, "2. What the dataset actually contains", 1)
    add_lead(
        doc,
        "The first scientific contribution is not a neural network. It is an audit that separates "
        "original isolated clips from aggregate recordings and derived copies."
    )
    add_figure(
        doc,
        "figure_1_study_flow.png",
        "Figure 1. Dataset audit and Version 1 study flow. The primary set is the intersection of original and individually labelled files.",
        "Flow diagram showing 630 decodable videos, exclusions for aggregate recordings and cropped or preprocessed copies, and the resulting 375 original isolated clips split into two held-out-signer folds.",
        width=6.25,
    )

    add_heading(doc, "Why 630 files and 1,205 reported gestures are not 1,205 independent clips", 2)
    add_body(
        doc,
        "A dataset can count several different things. A file is a container on disk. A gesture instance "
        "is a movement reported inside a file. A model sample is the exact bounded segment sent to the "
        "network. Those counts are equal only when every file contains exactly one gesture and no derived "
        "copy is treated as new evidence. NSL23 does not have that simple structure."
    )
    add_table(
        doc,
        ["Release category", "Count", "What it means for Version 1"],
        [
            ("Decodable video files", "630", "Inventory count; not automatically independent samples"),
            ("Individually labelled files", "607", "Includes originals and derived cropped/preprocessed copies"),
            ("Aggregate tapes", "23", "Long recordings containing multiple gestures without verified temporal boundaries"),
            ("Cropped/preprocessed files", "234", "Derived views that can duplicate an original performance"),
            ("Primary original isolated clips", "375", "The defensible Version 1 sample"),
        ],
        [2540, 900, 5920],
        font_size=9.1,
        first_col_bold=True,
        numeric_cols={1},
    )
    add_callout(
        doc,
        "Leakage example",
        "Suppose an original clip is in training and a cropped copy of that same performance is in test. "
        "The model appears to generalize, but it has already seen the underlying motion. Removing derived "
        "copies prevents this family-level duplication.",
        fill=PALE_GOLD,
        title_color=GOLD,
    )

    add_heading(doc, "What is inside the 375-clip primary sample", 2)
    add_body(
        doc,
        "The sample contains original isolated clips from S1, S2, S7, S8, S10, S11, and S12. All 49 "
        "alphabet classes are present: 36 consonants and 13 vowels. S1 and S2 each supply one bright and "
        "one dark clip for every class. The other five signers supply prepared consonant clips but not a "
        "complete 49-class evaluation seat."
    )
    for item in (
        "S1 and S2: complete alphabet; eligible for the two Version 1 test folds.",
        "S7, S8, S10, S11, S12: additional training evidence, primarily consonants.",
        "Each complete signer contributes 49 × 2 = 98 test clips.",
        "The manifest records file identity, signer, class, condition, origin/derivation status, and decode checks.",
    ):
        add_list_item(doc, item, bullet_id)

    add_heading(doc, "What the audit deliberately excludes from the primary claim", 2)
    add_body(
        doc,
        "Aggregate tapes are not discarded forever; they are deferred to Version 2 because their internal "
        "gesture boundaries require human verification. Cropped files are useful as derived inputs, but they "
        "cannot be counted as independent performances. Outdoor and other release conditions that do not "
        "meet the original-isolated-labelled rule are outside this primary sample."
    )


def add_protocol_section(doc: Document, bullet_id: int, decimal_id: int) -> None:
    add_heading(doc, "3. The Version 1 experimental design", 1)
    add_lead(
        doc,
        "Each run performs genuine signer exclusion: the person being scored does not appear in that run’s "
        "training set. The study as a whole is still exploratory because S1/S2 results influenced later model choices."
    )

    add_heading(doc, "The two folds", 2)
    add_table(
        doc,
        ["Fold", "Training signers", "Test signer", "Train / test"],
        [
            ("Held out S1", "S2, S7, S8, S10, S11, S12", "S1", "277 / 98 clips"),
            ("Held out S2", "S1, S7, S8, S10, S11, S12", "S2", "277 / 98 clips"),
        ],
        [1600, 4140, 1400, 2220],
        font_size=9.2,
        first_col_bold=True,
        numeric_cols={2, 3},
    )
    add_equation(
        doc,
        "49 classes × 2 lighting conditions = 98 predictions per held-out signer",
        "This is one complete test seat."
    )
    add_equation(
        doc,
        "98 predictions from S1 + 98 predictions from S2 = 196 pooled predictions per seed",
        "The pooling happens before accuracy and macro-F1 are computed."
    )
    add_body(
        doc,
        "There is no validation set and no early stopping. For the primary matrix, every model trains for "
        "30 epochs and the epoch-30 checkpoint is evaluated once. This is less flexible than tuning on a "
        "validation signer, but it avoids pretending that a missing validation population exists."
    )

    add_heading(doc, "Why three seeds are used", 2)
    add_body(
        doc,
        "Neural-network training begins from randomized weights and randomized minibatch order. Seeds 17, "
        "42, and 73 repeat the same experiment with different allowed randomness. Their spread describes "
        "optimization instability on the same data. It does not create more signers and it does not estimate "
        "how accuracy varies across Nepal’s signer population."
    )

    add_heading(doc, "Strict within-run holdout versus untouched confirmation", 2)
    add_table(
        doc,
        ["Idea", "Meaning here"],
        [
            ("Held out within a run", "S1 is absent when S1 is tested; S2 is absent when S2 is tested. This is true."),
            ("Untouched at study level", "No result from the test people influenced later feature/model choices. This is not true for the follow-on matrix."),
            ("Exploratory development estimate", "A transparent result on repeatedly examined development people. This is the correct description of 59.4%."),
            ("Confirmatory estimate", "A locked method scored once on prospectively sealed people. This is the intended Version 2 Stage 4 job."),
        ],
        [2700, 6660],
        font_size=9.0,
        first_col_bold=True,
    )
    add_callout(
        doc,
        "Why this wording matters",
        "The 59.4% number is not fake, leaked within a run, or numerically wrong. The limitation is a "
        "higher-level one: the same two complete signers were used while deciding that angles and "
        "prototypical training were promising.",
        fill=PALE_GOLD,
        title_color=GOLD,
    )

    add_heading(doc, "The reproducibility freeze", 2)
    add_body(
        doc,
        "Before the first full primary run, the project recorded SHA-256 hashes and byte sizes for 15 "
        "protocol, configuration, manifest, loader, model, training, and launcher files. The snapshot did "
        "not store a Git revision, so the audit searched history and found one exact tree: commit "
        "0eb1ff20337b584a5a7e4b06d5e4d5a350915fc7. All 15 inputs match it byte-for-byte. The human-readable "
        "tag v1-protocol-freeze is a convenience pointer; the dated snapshot remains the chronology record. "
        "A one-epoch smoke test had already accessed S1 before the snapshot. The later tag is not preregistration."
    )
    for item in (
        "Historical-tree verification answers: ‘Can we recover the exact inputs used?’",
        "Worktree drift answers: ‘What has legitimately changed since then?’",
        "Historical result receipts keep their original machine paths; active inputs now use portable repository-relative paths.",
    ):
        add_list_item(doc, item, bullet_id)


def add_representation_section(doc: Document, bullet_id: int) -> None:
    add_heading(doc, "4. From video to model input", 1)
    add_lead(
        doc,
        "Every model sees the same temporal sample—16 frames—but the representation determines whether it "
        "mostly sees a room or a hand."
    )
    add_heading(doc, "Frame sampling", 2)
    add_body(
        doc,
        "Each clip is divided into 16 equal time bins and one frame is taken from the center of each bin. "
        "Frames preserve aspect ratio, are letterboxed to 224 × 224 pixels, and receive ImageNet channel "
        "normalization. Training brightness and contrast perturbations are drawn once per clip and applied "
        "consistently across all 16 frames. Horizontal flipping is disabled because mirroring a sign may "
        "change its linguistic or handedness meaning."
    )

    add_heading(doc, "The three representations", 2)
    add_table(
        doc,
        ["Representation", "Per-frame information", "Main benefit / risk"],
        [
            ("Full RGB", "3 × 224 × 224 pixels; MobileNetV2 produces 1,280 features", "Keeps appearance, but background and clothing can dominate"),
            ("Wrist-relative xyz", "21 landmarks × 3 coordinates + one detection flag = 64 values", "Compact hand geometry; loses texture and depends on detector quality"),
            ("xyz + joint angles", "63 coordinates + 20 angles + flag = 84 values", "Adds finger geometry; only the angle component has rigid-transform invariance"),
            ("Hand crop RGB", "Square MediaPipe-guided crop resized to 224 × 224", "Makes the hand larger, but frozen ImageNet features remain mismatched"),
        ],
        [2050, 3550, 3760],
        font_size=8.9,
        first_col_bold=True,
    )

    add_heading(doc, "Wrist-relative, scale-normalized landmarks", 2)
    add_body(
        doc,
        "MediaPipe Hands returns 21 points. Point 0 is the wrist and point 9 is a middle-finger joint used "
        "as a hand-size reference. Subtracting the wrist moves the coordinate origin onto the hand. Dividing "
        "by a within-hand distance reduces differences caused by camera zoom or physical hand size."
    )
    add_equation(
        doc,
        "pᵢ = lᵢ − l₀     ·     s = ‖p₉(x,y)‖₂     ·     p̃ᵢ = pᵢ / s*",
        "The code uses s* = s when s ≥ 10⁻⁶, and s* = 1 otherwise. This avoids division by a nearly zero distance."
    )
    add_body(
        doc,
        "The 63 normalized coordinates are concatenated with a detection flag. If MediaPipe misses a frame, "
        "that frame becomes zeros and the flag is 0. In the 6,000 cached primary frames, detection succeeded "
        "on 5,868 (97.8%), and no clip had zero detected frames. These are availability counts, not a measurement of landmark accuracy."
    )

    add_heading(doc, "Joint angles", 2)
    add_body(
        doc,
        "Coordinates can still rotate when the wrist or camera rotates. The follow-on representation adds "
        "15 within-finger flexion angles and five wrist-level abduction angles. For two unit vectors meeting "
        "at a joint, the angle comes from their dot product."
    )
    add_equation(
        doc,
        "φ(u, v) = arccos( clip(uᵀv, −1, 1) )",
        "The clip prevents floating-point roundoff from sending arccos outside its valid range."
    )
    add_callout(
        doc,
        "Why angles help",
        "Two people can place their hands in different locations and orientations while bending the same "
        "finger joints. Angles explicitly encode that bending pattern. They are not magic: detector errors "
        "and visually similar classes still matter. Angles are unchanged by applying the same rigid rotation "
        "and uniform scale to valid points, but image-derived landmarks need not transform that way when "
        "the camera viewpoint changes. The concatenated xyz component still retains hand orientation.",
        fill=PALE_GREEN,
        title_color=GREEN,
    )

    add_figure(
        doc,
        "figure_2_architectures.png",
        "Figure 2. The paper’s RGB, landmark-softmax, and improved prototype pipelines. All operate on 16-frame clips.",
        "Three pipeline diagrams: frozen MobileNet RGB controls; wrist-relative xyz landmarks with MLP or BiLSTM softmax heads; and the improved xyz-plus-angle BiLSTM encoder classified by nearest training prototype.",
        width=6.0,
    )


def add_models_section(doc: Document, bullet_id: int) -> None:
    add_heading(doc, "5. What each model is doing", 1)
    add_lead(
        doc,
        "The model matrix is a controlled sequence of questions: Is one RGB frame enough? Does motion help? "
        "Do landmarks help? Do angles help? Does distance-based class structure help?"
    )
    add_table(
        doc,
        ["Model", "Trainable parameters", "Question answered"],
        [
            ("Single-frame CNN", "62,769", "Can a frozen ImageNet feature from one middle frame separate 49 classes?"),
            ("CNN–BiLSTM", "736,305", "Do ordered features across 16 RGB frames help?"),
            ("CNN–Transformer", "1,399,345", "Does attention across the 16 RGB frames help?"),
            ("Landmark MLP", "301,617", "Are flattened normalized hand coordinates enough without a temporal recurrence?"),
            ("Landmark BiLSTM", "73,137", "Does a compact temporal model over xyz landmarks generalize better?"),
            ("Hand-crop CNN", "62,769", "Was the hand simply too small in the full frame?"),
            ("xyz+angle BiLSTM", "83,377", "Do explicit joint angles improve the temporal landmark representation?"),
            ("Proto-BiLSTM", "83,377", "Does learning an embedding around class prototypes improve cross-signer transfer?"),
        ],
        [2450, 1650, 5260],
        font_size=8.7,
        first_col_bold=True,
        numeric_cols={1},
    )

    add_heading(doc, "CNN: a visual feature extractor", 2)
    add_body(
        doc,
        "MobileNetV2 was pretrained on ImageNet photographs. Here its weights and batch-normalization state "
        "stay frozen. The trainable head must use those generic visual features to recognize signs. This is "
        "computationally economical and a fair low-data baseline, but ImageNet is not a hand-shape alphabet."
    )

    add_heading(doc, "BiLSTM: read the clip in both directions", 2)
    add_body(
        doc,
        "An LSTM maintains a gated memory while reading a sequence. A bidirectional LSTM reads frame 1→16 "
        "and 16→1, concatenates both views, normalizes them, and averages across time. Because classification "
        "happens after the full isolated clip is available, future frames are allowed; this is not a causal "
        "streaming-language model."
    )

    add_heading(doc, "Transformer: compare every sampled time point", 2)
    add_body(
        doc,
        "The Transformer projects each frame feature to 256 values, adds a learned time position, and uses "
        "four attention heads in two layers. Each attention head measures which time points should influence "
        "one another. With only 277 training clips per fold, a large head can fit the training sample "
        "without transferring well to another signer. The scores do not identify which visual cues it learned."
    )

    add_heading(doc, "Prototypical encoder: learn a map where class means matter", 2)
    add_body(
        doc,
        "A conventional softmax model learns 49 classifier weight vectors. A prototypical network instead "
        "learns an embedding space. During an episode, five support clips per class are averaged into a class "
        "prototype and two query clips per class are pulled toward the correct prototype. At final unseen-signer "
        "evaluation, prototypes are recomputed from training clips only. The test signer contributes nothing."
    )
    add_equation(
        doc,
        "cᶜ = (1 / |Sᶜ|) Σ gψ(X)     ·     P(y=c | X) = exp(−‖gψ(X)−cᶜ‖₂) / Σc′ exp(−‖gψ(X)−cᶜ′‖₂)",
        "cᶜ is the mean embedding for class c; the closest prototype receives the largest probability."
    )
    for item in (
        "Every episode visits all 49 classes, so class imbalance is handled by the episode structure.",
        "When a class has fewer than seven clips, the five support and two query draws use replacement. A training clip can appear on both sides of an episode; the counts are draws, not distinct recordings.",
        "The benefit is not ‘one-shot magic’; the primary 59.4% evaluation still uses training-set prototypes and an unseen test signer.",
    ):
        add_list_item(doc, item, bullet_id)


def add_training_section(doc: Document, bullet_id: int) -> None:
    add_heading(doc, "6. Training without accidentally moving the goalposts", 1)
    add_heading(doc, "Primary softmax training", 2)
    add_body(
        doc,
        "The five primary models use inverse-frequency class-weighted cross entropy. Vowels have only two "
        "training clips in a fold, while some consonants have more. Giving rare classes larger weights keeps "
        "the loss from being dominated by the better-represented consonants."
    )
    add_equation(
        doc,
        "wᶜ ∝ 1/nᶜ, normalized so mean(wᶜ)=1     ·     L = −Σᶜ wᶜ yᶜ log(ŷᶜ)",
        "nᶜ is the training count for class c. This is one example's loss; the batch mean divides summed losses by the sum of target-class weights."
    )
    add_table(
        doc,
        ["Setting", "Primary matrix", "Prototypical follow-on"],
        [
            ("Optimizer", "AdamW", "AdamW"),
            ("Learning rate / weight decay", "3×10⁻⁴ / 10⁻⁴", "3×10⁻⁴ / 10⁻⁴"),
            ("Learning-rate schedule", "Cosine decay", "Constant"),
            ("Landmark feature jitter", "Gaussian SD 0.02; flag unchanged", "None"),
            ("Epochs", "30", "40"),
            ("Batch structure", "Batch size 8", "40 all-class episodes per epoch"),
            ("Seeds", "17, 42, 73", "17, 42, 73"),
            ("Validation / early stopping", "None; evaluate epoch 30", "None; evaluate epoch 40"),
            ("Class balancing", "Inverse-frequency loss weights", "Each episode visits every class"),
        ],
        [2600, 3380, 3380],
        font_size=8.8,
        first_col_bold=True,
    )

    add_heading(doc, "Determinism: useful, but not a time machine", 2)
    add_body(
        doc,
        "The code fixes seeds and asks PyTorch for deterministic algorithms on the recorded Apple M4 host. "
        "That reduces uncontrolled variation within the supported environment. It does not guarantee "
        "bit-for-bit identical results on another PyTorch version, CPU, GPU, or MPS implementation. The "
        "strongest reproducibility evidence is therefore the stored checkpoints, primitive predictions, "
        "per-run metrics, frozen inputs, and repeatable aggregation—not a promise that every machine will "
        "recreate every floating-point bit."
    )
    add_callout(
        doc,
        "Why no retraining was needed for the audit",
        "The audit recomputed paper values from 196 per-seed predictions, checked checkpoint and metrics "
        "inventories, verified frozen inputs, and regenerated figures/tables. The October scan also loaded "
        "all 60 checkpoints, replayed all landmark evaluation clips and representative RGB clips, and "
        "recomputed the separate closed-enrollment results. Retraining would answer a "
        "different question about cross-hardware repeatability and would take hours.",
        fill=PALE_BLUE,
        title_color=DARK_BLUE,
    )
    for item in (
        "Primary matrix size: 5 models × 2 folds × 3 seeds = 30 runs.",
        "Every run stores test predictions, metrics, per-class metrics, confusion matrices, environment information, training history, and a checkpoint.",
        "Seed-level summaries pool S1 and S2 predictions before computing nonlinear metrics.",
    ):
        add_list_item(doc, item, bullet_id)


def add_metrics_section(doc: Document, bullet_id: int) -> None:
    add_heading(doc, "7. The evaluation mathematics", 1)
    add_lead(
        doc,
        "The most important mathematical rule is the unit of aggregation: pool the two signer folds within "
        "each seed, compute the metric once on 196 predictions, then summarize the three seed scores."
    )

    add_heading(doc, "Accuracy", 2)
    add_equation(
        doc,
        "Accuracy = number of correct predictions / number of evaluated clips",
        "For 49 equally supported classes, chance top-1 accuracy is 1/49 ≈ 2.04%."
    )
    add_body(
        doc,
        "Each class has four pooled test clips per seed—two from S1 and two from S2. Equal support means "
        "overall accuracy is numerically equal to macro-recall in this specific pooled design. That equality "
        "would disappear if classes had unequal test counts."
    )

    add_heading(doc, "Precision, recall, and F1", 2)
    add_equation(
        doc,
        "Precisionᶜ = TPᶜ/(TPᶜ+FPᶜ)     ·     Recallᶜ = TPᶜ/(TPᶜ+FNᶜ)",
        "Precision asks whether predictions of class c are correct; recall asks whether true class-c clips are recovered."
    )
    add_equation(
        doc,
        "F1ᶜ = 2·Precisionᶜ·Recallᶜ / (Precisionᶜ+Recallᶜ)     ·     Macro-F1 = (1/49)Σᶜ F1ᶜ",
        "Macro-F1 gives every class equal weight after computing class-level F1."
    )

    add_heading(doc, "Mean and sample standard deviation over seeds", 2)
    add_equation(
        doc,
        "x̄ = (x₁+x₂+x₃)/3     ·     s = √[ Σᵢ(xᵢ−x̄)² / (3−1) ]",
        "The ± value in the paper is sample SD across seeds, not a 95% confidence interval."
    )
    add_body(
        doc,
        "For the strongest model, seed accuracies are 59.18%, 62.76%, and 56.12%. Their mean is 59.35% "
        "and their sample SD is 3.32 percentage points, reported as 59.4% ± 3.3%."
    )

    add_heading(doc, "The macro-F1 correction found in this audit", 2)
    add_body(
        doc,
        "Earlier follow-on scripts averaged the macro-F1 of the S1 fold and the macro-F1 of the S2 fold. "
        "The methods, however, say to pool all 196 predictions first. These procedures are not equivalent "
        "because F1 is nonlinear: false positives and false negatives can balance differently after pooling."
    )
    add_table(
        doc,
        ["Follow-on model", "Old fold-average F1", "Correct pooled F1"],
        [
            ("Hand-crop CNN", "15.9% ± 0.7%", "16.9% ± 0.9%"),
            ("xyz+angle MLP", "14.6% ± 2.9%", "15.6% ± 2.7%"),
            ("xyz+angle BiLSTM", "30.1% ± 2.6%", "31.5% ± 3.4%"),
            ("xyz Proto-BiLSTM", "50.6% ± 2.4%", "53.4% ± 2.8%"),
            ("xyz+angle Proto-BiLSTM", "54.7% ± 4.2%", "57.4% ± 3.6%"),
        ],
        [3960, 2600, 2800],
        font_size=9.0,
        first_col_bold=True,
        numeric_cols={1, 2},
    )
    add_callout(
        doc,
        "What changed—and what did not",
        "The paper, tables, summaries, figures, and audit script now use pooled macro-F1. No accuracy "
        "changed. The 59.4% headline remained exactly the same.",
        fill=PALE_GREEN,
        title_color=GREEN,
    )

    add_heading(doc, "What the uncertainty does and does not describe", 2)
    for item in (
        "Seed SD: variation caused by three training randomizations on the same signers and clips.",
        "Clip bootstrap: variation obtained by resampling these evaluated clips; useful descriptively, but it cannot invent new signers.",
        "Signer-population uncertainty: not estimable from only two complete evaluation people.",
        "A confidence statement about future Nepalese signers would require prospectively sampled, independent people and a suitable signer-level analysis.",
    ):
        add_list_item(doc, item, bullet_id)


def add_results_section(doc: Document, bullet_id: int) -> None:
    add_heading(doc, "8. Results: what improved, and what we can infer", 1)
    add_heading(doc, "The frozen five-model primary matrix", 2)
    add_table(
        doc,
        ["Model", "Accuracy", "Macro-F1", "Interpretation"],
        [
            ("Single-frame CNN", "5.8% ± 0.8%", "3.5% ± 0.2%", "Most evaluation clips are classified incorrectly"),
            ("CNN–BiLSTM", "7.0% ± 2.1%", "4.2% ± 2.1%", "Adding temporal capacity barely helps cross-signer transfer"),
            ("CNN–Transformer", "6.3% ± 0.8%", "4.4% ± 1.3%", "The observed gain over the single-frame model is small"),
            ("Landmark MLP", "18.0% ± 2.1%", "11.1% ± 1.8%", "Landmark features outperform the frozen RGB baselines"),
            ("Landmark BiLSTM", "24.8% ± 4.0%", "16.8% ± 3.8%", "Compact temporal hand geometry is the best frozen control"),
        ],
        [2200, 1660, 1660, 3840],
        font_size=8.6,
        first_col_bold=True,
        numeric_cols={1, 2},
    )
    add_figure(
        doc,
        "figure_3_primary_performance.png",
        "Figure 3. Frozen primary performance. Points are seed scores; error bars are sample SD across three seeds.",
        "Grouped bars for accuracy and macro-F1 across single-frame CNN, CNN-BiLSTM, CNN-Transformer, landmark MLP, and landmark BiLSTM, showing landmark models clearly higher than RGB baselines.",
        width=6.25,
    )

    add_heading(doc, "The RGB temporal heads overfit", 2)
    add_body(
        doc,
        "The CNN–BiLSTM and CNN–Transformer reach mean training accuracy 1.00 by the final epoch, while test "
        "accuracy remains around 6–7%. This is a generalization gap: fitting the training clips did not "
        "produce reliable classification on these two unseen signers. Background, hand resolution, and "
        "feature mismatch are possible explanations, but the experiment does not isolate their effects."
    )
    add_figure(
        doc,
        "figure_4_training_curves.png",
        "Figure 4. Mean training trajectories. Perfect training fit for the RGB temporal heads does not become unseen-signer accuracy.",
        "Training loss, accuracy, and macro-F1 curves for the primary models, highlighting RGB temporal heads reaching perfect training accuracy while evaluation performance remains low.",
        width=6.25,
    )

    add_heading(doc, "Lighting and sign type reveal different problems", 2)
    add_body(
        doc,
        "The xyz landmark BiLSTM scores 27.6% on bright clips and 22.1% on dark clips. Because lighting was "
        "not randomized as an intervention, this is a description, not a causal claim. The larger diagnostic "
        "is class type: the same model scores 66.0% on 13 vowels but only 10.0% on 36 consonants. Most of its "
        "24.8% overall score comes from vowels."
    )
    add_figure(
        doc,
        "figure_5_lighting_performance.png",
        "Figure 5. Bright and dark performance for the primary models. Landmark models retain the clearest advantage.",
        "Grouped bright-versus-dark accuracy and macro-F1 bars for five primary models, with landmark models outperforming RGB models and a modest dark-condition drop.",
        width=6.25,
    )

    add_heading(doc, "The follow-on ladder", 2)
    add_table(
        doc,
        ["Model", "Accuracy", "Pooled macro-F1", "Consonants", "Vowels"],
        [
            ("xyz BiLSTM (CE control)", "24.8% ± 4.0%", "16.8% ± 3.8%", "10.0% ± 3.9%", "66.0% ± 4.0%"),
            ("Hand-crop CNN", "19.2% ± 1.3%", "16.9% ± 0.9%", "15.5% ± 0.8%", "29.5% ± 2.9%"),
            ("xyz+angle BiLSTM (CE)", "37.9% ± 2.4%", "31.5% ± 3.4%", "28.7% ± 4.5%", "63.5% ± 3.8%"),
            ("xyz Proto-BiLSTM", "55.8% ± 1.8%", "53.4% ± 2.8%", "58.3% ± 2.5%", "48.7% ± 5.9%"),
            ("xyz+angle Proto-BiLSTM", "59.4% ± 3.3%", "57.4% ± 3.6%", "61.8% ± 5.7%", "52.6% ± 4.8%"),
        ],
        [2500, 1715, 1715, 1715, 1715],
        font_size=8.1,
        first_col_bold=True,
        numeric_cols={1, 2, 3, 4},
    )
    add_figure(
        doc,
        "figure_9_zero_shot_improvements.png",
        "Figure 6. Follow-on unseen-signer accuracy. Angles help; prototypical training is the larger step.",
        "Accuracy bars progressing from xyz cross-entropy BiLSTM to hand crop, xyz-plus-angle cross-entropy, xyz prototypical, and xyz-plus-angle prototypical BiLSTM, with the final model near 59 percent.",
        width=6.25,
    )
    add_body(
        doc,
        "The important scientific pattern is not just a taller final bar. Angles lift consonants from 10.0% "
        "to 28.7% under cross entropy. The xyz+angle prototypical model reaches "
        "61.8% consonant accuracy, although its loss, episode sampling, and epoch count all differ from the "
        "softmax model. Vowel accuracy falls from 66.0% to 52.6%, so the final "
        "gain is a more balanced alphabet representation rather than simply becoming an even stronger vowel classifier."
    )
    add_callout(
        doc,
        "Practical interpretation",
        "59.4% still means roughly two errors in every five first-choice predictions. It is encouraging for a "
        "low-resource research prototype, but weak for an unsupervised teaching tool that must reliably tell a learner they are correct.",
        fill=PALE_GOLD,
        title_color=GOLD,
    )


def add_confusion_section(doc: Document, bullet_id: int) -> None:
    add_heading(doc, "9. Errors, enrollment, and the danger of changing the exam", 1)
    add_heading(doc, "A confusion matrix is an error map", 2)
    add_body(
        doc,
        "Rows represent the true class and columns represent the predicted class. A perfect model has all mass "
        "on the diagonal. Off-diagonal cells show systematic confusions. Row normalization answers: out of all "
        "true examples of this class, where did the model send them?"
    )
    add_figure(
        doc,
        "figure_11_proto_bilstm_confusion.png",
        "Figure 7. Row-normalized confusion matrix for the xyz+angle prototypical BiLSTM, averaged over folds and seeds.",
        "A 49-by-49 row-normalized confusion heatmap for the final prototypical model, showing a stronger diagonal than the original landmark BiLSTM but remaining structured off-diagonal mistakes.",
        width=5.8,
    )
    add_body(
        doc,
        "The earlier xyz softmax BiLSTM collapsed several consonants onto vowel-like predictions—for example, "
        "BHA and RA repeatedly became AH, TA became RI, and NA became YAN. The prototypical angle model improves "
        "the diagonal but does not eliminate signer-specific ambiguity or detector noise. A confusion matrix is "
        "more informative than accuracy alone because it shows whether errors are diffuse or repeatable."
    )

    add_heading(doc, "Closed 1-shot enrollment is a different product situation", 2)
    add_body(
        doc,
        "Each complete signer has two clips per class: one bright and one dark. In the closed 1-shot exam, the "
        "system gets the other-lighting clip from the same signer for every class. Bright queries use 49 dark "
        "support clips and dark queries use 49 bright support clips. This is an enrolled user, not a first-time user."
    )
    add_figure(
        doc,
        "figure_10_enrollment.png",
        "Figure 8. Unseen-signer classification compared with closed 1-shot enrollment using the held-out signer’s other lighting.",
        "Paired bars showing much higher closed one-shot accuracy for xyz and xyz-plus-angle softmax models, with a smaller enrollment gain for the prototypical encoder.",
        width=6.25,
    )
    add_table(
        doc,
        ["Model", "No enrollment", "Closed 1-shot"],
        [
            ("xyz BiLSTM", "24.8% ± 4.0%", "76.2% ± 4.2%"),
            ("xyz+angle BiLSTM", "37.9% ± 2.4%", "78.2% ± 2.8%"),
            ("xyz+angle Proto-BiLSTM", "59.4% ± 3.3%", "73.0% ± 1.5%"),
        ],
        [3960, 2700, 2700],
        font_size=9.2,
        first_col_bold=True,
        numeric_cols={1, 2},
    )

    add_heading(doc, "Top-k and N-way scores are different exams", 2)
    add_body(
        doc,
        "Top-3 gives credit when the correct class appears anywhere among three suggestions. Top-5 gives five "
        "attempts. A 5-way episode chooses only five classes rather than all 49. These metrics can be useful "
        "for user-interface design or comparison with few-shot literature, but they cannot replace 49-way top-1."
    )
    add_figure(
        doc,
        "figure_12_protocol_ladder.png",
        "Figure 9. One encoder under different grading conventions. Higher bars partly reflect easier exams, not a newly trained model.",
        "Comparison of 49-way scores, 36-way consonants, a five-class restriction using full training means, and same-signer one-shot enrollment.",
        width=6.25,
        page_break_before=True,
    )
    for item in (
        "49-way unseen top-1: 59.4% ± 3.3%; chance 2.0%.",
        "49-way unseen top-3: 85.5% ± 6.0%; chance 6.1%.",
        "49-way unseen top-5: 91.3% ± 4.0%; chance 10.2%.",
        "36-way unseen consonants: 65.7% ± 3.6%; chance 2.8%.",
        "5-way restriction with full training-class means: 90.6% ± 2.5%; chance 20.0%. Each mean uses all training clips of that class, so this is not one-shot.",
    ):
        add_list_item(doc, item, bullet_id)
    add_callout(
        doc,
        "Reporting rule",
        "Never shorten ‘90.6% with five candidate classes’ to ‘90.6% NSL23 alphabet accuracy.’ That changes "
        "both the number of competing classes and the support information available to the model.",
        fill=PALE_RED,
        title_color=RED,
    )


def add_runtime_section(doc: Document, bullet_id: int) -> None:
    add_heading(doc, "10. Runtime: recognition quality and speed are separate", 1)
    add_table(
        doc,
        ["Model", "Model-only median", "End-to-end median", "What dominates"],
        [
            ("Single-frame CNN", "4.0 ms", "939.0 ms/clip", "Original video decode and preprocessing"),
            ("CNN–BiLSTM", "49.5 ms", "962.9 ms/clip", "Decode plus 16 frozen CNN frame passes"),
            ("CNN–Transformer", "49.7 ms", "970.9 ms/clip", "Decode plus 16 frozen CNN frame passes"),
            ("Landmark BiLSTM", "0.3 ms", "345.7 ms/clip", "MediaPipe over 16 frames"),
        ],
        [2300, 1750, 2050, 3260],
        font_size=8.8,
        first_col_bold=True,
        numeric_cols={1, 2},
    )
    add_body(
        doc,
        "A neural-network forward pass can be fast while the whole pipeline is slower. Loading and decoding a "
        "video, resizing frames, and running MediaPipe all take time. The table separates model-only latency "
        "from end-to-end clip processing so the paper does not call 0.3 ms a complete 16-frame system."
    )
    add_body(
        doc,
        "The rolling prototype webcam loop processed recorded NSL23 streams at roughly 33–42 frames per second "
        "on an Apple M4 CPU path, including MediaPipe. On one 90-frame S1 bright A clip it reached 39.6 FPS and "
        "the smoothed final prediction was A. This shows engineering feasibility for that replay; it is not a "
        "live recognition-accuracy study."
    )
    for item in (
        "FPS answers ‘Can the pipeline keep up with incoming frames on this machine?’",
        "Accuracy answers ‘How often is the predicted class correct under this evaluation protocol?’",
        "A product claim needs both, plus live-user testing, failure handling, calibration, accessibility, and a representative signer cohort.",
    ):
        add_list_item(doc, item, bullet_id)


def add_claims_section(doc: Document, bullet_id: int) -> None:
    add_heading(doc, "11. What can and cannot be claimed", 1)
    add_table(
        doc,
        ["Safe statement", "Why it is safe"],
        [
            ("The xyz+angle Proto-BiLSTM achieved 59.4% ± 3.3% accuracy on the pooled S1/S2 cross-signer development evaluation.", "It names the model, metric, people, and exploratory boundary."),
            ("Wrist-relative landmarks generalized better than frozen full-frame ImageNet features on these clips.", "All models used the same fixed folds and seeds; the pattern is large and consistent descriptively."),
            ("Angles and prototypical training improved consonant performance in the follow-on matrix.", "The corrected artifacts support 61.8% consonant accuracy for the final model."),
            ("A replayed-video prototype ran at 33–42 FPS on an Apple M4.", "It describes the timed host and task without calling it live-user accuracy."),
        ],
        [5550, 3810],
        font_size=8.6,
        first_col_bold=False,
    )
    add_table(
        doc,
        ["Unsafe shortcut", "What is wrong with it"],
        [
            ("‘59.4% accuracy for new NSL users’", "Only two complete signers were repeatedly used during development."),
            ("‘91.3% alphabet recognition’", "That is top-5, not top-1; the model gets five guesses."),
            ("‘90.6% 49-way recognition’", "That is a five-class restriction with full training means and 20% chance."),
            ("‘Real-time system is 39.6% accurate’", "39.6 is FPS, not accuracy."),
            ("‘State of the art in NSL’", "Other NSL studies use different classes, files, people, and splits."),
            ("‘Works in rural Nepal’", "The study lacks an in-country deployment cohort and field conditions."),
        ],
        [3880, 5480],
        font_size=8.7,
        header_fill="F6DFE1",
        first_col_bold=True,
    )

    add_heading(doc, "Why this can still be publishable", 2)
    add_body(
        doc,
        "A paper does not need a perfect deployment-ready score to be useful. Version 1 contributes a "
        "traceable dataset audit, strict within-run signer exclusion, negative RGB results, a clear feature "
        "ablation path, corrected primitive-level statistics, reproducibility artifacts, and honest limits. "
        "The publishable framing is exploratory low-resource evidence and protocol analysis—not a claim that "
        "the final model has solved Nepali Sign Language recognition."
    )
    for item in (
        "Keep ‘exploratory’ or ‘development’ next to the 59.4% result in the abstract, results, discussion, and conclusion.",
        "Report per-signer values where space permits: approximately 63.3% on S1 and 55.4% on S2 for the final model.",
        "Keep the task definition explicit: isolated, manually bounded 49-way letter classification.",
        "Do not compare percentages across papers unless class set, split unit, signer separation, and support protocol match.",
    ):
        add_list_item(doc, item, bullet_id)


def add_manual_review_section(doc: Document, bullet_id: int, decimal_id: int) -> None:
    add_heading(doc, "12. Why human review is necessary for Version 2", 1)
    add_lead(
        doc,
        "The dataset creators labelled the recordings at the file or sequence level. Version 2 needs a "
        "different annotation: exact, quality-controlled temporal segments inside long aggregate tapes."
    )
    add_heading(doc, "What is already known versus what is missing", 2)
    add_table(
        doc,
        ["Already supplied by the dataset", "Still needed for Version 2"],
        [
            ("Which signer and recording a tape belongs to", "Exact start and end of each usable letter performance"),
            ("The intended class sequence or file-level label context", "Whether a motion proposal truly contains the claimed sign"),
            ("The source video itself", "Whether the gesture is complete, clear, and suitable as an isolated sample"),
            ("Aggregate recordings with multiple gestures", "Detector misses, false proposals, repeats, transitions, and ambiguous motions"),
        ],
        [4680, 4680],
        font_size=8.8,
    )
    add_body(
        doc,
        "A motion detector has proposed 691 candidate intervals across 21 aggregate tapes. It detects change; "
        "it does not understand NSL. A candidate can begin late, end early, capture a transition between signs, "
        "contain a repeat, or miss a low-motion sign. Training directly on those proposals would convert "
        "detector mistakes into training labels and could make test coverage depend on an unverified algorithm."
    )
    add_callout(
        doc,
        "The reviewer’s actual job",
        "Watch each proposed interval in its source-video context, identify the signed letter, correct the "
        "temporal boundaries, choose a defined decision, and sweep the entire tape for missed gestures. The "
        "reviewer is not re-labelling the 375 isolated Version 1 files.",
        fill=PALE_BLUE,
        title_color=INK,
    )

    add_heading(doc, "Why two independent passes", 2)
    add_body(
        doc,
        "One knowledgeable person can make an honest mistake or inherit a subtle bias from the proposal order. "
        "Pass 2 therefore starts from the original detector proposal and hides pass-1 class, decision, bounds, "
        "and accept/reject outcome. For the sealed test-hierarchy tapes S5, S14, S4, and S6, the two passes must "
        "come from two different named reviewers. Disagreements are measured before an adjudicator resolves them."
    )
    for item in (
        "Independence makes agreement meaningful; copying pass 1 would create artificial agreement.",
        "Two reviewers reduce the chance that an incorrect boundary becomes frozen truth.",
        "Adjudication may resolve a disagreement only after both passes exist; it cannot replace a missing second pass.",
        "The project owner may serve as a reviewer only if they genuinely recognize all 49 NSL alphabet signs and follow the same blinded protocol.",
    ):
        add_list_item(doc, item, bullet_id)

    add_heading(doc, "The four decisions", 2)
    add_table(
        doc,
        ["Decision", "Use it when…", "What happens"],
        [
            ("Accept", "One complete, recognizable target sign is visible and its start/end can be bounded", "Keep the corrected segment for later agreement/freeze"),
            ("Reject", "The proposal is not a usable target sign: transition, duplicate, non-sign motion, or wrong proposal", "Exclude the proposal, while preserving the review record"),
            ("Incomplete", "The target sign is cut off or never reaches a complete form", "Do not use it as a verified sample"),
            ("Ambiguous", "A qualified reviewer cannot confidently determine class or boundary", "Flag for adjudication; never force certainty"),
        ],
        [1400, 4490, 3470],
        font_size=8.7,
        first_col_bold=True,
    )

    add_heading(doc, "What you need to do as coordinator", 2)
    steps = [
        "Recruit qualified NSL alphabet reviewers and record their names consistently. Annotation alone normally belongs in acknowledgements unless a reviewer later contributes intellectually to study design, analysis, interpretation, or writing.",
        "Run a disposable S3 calibration pilot. Discuss boundary examples and decision definitions, but do not merge pilot decisions into the real master annotation file.",
        "Lock any policy correction before real reviews begin. If the boundary policy changes, create a new protocol lock rather than silently moving the old one.",
        "Complete pass 1, then create a sealed pass-2 export for the second reviewer. Never send the master CSV to an outside pass-2 reviewer.",
        "Import pass 2, save the pre-adjudication agreement artifact, adjudicate completed disagreements, run coverage validation, and freeze only when every gate passes.",
    ]
    coordinator_num_id = add_numbering_definition(doc, "decimal")
    for item in steps:
        add_list_item(doc, item, coordinator_num_id)


def add_v2_section(doc: Document, bullet_id: int, decimal_id: int) -> None:
    add_heading(doc, "13. The Version 2 route to a confirmatory result", 1)
    add_lead(
        doc,
        "Version 2 is deliberately boring in the best scientific sense: test identities, features, trials, "
        "seeds, endpoint, and stop conditions are written down before the test data can influence the model."
    )
    add_heading(doc, "Current inventory and gate", 2)
    add_table(
        doc,
        ["Item", "Current state"],
        [
            ("Aggregate review tapes", "21 tapes; 22.13 source minutes; about 3.47 GiB"),
            ("Motion proposals", "691 unlabeled candidate intervals"),
            ("Pass-1 decisions", "0"),
            ("Pass-2 decisions", "0"),
            ("Verified segments", "0"),
            ("Frozen V2 manifests", "None"),
            ("Permission to train", "No—human annotation is the active gate"),
        ],
        [3300, 6060],
        font_size=9.1,
        first_col_bold=True,
    )

    add_heading(doc, "Permanent hierarchy", 2)
    add_table(
        doc,
        ["Role", "Signers / rule"],
        [
            ("Sealed test candidates", "S5, S14, S4, S6. Never release them into development, even if incomplete."),
            ("Development", "S1 and S2; S3 is validation only if it becomes complete."),
            ("Other incomplete non-test signers", "May contribute verified/original training clips under the frozen rules."),
            ("Minimum confirmatory Stage 4", "At least two sealed candidates must have verified coverage for all 49 classes."),
            ("Failure path", "Freeze/publish the annotation resource and Version 1 exploration without a confirmatory 49-way claim."),
        ],
        [2850, 6510],
        font_size=8.9,
        first_col_bold=True,
    )
    add_body(
        doc,
        "The hierarchy hash in configs/v2_protocol.json is "
        f"{V2_PROTOCOL['hierarchy_sha256']}. It binds the predeclared roles."
    )

    add_heading(doc, "The gated sequence", 2)
    stages = [
        "Calibration: two reviewers practice on disposable S3 proposals and align on the written boundary policy.",
        "Independent annotation: complete pass 1 and blinded pass 2, including full-tape sweeps for detector misses.",
        "Agreement and adjudication: save the hash-bound pre-adjudication report, then resolve completed disagreements.",
        "Coverage and freeze: validate hashes, durations, overlap, reviewer identities, and per-signer 49-class coverage; freeze the protocol and manifests.",
        "Stage 2 diagnostic: run the existing xyz+angle prototypical image model on S1/S2 development data only.",
        "Stage 3 selection: if S3 is complete, run exactly three preregistered geometry trials, selecting mean per-signer macro-recall across development LOSO folds and seeds. If S3 is incomplete, skip the search and carry the Stage 2 specification.",
        "Final retrain: train seeds 17/42/73 once on every permitted non-test sample, including validation data; tester rows are rejected by code.",
        "Stage 4 evaluation: run the three frozen checkpoints on the sealed test cohort once, with no adaptation or selection.",
        "Manuscript: report the equal-weight per-signer macro-recall endpoint, each tester separately, agreement, coverage, and every failure gate.",
    ]
    stage_num_id = add_numbering_definition(doc, "decimal")
    for item in stages:
        add_list_item(doc, item, stage_num_id)

    add_heading(doc, "Primary Version 2 estimand", 2)
    add_body(
        doc,
        "For each seed, average all verified clips within a signer × class cell. Then average the 49 class "
        "recalls within each signer, and average signers equally. Finally report the mean and sample SD over "
        "the three seeds. This prevents a signer with more accepted clips from dominating the headline. "
        "Clip-pooled accuracy remains a secondary diagnostic."
    )
    add_equation(
        doc,
        "Seed score = mean over frozen signers [ mean over 49 classes ( recall within signer × class ) ]",
        "Every signer and every class receives equal weight in the primary Stage 4 endpoint."
    )

    add_heading(doc, "Hard stop conditions", 2)
    for item in (
        "Do not train, build prototypes, compare features, or select checkpoints until the V2 protocol is frozen.",
        "Never use configs/signer_split.json for V2; it is obsolete and places sealed people in training.",
        "Never train on S5, S14, S4, or S6—even when a candidate lacks complete coverage.",
        "Never use model performance to decide which signer becomes a tester.",
        "Never force the detector output to contain exactly 13 vowels or 36 consonants.",
        "Never run or report a confirmatory 49-way Stage 4 when fewer than two sealed testers are complete.",
    ):
        add_list_item(doc, item, bullet_id)


def add_repro_section(doc: Document, bullet_id: int) -> None:
    add_heading(doc, "14. Reproducing and auditing the project", 1)
    add_heading(doc, "Evidence hierarchy", 2)
    add_table(
        doc,
        ["Layer", "What it proves"],
        [
            ("Source manifests and protocol files", "Which videos, classes, signers, folds, settings, and exclusions were intended"),
            ("Historical freeze snapshot", "The exact Version 1 input bytes and chronology anchor"),
            ("Per-run predictions and metrics", "The primitive outcomes from which paper values can be recomputed"),
            ("Aggregation scripts", "How folds, seeds, classes, and conditions become tables and figures"),
            ("Machine audit JSON", "A repeatable record of claim checks and corrected values"),
            ("Paper PDF", "The human-readable scientific narrative, visually checked page by page"),
        ],
        [3000, 6360],
        font_size=8.9,
        first_col_bold=True,
    )
    add_heading(doc, "Five checks to run before submission", 2)
    add_code_block(
        doc,
        [
            "PYTHONPATH=src .venv/bin/python scripts/verify_protocol_freeze.py",
            "PYTHONPATH=src .venv/bin/python scripts/verify_primary_protocol.py --verify-v1-freeze",
            "PYTHONPATH=src .venv/bin/python scripts/audit_paper_numbers.py",
            "PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v",
            "PYTHONPATH=src .venv/bin/python scripts/validate_primary_claims.py",
        ],
    )
    add_body(
        doc,
        "The strict verifier checks the 630-row manifest, 375 primary clips, fold sizes, test identities, "
        "class coverage, source existence, RGB caches, and the historical freeze. The numerical audit reads "
        "stored predictions and verifies the central manuscript tables. The claim validator checks several "
        "primary artifacts and that a compiled PDF exists; PDF text and page layout are checked separately."
    )

    add_heading(doc, "The files a future collaborator should read first", 2)
    add_table(
        doc,
        ["File", "Purpose"],
        [
            ("PAPER_DIRECTION.md", "Live dashboard: status, locks, next action, and stop conditions"),
            ("manuscript/nsl23_low_resource.pdf", "Current Version 1 exploratory paper"),
            ("docs/FINAL_SCAN_2026-10-11.md", "Latest corrections, checkpoint replay, source checks, and verification limits"),
            ("README_REPRODUCIBILITY.md", "Environment, artifacts, rerun boundary, and bundle instructions"),
            ("annotations/v2/REVIEWER_HANDOFF.md", "Recruitment text, exact tapes, workload, definitions, and safe transfer"),
            ("annotations/v2/ANNOTATION_GUIDE.md", "Operational annotation and freeze procedure"),
            ("VERSION2.md", "Full prospective Version 2 scientific protocol"),
        ],
        [3900, 5460],
        font_size=8.7,
        first_col_bold=True,
    )

    add_heading(doc, "Storage discipline", 2)
    add_body(
        doc,
        "The workspace had grown to roughly 23 GiB because it held both 9.6 GiB of official ZIP archives and "
        "9.7 GiB of extracted videos, plus caches and checkpoints. After all four ZIP MD5 hashes and all 630 "
        "extracted paths were verified, only the duplicate ZIP copies were removed. The extracted videos "
        "remain, and the archives can be downloaded again from Zenodo DOI 10.5281/zenodo.10478554."
    )
    for item in (
        "Do not commit source videos, downloaded archives, temporary review exports, smoke checkpoints, render scratch folders, or duplicate results trees.",
        "Keep authoritative predictions, metrics, compact checkpoints, manifests, protocols, audit receipts, manuscript figures, and final deliverables.",
        "Use scripts/manage_workspace_storage.py to verify before deleting recoverable archive duplicates.",
        "Build review tapes selectively with scripts/extract_v2_review_tapes.py instead of unpacking every archive again.",
    ):
        add_list_item(doc, item, bullet_id)


def add_submission_section(doc: Document, bullet_id: int) -> None:
    add_heading(doc, "15. Publication-readiness checklist", 1)
    add_heading(doc, "Version 1 submission", 2)
    for item in (
        "Keep the title, abstract, results, discussion, and conclusion explicit that the study is exploratory development evidence.",
        "Preserve 59.4% ± 3.3% accuracy and corrected 57.4% ± 3.6% pooled macro-F1 for the final model.",
        "Report that ± is sample SD across seeds, not a signer confidence interval.",
        "Keep Table 4’s supported class sets explicit: 36 consonants and 13 vowels.",
        "Retain the task boundary: isolated, manually bounded letters—not spotting, continuous recognition, or live-user accuracy.",
        "Run all five integrity commands, rebuild the PDF, and inspect every page after journal formatting.",
        "Adapt word count, anonymous-review details, citation style, and figure placement only after selecting a target journal.",
    ):
        add_list_item(doc, item, bullet_id)

    add_heading(doc, "Version 2 before annotation", 2)
    for item in (
        "Use the S3 pilot for calibration; keep pilot outcomes disposable.",
        "Make any policy correction before real outcomes and record a new immutable lock.",
        "Confirm two different qualified reviewers for each sealed test-hierarchy tape.",
        "Use the pass-2 export/import package for work on another computer; never send the master CSV.",
        "Back up the master annotation file and hash-bound sidecars after each completed milestone.",
    ):
        add_list_item(doc, item, bullet_id)

    add_heading(doc, "Version 2 after annotation", 2)
    for item in (
        "Save agreement before adjudication and report the agreement statistics honestly.",
        "Run duration, overlap, hash, identity, coverage, and hierarchy validators.",
        "Freeze even if coverage fails; a negative coverage result is data, not permission to move a sealed signer into training.",
        "Proceed to Stage 4 only with at least two complete sealed testers.",
        "Report every tester separately and the equal-weight primary endpoint.",
        "Do enrollment, top-k, broader LOSO, or webcam follow-ups only after the confirmatory batch is sealed and evaluated.",
    ):
        add_list_item(doc, item, bullet_id)

    add_callout(
        doc,
        "Final readiness verdict",
        "Version 1 is scientifically defensible for submission as an exploratory/negative-results paper, "
        "subject to journal fit. Version 2 is scientifically well specified but remains blocked—correctly—by "
        "zero human reviews. The next evidence-producing step is annotation, not training.",
        fill=PALE_GREEN,
        title_color=GREEN,
    )


def add_glossary(doc: Document) -> None:
    add_heading(doc, "Glossary", 1)
    rows = [
        ("Aggregate tape", "A longer recording containing multiple gesture performances rather than one already bounded class clip."),
        ("Annotation", "A human-created record of class, temporal start/end, decision, reviewer, and related quality information."),
        ("BiLSTM", "A recurrent network that reads a sequence forward and backward before combining both directions."),
        ("Cache", "A derived local array saved so expensive video decoding or landmark detection need not be repeated."),
        ("Class", "One of the 49 alphabet labels: 36 consonants or 13 vowels."),
        ("Closed 1-shot", "An exam where one support example per class from the same enrolled signer is available; here it is the other lighting."),
        ("Confirmatory", "An evaluation whose method and test cohort were frozen before test outcomes could guide development."),
        ("Cross entropy", "A loss that penalizes assigning low probability to the true class."),
        ("Cross-signer", "Training and evaluation use different people within a run."),
        ("Fold", "One train/test arrangement. This study has held-out-S1 and held-out-S2 folds."),
        ("Hierarchy hash", "A cryptographic digest binding the fixed V2 signer roles and protocol state."),
        ("Landmark", "A detector-estimated anatomical point on the hand; MediaPipe supplies 21 per detected hand."),
        ("Leakage", "Information from an evaluation performance, person, or derivative entering training or model selection improperly."),
        ("Macro-F1", "The arithmetic mean of per-class F1 values, giving each class equal importance."),
        ("Manifest", "A structured table that says which source files exist and how each sample is classified or assigned."),
        ("MPS", "Apple’s GPU acceleration backend used for the RGB training runs."),
        ("Pass 1 / pass 2", "Two independent reviews of the same original proposal, with pass-1 information hidden from pass 2."),
        ("Prototype", "The mean embedding of the support or training examples for one class."),
        ("Seed", "A number fixing permitted pseudorandom choices so training can be repeated in a controlled way."),
        ("Signer", "A person performing the signs."),
        ("Temporal boundary", "The start and end time/frame of one usable gesture inside a longer recording."),
        ("Top-k", "An evaluation that counts success when the correct class appears among the model’s k highest-ranked choices."),
        ("Unseen signer", "A test person whose clips were not used to train that run’s model or prototypes."),
    ]
    add_table(
        doc,
        ["Term", "Meaning in this project"],
        rows,
        [2400, 6960],
        font_size=8.7,
        first_col_bold=True,
    )
    add_heading(doc, "One-page mental model", 2)
    add_callout(
        doc,
        "Data",
        "Audit first: 375 independent original isolated clips for Version 1; 21 aggregate tapes need temporal annotation for Version 2.",
        fill=PALE_BLUE,
        title_color=DARK_BLUE,
    )
    add_callout(
        doc,
        "Representation",
        "Full-frame RGB scored poorly on these two signers. Wrist-relative landmarks removed direct background pixels; joint angles added finger geometry.",
        fill=PALE_BLUE,
        title_color=DARK_BLUE,
    )
    add_callout(
        doc,
        "Learning",
        "A BiLSTM encodes 16 frames. Prototypical episodes shape the embedding so class means become useful decision anchors.",
        fill=PALE_BLUE,
        title_color=DARK_BLUE,
    )
    add_callout(
        doc,
        "Evidence",
        "59.4% ± 3.3% is the honest Version 1 development result; it is not a population or deployment guarantee.",
        fill=PALE_GOLD,
        title_color=GOLD,
    )
    add_callout(
        doc,
        "Next gate",
        "Two independent NSL-aware reviews, agreement, adjudication, coverage, and freeze—then and only then train Version 2.",
        fill=PALE_GREEN,
        title_color=GREEN,
    )

    add_heading(doc, "Source trail", 2)
    add_body(
        doc,
        "This guide was updated after the repository scan on 11 October 2026. Numerical anchors trace "
        "to artifacts/results, artifacts/audit/full_paper_audit.json, and the manuscript. Operational Version 2 "
        "details trace to PAPER_DIRECTION.md, VERSION2.md, configs/v2_protocol.json, and annotations/v2. For "
        "formal citations to NSL23, MediaPipe, MobileNetV2, LSTM, Transformer, prototypical networks, metrics, "
        "and related NSL work, use the bibliography in manuscript/nsl23_low_resource.pdf."
    )


def build() -> Path:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    bullet_id, decimal_id = configure_document(doc)
    props = doc.core_properties
    props.title = "Understanding the NSL23 Low-Resource Alphabet Study"
    props.subject = "Plain-language technical companion to the NSL23 exploratory paper and Version 2 protocol"
    props.author = "NSL23 project"
    props.keywords = "Nepali Sign Language, NSL23, low-resource recognition, reproducibility, annotation"
    props.comments = "Generated from audited repository artifacts; compact_reference_guide preset."

    add_cover(doc)
    add_reading_map(doc, bullet_id, decimal_id)
    add_dataset_section(doc, bullet_id)
    add_protocol_section(doc, bullet_id, decimal_id)
    add_representation_section(doc, bullet_id)
    add_models_section(doc, bullet_id)
    add_training_section(doc, bullet_id)
    add_metrics_section(doc, bullet_id)
    add_results_section(doc, bullet_id)
    add_confusion_section(doc, bullet_id)
    add_runtime_section(doc, bullet_id)
    add_claims_section(doc, bullet_id)
    add_manual_review_section(doc, bullet_id, decimal_id)
    add_v2_section(doc, bullet_id, decimal_id)
    add_repro_section(doc, bullet_id)
    add_submission_section(doc, bullet_id)
    add_glossary(doc)

    doc.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())
