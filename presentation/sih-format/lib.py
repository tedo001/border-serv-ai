"""Shared helpers for filling the official SIH 2026 idea template.

Everything here writes into the template's own slides rather than building new
ones: the submission rules say to use the provided template without changing
the idea-detail pointers, so those headings stay verbatim and the content is
added around them.
"""

from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

# Sampled from the template itself, so additions sit in its palette.
NAVY = RGBColor(0x22, 0x49, 0x74)      # SIH title blue
BLUE = RGBColor(0x00, 0x71, 0xC1)      # footer bar
STEEL = RGBColor(0x2E, 0x5C, 0x8A)
AMBER = RGBColor(0xE0, 0x8A, 0x1E)
GREEN = RGBColor(0x1E, 0x8E, 0x5A)
RED = RGBColor(0xC0, 0x39, 0x2B)
INK = RGBColor(0x1A, 0x22, 0x2C)
GREY = RGBColor(0x54, 0x60, 0x6D)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
TINT = RGBColor(0xF2, 0xF6, 0xFA)
TINT_W = RGBColor(0xFD, 0xF6, 0xEC)
TINT_G = RGBColor(0xEF, 0xF8, 0xF3)
TINT_R = RGBColor(0xFD, 0xF1, 0xEF)
LINE = RGBColor(0xC9, 0xD6, 0xE4)

HEAD = "Arial"
BODY = "Arial"


def shape_by_id(slide, shape_id):
    for shape in slide.shapes:
        if shape.shape_id == shape_id:
            return shape
    raise KeyError(f"no shape {shape_id} on this slide")


def find_text(slide, needle):
    for shape in slide.shapes:
        if shape.has_text_frame and needle.lower() in shape.text_frame.text.lower():
            return shape
    raise KeyError(f"no shape containing {needle!r}")


def write(text_frame, blocks, *, wrap=True):
    """Replace a text frame's paragraphs.

    ``blocks`` is a list of dicts: text, size, bold, color, space_before,
    space_after, align, bullet. Rebuilt rather than edited run-by-run because
    the template's prompt text has a different number of lines than the content
    replacing it.
    """
    text_frame.clear()
    text_frame.word_wrap = wrap
    for index, block in enumerate(blocks):
        para = text_frame.paragraphs[0] if index == 0 else text_frame.add_paragraph()
        para.alignment = block.get("align", PP_ALIGN.LEFT)
        if block.get("space_before"):
            para.space_before = Pt(block["space_before"])
        if block.get("space_after") is not None:
            para.space_after = Pt(block["space_after"])
        run = para.add_run()
        run.text = block["text"]
        font = run.font
        font.name = block.get("font", BODY)
        font.size = Pt(block.get("size", 14))
        font.bold = block.get("bold", False)
        font.italic = block.get("italic", False)
        font.color.rgb = block.get("color", INK)
    return text_frame


def textbox(slide, x, y, w, h, blocks, *, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.margin_left = tf.margin_right = Inches(0.04)
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    write(tf, blocks)
    return box


def panel(slide, x, y, w, h, fill, outline=LINE, *, radius=True):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if outline is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = outline
        shape.line.width = Pt(1)
    shape.shadow.inherit = False
    if radius:
        shape.adjustments[0] = 0.08
    return shape


def card(slide, x, y, w, h, title, lines, colour, fill, *, title_size=12,
         body_size=11, gap=0.06):
    """A titled card: solid caption strip, tinted body, bulleted lines."""
    panel(slide, x, y, w, h, fill)
    cap = panel(slide, x, y, w, 0.34, colour, outline=None)
    tf = cap.text_frame
    tf.margin_left = tf.margin_right = Inches(0.06)
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    write(tf, [{"text": title, "size": title_size, "bold": True, "color": WHITE,
                "font": HEAD, "align": PP_ALIGN.CENTER}])

    blocks = []
    for i, line in enumerate(lines):
        if isinstance(line, tuple):
            label, rest = line
            blocks.append({"text": f"{label} {rest}", "size": body_size,
                           "color": INK, "space_before": 0 if i == 0 else 4,
                           "space_after": 0})
        else:
            blocks.append({"text": line, "size": body_size, "color": INK,
                           "space_before": 0 if i == 0 else 4, "space_after": 0})
    textbox(slide, x + 0.12, y + 0.34 + gap, w - 0.24, h - 0.34 - gap * 2, blocks)


def chip(slide, x, y, w, h, text, fill, colour=WHITE, size=10, bold=True):
    """A solid label block. ``text`` may be a string or a list of lines.

    Each line becomes its own paragraph rather than a newline inside one run:
    alignment is a paragraph property, so a run carrying "A\nB" centres only
    the part after the break and the first line sits left of the rest.
    """
    lines = text if isinstance(text, (list, tuple)) else [text]
    shape = panel(slide, x, y, w, h, fill, outline=None)
    tf = shape.text_frame
    tf.margin_left = tf.margin_right = Inches(0.04)
    tf.margin_top = tf.margin_bottom = 0
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    write(tf, [
        {"text": line, "size": size if i == 0 else size - 0.5,
         "bold": bold if i == 0 else False, "color": colour, "font": HEAD,
         "align": PP_ALIGN.CENTER, "space_after": 0}
        for i, line in enumerate(lines)
    ])
    return shape


def arrow(slide, x, y, w, h=0.12, colour=STEEL):
    shape = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y),
                                   Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = colour
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def pointer(slide, x, y, w, text, size=13):
    """The template's own idea-detail pointer, kept verbatim as a section label."""
    return textbox(slide, x, y, w, 0.26, [
        {"text": text, "size": size, "bold": True, "color": BLUE, "font": HEAD},
    ])
