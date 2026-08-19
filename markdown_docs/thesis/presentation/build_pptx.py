#!/usr/bin/env python3
import re
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image as PILImage

IMG = "/home/tpioda/Bachelor-Thesis/markdown_docs/thesis/images/"

# ---------------------------------------------------------------- palette --
BG          = RGBColor(0xF3, 0xF4, 0xF0)
SURFACE     = RGBColor(0xFF, 0xFF, 0xFF)
SURFACE2    = RGBColor(0xE9, 0xEA, 0xE4)
INK         = RGBColor(0x1B, 0x20, 0x23)
INK_SOFT    = RGBColor(0x4A, 0x52, 0x57)
INK_FAINT   = RGBColor(0x7B, 0x83, 0x88)
LINE        = RGBColor(0xD7, 0xD9, 0xD1)
BLEED       = RGBColor(0xB2, 0x4F, 0x2C)
BLEED_SOFT  = RGBColor(0xF1, 0xDC, 0xD1)
ISCH        = RGBColor(0x21, 0x5F, 0x73)
ISCH_SOFT   = RGBColor(0xD8, 0xE5, 0xE8)

SERIF = "Georgia"
SANS  = "Calibri"
MONO  = "Consolas"

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = 13.333, 7.5
BLANK = prs.slide_layouts[6]

TOTAL_SLIDES = 19
_slide_count = [0]

# ------------------------------------------------------------- primitives --
def new_slide():
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    _slide_count[0] += 1
    add_footer(s, _slide_count[0], TOTAL_SLIDES)
    return s

def add_footer(slide, idx, total):
    tb = slide.shapes.add_textbox(Inches(0.75), Inches(7.08), Inches(5), Inches(0.32))
    tf = tb.text_frame; tf.word_wrap = False
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = "CATE IN MASTER DAPT  ·  T. PIODA"
    r.font.size = Pt(8); r.font.name = MONO; r.font.color.rgb = INK_FAINT
    tb2 = slide.shapes.add_textbox(Inches(11.6), Inches(7.08), Inches(1.0), Inches(0.32))
    tf2 = tb2.text_frame; tf2.word_wrap = False
    p2 = tf2.paragraphs[0]; p2.alignment = PP_ALIGN.RIGHT
    r2 = p2.add_run(); r2.text = f"{idx:02d} / {total:02d}"
    r2.font.size = Pt(8); r2.font.name = MONO; r2.font.color.rgb = INK_FAINT

def _runs_from_markup(p, text, base_size, base_color, bold_color=None, font=SANS):
    parts = re.split(r'(\*\*.*?\*\*)', text)
    for part in parts:
        if not part:
            continue
        r = p.add_run()
        if part.startswith("**") and part.endswith("**"):
            r.text = part[2:-2]
            r.font.bold = True
            r.font.color.rgb = bold_color or INK
        else:
            r.text = part
            r.font.color.rgb = base_color
        r.font.size = Pt(base_size)
        r.font.name = font

def add_eyebrow(slide, label, color=INK_FAINT, x=0.75, y=0.42):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(9), Inches(0.35))
    tf = tb.text_frame; tf.word_wrap = False
    p = tf.paragraphs[0]
    r1 = p.add_run(); r1.text = "●  "
    r1.font.size = Pt(11); r1.font.name = SANS; r1.font.color.rgb = color
    r2 = p.add_run(); r2.text = label.upper()
    r2.font.size = Pt(11); r2.font.name = MONO; r2.font.bold = True; r2.font.color.rgb = color
    return tb

def add_title(slide, text, x=0.75, y=0.82, w=11.6, h=1.25, size=32):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.name = SERIF; r.font.color.rgb = INK
    return tb

def add_kicker(slide, text, x=0.75, y=1.55, w=9.5, h=0.6, size=15, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = 1.3
        r = p.add_run(); r.text = line
        r.font.size = Pt(size); r.font.italic = True; r.font.name = SERIF; r.font.color.rgb = INK_SOFT
    return tb

def add_bullets(slide, items, x, y, w, h, dot_color=INK_FAINT, size=14, gap=12, line_spacing=1.15):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    first = True
    for item in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(gap)
        p.line_spacing = line_spacing
        rdot = p.add_run(); rdot.text = "●  "
        rdot.font.size = Pt(size - 3); rdot.font.name = SANS; rdot.font.color.rgb = dot_color
        _runs_from_markup(p, item, size, INK_SOFT, bold_color=INK)
    return tb

def add_card(slide, x, y, w, h, fill=SURFACE, border=LINE):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    try:
        shp.adjustments[0] = 0.045
    except Exception:
        pass
    shp.fill.solid(); shp.fill.fore_color.rgb = fill
    shp.line.color.rgb = border; shp.line.width = Pt(0.75)
    shp.shadow.inherit = False
    return shp

def add_mini_card(slide, x, y, w, h, header, body, header_color=INK_FAINT, dashed=False):
    fill = SURFACE2 if dashed else SURFACE
    add_card(slide, x, y, w, h, fill=fill)
    tb = slide.shapes.add_textbox(Inches(x + 0.16), Inches(y + 0.13), Inches(w - 0.32), Inches(h - 0.26))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = header.upper()
    r.font.size = Pt(9.5); r.font.bold = True; r.font.name = MONO; r.font.color.rgb = header_color
    p2 = tf.add_paragraph(); p2.space_before = Pt(6); p2.line_spacing = 1.15
    _runs_from_markup(p2, body, 11, INK_SOFT, bold_color=INK)
    return tb

def add_callout(slide, text, x, y, w, h, color=INK, size=16):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(0.035), Inches(h))
    bar.fill.solid(); bar.fill.fore_color.rgb = color; bar.line.fill.background(); bar.shadow.inherit = False
    tb = slide.shapes.add_textbox(Inches(x + 0.2), Inches(y - 0.06), Inches(w - 0.2), Inches(h + 0.15))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.line_spacing = 1.2
    _runs_from_markup(p, text, size, INK, bold_color=INK, font=SERIF)
    for r in p.runs:
        r.font.italic = True
    return tb

def add_stat(slide, x, y, w, num, label, color=INK):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(0.62))
    tf = tb.text_frame; tf.word_wrap = False
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = num
    r.font.size = Pt(27); r.font.name = SERIF; r.font.color.rgb = color
    tb2 = slide.shapes.add_textbox(Inches(x), Inches(y + 0.56), Inches(w), Inches(0.4))
    tf2 = tb2.text_frame; tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    r2 = p2.add_run(); r2.text = label.upper()
    r2.font.size = Pt(9); r2.font.name = MONO; r2.font.color.rgb = INK_FAINT
    return tb

def add_tag(slide, x, y, text, color, soft):
    w = 0.16 + 0.092 * len(text)
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(0.32))
    try:
        shp.adjustments[0] = 0.4
    except Exception:
        pass
    shp.fill.solid(); shp.fill.fore_color.rgb = soft
    shp.line.fill.background(); shp.shadow.inherit = False
    tf = shp.text_frame; tf.word_wrap = False
    tf.margin_left = Inches(0.08); tf.margin_right = Inches(0.08)
    tf.margin_top = Inches(0.02); tf.margin_bottom = Inches(0.02)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = text
    r.font.size = Pt(10); r.font.bold = True; r.font.name = MONO; r.font.color.rgb = color
    return w

def add_table_rows(slide, x, y, w, col_fracs, headers, rows, row_h=0.42, row_colors=None):
    xs = [x]
    for f in col_fracs:
        xs.append(xs[-1] + f * w)
    # header
    for i, htxt in enumerate(headers):
        tb = slide.shapes.add_textbox(Inches(xs[i]), Inches(y), Inches(xs[i+1]-xs[i]), Inches(0.3))
        tf = tb.text_frame; tf.word_wrap = False
        p = tf.paragraphs[0]
        r = p.add_run(); r.text = htxt.upper()
        r.font.size = Pt(9.5); r.font.bold = True; r.font.name = MONO; r.font.color.rgb = INK_FAINT
    ln = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y + 0.34), Inches(w), Pt(1))
    ln.fill.solid(); ln.fill.fore_color.rgb = LINE; ln.line.fill.background(); ln.shadow.inherit = False
    cy = y + 0.42
    for ri, row in enumerate(rows):
        color = INK
        if row_colors and ri < len(row_colors) and row_colors[ri]:
            color = row_colors[ri]
        for ci, val in enumerate(row):
            tb = slide.shapes.add_textbox(Inches(xs[ci]), Inches(cy), Inches(xs[ci+1]-xs[ci]), Inches(row_h - 0.06))
            tf = tb.text_frame; tf.word_wrap = False
            p = tf.paragraphs[0]
            r = p.add_run(); r.text = val
            r.font.size = Pt(12.5); r.font.name = (MONO if ci > 0 else SANS)
            r.font.color.rgb = color
            r.font.bold = (color != INK)
        ln2 = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(cy + row_h - 0.06), Inches(w), Pt(0.75))
        ln2.fill.solid(); ln2.fill.fore_color.rgb = LINE; ln2.line.fill.background(); ln2.shadow.inherit = False
        cy += row_h
    return cy

def add_image_card(slide, path, x, y, maxw, maxh, caption=None, center_in_w=None):
    im = PILImage.open(path)
    iw, ih = im.size
    ar = iw / ih
    w = maxw
    h = w / ar
    if h > maxh:
        h = maxh
        w = h * ar
    pad = 0.16
    card_w = w + pad * 2
    card_h = h + pad * 2
    box_w = center_in_w if center_in_w else maxw
    cx = x + (box_w - card_w) / 2
    add_card(slide, cx, y, card_w, card_h, fill=SURFACE)
    slide.shapes.add_picture(path, Inches(cx + pad), Inches(y + pad), width=Inches(w), height=Inches(h))
    bottom = y + card_h
    if caption:
        cap_h = min(0.55, max(0.3, 7.35 - (bottom + 0.06)))
        cap = slide.shapes.add_textbox(Inches(cx), Inches(bottom + 0.06), Inches(card_w), Inches(cap_h))
        tf = cap.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.line_spacing = 1.15
        r = p.add_run(); r.text = caption
        r.font.size = Pt(8.5); r.font.name = MONO; r.font.color.rgb = INK_FAINT
    return bottom

# =============================================================== SLIDE 1 ==
s = new_slide()
tb = s.shapes.add_textbox(Inches(0.75), Inches(0.7), Inches(9), Inches(0.4))
tf = tb.text_frame; p = tf.paragraphs[0]
r = p.add_run(); r.text = "SUPSI · DTI — DIPLOMA THESIS      ●  bleeding      ●  ischaemia"
r.font.size = Pt(11); r.font.name = MONO; r.font.color.rgb = INK_FAINT
add_title(s, "CATE Estimation and Use in the MASTER DAPT Study", x=0.75, y=2.15, w=10.8, h=1.7, size=40)
add_kicker(s, "Can we find, validate, and act on individual treatment effect heterogeneity in a "
              "randomized cardiology trial — and if we can't, why not?", x=0.75, y=3.55, w=9.6, h=0.9, size=17)
meta = [("Candidate", "Tommaso Pioda"), ("Supervisor", "Prof. Laura Azzimonti"),
        ("Co-supervisor", "Gabriele Maroni"), ("Client", "EOC"), ("Programme", "Data Science & AI")]
mx = 0.75
for k, v in meta:
    tb = s.shapes.add_textbox(Inches(mx), Inches(4.9), Inches(2.15), Inches(0.8))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = k.upper()
    r.font.size = Pt(9); r.font.name = MONO; r.font.color.rgb = INK_FAINT
    p2 = tf.add_paragraph(); p2.space_before = Pt(4)
    r2 = p2.add_run(); r2.text = v
    r2.font.size = Pt(12.5); r2.font.name = SANS; r2.font.color.rgb = INK
    mx += 2.15

# =============================================================== SLIDE 2 ==
s = new_slide()
add_eyebrow(s, "The problem")
add_title(s, "One trial, two ways to be wrong")
body = ("After a coronary stent, dual antiplatelet therapy (DAPT) sits on two axes that pull in "
        "opposite directions: longer therapy protects against ischaemic events, shorter therapy "
        "protects against bleeding.\n\n"
        "MASTER DAPT randomized 4,579 high-bleeding-risk patients between abbreviated and prolonged "
        "DAPT, and concluded in favour of abbreviation — non-inferior on ischaemia, superior on bleeding.")
tb = s.shapes.add_textbox(Inches(0.75), Inches(2.15), Inches(6.6), Inches(2.4))
tf = tb.text_frame; tf.word_wrap = True
for i, para in enumerate(body.split("\n\n")):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.space_after = Pt(14); p.line_spacing = 1.3
    r = p.add_run(); r.text = para
    r.font.size = Pt(14); r.font.name = SANS; r.font.color.rgb = INK_SOFT
add_callout(s, "That is an average result. A favourable trial-level effect is compatible with "
               "subgroups that gain nothing — or are harmed.", x=0.75, y=4.7, w=6.6, h=1.0)
add_stat(s, 8.0, 2.3, 4.3, "↓ bleeding", "abbreviated DAPT", color=BLEED)
add_stat(s, 8.0, 3.25, 4.3, "↑ ischaemia risk", "if therapy is too short", color=ISCH)

# =============================================================== SLIDE 3 ==
s = new_slide()
add_eyebrow(s, "The problem")
add_title(s, "Why “treat the highest-risk patient” is not an answer")
add_bullets(s, [
    "A risk model ranks patients by **P(event | treatment received)** — one world, the one we observed.",
    "A treatment decision needs the **difference between two worlds**, and only one of them is ever observed for any patient. This is the fundamental problem of causal inference.",
    "The highest-risk patient may gain the most from shortening therapy, the least, or be harmed by it — risk alone cannot tell the three apart.",
], x=0.75, y=2.2, w=10.8, h=2.6, size=15, gap=16)
add_callout(s, "Risk does not rank benefit.", x=0.75, y=5.2, w=9, h=0.6, size=20)

# =============================================================== SLIDE 4 ==
s = new_slide()
add_eyebrow(s, "Research questions")
add_title(s, "Two questions, in order")
add_mini_card(s, 0.75, 2.3, 5.6, 2.0, "Q1 — Estimate & validate",
              "Can the individual treatment effect (CATE) be estimated at all — and, since the true "
              "individual effect is never observed, can it be **validated**?")
add_mini_card(s, 6.55, 2.3, 5.6, 2.0, "Q2 — Exploit",
              "If a credible effect exists, can enrolment be **concentrated** on the patients where "
              "it is visible — reaching the same precision with fewer patients?")
add_callout(s, "Five estimator families answer Q1. Five enrolment mechanisms answer Q2 — once Q1's "
               "answer changes what Q2 has to test.", x=0.75, y=4.75, w=10.8, h=1.0)

# =============================================================== SLIDE 5 ==
s = new_slide()
add_eyebrow(s, "Method — estimation")
add_title(s, "Five estimator families, chosen to err differently")
cards5 = [
    ("Meta-learner", "Two outcome models, one per arm; CATE is their difference. Simple, inherits every model error."),
    ("Causal Forest", "Estimates heterogeneity directly; double ML removes confounding, honest splits avoid overstating structure."),
    ("Bayesian Causal Forest", "Regularizes the effect surface separately from the prognostic one — a confounder can't disguise itself as heterogeneity."),
    ("In-context model", "Pre-trained on synthetic causal problems, not tuned to this dataset — a bias-driven, variance-independent control."),
    ("Interaction Forest", "One model on [X, T, X·T]; pools information across arms, compresses weak interactions toward zero."),
    ("Why five?", "Not “which is best” — a result that survives all five stops being attributable to any single algorithm's quirks."),
]
gx, gy, gw, gh, gap = 0.75, 2.25, 3.63, 1.55, 0.22
for i, (h, b) in enumerate(cards5):
    col = i % 3; row = i // 3
    x = gx + col * (gw + gap)
    y = gy + row * (gh + gap)
    add_mini_card(s, x, y, gw, gh, h, b, dashed=(h == "Why five?"))

# =============================================================== SLIDE 6 ==
s = new_slide()
add_eyebrow(s, "Method — validation")
add_title(s, "Validating what can never be observed")
add_bullets(s, [
    "τ(x) is never observed for **any** patient — CATE can't be checked prediction-vs-truth like a risk model.",
    "Instead, validate the **ranking**: the TOC curve — treat the top q%, gain vs. treating everyone. **AUTOC** weights the head of the ranking, where the decision happens.",
    "Evaluation is doubly-robust, plus a group-calibration check between estimated and observed CATE bands.",
    "Hyperparameters are tuned on the **same** metric used to validate — the model is trained for exactly the objective that matters.",
], x=0.75, y=2.25, w=6.7, h=4.2, size=13.5, gap=14)
add_mini_card(s, 7.75, 2.6, 4.6, 3.1, "The noise floor",
              "Even a random CATE produces some ranking dispersion. So: permute the treatment labels, "
              "re-run the whole pipeline — that distribution is chance. A result counts only if it "
              "clears **this floor**, not zero.")

# =============================================================== SLIDE 7 ==
s = new_slide()
add_eyebrow(s, "Case study — data")
add_title(s, "MASTER DAPT: the cohort")
bottom = add_table_rows(s, 0.75, 2.3, 5.6,
                         [0.55, 0.22, 0.23],
                         ["Endpoint (335 d)", "Events", "Rate"],
                         [["Bleeding", "506", "11.1%"],
                          ["BARC 2/3/5", "359", "7.8%"],
                          ["Myocardial infarction", "109", "2.4%"],
                          ["CV death", "81", "1.8%"],
                          ["Stroke", "35", "0.8%"]],
                         row_colors=[BLEED, None, ISCH, ISCH, ISCH])
add_callout(s, "4,579 randomized · 63 baseline covariates · T=1 = prolonged, so CATE always reads "
               "as the benefit of shortening.", x=0.75, y=bottom + 0.35, w=5.6, h=1.3)
add_image_card(s, IMG + "masterdapt_flow.png", 6.7, 1.85, 5.3, 4.15,
                caption="All 4,579 patients were enrolled because they were at high bleeding risk "
                        "— a design criterion, not a coincidence.")

# =============================================================== SLIDE 8 ==
s = new_slide()
add_eyebrow(s, "Case study — risk models")
add_title(s, "A predictive anomaly, left open on purpose")
add_image_card(s, IMG + "ROC_PR_out_of_fold.png", 0.75, 1.95, 7.0, 3.6,
                caption="Five endpoint-specific risk models, evaluated out-of-fold, threshold-free, "
                        "over the whole cohort.")
tb = s.shapes.add_textbox(Inches(8.15), Inches(2.0), Inches(4.4), Inches(1.0))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.line_spacing = 1.3
r = p.add_run(); r.text = "If risk tracked event frequency, bleeding — 506 events — should be the easiest endpoint to predict. It is the hardest."
r.font.size = Pt(12.5); r.font.name = SANS; r.font.color.rgb = INK_SOFT
add_stat(s, 8.15, 3.1, 1.8, "0.76", "CV death · 81 ev.", color=ISCH)
add_stat(s, 9.95, 3.1, 1.8, "0.72", "MI · 109 ev.", color=ISCH)
add_stat(s, 8.15, 4.15, 1.8, "0.62", "Bleeding · 506 ev.", color=BLEED)
add_callout(s, "Explained in full only in chapter 7 — the same cause reappears there for the causal result.",
            x=8.15, y=5.15, w=4.4, h=1.1)

# =============================================================== SLIDE 9 ==
s = new_slide()
add_eyebrow(s, "Results — 1 of 2", color=BLEED)
add_title(s, "The positive control: the tools work")
tb = s.shapes.add_textbox(Inches(0.75), Inches(2.15), Inches(10.8), Inches(0.5))
tf = tb.text_frame; p = tf.paragraphs[0]
r = p.add_run(); r.text = "All five estimator families independently recover the trial's average effect."
r.font.size = Pt(15); r.font.name = SANS; r.font.color.rgb = INK_SOFT
add_stat(s, 0.75, 2.9, 4, "+0.045 → +0.055", "bleeding CATE, mean (abbreviating helps)", color=BLEED)
add_stat(s, 4.9, 2.9, 3, "≈ 0", "ischaemic CATE, mean", color=ISCH)
add_stat(s, 8.0, 2.9, 3, "5 / 5", "estimator families agree", color=INK)
add_callout(s, "Consistent with the published trial, across five families that make different "
               "mistakes. The average effect is not in question.", x=0.75, y=4.5, w=9.5, h=1.0, color=BLEED)

# ============================================================== SLIDE 10 ==
s = new_slide()
add_eyebrow(s, "Results — 2 of 2", color=ISCH)
add_title(s, "The negative result: no heterogeneity above chance")
add_bullets(s, [
    "Individual CATE dispersion sits **on** the permutation noise floor, not above it.",
    "AUTOC is compatible with zero; credible bands cover the ATE across the whole ranking.",
    "Holds for **all five** independent estimator families.",
], x=0.75, y=2.2, w=6.2, h=2.2, dot_color=ISCH, size=14, gap=16)
add_callout(s, "Not “I found nothing” — “I searched with tools proven to work, and there is nothing.”",
            x=0.75, y=4.55, w=6.2, h=1.4, color=ISCH)
add_image_card(s, IMG + "CATE_placebo_floor.png", 7.2, 1.95, 5.2, 4.5,
                caption="Observed sd(τ(x)) (red) overlapping the placebo floor (grey) — bleeding and BARC 2/3/5.")

# ============================================================== SLIDE 11 ==
s = new_slide()
add_eyebrow(s, "Results — the trade-off plane")
add_title(s, "Where is the dilemma patient?")
add_image_card(s, IMG + "CATE_tradeoff_plane.png", 0.75, 1.95, 7.3, 4.6,
                caption="Every patient placed by bleeding-benefit (x) and ischaemic-benefit (y) of abbreviating.")
tb = s.shapes.add_textbox(Inches(8.4), Inches(2.1), Inches(4.2), Inches(1.4))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.line_spacing = 1.3
r = p.add_run(); r.text = "A personalized rule only pays off if a quadrant of real dilemma cases exists — large gain on one axis, real harm on the other."
r.font.size = Pt(13); r.font.name = SANS; r.font.color.rgb = INK_SOFT
add_callout(s, "Instead: a compact cloud around the trial's average effect. No quadrant to split on.",
            x=8.4, y=3.7, w=4.2, h=1.4, color=BLEED)

# ============================================================== SLIDE 12 ==
s = new_slide()
add_eyebrow(s, "Results — decision value")
add_title(s, "Even the weaker question gives the same answer")
tb = s.shapes.add_textbox(Inches(0.75), Inches(2.1), Inches(10.8), Inches(0.7))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]
_runs_from_markup(p, "Not “does the model rank well” — **does following the ranking produce a measurable "
                      "clinical gain**, evaluated with cross-fitted AIPW policy value.", 14.5, INK_SOFT, bold_color=INK)
cards12 = [
    ("Best rule vs. everyone", "No rule beats “abbreviate for everyone.” Net-neutral."),
    ("7 pre-specified subgroups", "Bleeding risk, ACS, diabetes, renal function, age, anaemia, OAC — none exclude zero."),
    ("Trade-off frontier", "Risk-magnification frontier is empty at every bleeding/ischaemic weight."),
]
gw, gh, gap = 3.63, 2.0, 0.22
for i, (h, b) in enumerate(cards12):
    x = 0.75 + i * (gw + gap)
    add_mini_card(s, x, 3.1, gw, gh, h, b)

# ============================================================== SLIDE 13 ==
s = new_slide()
add_eyebrow(s, "Results — how sure are we?")
add_title(s, "The minimum detectable δ")
add_image_card(s, IMG + "delta_sensitivity.png", 0.75, 1.95, 6.85, 4.25,
                caption="Synthetic heterogeneity of known size δ, injected and re-estimated; "
                        "compared against the floor recomputed at the same δ.")
bottom = add_table_rows(s, 8.05, 2.1, 4.5,
                         [0.55, 0.45],
                         ["Endpoint", "δ*"],
                         [["Bleeding", "0.05 risk-diff"],
                          ["BARC 2/3/5", "0.05 risk-diff"],
                          ["MI", "OR 3.0"],
                          ["CV death", "OR 5.0"]],
                         row_colors=[BLEED, BLEED, ISCH, ISCH])
add_callout(s, "On bleeding, δ* ≈ the entire average effect. It was not found — that is not the same claim as it does not exist.",
            x=8.05, y=bottom + 0.35, w=4.5, h=1.7)

# ============================================================== SLIDE 14 ==
s = new_slide()
add_eyebrow(s, "Why — three converging explanations")
add_title(s, "Not just “nothing found” — a mechanism")
cards14 = [
    ("01 · Cancellation", "The aggregate ischaemic score sums MI and stroke, which move in **opposite directions** — the sum hides both."),
    ("02 · Range restriction", "Enrolling only high-bleeding-risk patients compresses bleeding-risk variance before any model sees a datum — the same cause behind the ch.5 anomaly."),
    ("03 · Rare events", "MI, death, stroke: 35–109 events. Three endpoints of five are **not measurable** here, not “flat.”"),
]
gw, gh, gap = 3.63, 2.1, 0.22
for i, (h, b) in enumerate(cards14):
    x = 0.75 + i * (gw + gap)
    add_mini_card(s, x, 2.25, gw, gh, h, b)
add_callout(s, "The same feature of the trial's design — range restriction on bleeding risk — "
               "explains both the ch.5 predictive anomaly and the ch.6 absence of heterogeneity.",
            x=0.75, y=4.75, w=10.8, h=1.1)

# ============================================================== SLIDE 15 ==
s = new_slide()
add_eyebrow(s, "Testing it end-to-end")
add_title(s, "Guided enrolment: an exploitability test")
add_bullets(s, [
    "7 acquisition mechanisms — offline ceiling, pairwise duel, UCB1 / Thompson bandits, whole-pool ranking, ±45° geometry — over **100 paired cohorts**.",
    "Mechanisms **do** move the enrolled cohort on the estimated-CATE plane, stably and distinguishably from random.",
    "On **observed outcomes**, the shift is ≈ zero — even for the adaptive bandits.",
], x=0.75, y=2.15, w=6.7, h=2.9, size=14, gap=16)
add_mini_card(s, 7.75, 2.4, 4.6, 2.5, "ρ ≈ 0.91",
              "Win-win score correlates with the bleeding axis alone — confirms the cancellation.")
tb = s.shapes.add_textbox(Inches(7.9), Inches(4.55), Inches(4.4), Inches(0.9))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]
r = p.add_run(); r.text = "The adaptive cohort still loses to a plain randomized trial at equal enrolled n."
r.font.size = Pt(11.5); r.font.name = SANS; r.font.color.rgb = INK_SOFT
add_callout(s, "The machinery selects well on a signal that does not exist.",
            x=0.75, y=5.6, w=10.8, h=0.7, size=18)

# ============================================================== SLIDE 16 ==
s = new_slide()
add_eyebrow(s, "Conclusions — what holds")
add_title(s, "Six claims, each independently supported")
add_bullets(s, [
    "The trial's **average effect** is confirmed by five independent estimators.",
    "No exploitable heterogeneity — validated against the **noise floor**, not against zero.",
    "The same conclusion arrives a **second time**, from ch.8's observed events, without touching a single estimated CATE.",
    "The limit is in the **cohort**, not the tools — shown twice, by the positive control and by δ*.",
    "The mechanism is known: **HBR range restriction**, which also explains the ch.5 predictive anomaly.",
    "Two reusable findings beyond the case study: selecting ≠ allocating; linear scores collapse onto the higher-variance axis.",
], x=0.75, y=2.15, w=10.9, h=4.9, size=14, gap=13)

# ============================================================== SLIDE 17 ==
s = new_slide()
add_eyebrow(s, "Conclusions — limits")
add_title(s, "What doesn't hold, said before it's asked")
add_bullets(s, [
    "δ* = 0.05 is **large** — these data exclude heterogeneity the size of the whole ATE, not any heterogeneity.",
    "**3 of 5 endpoints** (35–109 events) are not assessable at all — “not measurable,” not “absent.”",
    "Stroke sensitivity analysis is not yet interpretable.",
    "Subgroup analyses are in-sample and descriptive; no external cohort; multiplicity declared but not formally corrected.",
], x=0.75, y=2.2, w=10.9, h=4.4, size=15, gap=18)

# ============================================================== SLIDE 18 ==
s = new_slide()
add_eyebrow(s, "Where to ask the same question next")
add_title(s, "The infrastructure is ready — the cohort wasn't")
add_bullets(s, [
    "A cohort **not** selected purely on bleeding risk — where the bleeding axis keeps its full variance.",
    "Longer follow-up, or a higher-ischaemic-risk population — 35 stroke events measure nothing, whatever the estimator.",
    "A trial **sized on the interaction**, not the average effect — δ* is exactly the tool to pre-register that target.",
], x=0.75, y=2.2, w=10.9, h=2.6, size=15, gap=18)
add_callout(s, "The guided-enrolment machinery of ch.8 is built, measured, and already equipped with "
               "its two known failure modes — ready for a cohort where heterogeneity actually exists.",
            x=0.75, y=5.0, w=10.9, h=1.3)

# ============================================================== SLIDE 19 ==
s = new_slide()
tb = s.shapes.add_textbox(Inches(0), Inches(2.7), Inches(SW), Inches(0.5))
tf = tb.text_frame; p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
r1 = p.add_run(); r1.text = "●  "
r1.font.size = Pt(11); r1.font.color.rgb = INK_FAINT
r2 = p.add_run(); r2.text = "THANK YOU"
r2.font.size = Pt(11); r2.font.bold = True; r2.font.name = MONO; r2.font.color.rgb = INK_FAINT
tb = s.shapes.add_textbox(Inches(0), Inches(3.15), Inches(SW), Inches(1.2))
tf = tb.text_frame; p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
r = p.add_run(); r.text = "Questions"
r.font.size = Pt(46); r.font.name = SERIF; r.font.color.rgb = INK
add_kicker(s, "Tommaso Pioda · CATE Estimation and Use in the MASTER DAPT Study\n"
              "Supervisor: Prof. Laura Azzimonti — Co-supervisor: Gabriele Maroni",
           x=0, y=4.35, w=SW, h=1.0, size=14, align=PP_ALIGN.CENTER)

# ------------------------------------------------------------------ save --
out = "/tmp/claude-1011/-home-tpioda-Bachelor-Thesis/1dfd5533-e0ad-4d86-8615-d1b88719c426/scratchpad/CATE_MASTER_DAPT.pptx"
prs.save(out)
print("Saved:", out)
print("Slides:", len(list(prs.slides)))
