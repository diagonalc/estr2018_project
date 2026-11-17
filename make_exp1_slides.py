"""Build a 2-slide PowerPoint for Experiment 1 from the recorded results."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
PLOTS = os.path.join(ROOT, "plots")

BLUE = RGBColor(0x25, 0x63, 0xEB)
DARK = RGBColor(0x1F, 0x29, 0x37)
GRAY = RGBColor(0x6B, 0x72, 0x80)
LIGHT = RGBColor(0xF3, 0xF4, 0xF6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
ORANGE = RGBColor(0xEA, 0x58, 0x0C)


def make_loglog_figure():
    init_df = pd.read_csv(os.path.join(RESULTS, "exp1_init.csv"))
    ep_df = pd.read_csv(os.path.join(RESULTS, "exp1_epoch1.csv"))
    with open(os.path.join(RESULTS, "exp1_init_summary.json")) as f:
        init_s = json.load(f)
    with open(os.path.join(RESULTS, "exp1_epoch1_summary.json")) as f:
        ep_s = json.load(f)

    fig, ax = plt.subplots(figsize=(7.8, 4.6), dpi=200)
    for df, s, color, label in [
        (init_df, init_s, "#2563EB", "init"),
        (ep_df, ep_s, "#EA580C", "after 1 epoch"),
    ]:
        B = df["batch_size"].values.astype(float)
        noise = df["noise_sq"].values
        ax.scatter(B, noise, s=58, color=color, zorder=3)
        xs = np.logspace(np.log10(B.min()), np.log10(B.max()), 100)
        ys = 10 ** (s["intercept"] + s["slope"] * np.log10(xs))
        ax.plot(xs, ys, color=color, lw=2.3,
                label=f"{label}: slope {s['slope']:.3f}, $R^2$={s['r2']:.4f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("batch size B", fontsize=14)
    ax.set_ylabel(r"noise$^2$ = E$\,\|g_B - g\|^2$", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10.5, loc="lower left", framealpha=0.95)
    ax.text(0.97, 0.95, r"theory: noise$^2 \propto 1/B$",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, color="#374151")
    fig.tight_layout()
    path = os.path.join(PLOTS, "slide_exp1_loglog.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def make_hist_figure():
    data = np.load(os.path.join(RESULTS, "exp1_init_tracked.npz"))
    vals = data["tracked"]              # (200 batches, 1000 coordinates)
    var_per_coord = vals.var(axis=0, ddof=1)
    k = int(np.argmax(var_per_coord))
    sample = vals[:, k]
    mu, sigma = float(sample.mean()), float(sample.std(ddof=1))
    g_val = float(data["g_tracked"][k])

    fig, ax = plt.subplots(figsize=(4.8, 2.55), dpi=200)
    ax.hist(sample, bins=22, density=True, alpha=0.7, color="#2563EB",
            label="200 batch gradients")
    xs = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    ax.plot(xs, np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
            / (sigma * np.sqrt(2 * np.pi)),
            color="#DC2626", lw=2, label="normal fit")
    ax.axvline(g_val, color="black", ls="--", lw=1.6, label="full gradient g")
    ax.set_xlabel("gradient value at one coordinate", fontsize=10.5)
    ax.set_ylabel("density", fontsize=10.5)
    ax.tick_params(labelsize=9)
    ax.set_title("B = 64, one coordinate (init)", fontsize=11)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOTS, "slide_exp1_clt.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def add_text(slide, x, y, w, h, text, size=18, bold=False, color=DARK,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = "Calibri"
    return box


def add_bullets(slide, x, y, w, h, items, size=14, color=DARK,
                space=8, bullet_color=BLUE):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space)
        r = p.add_run()
        r.text = "•  "
        r.font.bold = True
        r.font.size = Pt(size)
        r.font.color.rgb = bullet_color
        r.font.name = "Calibri"
        if isinstance(item, tuple):
            lead, rest = item
            r1 = p.add_run()
            r1.text = lead
            r1.font.bold = True
            r1.font.size = Pt(size)
            r1.font.color.rgb = color
            r1.font.name = "Calibri"
            r2 = p.add_run()
            r2.text = rest
            r2.font.size = Pt(size)
            r2.font.color.rgb = color
            r2.font.name = "Calibri"
        else:
            r1 = p.add_run()
            r1.text = item
            r1.font.size = Pt(size)
            r1.font.color.rgb = color
            r1.font.name = "Calibri"
    return box


def add_chrome(slide, prs, kicker, title, footer, page):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0,
                                 prs.slide_width, Inches(0.12))
    bar.fill.solid()
    bar.fill.fore_color.rgb = BLUE
    bar.line.fill.background()

    add_text(slide, 0.55, 0.34, 9, 0.3, kicker, size=11, bold=True, color=BLUE)
    add_text(slide, 0.55, 0.66, 12.2, 0.75, title, size=29, bold=True)
    add_text(slide, 0.55, 7.06, 10.8, 0.3, footer, size=9.5, color=GRAY)
    add_text(slide, 11.9, 7.06, 0.95, 0.3, page, size=9.5, color=GRAY,
             align=PP_ALIGN.RIGHT)


def style_cell(cell, text, size=12, bold=False, color=DARK,
               fill=None, align=PP_ALIGN.LEFT):
    cell.text = ""
    tf = cell.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = "Calibri"
    if fill is not None:
        cell.fill.solid()
        cell.fill.fore_color.rgb = fill
    else:
        cell.fill.background()
    cell.margin_left = Inches(0.09)
    cell.margin_right = Inches(0.06)
    cell.margin_top = Inches(0.03)
    cell.margin_bottom = Inches(0.03)


def build():
    img = make_loglog_figure()
    img_hist = make_hist_figure()
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    footer1 = ("MNIST  ·  MLP 784-256-128  ·  fixed subset |S| = 5,000  "
               "·  200 random batches per B  ·  seed 0")

    # ---------------- Slide 1 ----------------
    s1 = prs.slides.add_slide(blank)
    add_chrome(s1, prs, "EXPERIMENT 1  ·  GRADIENT NOISE",
               "Mini-batch gradient noise scales as 1/B", footer1, "1 / 2")
    s1.shapes.add_picture(img, Inches(0.5), Inches(1.62), width=Inches(7.75))

    add_bullets(s1, 8.5, 1.55, 4.35, 5.0, [
        ("Full gradient g ", "— computed from all 5,000 samples in a fixed subset"),
        ("Batch gradient g_B ", "— computed from B random samples"),
        ("Noise² ", "= E‖g_B − g‖²: how far one estimate strays"),
        ("Theory: ", "noise² = C / B, a straight line with slope −1"),
        ("Measured: ", "−0.996 (init), −0.981 (1 epoch); R² > 0.997"),
        ("Rule of thumb: ", "4× the batch → half the standard deviation"),
    ], size=13.5, space=9)

    s1.notes_slide.notes_text_frame.text = (
        "We fix a subset of 5,000 MNIST images and a fixed parameter state. "
        "The full gradient over this subset is our reference. For each batch size "
        "we draw 200 random batches and measure the average squared distance "
        "between the batch gradient and the full gradient. On a log-log plot the "
        "result is a straight line: slope -0.996 at initialization and -0.981 after "
        "one epoch, with R-squared above 0.997. In words: making the batch four "
        "times larger halves the standard deviation of the gradient estimate."
    )

    # ---------------- Slide 2 ----------------
    s2 = prs.slides.add_slide(blank)
    add_chrome(s2, prs, "EXPERIMENT 1  ·  WHAT IT MEANS",
               "Noise is unbiased, predictable — and state-dependent",
               footer1, "2 / 2")

    rows, cols = 5, 3
    tbl = s2.shapes.add_table(rows, cols, Inches(0.55), Inches(1.5),
                              Inches(6.75), Inches(2.45)).table
    tbl.columns[0].width = Inches(3.55)
    tbl.columns[1].width = Inches(1.6)
    tbl.columns[2].width = Inches(1.6)
    tbl.rows[0].height = Inches(0.45)
    for r in range(1, 5):
        tbl.rows[r].height = Inches(0.5)

    header = ["Metric", "Init", "After 1 epoch"]
    for c, text in enumerate(header):
        style_cell(tbl.cell(0, c), text, size=13, bold=True, color=WHITE,
                   fill=BLUE, align=PP_ALIGN.CENTER if c else PP_ALIGN.LEFT)

    body = [
        ("C  (per-sample gradient variance)", "28.5", "68.4"),
        ("Signal  ‖g‖²", "0.269", "0.0879"),
        ("Critical batch size  B* = C / ‖g‖²", "106", "778"),
        ("Mean-gradient rel. error  (B = 64)", "8.3%", "24.9%"),
    ]
    for r, (name, v1, v2) in enumerate(body, start=1):
        fill = WHITE if r % 2 else LIGHT
        style_cell(tbl.cell(r, 0), name, size=12, bold=False,
                   fill=fill, align=PP_ALIGN.LEFT)
        style_cell(tbl.cell(r, 1), v1, size=12, fill=fill,
                   align=PP_ALIGN.CENTER)
        style_cell(tbl.cell(r, 2), v2, size=12, fill=fill,
                   align=PP_ALIGN.CENTER)

    add_text(s2, 0.55, 4.05, 6.75, 0.32,
             "At B = B*: expected noise² equals the signal ‖g‖².",
             size=11.5, color=GRAY)

    s2.shapes.add_picture(img_hist, Inches(0.55), Inches(4.45),
                          width=Inches(4.45))

    add_bullets(s2, 7.75, 1.55, 5.1, 5.2, [
        ("Unbiased. ", "Averaging 200 batch gradients stays close to g; the 8.3% "
                       "residual matches noise / (‖g‖√200)."),
        ("Bell-shaped. ", "One coordinate’s batch gradients cluster around g "
                          "— the central limit theorem."),
        ("B* is the break-even point. ", "Below it, more samples cut noise; above "
                                         "it, extra samples are nearly wasted."),
        ("B* drifts upward. ", "After one epoch the signal falls ×0.33 while "
                               "disagreement rises ×2.4, so B* grows ×7."),
        ("Takeaway. ", "Gradient noise is measurable and predictable — batch "
                       "size can be set by principle, not trial and error."),
    ], size=14, space=10)

    s2.notes_slide.notes_text_frame.text = (
        "Two consequences. First, the estimator is unbiased: the average of many "
        "batch gradients lands close to the full gradient, and its residual error "
        "matches the theoretical noise divided by norm of g times root 200. Second, "
        "the break-even batch size, B star, is C over the squared norm of the "
        "gradient. It is about 106 at initialization and rises to about 778 after "
        "one epoch, because the signal weakens while per-sample disagreement grows. "
        "Batch size should scale with this signal-to-noise ratio, not be tuned blindly."
    )

    out = os.path.join(ROOT, "slides_exp1.pptx")
    prs.save(out)
    print(f"saved: {out}")
    print(f"saved: {img}")


if __name__ == "__main__":
    build()
