"""Build the README cover and comparison grid from the figures in docs/images/.

    python docs/make_figures.py

Reuses the panels demo/predict_image.py already drew, so no weights are needed.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

DOCS = Path(__file__).resolve().parent
IMAGES = DOCS / "images"

BG, CARD, MUTED, ACCENT = "#121419", "#1c2029", "#9aa0ab", "#2fb3a3"
INK, SUB = "#1e2a44", "#5a6275"
plt.rcParams.update({"font.family": "DejaVu Sans"})

# (ground truth, pretrained, fine-tuned), from results/qnrf_per_image.csv
COUNTS = {
    "0271": (916, 1169, 920),
    "0104": (1387, 1841, 1401),
    "0308": (3568, 4058, 3521),
    "0303": (2154, 2488, 2115),
}


def panels(image_id: str) -> list[Image.Image]:
    """Input, pretrained and fine-tuned panels of one predict_image.py figure."""
    im = Image.open(IMAGES / f"qnrf_img_{image_id}.jpg").convert("RGB")
    w, h = im.size
    gray = im.convert("L").load()

    def white_col(x: int) -> bool:
        return sum(gray[x, y] > 245 for y in range(0, h, 4)) > 0.9 * (h // 4)

    def white_row(y: int) -> bool:
        return sum(gray[x, y] > 245 for x in range(0, w, 4)) > 0.9 * (w // 4)

    cols, inside = [], False
    for x in range(w):
        if not white_col(x) and not inside:
            start, inside = x, True
        elif white_col(x) and inside:
            cols.append((start, x))
            inside = False
    rows, inside = [], False
    for y in range(h):
        if not white_row(y) and not inside:
            start, inside = y, True
        elif white_row(y) and inside:
            rows.append((start, y))
            inside = False
    top, bottom = max(rows, key=lambda r: r[1] - r[0])  # skip the title band
    return [im.crop((x0, top, x1, bottom)) for x0, x1 in cols[:3]]


def signed(n: int) -> str:
    return f"+{n:,}" if n > 0 else f"−{abs(n):,}"


# ---------------------------------------------------------------- cover
gt, pre, ft = COUNTS["0271"]
cover = panels("0271")
aspect = cover[0].width / cover[0].height

fig_w, fig_h = 16, 7.1
fig = plt.figure(figsize=(fig_w, fig_h), dpi=100, facecolor=BG)
fig.text(0.03, 0.915, "Mosque Crowd Counting with CLIP-EBC + LoRA", color="white",
         fontsize=27, fontweight="bold", va="center")
fig.text(0.03, 0.858, "Counts dense prayer crowds from a density map, not identities  ·  "
         "ViT-B/16 fine-tuned on 10,482 images, 0.43% of its weights trained",
         color=MUTED, fontsize=13.5, va="center")

left, right, gap = 0.03, 0.97, 0.0125
w = (right - left - 2 * gap) / 3
img_h = w * fig_w / aspect / fig_h
img_y = 0.80 - img_h
labels = [
    (f"ground truth {gt:,}", "INPUT"),
    (f"{pre:,}  ({signed(pre - gt)})", "PRETRAINED CLIP-EBC"),
    (f"{ft:,}  ({signed(ft - gt)})", "FINE-TUNED (THIS REPO)"),
]
for i, (panel, (big, small)) in enumerate(zip(cover, labels)):
    x = left + i * (w + gap)
    ax = fig.add_axes([x, img_y, w, img_h])
    ax.imshow(panel, aspect="auto")
    ax.set_axis_off()
    fig.add_artist(Rectangle((x, img_y - 0.085), w, 0.08, transform=fig.transFigure, color=CARD))
    fig.text(x + 0.009, img_y - 0.027, big, color=ACCENT if i == 2 else "#e8a33d" if i == 1 else "white",
             fontsize=15, fontweight="bold", va="center")
    fig.text(x + 0.009, img_y - 0.063, small, color="#c9ced6", fontsize=12, fontweight="bold", va="center")

cards = [
    ("−23%", "MAE on all 334 unseen UCF-QNRF\ntest images (130.00 → 100.06)"),
    ("−76%", "RMSE (694.16 → 164.79), and the\nmean error moves from +80 to −6"),
    ("0.43%", "of the parameters trained:\nLoRA rank 16 + 32 prompt tokens"),
    ("16.4M", "grid cells measured to re-derive the\ncount bins that were saturating"),
]
cw = (right - left - 3 * gap) / 4
card_y, card_h = 0.095, 0.15
for i, (big, small) in enumerate(cards):
    x = left + i * (cw + gap)
    fig.add_artist(Rectangle((x, card_y), cw, card_h, transform=fig.transFigure, color=CARD))
    fig.add_artist(Rectangle((x, card_y), 0.0022, card_h, transform=fig.transFigure, color=ACCENT))
    fig.text(x + 0.012, card_y + 0.105, big, color="white", fontsize=22, fontweight="bold", va="center")
    fig.text(x + 0.012, card_y + 0.042, small, color=MUTED, fontsize=11.5, va="center", linespacing=1.3)

fig.text(0.03, 0.042, "Image: UCF-QNRF test set, img_0271 (public dataset, ECCV 2018). "
         "Density maps from demo/predict_image.py.", color="#6b7280", fontsize=10.5, va="center")
fig.savefig(DOCS / "header.png", facecolor=BG)
plt.close(fig)
print("wrote docs/header.png")


# ---------------------------------------------------------------- comparison grid
ids = ["0104", "0308", "0303"]
rows = [panels(i) for i in ids]
fig_w = 15
col_w = 0.94 / 3
heights = [col_w * fig_w * r[0].height / r[0].width for r in rows]
head, caption, pad = 0.45, 0.42, 0.18
fig_h = head + sum(heights) + len(rows) * (caption + pad)
fig = plt.figure(figsize=(fig_w, fig_h), dpi=110, facecolor="white")

y = fig_h - head
for c, title in enumerate(["Input", "Pretrained CLIP-EBC", "Fine-tuned (this repo)"]):
    fig.text(0.03 + c * col_w + 0.004, (y + head * 0.45) / fig_h, title, color=INK,
             fontsize=14, fontweight="bold", va="center")
for image_id, row, h in zip(ids, rows, heights):
    y -= h
    for c, panel in enumerate(row):
        ax = fig.add_axes([0.03 + c * col_w + 0.003, y / fig_h, col_w - 0.006, h / fig_h])
        ax.imshow(panel, aspect="auto")
        ax.set_axis_off()
    gt, pre, ft = COUNTS[image_id]
    cy = (y - caption * 0.5) / fig_h
    fig.text(0.033, cy, f"img_{image_id}", color=INK, fontsize=12, fontweight="bold", va="center")
    fig.text(0.033 + col_w, cy, f"pretrained {pre:,} ({signed(pre - gt)})", color="#b4441b",
             fontsize=12, va="center")
    fig.text(0.033 + 2 * col_w, cy, f"fine-tuned {ft:,} ({signed(ft - gt)})", color="#147a6e",
             fontsize=12, fontweight="bold", va="center")
    fig.text(0.033 + 0.07, cy, f"ground truth {gt:,}", color=SUB, fontsize=12, va="center")
    y -= caption + pad
fig.savefig(IMAGES / "comparison.png", facecolor="white")
plt.close(fig)
print("wrote docs/images/comparison.png")
