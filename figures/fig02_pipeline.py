"""Figure 2 -- MovementGNN methodological pipeline.

Four-stage vertical flow:
(1) Acquisition, (2) Graph representation, (3) Model, (4) Product.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from utils_nato import FIG_WIDTH_IN, PALETTE, save_fig, set_mpl_style


def _box(ax, x, y, w, h, title, subtitle=None, *,
         bg=PALETTE["panel"], edge=PALETTE["edge"],
         title_color=None, title_size=7.8, sub_size=6.3, lw=0.8):
    title_color = title_color or PALETTE["primary"]
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.06",
        linewidth=lw, edgecolor=edge, facecolor=bg)
    ax.add_patch(p)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.70, title,
                ha="center", va="center", fontsize=title_size,
                fontweight="bold", color=title_color)
        ax.text(x + w / 2, y + h * 0.30, subtitle,
                ha="center", va="center", fontsize=sub_size,
                color=PALETTE["text"])
    else:
        ax.text(x + w / 2, y + h / 2, title,
                ha="center", va="center", fontsize=title_size,
                fontweight="bold", color=title_color)


def _arrow(ax, x1, y1, x2, y2, color=None, lw=0.8):
    color = color or PALETTE["muted"]
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=8,
        linewidth=lw, color=color, shrinkA=2, shrinkB=2)
    ax.add_patch(arr)


def build():
    set_mpl_style()
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN * 1.06, FIG_WIDTH_IN * 0.62))
    ax.set_xlim(-1.05, 16)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    bands = [
        (8.30, 1.55, "#EEF2F7", "1. Acquisition"),
        (5.80, 1.55, "#F4EFE4", "2. Graph"),
        (3.30, 1.55, "#EFE7E4", "3. Model"),
        (0.55, 1.40, "#E8EEE8", "4. Product"),
    ]
    for y, h, cor, rot in bands:
        ax.add_patch(FancyBboxPatch(
            (0.0, y), 16, h,
            boxstyle="round,pad=0.0,rounding_size=0.02",
            linewidth=0, facecolor=cor, alpha=0.65))
        # rotulo de estagio VERTICAL, fora da faixa (evita sobreposicao com as caixas)
        ax.text(-0.45, y + h / 2,
                rot, ha="center", va="center", fontsize=6.8, rotation=90,
                color=PALETTE["muted"], style="italic", fontweight="bold")

    sources = [
        ("Copernicus DEM", "30 m \u00b7 elev. + derivatives"),
        ("Sentinel-2", "10 m \u00b7 NDVI \u00b7 NDWI"),
        ("Sparse LiDAR", "support points"),
        ("Water mask", "water bodies"),
    ]
    wb, hb = 3.5, 1.20
    gap = 0.30
    x0 = (16 - 4 * wb - 3 * gap) / 2
    y1 = 8.45
    centers_1 = []
    for i, (t, s) in enumerate(sources):
        xi = x0 + i * (wb + gap)
        _box(ax, xi, y1, wb, hb, t, s, title_color=PALETTE["primary"])
        centers_1.append(xi + wb / 2)

    boxes_2 = [
        (1.20, "Rasterization", "30 m grid \u00b7 3598\u00d73354"),
        (5.20, "12.07 M nodes", "17 raw features / node"),
        (8.90, "8-neighbour graph", "dist. + \u0394elev. \u00b7 self-loops"),
        (12.60, "Spatial split", "5 km blocks \u00b7 no buffer"),
    ]
    wg, hg = 3.10, 1.20
    y2 = 5.95
    centers_2 = []
    for x, t, s in boxes_2:
        wi = wg if x != 12.60 else 2.95
        _box(ax, x, y2, wi, hg, t, s, title_color=PALETTE["accent"],
             title_size=7.1, sub_size=6.0)
        centers_2.append(x + wi / 2)

    _box(ax, 1.20, 3.55, 5.40, 1.10,
         "Encoder \u00b7 3 \u00d7 GATv2",
         "4 heads \u00b7 hidden 64 \u00b7 res + LayerNorm",
         title_color=PALETTE["highlight"], lw=1.0)
    _box(ax, 7.00, 3.55, 4.70, 1.10,
         "Rule-informed Loss",
         "CE + slope + NDVI + water",
         title_color=PALETTE["highlight"], lw=1.0)
    _box(ax, 12.10, 3.55, 3.30, 1.10,
         "4 MLP heads",
         "Dis \u00b7 Mot \u00b7 Mec \u00b7 Arm",
         title_color=PALETTE["highlight"], lw=1.0)
    ax.text(1.20, 3.34,
            "output: Go \u00b7 Slow Go \u00b7 No Go per profile",
            ha="left", va="top", fontsize=6.6,
            color=PALETTE["muted"], style="italic")
    ax.text(1.20, 3.06,
            "52,364 params  \u00b7  receptive horizon 90 \u2192 180 \u2192 270 m",
            ha="left", va="top", fontsize=6.6,
            color=PALETTE["muted"], style="italic")

    _box(ax, 3.80, 0.75, 8.40, 1.05,
         "Multi-band GeoTIFF \u00b7 EPSG:4326",
         "4 bands (one per profile) \u00b7 interoperable with ArcGIS Pro / ENVI",
         title_color=PALETTE["primary"], lw=1.0)

    target_rast_x = centers_2[0]
    for cx in centers_1:
        _arrow(ax, cx, y1 - 0.02, target_rast_x, y2 + hg + 0.02,
               color=PALETTE["muted"], lw=0.6)

    for i in range(3):
        x_a = centers_2[i] + (wg if boxes_2[i][0] != 12.60 else 2.95) / 2 + 0.02
        x_b = boxes_2[i + 1][0] - 0.02
        ya = y2 + hg / 2
        _arrow(ax, x_a - 0.05, ya, x_b + 0.05, ya,
               color=PALETTE["accent"], lw=0.9)

    _arrow(ax, centers_2[2], y2 - 0.02, 3.90, 4.65 + 0.02,
           color=PALETTE["highlight"], lw=0.9)
    _arrow(ax, 6.62, 4.10, 6.98, 4.10,
           color=PALETTE["highlight"], lw=0.9)
    _arrow(ax, 11.72, 4.10, 12.08, 4.10,
           color=PALETTE["highlight"], lw=0.9)
    _arrow(ax, 12.10 + 3.30 / 2, 3.53, 10.20, 1.84,
           color=PALETTE["primary"], lw=1.0)

    save_fig(fig, "fig02_pipeline")


if __name__ == "__main__":
    build()
