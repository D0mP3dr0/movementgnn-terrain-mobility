"""Figure 3 -- MovementGNN architecture with GATv2 attention mechanism.

Left: input encoder -> 3 GATv2 layers -> shared encoder -> 4 MLP heads -> softmax
Center-top: attention detail (1 node, 8 neighbors)
Bottom-right: COO-Informed Loss (CIL) composition
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch

from utils_nato import FIG_WIDTH_IN, PALETTE, save_fig, set_mpl_style


def _box(ax, x, y, w, h, title, sub=None, *,
         bg=PALETTE["panel"], edge=PALETTE["edge"],
         title_color=None, ts=7.5, ss=6.2, lw=0.9):
    title_color = title_color or PALETTE["primary"]
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.012,rounding_size=0.05",
                       linewidth=lw, edgecolor=edge, facecolor=bg)
    ax.add_patch(p)
    if sub:
        ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center",
                fontsize=ts, fontweight="bold", color=title_color)
        ax.text(x + w / 2, y + h * 0.32, sub, ha="center", va="center",
                fontsize=ss, color=PALETTE["text"])
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=ts, fontweight="bold", color=title_color)


def _arrow(ax, x1, y1, x2, y2, color=None, lw=0.8, style="-|>"):
    color = color or PALETTE["muted"]
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle=style, mutation_scale=8,
                          linewidth=lw, color=color, shrinkA=2, shrinkB=2)
    ax.add_patch(arr)


def _attention_inset(ax, cx, cy, r=0.95):
    np.random.seed(7)
    pesos = np.array([0.22, 0.05, 0.18, 0.04, 0.20, 0.06, 0.16, 0.09])
    cores = ["#2E9B4A", "#B8351F", "#F0C432", "#B8351F", "#2E9B4A",
             "#B8351F", "#F0C432", "#2E9B4A"]
    for i in range(8):
        ang = 2 * np.pi * i / 8 + np.pi / 8
        nx = cx + r * np.cos(ang)
        ny = cy + r * np.sin(ang)
        ax.add_patch(Circle((nx, ny), 0.13, facecolor=cores[i],
                            edgecolor=PALETTE["edge"], linewidth=0.6))
        lw = 0.4 + pesos[i] * 12
        ax.plot([cx, nx], [cy, ny], "-",
                color=PALETTE["primary"], linewidth=lw, alpha=0.6,
                solid_capstyle="round", zorder=1)
        rl = 0.62
        rx = cx + rl * np.cos(ang)
        ry = cy + rl * np.sin(ang)
        ax.text(rx, ry, f"{pesos[i]:.2f}",
                ha="center", va="center", fontsize=5.0,
                color=PALETTE["primary"], fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.06",
                          facecolor="white", edgecolor="none", alpha=0.85),
                zorder=3)
    ax.add_patch(Circle((cx, cy), 0.18, facecolor="#1F3A5F",
                        edgecolor="white", linewidth=1.5, zorder=4))
    ax.text(cx, cy, "v", ha="center", va="center",
            fontsize=7.0, color="white", fontweight="bold", zorder=5)


def build():
    set_mpl_style()

    fig_w = FIG_WIDTH_IN * 1.25
    fig_h = FIG_WIDTH_IN * 0.72
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 18.0)
    ax.set_ylim(0, 11.2)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(9.0, 10.85, "MovementGNN \u2014 Multi-task GATv2 Architecture",
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=PALETTE["text"])

    # --- Input box ---
    _box(ax, 0.15, 7.6, 2.30, 1.6,
         "Input",
         "271-d\n15 feat + 256 emb.",
         title_color=PALETTE["primary"], ts=7.0, ss=5.5)

    # --- GATv2 layers ---
    cx0, gy = 2.80, 7.6
    cw, ch = 2.30, 1.6
    gap = 0.18
    horizons = ["~90 m", "~180 m", "~270 m"]
    for i in range(3):
        x = cx0 + i * (cw + gap)
        _box(ax, x, gy, cw, ch,
             f"GATv2 #{i+1}",
             f"4 heads \u00b7 h64\nhorizon {horizons[i]}",
             bg="#FBF2EF",
             title_color=PALETTE["highlight"], ts=6.8, ss=5.5, lw=1.0)
        if i < 2:
            _arrow(ax, x + cw - 0.04, gy + ch / 2,
                   x + cw + 0.18, gy + ch / 2,
                   color=PALETTE["highlight"], lw=0.9)

    ax.annotate("", xy=(cx0 + 3 * (cw + gap) - 0.30, gy + ch + 0.12),
                xytext=(cx0 + 0.10, gy + ch + 0.12),
                arrowprops=dict(arrowstyle="-|>", color=PALETTE["muted"],
                                lw=0.6, connectionstyle="arc3,rad=-0.4"))
    ax.text(cx0 + 1.5 * (cw + gap), gy + ch + 0.95,
            "residual + LayerNorm + dropout 0.1",
            ha="center", va="center", fontsize=5.8, style="italic",
            color=PALETTE["muted"])

    # --- Input → GATv2 #1 arrow ---
    _arrow(ax, 0.15 + 2.30, 7.6 + 1.6 / 2,
           cx0, 7.6 + 1.6 / 2, color=PALETTE["highlight"], lw=0.9)

    # --- GATv2 → Shared encoder arrow ---
    enc_y = 5.5
    enc_w = cx0 + 3 * (cw + gap) - gap - 0.15
    _box(ax, 0.15, enc_y, enc_w, 0.95,
         "Shared encoder",
         "output per node: 64-d  \u00b7  3 message-passing layers",
         bg="#EEF2F7", title_color=PALETTE["primary"], ts=7.2, ss=5.8)

    last_gat_cx = cx0 + 2 * (cw + gap) + cw / 2
    _arrow(ax, last_gat_cx, gy,
           last_gat_cx, enc_y + 0.95,
           color=PALETTE["highlight"], lw=0.9)

    # --- Classification heads ---
    head_y = 3.0
    head_w = 2.40
    head_x0 = 0.15
    heads = ["Dismounted", "Motorised", "Mechanised", "Armoured"]
    head_gap = 0.12
    for i, c in enumerate(heads):
        x = head_x0 + i * (head_w + head_gap)
        _box(ax, x, head_y, head_w, 1.30,
             c, "MLP \u2192 3 classes",
             bg="#F4F2EE", title_color=PALETTE["accent"], ts=6.5, ss=5.2)
        _arrow(ax, x + head_w / 2, enc_y, x + head_w / 2, head_y + 1.30,
               color=PALETTE["accent"], lw=0.7)

    # --- Softmax bars ---
    sm_y = 0.95
    for i, c in enumerate(heads):
        x = head_x0 + i * (head_w + head_gap)
        for j, (cor, val, lab) in enumerate(zip(
                ["#2E9B4A", "#F0C432", "#B8351F"],
                [0.65, 0.20, 0.15], ["Go", "SlwGo", "NoGo"])):
            bw = 0.62
            bx = x + 0.24 + j * (bw + 0.06)
            ax.add_patch(FancyBboxPatch(
                (bx, sm_y), bw, 0.45 * val * 2.2,
                boxstyle="round,pad=0.0,rounding_size=0.02",
                linewidth=0.4, edgecolor=PALETTE["edge"],
                facecolor=cor, alpha=0.85))
            ax.text(bx + bw / 2, sm_y - 0.08, lab,
                    ha="center", va="top", fontsize=5.0,
                    color=PALETTE["muted"])
        _arrow(ax, x + head_w / 2, head_y - 0.02,
               x + head_w / 2, sm_y + 1.05,
               color=PALETTE["muted"], lw=0.6)

    ax.text(head_x0 + (4 * head_w + 3 * head_gap) / 2, 0.45,
            "softmax 3 classes", ha="center", va="top", fontsize=5.5,
            style="italic", color=PALETTE["muted"])

    # --- GATv2 Attention inset ---
    ix, iy = 13.2, 7.0
    inset_w, inset_h = 4.50, 3.50
    ax.add_patch(FancyBboxPatch(
        (ix - inset_w / 2, iy - 1.55), inset_w, inset_h,
        boxstyle="round,pad=0.05,rounding_size=0.06",
        facecolor="#FAFBFD",
        edgecolor=PALETTE["edge"], linewidth=0.8))
    ax.text(ix, iy + inset_h - 1.55 - 0.47,
            "GATv2 Attention\n(1 node \u00b7 8 neighbors)",
            ha="center", va="center", fontsize=6.5, fontweight="bold",
            color=PALETTE["primary"], linespacing=1.3)
    _attention_inset(ax, ix, iy - 0.05, r=0.95)
    ax.text(ix, iy - 1.42,
            r"$\alpha_{ij}$ = learned weight",
            ha="center", va="center", fontsize=6.0, style="italic",
            color=PALETTE["muted"])

    # --- COO-Informed Loss box ---
    lx, ly, lw_box, lh = 10.3, 0.40, 7.40, 3.40
    ax.add_patch(FancyBboxPatch((lx, ly), lw_box, lh,
                                boxstyle="round,pad=0.05,rounding_size=0.06",
                                facecolor="#FBF2EF",
                                edgecolor=PALETTE["highlight"],
                                linewidth=1.1))
    ax.text(lx + lw_box / 2, ly + lh - 0.30,
            "Rule-informed Loss (domain-informed)",
            ha="center", va="center", fontsize=7.0, fontweight="bold",
            color=PALETTE["highlight"])

    eq = (r"$\mathcal{L} = \sum_{f} e^{-s_f}\!\left[\mathcal{L}_{focal}^{f} + "
          r"0.5\,(\mathcal{L}_{slope} + \mathcal{L}_{NDVI} + \mathcal{L}_{water})\right] + s_f$")
    ax.text(lx + lw_box / 2, ly + lh - 0.95, eq,
            ha="center", va="center", fontsize=6.8,
            color=PALETTE["text"])

    items = [
        (r"$\mathcal{L}_{focal}$", "focal CE ($\\gamma$=2, ls=0.05), class weights [1; 3\u20135; 1]"),
        (r"$\mathcal{L}_{slope}$", r"$p_{Go}\cdot\mathrm{relu}(slope_{norm}-0.35)$"),
        (r"$\mathcal{L}_{NDVI}$", r"$p_{Go}\cdot\mathrm{relu}(NDVI-\tau_f)$, $\tau_f\in[0.5,0.7]$"),
        (r"$\mathcal{L}_{water}$", r"$p_{Go}\cdot\mathbb{1}[NDWI>0.3 \vee water]$; $s_f$: homoscedastic"),
    ]
    y_i = ly + lh - 1.45
    for nome, desc in items:
        ax.text(lx + 0.30, y_i, "\u2022  " + nome,
                ha="left", va="center", fontsize=6.5, fontweight="bold",
                color=PALETTE["highlight"])
        ax.text(lx + 1.40, y_i, "\u2014 " + desc,
                ha="left", va="center", fontsize=6.0,
                color=PALETTE["text"])
        y_i -= 0.36

    # --- Legend ---
    leg_handles = [
        Patch(facecolor="#2E9B4A", edgecolor="black", label="Go (Unrestricted)"),
        Patch(facecolor="#F0C432", edgecolor="black", label="Slow Go (Restricted)"),
        Patch(facecolor="#B8351F", edgecolor="black", label="No Go (Sev. Restricted)"),
    ]
    ax.legend(handles=leg_handles, loc="lower center",
              bbox_to_anchor=(0.55, -0.04), ncol=3, frameon=False, fontsize=6.0)

    save_fig(fig, "fig03_architecture")


if __name__ == "__main__":
    build()
