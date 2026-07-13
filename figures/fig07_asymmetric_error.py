"""Figure 7 -- Asymmetric error structure (dangerous vs conservative).

Diverging horizontal bars per fraction, based on D14_confusao_assimetrica
from 10_incerteza_results.json.

Dangerous error  = underestimates restriction (Go when should be No Go)
Conservative error = overestimates restriction (No Go when should be Go)
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt

from utils_nato import (ANALISE_DIR, FIG_WIDTH_IN, FRACTIONS,
                         FRACTION_TITLES, PALETTE, save_fig, set_mpl_style)


def load_data():
    with open(ANALISE_DIR / "10_incerteza" / "10_incerteza_results.json",
              encoding="utf-8") as f:
        inc = json.load(f)
    return inc["D14_confusao_assimetrica"]


def build():
    set_mpl_style()
    data = load_data()

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 0.50))
    y = np.arange(len(FRACTIONS))[::-1] * 1.2
    h = 0.6

    COLOR_DANGER = "#B5271F"
    COLOR_CONSERV = "#2E5984"

    for yi, frac in zip(y, FRACTIONS):
        d = data[frac]["GNN"]
        perig = d["erro_perigoso_pct"]
        conserv = d["erro_conservador_pct"]
        ratio = d["ratio_conserv_per_perigoso"]

        ax.barh(yi, -perig, h, color=COLOR_DANGER,
                edgecolor=PALETTE["edge"], linewidth=0.5, zorder=3)
        ax.barh(yi, conserv, h, color=COLOR_CONSERV,
                edgecolor=PALETTE["edge"], linewidth=0.5, zorder=3)

        ax.text(-perig - 0.05, yi, f"{perig:.2f}%",
                ha="right", va="center", fontsize=7.3, fontweight="bold",
                color=COLOR_DANGER, zorder=6)
        ax.text(conserv + 0.05, yi, f"{conserv:.2f}%",
                ha="left", va="center", fontsize=7.3, fontweight="bold",
                color=COLOR_CONSERV, zorder=6)

        cor_badge = "#2E5984" if ratio >= 1 else "#B5271F"
        ax.text(0, yi + h / 2 + 0.15,
                f"cons./danger ratio = {ratio:.2f}",
                ha="center", va="bottom", fontsize=6.6, style="italic",
                color=cor_badge, zorder=7,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                          edgecolor=cor_badge, linewidth=0.6))

    ax.axvline(0, color=PALETTE["edge"], linewidth=0.9, zorder=1)

    ax.set_yticks(y)
    ax.set_yticklabels([FRACTION_TITLES[f] for f in FRACTIONS], fontsize=8.2)

    max_abs = max(
        max(data[f]["GNN"]["erro_perigoso_pct"],
            data[f]["GNN"]["erro_conservador_pct"])
        for f in FRACTIONS)
    xlim = max_abs * 1.35
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(min(y) - 0.7, max(y) + 1.8)
    tks = [-2, -1, 0, 1, 2]
    ax.set_xticks(tks)
    ax.set_xticklabels([f"{abs(t):.0f}%" for t in tks], fontsize=7.4)
    ax.set_xlabel("Error (% of total nodes)", fontsize=8.5)

    ax.text(-xlim * 0.50, max(y) + 1.15,
            "\u2190 Dangerous error\n(underestimates restriction)",
            ha="center", va="bottom", fontsize=7.2, fontweight="bold",
            color=COLOR_DANGER)
    ax.text(xlim * 0.50, max(y) + 1.15,
            "Conservative error \u2192\n(overestimates restriction)",
            ha="center", va="bottom", fontsize=7.2, fontweight="bold",
            color=COLOR_CONSERV)

    ax.set_title(
        "Asymmetric error structure of MovementGNN by fraction",
        fontsize=9.5, loc="center", pad=28,
        color=PALETTE["text"], fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.4, color=PALETTE["grid"], alpha=0.8)
    ax.set_axisbelow(True)

    save_fig(fig, "fig07_asymmetric_error")


if __name__ == "__main__":
    build()
