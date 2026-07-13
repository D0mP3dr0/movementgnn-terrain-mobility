"""Figure -- Correlation between Shannon entropy and topographic indicators.

Two-panel figure:
  Left:  Entropy vs normalised slope (with doctrinal threshold τ_s ≈ 0.35)
  Right: Entropy vs NDVI (with fraction-specific thresholds τ_v)

Uses hexbin density to handle 12M+ nodes; overlays doctrinal thresholds as
dashed lines and the ZT entropy cut-off as horizontal dashed line.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))  # utils_nato local (copia, item 3.9)

from utils_nato import (ANALISE_DIR, COLAB_RESULTS_V2, COLAB_RESULTS_LOCAL,
                         FIG_WIDTH_IN, PALETTE, FRACTIONS,
                         load_features, save_fig, set_mpl_style)

OUT_DIR = Path(__file__).resolve().parent
FRAC = "blindada"
INTGAP_PERCENTILE = 85

NDVI_THRESHOLDS = {
    "a_pe": 0.70,
    "motorizada": 0.50,
    "mecanizada": 0.60,
    "blindada": 0.50,
}
SLOPE_THRESHOLD = 0.35


def build():
    set_mpl_style()

    print("[fig_corr] loading probabilities...")
    probs_path = COLAB_RESULTS_LOCAL / "probs_full.npz"  # RUN FINAL blindada_fix (D1/D2, 12/07)
    probs_data = np.load(probs_path)
    probs_key = f"probs_{FRAC}"
    if probs_key not in probs_data:
        probs_key = list(probs_data.keys())[0]
    probs = probs_data[probs_key]

    eps = 1e-9
    entropy = -np.sum(probs * np.log(probs + eps), axis=1) / np.log(3)
    thr_ent = np.percentile(entropy, INTGAP_PERCENTILE)
    print(f"[fig_corr] entropy P{INTGAP_PERCENTILE} = {thr_ent:.3f}")

    print("[fig_corr] loading features...")
    feats = load_features(["NDVI", "slope"])
    ndvi = feats["NDVI"]
    slope = feats["slope"]

    n = len(entropy)
    rng = np.random.default_rng(42)
    sample_size = min(500_000, n)
    idx = rng.choice(n, size=sample_size, replace=False)
    ent_s = entropy[idx]
    slope_s = slope[idx]
    ndvi_s = ndvi[idx]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 0.55),
        gridspec_kw={"wspace": 0.30})

    hb1 = ax1.hexbin(slope_s, ent_s, gridsize=80, cmap="cividis",
                      mincnt=1, linewidths=0.1, extent=[0, 0.8, 0, 1])
    ax1.axvline(SLOPE_THRESHOLD, color=PALETTE["highlight"], linewidth=1.2,
                linestyle="--", label=f"$\\tau_s = {SLOPE_THRESHOLD}$")
    ax1.axhline(thr_ent, color=PALETTE["accent2"], linewidth=1.0,
                linestyle=":", label=f"$H_t$ = P{INTGAP_PERCENTILE} ({thr_ent:.2f})")
    ax1.set_xlabel("Normalised slope $\\tilde{s}$", fontsize=8)
    ax1.set_ylabel("Normalised Shannon entropy", fontsize=8)
    ax1.set_title("A) Entropy vs. slope", fontsize=9, fontweight="bold",
                  color=PALETTE["text"], loc="left", pad=6)
    ax1.set_xlim(0, 0.8)
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=7, loc="upper left", frameon=True, framealpha=0.9)
    ax1.spines[["top", "right"]].set_visible(False)

    hb2 = ax2.hexbin(ndvi_s, ent_s, gridsize=80, cmap="cividis",
                      mincnt=1, linewidths=0.1, extent=[-0.2, 1.0, 0, 1])
    for frac, tau in NDVI_THRESHOLDS.items():
        ls = "-" if frac == FRAC else ":"
        lw = 1.2 if frac == FRAC else 0.7
        alpha = 1.0 if frac == FRAC else 0.6
        ax2.axvline(tau, color=PALETTE["highlight"], linewidth=lw,
                    linestyle=ls, alpha=alpha)
    ax2.axhline(thr_ent, color=PALETTE["accent2"], linewidth=1.0,
                linestyle=":", label=f"$H_t$ = P{INTGAP_PERCENTILE}")
    ax2.set_xlabel("NDVI", fontsize=8)
    ax2.set_ylabel("Normalised Shannon entropy", fontsize=8)
    ax2.set_title("B) Entropy vs. NDVI", fontsize=9, fontweight="bold",
                  color=PALETTE["text"], loc="left", pad=6)
    ax2.set_xlim(-0.2, 1.0)
    ax2.set_ylim(0, 1)
    ax2.spines[["top", "right"]].set_visible(False)

    legend_lines = [
        Line2D([0], [0], color=PALETTE["highlight"], linewidth=1.2,
               linestyle="-", label=f"$\\tau_v^{{arm}}$ = {NDVI_THRESHOLDS[FRAC]}"),
        Line2D([0], [0], color=PALETTE["highlight"], linewidth=0.7,
               linestyle=":", alpha=0.6, label="$\\tau_v$ other fractions"),
        Line2D([0], [0], color=PALETTE["accent2"], linewidth=1.0,
               linestyle=":", label=f"$H_t$ (P{INTGAP_PERCENTILE})"),
    ]
    ax2.legend(handles=legend_lines, fontsize=7, loc="upper left",
               frameon=True, framealpha=0.9)

    cb = fig.colorbar(hb2, ax=[ax1, ax2], shrink=0.85, pad=0.02,
                      aspect=30, label="Node count")
    cb.ax.tick_params(labelsize=6.5)

    # Override save location
    import utils_nato
    original_out = utils_nato.OUT_DIR
    utils_nato.OUT_DIR = OUT_DIR
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_fig(fig, "fig_correlation")
    utils_nato.OUT_DIR = original_out


if __name__ == "__main__":
    build()
