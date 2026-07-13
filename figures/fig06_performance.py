"""Figure 6 -- MovementGNN performance by class/fraction and spatial coherence.

Two panels:
  A) F1 by class (Go/SlowGo/NoGo) per military fraction, computed from
     confusion matrices in metrics.json
  B) Isolated cells (%) for MovementGNN vs baselines (spatial coherence)
     from D09 in 10_incerteza_results.json
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt

from utils_nato import (CLASS_COLORS, CLASS_NAMES, ANALISE_DIR,
                         COLAB_RESULTS_V2, COLAB_RESULTS_LOCAL,
                         FIG_WIDTH_IN, FRACTIONS, FRACTION_TITLES,
                         PALETTE, save_fig, set_mpl_style)


def _f1_from_cm(cm):
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = np.where(tp + fp > 0, tp / (tp + fp + 1e-9), 0.0)
    rec = np.where(tp + fn > 0, tp / (tp + fn + 1e-9), 0.0)
    return np.where(prec + rec > 0, 2 * prec * rec / (prec + rec + 1e-9), 0.0)


def load_data():
    metrics_path = COLAB_RESULTS_LOCAL / "metrics.json"  # RUN FINAL blindada_fix (D1, 12/07)
    with open(metrics_path, encoding="utf-8") as f:
        m = json.load(f)
    inc_path = ANALISE_DIR / "10_incerteza" / "10_incerteza_results.json"
    with open(inc_path, encoding="utf-8") as f:
        inc = json.load(f)
    f1_by_class = {frac: _f1_from_cm(m[frac]["confusion_matrix"]) for frac in FRACTIONS}
    return f1_by_class, inc["D09_coerencia_espacial"]


def build():
    set_mpl_style()
    f1_by_class, iso_by_model = load_data()

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 0.62),
        gridspec_kw={"wspace": 0.35})
    fig.subplots_adjust(bottom=0.26)

    frac_labels = [FRACTION_TITLES[f][:3] for f in FRACTIONS]
    x = np.arange(len(FRACTIONS))
    bar_w = 0.26
    colors = [CLASS_COLORS["go"], CLASS_COLORS["slowgo"], CLASS_COLORS["nogo"]]

    for j, (cname, cor) in enumerate(zip(CLASS_NAMES, colors)):
        vals = np.array([f1_by_class[f][j] for f in FRACTIONS])
        axA.bar(x + (j - 1) * bar_w, vals, bar_w, color=cor,
                edgecolor=PALETTE["edge"], linewidth=0.5, label=cname)
        for xi, v in zip(x + (j - 1) * bar_w, vals):
            axA.text(xi, v + 0.015, f"{v:.2f}",
                     ha="left", va="bottom", fontsize=6.3,
                     color=PALETTE["text"], rotation=45, rotation_mode="anchor")

    axA.set_xticks(x)
    axA.set_xticklabels(frac_labels, fontsize=7.5)
    axA.set_ylim(0, 1.18)
    axA.set_ylabel("F1 per class", fontsize=8.5)
    axA.set_title("A \u00b7 Performance by fraction and class",
                  fontsize=9, loc="center", pad=10,
                  color=PALETTE["text"], fontweight="bold")
    axA.spines[["top", "right"]].set_visible(False)
    axA.yaxis.grid(True, linestyle="--", linewidth=0.4, color=PALETTE["grid"], alpha=0.8)
    axA.set_axisbelow(True)

    modelos = ["GNN", "Rule", "RF", "MLP"]
    modelos_labels = ["MovementGNN", "Rule-Based", "RF", "MLP"]
    cores_mod = [PALETTE["highlight"], "#6B7280", "#8E99A5", "#B7B9BC"]

    bar_w = 0.19
    for i, (modelo, lab, cor) in enumerate(zip(modelos, modelos_labels, cores_mod)):
        vals = np.array([iso_by_model[f][modelo]["pct_isolados"] for f in FRACTIONS])
        xs = x + (i - 1.5) * bar_w
        axB.bar(xs, vals, bar_w, color=cor, edgecolor=PALETTE["edge"],
                linewidth=0.5, label=lab)
        if modelo == "GNN":
            for xi, v in zip(xs, vals):
                axB.text(xi, v + 0.10, f"{v:.1f}",
                         ha="left", va="bottom", fontsize=6.2,
                         color=PALETTE["text"], fontweight="bold",
                         rotation=45, rotation_mode="anchor")

    for i, f in enumerate(FRACTIONS):
        gnn = iso_by_model[f]["GNN"]["pct_isolados"]
        rule = iso_by_model[f]["Rule"]["pct_isolados"]
        red = (1 - gnn / rule) * 100
        axB.text(x[i] - 1.5 * bar_w, rule + 0.25, f"\u2212{red:.0f}%",
                 ha="center", va="bottom", fontsize=6.6,
                 color=PALETTE["highlight"], fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                           edgecolor=PALETTE["highlight"], linewidth=0.5))

    axB.set_xticks(x)
    axB.set_xticklabels(frac_labels, fontsize=7.5)
    axB.set_ylabel("% isolated cells", fontsize=8.5)
    axB.set_ylim(0, 8.2)
    axB.set_title("B \u00b7 Spatial coherence (isolated cells)",
                  fontsize=9, loc="center", pad=10,
                  color=PALETTE["text"], fontweight="bold")
    axB.spines[["top", "right"]].set_visible(False)
    axB.yaxis.grid(True, linestyle="--", linewidth=0.4, color=PALETTE["grid"], alpha=0.8)
    axB.set_axisbelow(True)

    from matplotlib.patches import Patch
    handles_cls = [Patch(facecolor=c, edgecolor=PALETTE["edge"], linewidth=0.5, label=n)
                   for c, n in zip(colors, CLASS_NAMES)]
    handles_mod = [Patch(facecolor=c, edgecolor=PALETTE["edge"], linewidth=0.5, label=n)
                   for c, n in zip(cores_mod, modelos_labels)]

    fig.legend(handles=handles_cls, loc="lower center",
               bbox_to_anchor=(0.50, 0.10), ncol=3, fontsize=6.8,
               frameon=False, handletextpad=0.4, columnspacing=1.0,
               title="Panel A \u2014 Classes", title_fontsize=7.0)
    fig.legend(handles=handles_mod, loc="lower center",
               bbox_to_anchor=(0.50, 0.01), ncol=4, fontsize=6.8,
               frameon=False, handletextpad=0.4, columnspacing=0.8,
               title="Panel B \u2014 Models", title_fontsize=7.0)

    save_fig(fig, "fig06_performance")


if __name__ == "__main__":
    build()
