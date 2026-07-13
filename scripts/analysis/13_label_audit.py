"""
13_label_audit.py - Label Quality Audit
========================================
Control hypothesis: Labels are nearly a direct function of features,
which explains the performance ceiling of tabular models.

Analyses:
  - Feature-to-label correlation (R^2) per fraction and feature
  - Class distribution per fraction (baseline for comparison)
  - Effect of class imbalance on performance

This script can run with or without the graph:
  - With graph: computes R^2 feature->label directly
  - Without graph: uses metrics.json + class_counts from baselines

Inputs:
  - pacaraima_q1q4_fixed.pt            (optional, for full D11)
  - metrics.json from RF, MLP, GNN, Rule (for D12, D17)

Outputs:
  - 13_label_audit_results.json
  - 13_d11_correlacao_features.png
  - 13_d12_distribuicao_classes.png
  - 13_d17_desbalanceamento.png
"""

import gc
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from src.paths import get_analysis_paths, is_colab

IS_COLAB = is_colab()
_analysis = get_analysis_paths("13_label_audit")
_graph = _analysis["graph"]
GRAPH_PATH = _graph if _graph.exists() else None
OUT_DIR    = _analysis["out_dir"]
DATA_DIR   = _analysis["data_dir"]
METRICS_SOURCES = _analysis["metrics_paths"]

FRACTIONS = ["a_pe", "motorizada", "mecanizada", "blindada"]

FEATURE_NAMES = [
    "elevation", "slope", "aspect_cos", "aspect_sin", "curvature",
    "tpi", "tri", "roughness", "B02", "B03", "B04", "B08",
    "NDVI", "NDWI", "water_mask", "lidar_avail", "lidar_elev",
]

# DAMEPLAN thresholds (expected)
NDVI_THRESH = {"a_pe": 0.7, "motorizada": 0.5, "mecanizada": 0.6, "blindada": 0.5}
SLOPE_THRESH = 0.35


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_metrics_json(paths):
    for p in paths:
        if p.exists():
            with open(str(p)) as f:
                return json.load(f)
    return None


# --- Main ---
def main():
    start = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    # --- Load metrics JSONs ---
    log("Loading metrics JSONs ...")
    metrics_all = {}
    for mname, paths in METRICS_SOURCES.items():
        m = load_metrics_json(paths)
        if m:
            metrics_all[mname] = m
            log(f"  {mname}: OK")
        else:
            log(f"  {mname}: NOT FOUND")

    # --- D11 - Feature->Label Correlation (requires graph) ---
    d11 = {}
    feat_source = None
    if IS_COLAB and GRAPH_PATH and GRAPH_PATH.exists():
        feat_source = "graph"
    elif not IS_COLAB and DATA_DIR and (DATA_DIR / "features_full.npz").exists():
        feat_source = "npz"

    if feat_source:
        log(f"D11 - Feature->label correlation (source={feat_source}) ...")
        import torch
        if feat_source == "graph":
            data  = torch.load(str(GRAPH_PATH), map_location="cpu", weights_only=False)
            dem   = data["dem"]
            feat  = dem.x.float().numpy()
            N     = feat.shape[0]
            labels_graph = {f: getattr(dem, f"y_{f}").numpy().astype(np.float32)
                            for f in FRACTIONS if hasattr(dem, f"y_{f}")}
        else:
            arr = np.load(str(DATA_DIR / "features_full.npz"))
            feat = arr["features"].astype(np.float32)
            N    = feat.shape[0]
            lbl  = np.load(str(DATA_DIR / "labels.npz"))
            labels_graph = {f: lbl[f].astype(np.float32) for f in FRACTIONS if f in lbl}

        log(f"  N={N:,}, features={feat.shape[1]}")

        for frac in FRACTIONS:
            if frac not in labels_graph:
                continue
            y = labels_graph[frac]

            row = {}
            for i, fname in enumerate(FEATURE_NAMES):
                col = feat[:, i]
                corr = float(np.corrcoef(col, y)[0, 1])
                row[fname] = {
                    "pearson_r":  round(corr, 4),
                    "r_squared":  round(corr ** 2, 4),
                }

            top3 = sorted(row.items(), key=lambda x: abs(x[1]["pearson_r"]), reverse=True)[:3]
            row["_top3"] = [{"feature": k, "r": v["pearson_r"], "r2": v["r_squared"]} for k, v in top3]
            row["_max_r2"] = max(v["r_squared"] for v in row.values() if isinstance(v, dict) and "r_squared" in v)

            d11[frac] = row

        results["D11_correlacao"] = d11

        fig, ax = plt.subplots(figsize=(14, 8))
        feat_labels = FEATURE_NAMES
        frac_labels = [f for f in FRACTIONS if f in d11]
        mat = np.array([
            [d11[frac].get(fname, {}).get("r_squared", 0) for fname in feat_labels]
            for frac in frac_labels
        ])

        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(len(feat_labels)))
        ax.set_xticklabels(feat_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(frac_labels)))
        ax.set_yticklabels([f.upper() for f in frac_labels], fontsize=9)
        ax.set_title("D11 - R^2 Feature->Label per Fraction\n(high R^2 = label is nearly a function of the feature)", fontsize=11)
        plt.colorbar(im, ax=ax, label="R^2 (Pearson^2)")
        for i in range(len(frac_labels)):
            for j in range(len(feat_labels)):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if mat[i, j] > 0.5 else "black")
        plt.tight_layout()
        plt.savefig(str(OUT_DIR / "13_d11_correlacao_features.png"), dpi=600, bbox_inches="tight")
        plt.close()
        log("  D11 done.")
        if feat_source == "graph":
            del data
        del feat
        gc.collect()
    else:
        log("D11 - Graph not available, using RF feature_importance as proxy ...")
        if "RF" in metrics_all:
            rf_data = metrics_all["RF"]
            fracs_key = "fractions" if "fractions" in rf_data else None
            if fracs_key:
                for frac in FRACTIONS:
                    if frac in rf_data[fracs_key] and "feature_importance" in rf_data[fracs_key][frac]:
                        fi = rf_data[fracs_key][frac]["feature_importance"]
                        d11[frac] = {k: {"rf_importance": v} for k, v in fi.items()}
        results["D11_correlacao"] = d11
        results["D11_nota"] = "Computed via RF feature_importance (proxy - graph not available)"

    # --- D12 - Class Distribution per Fraction ---
    log("D12 - Class distribution per fraction ...")
    d12 = {}

    rf_data = metrics_all.get("RF", {})
    fracs_rf = rf_data.get("fractions", rf_data)

    if fracs_rf:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("D12 - Class Distribution per Fraction (Pacaraima)", fontsize=13)

        for idx_f, frac in enumerate(FRACTIONS):
            if frac not in fracs_rf:
                continue
            frac_data = fracs_rf[frac]
            counts = frac_data.get("class_counts", {}).get("true", {})

            n_go    = counts.get("1", 0)
            n_slow  = counts.get("2", 0)
            n_nogo  = counts.get("3", 0)
            n_total = n_go + n_slow + n_nogo

            d12[frac] = {
                "n_Go":       n_go,
                "n_SlowGo":   n_slow,
                "n_NoGo":     n_nogo,
                "n_total":    n_total,
                "pct_Go":     round(n_go / n_total * 100, 2) if n_total > 0 else 0,
                "pct_SlowGo": round(n_slow / n_total * 100, 2) if n_total > 0 else 0,
                "pct_NoGo":   round(n_nogo / n_total * 100, 2) if n_total > 0 else 0,
            }

            ax = axes[idx_f // 2][idx_f % 2]
            labels_cls = ["Go\n(Unrestricted)", "SlowGo\n(Restricted)", "NoGo\n(Sev. Restricted)"]
            values = [n_go, n_slow, n_nogo]
            pcts   = [n_go/n_total*100, n_slow/n_total*100, n_nogo/n_total*100]
            colors = ["#2E7D32", "#F57F17", "#C62828"]
            bars = ax.bar(labels_cls, values, color=colors, alpha=0.85, edgecolor="white")
            for bar, pct in zip(bars, pcts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.5,
                        f"{pct:.1f}%", ha="center", va="center",
                        fontsize=10, fontweight="bold", color="white")
            ax.set_title(f"{frac.upper()} (n={n_total:,})", fontsize=10)
            ax.set_ylabel("Number of nodes")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1e6:.1f}M"))

        plt.tight_layout()
        plt.savefig(str(OUT_DIR / "13_d12_distribuicao_classes.png"), dpi=600, bbox_inches="tight")
        plt.close()

    results["D12_distribuicao"] = d12
    log("  D12 done.")

    # --- D17 - Effect of Class Imbalance on Performance ---
    log("D17 - Class imbalance effect on performance ...")
    d17 = {}

    model_f1_c2 = {}
    for mname, mdata in metrics_all.items():
        fracs_data = mdata.get("fractions", mdata)
        model_f1_c2[mname] = {}
        for frac in FRACTIONS:
            if frac in fracs_data:
                fdata = fracs_data[frac]
                if "f1_per_class" in fdata:
                    f1c2 = fdata["f1_per_class"].get("2", None)
                elif "confusion_matrix" in fdata:
                    cm = np.array(fdata["confusion_matrix"])
                    if cm.shape == (3, 3):
                        tp = cm[1, 1]
                        fp = cm[0, 1] + cm[2, 1]
                        fn = cm[1, 0] + cm[1, 2]
                        prec = tp / (tp + fp + 1e-8)
                        rec  = tp / (tp + fn + 1e-8)
                        f1c2 = 2 * prec * rec / (prec + rec + 1e-8)
                    else:
                        f1c2 = None
                else:
                    f1c2 = None
                model_f1_c2[mname][frac] = float(f1c2) if f1c2 is not None else None

    pct_slowgo = {}
    for frac in FRACTIONS:
        if frac in d12:
            pct_slowgo[frac] = d12[frac]["pct_SlowGo"]
        else:
            pct_slowgo[frac] = None

    d17["pct_slowgo_por_fracao"] = {f: pct_slowgo.get(f) for f in FRACTIONS}
    d17["f1_classe2_por_modelo"] = model_f1_c2

    for mname, f1s in model_f1_c2.items():
        valid_pairs = [(pct_slowgo[f], f1s[f]) for f in FRACTIONS
                       if pct_slowgo.get(f) is not None and f1s.get(f) is not None]
        if len(valid_pairs) >= 2:
            xs = [p[0] for p in valid_pairs]
            ys = [p[1] for p in valid_pairs]
            corr = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 1 else 0.0
            d17[f"correlacao_pctC2_vs_F1c2_{mname}"] = round(corr, 4)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("D17 - Imbalance and Performance: SlowGo Class", fontsize=13)

    ax1 = axes[0]
    x = np.arange(len(FRACTIONS))
    w = 0.2
    colors_m = {"GNN": "#0D47A1", "RF": "#2E7D32", "MLP": "#6A1B9A", "Rule": "#E65100"}
    for i, mname in enumerate(model_f1_c2.keys()):
        ys = [model_f1_c2[mname].get(f) or 0 for f in FRACTIONS]
        bars = ax1.bar(x + i * w, ys, w, label=mname,
                       color=colors_m.get(mname, "#455A64"), alpha=0.90)
        for bar, y in zip(bars, ys):
            if y > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, y + 0.002,
                         f"{y:.3f}", ha="center", va="bottom", fontsize=6)
    ax1.set_xticks(x + w * len(model_f1_c2) / 2)
    ax1.set_xticklabels([f.upper() for f in FRACTIONS])
    ax1.set_ylim(0.7, 1.02)
    ax1.set_ylabel("F1 Macro - SlowGo Class")
    ax1.set_title("F1 of Restricted Class by Model")
    ax1.legend(fontsize=8)

    ax2 = axes[1]
    colors_f = {"a_pe": "#C62828", "motorizada": "#0D47A1",
                "mecanizada": "#2E7D32", "blindada": "#F57F17"}
    for mname, f1s in model_f1_c2.items():
        for frac in FRACTIONS:
            pct = pct_slowgo.get(frac)
            f1  = f1s.get(frac)
            if pct is not None and f1 is not None:
                ax2.scatter(pct, f1, s=120, color=colors_f.get(frac, "gray"),
                            marker={"GNN": "o", "RF": "s", "MLP": "^", "Rule": "D"}.get(mname, "o"),
                            alpha=0.85, label=f"{mname}_{frac}", zorder=3)
                ax2.annotate(f"{mname[:3]}\n{frac[:3]}", (pct, f1),
                             textcoords="offset points", xytext=(5, 3), fontsize=6)

    ax2.set_xlabel("% SlowGo in fraction")
    ax2.set_ylabel("F1 SlowGo Class")
    ax2.set_title("Imbalance vs Performance on Restricted Class")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(OUT_DIR / "13_d17_desbalanceamento.png"), dpi=600, bbox_inches="tight")
    plt.close()

    results["D17_desbalanceamento"] = d17
    log("  D17 done.")

    # --- Save JSON ---
    results["meta"] = {
        "timestamp": datetime.now().isoformat(),
        "fracoes":   FRACTIONS,
        "modelos_carregados": list(metrics_all.keys()),
        "elapsed_s": round(time.perf_counter() - start, 1),
    }

    out_json = OUT_DIR / "13_label_audit_results.json"
    with open(str(out_json), "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.perf_counter() - start
    print("\n" + "=" * 60)
    print("SCRIPT 13 - COMPLETE")
    print(f"Total time: {elapsed:.0f}s")
    print(f"Results at: {OUT_DIR}")
    for p in sorted(OUT_DIR.iterdir()):
        print(f"  {p.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
