INTERNAL_DATA_ROOT = __import__("os").environ.get("MOVEMENTGNN_DATA_ROOT", ".")
import gc
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import f1_score

try:
    import google.colab  # noqa: F401
    IS_COLAB = True
except ImportError:
    IS_COLAB = False
_HERE    = Path(__file__).resolve().parent

def _glob_first(pattern_root, glob_pattern, fallback=None):
    candidates = sorted(pattern_root.glob(glob_pattern), reverse=True)
    return candidates[0] if candidates else (Path(fallback) if fallback else None)

if IS_COLAB:
    DRIVE_BASE   = Path(f"{INTERNAL_DATA_ROOT}")
    GRAPH_PATH   = DRIVE_BASE / "graph/pacaraima_q1q4_fixed.pt"
    GNN_PREDS    = DRIVE_BASE / "gnn/results_v2/predictions_full.npz"
    GNN_PROBS    = DRIVE_BASE / "gnn/results_v2/probs_full.npz"
    RF_PREDS     = DRIVE_BASE / "baseline_classic/results/random_forest/predictions_full.npz"
    RULE_PREDS   = DRIVE_BASE / "baseline_classic/results/rule_based/predictions_full.npz"
    OUT_DIR      = DRIVE_BASE / "gnn/analise/09_transicao"
    DATA_DIR     = None
else:
    DATA_DIR   = _HERE / "analise_dados"
    GNN_PREDS  = _HERE / "results" / "results_v2_local_blindada_fix" / "predictions_full.npz"
    GNN_PROBS  = _HERE / "results" / "results_v2_local_blindada_fix" / "probs_full.npz"
    RF_PREDS   = _glob_first(_HERE / "results",
                             "random_forest-*/random_forest/predictions_full.npz")
    RULE_PREDS = _glob_first(_HERE / "results",
                             "rule_based-*/rule_based/predictions_full.npz")
    GRAPH_PATH = None
    OUT_DIR    = _HERE / "analise_dados/09_transicao"
    print(f"[MODO LOCAL — v2_local]")
    print(f"  DATA_DIR : {DATA_DIR}")
    print(f"  GNN_PREDS: {GNN_PREDS}")
    print(f"  RF_PREDS : {RF_PREDS}")

NDVI_THRESH = {"a_pe": 0.7, "motorizada": 0.5, "mecanizada": 0.6, "blindada": 0.5}
SLOPE_THRESH = 0.35
MARGIN_NEAR  = 0.05
MARGIN_CLOSE = 0.02

FRACTIONS = ["a_pe", "motorizada", "mecanizada", "blindada"]
IDX_SLOPE = 1
IDX_NDVI  = 12
IDX_NDWI  = 13
IDX_TPI   = 5
IDX_TRI   = 6


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def f1_zone(y_true, y_pred, mask):
    if mask.sum() == 0:
        return float("nan"), 0
    return f1_score(y_true[mask], y_pred[mask], average="macro",
                    zero_division=0), int(mask.sum())


def load_preds(path, fractions):
    data = np.load(str(path))
    out = {}
    for f in fractions:
        if f in data:
            out[f] = data[f].astype(np.int32)
        elif f"y_{f}" in data:
            out[f] = data[f"y_{f}"].astype(np.int32)
    return out


def _norm0(arr):
    if arr is None:
        return None
    a = arr.astype(np.int32)
    return a - 1 if int(a.min()) >= 1 else a


def ndvi_zones(ndvi, threshold, margin_near, margin_close):
    diff = np.abs(ndvi - threshold)
    mask_close  = diff <= margin_close
    mask_near   = (diff <= margin_near) & ~mask_close
    mask_far    = diff > margin_near
    return mask_far, mask_near, mask_close


def slope_zones(slope, threshold, margin):
    mask_flat    = slope < (threshold - margin)
    mask_steep   = slope > (threshold + margin)
    mask_trans   = ~mask_flat & ~mask_steep
    return mask_flat, mask_trans, mask_steep


def main():
    start = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    if IS_COLAB:
        log(f"[Colab] Loading graph: {GRAPH_PATH.name} ...")
        data = torch.load(str(GRAPH_PATH), map_location="cpu", weights_only=False)
        dem  = data["dem"]
        feat = dem.x.float().numpy()
        N    = feat.shape[0]

        labels_true = {}
        for f in FRACTIONS:
            attr = f"y_{f}"
            if hasattr(dem, attr):
                labels_true[f] = getattr(dem, attr).numpy().astype(np.int32)

        et_main    = ("dem", "adjacent_to", "dem")
        _ei        = data[et_main].edge_index
        src        = _ei[0].numpy()
        dst        = _ei[1].numpy()
        raster_meta = getattr(data, "raster_meta", None)
        dem_obj    = dem
    else:
        log(f"[Local] Loading features from {DATA_DIR} ...")
        assert DATA_DIR and DATA_DIR.exists(), \
            f"DATA_DIR not found: {DATA_DIR}\nRun the script 00_extrair_dados_locais.py on Colab first."

        f_slim  = np.load(str(DATA_DIR / "features_slim.npz"))
        ndvi_   = f_slim["NDVI"]
        slope_  = f_slim["slope"]
        N       = len(ndvi_)

        feat_slim_dict = {k: f_slim[k] for k in f_slim.files}
        feat = None

        labels_true = {}
        lbl = np.load(str(DATA_DIR / "labels.npz"))
        for f in FRACTIONS:
            if f in lbl:
                labels_true[f] = lbl[f].astype(np.int32)

        edges = np.load(str(DATA_DIR / "edges.npz"))
        src = edges["src"]
        dst = edges["dst"]

        raster_meta = None
        rm_path = DATA_DIR / "raster_meta.json"
        if rm_path.exists():
            with open(str(rm_path)) as fp:
                raster_meta = json.load(fp)

        dem_obj  = None
        data     = None

        slim_keys = list(f_slim.files)
        slim_map  = {"elevation": 0, "slope": 1, "tpi": 5, "tri": 6,
                     "NDVI": 12, "NDWI": 13, "lidar_avail": 15, "lidar_elev": 16}
        feat = np.zeros((N, 17), dtype=np.float32)
        for k, idx in slim_map.items():
            if k in f_slim:
                feat[:, idx] = f_slim[k]

        log(f"  N={N:,} nos (modo local)")

    ndvi_all  = feat[:, IDX_NDVI]
    slope_all = feat[:, IDX_SLOPE]
    ndwi_all  = feat[:, IDX_NDWI]
    tpi_all   = feat[:, IDX_TPI]
    tri_all   = feat[:, IDX_TRI]
    has_lidar = feat[:, 15] > 0.5

    log(f"  N nos: {N:,} | LiDAR: {has_lidar.sum():,} ({has_lidar.mean()*100:.1f}%)")

    log("Carregando predicoes ...")
    preds_gnn  = load_preds(GNN_PREDS,  FRACTIONS)
    preds_rf   = load_preds(RF_PREDS,   FRACTIONS)
    preds_rule = load_preds(RULE_PREDS, FRACTIONS)
    log(f"  GNN: {len(preds_gnn)} fracoes | RF: {len(preds_rf)} | Rule: {len(preds_rule)}")

    for f in FRACTIONS:
        if f in preds_gnn:
            sz = preds_gnn[f].shape[0]
            if sz != N:
                log(f"  WARN: GNN {f} has {sz:,} nodes (graph has {N:,})")

    log('Normalizing prediction and label indices to 0-based ...')
    for _f in FRACTIONS:
        if _f in labels_true:
            labels_true[_f] = _norm0(labels_true[_f])
        if _f in preds_gnn:
            preds_gnn[_f]  = _norm0(preds_gnn[_f])
        if _f in preds_rf:
            preds_rf[_f]   = _norm0(preds_rf[_f])
        if _f in preds_rule:
            preds_rule[_f] = _norm0(preds_rule[_f])
    if preds_gnn:
        _sf = next(iter(preds_gnn))
        log(f"  Verificacao: GNN {_sf} range=[{preds_gnn[_sf].min()},{preds_gnn[_sf].max()}] "
            f"| label range=[{labels_true[_sf].min()},{labels_true[_sf].max()}]")

    gc.collect()

    log('D01 — F1 per NDVI band ...')
    d01 = {}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('D01 — Macro F1 per NDVI Band (vs DAMEPLAN Threshold)', fontsize=13)

    for idx_f, frac in enumerate(FRACTIONS):
        if frac not in labels_true or frac not in preds_gnn:
            continue
        y_true = labels_true[frac]
        thr    = NDVI_THRESH[frac]
        m_far, m_near, m_close = ndvi_zones(ndvi_all, thr, MARGIN_NEAR, MARGIN_CLOSE)

        row = {}
        zone_labels = ["Longe (>0.05)", "Perto (±0.05)", "No threshold (±0.02)"]
        masks = [m_far, m_near, m_close]
        models = {
            "GNN": preds_gnn.get(frac),
            "RF":  preds_rf.get(frac),
            "Rule": preds_rule.get(frac),
        }

        zone_data = {m: {} for m in models}
        for zone_name, mask in zip(zone_labels, masks):
            row[zone_name] = {}
            for mname, pred in models.items():
                if pred is None:
                    continue
                f1, cnt = f1_zone(y_true, pred, mask)
                row[zone_name][mname] = {"f1": round(float(f1), 4), "n": cnt}
                zone_data[mname][zone_name] = float(f1) if not np.isnan(f1) else 0.0

        d01[frac] = {"threshold_ndvi": thr, "zones": row}

        ax = axes[idx_f // 2][idx_f % 2]
        x = np.arange(len(zone_labels))
        w = 0.25
        colors = {"GNN": "#0D47A1", "RF": "#2E7D32", "Rule": "#E65100"}
        for i, (mname, vals) in enumerate(zone_data.items()):
            ys = [vals.get(z, 0) for z in zone_labels]
            bars = ax.bar(x + i * w, ys, w, label=mname, color=colors[mname], alpha=0.85)
            for bar, y in zip(bars, ys):
                if y > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                            f"{y:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_title(f"{frac.upper()} (NDVI thr={thr})", fontsize=10)
        ax.set_xticks(x + w)
        ax.set_xticklabels(zone_labels, fontsize=8)
        ax.set_ylim(0.7, 1.02)
        ax.set_ylabel("F1 Macro")
        ax.legend(fontsize=8)
        ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

    plt.tight_layout()
    plt.savefig(str(OUT_DIR / "09_d01_f1_ndvi.png"), dpi=600, bbox_inches="tight")
    plt.close()
    results["D01_f1_ndvi"] = d01
    log(f"  D01 concluido.")

    log('D02 — F1 per slope band ...')
    d02 = {}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"D02 — Macro F1 per Slope Band (threshold={SLOPE_THRESH})", fontsize=13)

    for idx_f, frac in enumerate(FRACTIONS):
        if frac not in labels_true or frac not in preds_gnn:
            continue
        y_true = labels_true[frac]
        m_flat, m_trans, m_steep = slope_zones(slope_all, SLOPE_THRESH, MARGIN_NEAR)

        row = {}
        zone_labels = [f"Plano (<{SLOPE_THRESH-MARGIN_NEAR:.2f})",
                       f"Transicao (±{MARGIN_NEAR})",
                       f"Inclinado (>{SLOPE_THRESH+MARGIN_NEAR:.2f})"]
        masks = [m_flat, m_trans, m_steep]
        models = {"GNN": preds_gnn.get(frac), "RF": preds_rf.get(frac), "Rule": preds_rule.get(frac)}

        zone_data = {m: {} for m in models}
        for zone_name, mask in zip(zone_labels, masks):
            row[zone_name] = {}
            for mname, pred in models.items():
                if pred is None:
                    continue
                f1, cnt = f1_zone(y_true, pred, mask)
                row[zone_name][mname] = {"f1": round(float(f1), 4), "n": cnt}
                zone_data[mname][zone_name] = float(f1) if not np.isnan(f1) else 0.0

        d02[frac] = {"threshold_slope": SLOPE_THRESH, "zones": row}

        ax = axes[idx_f // 2][idx_f % 2]
        x = np.arange(len(zone_labels))
        w = 0.25
        colors = {"GNN": "#0D47A1", "RF": "#2E7D32", "Rule": "#E65100"}
        for i, (mname, vals) in enumerate(zone_data.items()):
            ys = [vals.get(z, 0) for z in zone_labels]
            bars = ax.bar(x + i * w, ys, w, label=mname, color=colors[mname], alpha=0.85)
            for bar, y in zip(bars, ys):
                if y > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                            f"{y:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_title(f"{frac.upper()}", fontsize=10)
        ax.set_xticks(x + w)
        ax.set_xticklabels(zone_labels, fontsize=8)
        ax.set_ylim(0.7, 1.02)
        ax.set_ylabel("F1 Macro")
        ax.legend(fontsize=8)
        ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(str(OUT_DIR / "09_d02_f1_slope.png"), dpi=600, bbox_inches="tight")
    plt.close()
    results["D02_f1_slope"] = d02
    log(f"  D02 concluido.")

    log("D03 — Concordancia GNN vs Rule-Based ...")
    d03 = {}

    for frac in FRACTIONS:
        if frac not in preds_gnn or frac not in preds_rule:
            continue
        gnn  = preds_gnn[frac]
        rule = preds_rule[frac]
        y_true = labels_true.get(frac)

        concordante   = (gnn == rule)
        discordante   = ~concordante
        gnn_mais_rest = discordante & (gnn > rule)
        gnn_menos_rest = discordante & (gnn < rule)

        n_total     = len(gnn)
        n_discord   = int(discordante.sum())
        n_mais_rest = int(gnn_mais_rest.sum())
        n_menos_rest = int(gnn_menos_rest.sum())

        f1_discord_gnn, _ = f1_zone(y_true, gnn, discordante) if y_true is not None else (float("nan"), 0)
        f1_discord_rule, _ = f1_zone(y_true, rule, discordante) if y_true is not None else (float("nan"), 0)

        ndvi_discord_mean = float(ndvi_all[discordante].mean()) if discordante.sum() > 0 else float("nan")
        slope_discord_mean = float(slope_all[discordante].mean()) if discordante.sum() > 0 else float("nan")

        d03[frac] = {
            "n_total": n_total,
            "n_discordante": n_discord,
            "pct_discordante": round(n_discord / n_total * 100, 2),
            "n_gnn_mais_restritiva": n_mais_rest,
            "n_gnn_menos_restritiva": n_menos_rest,
            "f1_discordantes_gnn": round(float(f1_discord_gnn), 4),
            "f1_discordantes_rule": round(float(f1_discord_rule), 4),
            "ndvi_medio_discordantes": round(ndvi_discord_mean, 4),
            "slope_medio_discordantes": round(slope_discord_mean, 4),
        }

        if raster_meta is not None and dem_obj is not None:
            _export_discordance_tif(
                concordante, gnn_mais_rest, gnn_menos_rest,
                dem_obj, raster_meta, OUT_DIR / f"09_d03_discordancia_{frac}.tif", frac
            )

    results["D03_concordancia"] = d03
    log(f"  D03 concluido. Discordancias: "
        + " | ".join([f"{f}: {d03.get(f, {}).get('pct_discordante', 0):.1f}%" for f in FRACTIONS]))

    log('D04 — Feature profile of the discordant pixels ...')
    d04 = {}
    feat_names = ["elevation", "slope", "aspect_cos", "aspect_sin", "curvature",
                  "tpi", "tri", "roughness", "B02", "B03", "B04", "B08",
                  "NDVI", "NDWI", "water_mask", "lidar_avail", "lidar_elev"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('D04 — NDVI and Slope: Discordant vs Total', fontsize=13)

    for idx_f, frac in enumerate(FRACTIONS):
        if frac not in preds_gnn or frac not in preds_rule:
            continue
        gnn  = preds_gnn[frac]
        rule = preds_rule[frac]
        y_true = labels_true.get(frac)
        discord = gnn != rule

        row = {}
        for i, fname in enumerate(feat_names):
            col = feat[:, i]
            row[fname] = {
                "mean_all":       round(float(col.mean()), 4),
                "mean_discord":   round(float(col[discord].mean()), 4) if discord.sum() > 0 else None,
                "std_discord":    round(float(col[discord].std()), 4) if discord.sum() > 0 else None,
            }
            if discord.sum() > 0:
                z = (float(col[discord].mean()) - float(col.mean())) / (float(col.std()) + 1e-8)
                row[fname]["z_score"] = round(z, 3)

        d04[frac] = row

        ax = axes[idx_f // 2][idx_f % 2]
        thr = NDVI_THRESH[frac]
        ax.hist(ndvi_all, bins=80, alpha=0.4, color="blue", density=True, label="Todos nos")
        if discord.sum() > 0:
            ax.hist(ndvi_all[discord], bins=80, alpha=0.6, color="red", density=True, label=f"Discordantes GNN≠Rule")
        ax.axvline(x=thr, color="black", linestyle="--", linewidth=1.5, label=f"Thr DAMEPLAN={thr}")
        ax.set_title(f"{frac.upper()} — NDVI (discord={discord.sum():,})", fontsize=9)
        ax.set_xlabel("NDVI")
        ax.set_ylabel("Densidade")
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(str(OUT_DIR / "09_d04_feature_profile.png"), dpi=600, bbox_inches="tight")
    plt.close()
    results["D04_feature_profile"] = d04
    log(f"  D04 concluido.")

    log('D05 — NDVI gradient between neighbours ...')

    ndvi_grad = np.abs(ndvi_all[src] - ndvi_all[dst])
    slope_grad = np.abs(slope_all[src] - slope_all[dst])

    ndvi_grad_node  = np.zeros(N, dtype=np.float32)
    slope_grad_node = np.zeros(N, dtype=np.float32)
    count_node      = np.zeros(N, dtype=np.int32)

    np.add.at(ndvi_grad_node, dst, ndvi_grad)
    np.add.at(slope_grad_node, dst, slope_grad)
    np.add.at(count_node, dst, 1)

    valid = count_node > 0
    ndvi_grad_node[valid]  = ndvi_grad_node[valid]  / count_node[valid]
    slope_grad_node[valid] = slope_grad_node[valid] / count_node[valid]

    p75_ndvi  = float(np.percentile(ndvi_grad_node[valid], 75))
    p75_slope = float(np.percentile(slope_grad_node[valid], 75))

    mask_high_ndvi_grad  = ndvi_grad_node > p75_ndvi
    mask_high_slope_grad = slope_grad_node > p75_slope
    mask_low_ndvi_grad   = ndvi_grad_node <= p75_ndvi
    mask_low_slope_grad  = slope_grad_node <= p75_slope

    d05 = {"p75_ndvi_grad": round(p75_ndvi, 4), "p75_slope_grad": round(p75_slope, 4)}
    d06 = {}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('D05 — F1 at Biome Boundaries (High NDVI Gradient between Neighbours)', fontsize=12)
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('D06 — F1 at Topographic Boundaries (High Slope Gradient between Neighbours)', fontsize=12)

    for idx_f, frac in enumerate(FRACTIONS):
        if frac not in labels_true or frac not in preds_gnn:
            continue
        y_true = labels_true[frac]
        models = {"GNN": preds_gnn.get(frac), "RF": preds_rf.get(frac), "Rule": preds_rule.get(frac)}

        row_d05 = {}
        d05_plot = {}
        for zone_name, mask in [("Baixo grad. NDVI", mask_low_ndvi_grad),
                                  ("Alto grad. NDVI (fronteira bioma)", mask_high_ndvi_grad)]:
            row_d05[zone_name] = {}
            for mname, pred in models.items():
                if pred is None:
                    continue
                f1, cnt = f1_zone(y_true, pred, mask)
                row_d05[zone_name][mname] = {"f1": round(float(f1), 4), "n": cnt}
                d05_plot.setdefault(mname, {})[zone_name] = float(f1) if not np.isnan(f1) else 0.0
        d05[frac] = row_d05

        ax = axes[idx_f // 2][idx_f % 2]
        zone_labels_d05 = ["Baixo grad. NDVI", "Alto grad. NDVI (fronteira bioma)"]
        x = np.arange(len(zone_labels_d05))
        w = 0.25
        colors = {"GNN": "#0D47A1", "RF": "#2E7D32", "Rule": "#E65100"}
        for i, (mname, vals) in enumerate(d05_plot.items()):
            ys = [vals.get(z, 0) for z in zone_labels_d05]
            bars = ax.bar(x + i * w, ys, w, label=mname, color=colors[mname], alpha=0.85)
            for bar, y in zip(bars, ys):
                if y > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                            f"{y:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_title(f"{frac.upper()}", fontsize=10)
        ax.set_xticks(x + w)
        ax.set_xticklabels(zone_labels_d05, fontsize=8)
        ax.set_ylim(0.7, 1.02)
        ax.set_ylabel("F1 Macro")
        ax.legend(fontsize=8)

        row_d06 = {}
        d06_plot = {}
        for zone_name, mask in [("Baixo grad. slope", mask_low_slope_grad),
                                  ("Alto grad. slope (transicao topog.)", mask_high_slope_grad)]:
            row_d06[zone_name] = {}
            for mname, pred in models.items():
                if pred is None:
                    continue
                f1, cnt = f1_zone(y_true, pred, mask)
                row_d06[zone_name][mname] = {"f1": round(float(f1), 4), "n": cnt}
                d06_plot.setdefault(mname, {})[zone_name] = float(f1) if not np.isnan(f1) else 0.0
        d06[frac] = row_d06

        ax2 = axes2[idx_f // 2][idx_f % 2]
        zone_labels_d06 = ["Baixo grad. slope", "Alto grad. slope (transicao topog.)"]
        x2 = np.arange(len(zone_labels_d06))
        for i, (mname, vals) in enumerate(d06_plot.items()):
            ys = [vals.get(z, 0) for z in zone_labels_d06]
            bars = ax2.bar(x2 + i * w, ys, w, label=mname, color=colors[mname], alpha=0.85)
            for bar, y in zip(bars, ys):
                if y > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                             f"{y:.3f}", ha="center", va="bottom", fontsize=7)
        ax2.set_title(f"{frac.upper()}", fontsize=10)
        ax2.set_xticks(x2 + w)
        ax2.set_xticklabels(zone_labels_d06, fontsize=8)
        ax2.set_ylim(0.7, 1.02)
        ax2.set_ylabel("F1 Macro")
        ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(str(OUT_DIR / "09_d05_gradiente_ndvi.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    fig2.tight_layout()
    fig2.savefig(str(OUT_DIR / "09_d06_gradiente_slope.png"), dpi=600, bbox_inches="tight")
    plt.close(fig2)

    results["D05_gradiente_ndvi"]  = d05
    results["D06_gradiente_slope"] = d06
    log("  D05/D06 concluidos.")

    log('D15 — Micro-basins (valleys with neighbours on ridges) ...')
    tpi_all_np = tpi_all if isinstance(tpi_all, np.ndarray) else feat[:, IDX_TPI]

    is_vale   = tpi_all_np < 0
    is_crista = tpi_all_np > 0

    has_crista_neighbor = np.zeros(N, dtype=bool)
    crista_src_mask = is_crista[src]
    np.logical_or.at(has_crista_neighbor, dst[crista_src_mask], True)

    mask_microbacia = is_vale & has_crista_neighbor
    mask_plano      = (np.abs(tpi_all_np) <= 0.05)

    n_micro = int(mask_microbacia.sum())
    log(f"  Microbacias: {n_micro:,} nos ({n_micro/N*100:.1f}%)")

    d15 = {"n_microbacias": n_micro, "pct_microbacias": round(n_micro/N*100, 2)}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("D15 — F1 em Microbacias vs Terreno Plano", fontsize=13)

    for idx_f, frac in enumerate(FRACTIONS):
        if frac not in labels_true or frac not in preds_gnn:
            continue
        y_true = labels_true[frac]
        models = {"GNN": preds_gnn.get(frac), "RF": preds_rf.get(frac), "Rule": preds_rule.get(frac)}

        row = {}
        plot_data = {}
        for zone_name, mask in [("Plano/uniforme", mask_plano),
                                  ("Microbacia (vale+crista vizinha)", mask_microbacia)]:
            row[zone_name] = {}
            for mname, pred in models.items():
                if pred is None:
                    continue
                f1, cnt = f1_zone(y_true, pred, mask)
                row[zone_name][mname] = {"f1": round(float(f1), 4), "n": cnt}
                plot_data.setdefault(mname, {})[zone_name] = float(f1) if not np.isnan(f1) else 0.0
        d15[frac] = row

        ax = axes[idx_f // 2][idx_f % 2]
        zone_labels_m = ["Plano/uniforme", "Microbacia (vale+crista vizinha)"]
        x = np.arange(len(zone_labels_m))
        w = 0.25
        colors = {"GNN": "#0D47A1", "RF": "#2E7D32", "Rule": "#E65100"}
        for i, (mname, vals) in enumerate(plot_data.items()):
            ys = [vals.get(z, 0) for z in zone_labels_m]
            bars = ax.bar(x + i * w, ys, w, label=mname, color=colors[mname], alpha=0.85)
            for bar, y in zip(bars, ys):
                if y > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                            f"{y:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_title(f"{frac.upper()}", fontsize=10)
        ax.set_xticks(x + w)
        ax.set_xticklabels(zone_labels_m, fontsize=8, rotation=10)
        ax.set_ylim(0.7, 1.02)
        ax.set_ylabel("F1 Macro")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(str(OUT_DIR / "09_d15_microbacias.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    results["D15_microbacias"] = d15
    log("  D15 concluido.")

    results["meta"] = {
        "timestamp": datetime.now().isoformat(),
        "n_nos": N,
        "fracoes": FRACTIONS,
        "ndvi_thresholds": NDVI_THRESH,
        "slope_threshold": SLOPE_THRESH,
        "margin_near": MARGIN_NEAR,
        "margin_close": MARGIN_CLOSE,
        "p75_ndvi_grad": d05.get("p75_ndvi_grad"),
        "p75_slope_grad": d05.get("p75_slope_grad"),
        "elapsed_s": round(time.perf_counter() - start, 1),
    }

    out_json = OUT_DIR / "09_transicao_results.json"
    with open(str(out_json), "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.perf_counter() - start
    print("\n" + "=" * 60)
    print("SCRIPT 09 — CONCLUIDO")
    print(f"Tempo total: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Resultados em: {OUT_DIR}")
    print("Arquivos gerados:")
    for p in sorted(OUT_DIR.iterdir()):
        print(f"  {p.name}")
    print("=" * 60)


def _export_discordance_tif(concordante, mais_rest, menos_rest, dem, raster_meta, path, frac):
    try:
        import rasterio
        from rasterio.transform import Affine
        from rasterio.crs import CRS
    except ImportError:
        return

    H, W = raster_meta["dem_shape"]
    crs = CRS.from_user_input(raster_meta["crs"])
    t = raster_meta["dem_transform"]
    transform = Affine(*t[:6]) if isinstance(t, (list, tuple)) else Affine(t.a, t.b, t.c, t.d, t.e, t.f)

    pos = dem.pos.numpy() if hasattr(dem, "pos") else None
    if pos is None:
        return

    lats, lons = pos[:, 0], pos[:, 1]
    rows_px = np.clip(((transform.f - lats) / abs(transform.e)).astype(int), 0, H - 1)
    cols_px = np.clip(((lons - transform.c) / abs(transform.a)).astype(int), 0, W - 1)

    arr = np.zeros((H, W), dtype=np.uint8)
    arr[rows_px[concordante], cols_px[concordante]] = 0
    arr[rows_px[mais_rest], cols_px[mais_rest]] = 1
    arr[rows_px[menos_rest], cols_px[menos_rest]] = 2

    cmap = {0: (180, 180, 180), 1: (220, 50, 50), 2: (50, 50, 220)}
    with rasterio.open(str(path), "w", driver="GTiff", height=H, width=W,
                       count=1, dtype="uint8", crs=crs, transform=transform,
                       nodata=255, compress="deflate") as dst:
        dst.write(arr, 1)
        dst.write_colormap(1, cmap)
        dst.update_tags(FRAC=frac, CLASSES="0=Concordante,1=GNN_mais_restritiva,2=GNN_menos_restritiva")


if __name__ == "__main__":
    main()
