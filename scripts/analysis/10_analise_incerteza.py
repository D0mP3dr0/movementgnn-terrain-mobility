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
from sklearn.metrics import f1_score

try:
    import google.colab  # noqa: F401
    IS_COLAB = True
except ImportError:
    IS_COLAB = False
_HERE    = Path(__file__).resolve().parent

def _glob_first(root, pattern):
    candidates = sorted(root.glob(pattern), reverse=True)
    return candidates[0] if candidates else None

if IS_COLAB:
    DRIVE_BASE   = Path(f"{INTERNAL_DATA_ROOT}")
    GRAPH_PATH   = DRIVE_BASE / "graph/pacaraima_q1q4_fixed.pt"
    GNN_PREDS    = DRIVE_BASE / "gnn/results_v2/predictions_full.npz"
    GNN_PROBS    = DRIVE_BASE / "gnn/results_v2/probs_full.npz"
    RF_PREDS     = DRIVE_BASE / "baseline_classic/results/random_forest/predictions_full.npz"
    MLP_PREDS    = DRIVE_BASE / "baseline_classic/results/mlp/predictions_full.npz"
    RULE_PREDS   = DRIVE_BASE / "baseline_classic/results/rule_based/predictions_full.npz"
    OUT_DIR      = DRIVE_BASE / "gnn/analise/10_incerteza"
    DATA_DIR     = None
else:
    DATA_DIR   = _HERE / "analise_dados"
    GNN_PREDS  = _HERE / "results" / "results_v2_local_blindada_fix" / "predictions_full.npz"
    GNN_PROBS  = _HERE / "results" / "results_v2_local_blindada_fix" / "probs_full.npz"
    RF_PREDS   = _glob_first(_HERE / "results",
                             "random_forest-*/random_forest/predictions_full.npz")
    MLP_PREDS  = _glob_first(_HERE / "results",
                             "mlp-*/mlp/predictions_full.npz")
    RULE_PREDS = _glob_first(_HERE / "results",
                             "rule_based-*/rule_based/predictions_full.npz")
    GRAPH_PATH = None
    OUT_DIR    = _HERE / "analise_dados/10_incerteza"
    print(f"[MODO LOCAL — v2_local] DATA_DIR={DATA_DIR}")

FRACTIONS = ["a_pe", "motorizada", "mecanizada", "blindada"]
IDX_NDVI  = 12
IDX_NDWI  = 13
IDX_SLOPE = 1
IDX_TPI   = 5

NDVI_THRESH = {"a_pe": 0.7, "motorizada": 0.5, "mecanizada": 0.6, "blindada": 0.5}
SLOPE_THRESH = 0.35
MARGIN = 0.05


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_preds(path, fractions):
    data = np.load(str(path))
    out = {}
    for f in fractions:
        if f in data:
            out[f] = data[f].astype(np.int32)
        elif f"y_{f}" in data:
            out[f] = data[f"y_{f}"].astype(np.int32)
    return out


def load_probs(path, fractions):
    data = np.load(str(path))
    out = {}
    for f in fractions:
        key = f"probs_{f}" if f"probs_{f}" in data else f
        if key in data:
            out[f] = data[key].astype(np.float32)
    return out


def _norm0(arr):
    if arr is None:
        return None
    a = arr.astype(np.int32)
    return a - 1 if int(a.min()) >= 1 else a


def f1_zone(y_true, y_pred, mask):
    if mask.sum() == 0:
        return float("nan"), 0
    return f1_score(y_true[mask], y_pred[mask], average="macro",
                    zero_division=0), int(mask.sum())


def entropy(probs):
    eps = 1e-8
    h = -np.sum(probs * np.log(probs + eps), axis=1)
    return h


def main():
    start = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    if IS_COLAB:
        log("Carregando grafo ...")
        data = torch.load(str(GRAPH_PATH), map_location="cpu", weights_only=False)
        dem  = data["dem"]
        feat = dem.x.float().numpy()
        N    = feat.shape[0]
        labels_true = {}
        for f in FRACTIONS:
            attr = f"y_{f}"
            if hasattr(dem, attr):
                labels_true[f] = getattr(dem, attr).numpy().astype(np.int32)
        raster_meta = getattr(data, "raster_meta", None)
        ei = data[("dem", "adjacent_to", "dem")].edge_index.numpy()
        src, dst = ei[0], ei[1]
        dem_obj  = dem
        lidar_et = next((et for et in data.edge_types if "lidar" in str(et).lower()), None)
        lidar_ei = data[lidar_et].edge_index.numpy() if lidar_et else None
    else:
        log(f"[Local] Loading from {DATA_DIR} ...")
        assert DATA_DIR and DATA_DIR.exists(), \
            f"DATA_DIR not found: {DATA_DIR}\nRun 00_extrair_dados_locais.py on Colab."
        f_slim = np.load(str(DATA_DIR / "features_slim.npz"))
        N = len(f_slim["NDVI"])
        feat = np.zeros((N, 17), dtype=np.float32)
        slim_map = {"elevation":0,"slope":1,"tpi":5,"tri":6,
                    "NDVI":12,"NDWI":13,"lidar_avail":15,"lidar_elev":16}
        for k, idx in slim_map.items():
            if k in f_slim:
                feat[:, idx] = f_slim[k]
        lbl = np.load(str(DATA_DIR / "labels.npz"))
        labels_true = {f: lbl[f].astype(np.int32) for f in FRACTIONS if f in lbl}
        edges = np.load(str(DATA_DIR / "edges.npz"))
        src, dst = edges["src"], edges["dst"]
        raster_meta = None
        rm_path = DATA_DIR / "raster_meta.json"
        if rm_path.exists():
            with open(str(rm_path)) as fp:
                raster_meta = json.load(fp)
        dem_obj  = None
        data     = None
        lidar_et = None
        lidar_np_path = DATA_DIR / "lidar_edges.npz"
        lidar_ei = np.load(str(lidar_np_path)) if lidar_np_path.exists() else None

    ndvi_all  = feat[:, IDX_NDVI]
    slope_all = feat[:, IDX_SLOPE]
    has_lidar = feat[:, 15] > 0.5
    tpi_all   = feat[:, IDX_TPI]
    log(f"  N={N:,} | arestas={len(src):,} | LiDAR: {has_lidar.sum():,} ({has_lidar.mean()*100:.1f}%)")

    log("Carregando predicoes ...")
    preds_gnn  = load_preds(GNN_PREDS,  FRACTIONS)
    preds_rf   = load_preds(RF_PREDS,   FRACTIONS)
    preds_mlp  = load_preds(MLP_PREDS,  FRACTIONS)
    preds_rule = load_preds(RULE_PREDS, FRACTIONS)

    has_probs = GNN_PROBS is not None and GNN_PROBS.exists()
    probs_gnn = {}
    if has_probs:
        log("Carregando probabilidades GNN ...")
        probs_gnn = load_probs(GNN_PROBS, FRACTIONS)
        log(f"  Probabilidades: {list(probs_gnn.keys())}")
    else:
        log('  WARNING: probs_full.npz not found — D07/D08 will be skipped')

    log('Normalizing prediction and label indices to 0-based ...')
    for _f in FRACTIONS:
        if _f in labels_true:
            labels_true[_f] = _norm0(labels_true[_f])
        if _f in preds_gnn:
            preds_gnn[_f]  = _norm0(preds_gnn[_f])
        if _f in preds_rf:
            preds_rf[_f]   = _norm0(preds_rf[_f])
        if _f in preds_mlp:
            preds_mlp[_f]  = _norm0(preds_mlp[_f])
        if _f in preds_rule:
            preds_rule[_f] = _norm0(preds_rule[_f])
    if preds_gnn:
        sample_frac = next(iter(preds_gnn))
        log(f"  Verificacao pos-norm — GNN {sample_frac}: "
            f"min={preds_gnn[sample_frac].min()} max={preds_gnn[sample_frac].max()} | "
            f"label min={labels_true[sample_frac].min()} max={labels_true[sample_frac].max()}")

    gc.collect()

    if has_probs and raster_meta is not None:
        log("D07 — Mapa de entropia softmax ...")
        d07 = {}
        for frac in FRACTIONS:
            if frac not in probs_gnn:
                continue
            p = probs_gnn[frac]
            H = entropy(p)

            thr = NDVI_THRESH[frac]
            in_trans = np.abs(ndvi_all - thr) <= MARGIN

            d07[frac] = {
                "entropia_media_total":      round(float(H.mean()), 4),
                "entropia_media_transicao":  round(float(H[in_trans].mean()), 4) if in_trans.sum() > 0 else None,
                "entropia_media_fora_trans": round(float(H[~in_trans].mean()), 4) if (~in_trans).sum() > 0 else None,
                "pct_alta_entropia_em_trans": None,
            }

            high_H = H > np.percentile(H, 75)
            if in_trans.sum() > 0:
                d07[frac]["pct_alta_entropia_em_trans"] = round(
                    float((high_H & in_trans).sum() / in_trans.sum() * 100), 2)

            if dem_obj is not None and raster_meta is not None:
                _export_entropy_tif(H, dem_obj, raster_meta, OUT_DIR / f"10_d07_entropia_{frac}.tif", frac)

        results["D07_entropia"] = d07
        log("  D07 concluido.")
    else:
        results["D07_entropia"] = "PULADO — probs_full.npz nao disponivel"

    if has_probs:
        log("D08 — Distribuicao probabilidade: acertos vs erros ...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("D08 — Confianca GNN: Acertos vs Erros", fontsize=13)
        d08 = {}

        for idx_f, frac in enumerate(FRACTIONS):
            if frac not in probs_gnn or frac not in labels_true or frac not in preds_gnn:
                continue
            p       = probs_gnn[frac]
            y_true  = labels_true[frac]
            y_pred  = preds_gnn[frac]
            conf    = p.max(axis=1)

            acerto = y_pred == y_true

            conf_acerto = conf[acerto]
            conf_erro   = conf[~acerto]

            d08[frac] = {
                "confianca_media_acerto": round(float(conf_acerto.mean()), 4),
                "confianca_media_erro":   round(float(conf_erro.mean()), 4),
                "pct_erro_alta_conf":     round(float((conf_erro > 0.8).mean() * 100), 2),
                "n_erros":                int((~acerto).sum()),
                "n_acertos":              int(acerto.sum()),
            }

            ax = axes[idx_f // 2][idx_f % 2]
            ax.hist(conf_acerto, bins=60, alpha=0.5, color="blue", density=True, label=f"Acertos (n={acerto.sum():,})")
            ax.hist(conf_erro,   bins=60, alpha=0.6, color="red",  density=True, label=f"Erros (n={(~acerto).sum():,})")
            ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
            ax.axvline(x=0.8, color="orange", linestyle="--", alpha=0.5, label="Thr alta confianca=0.8")
            ax.set_title(f"{frac.upper()} | Conf.err={conf_erro.mean():.3f}", fontsize=9)
            ax.set_xlabel("Probabilidade maxima (confianca)")
            ax.set_ylabel("Densidade")
            ax.legend(fontsize=7)

        plt.tight_layout()
        plt.savefig(str(OUT_DIR / "10_d08_prob_distribuicao.png"), dpi=600, bbox_inches="tight")
        plt.close()
        results["D08_prob_dist"] = d08
        log("  D08 concluido.")
    else:
        results["D08_prob_dist"] = "PULADO — probs_full.npz nao disponivel"

    log("D09 — Coerencia espacial (pixels solitarios) ...")
    d09 = {}

    all_models = {"GNN": preds_gnn, "RF": preds_rf, "MLP": preds_mlp, "Rule": preds_rule}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('D09 — Isolated Pixels per Model (Spatial Incoherence)', fontsize=12)

    for idx_f, frac in enumerate(FRACTIONS):
        row = {}
        model_pcts = {}

        for mname, preds in all_models.items():
            if frac not in preds:
                continue
            pred = preds[frac]

            edge_discord = (pred[src] != pred[dst]).astype(np.int32)
            discord_count = np.zeros(N, dtype=np.int32)
            np.add.at(discord_count, dst, edge_discord)

            degree = np.zeros(N, dtype=np.int32)
            np.add.at(degree, dst, 1)

            valid = degree > 0
            frac_discord = np.zeros(N, dtype=np.float32)
            frac_discord[valid] = discord_count[valid] / degree[valid]

            isolado = valid & (frac_discord > 0.5)
            pct_isolado = float(isolado.sum()) / float(valid.sum()) * 100

            row[mname] = {
                "n_isolados": int(isolado.sum()),
                "pct_isolados": round(pct_isolado, 3),
                "discord_medio": round(float(frac_discord[valid].mean()), 4),
            }
            model_pcts[mname] = pct_isolado

        d09[frac] = row

        ax = axes[idx_f // 2][idx_f % 2]
        mnames = list(model_pcts.keys())
        pcts   = [model_pcts[m] for m in mnames]
        colors = ["#0D47A1", "#2E7D32", "#6A1B9A", "#E65100"]
        bars = ax.bar(mnames, pcts, color=colors[:len(mnames)], alpha=0.85)
        for bar, pct in zip(bars, pcts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f"{pct:.3f}%", ha="center", va="bottom", fontsize=9)
        ax.set_title(f"{frac.upper()} — % Pixels Solitarios", fontsize=10)
        ax.set_ylabel('% nodes with >50% neighbours of a different class')
        ax.set_ylim(0, max(pcts) * 1.3 if pcts else 1)

    fig.tight_layout()
    fig.savefig(str(OUT_DIR / "10_d09_coerencia_espacial.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    results["D09_coerencia_espacial"] = d09
    log("  D09 concluido.")

    log('D10 — LiDAR impact by proximity ...')
    if lidar_ei is not None:
        lidar_arr = lidar_ei if isinstance(lidar_ei, np.ndarray) else np.stack(
            [lidar_ei["src"], lidar_ei["dst"]])
        dem_nodes_with_lidar_neighbor = np.zeros(N, dtype=bool)
        dem_nodes_with_lidar_neighbor[lidar_arr[1] if lidar_arr.ndim == 2 else lidar_ei["dst"]] = True
        has_lidar_neighbor = dem_nodes_with_lidar_neighbor
    else:
        has_lidar_neighbor = has_lidar

    log(f"  DEM nodes with a LiDAR neighbour: {has_lidar_neighbor.sum():,} ({has_lidar_neighbor.mean()*100:.1f}%)")

    d10 = {}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('D10 — GNN F1: With vs Without LiDAR Neighbour', fontsize=13)

    for idx_f, frac in enumerate(FRACTIONS):
        if frac not in labels_true or frac not in preds_gnn:
            continue
        y_true = labels_true[frac]
        models = {"GNN": preds_gnn.get(frac), "RF": preds_rf.get(frac)}

        row = {}
        plot_data = {}
        for zone_name, mask in [("Sem LiDAR vizinho", ~has_lidar_neighbor),
                                  ("Com LiDAR vizinho", has_lidar_neighbor)]:
            row[zone_name] = {}
            for mname, pred in models.items():
                if pred is None:
                    continue
                f1, cnt = f1_zone(y_true, pred, mask)
                row[zone_name][mname] = {"f1": round(float(f1), 4), "n": cnt}
                plot_data.setdefault(mname, {})[zone_name] = float(f1) if not np.isnan(f1) else 0.0
        d10[frac] = row

        ax = axes[idx_f // 2][idx_f % 2]
        zones = ["Sem LiDAR vizinho", "Com LiDAR vizinho"]
        x = np.arange(len(zones))
        w = 0.35
        colors = {"GNN": "#0D47A1", "RF": "#2E7D32"}
        for i, (mname, vals) in enumerate(plot_data.items()):
            ys = [vals.get(z, 0) for z in zones]
            bars = ax.bar(x + i * w, ys, w, label=mname, color=colors[mname], alpha=0.85)
            for bar, y in zip(bars, ys):
                if y > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                            f"{y:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"{frac.upper()}", fontsize=10)
        ax.set_xticks(x + w / 2)
        ax.set_xticklabels(zones, fontsize=9)
        ax.set_ylim(0.8, 1.02)
        ax.set_ylabel("F1 Macro")
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(str(OUT_DIR / "10_d10_lidar_impacto.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    results["D10_lidar_impacto"] = d10
    log("  D10 concluido.")

    log("D13 — Consensus-hard nodes (onde RF+MLP erram simultaneamente) ...")
    d13 = {}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("D13 — F1 nos Consensus-Hard Nodes (RF+MLP erram juntos)", fontsize=12)

    for idx_f, frac in enumerate(FRACTIONS):
        if frac not in labels_true:
            continue
        y_true  = labels_true[frac]
        rf_pred  = preds_rf.get(frac)
        mlp_pred = preds_mlp.get(frac)
        gnn_pred = preds_gnn.get(frac)

        if rf_pred is None or mlp_pred is None:
            continue

        rf_wrong  = rf_pred  != y_true
        mlp_wrong = mlp_pred != y_true
        mask_hard = rf_wrong & mlp_wrong

        row = {
            "n_hard_nodes": int(mask_hard.sum()),
            "pct_hard":     round(float(mask_hard.mean() * 100), 3),
        }

        for mname, pred in [("GNN", gnn_pred), ("RF", rf_pred), ("MLP", mlp_pred)]:
            if pred is None:
                continue
            f1, cnt = f1_zone(y_true, pred, mask_hard)
            row[f"f1_{mname}_em_hard"] = round(float(f1), 4)

        mask_easy = ~rf_wrong & ~mlp_wrong
        for mname, pred in [("GNN", gnn_pred), ("RF", rf_pred), ("MLP", mlp_pred)]:
            if pred is None:
                continue
            f1, _ = f1_zone(y_true, pred, mask_easy)
            row[f"f1_{mname}_em_easy"] = round(float(f1), 4)

        d13[frac] = row

        ax = axes[idx_f // 2][idx_f % 2]
        models_names = ["GNN", "RF", "MLP"]
        f1_hard  = [row.get(f"f1_{m}_em_hard", 0) for m in models_names if row.get(f"f1_{m}_em_hard") is not None]
        f1_easy  = [row.get(f"f1_{m}_em_easy", 0) for m in models_names if row.get(f"f1_{m}_em_easy") is not None]
        valid_m  = [m for m in models_names if row.get(f"f1_{m}_em_hard") is not None]

        x = np.arange(len(valid_m))
        w = 0.35
        bars1 = ax.bar(x - w/2, f1_hard, w, label=f"Hard (n={row['n_hard_nodes']:,})",
                       color=["#C62828", "#E65100", "#6A1B9A"][:len(valid_m)], alpha=0.90)
        bars2 = ax.bar(x + w/2, f1_easy, w, label=f"Easy (n={mask_easy.sum():,})",
                       color=["#0D47A1", "#2E7D32", "#00695C"][:len(valid_m)], alpha=0.90)
        for bars in [bars1, bars2]:
            for bar in bars:
                y = bar.get_height()
                if y > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, y + 0.001,
                            f"{y:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(valid_m)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("F1 Macro")
        ax.set_title(f"{frac.upper()} — Hard={row['pct_hard']:.2f}%", fontsize=9)
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(str(OUT_DIR / "10_d13_consensus_hard.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    results["D13_consensus_hard"] = d13
    log(f"  D13 done. Hard nodes: "
        + " | ".join([f"{f}: {d13.get(f, {}).get('n_hard_nodes', 0):,}" for f in FRACTIONS]))

    log("D14 — Confusao assimetrica (conservador vs perigoso) ...")

    d14 = {}
    gnn_cm_path = DRIVE_BASE / "gnn/results_v2/metrics.json" if IS_COLAB else \
                  (_glob_first(_HERE / "GNN/results", "results_v2-*/results_v2/metrics.json")
                   or Path("_nao_existe"))

    gnn_cm = {}
    if gnn_cm_path.exists():
        with open(str(gnn_cm_path)) as f:
            gnn_metrics = json.load(f)
        for frac in FRACTIONS:
            if frac in gnn_metrics and "confusion_matrix" in gnn_metrics[frac]:
                gnn_cm[frac] = np.array(gnn_metrics[frac]["confusion_matrix"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("D14 — Erros Conservadores vs Perigosos por Fracao", fontsize=13)

    for idx_f, frac in enumerate(FRACTIONS):
        row = {}
        ax = axes[idx_f // 2][idx_f % 2]

        if frac in gnn_cm:
            cm = gnn_cm[frac]
            n  = cm.sum()

            erro_perigoso_gnn = (cm[1, 0] + cm[2, 0]) / n * 100
            erro_conserv_gnn  = (cm[0, 2] + cm[1, 2]) / n * 100

            row["GNN"] = {
                "erro_perigoso_pct":    round(erro_perigoso_gnn, 4),
                "erro_conservador_pct": round(erro_conserv_gnn, 4),
                "ratio_conserv_per_perigoso": round(erro_conserv_gnn / (erro_perigoso_gnn + 1e-8), 2),
            }

        y_true = labels_true.get(frac)
        rf_pred = preds_rf.get(frac)
        if y_true is not None and rf_pred is not None:
            n_total = len(y_true)

            perig_rf   = ((y_true == 1) | (y_true == 2)) & (rf_pred == 0)
            conserv_rf = ((y_true == 0) | (y_true == 1)) & (rf_pred == 2)

            row["RF"] = {
                "erro_perigoso_pct":    round(float(perig_rf.sum()) / n_total * 100, 4),
                "erro_conservador_pct": round(float(conserv_rf.sum()) / n_total * 100, 4),
            }

        d14[frac] = row

        model_names = list(row.keys())
        perigos     = [row[m]["erro_perigoso_pct"] for m in model_names]
        conservs    = [row[m]["erro_conservador_pct"] for m in model_names]

        x = np.arange(len(model_names))
        w = 0.35
        bars1 = ax.bar(x - w/2, perigos,  w, label="Erro Perigoso (subest. restricao)",
                       color="#C62828", alpha=0.90)
        bars2 = ax.bar(x + w/2, conservs, w, label="Erro Conservador (superest. restricao)",
                       color="#0D47A1", alpha=0.90)
        for bars in [bars1, bars2]:
            for bar in bars:
                y = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, y + 0.001,
                        f"{y:.4f}%", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.set_title(f"{frac.upper()}", fontsize=10)
        ax.set_ylabel('% of nodes misclassified')
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(str(OUT_DIR / "10_d14_confusao_assimetrica.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    results["D14_confusao_assimetrica"] = d14
    log("  D14 concluido.")

    results["meta"] = {
        "timestamp": datetime.now().isoformat(),
        "n_nos": N,
        "fracoes": FRACTIONS,
        "probs_disponivel": has_probs,
        "elapsed_s": round(time.perf_counter() - start, 1),
    }

    out_json = OUT_DIR / "10_incerteza_results.json"
    with open(str(out_json), "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.perf_counter() - start
    print("\n" + "=" * 60)
    print("SCRIPT 10 — CONCLUIDO")
    print(f"Tempo total: {elapsed:.0f}s")
    print(f"Resultados em: {OUT_DIR}")
    for p in sorted(OUT_DIR.iterdir()):
        print(f"  {p.name}")
    print("=" * 60)


def _export_entropy_tif(entropy_arr, dem, raster_meta, path, frac):
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

    arr = np.full((H, W), fill_value=-1.0, dtype=np.float32)
    arr[rows_px, cols_px] = entropy_arr

    with rasterio.open(str(path), "w", driver="GTiff", height=H, width=W,
                       count=1, dtype="float32", crs=crs, transform=transform,
                       nodata=-1.0, compress="deflate") as dst:
        dst.write(arr, 1)
        dst.update_tags(FRAC=frac, CONTENT="Entropia_Softmax_GNN")


if __name__ == "__main__":
    main()
