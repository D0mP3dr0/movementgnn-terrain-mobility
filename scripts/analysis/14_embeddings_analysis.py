INTERNAL_DATA_ROOT = __import__("os").environ.get("MOVEMENTGNN_DATA_ROOT", ".")
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import google.colab  # noqa: F401
    IS_COLAB = True
except ImportError:
    IS_COLAB = False
_HERE    = Path(__file__).resolve().parent

if IS_COLAB:
    DRIVE_BASE = Path(f"{INTERNAL_DATA_ROOT}")
    EMB_PATH   = DRIVE_BASE / "graph/pacaraima_q1q4_embeddings_256.pt"
    GRAPH_PATH = DRIVE_BASE / "graph/pacaraima_q1q4_fixed.pt"
    OUT_DIR    = DRIVE_BASE / "gnn/analise/14_embeddings"
    DATA_DIR   = None
else:
    DATA_DIR   = _HERE / "analise_dados"
    _GRAPH_DATA = (_HERE.parent / "GNN_RESTRICAO_MOVIMENTO" / "graph_data")
    EMB_PATH   = _GRAPH_DATA / "pacaraima_q1q4_embeddings_256.pt"
    GRAPH_PATH = _GRAPH_DATA / "pacaraima_q1q4_ready.pt"
    OUT_DIR    = _HERE / "analise_dados/14_embeddings"
    print(f"[MODO LOCAL — v2_local] EMB_PATH={EMB_PATH}")

FRACTIONS = ["a_pe", "motorizada", "mecanizada", "blindada"]

FEATURE_NAMES_ALL = [
    "elevation", "slope", "aspect_cos", "aspect_sin", "curvature",
    "tpi", "tri", "roughness", "B02", "B03", "B04", "B08",
    "NDVI", "NDWI", "water_mask", "lidar_avail", "lidar_elev",
]
FEATURE_NAMES = [f for f in FEATURE_NAMES_ALL if f not in ("lidar_avail", "lidar_elev")]
FEATURE_INDICES = [i for i, f in enumerate(FEATURE_NAMES_ALL) if f not in ("lidar_avail", "lidar_elev")]

N_SAMPLE = 500_000


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main():
    start = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    pca_precomputed = False
    if IS_COLAB or (not IS_COLAB and EMB_PATH is not None and EMB_PATH.exists()):
        _path = EMB_PATH
        log(f"Carregando embeddings: {_path.name} ...")
        emb_data = torch.load(str(_path), map_location="cpu", weights_only=False)
        if isinstance(emb_data, torch.Tensor):
            emb = emb_data.float().numpy()
        elif isinstance(emb_data, dict):
            emb = emb_data.get("embeddings", list(emb_data.values())[0]).float().numpy()
        else:
            emb = emb_data.float().numpy()
        N, D = emb.shape
    else:
        pca_npz = DATA_DIR / "embeddings_pca.npz"
        assert pca_npz.exists(), \
            f"embeddings_pca.npz not found in {DATA_DIR}\nRun 00_extrair_dados_locais.py on Colab."
        log(f"[Local] Carregando embeddings_pca.npz ...")
        pca_npz_data = np.load(str(pca_npz))
        pca_components = pca_npz_data["pca_components"]
        evr = pca_npz_data["explained_variance_ratio"]
        N, _n_comp = pca_components.shape
        D = 256
        pca_precomputed = True
        log(f"  PCA pre-calculado: {N:,} nos x {_n_comp} PCs | var acumulada: {evr.sum()*100:.1f}%")
        emb = None

    log(f"  Embeddings: {N:,} nos x {D} dims")

    if not pca_precomputed:
        results["meta_embeddings"] = {
            "n_nos": N, "n_dims": D,
            "mean_norm": round(float(np.linalg.norm(emb, axis=1).mean()), 4),
            "std_norm":  round(float(np.linalg.norm(emb, axis=1).std()), 4),
        }

    log("D18 — PCA nos embeddings ...")
    if pca_precomputed:
        explained   = evr
        cumulative  = np.cumsum(explained)
        pca         = None
        scaler      = None
        n_sample    = min(N_SAMPLE, N)
        idx_sample  = np.arange(n_sample)
        emb_sample  = pca_components[idx_sample]
    else:
        rng = np.random.default_rng(42)
        n_sample = min(N_SAMPLE, N)
        idx_sample = rng.choice(N, size=n_sample, replace=False)
        emb_sample = emb[idx_sample]
        log(f"  Amostra PCA: {n_sample:,} nos")
        scaler = StandardScaler()
        emb_scaled = scaler.fit_transform(emb_sample)
        pca = PCA(n_components=min(D, 64), random_state=42)
        pca.fit(emb_scaled)
        explained = pca.explained_variance_ratio_
        cumulative = np.cumsum(explained)

    dims_90 = int(np.searchsorted(cumulative, 0.90)) + 1
    dims_95 = int(np.searchsorted(cumulative, 0.95)) + 1
    dims_99 = int(np.searchsorted(cumulative, 0.99)) + 1

    d18 = {
        "n_dims_total":  D,
        "n_sample":      n_sample,
        "dims_para_90pct_variancia": dims_90,
        "dims_para_95pct_variancia": dims_95,
        "dims_para_99pct_variancia": dims_99,
        "variancia_PC1": round(float(explained[0]) * 100, 2),
        "variancia_PC2": round(float(explained[1]) * 100, 2),
        "variancia_PC3": round(float(explained[2]) * 100, 2),
        "variancia_top5": round(float(cumulative[4]) * 100, 2),
        "variancia_top10": round(float(cumulative[9]) * 100, 2),
        "interpretacao": (
            "REDUNDANTE — poucas dims explicam a variancia (embeddings comprimidos)"
            if dims_90 <= 10 else
            "INFORMATIVO — muitas dims necessarias para 90% variancia (embeddings ricos)"
        )
    }
    results["D18_pca"] = d18

    log(f"  PC1={explained[0]*100:.1f}% | PC2={explained[1]*100:.1f}%")
    log(f"  Dims para 90%: {dims_90} | 95%: {dims_95} | 99%: {dims_99}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"PCA on {D}-dim GNN Embeddings", fontsize=13)

    ax1 = axes[0]
    n_show = min(50, len(explained))
    ax1.bar(range(1, n_show + 1), explained[:n_show] * 100,
            color="#0D47A1", alpha=0.85, label="Individual variance")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance (%)")
    ax1.set_title("Individual Variance per PC")
    ax1.axhline(y=float(explained[0]) * 100, color="gray", linestyle="--", alpha=0.4)

    ax2 = axes[1]
    ax2.plot(range(1, len(cumulative) + 1), cumulative * 100,
             color="#C62828", linewidth=2.5, label="Cumulative variance")
    ax2.axhline(y=90, color="gray", linestyle="--", alpha=0.7, label="90%")
    ax2.axhline(y=95, color="orange", linestyle="--", alpha=0.7, label="95%")
    ax2.axvline(x=dims_90, color="gray", linestyle=":", alpha=0.7,
                label=f"n={dims_90} dims for 90%")
    ax2.fill_between(range(1, dims_90 + 1), 0, cumulative[:dims_90] * 100,
                     alpha=0.20, color="#0D47A1")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Explained Variance (%)")
    ax2.set_title(f"{dims_90} dims for 90% variance ({dims_90/D*100:.0f}% of {D} dims)")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 102)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(OUT_DIR / "14_d18_pca_variancia.png"), dpi=600, bbox_inches="tight")
    plt.close()

    feat_avail = (IS_COLAB and GRAPH_PATH and GRAPH_PATH.exists()) or \
                 (not IS_COLAB and DATA_DIR and (DATA_DIR / "features_full.npz").exists())
    if feat_avail:
        log("Correlacionando PCs com features topograficas ...")
        if IS_COLAB:
            data = torch.load(str(GRAPH_PATH), map_location="cpu", weights_only=False)
            dem  = data["dem"]
            feat = dem.x.float().numpy()[idx_sample]
        else:
            arr  = np.load(str(DATA_DIR / "features_full.npz"))
            feat = arr["features"][idx_sample].astype(np.float32)

        if pca_precomputed:
            pcs = emb_sample
        else:
            pcs = pca.transform(emb_scaled)

        n_pcs_check = min(10, pcs.shape[1])
        corr_matrix = np.zeros((n_pcs_check, len(FEATURE_NAMES)))

        for i in range(n_pcs_check):
            for j, fidx in enumerate(FEATURE_INDICES):
                corr = float(np.corrcoef(pcs[:, i], feat[:, fidx])[0, 1])
                corr_matrix[i, j] = corr

        fig, ax = plt.subplots(figsize=(14, 7))
        im = ax.imshow(np.abs(corr_matrix), aspect="auto", cmap="Blues", vmin=0, vmax=0.5)
        ax.set_xticks(range(len(FEATURE_NAMES)))
        ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_pcs_check))
        ax.set_yticklabels([f"PC{i+1}" for i in range(n_pcs_check)], fontsize=9)
        ax.set_title("|Pearson r| between Embedding PCs and Input Features", fontsize=11)
        plt.colorbar(im, ax=ax, label="|Pearson r|")
        for i in range(n_pcs_check):
            for j in range(len(FEATURE_NAMES)):
                ax.text(j, i, f"{abs(corr_matrix[i, j]):.2f}",
                        ha="center", va="center", fontsize=5.5,
                        color="white" if abs(corr_matrix[i, j]) > 0.3 else "black")
        plt.tight_layout()
        plt.savefig(str(OUT_DIR / "14_d18_pca_correlacoes.png"), dpi=600, bbox_inches="tight")
        plt.close()

        top_corrs = {}
        for i in range(n_pcs_check):
            top_idx = np.argsort(np.abs(corr_matrix[i]))[::-1][:3]
            top_corrs[f"PC{i+1}"] = [
                {"feature": FEATURE_NAMES[j], "r": round(float(corr_matrix[i, j]), 4)}
                for j in top_idx
            ]
        results["D18_pc_feature_correlacoes"] = top_corrs
        log("  Correlacoes PCA-features concluidas.")

        log("Scatter PC1 vs PC2 ...")
        ndvi_sample = feat[:, 12]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Embedding Space: PC1 vs PC2", fontsize=13)

        sc1 = axes[0].scatter(pcs[:5000, 0], pcs[:5000, 1],
                               c=ndvi_sample[:5000], cmap="RdYlGn",
                               s=2, alpha=0.5, vmin=0, vmax=1)
        axes[0].set_xlabel("PC1")
        axes[0].set_ylabel("PC2")
        axes[0].set_title("Colored by NDVI")
        plt.colorbar(sc1, ax=axes[0], label="NDVI")

        slope_sample = feat[:, 1]
        sc2 = axes[1].scatter(pcs[:5000, 0], pcs[:5000, 1],
                               c=slope_sample[:5000], cmap="YlOrRd",
                               s=2, alpha=0.5, vmin=0, vmax=1)
        axes[1].set_xlabel("PC1")
        axes[1].set_ylabel("PC2")
        axes[1].set_title("Colored by Slope")
        plt.colorbar(sc2, ax=axes[1], label="Slope")

        plt.tight_layout()
        plt.savefig(str(OUT_DIR / "14_d18_pca_scatter.png"), dpi=600, bbox_inches="tight")
        plt.close()

        if IS_COLAB:
            raster_meta = getattr(data, "raster_meta", None)
        else:
            rm_path = DATA_DIR / "raster_meta.json"
            if rm_path.exists():
                with open(str(rm_path)) as _f:
                    raster_meta = json.load(_f)
            else:
                raster_meta = None
        if raster_meta is not None:
            log("Exportando PC1 como GeoTIFF ...")
            pc1_all = np.zeros(N, dtype=np.float32)
            if pca_precomputed:
                log("  PC1 ja disponivel (modo pre-calculado) — exportando diretamente")
                pc1_all = pca_components[:, 0] if pca_components.shape[1] > 0 else pc1_all
            elif pca is not None and emb is not None and scaler is not None:
                chunk = 100_000
                for i in range(0, N, chunk):
                    chunk_scaled = scaler.transform(emb[i:i+chunk])
                    pc1_all[i:i+chunk] = pca.transform(chunk_scaled)[:, 0]
            if IS_COLAB and "dem" in dir():
                _dem = dem
            else:
                _dem = None
            _pos_npz = DATA_DIR / "pos.npz" if not IS_COLAB else None
            _export_pc1_tif(pc1_all, _dem, raster_meta, OUT_DIR / "14_d18_embeddings_pc1.tif",
                            pos_npz=_pos_npz)
            log("  PC1 GeoTIFF exportado.")

        if IS_COLAB and "data" in dir():
            del data
        del feat
        gc.collect()
    else:
        log('  Graph not available — PCA-feature correlations skipped')

    results["meta"] = {
        "timestamp": datetime.now().isoformat(),
        "n_nos":    N,
        "n_dims":   D,
        "n_sample_pca": n_sample,
        "elapsed_s": round(time.perf_counter() - start, 1),
    }

    out_json = OUT_DIR / "14_embeddings_results.json"
    with open(str(out_json), "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.perf_counter() - start
    print("\n" + "=" * 60)
    print("SCRIPT 14 — CONCLUIDO")
    print(f"Tempo total: {elapsed:.0f}s")
    print(f"\nRESUMO D18:")
    print(f"  Dims para 90% variancia: {d18['dims_para_90pct_variancia']} / {D}")
    print(f"  Interpretacao: {d18['interpretacao']}")
    print(f"\nResultados em: {OUT_DIR}")
    for p in sorted(OUT_DIR.iterdir()):
        print(f"  {p.name}")
    print("=" * 60)


def _export_pc1_tif(pc1, dem, raster_meta, path, pos_npz=None):
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

    if dem is not None and hasattr(dem, "pos"):
        pos = dem.pos.numpy()
        lons, lats = pos[:, 0], pos[:, 1]
    elif pos_npz is not None and Path(str(pos_npz)).exists():
        pos = np.load(str(pos_npz))["pos"]
        lons, lats = pos[:, 0], pos[:, 1]
    else:
        if len(pc1) == H * W:
            arr = pc1.reshape(H, W)
            with rasterio.open(str(path), "w", driver="GTiff", height=H, width=W,
                               count=1, dtype="float32", crs=crs, transform=transform,
                               nodata=-9999.0, compress="deflate") as dst:
                dst.write(arr, 1)
                dst.update_tags(CONTENT="PC1_GNN_TOPO_embeddings_reshape")
        return

    rows_px = np.clip(((transform.f - lats) / abs(transform.e)).astype(int), 0, H - 1)
    cols_px = np.clip(((lons - transform.c) / abs(transform.a)).astype(int), 0, W - 1)

    arr = np.full((H, W), fill_value=-9999.0, dtype=np.float32)
    arr[rows_px, cols_px] = pc1

    with rasterio.open(str(path), "w", driver="GTiff", height=H, width=W,
                       count=1, dtype="float32", crs=crs, transform=transform,
                       nodata=-9999.0, compress="deflate") as dst:
        dst.write(arr, 1)
        dst.update_tags(CONTENT="PC1_GNN_TOPO_embeddings")


if __name__ == "__main__":
    main()
