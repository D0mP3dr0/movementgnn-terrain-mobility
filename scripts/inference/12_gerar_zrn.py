"""
12_gerar_zrn.py — Post-processing: Necessary Reconnaissance Zone (ZRN)
=======================================================================
Reads probs_full.npz (3 classes x 4 fractions) and applies entropy-based
logic to generate the 4th class (ZRN) for high-uncertainty pixels.

Class scheme (output):
    1 = Adequate (Go)
    2 = Restricted (SlowGo)
    3 = Necessary Reconnaissance Zone (ZRN)
    4 = Impeditive (NoGo)

Outputs:
    - predictions_4class.npz  (4-class predictions)
    - zrn_stats.json          (per-fraction statistics)
    - entropy_maps.npz        (per-fraction entropy maps)
    - GeoTIFF 4-class per fraction + multiband
    - GeoTIFF continuous entropy per fraction

Dependencies: numpy, rasterio, torch (graph metadata only)
"""

import gc
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# --- CONFIGURATION ---
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from src.paths import default_graph_path, get_inference_paths

_paths = get_inference_paths("v2_local")
PROBS_PATH = _paths["out_dir"] / "probs_full.npz"
GRAPH_PATH = default_graph_path()
OUT_DIR    = _paths["out_dir"]

ENTROPY_THRESHOLD = 0.5
FRACTIONS = ["a_pe", "motorizada", "mecanizada", "blindada"]

COLORMAP_4CLASS = {
    0: (0, 0, 0, 0),
    1: (0, 200, 0, 255),
    2: (255, 200, 0, 255),
    3: (255, 140, 0, 255),
    4: (200, 0, 0, 255),
}


def log(msg, t0):
    ts = datetime.now().strftime("%H:%M:%S")
    elapsed = time.perf_counter() - t0
    print(f"[{ts}] (+{elapsed:5.0f}s) {msg}", flush=True)


def compute_entropy(probs):
    """Normalized Shannon entropy (0-1) for a 3-class distribution."""
    eps = 1e-10
    p = np.clip(probs, eps, 1.0)
    h = -np.sum(p * np.log2(p), axis=1)
    max_entropy = np.log2(p.shape[1])
    return h / max_entropy


def apply_zrn(preds_3class, entropy, threshold):
    """
    Convert 3-class predictions to 4-class with ZRN.

    Mapping:
        original Go(1)     + low entropy  -> 1 (Adequate)
        original SlowGo(2) + low entropy  -> 2 (Restricted)
        any class          + high entropy -> 3 (ZRN)
        original NoGo(3)   + low entropy  -> 4 (Impeditive)
    """
    result = np.zeros_like(preds_3class)
    zrn_mask = entropy >= threshold

    result[~zrn_mask & (preds_3class == 1)] = 1
    result[~zrn_mask & (preds_3class == 2)] = 2
    result[zrn_mask] = 3
    result[~zrn_mask & (preds_3class == 3)] = 4

    return result


def export_geotiff_4class(preds_dict, entropy_dict, raster_meta, pos_np, out_dir):
    """Export GeoTIFFs with 4 classes and entropy maps."""
    try:
        import rasterio
        from rasterio.transform import Affine
        from rasterio.crs import CRS
    except ImportError:
        print("rasterio not available — GeoTIFF not exported")
        return

    if raster_meta is None or pos_np is None:
        print("raster_meta or pos missing — GeoTIFF not exported")
        return

    H, W = raster_meta["dem_shape"]
    crs = CRS.from_user_input(raster_meta["crs"])
    t = raster_meta["dem_transform"]
    if isinstance(t, (list, tuple)):
        transform = Affine(*t[:6])
    else:
        transform = Affine(t.a, t.b, t.c, t.d, t.e, t.f)

    lats, lons = pos_np[:, 0], pos_np[:, 1]
    rows_px = np.clip(((transform.f - lats) / abs(transform.e)).astype(int), 0, H - 1)
    cols_px = np.clip(((lons - transform.c) / abs(transform.a)).astype(int), 0, W - 1)

    out_dir = Path(out_dir)

    for frac, vals in preds_dict.items():
        arr = np.zeros((H, W), dtype=np.uint8)
        arr[rows_px, cols_px] = vals.astype(np.uint8)
        fpath = out_dir / f"restricao_4class_{frac}.tif"
        with rasterio.open(fpath, "w", driver="GTiff", height=H, width=W,
                           count=1, dtype="uint8", crs=crs, transform=transform,
                           nodata=0, compress="deflate") as dst:
            dst.write(arr, 1)
            dst.set_band_description(1, f"restricao_4class_{frac}")
            dst.write_colormap(1, COLORMAP_4CLASS)
            dst.update_tags(
                MODEL="movement_gnn_v2_zrn",
                CLASSES="1=Adequate,2=Restricted,3=ZRN,4=Impeditive",
                FRACTION=frac,
                ENTROPY_THRESHOLD=str(ENTROPY_THRESHOLD),
            )
        print(f"  GeoTIFF 4-class: {fpath.name}")

    bands = np.zeros((4, H, W), dtype=np.uint8)
    for i, frac in enumerate(FRACTIONS):
        bands[i, rows_px, cols_px] = preds_dict[frac].astype(np.uint8)
    mpath = out_dir / "restricao_4class_multiband.tif"
    with rasterio.open(mpath, "w", driver="GTiff", height=H, width=W,
                       count=4, dtype="uint8", crs=crs, transform=transform,
                       nodata=0, compress="deflate") as dst:
        for i in range(4):
            dst.write(bands[i], i + 1)
            dst.set_band_description(i + 1, f"restricao_4class_{FRACTIONS[i]}")
        dst.update_tags(
            MODEL="movement_gnn_v2_zrn",
            CLASSES="1=Adequate,2=Restricted,3=ZRN,4=Impeditive",
            VERSION="v2_zrn",
        )
    print(f"  GeoTIFF 4-class multiband: {mpath.name}")

    for frac, ent in entropy_dict.items():
        arr_f = np.full((H, W), np.nan, dtype=np.float32)
        arr_f[rows_px, cols_px] = ent.astype(np.float32)
        epath = out_dir / f"entropy_{frac}.tif"
        with rasterio.open(epath, "w", driver="GTiff", height=H, width=W,
                           count=1, dtype="float32", crs=crs, transform=transform,
                           nodata=np.nan, compress="deflate") as dst:
            dst.write(arr_f, 1)
            dst.set_band_description(1, f"entropy_{frac}")
            dst.update_tags(MODEL="movement_gnn_v2", METRIC="normalized_shannon_entropy")
        print(f"  GeoTIFF entropy: {epath.name}")


def main():
    t0 = time.perf_counter()

    print("=" * 70)
    print("  12_GERAR_ZRN — Necessary Reconnaissance Zone (post-processing)")
    print("=" * 70)

    assert PROBS_PATH.exists(), f"probs_full.npz not found: {PROBS_PATH}"
    assert GRAPH_PATH.exists(), f"Graph not found: {GRAPH_PATH}"

    # --- 1. LOAD PROBABILITIES ---
    log(f"[1/5] Loading {PROBS_PATH.name} ...", t0)
    probs_data = np.load(str(PROBS_PATH))

    probs = {}
    for frac in FRACTIONS:
        key = f"probs_{frac}"
        if key in probs_data:
            probs[frac] = probs_data[key]
        elif frac in probs_data:
            probs[frac] = probs_data[frac]
        else:
            raise KeyError(f"Key {key} not found in probs_full.npz. "
                           f"Available keys: {list(probs_data.keys())}")

    n_nodes = probs[FRACTIONS[0]].shape[0]
    log(f"  {n_nodes:,} nodes | {len(FRACTIONS)} fractions | 3 classes", t0)
    for frac in FRACTIONS:
        s = probs[frac][:5].sum(axis=1)
        log(f"  {frac}: shape={probs[frac].shape} | sum_check={s.round(3).tolist()}", t0)

    # --- 2. COMPUTE ENTROPY ---
    log("[2/5] Computing normalized entropy ...", t0)
    entropy = {}
    for frac in FRACTIONS:
        entropy[frac] = compute_entropy(probs[frac])
        h = entropy[frac]
        log(f"  {frac}: mean={h.mean():.4f} | median={np.median(h):.4f} | "
            f"p95={np.percentile(h, 95):.4f} | max={h.max():.4f} | "
            f">{ENTROPY_THRESHOLD}: {(h >= ENTROPY_THRESHOLD).sum():,} "
            f"({(h >= ENTROPY_THRESHOLD).mean() * 100:.2f}%)", t0)

    # --- 3. APPLY ZRN (4th CLASS) ---
    log(f"[3/5] Applying ZRN (threshold={ENTROPY_THRESHOLD}) ...", t0)
    preds_3class = {}
    preds_4class = {}
    stats = {}

    for frac in FRACTIONS:
        p3 = probs[frac].argmax(axis=1) + 1
        preds_3class[frac] = p3

        p4 = apply_zrn(p3, entropy[frac], ENTROPY_THRESHOLD)
        preds_4class[frac] = p4

        dist = {
            "Adequado_Go": int((p4 == 1).sum()),
            "Restrito_SlowGo": int((p4 == 2).sum()),
            "ZRN": int((p4 == 3).sum()),
            "Impeditivo_NoGo": int((p4 == 4).sum()),
        }
        total = sum(dist.values())
        pct = {k: round(v / total * 100, 2) for k, v in dist.items()}
        stats[frac] = {"counts": dist, "percentages": pct, "total": total}

        from_3class = {
            "Go_to_ZRN": int(((p3 == 1) & (p4 == 3)).sum()),
            "SlowGo_to_ZRN": int(((p3 == 2) & (p4 == 3)).sum()),
            "NoGo_to_ZRN": int(((p3 == 3) & (p4 == 3)).sum()),
        }
        stats[frac]["reclassified_to_ZRN"] = from_3class

        log(f"  {frac}: Go={dist['Adequado_Go']:,} ({pct['Adequado_Go']}%) | "
            f"SlowGo={dist['Restrito_SlowGo']:,} ({pct['Restrito_SlowGo']}%) | "
            f"ZRN={dist['ZRN']:,} ({pct['ZRN']}%) | "
            f"NoGo={dist['Impeditivo_NoGo']:,} ({pct['Impeditivo_NoGo']}%)", t0)

    # --- 4. SAVE RESULTS ---
    log("[4/5] Saving results ...", t0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pred_path = OUT_DIR / "predictions_4class.npz"
    np.savez_compressed(str(pred_path), **{f: preds_4class[f] for f in FRACTIONS})
    log(f"  {pred_path.name}: {pred_path.stat().st_size / 1e6:.1f} MB", t0)

    ent_path = OUT_DIR / "entropy_maps.npz"
    np.savez_compressed(str(ent_path), **{f"entropy_{f}": entropy[f] for f in FRACTIONS})
    log(f"  {ent_path.name}: {ent_path.stat().st_size / 1e6:.1f} MB", t0)

    stats_global = {
        "timestamp": datetime.now().isoformat(),
        "entropy_threshold": ENTROPY_THRESHOLD,
        "n_nodes": n_nodes,
        "fractions": stats,
    }
    stats_path = OUT_DIR / "zrn_stats.json"
    with open(str(stats_path), "w") as f:
        json.dump(stats_global, f, indent=2)
    log(f"  {stats_path.name}", t0)

    # --- 5. GEOTIFF 4 CLASSES + ENTROPY ---
    log("[5/5] Exporting GeoTIFFs ...", t0)

    import torch
    data = torch.load(str(GRAPH_PATH), map_location="cpu", weights_only=False)
    dem_store = data["dem"] if hasattr(data, "node_types") else data
    raster_meta = getattr(data, "raster_meta", None)
    pos = getattr(dem_store, "pos", None)

    if raster_meta is not None and pos is not None:
        pos_np = pos.numpy() if torch.is_tensor(pos) else pos
        export_geotiff_4class(preds_4class, entropy, raster_meta, pos_np, OUT_DIR)
    else:
        log("  WARNING: raster_meta or pos missing — GeoTIFFs not exported", t0)

    del data
    gc.collect()

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 70)
    print("  DONE — ZRN generated")
    print(f"  Entropy threshold: {ENTROPY_THRESHOLD}")
    for frac in FRACTIONS:
        pct_zrn = stats[frac]["percentages"]["ZRN"]
        print(f"  {frac}: {pct_zrn}% ZRN ({stats[frac]['counts']['ZRN']:,} nodes)")
    print(f"  Time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"  Output: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
