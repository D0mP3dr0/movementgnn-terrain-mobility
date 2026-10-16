import sys
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS

BASE     = Path(__file__).resolve().parent
RESULTS  = BASE / "results" / "results_v2_local"
CKPT     = RESULTS / "checkpoints" / "checkpoint_best.pt"
PREDS    = RESULTS / "predictions_full.npz"
OUT_DIR  = RESULTS

FRACTIONS = ["a_pe", "motorizada", "mecanizada", "blindada"]

CMAP = {
    1: (0,   200, 0),
    2: (255, 200, 0),
    3: (200, 0,   0),
}


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_raster_meta(ckpt_path):
    log(f"Carregando checkpoint: {ckpt_path.name} ...")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    rm = ckpt.get("raster_meta")
    if rm is None:
        raise RuntimeError('raster_meta not found in the checkpoint.')
    return rm


def build_transform(rm):
    t = rm["dem_transform"]
    if isinstance(t, (list, tuple)):
        tf = Affine(*t[:6])
    elif isinstance(t, Affine):
        tf = t
    else:
        tf = Affine(t.a, t.b, t.c, t.d, t.e, t.f)
    return tf


def pixel_stats(arr, fracs_label):
    total = arr.size
    for v in [1, 2, 3]:
        cnt = int((arr == v).sum())
        pct = 100.0 * cnt / total if total > 0 else 0.0
        print(f"    classe {v}: {cnt:>10,}  ({pct:5.1f}%)")
    zeros = int((arr == 0).sum())
    print(f"    nodata  : {zeros:>10,}  ({100.0*zeros/total:5.1f}%)")


def main():
    start = time.perf_counter()

    rm        = load_raster_meta(CKPT)
    H, W      = rm["dem_shape"]
    crs       = CRS.from_user_input(rm["crs"])
    transform = build_transform(rm)

    log(f"Raster: {H}x{W} px | CRS={rm['crs']}")
    log(f"Transform: a={transform.a:.10f}  e={transform.e:.10f}")
    log(f"Bounds: {rm.get('bounds')}")

    expected_n = H * W
    log(f"N nodes esperado (H*W): {expected_n:,}")

    log(f"Carregando predicoes: {PREDS.name} ...")
    preds_data = np.load(str(PREDS))
    preds = {}
    for frac in FRACTIONS:
        arr_flat = preds_data[frac].astype(np.int64)
        if arr_flat.shape[0] != expected_n:
            raise RuntimeError(
                f"Incompatible shape for '{frac}': expected {expected_n}, found {arr_flat.shape[0]}"
            )
        preds[frac] = arr_flat
        log(f"  {frac}: shape={arr_flat.shape}, "
            f"classes={np.unique(arr_flat).tolist()}")

    log("Exportando GeoTIFFs ...")
    export_log = {}

    for frac in FRACTIONS:
        grid = preds[frac].reshape(H, W).astype(np.uint8)

        fpath = OUT_DIR / f"restricao_{frac}_movement_gnn_v2.tif"
        with rasterio.open(
            fpath, "w",
            driver   = "GTiff",
            height   = H,
            width    = W,
            count    = 1,
            dtype    = "uint8",
            crs      = crs,
            transform= transform,
            nodata   = 0,
            compress = "deflate",
        ) as dst:
            dst.write(grid, 1)
            dst.set_band_description(1, f"restricao_{frac}")
            dst.write_colormap(1, CMAP)
            dst.update_tags(
                MODEL="MovementGNN_v2_local",
                CLASSES="1=Go,2=SlowGo,3=NoGo",
                FRACTION=frac,
                FIX="reshape_row_major",
            )

        log(f"  {frac}: {fpath.name}")
        pixel_stats(grid, frac)
        export_log[frac] = {
            "path": str(fpath),
            "shape": f"{H}x{W}",
            "crs": rm["crs"],
            "transform": [transform.a, transform.b, transform.c,
                          transform.d, transform.e, transform.f],
            "class_counts": {
                str(v): int((grid == v).sum()) for v in [1, 2, 3]
            },
            "nodata_count": int((grid == 0).sum()),
        }

    bands = np.stack(
        [preds[f].reshape(H, W).astype(np.uint8) for f in FRACTIONS],
        axis=0
    )

    mpath = OUT_DIR / "restricao_multiband_movement_gnn_v2.tif"
    with rasterio.open(
        mpath, "w",
        driver   = "GTiff",
        height   = H,
        width    = W,
        count    = 4,
        dtype    = "uint8",
        crs      = crs,
        transform= transform,
        nodata   = 0,
        compress = "deflate",
    ) as dst:
        for i, frac in enumerate(FRACTIONS):
            dst.write(bands[i], i + 1)
            dst.set_band_description(i + 1, f"restricao_{frac}")
        dst.update_tags(
            MODEL="MovementGNN_v2_local",
            CLASSES="1=Go,2=SlowGo,3=NoGo",
            VERSION="v2_local",
            FIX="reshape_row_major",
        )

    log(f"  multiband: {mpath.name} ({bands.shape})")
    export_log["multiband"] = {
        "path": str(mpath),
        "bands": FRACTIONS,
        "shape": f"4x{H}x{W}",
    }

    log_path = OUT_DIR / "geotiff_export_log.json"
    export_log["_meta"] = {
        "timestamp": datetime.now().isoformat(),
        "script": "re_export_geotiff.py",
        "bug_fix": "pos lat/lon swap + reshape row-major",
        "elapsed_s": round(time.perf_counter() - start, 1),
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(export_log, f, indent=2, ensure_ascii=False)
    log(f"Log salvo: {log_path.name}")

    elapsed = time.perf_counter() - start
    log(f"Concluido em {elapsed:.1f}s")


if __name__ == "__main__":
    main()
