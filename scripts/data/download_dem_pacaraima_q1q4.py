"""
Download Copernicus DEM GLO-30 for the Pacaraima study area (Q1-Q4 bbox).

Outputs:
- DEM tiles in data_raw/satelites_raw/dem/tiles/pacaraima_q1q4/
- Consolidated mosaic in data_raw/satelites_raw/dem/pacaraima_dem_consolidated_q1q4.tif
- JSON report in data_raw/satelites_raw/dem/pacaraima_dem_download_report_q1q4.json
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import requests
import rasterio
from rasterio.merge import merge

import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

COPERNICUS_DEM_URL = "https://copernicus-dem-30m.s3.amazonaws.com"
FULL_BBOX = [-61.6138630116465, 3.9760673588888893, -60.682286988353496, 4.97551041888889]

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from src.paths import RAW_DEM

ROOT = RAW_DEM
TILES_DIR = ROOT / "tiles" / "pacaraima_q1q4"
MOSAIC_PATH = ROOT / "pacaraima_dem_consolidated_q1q4.tif"
REPORT_PATH = ROOT / "pacaraima_dem_download_report_q1q4.json"


def get_dem_tile_name(lat: int, lon: int) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"


def get_required_tiles(bbox: List[float]) -> List[Tuple[int, int]]:
    min_lon, min_lat, max_lon, max_lat = bbox
    start_lat = math.floor(min_lat)
    end_lat = math.ceil(max_lat)
    start_lon = math.floor(min_lon)
    end_lon = math.ceil(max_lon)
    out = []
    for lat in range(start_lat, end_lat):
        for lon in range(start_lon, end_lon):
            out.append((lat, lon))
    return out


def download_tile(lat: int, lon: int) -> dict:
    TILES_DIR.mkdir(parents=True, exist_ok=True)
    tile_name = get_dem_tile_name(lat, lon)
    output_path = TILES_DIR / f"{tile_name}.tif"

    if output_path.exists() and output_path.stat().st_size > 1_000:
        return {"tile": tile_name, "status": "exists", "path": str(output_path), "size_bytes": output_path.stat().st_size}

    url = f"{COPERNICUS_DEM_URL}/{tile_name}/{tile_name}.tif"
    try:
        log.info(f"DEM -> {tile_name}")
        r = requests.get(url, stream=True, timeout=180)
        if r.status_code == 404:
            return {"tile": tile_name, "status": "not_found", "path": None}
        r.raise_for_status()
        total = 0
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 64):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)
        return {"tile": tile_name, "status": "downloaded", "path": str(output_path), "size_bytes": total}
    except Exception as exc:
        return {"tile": tile_name, "status": "error", "error": str(exc)}


def build_mosaic_if_possible() -> dict:
    tif_files = sorted(TILES_DIR.glob("*.tif"))
    if not tif_files:
        return {"mosaic_status": "skipped", "reason": "no_tiles"}

    srcs = []
    try:
        for p in tif_files:
            srcs.append(rasterio.open(p))
        mosaic_arr, mosaic_transform = merge(srcs, bounds=FULL_BBOX)
        meta = srcs[0].meta.copy()
        meta.update(
            driver="GTiff",
            height=mosaic_arr.shape[1],
            width=mosaic_arr.shape[2],
            transform=mosaic_transform,
            compress="lzw",
            tiled=True,
            blockxsize=512,
            blockysize=512,
            bigtiff="YES",
        )
        MOSAIC_PATH.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(MOSAIC_PATH, "w", **meta) as dst:
            dst.write(mosaic_arr)
        return {"mosaic_status": "ok", "mosaic_path": str(MOSAIC_PATH), "size_gb": round(MOSAIC_PATH.stat().st_size / 1e9, 3)}
    finally:
        for s in srcs:
            try:
                s.close()
            except Exception:
                pass


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    start = datetime.now()

    tiles = get_required_tiles(FULL_BBOX)
    log.info(f"Pacaraima Q1-Q4 DEM bbox: {FULL_BBOX}")
    log.info(f"Expected tiles: {len(tiles)}")

    results = [download_tile(lat, lon) for lat, lon in tiles]
    stats = {
        "total": len(results),
        "downloaded": sum(1 for x in results if x["status"] == "downloaded"),
        "exists": sum(1 for x in results if x["status"] == "exists"),
        "not_found": sum(1 for x in results if x["status"] == "not_found"),
        "errors": sum(1 for x in results if x["status"] == "error"),
        "size_mb": round(sum(x.get("size_bytes", 0) for x in results) / 1e6, 2),
    }

    mosaic_info = build_mosaic_if_possible()

    report = {
        "pipeline": "download_dem_pacaraima_q1q4",
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": (datetime.now() - start).total_seconds(),
        "bbox": FULL_BBOX,
        "tiles": results,
        "stats": stats,
        "mosaic": mosaic_info,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log.info(f"Report: {REPORT_PATH}")
    log.info(f"Stats: {stats}")
    if mosaic_info.get("mosaic_status") == "ok":
        log.info(f"Mosaic: {mosaic_info['mosaic_path']}")


if __name__ == "__main__":
    main()
