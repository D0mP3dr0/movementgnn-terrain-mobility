"""
Download Sentinel-2 L2A 10m for Pacaraima (Q1-Q4) using a quadrant strategy.

Pipeline:
- STAC search (Element84 Earth Search)
- Split into 4 quadrants (~1x1 degree each)
- Final 2x2 degree mosaic
- Output: GeoTIFF + JSON metadata
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject as rio_reproject


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# COG/GDAL optimized for HTTP reads with explicit timeouts
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
os.environ.setdefault("GDAL_HTTP_VERSION", "2")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("VSI_CACHE_SIZE", "1000000000")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "90")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "5")
os.environ.setdefault("CPL_CURL_GZIP", "YES")

# Python-level timeout per band download (seconds)
BAND_DOWNLOAD_TIMEOUT = 120


PACARAIMA = {
    "name": "Pacaraima RR",
    "lon": -61.148075,
    "lat": 4.475788888888889,
}

# Study area bounding box equivalent to Q1-Q4
FULL_BBOX = [
    -61.6138630116465,
    3.9760673588888893,
    -60.682286988353496,
    4.97551041888889,
]

RESOLUTION_DEG = 10.0 / 111_320.0
BANDS = ["B02", "B03", "B04", "B08"]
BAND_ASSET_MAP = {"B02": "blue", "B03": "green", "B04": "red", "B08": "nir"}

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from src.paths import RAW_SENTINEL2

OUT_DIR = RAW_SENTINEL2
OUT_TIF = OUT_DIR / "pacaraima_s2_10m_real_q1q4.tif"
OUT_JSON = OUT_DIR / "pacaraima_s2_10m_real_q1q4.json"
CHECKPOINT_DIR = OUT_DIR / "_checkpoint_q1q4"
PROGRESS_JSON = OUT_DIR / "pacaraima_s2_q1q4_progress.json"


def search_scenes(bbox: list, max_cloud: float, year: int, year_start: int, label: str) -> list:
    """Search STAC catalog for scenes in a quadrant."""
    try:
        import pystac_client
    except ImportError:
        log.error("pystac-client not installed. Run: pip install pystac-client")
        raise

    date_start = f"{year_start}-01-01T00:00:00Z"
    date_end = f"{year}-12-31T23:59:59Z"

    log.info(
        f"[{label}] STAC search bbox={[round(x, 6) for x in bbox]} "
        f"cloud<={max_cloud}% period={year_start}-{year}"
    )

    catalog = pystac_client.Client.open(STAC_URL)
    search = catalog.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=f"{date_start}/{date_end}",
        query={"eo:cloud_cover": {"lte": max_cloud}},
        max_items=1000,
        sortby=["+properties.eo:cloud_cover"],
    )

    items = list(search.items())
    log.info(f"[{label}] Scenes found: {len(items)}")
    return items


def build_target_grid(bbox: list, res_deg: float):
    lon_min, lat_min, lon_max, lat_max = bbox
    width = int(round((lon_max - lon_min) / res_deg))
    height = int(round((lat_max - lat_min) / res_deg))
    tf = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
    crs = CRS.from_epsg(4326)
    return tf, width, height, crs


def _download_band_cog_inner(href, band_name, dst_transform, dst_w, dst_h, dst_crs) -> np.ndarray | None:
    """Download and reproject a single band from a COG asset."""
    with rasterio.open(href) as src:
        tmp = np.zeros((dst_h, dst_w), dtype=np.float32)
        rio_reproject(
            source=rasterio.band(src, 1),
            destination=tmp,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            num_threads=4,
        )
    return tmp


def download_band_cog(item, band_name, dst_transform, dst_w, dst_h, dst_crs, acc, cnt, retry=3) -> bool:
    """Download/reproject one band from a COG scene with Python-level timeout."""
    asset_key = BAND_ASSET_MAP.get(band_name, band_name)
    asset = item.assets.get(asset_key)
    if asset is None:
        return False

    href = asset.href
    for attempt in range(retry):
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    _download_band_cog_inner,
                    href, band_name, dst_transform, dst_w, dst_h, dst_crs,
                )
                tmp = future.result(timeout=BAND_DOWNLOAD_TIMEOUT)

            if tmp is not None:
                valid = tmp > 0
                acc[valid] += tmp[valid]
                cnt[valid] += 1
                return True
        except FuturesTimeoutError:
            log.warning(
                f"TIMEOUT ({BAND_DOWNLOAD_TIMEOUT}s) attempt {attempt+1}/{retry} "
                f"for {band_name}: {href[:80]}..."
            )
            time.sleep(5 * (attempt + 1))
        except Exception as exc:
            log.warning(f"Failure attempt {attempt+1}/{retry} for {band_name}: {type(exc).__name__}: {exc}")
            time.sleep(4 * (attempt + 1))
    return False


def mosaic_quadrant(
    items: list,
    bbox: list,
    max_scenes: int,
    label: str,
    coverage_target: float = 99.0,
    coverage_min_accept: float = 60.0,
) -> dict:
    """
    Mosaic a single quadrant (~1x1 degree).

    Stops accumulating scenes once coverage_target is reached.
    Accepts results below coverage_target if coverage >= coverage_min_accept
    (avoids infinite loop in structurally limited quadrants, e.g. border
    areas with persistent cloud cover).
    """
    dst_transform, dst_w, dst_h, dst_crs = build_target_grid(bbox, RESOLUTION_DEG)
    selected = items[:max_scenes]

    qdata = {}
    for band in BANDS:
        log.info(f"[{label}] Band {band} with up to {len(selected)} scenes (target={coverage_target:.0f}%)...")
        acc = np.zeros((dst_h, dst_w), dtype=np.float32)
        cnt = np.zeros((dst_h, dst_w), dtype=np.float32)

        ok = 0
        coverage = 0.0
        for i, item in enumerate(selected, start=1):
            cloud = item.properties.get("eo:cloud_cover", 999)
            date = str(item.properties.get("datetime", "?"))[:10]
            log.info(f"[{label}] {band} scene {i}/{len(selected)} date={date} cloud={cloud:.1f}% coverage={coverage:.1f}%")
            if download_band_cog(item, band, dst_transform, dst_w, dst_h, dst_crs, acc, cnt):
                ok += 1
                coverage = float((cnt > 0).mean() * 100)
                if coverage >= coverage_target:
                    log.info(f"[{label}] {band} target {coverage_target:.0f}% reached ({coverage:.1f}%) in {i} scenes.")
                    break

        if ok == 0:
            log.error(f"[{label}] No valid scenes for band {band}")
            return {}

        if coverage < coverage_min_accept:
            log.warning(
                f"[{label}] {band} final coverage {coverage:.1f}% below minimum "
                f"({coverage_min_accept:.0f}%). Check scene availability in the region."
            )
        else:
            log.info(f"[{label}] {band} finished with {ok} valid scenes, coverage={coverage:.1f}%")

        arr = np.clip(acc / np.maximum(cnt, 1.0) / 10000.0, 0.0, 1.0).astype(np.float32)
        qdata[band] = arr
        del acc, cnt
        gc.collect()

    qdata["__bbox"] = bbox
    return qdata


def combine_quadrants(quadrants: dict, full_bbox: list) -> dict:
    """Combine SW/SE/NW/NE quadrants into the full 2x2 degree grid."""
    lon_min, lat_min, lon_max, lat_max = full_bbox
    full_w = int(round((lon_max - lon_min) / RESOLUTION_DEG))
    full_h = int(round((lat_max - lat_min) / RESOLUTION_DEG))

    result = {}
    for band in BANDS:
        full = np.zeros((full_h, full_w), dtype=np.float32)

        for qname, qdata in quadrants.items():
            if not qdata or band not in qdata:
                continue
            ql, qb, qr, qt = qdata["__bbox"]
            col0 = int(round((ql - lon_min) / RESOLUTION_DEG))
            row0 = int(round((lat_max - qt) / RESOLUTION_DEG))
            col1 = int(round((qr - lon_min) / RESOLUTION_DEG))
            row1 = int(round((lat_max - qb) / RESOLUTION_DEG))
            full[row0:row1, col0:col1] = qdata[band]

        result[band] = full

    result["__transform"] = from_bounds(lon_min, lat_min, lon_max, lat_max, full_w, full_h)
    result["__width"] = full_w
    result["__height"] = full_h
    result["__crs"] = CRS.from_epsg(4326)
    return result


def save_geotiff(mosaic: dict, out_path: Path):
    """Save final 4-band GeoTIFF."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "driver": "GTiff",
        "dtype": "uint16",
        "nodata": 0,
        "width": mosaic["__width"],
        "height": mosaic["__height"],
        "count": len(BANDS),
        "crs": mosaic["__crs"],
        "transform": mosaic["__transform"],
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "bigtiff": "YES",
    }
    with rasterio.open(out_path, "w", **meta) as dst:
        for i, band in enumerate(BANDS, start=1):
            arr_u16 = (mosaic[band] * 10000).astype(np.uint16)
            dst.write(arr_u16, i)
            dst.update_tags(i, name=band)
        dst.update_tags(
            version="PACARAIMA-Q1Q4",
            source=STAC_URL,
            bands=",".join(BANDS),
            generated=datetime.now().isoformat(),
        )


def save_quadrant_checkpoint(qname: str, qdata: dict):
    """
    Save quadrant checkpoint for resumption.
    One file per quadrant avoids rework on late failures.
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    qpath = CHECKPOINT_DIR / f"{qname}.tif"

    bbox = qdata["__bbox"]
    transform, width, height, crs = build_target_grid(bbox, RESOLUTION_DEG)
    meta = {
        "driver": "GTiff",
        "dtype": "uint16",
        "nodata": 0,
        "width": width,
        "height": height,
        "count": len(BANDS),
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "bigtiff": "YES",
    }
    with rasterio.open(qpath, "w", **meta) as dst:
        for i, band in enumerate(BANDS, start=1):
            dst.write((qdata[band] * 10000).astype(np.uint16), i)
            dst.update_tags(i, name=band)
        dst.update_tags(quadrant=qname, bbox=json.dumps(bbox), generated=datetime.now().isoformat())


def load_quadrant_checkpoint(qname: str) -> dict:
    """Load a previously saved quadrant checkpoint. Returns {} if not found."""
    qpath = CHECKPOINT_DIR / f"{qname}.tif"
    if not qpath.exists():
        return {}
    out = {}
    with rasterio.open(qpath) as src:
        for i, band in enumerate(BANDS, start=1):
            out[band] = (src.read(i).astype(np.float32) / 10000.0)
        try:
            bbox = json.loads(src.tags().get("bbox", "[]"))
        except Exception:
            bbox = []
    if len(bbox) != 4:
        return {}
    out["__bbox"] = bbox
    return out


def load_progress() -> dict:
    if not PROGRESS_JSON.exists():
        return {"quadrants_done": [], "updated_at": None}
    try:
        with open(PROGRESS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"quadrants_done": [], "updated_at": None}


def save_progress(progress: dict):
    progress["updated_at"] = datetime.now().isoformat()
    with open(PROGRESS_JSON, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def main():
    global BAND_DOWNLOAD_TIMEOUT  # noqa: PLW0603
    parser = argparse.ArgumentParser(description="Download Sentinel-2 Pacaraima Q1-Q4 (2x2 degrees).")
    parser.add_argument("--max-cloud", type=float, default=20.0)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--year-start", type=int, default=2023)
    parser.add_argument("--max-scenes", type=int, default=20)
    parser.add_argument("--coverage-target", type=float, default=99.0,
                        help="Target coverage %% per quadrant to stop accumulating scenes (default: 99).")
    parser.add_argument("--coverage-min", type=float, default=60.0,
                        help="Minimum acceptable coverage %%; emits warning if not reached (default: 60).")
    parser.add_argument("--band-timeout", type=int, default=BAND_DOWNLOAD_TIMEOUT,
                        help=f"Timeout in seconds per band download (default: {BAND_DOWNLOAD_TIMEOUT}).")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    BAND_DOWNLOAD_TIMEOUT = args.band_timeout

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lon = PACARAIMA["lon"]
    lat = PACARAIMA["lat"]

    # Quadrants must derive from FULL_BBOX to avoid shape mismatch in combine_quadrants
    lon_min, lat_min, lon_max, lat_max = FULL_BBOX
    lon_mid = (lon_min + lon_max) / 2.0
    lat_mid = (lat_min + lat_max) / 2.0
    quadrant_bboxes = {
        "SW": [lon_min, lat_min, lon_mid, lat_mid],
        "SE": [lon_mid, lat_min, lon_max, lat_mid],
        "NW": [lon_min, lat_mid, lon_mid, lat_max],
        "NE": [lon_mid, lat_mid, lon_max, lat_max],
    }

    log.info("=" * 72)
    log.info("PACARAIMA S2 DOWNLOAD — GNN_RF STRUCTURE (Q1-Q4)")
    log.info(f"Center: lon={lon:.6f}, lat={lat:.6f}")
    log.info(f"Target bbox equivalent: {FULL_BBOX}")
    log.info("=" * 72)

    if OUT_TIF.exists() and not args.force and not args.dry_run:
        log.info(f"File already exists: {OUT_TIF}")
        return

    quadrants_data = {}
    scenes_summary = {}
    progress = load_progress()
    done_quadrants = set(progress.get("quadrants_done", []))

    for qname, qbbox in quadrant_bboxes.items():
        if (qname in done_quadrants) and (not args.force):
            cached = load_quadrant_checkpoint(qname)
            if cached:
                log.info(f"[{qname}] Checkpoint found. Reusing quadrant.")
                quadrants_data[qname] = cached
                scenes_summary[qname] = progress.get("scenes_found_per_quadrant", {}).get(qname, -1)
                continue

        items = search_scenes(
            bbox=qbbox,
            max_cloud=args.max_cloud,
            year=args.year,
            year_start=args.year_start,
            label=qname,
        )
        scenes_summary[qname] = len(items)
        if args.dry_run:
            continue
        qdata = mosaic_quadrant(
            items, qbbox, args.max_scenes, qname,
            coverage_target=args.coverage_target,
            coverage_min_accept=args.coverage_min,
        )
        if not qdata:
            raise RuntimeError(f"[{qname}] Failed to generate quadrant mosaic.")
        quadrants_data[qname] = qdata
        save_quadrant_checkpoint(qname, qdata)
        done_quadrants.add(qname)
        progress["quadrants_done"] = sorted(done_quadrants)
        progress["scenes_found_per_quadrant"] = scenes_summary
        save_progress(progress)

    if args.dry_run:
        log.info(f"Dry-run complete. Scenes per quadrant: {scenes_summary}")
        return

    mosaic = combine_quadrants(quadrants_data, FULL_BBOX)
    save_geotiff(mosaic, OUT_TIF)

    metadata = {
        "city": PACARAIMA["name"],
        "center_lon": lon,
        "center_lat": lat,
        "full_bbox_equivalente_q1q4": FULL_BBOX,
        "quadrants": quadrant_bboxes,
        "resolution_m": 10.0,
        "year": args.year,
        "year_start": args.year_start,
        "max_cloud_pct": args.max_cloud,
        "max_scenes_per_quadrant": args.max_scenes,
        "scenes_found_per_quadrant": scenes_summary,
        "source": STAC_URL,
        "collection": COLLECTION,
        "output_tif": str(OUT_TIF),
        "generated_at": datetime.now().isoformat(),
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "progress_json": str(PROGRESS_JSON),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    log.info(f"GeoTIFF saved: {OUT_TIF}")
    log.info(f"Metadata saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
