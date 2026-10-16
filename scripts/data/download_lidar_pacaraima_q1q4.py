from __future__ import annotations
INTERNAL_DATA_ROOT = __import__("os").environ.get("MOVEMENTGNN_DATA_ROOT", ".")

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List

import h5py
import pandas as pd

try:
    import earthaccess
except Exception as exc:
    raise RuntimeError("Dependencia ausente: earthaccess. Instale com pip install earthaccess") from exc

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

FULL_BBOX = [-61.6138630116465, 3.9760673588888893, -60.682286988353496, 4.97551041888889]
DATE_RANGE = ("2022-01-01", "2024-12-31")

ROOT = Path(rf"{INTERNAL_DATA_ROOT}\GNN_RESTRICAO_MOVIMENTO\data_raw\satelites_raw\lidar")
CSV_OUT = ROOT / "pacaraima_q1q4_lidar_anchors.csv"
REPORT_OUT = ROOT / "pacaraima_q1q4_lidar_report.json"
CHECKPOINT_DIR = ROOT / "_checkpoint_q1q4"
STATE_OUT = CHECKPOINT_DIR / "pacaraima_q1q4_lidar_state.json"
ENV_PATH = ROOT.parent / ".env"


def filter_bbox(df: pd.DataFrame, bbox: List[float]) -> pd.DataFrame:
    min_lon, min_lat, max_lon, max_lat = bbox
    return df[
        (df["lon"] >= min_lon) &
        (df["lon"] <= max_lon) &
        (df["lat"] >= min_lat) &
        (df["lat"] <= max_lat)
    ]


def setup_auth(allow_netrc: bool = False):
    if load_dotenv and ENV_PATH.exists():
        load_dotenv(ENV_PATH)
        log.info(f"Credenciais carregadas: {ENV_PATH}")

    has_user = bool((os.getenv("EARTHDATA_USERNAME") or "").strip())
    has_pass = bool((os.getenv("EARTHDATA_PASSWORD") or "").strip())
    has_token = bool((os.getenv("EARTHDATA_TOKEN") or "").strip())
    log.info(f"Earthdata env -> user={has_user}, pass={has_pass}, token={has_token}")

    try:
        earthaccess.login(strategy="environment", persist=False)
        log.info("Login Earthdata via environment: OK")
        return
    except Exception as e_env:
        log.warning(f"Falha login environment: {e_env}")
    if not allow_netrc:
        raise RuntimeError(
            'Earthdata credentials missing/invalid in the environment. Set EARTHDATA_USERNAME+EARTHDATA_PASSWORD or EARTHDATA_TOKEN.'
        )

    earthaccess.login(persist=True)
    log.info("Login Earthdata via netrc: OK")


def process_atl08(path: Path, bbox: List[float]) -> pd.DataFrame:
    out = []
    with h5py.File(path, "r") as f:
        beams = [k for k in f.keys() if k.startswith("gt")]
        for beam in beams:
            if "land_segments" not in f[beam]:
                continue
            try:
                l_seg = f[beam]["land_segments"]
                terrain = l_seg["terrain"]
                elev = terrain["h_te_best_fit"][:].flatten()
                lat = l_seg["latitude"][:].flatten()
                lon = l_seg["longitude"][:].flatten()
                df = pd.DataFrame({"lat": lat, "lon": lon, "elevation": elev, "source": "ICESat-2", "beam": beam})
                df = df[(df["elevation"] > -100) & (df["elevation"] < 10000)]
                df = filter_bbox(df, bbox)
                if not df.empty:
                    out.append(df)
            except Exception:
                continue
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def process_gedi(path: Path, bbox: List[float]) -> pd.DataFrame:
    out = []
    with h5py.File(path, "r") as f:
        beams = [k for k in f.keys() if k.startswith("BEAM")]
        for beam in beams:
            try:
                g = f[beam]
                lat = g["lat_lowestmode"][:]
                lon = g["lon_lowestmode"][:]
                elev = g["elev_lowestmode"][:]
                qual = g["quality_flag"][:]
                df = pd.DataFrame({"lat": lat, "lon": lon, "elevation": elev, "quality": qual, "source": "GEDI", "beam": beam})
                df = df[df["quality"] == 1]
                df = df[(df["elevation"] > -100) & (df["elevation"] < 10000)]
                df = filter_bbox(df, bbox)
                if not df.empty:
                    out.append(df[["lat", "lon", "elevation", "source", "beam"]])
            except Exception:
                continue
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def granule_uid(granule) -> str:
    uid = getattr(granule, "id", None)
    if uid:
        return str(uid)
    return str(granule)


def load_state() -> dict:
    if not STATE_OUT.exists():
        return {
            "processed_icesat2": [],
            "processed_gedi": [],
            "total_points": 0,
            "csv_size_mb": 0.0,
            "updated_at": None,
        }
    try:
        with open(STATE_OUT, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "processed_icesat2": [],
            "processed_gedi": [],
            "total_points": 0,
            "csv_size_mb": 0.0,
            "updated_at": None,
        }


def save_state(state: dict):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = datetime.now().isoformat()
    with open(STATE_OUT, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def append_points_csv(df: pd.DataFrame):
    ROOT.mkdir(parents=True, exist_ok=True)
    write_header = not CSV_OUT.exists()
    df.to_csv(CSV_OUT, mode="a", header=write_header, index=False)


def save_report(report: dict):
    ROOT.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download LiDAR Pacaraima Q1-Q4")
    parser.add_argument(
        "--allow-netrc",
        action="store_true",
        help='Allows falling back to .netrc credentials (may fail if they are invalid).',
    )
    args = parser.parse_args()

    ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    setup_auth(allow_netrc=args.allow_netrc)
    state = load_state()
    processed_ice = set(state.get("processed_icesat2", []))
    processed_gedi = set(state.get("processed_gedi", []))

    report = {
        "pipeline": "download_lidar_pacaraima_q1q4",
        "timestamp": datetime.now().isoformat(),
        "bbox": FULL_BBOX,
        "date_range": DATE_RANGE,
        "icesat2": {"granules_found": 0, "processed_files": 0, "points": 0, "errors": 0},
        "gedi": {"granules_found": 0, "processed_files": 0, "points": 0, "errors": 0},
        "output_csv": str(CSV_OUT),
        "checkpoint_state": str(STATE_OUT),
    }
    d_ice = CHECKPOINT_DIR / "downloads" / "icesat2"
    d_gedi = CHECKPOINT_DIR / "downloads" / "gedi"
    d_ice.mkdir(parents=True, exist_ok=True)
    d_gedi.mkdir(parents=True, exist_ok=True)

    total_points = int(state.get("total_points", 0))

    try:
        log.info("[Pacaraima] ICESat-2 ATL08")
        res_ice = earthaccess.search_data(short_name="ATL08", bounding_box=tuple(FULL_BBOX), temporal=DATE_RANGE)
        report["icesat2"]["granules_found"] = len(res_ice)
        for g in res_ice:
            gid = granule_uid(g)
            if gid in processed_ice:
                continue
            try:
                files = earthaccess.download([g], str(d_ice))
                if files:
                    fp = Path(files[0])
                    df = process_atl08(fp, FULL_BBOX)
                    report["icesat2"]["processed_files"] += 1
                    if not df.empty:
                        append_points_csv(df)
                        n = int(len(df))
                        report["icesat2"]["points"] += n
                        total_points += n
                    processed_ice.add(gid)
                    fp.unlink(missing_ok=True)
                    state["processed_icesat2"] = sorted(processed_ice)
                    state["processed_gedi"] = sorted(processed_gedi)
                    state["total_points"] = total_points
                    state["csv_size_mb"] = round(CSV_OUT.stat().st_size / 1e6, 3) if CSV_OUT.exists() else 0.0
                    save_state(state)
                    report["total_points"] = total_points
                    report["csv_size_mb"] = state["csv_size_mb"]
                    save_report(report)
            except Exception:
                report["icesat2"]["errors"] += 1
                save_report(report)
    except Exception as exc:
        log.warning(f"[Pacaraima] ICESat-2 falhou: {exc}")
        report["icesat2"]["errors"] += 1

    try:
        log.info("[Pacaraima] GEDI02_A")
        res_gedi = earthaccess.search_data(short_name="GEDI02_A", bounding_box=tuple(FULL_BBOX), temporal=DATE_RANGE)
        report["gedi"]["granules_found"] = len(res_gedi)
        for g in res_gedi:
            gid = granule_uid(g)
            if gid in processed_gedi:
                continue
            try:
                files = earthaccess.download([g], str(d_gedi))
                if files:
                    fp = Path(files[0])
                    df = process_gedi(fp, FULL_BBOX)
                    report["gedi"]["processed_files"] += 1
                    if not df.empty:
                        append_points_csv(df)
                        n = int(len(df))
                        report["gedi"]["points"] += n
                        total_points += n
                    processed_gedi.add(gid)
                    fp.unlink(missing_ok=True)
                    state["processed_icesat2"] = sorted(processed_ice)
                    state["processed_gedi"] = sorted(processed_gedi)
                    state["total_points"] = total_points
                    state["csv_size_mb"] = round(CSV_OUT.stat().st_size / 1e6, 3) if CSV_OUT.exists() else 0.0
                    save_state(state)
                    report["total_points"] = total_points
                    report["csv_size_mb"] = state["csv_size_mb"]
                    save_report(report)
            except Exception:
                report["gedi"]["errors"] += 1
                save_report(report)
    except Exception as exc:
        log.warning(f"[Pacaraima] GEDI falhou: {exc}")
        report["gedi"]["errors"] += 1

    report["total_points"] = total_points
    report["csv_size_mb"] = round(CSV_OUT.stat().st_size / 1e6, 3) if CSV_OUT.exists() else 0.0
    save_report(report)

    log.info(f"Relatorio: {REPORT_OUT}")
    log.info(f"Total pontos LiDAR: {report['total_points']}")


if __name__ == "__main__":
    main()

