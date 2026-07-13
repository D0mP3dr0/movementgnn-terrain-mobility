"""
Export baseline predictions to a multi-band GeoTIFF (4 fractions).

Usage:
    python -m src.baselines.export_baseline_to_geotiff \
        --data_path "path/to/graph.pt" \
        --baseline random_forest \
        --output_tif "output/restriction_map.tif"
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from rasterio.transform import from_origin
import rasterio
from torch_geometric.data import Data

from .random_forest_baseline import train_rf_baseline
from .mlp_baseline import train_mlp_baseline
from .rule_based_baseline import train_rule_baseline


def _load_dem_data(data_path: str) -> Data:
    """Load HeteroData and extract the DEM subgraph as a Data object."""
    hetero_data = torch.load(data_path, map_location="cpu", weights_only=False)
    dem_store = hetero_data["dem"]

    edge_index = None
    for edge_type in hetero_data.edge_types:
        if edge_type[0] == "dem" and edge_type[2] == "dem":
            edge_index = hetero_data[edge_type].edge_index
            break

    if edge_index is None:
        raise ValueError("Could not find DEM-DEM edges in the graph.")

    return Data(
        x=dem_store.x,
        edge_index=edge_index,
        pos=getattr(dem_store, "pos", None),
        y_a_pe=dem_store.y_a_pe,
        y_motorizada=dem_store.y_motorizada,
        y_mecanizada=dem_store.y_mecanizada,
        y_blindada=dem_store.y_blindada,
        train_mask=dem_store.train_mask,
        val_mask=dem_store.val_mask,
        test_mask=dem_store.test_mask,
    )


def _infer_grid_and_transform(pos: np.ndarray) -> Tuple[int, int, np.ndarray, np.ndarray, object]:
    """
    Build 2D grid (row/col per node) and geospatial transform.
    Assumes lat/lon coordinates in EPSG:4326.
    """
    if pos is None:
        raise ValueError("DEM graph missing 'pos' attribute; cannot georeference.")

    lats = pos[:, 0]
    lons = pos[:, 1]

    unique_lats = np.unique(lats)
    unique_lons = np.unique(lons)

    rows = unique_lats.size
    cols = unique_lons.size
    n_nodes = pos.shape[0]
    if rows * cols != n_nodes:
        raise ValueError(
            f"Irregular grid detected: rows*cols={rows*cols} != n_nodes={n_nodes}. "
            "Grid mapping adjustment needed before raster export."
        )

    lat_idx = np.searchsorted(unique_lats, lats)
    lon_idx = np.searchsorted(unique_lons, lons)

    # Raster row 0 at north (highest latitude)
    row_idx = (rows - 1) - lat_idx
    col_idx = lon_idx

    if cols > 1:
        xres = float(np.median(np.diff(unique_lons)))
    else:
        xres = 1.0

    if rows > 1:
        yres = float(np.median(np.diff(unique_lats)))
    else:
        yres = 1.0

    west = float(unique_lons.min() - (xres / 2.0))
    north = float(unique_lats.max() + (yres / 2.0))
    transform = from_origin(west, north, abs(xres), abs(yres))

    return rows, cols, row_idx, col_idx, transform


def _predictions_to_raster_bands(
    preds: Dict[str, np.ndarray],
    rows: int,
    cols: int,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
) -> np.ndarray:
    """Build a [4, rows, cols] stack with restriction codes 1..3."""
    fractions = ["a_pe", "motorizada", "mecanizada", "blindada"]
    bands = np.zeros((4, rows, cols), dtype=np.uint8)

    for band_i, frac in enumerate(fractions):
        values = preds[frac].astype(np.uint8)
        bands[band_i, row_idx, col_idx] = values

    return bands


def _run_baseline_and_get_full_predictions(
    baseline_name: str,
    data: Data,
    artifacts_dir: Path,
) -> Dict[str, np.ndarray]:
    """Train chosen baseline and return predictions for ALL nodes."""
    if baseline_name == "random_forest":
        _, _, preds = train_rf_baseline(
            data,
            output_dir=str(artifacts_dir / "metrics"),
            model_dir=str(artifacts_dir / "models" / "random_forest"),
            full_predictions_path=str(artifacts_dir / "predictions_full" / "random_forest_full_predictions.npz"),
            return_full_predictions=True,
        )
        return preds

    if baseline_name == "mlp":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _, _, preds = train_mlp_baseline(
            data,
            output_dir=str(artifacts_dir / "metrics"),
            device=device,
            model_dir=str(artifacts_dir / "models" / "mlp"),
            full_predictions_path=str(artifacts_dir / "predictions_full" / "mlp_full_predictions.npz"),
            return_full_predictions=True,
        )
        return preds

    if baseline_name == "rule_based":
        _, _, preds = train_rule_baseline(
            data,
            output_dir=str(artifacts_dir / "metrics"),
            full_predictions_path=str(artifacts_dir / "predictions_full" / "rule_based_full_predictions.npz"),
            return_full_predictions=True,
        )
        return preds

    raise ValueError(f"Unknown baseline: {baseline_name}")


def export_baseline_to_geotiff(
    data_path: str,
    baseline_name: str,
    output_tif: str,
    artifacts_dir: str,
    crs: str = "EPSG:4326",
):
    """Full pipeline: baseline -> full prediction -> GeoTIFF."""
    artifacts = Path(artifacts_dir)
    (artifacts / "metrics").mkdir(parents=True, exist_ok=True)
    (artifacts / "models").mkdir(parents=True, exist_ok=True)
    (artifacts / "predictions_full").mkdir(parents=True, exist_ok=True)

    data = _load_dem_data(data_path)
    pos = data.pos.cpu().numpy() if data.pos is not None else None

    preds = _run_baseline_and_get_full_predictions(baseline_name, data, artifacts)

    rows, cols, row_idx, col_idx, transform = _infer_grid_and_transform(pos)
    bands = _predictions_to_raster_bands(preds, rows, cols, row_idx, col_idx)

    out = Path(output_tif)
    out.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        out,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=4,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
        nodata=0,
        compress="deflate",
    ) as dst:
        dst.write(bands[0], 1)
        dst.write(bands[1], 2)
        dst.write(bands[2], 3)
        dst.write(bands[3], 4)
        dst.set_band_description(1, "restricao_a_pe")
        dst.set_band_description(2, "restricao_motorizada")
        dst.set_band_description(3, "restricao_mecanizada")
        dst.set_band_description(4, "restricao_blindada")
        dst.update_tags(
            MODEL=baseline_name,
            CLASSES="1=Unrestricted,2=Restricted,3=SeverelyRestricted",
            SOURCE_DATA=str(data_path),
        )

    print(f"GeoTIFF generated: {out}")


def main():
    parser = argparse.ArgumentParser(description="Export baseline predictions to multi-band GeoTIFF.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to prepared graph .pt file.")
    parser.add_argument(
        "--baseline",
        type=str,
        choices=["random_forest", "mlp", "rule_based"],
        default="random_forest",
    )
    parser.add_argument("--output_tif", type=str, required=True, help="Output GeoTIFF path.")
    parser.add_argument("--artifacts_dir", type=str, required=True, help="Directory for intermediate artifacts.")
    parser.add_argument("--crs", type=str, default="EPSG:4326")
    args = parser.parse_args()

    export_baseline_to_geotiff(
        data_path=args.data_path,
        baseline_name=args.baseline,
        output_tif=args.output_tif,
        artifacts_dir=args.artifacts_dir,
        crs=args.crs,
    )


if __name__ == "__main__":
    main()
