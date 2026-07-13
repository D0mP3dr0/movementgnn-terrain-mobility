"""Shared utilities for Defence Technology article figures (NATO terminology).

Adapted from GNN_RESTRICAO_MOV_OB_V1/artigo_tcc_esimex/figuras/utils.py
with all labels translated to NATO/English doctrine.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

WORK_DIR = Path(__file__).resolve().parent
OUT_DIR = WORK_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE = Path(r"F:\arpia_topo_refinado\TOPO_RESTRICAO_MOVIMENTO")
COLAB_DIR = BASE / "colab"
ANALISE_DIR = COLAB_DIR / "analise_dados"
COLAB_RESULTS_V2 = COLAB_DIR / "results" / "results_v2"
COLAB_RESULTS_LOCAL = COLAB_DIR / "results" / "results_v2_local_blindada_fix"
DATA_DIR = BASE / "GNN_RESTRICAO_MOV_OB_V1" / "dados_reunidos"
GNN_DIR = BASE / "GNN_RESTRICAO_MOVIMENTO"
RESULTS_DIR = GNN_DIR / "results"
RAW_DIR = GNN_DIR / "data_raw" / "satelites_raw"
GEO_DIR = (BASE / "GNN_RESTRICAO_MOV_OB_V1" / "artigo_tcc_esimex"
           / "figuras" / "data_geo")
GEOTIFF_DIR = COLAB_DIR / "results" / "geotiff-20260404T200839Z-3-001"

FIG_WIDTH_CM = 16.0
FIG_WIDTH_IN = FIG_WIDTH_CM / 2.54
DPI = 600

PALETTE = {
    "bg":        "#F5F7FA",
    "panel":     "#FFFFFF",
    "edge":      "#1F2D3D",
    "text":      "#111827",
    "muted":     "#6B7280",
    "primary":   "#1F3A5F",
    "accent":    "#8C6E2B",
    "accent2":   "#5B7A99",
    "highlight": "#B04B2F",
    "grid":      "#D1D5DB",
}

CLASS_COLORS = {
    "go":     "#2E9B4A",
    "slowgo": "#F0C432",
    "nogo":   "#B8351F",
}
CLASS_NAMES = ["Go (Unrestricted)", "Slow Go (Restricted)",
               "No Go (Severely Restricted)"]
CLASS_NAMES_SHORT = ["Go", "SlwGo", "NoGo"]
CLASS_VALUES = [1, 2, 3]

FRACTIONS = ["a_pe", "motorizada", "mecanizada", "blindada"]
FRACTION_TITLES = {
    "a_pe":       "Dismounted",
    "motorizada": "Motorized",
    "mecanizada": "Mechanized",
    "blindada":   "Armored",
}
FRACTION_SHORT = {
    "a_pe":       "Dis",
    "motorizada": "Mot",
    "mecanizada": "Mec",
    "blindada":   "Arm",
}


def set_mpl_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "axes.edgecolor": PALETTE["edge"],
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_fig(fig: plt.Figure, name: str, **kwargs) -> Path:
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight",
                facecolor="white", pad_inches=0.1, **kwargs)
    plt.close(fig)
    print(f"  -> {path.name}  [{path.stat().st_size / 1e6:.1f} MB @ {DPI} DPI]")
    return path


def class_cmap():
    from matplotlib.colors import ListedColormap
    return ListedColormap(
        [CLASS_COLORS["go"], CLASS_COLORS["slowgo"], CLASS_COLORS["nogo"]]
    )


def load_labels() -> dict[str, np.ndarray]:
    data = np.load(ANALISE_DIR / "labels.npz")
    return {f: data[f] for f in FRACTIONS}


def load_predictions() -> dict[str, np.ndarray]:
    path = COLAB_RESULTS_V2 / "predictions_full.npz"
    if not path.exists():
        path = COLAB_RESULTS_LOCAL / "predictions_full.npz"
    data = np.load(path)
    return {f: data[f] for f in FRACTIONS}


def load_features(columns: list[str] | None = None) -> dict[str, np.ndarray]:
    data = np.load(ANALISE_DIR / "features_full.npz")
    feat_names = list(data["feature_names"])
    feats = data["features"]
    if columns is None:
        columns = feat_names
    d = {}
    for name in columns:
        if name in feat_names:
            d[name] = feats[:, feat_names.index(name)]
    return d


def load_positions() -> np.ndarray:
    data = np.load(ANALISE_DIR / "pos.npz")
    return data[data.files[0]]


def raster_meta() -> dict:
    import json
    with open(ANALISE_DIR / "raster_meta.json", encoding="utf-8") as f:
        return json.load(f)


def flat_to_2d(values: np.ndarray, pos: np.ndarray,
               H: int, W: int, transform,
               fill: float = np.nan) -> np.ndarray:
    from rasterio.transform import Affine
    if not isinstance(transform, Affine):
        transform = Affine(*transform[:6])
    arr = np.full((H, W), fill, dtype=float)
    inv = ~transform
    lats = pos[:, 1] if pos[:, 0].min() < -10 else pos[:, 0]
    lons = pos[:, 0] if pos[:, 0].min() < -10 else pos[:, 1]
    cols, rows = inv * (lons, lats)
    rows_i = np.round(rows).astype(int)
    cols_i = np.round(cols).astype(int)
    ok = (rows_i >= 0) & (rows_i < H) & (cols_i >= 0) & (cols_i < W)
    arr[rows_i[ok], cols_i[ok]] = values[ok]
    return arr


def hillshade(elev: np.ndarray, az: float = 315.0,
              alt: float = 45.0, vert_exag: float = 3.0) -> np.ndarray:
    from matplotlib.colors import LightSource
    ls = LightSource(azdeg=az, altdeg=alt)
    return ls.hillshade(elev, vert_exag=vert_exag)
