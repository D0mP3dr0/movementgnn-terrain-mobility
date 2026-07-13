"""
15_analise_geoespacial.py - GEOINT Cartographic Maps for Publication
=====================================================================
Generates publication-quality maps for cartographic journals and GEOINT theses.

Three independent scales (x3 images per map):
  S1 - 1:100,000  center (4.47659 N, -61.14721 W)  A2 paper
  S2 - 1:50,000   center (4.59310 N, -61.11385 W)  A3 paper
  S3 - 1:25,000   center (4.423705 N,-61.148943 W) A3 paper

Maps per scale (7 types x 4 fractions = 28 maps x 3 scales = 84 PNG files):
  - Comparative 4 models (GNN/RF/MLP/Rule)
  - Disagreement GNN vs Rule-Based
  - Uncertainty (softmax entropy; requires probs_full.npz)
  - Dangerous vs conservative errors
  - Spatial coherence: isolated pixels
  - NDVI transition zones
  - Topographic profile with classification

Quality:
  - DPI: 600 for final print
  - Fill nearest-neighbor on all rasters (no gaps)
  - Multi-directional hillshade + hypsometric tinting
  - Precise cartographic scale (computed from paper/scale/latitude)
  - Elements: scale bar, north arrow, graticule, double neat line,
    full cartouche, South America inset
"""

import gc
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
from matplotlib.colors import (LightSource, ListedColormap, BoundaryNorm,
                                LinearSegmentedColormap)
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

try:
    import contextily as cx
    HAS_CTX = True
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "contextily"])
    import contextily as cx
    HAS_CTX = True

SATELLITE_SOURCE = cx.providers.Esri.WorldImagery

# --- Publication Quality Settings ---
PUBLICATION_DPI = 600
CLASS_ALPHAS = [0.25, 0.30, 0.35]

FIGSIZE_A2  = (23.39, 16.54)
FIGSIZE_A3  = (16.54, 11.69)
FIGSIZE_A4P = (11.69,  8.27)

plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["DejaVu Serif", "Liberation Serif",
                         "Times New Roman", "serif"],
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   9,
    "xtick.labelsize":  7.5,
    "ytick.labelsize":  7.5,
    "legend.fontsize":  8,
    "legend.title_fontsize": 8.5,
    "figure.dpi":       PUBLICATION_DPI,
    "savefig.dpi":      PUBLICATION_DPI,
    "figure.facecolor": "white",
    "axes.facecolor":   "#f5f5f0",
    "axes.edgecolor":   "#2c3e50",
    "axes.linewidth":   1.2,
    "grid.alpha":       0.45,
    "grid.linewidth":   0.35,
})

# Three Cartographic Scales
# half_lon = (scale * map_w_in * 0.0254) / (111320 * cos(lat)) / 2
# half_lat = (scale * map_h_in * 0.0254) / 111320 / 2
def _calc_half(scale, map_w_in, map_h_in, center_lat):
    """Compute half_lon and half_lat for exact-scale map window."""
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(center_lat))
    half_lon = (scale * map_w_in * 0.0254) / m_per_deg_lon / 2.0
    half_lat = (scale * map_h_in * 0.0254) / m_per_deg_lat / 2.0
    return round(half_lon, 6), round(half_lat, 6)


_S1_c  = (4.47659,   -61.14721)
_S2_c  = (4.59310,   -61.11385)
_S3_c  = (4.423705,  -61.148943)

_hl1_2, _hla1_2 = _calc_half(100_000, 6.5, 8.0, _S1_c[0])
_hl2_2, _hla2_2 = _calc_half( 50_000, 6.5, 8.0, _S2_c[0])
_hl3_2, _hla3_2 = _calc_half( 25_000, 6.5, 8.0, _S3_c[0])

_hl1_4, _hla1_4 = _calc_half(100_000, 8.5, 5.5, _S1_c[0])
_hl2_4, _hla2_4 = _calc_half( 50_000, 6.0, 3.8, _S2_c[0])
_hl3_4, _hla3_4 = _calc_half( 25_000, 6.0, 3.8, _S3_c[0])

SCALE_CONFIGS = [
    {
        "id":          "s1_100k",
        "label":       "1:100,000",
        "scale":       100_000,
        "center":      _S1_c,
        "half_2panel": (_hl1_2, _hla1_2),
        "half_4panel": (_hl1_4, _hla1_4),
        "fig_4p":      FIGSIZE_A2,
        "fig_2p":      FIGSIZE_A3,
        "fig_pro":     FIGSIZE_A4P,
        "paper":       "A2 / A3",
    },
    {
        "id":          "s2_50k",
        "label":       "1:50,000",
        "scale":       50_000,
        "center":      _S2_c,
        "half_2panel": (_hl2_2, _hla2_2),
        "half_4panel": (_hl2_4, _hla2_4),
        "fig_4p":      FIGSIZE_A3,
        "fig_2p":      FIGSIZE_A3,
        "fig_pro":     FIGSIZE_A4P,
        "paper":       "A3",
    },
    {
        "id":          "s3_25k",
        "label":       "1:25,000",
        "scale":       25_000,
        "center":      _S3_c,
        "half_2panel": (_hl3_2, _hla3_2),
        "half_4panel": (_hl3_4, _hla3_4),
        "fig_4p":      FIGSIZE_A3,
        "fig_2p":      FIGSIZE_A3,
        "fig_pro":     FIGSIZE_A4P,
        "paper":       "A3",
    },
]

def extent_from_cfg(cfg, mode="2p"):
    """Return [lon_min, lon_max, lat_min, lat_max] for scale config."""
    lat, lon = cfg["center"]
    if mode == "4p":
        hl, hla = cfg["half_4panel"]
    else:
        hl, hla = cfg["half_2panel"]
    return [lon - hl, lon + hl, lat - hla, lat + hla]


# --- NATO / GEOINT Cartographic Palette ---
CLASS_COLORS = {0: "#00E626", 1: "#FFD000", 2: "#FF1A1A"}
CLASS_NAMES  = {0: "Go (Unrestricted)", 1: "Slow Go (Restricted)", 2: "No Go (Sev. Restricted)"}
CLASS_CMAP   = ListedColormap([CLASS_COLORS[i] for i in range(3)])
CLASS_NORM   = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], ncolors=3)

MODEL_COLORS = {"GNN": "#1565C0", "RF": "#2e7d32", "MLP": "#6a1b9a", "Rule": "#bf360c"}
FRAC_TITLES  = {"a_pe": "Dismounted", "motorizada": "Motorized",
                "mecanizada": "Mechanized", "blindada": "Armored"}

HYPSO_CMAP = LinearSegmentedColormap.from_list("hypso", [
    (0.00, "#3d6b2e"), (0.20, "#7daa3e"), (0.40, "#c8a04a"),
    (0.60, "#9b7a4f"), (0.80, "#8a7060"), (1.00, "#e0dcd8"),
])
ERR_CMAP2 = ListedColormap(["#b2182b", "#d9d9d9", "#2166ac"])
ERR_NORM2 = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], ncolors=3)

import sys
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from src.paths import get_analysis_paths, is_colab

IS_COLAB = is_colab()
_analysis = get_analysis_paths("15_geoespacial")
GRAPH_PATH = _analysis["graph"]
GNN_PREDS  = _analysis["gnn_preds"]
GNN_PROBS  = _analysis["gnn_probs"]
RF_PREDS   = _analysis["rf_preds"]
MLP_PREDS  = _analysis["mlp_preds"]
RULE_PREDS = _analysis["rule_preds"]
OUT_DIR    = _analysis["out_dir"]

FRACTIONS    = ["a_pe", "motorizada", "mecanizada", "blindada"]
NDVI_THRESH  = {"a_pe": 0.7, "motorizada": 0.5, "mecanizada": 0.6, "blindada": 0.5}
MARGIN       = 0.05
IDX_ELEV = 0; IDX_SLOPE = 1; IDX_TPI = 5; IDX_NDVI = 12


# --- Data Helpers ---
def log(msg): print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _norm0(arr):
    if arr is None: return None
    a = arr.astype(np.int32)
    return a - 1 if int(a.min()) >= 1 else a


def load_preds(path, fractions):
    if path is None or not Path(str(path)).exists(): return {}
    d = np.load(str(path))
    return {f: _norm0(d[f if f in d else f"y_{f}"].astype(np.int32))
            for f in fractions if f in d or f"y_{f}" in d}


def load_probs(path, fractions):
    if path is None or not Path(str(path)).exists(): return {}
    d = np.load(str(path))
    return {f: d[f"probs_{f}" if f"probs_{f}" in d else f].astype(np.float32)
            for f in fractions if f"probs_{f}" in d or f in d}


def entropy_norm(probs):
    """Normalized Shannon entropy (0-1 range)."""
    eps = 1e-8
    h = -np.sum(probs * np.log(probs + eps), axis=1)
    return h / np.log(probs.shape[1])


# --- Raster: Reconstruction and Fill ---
_FLAT2D_LOGGED = False

def flat_to_2d(values, pos, H, W, transform, fill=np.nan):
    """Reconstruct 2D array from flat array + positions (lat/lon)."""
    global _FLAT2D_LOGGED
    arr = np.full((H, W), fill, dtype=np.float32)

    if hasattr(transform, 'a'):
        ta, tc, te, tf = transform.a, transform.c, transform.e, transform.f
    else:
        ta, _, tc, _, te, tf = transform[:6]

    # Auto-detect if pos is [lat,lon] or [lon,lat]
    col0_near_lon = abs(pos[:, 0].mean() - tc) < abs(pos[:, 0].mean() - tf)
    if col0_near_lon:
        lats = pos[:, 1]
        lons = pos[:, 0]
    else:
        lats = pos[:, 0]
        lons = pos[:, 1]

    rows = np.clip(((tf - lats) / abs(te)).astype(int), 0, H - 1)
    cols = np.clip(((lons - tc) / abs(ta)).astype(int), 0, W - 1)

    if not _FLAT2D_LOGGED:
        _FLAT2D_LOGGED = True
        log(f"  flat_to_2d: transform a={ta:.8f} c={tc:.6f} e={te:.8f} f={tf:.6f}")
        log(f"    pos order: {'[lon,lat]' if col0_near_lon else '[lat,lon]'}")
        log(f"    rows: [{rows.min()},{rows.max()}] (H={H}) | cols: [{cols.min()},{cols.max()}] (W={W})")

    arr[rows, cols] = values
    return arr


def fill_nearest(arr):
    """Fill all NaN values with nearest valid neighbor value."""
    from scipy.ndimage import distance_transform_edt
    nan_mask = np.isnan(arr)
    if not nan_mask.any():
        return arr
    idx = distance_transform_edt(nan_mask,
                                  return_distances=False,
                                  return_indices=True)
    return arr[tuple(idx)]


def crop(arr, src_ext, tgt_ext, fill=np.nan):
    """Crop 2D or 3D array from src_ext to tgt_ext.

    src_ext = tgt_ext = [lon_min, lon_max, lat_min, lat_max]
    Returns (cropped_arr, actual_extent).
    """
    if arr is None:
        return None, tgt_ext
    H, W = arr.shape[:2]
    lon_min_s, lon_max_s, lat_min_s, lat_max_s = src_ext
    lon_min_t, lon_max_t, lat_min_t, lat_max_t = tgt_ext

    px_lon = W / (lon_max_s - lon_min_s)
    px_lat = H / (lat_max_s - lat_min_s)

    c0 = int(round((lon_min_t - lon_min_s) * px_lon))
    c1 = int(round((lon_max_t - lon_min_s) * px_lon))
    r0 = int(round((lat_max_s - lat_max_t) * px_lat))
    r1 = int(round((lat_max_s - lat_min_t) * px_lat))

    c0c = max(0, c0); c1c = min(W, c1)
    r0c = max(0, r0); r1c = min(H, r1)

    if r0c >= r1c or c0c >= c1c:
        log(f"  WARNING crop: window outside data extent!")
        return None, tgt_ext

    out = arr[r0c:r1c, c0c:c1c] if arr.ndim == 2 else arr[r0c:r1c, c0c:c1c, :]

    actual = [
        lon_min_s + c0c / px_lon,
        lon_min_s + c1c / px_lon,
        lat_max_s - r1c / px_lat,
        lat_max_s - r0c / px_lat,
    ]
    return out.copy(), actual


def get_full_extent(transform, H, W):
    if hasattr(transform, 'a'):
        return [transform.c, transform.c + W*transform.a,
                transform.f + H*transform.e, transform.f]
    t = transform
    return [t[2], t[2]+W*t[0], t[5]+H*t[4], t[5]]


# --- Hillshade + Vegetation + Hypsometry (pseudo-satellite 3D) ---

VEG_CMAP = LinearSegmentedColormap.from_list("veg_sat", [
    (0.00, "#c4b799"),
    (0.10, "#bfb48f"),
    (0.20, "#a8a878"),
    (0.30, "#8c9e58"),
    (0.40, "#6b8c3a"),
    (0.50, "#4e7a28"),
    (0.65, "#3a6620"),
    (0.80, "#2a5418"),
    (1.00, "#1a3e10"),
])


def build_terrain(elev_2d_m, ndvi_2d=None, vert_exag=6.0):
    """Return (hillshade, rgb_blend, contour_levels, elev_smooth).

    Blends three layers for pseudo-satellite 3D appearance:
      1. Vegetation (NDVI) - land cover texture
      2. Hypsometry - elevation gradient
      3. Multi-directional hillshade - 3D relief
    """
    from scipy.ndimage import gaussian_filter
    e = np.where(np.isnan(elev_2d_m), float(np.nanmean(elev_2d_m)), elev_2d_m)
    e_s = gaussian_filter(e, sigma=1.0)
    dx = 30.0

    # Multi-directional hillshade (4 light sources)
    hs_nw = LightSource(azdeg=315, altdeg=40).hillshade(e_s, vert_exag=vert_exag,     dx=dx, dy=dx)
    hs_ne = LightSource(azdeg=45,  altdeg=35).hillshade(e_s, vert_exag=vert_exag*0.6, dx=dx, dy=dx)
    hs_s  = LightSource(azdeg=180, altdeg=55).hillshade(e_s, vert_exag=vert_exag*0.4, dx=dx, dy=dx)
    hs_w  = LightSource(azdeg=270, altdeg=25).hillshade(e_s, vert_exag=vert_exag*0.3, dx=dx, dy=dx)
    hs = np.clip(0.50*hs_nw + 0.22*hs_ne + 0.15*hs_s + 0.13*hs_w, 0, 1)

    p2, p98 = np.percentile(hs, [2, 98])
    hs = np.clip((hs - p2) / (p98 - p2 + 1e-8), 0, 1)

    e_norm = (e - e.min()) / (e.max() - e.min() + 1e-8)
    hypso  = HYPSO_CMAP(e_norm)

    if ndvi_2d is not None:
        ndvi = np.where(np.isnan(ndvi_2d), 0.0, ndvi_2d).astype(np.float32)
        n_min, n_max = float(ndvi.min()), float(ndvi.max())
        if n_max <= 1.0 and n_min >= -1.0:
            ndvi_n = np.clip((ndvi + 1) / 2, 0, 1) if n_min < 0 else np.clip(ndvi, 0, 1)
        else:
            ndvi_n = np.clip((ndvi - n_min) / (n_max - n_min + 1e-8), 0, 1)
        veg = VEG_CMAP(ndvi_n)

        # Blend: vegetation dominant + hypsometry subtle + hillshade strong
        rgb = np.zeros((*e.shape, 4), dtype=np.float32)
        for c in range(3):
            rgb[:, :, c] = np.clip(
                veg[:, :, c]   * 0.42
                + hypso[:, :, c] * 0.12
                + hs             * 0.30
                + veg[:, :, c] * hs * 0.16
                , 0, 1)
    else:
        rgb = np.zeros((*e.shape, 4), dtype=np.float32)
        for c in range(3):
            rgb[:, :, c] = np.clip(
                hypso[:, :, c] * 0.38 + hs * 0.42 + hypso[:, :, c] * hs * 0.20, 0, 1)
    rgb[:, :, 3] = 1.0

    e_range = e.max() - e.min()
    if e_range > 200:
        step = 100
    elif e_range > 50:
        step = 25
    else:
        step = max(10, int(e_range / 8))
    base = int(np.ceil(e.min() / step) * step)
    contour_levels = list(range(base, int(e.max()) + 1, step))

    return hs, rgb, contour_levels, gaussian_filter(e, sigma=2.0)


# --- Cartographic Elements ---
def setup_axis(ax, extent):
    """Graticule, formatted ticks, subtle grid."""
    lon_min, lon_max, lat_min, lat_max = extent
    span = max(lon_max - lon_min, lat_max - lat_min)
    step = next((s for s in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
                 if span / s <= 7), 0.5)

    lons = np.arange(np.ceil(lon_min/step)*step, lon_max+step*0.1, step)
    lats = np.arange(np.ceil(lat_min/step)*step, lat_max+step*0.1, step)
    ax.set_xticks(lons[lons <= lon_max])
    ax.set_yticks(lats[lats <= lat_max])

    def fmt(v, _, is_lon):
        d = abs(int(v))
        m = (abs(v) - d) * 60
        h = ("E" if v >= 0 else "W") if is_lon else ("N" if v >= 0 else "S")
        return f"{d}\u00b0{m:04.1f}'{h}" if step < 0.1 else f"{d}\u00b0{m:.0f}'{h}"

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,p: fmt(v,p,True)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,p: fmt(v,p,False)))
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.tick_params(labelsize=7, length=3, direction="in", top=True, right=True)
    ax.grid(True, color="white", linewidth=0.3, alpha=0.55, linestyle=":")
    return step


def add_scale_bar(ax, extent, scale_denom):
    """Dual-segment scale bar computed from real distance."""
    lon_min, lon_max, lat_min, lat_max = extent
    mid_lat  = (lat_min + lat_max) / 2
    km_per_deg = 111.32 * math.cos(math.radians(mid_lat))
    span_km    = (lon_max - lon_min) * km_per_deg

    target  = span_km * 0.22
    mag     = 10 ** int(math.log10(max(target, 0.001)))
    bar_km  = round(target / mag) * mag or mag
    bar_deg = bar_km / km_per_deg

    x0  = lon_min + (lon_max - lon_min) * 0.30
    y0  = lat_min + (lat_max - lat_min) * 0.030
    y1  = lat_min + (lat_max - lat_min) * 0.050
    seg = bar_deg / 4

    for i, col in enumerate(["#111", "#fff", "#111", "#fff"]):
        ax.fill_between([x0+i*seg, x0+(i+1)*seg], [y0, y0], [y1, y1],
                        color=col, zorder=10, linewidth=0)
    ax.plot([x0, x0+bar_deg, x0+bar_deg, x0, x0],
            [y0, y0, y1, y1, y0], color="#111", lw=0.7, zorder=11)

    dy = (lat_max - lat_min) * 0.012
    kw = dict(ha="center", va="bottom", fontsize=6.5, zorder=12,
              path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])
    ax.text(x0,             y1+dy, "0",               **kw)
    ax.text(x0+bar_deg/2,   y1+dy, f"{bar_km/2:.0f}", **kw)
    ax.text(x0+bar_deg,     y1+dy, f"{bar_km:.0f} km",
            fontweight="bold", **kw)


def add_north_arrow(ax, extent):
    """Professional bicolor north arrow (ICA standard)."""
    lon_min, lon_max, lat_min, lat_max = extent
    sx, sy = lon_max - lon_min, lat_max - lat_min
    cx = lon_max - sx * 0.055
    cy = lat_max - sy * 0.085
    h, w = sy * 0.065, sx * 0.012

    ax.fill([cx-w/2, cx, cx+w/2, cx-w/2], [cy, cy+h, cy, cy],
            color="#111", zorder=15, linewidth=0)
    ax.fill([cx-w/2, cx, cx+w/2, cx-w/2], [cy, cy-h, cy, cy],
            color="white", zorder=15, linewidth=0)
    ax.plot([cx-w/2,cx,cx+w/2,cx-w/2,cx,cx-w/2],
            [cy,cy+h,cy,cy,cy-h,cy], color="#111", lw=0.6, zorder=16)
    ax.add_patch(plt.Circle((cx,cy), h*0.11, color="#111", zorder=17))
    ax.text(cx, cy+h*1.22, "N", ha="center", va="center",
            fontsize=8.5, fontweight="bold", zorder=18,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])


def add_neat_line(ax):
    """Double neat line (atlas cartographic standard)."""
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.8)
        sp.set_edgecolor("#2c3e50")
    outer = FancyBboxPatch((0,0),1,1, boxstyle="square,pad=0",
                           transform=ax.transAxes, fill=False,
                           linewidth=2.2, edgecolor="#111111",
                           zorder=25, clip_on=False)
    ax.add_patch(outer)


def add_class_legend(ax, loc="lower right", extra_handles=None):
    """Movement restriction class legend."""
    patches = [mpatches.Patch(facecolor=CLASS_COLORS[i], edgecolor="#333",
                               linewidth=0.5, label=CLASS_NAMES[i],
                               hatch=("///" if i == 2 else None))
               for i in range(3)]
    leg = ax.legend(handles=patches, loc="lower right", fontsize=7,
                    title="Movement Restriction", title_fontsize=7.5,
                    framealpha=1.0, edgecolor="#2c3e50",
                    fancybox=False, borderpad=0.45,
                    handlelength=1.2, handleheight=0.9)
    leg.set_zorder(30)
    leg.get_frame().set_linewidth(0.8)
    ax.add_artist(leg)

    if extra_handles:
        leg2 = ax.legend(handles=extra_handles, loc="center right",
                         fontsize=6.5, title="Additional info",
                         title_fontsize=7.0, framealpha=1.0,
                         edgecolor="#999", fancybox=False,
                         borderpad=0.40, handlelength=1.2)
        leg2.set_zorder(30)
        leg2.get_frame().set_linewidth(0.6)
    return leg


def add_cartouche(fig, title, subtitle="", source="GNN GEOINT Lab",
                  scale_str="", crs_str="", date_str=None, y0=0.003):
    if not date_str:
        date_str = datetime.now().strftime("%B %Y")
    fig.text(0.5, y0+0.026, title, ha="center", va="bottom",
             fontsize=12, fontweight="bold",
             path_effects=[pe.withStroke(linewidth=0.4, foreground="white")])
    if subtitle:
        fig.text(0.5, y0+0.014, subtitle, ha="center", va="bottom",
                 fontsize=8.5, style="italic", color="#444")
    fig.text(0.01, y0+0.003, f"Source: {source}", ha="left",
             va="bottom", fontsize=6.5, color="#555")
    right = "  |  ".join(filter(None, [scale_str, crs_str, date_str]))
    fig.text(0.99, y0+0.003, right, ha="right", va="bottom",
             fontsize=6.5, color="#555")
    fig.add_artist(Line2D([0.01,0.99],[y0+0.001,y0+0.001],
                          transform=fig.transFigure,
                          color="#aaa", linewidth=0.6))


def add_inset(ax, extent):
    """Study area location inset (South America)."""
    ax_i = ax.inset_axes([0.01, 0.08, 0.14, 0.20])
    ax_i.set_facecolor("#c9e6f5")
    ax_i.set_aspect("equal", adjustable="datalim")

    sa_x = [-82,-73,-71,-68,-60,-50,-35,-34,-38,-43,-48,-52,-53,-73,-82]
    sa_y = [  9, 12, -1, -4,  5,  4, -5,-10,-16,-23,-28,-34,-34,  9,  9]
    ax_i.fill(sa_x, sa_y, "#d4c9a8", edgecolor="#888", linewidth=0.5, zorder=2)
    ax_i.add_patch(Rectangle((-74,-34), 41, 39, fill=True,
                              facecolor="#b8d4a0", edgecolor="#555",
                              linewidth=0.5, alpha=0.65, zorder=3))

    lon_min, lon_max, lat_min, lat_max = extent
    w_r = max(lon_max - lon_min, 0.01)
    h_r = max(lat_max - lat_min, 0.01)
    ax_i.add_patch(Rectangle((lon_min, lat_min), w_r, h_r,
                              fill=True, facecolor="#e74c3c",
                              edgecolor="#900", linewidth=1.0,
                              alpha=0.92, zorder=5))

    ax_i.set_xlim(-82, -30)
    ax_i.set_ylim(-36, 14)
    ax_i.set_xticks([]); ax_i.set_yticks([])
    ax_i.set_title("Location", fontsize=5.5, pad=1.5,
                   color="#333", loc="center")

    for sp in ax_i.spines.values():
        sp.set_linewidth(0.8)
        sp.set_edgecolor("#444")
    ax_i.patch.set_edgecolor("#444")
    ax_i.patch.set_linewidth(0.8)
    return ax_i


def render_satellite(ax, extent):
    """Render real satellite imagery background (Esri World Imagery)."""
    lon_min, lon_max, lat_min, lat_max = extent
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    try:
        cx.add_basemap(ax, source=SATELLITE_SOURCE, crs="EPSG:4326",
                       attribution=False, zoom="auto")
    except Exception as e:
        log(f"    WARNING satellite: {e} - using gray background")
        ax.set_facecolor("#d0ccc0")


def render_class(ax, p2d, extent, alpha=0.25):
    """Render classification overlay on satellite with vivid colors.

    Uses a white sub-layer to isolate colors from dark satellite background.
    """
    if p2d is None: return
    valid = ~np.isnan(p2d)
    white_base = np.zeros((*p2d.shape, 4), dtype=np.float32)
    white_base[valid, :3] = 1.0
    white_base[valid, 3]  = alpha * 1.6
    ax.imshow(white_base, extent=extent, interpolation="nearest",
              aspect="auto", zorder=3)
    masked = np.ma.masked_where(~valid, p2d)
    ax.imshow(masked, cmap=CLASS_CMAP, norm=CLASS_NORM,
              extent=extent, alpha=alpha * 2.5,
              interpolation="nearest", aspect="auto", zorder=4)


def stats_box(ax, text, loc="upper left"):
    """Opaque statistics box."""
    x  = 0.02 if "left" in loc else 0.98
    y  = 0.97 if "upper" in loc else 0.55
    ha = "left" if "left" in loc else "right"
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=6.5, va="top", ha=ha, zorder=30,
            fontfamily="monospace", linespacing=1.35,
            bbox=dict(facecolor="white", alpha=1.0,
                      edgecolor="#666", linewidth=0.8,
                      boxstyle="round,pad=0.35"))


def save_map(fig, path):
    fig.savefig(str(path), dpi=PUBLICATION_DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    mb = path.stat().st_size / 1_048_576
    log(f"    -> {path.name}  [{mb:.1f} MB @ {PUBLICATION_DPI} DPI]")


def decorate(ax, extent, scale_denom):
    """Apply all decorative elements to a map axis."""
    setup_axis(ax, extent)
    add_scale_bar(ax, extent, scale_denom)
    add_north_arrow(ax, extent)
    add_neat_line(ax)


# --- Map Functions (one per type, parameterized by extent+scale) ---
MODEL_ORDER = ["GNN", "RF", "MLP", "Rule"]


def _crop_all(rasters_dict, full_ext, tgt_ext):
    return {k: crop(v, full_ext, tgt_ext)[0] for k, v in rasters_dict.items()}


def map_m01(frac, all_preds_2d,
            full_ext, cfg, out_dir, class_alpha=0.25, alpha_tag=""):
    """M01 - Comparative 4 models."""
    ext = extent_from_cfg(cfg, "4p")

    fig = plt.figure(figsize=cfg["fig_4p"], facecolor="white")
    gs  = fig.add_gridspec(2, 2, left=0.05, right=0.97,
                           top=0.90, bottom=0.10,
                           hspace=0.11, wspace=0.05)

    ax_tl = None
    for idx_m, mname in enumerate(MODEL_ORDER):
        ax  = fig.add_subplot(gs[idx_m//2, idx_m%2])
        if idx_m == 0:
            ax_tl = ax
        p2d = crop(all_preds_2d.get(mname, {}).get(frac), full_ext, ext)[0]
        render_satellite(ax, ext)
        render_class(ax, p2d, ext, alpha=class_alpha)

        if p2d is not None:
            v = p2d[~np.isnan(p2d)]
            c = {cl: float((v==cl).sum()/len(v)*100) for cl in range(3)}
            stats_box(ax, f"Go:          {c[0]:5.1f}%\n"
                          f"Slow Go:     {c[1]:5.1f}%\n"
                          f"No Go:       {c[2]:5.1f}%")
        else:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color="#888", style="italic")

        ax.set_title(mname, fontsize=12, fontweight="bold",
                     color=MODEL_COLORS.get(mname,"#333"), pad=5)
        decorate(ax, ext, cfg["scale"])
        add_class_legend(ax)
        ax.set_ylabel("Latitude" if idx_m in [0,2] else "", fontsize=8)
        ax.set_xlabel("Longitude" if idx_m in [2,3] else "", fontsize=8)
        if idx_m in [0,1]: ax.set_xticklabels([])
        if idx_m in [1,3]: ax.set_yticklabels([])

    fig.text(0.5, 0.945,
             f"Movement Restriction Comparison - {FRAC_TITLES[frac]}",
             ha="center", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.924,
             f"Pacaraima, Roraima (Brazil) | {cfg['label']} | {cfg['paper']}",
             ha="center", fontsize=9, style="italic", color="#555")
    if ax_tl is not None:
        add_inset(ax_tl, ext)
    add_cartouche(fig,
        f"Movement Restriction - {FRAC_TITLES[frac]} | GNN vs Classical Methods",
        subtitle="Graph Neural Network (GNN) x Random Forest x MLP x DAMEPLAN Rule",
        source="GNN Pacaraima v2 | GEOINT Lab",
        scale_str=cfg["label"], crs_str="SIRGAS 2000")
    save_map(fig, out_dir / f"15_m01_{cfg['id']}_{frac}{alpha_tag}.png")
    plt.close(fig)


def map_m02(frac, all_preds_2d,
            full_ext, cfg, out_dir, class_alpha=0.25, alpha_tag=""):
    """M02 - Disagreement GNN vs Rule."""
    ext = extent_from_cfg(cfg, "2p")

    gnn_c  = crop(all_preds_2d.get("GNN",  {}).get(frac), full_ext, ext)[0]
    rule_c = crop(all_preds_2d.get("Rule", {}).get(frac), full_ext, ext)[0]

    fig = plt.figure(figsize=cfg["fig_2p"], facecolor="white")
    gs  = fig.add_gridspec(1, 2, left=0.05, right=0.97,
                           top=0.87, bottom=0.10, wspace=0.05)

    ax_l = fig.add_subplot(gs[0,0])
    render_satellite(ax_l, ext)
    render_class(ax_l, gnn_c, ext, alpha=class_alpha)
    ax_l.set_title("GNN - Prediction", fontsize=11, fontweight="bold",
                   color=MODEL_COLORS["GNN"], pad=5)
    decorate(ax_l, ext, cfg["scale"])
    add_class_legend(ax_l)
    ax_l.set_xlabel("Longitude", fontsize=8)
    ax_l.set_ylabel("Latitude", fontsize=8)
    add_inset(ax_l, ext)

    ax_r = fig.add_subplot(gs[0,1])
    render_satellite(ax_r, ext)

    disc_stats = {}
    if gnn_c is not None and rule_c is not None:
        valid  = ~np.isnan(gnn_c) & ~np.isnan(rule_c)
        disc   = np.full_like(gnn_c, np.nan)
        disc[valid] = np.where(gnn_c[valid]>rule_c[valid],  1.0,
                      np.where(gnn_c[valid]<rule_c[valid], -1.0, 0.0))

        mais_r = np.ma.masked_where(disc != 1,  np.ones_like(disc))
        meno_r = np.ma.masked_where(disc != -1, np.ones_like(disc))
        ax_r.imshow(mais_r, cmap=ListedColormap(["#c0392b"]),
                    extent=ext, alpha=0.82, interpolation="nearest",
                    aspect="auto", zorder=5)
        ax_r.imshow(meno_r, cmap=ListedColormap(["#1a7a3c"]),
                    extent=ext, alpha=0.82, interpolation="nearest",
                    aspect="auto", zorder=5)

        nv = int(valid.sum())
        nm = int((disc[valid]==1).sum())
        nl = int((disc[valid]==-1).sum())
        nc = int((disc[valid]==0).sum())
        disc_stats = {"agreement":         round(nc/nv*100,2),
                      "gnn_more_restr":    round(nm/nv*100,2),
                      "gnn_less_restr":    round(nl/nv*100,2)}
        stats_box(ax_r,
            f"Agreement:         {nc/nv*100:5.1f}%\n"
            f"GNN more restr.:   {nm/nv*100:5.1f}%\n"
            f"GNN less restr.:   {nl/nv*100:5.1f}%")

    ptchs = [mpatches.Patch(facecolor="#c0392b",edgecolor="#333",lw=0.5,label="GNN more restrictive"),
             mpatches.Patch(facecolor="#e0e0e0",edgecolor="#333",lw=0.5,label="Agreement"),
             mpatches.Patch(facecolor="#1a7a3c",edgecolor="#333",lw=0.5,label="GNN less restrictive")]
    leg = ax_r.legend(handles=ptchs, loc="lower right", fontsize=7,
                      title="Disagreement", title_fontsize=8,
                      framealpha=0.92, edgecolor="#2c3e50", fancybox=False)
    leg.get_frame().set_linewidth(0.7)
    ax_r.set_title("Disagreement GNN vs Rule-Based", fontsize=11,
                   fontweight="bold", pad=5)
    decorate(ax_r, ext, cfg["scale"])
    ax_r.set_xlabel("Longitude", fontsize=8)
    ax_r.set_yticklabels([])

    fig.text(0.5,0.918,
             f"Disagreement GNN vs Rule-Based - {FRAC_TITLES[frac]}",
             ha="center", fontsize=13, fontweight="bold")
    fig.text(0.5,0.899,
             f"Hypothesis H1 | {cfg['label']} | {cfg['paper']}",
             ha="center", fontsize=8.5, style="italic", color="#555")
    add_cartouche(fig, f"Disagreement GNN vs Rule - {FRAC_TITLES[frac]}",
                  source="GNN Pacaraima v2 | GEOINT Lab",
                  scale_str=cfg["label"], crs_str="SIRGAS 2000")
    save_map(fig, out_dir / f"15_m02_{cfg['id']}_{frac}{alpha_tag}.png")
    plt.close(fig)
    return disc_stats


def map_m03(frac, entr_2d, gnn_pred_2d, trans_2d,
            full_ext, cfg, out_dir, p90, p75, thr_ndvi,
            class_alpha=0.25, alpha_tag=""):
    """M03 - Uncertainty (softmax entropy)."""
    ext = extent_from_cfg(cfg, "2p")
    entr_c  = crop(entr_2d, full_ext, ext)[0]
    gnn_c   = crop(gnn_pred_2d, full_ext, ext)[0]
    trans_c = crop(trans_2d, full_ext, ext)[0]

    UNC_CMAP = LinearSegmentedColormap.from_list("unc", [
        (0.00,"#ffffcc"),(0.25,"#fecc5c"),(0.50,"#fd8d3c"),
        (0.75,"#e31a1c"),(1.00,"#800026")])

    fig = plt.figure(figsize=cfg["fig_2p"], facecolor="white")
    gs  = fig.add_gridspec(1,2,left=0.05,right=0.95,
                           top=0.87,bottom=0.10,wspace=0.07)

    ax_l = fig.add_subplot(gs[0,0])
    render_satellite(ax_l, ext)
    if entr_c is not None:
        im_u = ax_l.imshow(np.ma.masked_where(np.isnan(entr_c),entr_c),
                           cmap=UNC_CMAP, vmin=0, vmax=1,
                           extent=ext, alpha=0.72, interpolation="bilinear",
                           aspect="auto", zorder=4)
        if trans_c is not None:
            ax_l.contour(np.flip(trans_c,0), levels=[0.5], extent=ext,
                         colors=["#00bcd4"], linewidths=1.8, zorder=7)
        fig.colorbar(im_u, ax=ax_l, fraction=0.028, pad=0.015,
                     label="Normalized Entropy", shrink=0.85)
        stats_box(ax_l,
            f"Mean entropy: {float(np.nanmean(entr_c)):.3f}\n"
            f"P75: {p75:.3f}  P90: {p90:.3f}\n"
            "  cyan = NDVI transition zone")
    ax_l.set_title("Softmax Entropy - GNN Uncertainty", fontsize=11,
                   fontweight="bold", pad=5)
    decorate(ax_l, ext, cfg["scale"])
    ax_l.set_xlabel("Longitude", fontsize=8); ax_l.set_ylabel("Latitude", fontsize=8)
    add_inset(ax_l, ext)

    ax_r = fig.add_subplot(gs[0,1])
    render_satellite(ax_r, ext)
    render_class(ax_r, gnn_c, ext, alpha=class_alpha)
    if entr_c is not None:
        high = np.ma.masked_where(entr_c < p90, np.ones_like(entr_c))
        ax_r.imshow(high, cmap=ListedColormap(["#ffd600"]),
                    extent=ext, alpha=0.88, interpolation="nearest",
                    aspect="auto", zorder=6)
    if trans_c is not None:
        ax_r.contour(np.flip(trans_c,0), levels=[0.5], extent=ext,
                     colors=["#00bcd4"], linewidths=1.5, zorder=7)
    add_class_legend(ax_r, extra_handles=[
        mpatches.Patch(facecolor="#ffd600",edgecolor="#333",lw=0.5,
                       label=f"High uncertainty P90 (>{p90:.2f})"),
        Line2D([0],[0],color="#00bcd4",lw=1.8,
               label=f"Trans.NDVI ({thr_ndvi})")
    ])
    ax_r.set_title("GNN + High Uncertainty Regions (P90)",
                   fontsize=11,fontweight="bold",pad=5)
    decorate(ax_r, ext, cfg["scale"])
    ax_r.set_xlabel("Longitude",fontsize=8); ax_r.set_yticklabels([])

    fig.text(0.5,0.918,f"GNN Uncertainty (Softmax Entropy) - {FRAC_TITLES[frac]}",
             ha="center",fontsize=13,fontweight="bold")
    fig.text(0.5,0.899,f"Hypothesis H5 | {cfg['label']} | {cfg['paper']}",
             ha="center",fontsize=8.5,style="italic",color="#555")
    add_cartouche(fig,f"GNN Uncertainty - {FRAC_TITLES[frac]}",
                  source="GNN Pacaraima v2 | probs_full.npz",
                  scale_str=cfg["label"],crs_str="SIRGAS 2000")
    save_map(fig, out_dir/f"15_m03_{cfg['id']}_{frac}{alpha_tag}.png")
    plt.close(fig)


def map_m04(frac, all_preds_2d, labels_2d,
            full_ext, cfg, out_dir, class_alpha=0.25, alpha_tag=""):
    """M04 - Dangerous vs conservative errors."""
    ext = extent_from_cfg(cfg, "4p")
    lbl_c  = crop(labels_2d.get(frac), full_ext, ext)[0]
    if lbl_c is None: return

    fig = plt.figure(figsize=cfg["fig_4p"], facecolor="white")
    gs  = fig.add_gridspec(2, 2, left=0.05, right=0.97,
                           top=0.90, bottom=0.10, hspace=0.11, wspace=0.05)
    m04  = {}
    ax_tl = None

    for idx_m, mname in enumerate(MODEL_ORDER):
        ax  = fig.add_subplot(gs[idx_m//2, idx_m%2])
        if idx_m == 0:
            ax_tl = ax
        p2d = crop(all_preds_2d.get(mname,{}).get(frac), full_ext, ext)[0]
        render_satellite(ax, ext)

        if p2d is not None and lbl_c is not None:
            err = np.where(p2d==lbl_c, 0.0,
                  np.where(p2d<lbl_c, -1.0, 1.0)).astype(np.float32)
            err[np.isnan(p2d)|np.isnan(lbl_c)] = np.nan

            ok  = np.ma.masked_where(err != 0, np.ma.masked_invalid(p2d))
            ax.imshow(ok, cmap=CLASS_CMAP, norm=CLASS_NORM,
                      extent=ext, alpha=0.30, interpolation="nearest",
                      aspect="auto", zorder=4)

            per = np.ma.masked_where(err!=-1, np.ones_like(err))
            con = np.ma.masked_where(err!= 1, np.ones_like(err))
            ax.imshow(per, cmap=ListedColormap(["#b2182b"]),
                      extent=ext, alpha=0.90, interpolation="nearest",
                      aspect="auto", zorder=5)
            ax.imshow(con, cmap=ListedColormap(["#2166ac"]),
                      extent=ext, alpha=0.90, interpolation="nearest",
                      aspect="auto", zorder=5)

            nv = int((~np.isnan(err)).sum())
            np_ = int((err==-1).sum()); nc = int((err==1).sum())
            na  = int((err== 0).sum())
            m04[mname] = {"pct_correct":round(na/nv*100,3),
                          "pct_dangerous":round(np_/nv*100,3),
                          "pct_conservative":round(nc/nv*100,3)}
            ax.set_title(f"{mname}  -  Danger.:{np_/nv*100:.2f}%  |  "
                         f"Conserv.:{nc/nv*100:.2f}%",
                         fontsize=9, fontweight="bold",
                         color=MODEL_COLORS.get(mname,"#333"), pad=5)
            stats_box(ax, f"Correct:     {na/nv*100:5.1f}%\n"
                          f"Dangerous:   {np_/nv*100:5.2f}%\n"
                          f"Conserv.:    {nc/nv*100:5.2f}%")
        else:
            ax.set_title(f"{mname} - no data",fontsize=9,color="gray")

        ptchs = [mpatches.Patch(facecolor="#b2182b",edgecolor="#333",lw=0.5,
                                label="Dangerous Error - underestimated"),
                 mpatches.Patch(facecolor="#d9d9d9",edgecolor="#333",lw=0.5,
                                label="Correct"),
                 mpatches.Patch(facecolor="#2166ac",edgecolor="#333",lw=0.5,
                                label="Conservative Error - overestimated")]
        leg = ax.legend(handles=ptchs,loc="lower right",fontsize=7,
                        title="Error Type",title_fontsize=8,
                        framealpha=0.92,edgecolor="#2c3e50",fancybox=False)
        leg.get_frame().set_linewidth(0.7)
        decorate(ax, ext, cfg["scale"])
        ax.set_ylabel("Latitude" if idx_m in [0,2] else "", fontsize=8)
        ax.set_xlabel("Longitude" if idx_m in [2,3] else "", fontsize=8)
        if idx_m in [0,1]: ax.set_xticklabels([])
        if idx_m in [1,3]: ax.set_yticklabels([])

    fig.text(0.5, 0.945, f"Dangerous vs Conservative Errors - {FRAC_TITLES[frac]}",
             ha="center", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.924,
             f"Red=underestimated restriction (risk)  |  "
             f"Blue=overestimated (cost) | {cfg['label']}",
             ha="center", fontsize=9, style="italic", color="#555")
    if ax_tl is not None:
        add_inset(ax_tl, ext)
    add_cartouche(fig,f"Error Analysis - {FRAC_TITLES[frac]}",
                  subtitle="Hypothesis H2: GNN has lower dangerous error rate",
                  source="GNN Pacaraima v2 | GEOINT Lab",
                  scale_str=cfg["label"],crs_str="SIRGAS 2000")
    save_map(fig, out_dir/f"15_m04_{cfg['id']}_{frac}{alpha_tag}.png")
    plt.close(fig)
    return m04


def map_m05(frac, all_preds_2d, iso_2d_dict,
            full_ext, cfg, out_dir, class_alpha=0.25, alpha_tag=""):
    """M05 - Spatial coherence: isolated pixels."""
    ext   = extent_from_cfg(cfg, "2p")

    fig = plt.figure(figsize=cfg["fig_2p"], facecolor="white")
    gs  = fig.add_gridspec(1,2,left=0.05,right=0.97,
                           top=0.87,bottom=0.10,wspace=0.05)

    iso_colors = {"GNN": "#e74c3c", "RF": "#f39c12"}
    ax_first = None
    for idx_m, mname in enumerate(["GNN", "RF"]):
        ax  = fig.add_subplot(gs[0, idx_m])
        if idx_m == 0:
            ax_first = ax
        p2d = crop(all_preds_2d.get(mname, {}).get(frac), full_ext, ext)[0]
        iso = crop(iso_2d_dict.get(mname, {}).get(frac), full_ext, ext)[0]
        render_satellite(ax, ext)
        render_class(ax, p2d, ext, alpha=class_alpha)
        if iso is not None:
            ax.imshow(np.ma.masked_where(iso < 0.5, iso),
                      cmap=ListedColormap([iso_colors[mname]]),
                      extent=ext, alpha=0.92, interpolation="nearest",
                      aspect="auto", zorder=6)
            n_iso = int(np.nansum(iso))
            pct   = float(np.nansum(iso > 0.5) / max(np.isfinite(iso).sum(), 1) * 100)
            stats_box(ax, f"Isolated pixels: {n_iso:,}\n"
                          f"% of total: {pct:.3f}%\n"
                          "(>50% diff. class neighbors)")

        add_class_legend(ax, extra_handles=[
            mpatches.Patch(facecolor=iso_colors[mname], edgecolor="#333", lw=0.5,
                           label=f"Isolated pixel {mname}")])
        ax.set_title(f"{mname} - Isolated Pixels", fontsize=11,
                     fontweight="bold", color=MODEL_COLORS.get(mname, "#333"), pad=5)
        decorate(ax, ext, cfg["scale"])
        ax.set_xlabel("Longitude", fontsize=8)
        if idx_m == 0:
            ax.set_ylabel("Latitude", fontsize=8)
        else:
            ax.set_yticklabels([])

    if ax_first is not None:
        add_inset(ax_first, ext)
    fig.text(0.5,0.918,f"Spatial Coherence - {FRAC_TITLES[frac]}",
             ha="center",fontsize=13,fontweight="bold")
    fig.text(0.5,0.899,
             f"Hypothesis H2: GNN produces more coherent classifications | {cfg['label']}",
             ha="center",fontsize=8.5,style="italic",color="#555")
    add_cartouche(fig,f"Spatial Coherence - {FRAC_TITLES[frac]}",
                  source="GNN Pacaraima v2 | GEOINT Lab",
                  scale_str=cfg["label"],crs_str="SIRGAS 2000")
    save_map(fig, out_dir/f"15_m05_{cfg['id']}_{frac}{alpha_tag}.png")
    plt.close(fig)


def map_m06(frac, all_preds_2d, labels_2d, ndvi_2d, trans_2d,
            full_ext, cfg, out_dir, thr,
            class_alpha=0.25, alpha_tag=""):
    """M06 - NDVI transition zones."""
    ext    = extent_from_cfg(cfg, "2p")
    ndvi_c = crop(ndvi_2d,     full_ext, ext)[0]
    trans_c= crop(trans_2d,    full_ext, ext)[0]
    lbl_c  = crop(labels_2d.get(frac), full_ext, ext)[0]
    gnn_c  = crop(all_preds_2d.get("GNN",{}).get(frac), full_ext, ext)[0]
    rf_c   = crop(all_preds_2d.get("RF", {}).get(frac), full_ext, ext)[0]

    NDVI_CMAP = LinearSegmentedColormap.from_list("ndvi_pub", [
        (0.00,"#8B4513"),(0.15,"#DEB887"),(0.30,"#F5DEB3"),
        (0.50,"#9ACD32"),(0.75,"#228B22"),(1.00,"#006400")])

    fig = plt.figure(figsize=cfg["fig_2p"], facecolor="white")
    gs  = fig.add_gridspec(1,2,left=0.05,right=0.95,
                           top=0.87,bottom=0.10,wspace=0.07)

    ax_l = fig.add_subplot(gs[0,0])
    render_satellite(ax_l, ext)
    if ndvi_c is not None:
        im_n = ax_l.imshow(np.ma.masked_where(np.isnan(ndvi_c),ndvi_c),
                           cmap=NDVI_CMAP, vmin=0, vmax=1, extent=ext,
                           alpha=0.62, interpolation="bilinear",
                           aspect="auto", zorder=4)
        fig.colorbar(im_n, ax=ax_l, fraction=0.028, pad=0.015,
                     label="NDVI", shrink=0.85)
        if trans_c is not None:
            ax_l.contour(np.flip(trans_c, 0), levels=[0.5], extent=ext,
                         colors=["white"], linewidths=2.0, zorder=7)
        ax_l.contour(np.flip(np.where(np.isnan(ndvi_c), 0, ndvi_c), 0),
                     levels=[thr-MARGIN, thr, thr+MARGIN], extent=ext,
                     colors=["#00bcd4", "#ffffff", "#00bcd4"],
                     linewidths=[1.0, 1.8, 1.0], linestyles=["--", "-", "--"],
                     alpha=0.9, zorder=8)
    ax_l.set_title(f"NDVI - Transition Zone (thr={thr})", fontsize=11,
                   fontweight="bold", pad=5)
    decorate(ax_l, ext, cfg["scale"])
    ax_l.set_xlabel("Longitude", fontsize=8); ax_l.set_ylabel("Latitude", fontsize=8)
    add_inset(ax_l, ext)

    ax_r = fig.add_subplot(gs[0,1])
    render_satellite(ax_r, ext)
    m06_stats = {}

    if lbl_c is not None and gnn_c is not None and trans_c is not None:
        in_t = trans_c > 0.5
        cat  = np.full_like(gnn_c, np.nan)
        if rf_c is not None:
            gok = in_t & (gnn_c==lbl_c); rok = in_t & (rf_c==lbl_c)
            cat[in_t & gok &  rok] = 2.0
            cat[in_t & gok & ~rok] = 1.0
            cat[in_t &~gok &  rok] = 3.0
            cat[in_t &~gok & ~rok] = 4.0
        else:
            cat[in_t & (gnn_c==lbl_c)] = 2.0
            cat[in_t & (gnn_c!=lbl_c)] = 4.0

        PERF_CMAP = ListedColormap(["#1565C0","#2e7d32","#e65100","#b71c1c"])
        PERF_NORM = BoundaryNorm([0.5,1.5,2.5,3.5,4.5], ncolors=4)
        ax_r.imshow(np.ma.masked_where(np.isnan(cat),cat),
                    cmap=PERF_CMAP, norm=PERF_NORM,
                    extent=ext, alpha=0.82, interpolation="nearest",
                    aspect="auto", zorder=5)
        ax_r.contour(np.flip(trans_c,0), levels=[0.5], extent=ext,
                     colors=["white"], linewidths=1.8, zorder=7)

        nt = int(in_t.sum())
        n1=int((cat[in_t]==1).sum()); n2=int((cat[in_t]==2).sum())
        n3=int((cat[in_t]==3).sum()); n4=int((cat[in_t]==4).sum())
        stats_box(ax_r,
            f"In zone ({nt:,} nodes):\n"
            f" GNN ok, RF fail:  {n1:,} ({n1/max(nt,1)*100:.1f}%)\n"
            f" Both ok:          {n2:,} ({n2/max(nt,1)*100:.1f}%)\n"
            f" RF ok, GNN fail:  {n3:,} ({n3/max(nt,1)*100:.1f}%)\n"
            f" Both fail:        {n4:,} ({n4/max(nt,1)*100:.1f}%)\n"
            f" GNN advantage:  {(n1-n3)/max(nt,1)*100:+.1f}%")
        m06_stats = {"n_trans":nt,"gnn_ok_rf_fail":n1,"both_ok":n2,
                     "rf_ok_gnn_fail":n3,"both_fail":n4,
                     "gnn_advantage_pct":round((n1-n3)/max(nt,1)*100,2)}
        perf_ptchs = [
            mpatches.Patch(facecolor="#1565C0",lw=0.5,edgecolor="#333",label="GNN ok, RF fails"),
            mpatches.Patch(facecolor="#2e7d32",lw=0.5,edgecolor="#333",label="Both ok"),
            mpatches.Patch(facecolor="#e65100",lw=0.5,edgecolor="#333",label="RF ok, GNN fails"),
            mpatches.Patch(facecolor="#b71c1c",lw=0.5,edgecolor="#333",label="Both fail"),
        ]
        leg = ax_r.legend(handles=perf_ptchs,loc="lower right",fontsize=7,
                          title="In Transition Zone",title_fontsize=8,
                          framealpha=0.92,edgecolor="#2c3e50",fancybox=False)
        leg.get_frame().set_linewidth(0.7)

    ax_r.set_title("GNN vs RF in NDVI Transition Zone",
                   fontsize=11,fontweight="bold",pad=5)
    decorate(ax_r, ext, cfg["scale"])
    ax_r.set_xlabel("Longitude",fontsize=8); ax_r.set_yticklabels([])

    fig.text(0.5,0.918,f"Performance in Transition Zones - {FRAC_TITLES[frac]}",
             ha="center",fontsize=13,fontweight="bold")
    fig.text(0.5,0.899,
             f"Hypothesis H1: GNN outperforms RF in ambiguous DAMEPLAN regions | {cfg['label']}",
             ha="center",fontsize=8.5,style="italic",color="#555")
    add_cartouche(fig,f"NDVI Transition Zones - {FRAC_TITLES[frac]}",
                  source="GNN Pacaraima v2 | GEOINT Lab",
                  scale_str=cfg["label"],crs_str="SIRGAS 2000")
    save_map(fig, out_dir/f"15_m06_{cfg['id']}_{frac}{alpha_tag}.png")
    plt.close(fig)
    return m06_stats


def map_m07(all_preds_2d, elev_m_2d,
            full_ext, cfg, out_dir, transform,
            class_alpha=0.25, alpha_tag=""):
    """M07 - Topographic profile with classification (E-W transect)."""
    ext   = extent_from_cfg(cfg, "2p")
    elv_c = crop(elev_m_2d,    full_ext, ext)[0]
    if elv_c is None: return

    fig = plt.figure(figsize=cfg["fig_pro"], facecolor="white")
    gs  = fig.add_gridspec(2,1,left=0.06,right=0.97,
                           top=0.88,bottom=0.09,hspace=0.0,
                           height_ratios=[1.5,1.0])

    ax_map = fig.add_subplot(gs[0,0])
    render_satellite(ax_map, ext)
    if elv_c is not None:
        im_e = ax_map.imshow(np.ma.masked_where(np.isnan(elv_c),elv_c),
                             cmap="terrain", alpha=0.42, extent=ext,
                             interpolation="bilinear", aspect="auto", zorder=4)
        fig.colorbar(im_e, ax=ax_map, fraction=0.02, pad=0.01,
                     label="Elevation (m)", shrink=0.8)

    H_c = elv_c.shape[0] if elv_c is not None else 100
    W_c = elv_c.shape[1] if elv_c is not None else 100
    lon_min,lon_max,lat_min,lat_max = ext
    lat_ew = (lat_min + lat_max) / 2
    lon_ns = (lon_min + lon_max) / 2
    ax_map.axhline(y=lat_ew,color="#FFD600",lw=1.8,ls="--",
                   label=f"E-W Transect (lat={lat_ew:.4f}\u00b0)")
    ax_map.axvline(x=lon_ns,color="#00E5FF",lw=1.8,ls="--",
                   label=f"N-S Transect (lon={lon_ns:.4f}\u00b0)")
    ax_map.legend(loc="upper right",fontsize=7,framealpha=0.90,
                  edgecolor="#2c3e50",fancybox=False)
    ax_map.set_title("Transect Locations", fontsize=9, pad=4)
    decorate(ax_map, ext, cfg["scale"])
    ax_map.set_xticklabels([]); ax_map.set_ylabel("Latitude", fontsize=8)
    add_inset(ax_map, ext)

    ax_p = fig.add_subplot(gs[1, 0])
    lons_arr = np.linspace(lon_min, lon_max, W_c)
    y0 = 0.0; y_gap = 1.2

    if elv_c is not None:
        mid_row = H_c // 2
        elev_ew = elv_c[mid_row, :]
        valid   = ~np.isnan(elev_ew)
        ax2 = ax_p.twinx()
        ax2.fill_between(lons_arr[valid],
                         float(np.nanmin(elev_ew)), elev_ew[valid],
                         alpha=0.15, color="#8B7355", zorder=2)
        ax2.plot(lons_arr[valid], elev_ew[valid],
                 color="#6B5B45", lw=1.4, alpha=0.6, zorder=3)
        ax2.set_ylabel("Elevation (m)", fontsize=7.5, color="#6B5B45")
        ax2.tick_params(axis="y", labelcolor="#6B5B45", labelsize=6.5)

    frac_p = "a_pe"
    y_total_est = y_gap * len(MODEL_ORDER) + 0.2
    for mname in MODEL_ORDER:
        p2d_full = all_preds_2d.get(mname, {}).get(frac_p)
        pc = crop(p2d_full, full_ext, ext)[0]
        if pc is None:
            y0 += y_gap; continue
        mid = pc.shape[0] // 2
        pew = pc[mid, :]
        for cls, col in CLASS_COLORS.items():
            m = ~np.isnan(pew) & (pew == cls)
            if m.sum() > 0:
                ax_p.fill_between(lons_arr, y0, y0 + 1.0, where=m,
                                  color=col, alpha=0.82, step="mid", zorder=4)
        y_frac = (y0 + 0.42) / max(y_total_est, 0.1)
        ax_p.annotate(mname,
                      xy=(1.0, y_frac), xycoords="axes fraction",
                      xytext=(4, 0), textcoords="offset points",
                      fontsize=8, fontweight="bold",
                      color=MODEL_COLORS.get(mname, "#333"),
                      va="center", ha="left", clip_on=False)
        y0 += y_gap

    for yi in np.arange(y_gap, y0 - 0.1, y_gap):
        ax_p.axhline(y=yi, color="#ccc", lw=0.5, zorder=3)
    ax_p.set_xlim(lon_min, lon_max)
    ax_p.set_ylim(-0.2, y0)
    ax_p.set_yticks([])
    ax_p.set_xlabel("Longitude - E-W Transect", fontsize=8)
    ax_p.set_title(f"Classification along transect - {FRAC_TITLES[frac_p]} Fraction",
                   fontsize=9, pad=4)
    ax_p.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v,p: f"{abs(v):.3f}\u00b0{'W' if v<0 else 'E'}"))
    ax_p.tick_params(axis="x",labelsize=7)
    ax_p.grid(True,axis="x",alpha=0.3,lw=0.5)
    ax_p.legend(handles=[mpatches.Patch(facecolor=CLASS_COLORS[i],
                edgecolor="#333",lw=0.5,label=CLASS_NAMES[i])
                for i in range(3)], loc="lower right",
                fontsize=7,title=f"Restriction {FRAC_TITLES[frac_p]}",
                title_fontsize=7.5,framealpha=0.90,fancybox=False)
    add_neat_line(ax_p)

    fig.text(0.5,0.924,"Topographic Profile - Movement Restriction",
             ha="center",fontsize=12,fontweight="bold")
    fig.text(0.5,0.906,
             f"E-W Transect | Pacaraima, Roraima | {cfg['label']} | {cfg['paper']}",
             ha="center",fontsize=8.5,style="italic",color="#555")
    add_cartouche(fig,"Topographic Profile - Movement Restriction",
                  source="GNN Pacaraima v2 | GEOINT Lab",
                  scale_str=cfg["label"],crs_str="SIRGAS 2000")
    save_map(fig, out_dir/f"15_m07_{cfg['id']}{alpha_tag}.png")
    plt.close(fig)


# --- Main ---
def main():
    start = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    assert GRAPH_PATH is not None and GRAPH_PATH.exists(), (
        f"Graph not found: {GRAPH_PATH}"
    )

    # -- 1. Graph --
    log(f"Loading graph: {GRAPH_PATH.name} ...")
    data = torch.load(str(GRAPH_PATH), map_location="cpu", weights_only=False)
    dem  = data["dem"]
    feat = dem.x.float().numpy()
    N    = feat.shape[0]
    log(f"  N={N:,} nodes, {feat.shape[1]} features")

    labels_true = {}
    for f in FRACTIONS:
        if hasattr(dem, f"y_{f}"):
            labels_true[f] = _norm0(getattr(dem, f"y_{f}").numpy().astype(np.int32))

    raster_meta = getattr(data, "raster_meta", None)
    assert raster_meta, "raster_meta not found!"
    pos = dem.pos.numpy()

    H, W  = raster_meta["dem_shape"]
    crs   = raster_meta.get("crs", "SIRGAS 2000")
    t_raw = raster_meta["dem_transform"]

    try:
        from rasterio.transform import Affine
        transform = Affine(*t_raw[:6]) if isinstance(t_raw,(list,tuple)) \
                    else Affine(t_raw.a,t_raw.b,t_raw.c,t_raw.d,t_raw.e,t_raw.f)
    except ImportError:
        transform = t_raw

    full_ext = get_full_extent(transform, H, W)
    log(f"  Raster={H}x{W} | Extent={[round(v,4) for v in full_ext]}")

    ei      = data[("dem","adjacent_to","dem")].edge_index.numpy()
    src_ei  = ei[0]; dst_ei = ei[1]
    del data, dem; gc.collect()

    # -- 2. Features --
    elev_flat = feat[:, IDX_ELEV]
    ndvi_flat = feat[:, IDX_NDVI]
    tpi_flat  = feat[:, IDX_TPI]

    # -- 3. Rasters (reconstruct + fill nearest) --
    log("Reconstructing 2D rasters + fill nearest-neighbor ...")

    def mk(vals, fill=np.nan):
        r = flat_to_2d(vals.astype(np.float32), pos, H, W, transform, fill=fill)
        return fill_nearest(r)

    elev_2d = mk(elev_flat)
    e_min, e_max = float(np.nanmin(elev_2d)), float(np.nanmax(elev_2d))

    if 0.0 <= e_min and e_max <= 1.0:
        elev_m = elev_2d * (1400 - 800) + 800
    elif e_max < 10.0:
        elev_m = 800 + (elev_2d - e_min) / max(e_max - e_min, 1e-8) * 600
    else:
        elev_m = elev_2d
    log(f"  Elevation: {float(np.nanmin(elev_m)):.0f}m - {float(np.nanmax(elev_m)):.0f}m")

    ndvi_2d = mk(ndvi_flat)

    # -- 4. Background --
    log("Background: real satellite imagery (Esri World Imagery via contextily)")

    # -- 5. Predictions --
    log("Loading predictions ...")
    preds_gnn  = load_preds(GNN_PREDS,  FRACTIONS)
    preds_rf   = load_preds(RF_PREDS,   FRACTIONS)
    preds_mlp  = load_preds(MLP_PREDS,  FRACTIONS)
    preds_rule = load_preds(RULE_PREDS, FRACTIONS)
    has_probs  = GNN_PROBS is not None and Path(str(GNN_PROBS)).exists()
    probs_gnn  = load_probs(GNN_PROBS, FRACTIONS) if has_probs else {}
    log(f"  GNN={len(preds_gnn)} RF={len(preds_rf)} "
        f"MLP={len(preds_mlp)} Rule={len(preds_rule)} "
        f"Probs={'yes' if probs_gnn else 'no'}")

    log("Reconstructing prediction rasters (fill nearest) ...")
    all_preds_2d = {}
    for mname, pdict in [("GNN",preds_gnn),("RF",preds_rf),
                          ("MLP",preds_mlp),("Rule",preds_rule)]:
        all_preds_2d[mname] = {}
        for frac, arr in pdict.items():
            r = flat_to_2d(arr.astype(np.float32), pos, H, W, transform, fill=np.nan)
            all_preds_2d[mname][frac] = fill_nearest(r)

    labels_2d = {}
    for frac, arr in labels_true.items():
        r = flat_to_2d(arr.astype(np.float32), pos, H, W, transform, fill=np.nan)
        labels_2d[frac] = fill_nearest(r)

    trans_2d_dict = {}
    for frac in FRACTIONS:
        thr = NDVI_THRESH[frac]
        mask = (np.abs(ndvi_flat - thr) <= MARGIN).astype(np.float32)
        trans_2d_dict[frac] = mk(mask, fill=0)

    log("Computing spatial coherence ...")
    degree = np.zeros(N, dtype=np.int32)
    np.add.at(degree, dst_ei, 1)
    iso_2d_dict = {}
    for mname, pdict in [("GNN",preds_gnn),("RF",preds_rf)]:
        iso_2d_dict[mname] = {}
        for frac, arr in pdict.items():
            edg = (arr[src_ei] != arr[dst_ei]).astype(np.int32)
            cnt = np.zeros(N, dtype=np.int32)
            np.add.at(cnt, dst_ei, edg)
            vd  = degree > 0
            fd  = np.zeros(N, dtype=np.float32)
            fd[vd] = cnt[vd] / degree[vd]
            iso = (vd & (fd > 0.5)).astype(np.float32)
            r   = flat_to_2d(iso, pos, H, W, transform, fill=0)
            iso_2d_dict[mname][frac] = fill_nearest(r)

    entr_2d_dict = {}
    entr_stats   = {}
    if probs_gnn:
        log("Computing softmax entropy ...")
        for frac, probs in probs_gnn.items():
            ef = entropy_norm(probs)
            entr_2d_dict[frac] = mk(ef)
            entr_stats[frac]   = {
                "p75": float(np.nanpercentile(ef, 75)),
                "p90": float(np.nanpercentile(ef, 90)),
            }

    gc.collect()

    # -- 6. Generate maps for each scale --
    total_maps = 0

    for cfg in SCALE_CONFIGS:
        sid = cfg["id"]
        log(f"\n{'='*60}")
        log(f"Generating maps - {cfg['label']} ({cfg['paper']}) ...")

        for ca in CLASS_ALPHAS:
            atag = f"_a{int(ca*100)}"
            akw = dict(class_alpha=ca, alpha_tag=atag)

            for frac in FRACTIONS:
                map_m01(frac, all_preds_2d, full_ext, cfg, OUT_DIR, **akw)
                total_maps += 1

            disc_r = {}
            for frac in FRACTIONS:
                ds = map_m02(frac, all_preds_2d, full_ext, cfg, OUT_DIR, **akw)
                if ds: disc_r[frac] = ds
                total_maps += 1
            results.setdefault("M02",{})[sid] = disc_r

            if probs_gnn:
                for frac in FRACTIONS:
                    if frac not in entr_2d_dict: continue
                    map_m03(frac, entr_2d_dict[frac],
                            all_preds_2d.get("GNN",{}).get(frac),
                            trans_2d_dict.get(frac),
                            full_ext, cfg, OUT_DIR,
                            entr_stats[frac]["p90"],
                            entr_stats[frac]["p75"],
                            NDVI_THRESH[frac], **akw)
                    total_maps += 1

            for frac in FRACTIONS:
                map_m04(frac, all_preds_2d, labels_2d,
                        full_ext, cfg, OUT_DIR, **akw)
                total_maps += 1

            for frac in FRACTIONS:
                map_m05(frac, all_preds_2d, iso_2d_dict,
                        full_ext, cfg, OUT_DIR, **akw)
                total_maps += 1

            m06_r = {}
            for frac in FRACTIONS:
                if frac not in labels_2d: continue
                st = map_m06(frac, all_preds_2d, labels_2d, ndvi_2d,
                             trans_2d_dict.get(frac),
                             full_ext, cfg, OUT_DIR,
                             NDVI_THRESH[frac], **akw)
                if st: m06_r[frac] = st
                total_maps += 1
            results.setdefault("M06",{})[sid] = m06_r

            map_m07(all_preds_2d, elev_m,
                    full_ext, cfg, OUT_DIR, transform, **akw)
            total_maps += 1

    # -- 7. Summary JSON --
    elapsed = time.perf_counter() - start
    results["meta"] = {
        "timestamp": datetime.now().isoformat(),
        "n_nos": N, "raster_shape": [H,W], "crs": str(crs),
        "fracoes": FRACTIONS, "has_probs": bool(probs_gnn),
        "dpi": PUBLICATION_DPI, "total_mapas": total_maps,
        "escalas": [c["label"] for c in SCALE_CONFIGS],
        "elapsed_s": round(elapsed,1),
    }
    with open(str(OUT_DIR/"15_geoespacial_resumo.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total_mb = sum(p.stat().st_size for p in OUT_DIR.glob("*.png")) / 1_048_576
    print("\n" + "="*65)
    print("SCRIPT 15 - GEOSPATIAL ANALYSIS COMPLETE")
    print(f"  DPI: {PUBLICATION_DPI} | Scales: "
          f"{', '.join(c['label'] for c in SCALE_CONFIGS)}")
    print(f"  Total maps generated: {total_maps}")
    print(f"  Total size:           {total_mb:.0f} MB")
    print(f"  Time:                 {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Output:               {OUT_DIR}")
    print("="*65)


if __name__ == "__main__":
    main()
