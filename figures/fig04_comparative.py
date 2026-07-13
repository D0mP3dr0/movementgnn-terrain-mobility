"""Figure 4 -- Cartographic comparison MovementGNN v2 vs baselines (S2 1:50,000).

Three PNG plates with the SAME tactical S2 1:50,000 crop, centered at
(4.59310 N, -61.11385 W):

  * fig04a_terrain.png   -- S2 natural color (B4/B3/B2) without overlays
  * fig04b_dismounted.png -- 2x2 mosaic: GNN | Rule-Based | RF | MLP (dismounted)
  * fig04c_armored.png    -- 2x2 mosaic: GNN | Rule-Based | RF | MLP (armored)

Divergence overlay (blue) = pixel-by-pixel difference between each baseline
and the MovementGNN v2 official GeoTIFF.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio.windows import from_bounds

from utils_nato import (FIG_WIDTH_IN, PALETTE, COLAB_RESULTS_LOCAL,
                         GEOTIFF_DIR, RAW_DIR, class_cmap, save_fig,
                         set_mpl_style)

SENTINEL2_TIF = RAW_DIR / "sentinel2" / "pacaraima" / "pacaraima_s2_10m_real_q1q4.tif"
S2_BAND_B02, S2_BAND_B03, S2_BAND_B04 = 1, 2, 3

_GEO = GEOTIFF_DIR / "geotiff" if (GEOTIFF_DIR / "geotiff").exists() else GEOTIFF_DIR

TIF_PATHS = {
    "GNN": {
        "a_pe":       COLAB_RESULTS_LOCAL / "restricao_a_pe_movement_gnn_v2_fix.tif",
        "motorizada": COLAB_RESULTS_LOCAL / "restricao_motorizada_movement_gnn_v2_fix.tif",
        "blindada":   COLAB_RESULTS_LOCAL / "restricao_blindada_movement_gnn_v2_fix.tif",
    },
    "Rule": {
        "a_pe":       _GEO / "rule_based" / "rule_based_a_pe.tif",
        "motorizada": _GEO / "rule_based" / "rule_based_motorizada.tif",
        "blindada":   _GEO / "rule_based" / "rule_based_blindada.tif",
    },
    "RF": {
        "a_pe":       _GEO / "random_forest" / "random_forest_a_pe.tif",
        "motorizada": _GEO / "random_forest" / "random_forest_motorizada.tif",
        "blindada":   _GEO / "random_forest" / "random_forest_blindada.tif",
    },
    "MLP": {
        "a_pe":       _GEO / "mlp" / "mlp_a_pe.tif",
        "motorizada": _GEO / "mlp" / "mlp_motorizada.tif",
        "blindada":   _GEO / "mlp" / "mlp_blindada.tif",
    },
}
MODELS = ["GNN", "Rule", "RF", "MLP"]
MODEL_LABELS = {"GNN": "MovementGNN v2", "Rule": "Rule-Based",
                "RF": "Random Forest", "MLP": "MLP"}
FRACTIONS_FIG = ["a_pe", "motorizada", "blindada"]
FRACTION_LABELS = {"a_pe": "Dismounted", "motorizada": "Motorized",
                   "blindada": "Armored"}

S2_CENTER_LAT, S2_CENTER_LON = 4.59310, -61.11385
S2_SCALE = 50_000
S2_SCALE_TXT = "1:50,000"
S2_PANEL_W_IN, S2_PANEL_H_IN = 6.0, 3.8

DIVERG_COLOR = "#0B55D8"
CLASS_ALPHA = 0.40
DIVERG_ALPHA = 0.70
S2_STRETCH_LO, S2_STRETCH_HI = 2.0, 98.0
S2_GAMMA = 0.95


def _calc_half(scale, map_w_in, map_h_in, center_lat):
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(center_lat))
    return ((scale * map_w_in * 0.0254) / m_per_deg_lon / 2.0,
            (scale * map_h_in * 0.0254) / m_per_deg_lat / 2.0)


_HL, _HLA = _calc_half(S2_SCALE, S2_PANEL_W_IN, S2_PANEL_H_IN, S2_CENTER_LAT)
S2_EXTENT = {
    "lon_min": S2_CENTER_LON - _HL, "lon_max": S2_CENTER_LON + _HL,
    "lat_min": S2_CENTER_LAT - _HLA, "lat_max": S2_CENTER_LAT + _HLA,
}
S2_WIDTH_KM = S2_SCALE * S2_PANEL_W_IN * 0.0254 / 1000.0
S2_HEIGHT_KM = S2_SCALE * S2_PANEL_H_IN * 0.0254 / 1000.0


def _read_tif_window(tif_path, bounds):
    with rasterio.open(tif_path) as src:
        win = from_bounds(bounds["lon_min"], bounds["lat_min"],
                          bounds["lon_max"], bounds["lat_max"],
                          transform=src.transform)
        arr = src.read(1, window=win).astype(np.int16)
        win_tf = src.window_transform(win)
        H, W = arr.shape
        left, top = win_tf.c, win_tf.f
        right = left + win_tf.a * W
        bottom = top + win_tf.e * H
    arr = np.where(arr == 0, -1, arr)
    return arr, (left, right, bottom, top)


def _percentile_stretch(band, mask_valid, lo=S2_STRETCH_LO, hi=S2_STRETCH_HI):
    vals = band[mask_valid]
    if vals.size == 0:
        return np.zeros_like(band, dtype=np.float32)
    lo_v, hi_v = np.percentile(vals, [lo, hi])
    return np.clip((band.astype(np.float32) - lo_v) / max(hi_v - lo_v, 1.0), 0.0, 1.0)


def _build_terrain_rgb_s2(bounds):
    with rasterio.open(SENTINEL2_TIF) as src:
        win = from_bounds(bounds["lon_min"], bounds["lat_min"],
                          bounds["lon_max"], bounds["lat_max"],
                          transform=src.transform)
        red = src.read(S2_BAND_B04, window=win).astype(np.float32)
        green = src.read(S2_BAND_B03, window=win).astype(np.float32)
        blue = src.read(S2_BAND_B02, window=win).astype(np.float32)
        win_tf = src.window_transform(win)
        H, W = red.shape
        left, top = win_tf.c, win_tf.f
        right = left + win_tf.a * W
        bottom = top + win_tf.e * H
    mask = (red > 0) & (green > 0) & (blue > 0)
    r01 = _percentile_stretch(red, mask)
    g01 = _percentile_stretch(green, mask)
    b01 = _percentile_stretch(blue, mask)
    if S2_GAMMA != 1.0:
        r01, g01, b01 = np.power(r01, S2_GAMMA), np.power(g01, S2_GAMMA), np.power(b01, S2_GAMMA)
    rgb = np.dstack([r01, g01, b01])
    rgb[~mask] = 1.0
    return rgb, (left, right, bottom, top)


def _add_scalebar(ax, bounds, lat_center):
    x0, x1 = bounds["lon_min"], bounds["lon_max"]
    y0 = bounds["lat_min"]
    span_x, span_y = x1 - x0, bounds["lat_max"] - y0
    km_per_deg_lon = 111.32 * math.cos(math.radians(lat_center))
    deg_2km = 2.0 / km_per_deg_lon
    sx, sy = x0 + 0.04 * span_x, y0 + 0.05 * span_y
    bar_h = 0.010 * span_y
    ax.add_patch(plt.Rectangle((sx, sy), deg_2km / 2, bar_h,
                                facecolor="black", edgecolor="black", lw=0.5, zorder=9))
    ax.add_patch(plt.Rectangle((sx + deg_2km / 2, sy), deg_2km / 2, bar_h,
                                facecolor="white", edgecolor="black", lw=0.5, zorder=9))
    for val, x, ha in [("0", sx, "left"), ("1", sx + deg_2km / 2, "center"),
                        ("2 km", sx + deg_2km, "right")]:
        ax.text(x, sy + bar_h * 2.2, val, fontsize=6.3, color="white",
                fontweight="bold", ha=ha, va="bottom", zorder=10)


def _add_north_arrow(ax, bounds):
    x1, y1 = bounds["lon_max"], bounds["lat_max"]
    span_x = x1 - bounds["lon_min"]
    span_y = y1 - bounds["lat_min"]
    x = x1 - 0.06 * span_x
    ax.annotate("N", xy=(x, y1 - 0.06 * span_y),
                xytext=(x, y1 - 0.18 * span_y),
                ha="center", va="center", fontsize=8.5, fontweight="bold",
                color="white", zorder=10,
                arrowprops=dict(arrowstyle="-|>", color="white", lw=1.2))


def _build_terrain_fig(terrain_rgb, terrain_extent, bounds):
    fig_w = FIG_WIDTH_IN * 1.35
    fig_h = fig_w * (S2_HEIGHT_KM / S2_WIDTH_KM) * 1.07
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(1, 1, left=0.08, right=0.97, top=0.93, bottom=0.13)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(terrain_rgb, extent=terrain_extent, origin="upper",
              interpolation="bilinear", aspect="auto")
    ax.set_xlim(bounds["lon_min"], bounds["lon_max"])
    ax.set_ylim(bounds["lat_min"], bounds["lat_max"])
    ax.set_xlabel("Longitude (\u00b0)", fontsize=7.8)
    ax.set_ylabel("Latitude (\u00b0)", fontsize=7.8)
    ax.tick_params(labelsize=6.8)
    ax.grid(True, linestyle=":", linewidth=0.3, color="white", alpha=0.5)
    ax.set_title(
        f"Terrain reference \u2014 Sentinel-2 L2A (B4/B3/B2) \u00b7 {S2_SCALE_TXT} \u00b7 "
        f"center ({S2_CENTER_LAT:.4f}\u00b0 N, {abs(S2_CENTER_LON):.4f}\u00b0 W)",
        loc="center", fontsize=9.2, fontweight="bold", pad=4, color=PALETTE["text"])
    lat_center = 0.5 * (bounds["lat_min"] + bounds["lat_max"])
    _add_scalebar(ax, bounds, lat_center)
    _add_north_arrow(ax, bounds)
    fig.text(0.5, 0.015,
             f"Scale {S2_SCALE_TXT} (A3 sheet)   \u00b7   "
             f"CRS WGS 84 (EPSG:4326)   \u00b7   "
             f"Background: Sentinel-2 L2A natural color (10 m)   \u00b7   "
             f"Classes: official MovementGNN v2 GeoTIFFs",
             ha="center", va="bottom", fontsize=6.4,
             color=PALETTE["muted"], style="italic")
    save_fig(fig, "fig04a_terrain")


def _build_mosaic_fig(frac, frac_label, terrain_rgb, terrain_extent,
                      tif_arrays, bounds, out_name):
    cmap_class = class_cmap()
    cmap_div = ListedColormap([DIVERG_COLOR])
    panel_w = FIG_WIDTH_IN * 0.68
    panel_h = panel_w * (S2_HEIGHT_KM / S2_WIDTH_KM)
    fig_w = 2 * panel_w + 0.6
    fig_h = 2 * panel_h + 2.25
    fig = plt.figure(figsize=(fig_w, fig_h))
    top_margin = 1.0 - 1.05 / fig_h
    bot_margin = 1.05 / fig_h
    gs = fig.add_gridspec(2, 2, wspace=0.08, hspace=0.18,
                          left=0.07, right=0.97, top=top_margin, bottom=bot_margin)
    lat_center = 0.5 * (bounds["lat_min"] + bounds["lat_max"])
    gnn_arr = tif_arrays["GNN"]

    for idx, m in enumerate(MODELS):
        r_row, r_col = divmod(idx, 2)
        ax = fig.add_subplot(gs[r_row, r_col])
        ax.imshow(terrain_rgb, extent=terrain_extent, origin="upper",
                  interpolation="bilinear", aspect="auto")
        arr_m = tif_arrays[m]
        class_masked = np.where(arr_m > 0, arr_m, np.nan)
        ax.imshow(class_masked, extent=terrain_extent, cmap=cmap_class,
                  vmin=1, vmax=3, origin="upper", aspect="auto",
                  interpolation="nearest", alpha=CLASS_ALPHA)
        if m != "GNN":
            mask_valid = (arr_m > 0) & (gnn_arr > 0)
            mask_div = mask_valid & (arr_m != gnn_arr)
            divergence = np.where(mask_div, 1.0, np.nan)
            ax.imshow(divergence, extent=terrain_extent, cmap=cmap_div,
                      vmin=0, vmax=1, origin="upper", aspect="auto",
                      interpolation="nearest", alpha=DIVERG_ALPHA)
            pct = 100.0 * mask_div.sum() / max(mask_valid.sum(), 1)
        else:
            pct = 0.0
        ax.set_xlim(bounds["lon_min"], bounds["lon_max"])
        ax.set_ylim(bounds["lat_min"], bounds["lat_max"])
        if r_col == 0:
            ax.set_ylabel("Latitude (\u00b0)", fontsize=7.0)
            ax.tick_params(labelsize=6.2)
        else:
            ax.tick_params(labelleft=False, labelsize=6.2)
        if r_row == 1:
            ax.set_xlabel("Longitude (\u00b0)", fontsize=7.0)
        else:
            ax.tick_params(labelbottom=False)
        ax.grid(True, linestyle=":", linewidth=0.25, color="white", alpha=0.4)
        if m == "GNN":
            ax.set_title(MODEL_LABELS[m], fontsize=9.2, fontweight="bold",
                         loc="center", pad=3, color=PALETTE["highlight"])
        else:
            ax.set_title(f"{MODEL_LABELS[m]}  \u00b7  \u0394 GNN = {pct:.1f} %",
                         fontsize=9.2, fontweight="bold", loc="center", pad=3,
                         color=PALETTE["text"])
        if idx == 0:
            _add_scalebar(ax, bounds, lat_center)
        _add_north_arrow(ax, bounds)
        for sp in ax.spines.values():
            sp.set_color(PALETTE["muted"])
            sp.set_linewidth(0.6)

    title_y = 1.0 - 0.30 / fig_h
    sub_y1 = 1.0 - 0.55 / fig_h
    sub_y2 = 1.0 - 0.78 / fig_h
    fig.suptitle(
        f"Cartographic comparison {S2_SCALE_TXT} \u2014 {frac_label} fraction \u00b7 "
        f"MovementGNN v2 vs baselines",
        x=0.5, y=title_y, ha="center", fontsize=10.5, fontweight="bold",
        color=PALETTE["text"])
    fig.text(0.5, sub_y1,
             f"Background: Sentinel-2 L2A (B4/B3/B2) \u00b7 classes (Go/SlwGo/NoGo) from "
             f"official GeoTIFFs at \u03b1 40% \u00b7 blue (\u03b1 70%) = pixel-wise "
             f"divergence vs MovementGNN v2",
             ha="center", fontsize=7.2, style="italic", color=PALETTE["muted"])
    fig.text(0.5, sub_y2,
             f"Crop S2: {S2_WIDTH_KM:.2f} \u00d7 {S2_HEIGHT_KM:.2f} km \u00b7 "
             f"center ({S2_CENTER_LAT:.4f}\u00b0 N, {abs(S2_CENTER_LON):.4f}\u00b0 W) \u00b7 "
             f"CRS WGS 84 (EPSG:4326)",
             ha="center", fontsize=6.8, style="italic", color=PALETTE["muted"])

    legend_handles = [
        Patch(facecolor="#2E9B4A", edgecolor="black", alpha=CLASS_ALPHA,
              label="Go (Unrestricted, \u03b1 40%)"),
        Patch(facecolor="#F0C432", edgecolor="black", alpha=CLASS_ALPHA,
              label="Slow Go (Restricted, \u03b1 40%)"),
        Patch(facecolor="#B8351F", edgecolor="black", alpha=CLASS_ALPHA,
              label="No Go (Sev. Restricted, \u03b1 40%)"),
        Patch(facecolor=DIVERG_COLOR, edgecolor="black", alpha=DIVERG_ALPHA,
              label="Divergence vs MovementGNN v2 (\u03b1 70%)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.20 / fig_h), fontsize=7.3, frameon=False)
    save_fig(fig, out_name)


def build():
    set_mpl_style()
    print(f"[fig4] S2 crop {S2_SCALE_TXT} = {S2_WIDTH_KM:.2f} x {S2_HEIGHT_KM:.2f} km "
          f"centered at ({S2_CENTER_LAT}, {S2_CENTER_LON})")

    print("[fig4] reading official GeoTIFFs (S2 window)...")
    tif_arrays = {f: {} for f in FRACTIONS_FIG}
    tif_extent = None
    for m in MODELS:
        for f in FRACTIONS_FIG:
            p = TIF_PATHS[m][f]
            if not p.exists():
                print(f"  [WARN] TIFF not found: {p} -- skipping model {m}/{f}")
                continue
            arr, ext = _read_tif_window(p, S2_EXTENT)
            tif_arrays[f][m] = arr
            if tif_extent is None:
                tif_extent = ext

    print("[fig4] reading Sentinel-2 10 m (natural color B04/B03/B02)...")
    terrain_rgb, terrain_extent = _build_terrain_rgb_s2(S2_EXTENT)

    print("[fig4] saving fig04a (terrain reference)...")
    _build_terrain_fig(terrain_rgb, terrain_extent, S2_EXTENT)

    _OUT_NAMES = {
        "a_pe": "fig04b_dismounted",
        "motorizada": "fig04d_motorized",
        "blindada": "fig04c_armored",
    }
    for frac in FRACTIONS_FIG:
        if len(tif_arrays[frac]) < 4:
            print(f"  [WARN] Not all models available for {frac}, skipping mosaic")
            continue
        label = FRACTION_LABELS[frac]
        out = _OUT_NAMES[frac]
        print(f"[fig4] saving {out} ({label})...")
        _build_mosaic_fig(frac, label, terrain_rgb, terrain_extent,
                          tif_arrays[frac], S2_EXTENT, out)


if __name__ == "__main__":
    build()
