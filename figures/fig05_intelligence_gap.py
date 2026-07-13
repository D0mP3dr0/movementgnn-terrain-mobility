"""Figure 5 -- Intelligence Gap identification via softmax entropy.

NATO terminology: "Intelligence Gap" replaces the Brazilian "ZRN"
(Zona de Reconhecimento Necessario). These are areas where the model's
classification confidence is lowest, requiring additional reconnaissance.

Layout (constrained_layout):
  Row 1: [A) Predicted class] [B) Entropy + colorbar] [E) Boxplot NDVI/slope]
  Row 2: [C) Intel. Gap     ] [D) Histogram          ] [E) continues       ]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, LightSource
from matplotlib.patches import Patch

from utils_nato import (CLASS_COLORS, CLASS_NAMES, FIG_WIDTH_IN, PALETTE,
                         COLAB_RESULTS_LOCAL, COLAB_RESULTS_V2, RAW_DIR,
                         ANALISE_DIR, class_cmap, load_features,
                         load_positions, raster_meta, save_fig, set_mpl_style)

PROBS_PATH = COLAB_RESULTS_LOCAL / "probs_full.npz"  # RUN FINAL blindada_fix (decisao D1/D2, 12/07)

GEOTIFF_ARMORED = COLAB_RESULTS_LOCAL / "restricao_blindada_movement_gnn_v2_fix.tif"
DEM_PATH = RAW_DIR / "dem" / "pacaraima_dem_consolidated_q1q4.tif"

FRAC = "blindada"
LABEL_FRAC = "Armoured"
INTGAP_PERCENTILE = 85  # Transition Zone: p85 seleciona ~15% dos nos POR CONSTRUCAO


def _downsample(arr, factor=2):
    return arr[::factor, ::factor]


def _load_classes_from_geotiff():
    with rasterio.open(GEOTIFF_ARMORED) as src:
        classes = src.read(1).astype(float)
        nodata = src.nodata
        bounds = src.bounds
    if nodata is not None:
        classes[classes == nodata] = np.nan
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    return classes, extent


def _load_dem_hillshade():
    with rasterio.open(DEM_PATH) as src:
        dem = src.read(1).astype(float)
    ls = LightSource(azdeg=315.0, altdeg=45.0)
    return ls.hillshade(dem, vert_exag=3.0)


def build():
    set_mpl_style()

    print("[fig5] loading classes from official GeoTIFF...")
    class_rast, extent = _load_classes_from_geotiff()
    print(f"[fig5]   shape = {class_rast.shape}")

    print("[fig5] loading DEM (Copernicus) for hillshade...")
    hs = _load_dem_hillshade()

    print("[fig5] loading model v2 probabilities...")
    probs_data = np.load(PROBS_PATH)
    probs_key = f"probs_{FRAC}"
    if probs_key not in probs_data:
        available = list(probs_data.keys())
        print(f"  [WARN] Key '{probs_key}' not found. Available: {available}")
        probs_key = available[0]
    probs = probs_data[probs_key]
    print(f"[fig5]   probs shape = {probs.shape}")

    print("[fig5] loading features (NDVI, slope)...")
    feats = load_features(["NDVI", "slope"])
    ndvi = feats["NDVI"]
    slope = feats["slope"]

    print("[fig5] loading graph positions...")
    pos = load_positions()

    eps = 1e-9
    entropy = -np.sum(probs * np.log(probs + eps), axis=1) / np.log(3)
    thr = np.percentile(entropy, INTGAP_PERCENTILE)
    igap_mask = entropy >= thr
    print(f"[fig5] entropy threshold P{INTGAP_PERCENTILE} = {thr:.3f}  "
          f"-> Intelligence Gap covers {igap_mask.mean()*100:.1f}% of nodes")

    meta = raster_meta()
    H, W = meta["dem_shape"]
    from rasterio.transform import Affine
    tr = Affine(*meta["dem_transform"][:6])

    def _scatter_to_raster(values, fill=np.nan):
        arr = np.full((H, W), fill, dtype=float)
        inv = ~tr
        cols, rows = inv * (pos[:, 0], pos[:, 1])
        rows_i = np.round(rows).astype(int)
        cols_i = np.round(cols).astype(int)
        ok = (rows_i >= 0) & (rows_i < H) & (cols_i >= 0) & (cols_i < W)
        arr[rows_i[ok], cols_i[ok]] = values[ok]
        return arr

    ent_rast = _scatter_to_raster(entropy)
    igap_rast = _scatter_to_raster(igap_mask.astype(float), fill=0.0)

    ds = 2
    class_plot = _downsample(class_rast, ds)
    ent_plot = _downsample(ent_rast, ds)
    igap_plot = _downsample(igap_rast, ds)
    hs_plot = _downsample(hs, ds)

    fig = plt.figure(figsize=(FIG_WIDTH_IN * 1.15, FIG_WIDTH_IN * 0.95),
                     constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.95],
                          height_ratios=[1.0, 1.0])
    cmap_class = class_cmap()
    cmap_ent = LinearSegmentedColormap.from_list(
        "ent", ["#1F3A5F", "#5B7A99", "#F0C432", "#B04B2F"], N=256)

    axA = fig.add_subplot(gs[0, 0])
    axA.imshow(hs_plot, extent=extent, cmap="gray", vmin=0.3, vmax=1.0,
               alpha=0.55, origin="upper", interpolation="nearest")
    axA.imshow(class_plot, extent=extent, cmap=cmap_class, vmin=1, vmax=3,
               origin="upper", interpolation="nearest", alpha=0.82)
    axA.set_xticks([]); axA.set_yticks([])
    axA.set_title(f"A) MovementGNN v2 \u2014 {LABEL_FRAC}",
                  fontsize=8.8, fontweight="bold", color=PALETTE["text"],
                  loc="left", pad=4)
    for sp in axA.spines.values():
        sp.set_color(PALETTE["muted"]); sp.set_linewidth(0.6)

    axB = fig.add_subplot(gs[0, 1])
    axB.imshow(hs_plot, extent=extent, cmap="gray", vmin=0.3, vmax=1.0,
               alpha=0.5, origin="upper", interpolation="nearest")
    im = axB.imshow(ent_plot, extent=extent, cmap=cmap_ent, vmin=0, vmax=1,
                    origin="upper", interpolation="nearest", alpha=0.88)
    axB.set_xticks([]); axB.set_yticks([])
    axB.set_title("B) Softmax entropy", fontsize=8.8, fontweight="bold",
                  color=PALETTE["text"], loc="left", pad=4)
    for sp in axB.spines.values():
        sp.set_color(PALETTE["muted"]); sp.set_linewidth(0.6)
    cb = fig.colorbar(im, ax=axB, fraction=0.046, pad=0.03, shrink=0.95)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.set_label("Normalized entropy", fontsize=6.8, labelpad=3)

    axE = fig.add_subplot(gs[:, 2])
    ndvi_in, ndvi_out = ndvi[igap_mask], ndvi[~igap_mask]
    slope_in, slope_out = slope[igap_mask], slope[~igap_mask]
    smax = float(np.percentile(np.concatenate([slope_in, slope_out]), 98)) + 1e-6
    slope_in_n = np.clip(slope_in / smax, 0, 1)
    slope_out_n = np.clip(slope_out / smax, 0, 1)
    positions = [0, 1, 3, 4]
    data = [ndvi_out, ndvi_in, slope_out_n, slope_in_n]
    colors_b = [PALETTE["muted"], PALETTE["highlight"],
                PALETTE["muted"], PALETTE["highlight"]]
    bp = axE.boxplot(data, positions=positions, widths=0.7,
                     patch_artist=True, showfliers=False,
                     medianprops=dict(color="white", linewidth=1.2))
    for patch, c in zip(bp["boxes"], colors_b):
        patch.set_facecolor(c); patch.set_edgecolor("#1F2D3D")
        patch.set_alpha(0.85)
    for w in bp["whiskers"] + bp["caps"]:
        w.set_color("#1F2D3D"); w.set_linewidth(0.7)
    axE.set_xticks([0.5, 3.5])
    axE.set_xticklabels(["NDVI", "Slope\n(normalized)"], fontsize=7.5)
    axE.set_ylabel("Feature value", fontsize=7.5)
    axE.tick_params(labelsize=6.8)
    axE.set_title("E) Features: inside vs.\noutside Transition Zone",
                  fontsize=8.8, fontweight="bold", color=PALETTE["text"],
                  loc="left", pad=4)
    axE.set_ylim(-0.05, 1.05)
    for sp in axE.spines.values():
        sp.set_color(PALETTE["muted"]); sp.set_linewidth(0.5)
    legend_bp = [
        Patch(facecolor=PALETTE["muted"], alpha=0.85, label="Outside Transition Zone"),
        Patch(facecolor=PALETTE["highlight"], alpha=0.85, label="Inside Transition Zone"),
    ]
    axE.legend(handles=legend_bp, fontsize=6.8, loc="upper right",
               frameon=True, framealpha=0.92)

    axC = fig.add_subplot(gs[1, 0])
    axC.imshow(hs_plot, extent=extent, cmap="gray", vmin=0.3, vmax=1.0,
               alpha=0.45, origin="upper", interpolation="nearest")
    axC.imshow(class_plot, extent=extent, cmap=cmap_class, vmin=1, vmax=3,
               origin="upper", interpolation="nearest", alpha=0.32)
    igap_over = np.where(igap_plot > 0.5, 1.0, np.nan)
    axC.imshow(igap_over, extent=extent,
               cmap=ListedColormap([PALETTE["highlight"]]),
               vmin=0, vmax=1, origin="upper", interpolation="nearest",
               alpha=0.92)
    axC.set_xticks([]); axC.set_yticks([])
    axC.set_title(f"C) Transition Zone (P{INTGAP_PERCENTILE}) \u2014\n"
                  f"{igap_mask.mean()*100:.1f}% of terrain (by construction)",
                  fontsize=8.8, fontweight="bold", color=PALETTE["text"],
                  loc="left", pad=4)
    for sp in axC.spines.values():
        sp.set_color(PALETTE["highlight"]); sp.set_linewidth(0.8)

    axD = fig.add_subplot(gs[1, 1])
    axD.hist(entropy, bins=60, color=PALETTE["primary"],
             edgecolor="white", linewidth=0.3)
    axD.axvline(thr, color=PALETTE["highlight"], linewidth=1.3, linestyle="--",
                label=f"Threshold P{INTGAP_PERCENTILE} = {thr:.2f}")
    axD.set_xlabel("Normalized entropy", fontsize=7.5)
    axD.set_ylabel("Frequency (nodes, log scale)", fontsize=7.5)
    axD.set_title("D) Entropy distribution", fontsize=8.8, fontweight="bold",
                  color=PALETTE["text"], loc="left", pad=4)
    axD.legend(fontsize=6.8, loc="upper right", frameon=True, framealpha=0.92)
    for sp in axD.spines.values():
        sp.set_color(PALETTE["muted"]); sp.set_linewidth(0.5)
    axD.tick_params(labelsize=6.8)
    axD.set_yscale("log")
    axD.set_xlim(0, 1)

    cls_handles = [Patch(facecolor=CLASS_COLORS[c], edgecolor="black", label=lbl)
                   for c, lbl in zip(["go", "slowgo", "nogo"], CLASS_NAMES)]
    axA.legend(handles=cls_handles, loc="lower right", fontsize=4.8,
               handlelength=1.1, borderpad=0.35, labelspacing=0.35,
               frameon=True, framealpha=0.92, edgecolor="#CCCCCC")

    fig.suptitle(
        f"Transition Zone identification \u2014 {LABEL_FRAC} profile",
        x=0.01, y=0.995, ha="left", fontsize=10.6, fontweight="bold",
        color=PALETTE["text"])

    # Elementos cartograficos (exigencia Migon): escala + norte no painel A, CRS no rodape
    import math as _math
    lat_c = 0.5 * (extent[2] + extent[3])
    deg_20km = 20.0 / (111.32 * _math.cos(_math.radians(lat_c)))
    sx = extent[0] + 0.05 * (extent[1] - extent[0])
    sy = extent[2] + 0.06 * (extent[3] - extent[2])
    axA.plot([sx, sx + deg_20km], [sy, sy], "-", color="black", lw=2.4, zorder=9)
    axA.plot([sx, sx + deg_20km / 2], [sy, sy], "-", color="white", lw=2.4, zorder=10)
    axA.text(sx + deg_20km / 2, sy + 0.015 * (extent[3] - extent[2]), "0     10     20 km",
             fontsize=5.6, ha="center", va="bottom", color="white", fontweight="bold", zorder=11)
    xn = extent[1] - 0.07 * (extent[1] - extent[0])
    axA.annotate("N", xy=(xn, extent[3] - 0.05 * (extent[3] - extent[2])),
                 xytext=(xn, extent[3] - 0.16 * (extent[3] - extent[2])),
                 ha="center", va="center", fontsize=8.5, fontweight="bold",
                 color="white", zorder=10,
                 arrowprops=dict(arrowstyle="-|>", color="white", lw=1.3))
    fig.get_layout_engine().set(rect=[0, 0.032, 1, 0.968])
    fig.text(0.99, 0.006,
             "CRS: EPSG:4326 (WGS 84) \u00b7 30 m \u00b7 run results_v2_local_blindada_fix "
             "(2026-05-24) \u00b7 P85 selects ~15% of nodes by construction",
             ha="right", va="bottom", fontsize=5.6, color=PALETTE["muted"], style="italic")

    save_fig(fig, "fig05_intelligence_gap")


if __name__ == "__main__":
    build()
