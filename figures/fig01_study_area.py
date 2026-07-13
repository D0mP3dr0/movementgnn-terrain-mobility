"""Figure 1 -- Study area: Pacaraima (Roraima, Brazil).

Main panel: pseudo-natural RGB from Sentinel-2 B4/B3/B2 with DEM hillshade.
Inset A: regional location (Brazil / Roraima / Venezuela border).
Inset B: elevation histogram from Copernicus GLO-30 DEM.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import rasterio
import geopandas as gpd
import matplotlib.patheffects as pe
from matplotlib.colors import LightSource
from matplotlib.patches import Patch, Rectangle

from utils_nato import (FIG_WIDTH_IN, PALETTE, GEO_DIR, RAW_DIR,
                        save_fig, set_mpl_style)

DEM_PATH = RAW_DIR / "dem" / "pacaraima_dem_consolidated_q1q4.tif"
S2_PATH_PATCHED = RAW_DIR / "sentinel2" / "pacaraima" / "pacaraima_s2_10m_real_q1q4_patched.tif"
S2_PATH_ORIG = RAW_DIR / "sentinel2" / "pacaraima" / "pacaraima_s2_10m_real_q1q4.tif"
S2_PATH = S2_PATH_PATCHED if S2_PATH_PATCHED.exists() else S2_PATH_ORIG
PACARAIMA_LON, PACARAIMA_LAT = -61.15, 4.47


def _load_dem():
    with rasterio.open(DEM_PATH) as src:
        elev = src.read(1).astype(np.float32)
        bounds = src.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    elev[elev < -100] = np.nan
    return elev, extent, bounds


def _load_sentinel2():
    """Load Sentinel-2 as RGB natural color (B04=R, B03=G, B02=B)."""
    with rasterio.open(S2_PATH) as src:
        bounds = src.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        b02 = src.read(1).astype(np.float32)  # Blue
        b03 = src.read(2).astype(np.float32)  # Green
        b04 = src.read(3).astype(np.float32)  # Red
    mask = (b02 > 0) & (b03 > 0) & (b04 > 0)
    channels = []
    for band in [b04, b03, b02]:  # R, G, B
        valid = band[mask]
        lo, hi = np.percentile(valid, [2, 98])
        stretched = np.clip((band - lo) / max(hi - lo, 1.0), 0, 1)
        channels.append(stretched)
    rgb = np.dstack(channels)
    rgb[~mask] = 1.0  # nodata -> white
    return rgb, extent, bounds


def build():
    set_mpl_style()
    print("[fig01] Loading DEM...")
    elev, dem_extent, dem_bounds = _load_dem()
    print(f"[fig01] DEM shape: {elev.shape}, "
          f"range: {np.nanmin(elev):.0f}--{np.nanmax(elev):.0f} m")

    print("[fig01] Loading Sentinel-2...")
    try:
        rgb, s2_extent, s2_bounds = _load_sentinel2()
        has_s2 = True
        print(f"[fig01] S2 shape: {rgb.shape}")
    except Exception as e:
        print(f"[fig01] S2 not available ({e}), using NDVI-derived RGB")
        has_s2 = False

    elev_filled = np.where(np.isnan(elev), np.nanmean(elev), elev)
    ls = LightSource(azdeg=315, altdeg=45)

    dem_terrain = plt.cm.terrain(
        (elev_filled - np.nanmin(elev)) / (np.nanmax(elev) - np.nanmin(elev))
    )[:, :, :3]
    dem_rgb = ls.shade_rgb(dem_terrain, elev_filled,
                           vert_exag=2.5, blend_mode="overlay")

    if has_s2:
        from rasterio.windows import from_bounds as rb_from_bounds
        from scipy.ndimage import zoom as nd_zoom
        with rasterio.open(S2_PATH) as src_s2:
            win = rb_from_bounds(
                dem_bounds.left, dem_bounds.bottom,
                dem_bounds.right, dem_bounds.top,
                transform=src_s2.transform)
            win_r = win.round_offsets().round_lengths()
            b02 = src_s2.read(1, window=win_r).astype(np.float32)
            b03 = src_s2.read(2, window=win_r).astype(np.float32)
            b04 = src_s2.read(3, window=win_r).astype(np.float32)
        mask_s2 = (b02 > 0) & (b03 > 0) & (b04 > 0)
        channels = []
        for band in [b04, b03, b02]:
            valid = band[mask_s2]
            lo, hi = np.percentile(valid, [2, 98])
            stretched = np.clip((band - lo) / max(hi - lo, 1.0), 0, 1)
            channels.append(stretched)
        rgb_s2 = np.dstack(channels)

        elev_hi = nd_zoom(elev_filled, (rgb_s2.shape[0] / elev_filled.shape[0],
                                         rgb_s2.shape[1] / elev_filled.shape[1]), order=1)
        s2_shaded = ls.shade_rgb(rgb_s2, elev_hi,
                                 vert_exag=2.2, blend_mode="overlay")

        dem_hi = nd_zoom(dem_rgb, (rgb_s2.shape[0] / dem_rgb.shape[0],
                                    rgb_s2.shape[1] / dem_rgb.shape[1],
                                    1), order=1)
        mask_3d = np.dstack([mask_s2] * 3)
        rgb_shaded = np.where(mask_3d, s2_shaded, dem_hi)
        nodata_pct = (~mask_s2).sum() / mask_s2.size * 100
        print(f"[fig01] S2 valid: {mask_s2.sum()/1e6:.1f}M px, "
              f"DEM fill: {nodata_pct:.1f}%")
    else:
        rgb_shaded = dem_rgb

    extent = dem_extent
    bounds = dem_bounds

    fig = plt.figure(figsize=(FIG_WIDTH_IN * 1.35, FIG_WIDTH_IN * 0.85))
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        height_ratios=[1, 1], width_ratios=[2.5, 1],
        wspace=0.18, hspace=0.28,
        left=0.06, right=0.97, top=0.90, bottom=0.08,
    )
    ax_main = fig.add_subplot(gs[:, 0])
    ax_inset = fig.add_subplot(gs[0, 1])
    ax_hist = fig.add_subplot(gs[1, 1])

    ax_main.imshow(rgb_shaded, extent=extent, origin="upper",
                   interpolation="nearest", aspect="auto")
    ax_main.set_xlim(extent[0], extent[1])
    ax_main.set_ylim(extent[2], extent[3])
    ax_main.set_xlabel("Longitude (\u00b0, EPSG:4326)", fontsize=7.5)
    ax_main.set_ylabel("Latitude (\u00b0, EPSG:4326)", fontsize=7.5)
    ax_main.text(0.993, 0.012,
                 "CRS: EPSG:4326 (WGS 84) \u00b7 Copernicus DEM GLO-30 (30 m) \u00b7 "
                 "Sentinel-2 L2A (10 m) \u00b7 12,067,692 grid cells",
                 transform=ax_main.transAxes, ha="right", va="bottom",
                 fontsize=5.6, color="#1F2D3D", zorder=9,
                 bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                           edgecolor="none", alpha=0.85))
    ax_main.tick_params(labelsize=6.8)
    ax_main.grid(True, linestyle=":", linewidth=0.3, color="white", alpha=0.45)
    ax_main.set_title(
        "Study area \u2014 Pacaraima (RR, Brazil) \u00b7 ~10,900 km\u00b2 \u00b7 border zone",
        loc="center", fontsize=9.0, fontweight="bold", pad=4,
        color=PALETTE["text"],
    )

    ax_main.text(0.99, 0.985,
                 "Contiguous to the\nBrazil\u2013Venezuela border",
                 transform=ax_main.transAxes,
                 ha="right", va="top", fontsize=6.4,
                 color="#1F2D3D", fontweight="bold", zorder=8,
                 bbox=dict(boxstyle="round,pad=0.22", facecolor="#F2E06A",
                           edgecolor="#1F2D3D", linewidth=0.5, alpha=0.95))

    ax_main.annotate("N",
                     xy=(extent[1] - 0.05, extent[3] - 0.20),
                     xytext=(extent[1] - 0.05, extent[3] - 0.31),
                     ha="center", va="center", fontsize=13, fontweight="bold",
                     color="white", zorder=9,
                     path_effects=[pe.withStroke(linewidth=2.5,
                                                 foreground="#1F2D3D")],
                     arrowprops=dict(arrowstyle="-|>", color="white", lw=2.6,
                                     shrinkA=4,
                                     path_effects=[pe.withStroke(
                                         linewidth=4.5,
                                         foreground="#1F2D3D")]))

    deg_per_20km = 20 / 111
    sx = extent[0] + 0.05
    sy = extent[2] + 0.04
    ax_main.plot([sx, sx + deg_per_20km], [sy, sy],
                 "-", color="black", lw=2.2, zorder=7)
    ax_main.plot([sx, sx + deg_per_20km / 2], [sy, sy],
                 "-", color="white", lw=2.2, zorder=8)
    ax_main.text(sx + deg_per_20km / 2, sy + 0.010,
                 "0            10           20 km",
                 fontsize=5.8, ha="center", va="bottom", zorder=9,
                 color="white", fontweight="bold")

    leg_patches = [
        Patch(facecolor="#A89868", label="Savanna / bare soil"),
        Patch(facecolor="#85884B", label="Cerrado / transition"),
        Patch(facecolor="#3E6E2F", label="Tropical forest"),
        Patch(facecolor="#3A5990", label="Water body"),
    ]
    leg = ax_main.legend(handles=leg_patches, loc="upper left",
                         bbox_to_anchor=(0.01, 0.99),
                         fontsize=6.4, title="Land cover domains",
                         title_fontsize=6.8,
                         frameon=True, facecolor="white",
                         framealpha=0.92, edgecolor=PALETTE["edge"])
    leg.get_frame().set_linewidth(0.4)

    countries = gpd.read_file(GEO_DIR / "_ne_countries_ba_ve.geojson")
    roraima = gpd.read_file(GEO_DIR / "_ne_roraima.geojson")
    ven = countries[countries["NAME"] == "Venezuela"]
    bra = countries[countries["NAME"] == "Brazil"]

    reg_xmin, reg_xmax = -64.5, -58.0
    reg_ymin, reg_ymax = 1.5, 6.2

    ven.plot(ax=ax_inset, facecolor="#F1E8D5", edgecolor="#8C6E2B",
             linewidth=0.8, alpha=0.9)
    bra.plot(ax=ax_inset, facecolor="#DDE8DC", edgecolor="#2E6A41",
             linewidth=0.8, alpha=0.9)
    roraima.plot(ax=ax_inset, facecolor="#B9D1BC", edgecolor="#1F3A5F",
                 linewidth=1.0, alpha=0.95)

    ax_inset.set_xlim(reg_xmin, reg_xmax)
    ax_inset.set_ylim(reg_ymin, reg_ymax)
    ax_inset.set_aspect("equal")

    ax_inset.text(-62.5, 5.6, "VENEZUELA", fontsize=6.6,
                  color="#8C6E2B", fontweight="bold", ha="center", va="center",
                  bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                            edgecolor="none", alpha=0.75))
    ax_inset.text(-61.5, 2.6, "BRAZIL\nRoraima", fontsize=6.6,
                  color="#1F3A5F", fontweight="bold", ha="center", va="center",
                  bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                            edgecolor="none", alpha=0.75))

    ax_inset.add_patch(Rectangle(
        (bounds.left, bounds.bottom),
        bounds.right - bounds.left, bounds.top - bounds.bottom,
        facecolor="none", edgecolor="#B04B2F", linewidth=1.3, zorder=8))

    ax_inset.plot(PACARAIMA_LON, PACARAIMA_LAT, "o", color="#B04B2F",
                  markersize=6.5, markeredgecolor="white",
                  markeredgewidth=1.3, zorder=9)
    ax_inset.annotate("Pacaraima",
                      xy=(PACARAIMA_LON, PACARAIMA_LAT),
                      xytext=(-59.6, 4.9),
                      fontsize=6.6, color="#B04B2F", fontweight="bold",
                      arrowprops=dict(arrowstyle="-|>", color="#B04B2F", lw=0.8),
                      zorder=10)

    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    ax_inset.set_title("A \u00b7 Regional location \u2014 BR/VE border",
                       fontsize=7.6, loc="left", pad=3,
                       fontweight="bold", color=PALETTE["text"])
    for spine in ax_inset.spines.values():
        spine.set_color(PALETTE["muted"])
        spine.set_linewidth(0.5)

    ax_locator = ax_inset.inset_axes([0.015, 0.015, 0.30, 0.32])
    bra.plot(ax=ax_locator, facecolor="#DDE8DC", edgecolor="#2E6A41", linewidth=0.5)
    roraima.plot(ax=ax_locator, facecolor="#B9D1BC", edgecolor="#1F3A5F", linewidth=0.5)
    ax_locator.plot(PACARAIMA_LON, PACARAIMA_LAT, "o", color="#B04B2F",
                    markersize=3, markeredgecolor="white", markeredgewidth=0.5)
    ax_locator.set_xlim(-74, -34)
    ax_locator.set_ylim(-34, 6)
    ax_locator.set_aspect("equal")
    ax_locator.set_xticks([])
    ax_locator.set_yticks([])
    ax_locator.set_facecolor("#FFFFFF")
    for spine in ax_locator.spines.values():
        spine.set_color("#1F2D3D")
        spine.set_linewidth(0.5)

    elev_valid = elev[~np.isnan(elev)]
    ax_hist.hist(elev_valid, bins=60, color="#5D7A3A", alpha=0.90,
                 edgecolor="white", linewidth=0.3)
    ax_hist.set_xlabel("Elevation (m)", fontsize=7)
    ax_hist.set_ylabel("Cell count", fontsize=7)
    ax_hist.tick_params(labelsize=6.2)
    ax_hist.set_title("B \u00b7 Elevation distribution",
                      fontsize=7.8, loc="left", pad=3,
                      fontweight="bold", color=PALETTE["text"])
    ax_hist.spines[["top", "right"]].set_visible(False)
    ax_hist.yaxis.grid(True, linestyle="--", linewidth=0.3, color=PALETTE["grid"])
    ax_hist.set_axisbelow(True)
    ax_hist.text(0.97, 0.95,
                 f"mean: {np.mean(elev_valid):.0f} m\n"
                 f"median: {np.median(elev_valid):.0f} m\n"
                 f"n = {len(elev_valid)/1e6:.2f} M",
                 transform=ax_hist.transAxes,
                 ha="right", va="top", fontsize=6.2,
                 color=PALETTE["text"],
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                           edgecolor=PALETTE["muted"], linewidth=0.4))

    save_fig(fig, "fig01_study_area")


if __name__ == "__main__":
    build()
