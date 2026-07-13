# -*- coding: utf-8 -*-
"""
fig_spatial_split_map.png — VERSAO 2 (feedback do autor, 12/07):
particoes com TRANSPARENCIA sobre o TERRENO REAL (Sentinel-2 cor natural),
no padrao da fig04b_dismounted. Substitui a v1 (bloco chapado, ilegivel).

Dados: split_indices.npz (mascaras reais, item 3.5) + fronteiras de blocos medidas
(T-4b) + mosaico Sentinel-2 patched (fundo visual) + bounds de pos.npz.
Paleta categorica validada (dataviz): train #2a78d6 / val #1baf7a / test #eda100.
"""
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = Path(r"F:\arpia_topo_refinado\TOPO_RESTRICAO_MOVIMENTO")
ENT = ROOT / "roadmap_revisao_tutor_migon" / "entregaveis"
S2 = ROOT / "GNN_RESTRICAO_MOVIMENTO/data_raw/satelites_raw/sentinel2/pacaraima/pacaraima_s2_10m_real_q1q4_patched.tif"

N, H, W = 12067692, 3598, 3354
idx = np.load(ENT / "split_indices.npz")
part = np.zeros(N, dtype=np.int8)
part[idx["train_idx"]] = 1
part[idx["val_idx"]] = 2
part[idx["test_idx"]] = 3
p2d = part.reshape(H, W)

pos = np.load(ROOT / "colab/analise_dados/pos.npz")["pos"]
lon_min, lon_max = float(pos[:, 0].min()), float(pos[:, 0].max())
lat_min, lat_max = float(pos[:, 1].min()), float(pos[:, 1].max())
extent = [lon_min, lon_max, lat_min, lat_max]

# fundo: Sentinel-2 cor natural (B04/B03/B02), recortado aos bounds e decimado
print("Lendo fundo Sentinel-2...")
Hb, Wb = H // 2, W // 2
with rasterio.open(S2) as src:
    win = from_bounds(lon_min, lat_min, lon_max, lat_max, transform=src.transform)
    rgb = np.stack([
        src.read(b, window=win, out_shape=(Hb, Wb), resampling=Resampling.bilinear).astype(np.float32)
        for b in (3, 2, 1)  # R=B04, G=B03, B=B02
    ], axis=-1)
mask = rgb.sum(axis=-1) > 0
for c in range(3):
    v = rgb[..., c][mask]
    lo, hi = np.percentile(v, [2, 98])
    rgb[..., c] = np.clip((rgb[..., c] - lo) / max(hi - lo, 1.0), 0, 1)
rgb = np.power(rgb, 0.9)
rgb[~mask] = 1.0

# fronteiras reais dos blocos (mesma deteccao da T-4b)
vcols = np.flatnonzero((p2d[:, 1:] != p2d[:, :-1]).sum(axis=0) > H * 0.05) + 1
hrows = np.flatnonzero((p2d[1:, :] != p2d[:-1, :]).sum(axis=1) > W * 0.05) + 1

part_plot = p2d[::2, ::2].astype(float)
part_masked = np.ma.masked_where(part_plot == 0, part_plot)

SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#898781"
COLORS = {"train": "#2a78d6", "val": "#1baf7a", "test": "#eda100"}
cmap = ListedColormap([COLORS["train"], COLORS["val"], COLORS["test"]])

fig, ax = plt.subplots(figsize=(9.5, 10.4), dpi=200, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.imshow(rgb, extent=extent, origin="upper", aspect="auto", interpolation="bilinear")
ax.imshow(part_masked, extent=extent, origin="upper", aspect="auto",
          cmap=cmap, vmin=1, vmax=3, interpolation="nearest", alpha=0.42)
# linhas das fronteiras de blocos (finas, brancas, semitransparentes)
for c in vcols:
    x = lon_min + (c / W) * (lon_max - lon_min)
    ax.plot([x, x], [lat_min, lat_max], color="white", lw=0.45, alpha=0.55, zorder=5)
for r in hrows:
    y = lat_max - (r / H) * (lat_max - lat_min)
    ax.plot([lon_min, lon_max], [y, y], color="white", lw=0.45, alpha=0.55, zorder=5)

ax.set_xlabel("Longitude (°, EPSG:4326)", color=MUTED, fontsize=9)
ax.set_ylabel("Latitude (°, EPSG:4326)", color=MUTED, fontsize=9)
ax.tick_params(colors=MUTED, labelsize=8)
for s in ax.spines.values():
    s.set_color("#e1e0d9")
ax.set_title("Spatially blocked train–validation–test split over Sentinel-2 natural colour\n"
             "Pacaraima Q1Q4 · 12,067,692 DEM nodes (30 m) · 483 blocks of ~5 km, 100% block-pure · no buffer",
             color=INK, fontsize=10.2, pad=12)

tr, va, te = int((part == 1).sum()), int((part == 2).sum()), int((part == 3).sum())
legend = [Patch(facecolor=COLORS["train"], alpha=0.55, edgecolor=INK, linewidth=0.4,
                label=f"Train — {tr:,} nodes (70.06%) · 338 blocks".replace(",", ".")),
          Patch(facecolor=COLORS["val"], alpha=0.55, edgecolor=INK, linewidth=0.4,
                label=f"Validation — {va:,} nodes (15.31%) · 72 blocks".replace(",", ".")),
          Patch(facecolor=COLORS["test"], alpha=0.55, edgecolor=INK, linewidth=0.4,
                label=f"Test — {te:,} nodes (14.62%) · 73 blocks".replace(",", "."))]
leg = ax.legend(handles=legend, loc="upper right", fontsize=8.5, framealpha=0.95,
                facecolor="white", edgecolor="#e1e0d9")
for t_ in leg.get_texts():
    t_.set_color(INK)

# escala (preto/branco alternado, estilo carta) e norte com halo
lat_mid = (lat_min + lat_max) / 2
deg_10 = 10.0 / (111.32 * math.cos(math.radians(lat_mid)))
sx = lon_min + 0.035 * (lon_max - lon_min)
sy = lat_min + 0.035 * (lat_max - lat_min)
ax.plot([sx, sx + deg_10], [sy, sy], color="black", lw=3.2, zorder=9, solid_capstyle="butt")
ax.plot([sx, sx + deg_10 / 2], [sy, sy], color="white", lw=3.2, zorder=10, solid_capstyle="butt")
for val, x, ha in [("0", sx, "center"), ("5", sx + deg_10 / 2, "center"), ("10 km", sx + deg_10, "center")]:
    ax.text(x, sy + 0.011 * (lat_max - lat_min), val, fontsize=7.5, ha=ha, va="bottom",
            color="white", fontweight="bold", zorder=11,
            path_effects=[pe.withStroke(linewidth=1.6, foreground="black")])
xn = lon_min + 0.045 * (lon_max - lon_min)
yn = lat_max - 0.115 * (lat_max - lat_min)
ax.annotate("N", xy=(xn, yn + 0.05 * (lat_max - lat_min)), xytext=(xn, yn),
            ha="center", fontsize=12, fontweight="bold", color="white", zorder=11,
            path_effects=[pe.withStroke(linewidth=1.8, foreground="black")],
            arrowprops=dict(arrowstyle="-|>", color="white", lw=2.0,
                            path_effects=[pe.withStroke(linewidth=3.4, foreground="black")]))

fig.text(0.985, 0.008,
         "CRS: EPSG:4326 (WGS 84) · background: Sentinel-2 L2A natural colour (gap-filled mosaic) · "
         "masks measured from pacaraima_q1q4_ready.pt (audits T-4/T-4b, 2026-07-12)",
         ha="right", va="bottom", fontsize=6.2, color=MUTED)
fig.tight_layout(rect=[0, 0.02, 1, 1])
fig.savefig(ENT / "fig_spatial_split_map.png", facecolor=SURFACE, bbox_inches="tight")
print("OK: fig_spatial_split_map.png (v2, terreno + transparencia)")
