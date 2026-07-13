# movementgnn-terrain-mobility

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21329214.svg)](https://doi.org/10.5281/zenodo.21329214)

Reproducibility package for the manuscript **"Predictive Entropy in Graph-Based Terrain
Mobility Classification for Known Off-Road Vehicle Profiles"** (under peer review).

MovementGNN is a multi-task GATv2 network that classifies every 30 m cell of a
12,067,692-node terrain graph (Pacaraima, RR, Brazil) into Go / Slow Go / No Go for four
doctrinal mobility profiles (dismounted, motorised, mechanised, armoured), against
rule-based reference labels, and quantifies predictive entropy to delimit a
**Transition Zone** of elevated classification ambiguity.

> Frozen source version: `MovementGNN_JT_Submission_v1`, commit
> `e00fa11d8919151a2674e731d4663c0ef4c5e69e` (2026-07-12). Result tables:
> `results/Results_Master.xlsx`.

## Repository layout

| Folder | Contents |
|---|---|
| `src/` | Library: model (`models/movement_gnn.py`, production defaults 271-d / 3 GATv2 layers), rule-informed loss, label rules (EB 60-ME-11.401 thresholds), graph integration, spatial splitter, baselines, training utilities |
| `scripts/` | Pipeline scripts: data download, training (`training/07_train_gnn_v2.py`), fine-tune, inference, analyses (transition bands, uncertainty, label audit, geospatial) |
| `configs/` | `config_movementgnn_v2.yaml` — complete architecture/loss/training configuration |
| `models/` | `movement_gnn_best.pt` — final trained weights (SHA256 `85F21530…`, 2026-05-24) |
| `data/` | `Source_Manifest.csv` (raster provenance + SHA256), `split_indices.npz` (train/val/test node indices measured from the frozen graph) |
| `docs/` | `Graph_Construction.md`, `Spatial_Split_Manifest.csv`, `Feature_Dictionary.xlsx` (17 attributes), `Label_Generation_Rules.xlsx`, `Model_and_Training_Config.xlsx` |
| `results/` | `metrics_final_run.json`, `Results_Master.xlsx`, `t2_threshold_bands.json`, `t12_rebuild_timing.json` |
| `figures/` | Figure-generation scripts (+ `preview/` renders); set `MOVEMENTGNN_DATA_ROOT` to your data root |
| `supplementary/` | `Supplementary_Information.pdf` (S1–S9), `Supplementary_Tables.xlsx`, `Supplementary_Spatial_Files.zip` (Transition-Zone masks, class GeoTIFFs, split indices) |

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Training was performed with Python 3.11.9, PyTorch 2.10.0+cu128 (CUDA 12.8) and PyG
2.8.0.dev20260214 on an RTX 5070 Ti (16 GB) — the exact 220-package environment is in
`requirements_freeze_full.txt`.

## Data

All inputs are open data: Copernicus DEM GLO-30 and Sentinel-2 L2A (B02/B03/B04/B08,
2023–2024 search window, ≤20 % cloud). `data/Source_Manifest.csv` lists every raster with
checksum and embedded acquisition tags; `scripts/data/` re-downloads them. Scenes are NOT
redistributed here — only identifiers, areas of interest and scripts.

## Minimal reproduction (no 12 M-node graph rebuild required)

1. Install the environment (above).
2. Load the shipped weights `models/movement_gnn_best.pt` and the model in
   `src/models/movement_gnn.py` (in_channels 271, 3 layers, 4 heads).
3. Run inference on a node subset, compute normalised entropy
   `H = -Σ p ln p / ln 3` and apply the per-profile Transition-Zone thresholds
   (`supplementary/Supplementary_Spatial_Files.zip → tz_masks_final_run.npz`:
   dismounted 0.4591, motorised 0.4472, mechanised 0.4207, armoured 0.4402).
4. Reproduce a summary table from `results/metrics_final_run.json`
   (per-profile accuracy/macro-F1 and confusion matrices).

## Full reproduction

Documented end-to-end in `docs/Graph_Construction.md` and
`docs/Model_and_Training_Config.xlsx`: download rasters → build the HeteroData graph
(measured: 184 s, 42 GB peak RAM) → generate rule-based reference labels → spatially
blocked 5 km split (483 blocks, 100 % block-purity, no buffer) → train (85 epochs,
NeighborLoader 750 k / k=8) → fine-tune → analyses. Full training requires a CUDA GPU
(≈16 GB VRAM) and ≈128 GB RAM.

## Known limitations (declared)

- Reference labels are rule-based proxies, not field-verified trafficability.
- Single humid-tropical study area; three of the six classic terrain factors modelled.
- GNN training was not bit-reproducible (seed 42 covers the spatial split and baselines);
  a single training run plus one fine-tune run were performed.

## License and citation

Code: MIT (see `LICENSE`; institutionally authorised).
Cite via `CITATION.cff`. Version `v1.0.1-submission` is archived on Zenodo:
DOI [10.5281/zenodo.21329214](https://doi.org/10.5281/zenodo.21329214)
(concept DOI for all versions: [10.5281/zenodo.21329213](https://doi.org/10.5281/zenodo.21329213)).
The derived spatial data (Transition-Zone masks, per-profile classification GeoTIFFs,
spatial-split indices) are archived separately under
DOI [10.5281/zenodo.21329327](https://doi.org/10.5281/zenodo.21329327) (CC BY 4.0).

## Contact

Technical contact: L. F. C. Seelig (s33l1g@gmail.com).
