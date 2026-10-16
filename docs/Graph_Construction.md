# Graph_Construction: Pacaraima Q1Q4 terrain graph (MovementGNN)

> Compiled on 2026-07-12. Frozen reference version: commit
> `e00fa11d8919151a2674e731d4663c0ef4c5e69e`. Paths are relative to the root of the frozen
> source; `file:line` references point to the files as run (the published scripts have
> different line numbers because their comments were removed). Each item cites its source
> (file:line or measured artefact).

## 1. Exact counts

| Item | Value | Source |
|---|---|---|
| Total nodes (HeteroData) | **121,076,487** | `GNN_RESTRICAO_MOV_OB_V1/dados_reunidos/08_grafo/pacaraima_q1q4_ready.json` |
| Total edges | **222,974,375** | same |
| `dem` nodes (1 per DEM cell, the classification unit) | **12,067,692** | same; measured on the graph (2026-07-12) |
| `sentinel` nodes | **108,609,228** | same |
| `lidar` nodes | **399,567** | same |
| `(dem, adjacent_to, dem)` edges | **108,567,520** = **96,499,828 from the 8-neighbourhood + 12,067,692 self-loops (1 per node)** | ready.json; `colab/analise_dados/edges.npz` (count of `src==dst` = 12,067,692) |
| `(sentinel, belongs_to, dem)` edges | **108,609,228** | ready.json |
| `(lidar, belongs_to, dem)` edges | **399,567** | ready.json |
| `(lidar, near_to, dem)` edges | **3,196,536** (k=8) | ready.json |
| `(lidar, near_to, lidar)` edges | **2,201,524** | ready.json |
| DEM grid | **3354 × 3598** | ready.json; raw DEM `pacaraima_dem_consolidated_q1q4.tif` (3354 wide × 3598 high) |

## 2. Resolution, CRS and coverage

- **Spatial resolution:** 30 m (Copernicus DEM GLO-30; resolution computed from the transform: 0.0002777…° × 111,320 ≈ 30.9 m, `generate_graph.py:120-124`).
- **CRS:** EPSG:4326 (ready.json; `data/Source_Manifest.csv`).
- **Coverage:** lon −61.6139 to −60.6825; lat 3.9763 to 4.9755 (bbox measured on the graph `pos`; `download_dem_pacaraima_q1q4.py:27-35`). Area ≈ 10,900 km².

## 3. Node definitions

- **`dem`**: one 30 m DEM cell with 17 features (full dictionary: `docs/Feature_Dictionary.xlsx`). It is the labelled and classified node (`y_a_pe`, `y_motorizada`, `y_mecanizada`, `y_blindada` + split masks).
- **`sentinel`**: one 10 m Sentinel-2 pixel (9 per DEM cell), 8 features (`generate_graph.py:299-307`).
- **`lidar`**: one ICESat-2 ATL08/GEDI02_A point, 3 features (2 of which are constant).

## 4. Neighbourhood rule and edge properties

- **8 neighbours (8-connectivity)** on the DEM grid: offsets `[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]`, `generate_graph.py:315`.
- **Directionality:** each neighbouring pair produces an edge in both directions when all nodes are scanned (effectively bidirectional; stored as a directed graph with reciprocal pairs), `generate_graph.py:312-320`. Check: 108,567,520 = sum of the degrees of the 3354×3598 grid with 8-connectivity (border cells have fewer neighbours).
- **Self-loops:** **YES, the graph contains exactly 1 self-loop per dem node (12,067,692)**, in addition to the 96,499,828 8-neighbourhood edges (measured in `edges.npz`: `src==dst` = 12,067,692; 108,567,520 − 96,499,828 = 12,067,692). The model also uses `GATv2Conv(add_self_loops=True)` (`colab/07_train_gnn_v2.py:104-110`).
- **Edge attributes [2]:** Euclidean distance (30 m orthogonal, 30·√2 m diagonal, `generate_graph.py:322`) and Δz between the nodes (normalised elevation, target minus source, `generate_graph.py:330-332`).

## 5. Borders, water and missing data

- **Area borders:** border nodes simply have fewer edges (test `0 ≤ nr < H and 0 ≤ nc < W`, `generate_graph.py:317`); topographic derivatives use `mode='edge'` padding (`generate_graph.py:40`).
- **Water:** water nodes are not removed; water enters as a feature (`water_mask = ndwi > 0.15`, `generate_graph.py:233`) and as a No-Go label rule (`dameplan_rules.py:161-162,241-242`).
- **Missing data:** NaN/Inf are set to zero when loading for training (`torch.nan_to_num`, `colab/07_train_gnn_v2.py:427-478`); the Sentinel bands have ~4.3% zeros (pixels without data).

## 6. Feature normalisation

Applied at construction time (`generate_graph.py:207-246`): Sentinel /10000 clipped to [0,1]; NDVI/NDWI clipped to [−1,1]; elevation as a local z-score (mean and standard deviation of the area); slope/90; TPI/TRI/roughness divided by the standard deviation of the elevation. The statistics are stored in the graph itself (`data.normalization`, `generate_graph.py:403-409`, version "V18.4"). Per-feature detail: `docs/Feature_Dictionary.xlsx`.

## 7. Storage format

PyTorch Geometric **`HeteroData`** serialised as `.pt` (`generate_graph.py:362,416`). Reference file: `GNN_RESTRICAO_MOVIMENTO/graph_data/pacaraima_q1q4_ready.pt`, **14,011,792,863 bytes (13.05 GB)**, mtime 2026-05-23 18:46, SHA256 `BE189F0571671BF413E1335D6B166828560005FA5C3F60A69F5B936A346F3393` (hash list of the frozen source).

## 8. Construction time and memory (measured on 2026-07-12)

> Protocol: construction from the raw rasters
> (`pacaraima_dem_consolidated_q1q4.tif` + `pacaraima_s2_10m_real_q1q4.tif`) with the builder
> script unchanged, in an isolated directory. Result: `results/t12_rebuild_timing.json`.

- **Total time: 184.1 s (3.1 min)**.
- **Peak memory (RSS): 42.36 GB** (sampled every 2 s).
- **Generated file: 9.53 GB** (without labels/masks/embeddings/LiDAR, which later stages add; the full graph with those fields has 13.05 GB).
- **Measurement machine:** 128 GB RAM, CPU (graph construction is CPU-bound); training machine: RTX 5070 Ti 16 GB.

## 9. Builder script

- **Implementation:** `colab_geoint_restricao_mov/2-data_processed/generate_graph.py` (frozen copy in the frozen source; not included in this repository). Raster to HeteroData generator of the project.
- **Later stages** (outside this script): label integration (`src/graph_construction/graph_integrator.py:112-126`), split masks (`src/graph_construction/spatial_splitter.py`) and 256-D embeddings (`scripts/data/06b_extract_topo_embeddings*.py`).

## 10. Edges between training, validation and test (measured on 2026-07-12)

- The masks cover 100% of the 12,067,692 dem nodes (train 8,454,917 = 70.06% / val 1,847,936 = 15.31% / test 1,764,839 = 14.62%), **with no buffer zone** (zero unassigned nodes).
- **Edges between partitions kept in the graph:** **183,702 train↔val, 182,260 train↔test and 38,836 val↔test** (0.37% of the 108,567,520 dem–dem edges). In evaluation with NeighborLoader (K=8), message passing can cross partition boundaries in the border band (~1 cell = 30 m per layer; 3 layers ≈ 90 m).
- Block structure (mask boundaries): **regular grid of 483 blocks of 162–163 cells (4.86–4.89 km ≈ 5 km), purity 100%**, train 338 / val 72 / test 73 blocks. Offset of the first band: 30 rows (`results/t4b_block_structure.json`).
- Data: `results/t4_audit_results.json` + `results/t4b_block_structure.json` + `docs/Spatial_Split_Manifest.csv`.
