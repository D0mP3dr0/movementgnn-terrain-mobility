# movementgnn-terrain-mobility

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21328554.svg)](https://doi.org/10.5281/zenodo.21328554)

Code, trained weights and result records of MovementGNN, a multi-task GATv2 network that
classifies every cell of a 12,067,692-node terrain graph of Pacaraima (Roraima, Brazil)
into Go / Slow Go / No Go for four doctrinal mobility profiles (dismounted, motorised,
mechanised, armoured), against rule-based reference labels, and uses predictive entropy
to delimit a Transition Zone of elevated classification ambiguity.

Status: manuscripts in preparation.

## Version 1.1.0

The Python sources in this version are the scripts and library **as they were run**, taken
from the frozen source version (commit `e00fa11d8919151a2674e731d4663c0ef4c5e69e`,
2026-07-12), with comments removed, console messages translated into English and the
absolute data-root prefix turned into a variable. `verify_code.py` proves that the
executable code is identical to the code as run. Version 1.0.x carried a restructured copy
of the pipeline (paths routed through a `src/paths.py` helper and Colab branches removed)
that was never run; it is superseded by this version.

## Repository layout

| Folder | Contents |
|---|---|
| `src/` | Library: model (`models/movement_gnn.py`), rule-informed loss (`models/dameplan_loss.py`), label rules, graph integration, spatial splitter, baselines, training utilities |
| `scripts/` | Pipeline scripts as run: data download, training (`training/07_train_gnn_v2.py`), fine-tune of the final run (`training/07b_retrain_blindada_fix.py`), inference, analyses (transition bands, uncertainty, label audit, embeddings, geospatial maps) |
| `train.py` | Modular training entry point of the library |
| `configs/` | `config_movementgnn_v2.yaml`, consolidated architecture, loss and training configuration |
| `models/` | `movement_gnn_best.pt`, final trained weights (SHA256 `85f21530…`, listed in `SHA256SUMS`) |
| `data/` | `Source_Manifest.csv` (raster provenance and checksums), `split_indices.npz` (train/val/test node indices of the spatial split) |
| `docs/` | `Graph_Construction.md`, `Spatial_Split_Manifest.csv`, `Feature_Dictionary.xlsx`, `Label_Generation_Rules.xlsx`, `Model_and_Training_Config.xlsx` |
| `results/` | `metrics_final_run.json` (final run), `Results_Master.xlsx`, `t2_threshold_bands.json`, `t4_audit_results.json`, `t4b_block_structure.json`, `t12_rebuild_timing.json` |
| `figures/` | Figure-generation scripts, `Figures_Manifest.csv` and `preview/` renders |
| `supplementary/` | `Supplementary_Tables.xlsx` |

Apart from the split indices, files larger than 5 MB are not in this repository: the full-resolution render of the study
area figure, the supplementary information PDF and the derived spatial data. The derived
spatial data (Transition-Zone masks, per-profile classification GeoTIFFs, split indices)
are archived under DOI [10.5281/zenodo.21329327](https://doi.org/10.5281/zenodo.21329327)
(CC BY 4.0).

## Model and run, as recorded

The final run (`results/metrics_final_run.json`) uses 271 input channels, 3 GATv2 layers,
4 attention heads and 52,364 parameters, trained with NeighborLoader in
batches of 750,000 nodes with k = 8 neighbours. The spatial split assigns
8,454,917 / 1,847,936 / 1,764,839 nodes to train / validation / test
(`results/t4_audit_results.json`) in 483 blocks with block purity 1.0
(`results/t4b_block_structure.json`). Rebuilding the
graph from the raw rasters took 184.1 s with a sampled peak of 42.36 GB of resident memory
(`results/t12_rebuild_timing.json`). `verify_payload.py` checks each of these numbers
against its record.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The pinned versions in `requirements.txt` are those of the training environment; the full
freeze of that environment is in `requirements_freeze_full.txt`. PyTorch Geometric needs a
platform-specific installation.

## Data

All inputs are open data: Copernicus DEM GLO-30 and Sentinel-2 L2A. `data/Source_Manifest.csv`
lists every raster with its checksum and embedded tags; `scripts/data/` downloads them. The
scenes are not redistributed here. The scripts read their data under the folder given by
the environment variable `MOVEMENTGNN_DATA_ROOT` (default: the current directory).

## Names kept from the code

The code and the records it writes use Portuguese identifiers, kept so that the code stays
identical to the code as run:

| Identifier | Meaning |
|---|---|
| `a_pe`, `motorizada`, `mecanizada`, `blindada` | dismounted, motorised, mechanised and armoured profiles (keys of `metrics_final_run.json`, `t2_threshold_bands.json`, `t4_audit_results.json` and the label fields `y_*`) |
| `09_diagnostico_transicao.py` | transition-zone diagnosis |
| `10_analise_incerteza.py` | uncertainty analysis |
| `11_gerar_probs.py`, `12_gerar_zrn.py` | generate full-graph probabilities; generate the entropy-based zone |
| `15_analise_geoespacial.py` | geospatial maps |
| `07b_retrain_blindada_fix.py` | fine-tune with the corrected armoured labels |
| `DAMEPLAN` | name of the doctrinal terrain-rule set used for the labels and the loss |

A few short console strings (fewer than four words) and some notes inside the `.xlsx`
workbooks remain in Portuguese, because changing them would change the verified content.
The keys of the four audit records `t2`, `t4`, `t4b` and `t12` were translated; the map
back to the keys as written is in `verify_payload.py`.

## Verification

```bash
sha256sum -c SHA256SUMS          # every file, byte for byte
python3.11 verify_code.py        # code identical to the code as run
python3 verify_payload.py        # records and README numbers identical to the records as written
```

`PROVENANCE.md` lists, for every file, the SHA-256 of the file as run or as written next to
the published one, and says what differs.

## Known limitations

- Reference labels are rule-based proxies, not field-verified trafficability.
- A single humid-tropical study area, with only part of the classic terrain factors modelled.
- GNN training was not bit-reproducible (the fixed seed covers the spatial split and the
  baselines); one training run and one fine-tune run were performed.

## License and citation

Code and weights: MIT (`LICENSE`). Data, records and documents (`data/`, `docs/`,
`results/`, `figures/preview/`, `supplementary/`): CC BY 4.0 (`LICENSE-DATA`).
Cite through `CITATION.cff` or the concept DOI
[10.5281/zenodo.21328554](https://doi.org/10.5281/zenodo.21328554), which resolves to the
latest archived version.

## Contact

L. F. C. Seelig (seelig.felipe@eb.mil.br).
