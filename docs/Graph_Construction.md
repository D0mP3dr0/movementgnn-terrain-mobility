# Graph_Construction — Grafo de Terreno Pacaraima Q1Q4 (MovementGNN)

> Consolidado em 2026-07-12. Versão congelada de referência: `MovementGNN_JT_Submission_v1`,
> commit `e00fa11d8919151a2674e731d4663c0ef4c5e69e`.
> Convenção: caminhos relativos a `TOPO_RESTRICAO_MOVIMENTO/`. Cada item cita a fonte
> (arquivo:linha ou artefato medido).

## 1. Números exatos

| Item | Valor | Fonte |
|---|---|---|
| Nós totais (HeteroData) | **121.076.487** | `GNN_RESTRICAO_MOV_OB_V1/dados_reunidos/08_grafo/pacaraima_q1q4_ready.json` |
| Arestas totais | **222.974.375** | idem |
| Nós `dem` (1 por célula do DEM — unidade de classificação) | **12.067.692** | idem; medido no grafo (2026-07-12) |
| Nós `sentinel` | **108.609.228** | idem |
| Nós `lidar` | **399.567** | idem |
| Arestas `(dem, adjacent_to, dem)` | **108.567.520** = **96.499.828 de 8-vizinhança + 12.067.692 self-loops (1 por nó)** | ready.json; `colab/analise_dados/edges.npz` (contagem `src==dst` = 12.067.692) |
| Arestas `(sentinel, belongs_to, dem)` | **108.609.228** | ready.json |
| Arestas `(lidar, belongs_to, dem)` | **399.567** | ready.json |
| Arestas `(lidar, near_to, dem)` | **3.196.536** (k=8) | ready.json |
| Arestas `(lidar, near_to, lidar)` | **2.201.524** | ready.json |
| Grid DEM | **3354 × 3598** | ready.json; DEM bruto `pacaraima_dem_consolidated_q1q4.tif` (3354 larg. × 3598 alt.) |

## 2. Resolução, CRS e cobertura

- **Resolução espacial:** 30 m (Copernicus DEM GLO-30; resolução calculada do transform: 0,0002777…° × 111.320 ≈ 30,9 m — `generate_graph.py:120-124`).
- **CRS:** EPSG:4326 (ready.json; `Source_Manifest.csv`).
- **Cobertura:** lon −61,6139 a −60,6825; lat 3,9763 a 4,9755 (bbox medida sobre `pos` do grafo; `download_dem_pacaraima_q1q4.py:27-35`). Área ≈ 10.900 km² (`Dataset_Pacaraima_Q1Q4.md`).

## 3. Definição de cada nó

- **`dem`**: uma célula 30 m do DEM, com 17 features (dicionário completo: `docs/Feature_Dictionary.xlsx`). É o nó rotulado e classificado (`y_a_pe`, `y_motorizada`, `y_mecanizada`, `y_blindada` + máscaras de split).
- **`sentinel`**: um pixel 10 m do Sentinel-2 (9 por célula DEM), 8 features (`generate_graph.py:299-307`; estatísticas em `Features_Terreno_17D.md`).
- **`lidar`**: ponto ICESat-2 ATL08/GEDI02_A, 3 features (das quais 2 constantes/mortas — `Features_Terreno_17D.md`).

## 4. Regra de vizinhança e propriedades das arestas

- **8 vizinhos (8-conectividade)** no grid DEM: offsets `[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]` — `generate_graph.py:315`.
- **Direcionalidade:** cada par vizinho gera aresta nos dois sentidos ao varrer todos os nós (efetivamente bidirecional; armazenado como grafo direcionado com pares recíprocos) — `generate_graph.py:312-320`. Verificação: 108.567.520 = Σ graus do grid 3354×3598 com 8-conectividade (bordas com menos vizinhos).
- **Self-loops:** **SIM — o grafo contém exatamente 1 self-loop por nó dem (12.067.692)**, além das 96.499.828 arestas de 8-vizinhança (medido em `edges.npz`: `src==dst` = 12.067.692; 108.567.520 − 96.499.828 = 12.067.692). Adicionalmente, o modelo usa `GATv2Conv(add_self_loops=True)` (`colab/07_train_gnn_v2.py:104-110`).
- **Atributos de aresta [2]:** distância euclidiana (30 m ortogonal, 30·√2 m diagonal — `generate_graph.py:322`) e Δz entre os nós (elevação normalizada destino − origem — `generate_graph.py:330-332`).

## 5. Tratamento de limites, água e ausentes

- **Bordas da área:** nós de borda simplesmente têm menos arestas (teste `0 ≤ nr < H e 0 ≤ nc < W`, `generate_graph.py:317`); derivadas topográficas usam padding `mode='edge'` (`generate_graph.py:40`).
- **Água:** não há remoção de nós de água; a água entra como feature (`water_mask = ndwi > 0.15`, `generate_graph.py:233`) e como regra de rótulo No-Go (`dameplan_rules.py:161-162,241-242`).
- **Dados ausentes:** NaN/Inf zerados no carregamento para treino (`torch.nan_to_num`, `colab/07_train_gnn_v2.py:427-478`); bandas Sentinel têm ~4,3% de zeros (pixels sem dado — `Features_Terreno_17D.md`).

## 6. Normalização das features

Aplicada na construção (`generate_graph.py:207-246`): Sentinel /10000 com clip [0,1]; NDVI/NDWI clip [−1,1]; elevação em z-score local (média/desvio da área); slope/90; TPI/TRI/roughness divididos pelo desvio da elevação. Estatísticas salvas no próprio grafo (`data.normalization`, `generate_graph.py:403-409`, versão "V18.4"). Detalhe por feature: `Feature_Dictionary.xlsx`.

## 7. Formato de armazenamento

PyTorch Geometric **`HeteroData`** serializado em `.pt` (`generate_graph.py:362,416`). Arquivo oficial: `GNN_RESTRICAO_MOVIMENTO/graph_data/pacaraima_q1q4_ready.pt` — **14.011.792.863 bytes (13,05 GB)**, mtime 2026-05-23 18:46, SHA256 `BE189F0571671BF413E1335D6B166828560005FA5C3F60A69F5B936A346F3393` (`MovementGNN_JT_Submission_v1/HASHES_SHA256.txt`).

## 8. Tempo e memória de construção (medidos em 2026-07-12)

> Protocolo: construção a partir dos rasters brutos
> (`pacaraima_dem_consolidated_q1q4.tif` + `pacaraima_s2_10m_real_q1q4.tif`), com o script
> construtor inalterado, em diretório isolado. Resultado: `results/t12_rebuild_timing.json`.

- **Tempo total: 184,1 s (3,1 min)**.
- **Pico de memória (RSS): 42,36 GB** (amostragem a cada 2 s).
- **Arquivo gerado: 9,53 GB** (sem labels/máscaras/embeddings/LiDAR, adicionados por etapas posteriores; o grafo completo com esses campos tem 13,05 GB).
- **Máquina da medição:** 128 GB RAM, CPU (a construção do grafo é CPU-bound); máquina do treino: RTX 5070 Ti 16 GB.

## 9. Script construtor

- **Implementação:** `colab_geoint_restricao_mov/2-data_processed/generate_graph.py` (cópia congelada em `MovementGNN_JT_Submission_v1/`). Gerador raster→HeteroData do projeto.
- **Etapas posteriores** (fora deste script): integração de labels (`src/graph_construction/graph_integrator.py:112-126`), máscaras de split (`src/graph_construction/spatial_splitter.py`) e embeddings 256D (`colab/06b_extract_topo_embeddings*.py`).

## 10. Arestas entre treino, validação e teste (medido em 2026-07-12)

- As máscaras cobrem 100% dos 12.067.692 nós dem (train 8.454.917 = 70,06% / val 1.847.936 = 15,31% / test 1.764.839 = 14,62%), **sem zona de buffer** (zero nós não atribuídos).
- **Arestas entre partições mantidas no grafo:** **183.702 train↔val, 182.260 train↔test e 38.836 val↔test** (0,37% das 108.567.520 arestas dem–dem). Na avaliação com NeighborLoader (K=8), a passagem de mensagens pode atravessar fronteiras de partição na banda limítrofe (~1 célula = 30 m por camada; 3 camadas ≈ 90 m).
- Estrutura de blocos (fronteiras das máscaras): **grade regular de 483 blocos de 162–163 células (4,86–4,89 km ≈ 5 km), pureza 100%** — train 338 / val 72 / test 73 blocos. Offset da primeira banda: 30 linhas (`results/t4b_block_structure.json`).
- Dados: `results/t4_audit_results.json` + `results/t4b_block_structure.json` + `docs/Spatial_Split_Manifest.csv`.
