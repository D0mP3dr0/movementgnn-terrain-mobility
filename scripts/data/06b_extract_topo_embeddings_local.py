"""
06b_extract_topo_embeddings_local.py — GNN_TOPO embedding extraction (local)
=============================================================================
Local-execution variant. Extracts terrain embeddings from a pre-trained
HierarchicalTerrainGNN checkpoint using spatial chunking.

Key design choices:
  1. No AMP — pure float32 (float16 causes overflow in GATv2 exponentials)
  2. Spatial chunking with guaranteed 100% coverage (overlap=50, border_crop=25)
  3. Auto-detection of hidden_dim/heads/num_layers from checkpoint

Input:
  - pacaraima_q1q4_ready.pt (HeteroData)
  - baseline_Q4_p5_epoch=01_val_loss=0.0653.ckpt (PyTorch Lightning)

Output:
  - pacaraima_q1q4_embeddings_{dim}.pt
"""

import gc
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv

import sys
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from src.paths import get_embedding_extraction_paths

_emb = get_embedding_extraction_paths(local=True)
CKPT_PATH  = _emb["ckpt"]
GRAPH_PATH = _emb["graph"]
OUT_DIR    = _emb["out_dir"]

CHUNK_SIZE   = 250
OVERLAP      = 50
BORDER_CROP  = 25

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- AUTO-DETECT HYPERPARAMETERS FROM CHECKPOINT ---
def infer_hparams(state):
    """Infer hidden_dim, heads, num_layers directly from the state_dict."""
    hidden = state["dem_encoder.0.weight"].shape[0]
    att_key = "convs.0.convs.<dem___adjacent_to___dem>.att"
    heads = state[att_key].shape[1]
    layers = 0
    while f"convs.{layers}.convs.<dem___adjacent_to___dem>.att" in state:
        layers += 1
    return hidden, heads, layers


# --- MODEL: HierarchicalTerrainGNN (must match training architecture) ---
class HierarchicalTerrainGNN(nn.Module):
    def __init__(self, metadata, hidden_dim=256, num_layers=4, heads=8,
                 edge_dim=2, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.dem_encoder = nn.Sequential(
            nn.Linear(17, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout))
        self.sent_encoder = nn.Sequential(
            nn.Linear(8, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout))
        self.lidar_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout))

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            conv_dict = {
                ("dem", "adjacent_to", "dem"): GATv2Conv(
                    hidden_dim, hidden_dim, heads=heads, concat=False,
                    edge_dim=edge_dim, add_self_loops=False),
                ("sentinel", "belongs_to", "dem"): GATv2Conv(
                    (hidden_dim, hidden_dim), hidden_dim, heads=1,
                    concat=False, edge_dim=3, add_self_loops=False),
                ("lidar", "belongs_to", "dem"): GATv2Conv(
                    (hidden_dim, hidden_dim), hidden_dim, heads=1,
                    concat=False, edge_dim=2, add_self_loops=False),
                ("lidar", "near_to", "dem"): GATv2Conv(
                    (hidden_dim, hidden_dim), hidden_dim, heads=1,
                    concat=False, edge_dim=2, add_self_loops=False),
            }
            # Checkpoint compatibility: zip(*metadata[1]) never produces
            # 3-tuples, so this conv is never instantiated. The checkpoint
            # lacks these weights.
            if ("lidar", "near_to", "lidar") in list(zip(*metadata[1])):
                conv_dict[("lidar", "near_to", "lidar")] = GATv2Conv(
                    hidden_dim, hidden_dim, heads=1, concat=False,
                    edge_dim=4, add_self_loops=False)

            self.convs.append(HeteroConv(conv_dict, aggr="sum"))
            self.norms.append(nn.ModuleDict({
                "dem": nn.LayerNorm(hidden_dim),
            }))

        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
                nn.Linear(hidden_dim // 2, 1))
            for name in ["elevation", "slope", "aspect_cos", "aspect_sin",
                         "curvature", "tpi", "tri", "roughness", "flow_accum",
                         "ndvi", "ndwi", "bsi", "shadow"]
        })
        self.heads["canopy"] = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1), nn.Sigmoid())


# --- SPATIAL CHUNKING ---
def generate_chunks(H, W, chunk_size, overlap, border_crop):
    """Generate spatial chunk coordinates with 100% coverage."""
    step = chunk_size - overlap
    chunks = []
    r0 = 0
    while r0 < H:
        r1 = min(r0 + chunk_size, H)
        c0 = 0
        while c0 < W:
            c1 = min(c0 + chunk_size, W)
            rv0 = r0 + border_crop if r0 > 0 else r0
            rv1 = r1 - border_crop if r1 < H else r1
            cv0 = c0 + border_crop if c0 > 0 else c0
            cv1 = c1 - border_crop if c1 < W else c1
            if rv0 < rv1 and cv0 < cv1:
                chunks.append((r0, c0, r1, c1, rv0, cv0, rv1, cv1))
            if c1 >= W:
                break
            c0 += step
        if r1 >= H:
            break
        r0 += step
    return chunks


# --- SUBGRAPH EXTRACTION PER CHUNK ---
def build_chunk_subgraph(data, dem_mask, dem_row_np, dem_col_np,
                         edge_cache, n_dem, n_sent, n_lidar):
    """
    Build a HeteroData subgraph for one spatial chunk.

    Args:
        dem_mask: boolean numpy array (n_dem,) indicating DEM nodes in the chunk
        edge_cache: dict with pre-extracted edge_index numpy arrays
    """
    chunk_dem_ids = np.where(dem_mask)[0]
    n_chunk_dem = len(chunk_dem_ids)
    if n_chunk_dem == 0:
        return None, chunk_dem_ids

    g2l_dem = np.full(n_dem, -1, dtype=np.int64)
    g2l_dem[chunk_dem_ids] = np.arange(n_chunk_dem)

    x_dict = {"dem": data["dem"].x[chunk_dem_ids].to(DEVICE)}
    edge_index_dict = {}
    edge_attr_dict = {}

    # ── DEM-DEM ───────────────────────────────────────────────────────
    dd = edge_cache["dem_dem"]
    dd_mask = dem_mask[dd["src"]] & dem_mask[dd["dst"]]
    n_dd = dd_mask.sum()
    if n_dd > 0:
        local_src = torch.from_numpy(g2l_dem[dd["src"][dd_mask]]).long()
        local_dst = torch.from_numpy(g2l_dem[dd["dst"][dd_mask]]).long()
        et = ("dem", "adjacent_to", "dem")
        edge_index_dict[et] = torch.stack([local_src, local_dst]).to(DEVICE)
        if dd["ea"] is not None:
            edge_attr_dict[et] = dd["ea"][torch.from_numpy(np.where(dd_mask)[0])].to(DEVICE)

    def extract_other_nodes(key, node_type, edge_type, n_other, other_x):
        info = edge_cache.get(key)
        if info is None:
            return
        mask = dem_mask[info["dst"]]
        if not mask.any():
            return

        other_global = np.unique(info["src"][mask])
        n_other_chunk = len(other_global)
        g2l_other = np.full(n_other, -1, dtype=np.int64)
        g2l_other[other_global] = np.arange(n_other_chunk)

        if node_type not in x_dict:
            x_dict[node_type] = other_x[other_global].to(DEVICE)
            extract_other_nodes._g2l[node_type] = g2l_other
            extract_other_nodes._global[node_type] = other_global
        else:
            existing = extract_other_nodes._global.get(node_type, np.array([]))
            merged = np.unique(np.concatenate([existing, other_global]))
            if len(merged) > len(existing):
                g2l_merged = np.full(n_other, -1, dtype=np.int64)
                g2l_merged[merged] = np.arange(len(merged))
                x_dict[node_type] = other_x[merged].to(DEVICE)
                extract_other_nodes._g2l[node_type] = g2l_merged
                extract_other_nodes._global[node_type] = merged
            g2l_other = extract_other_nodes._g2l[node_type]

        local_src = torch.from_numpy(g2l_other[info["src"][mask]]).long()
        local_dst = torch.from_numpy(g2l_dem[info["dst"][mask]]).long()
        edge_index_dict[edge_type] = torch.stack([local_src, local_dst]).to(DEVICE)
        if info["ea"] is not None:
            edge_attr_dict[edge_type] = info["ea"][
                torch.from_numpy(np.where(mask)[0])].to(DEVICE)

    extract_other_nodes._g2l = {}
    extract_other_nodes._global = {}

    # ── Sentinel -> DEM ───────────────────────────────────────────────
    if n_sent > 0:
        extract_other_nodes(
            "sent_belongs_dem", "sentinel",
            ("sentinel", "belongs_to", "dem"),
            n_sent, data["sentinel"].x)

    # ── LiDAR -> DEM (belongs_to + near_to) ───────────────────────────
    if n_lidar > 0:
        extract_other_nodes(
            "lidar_belongs_dem", "lidar",
            ("lidar", "belongs_to", "dem"),
            n_lidar, data["lidar"].x)
        extract_other_nodes(
            "lidar_near_dem", "lidar",
            ("lidar", "near_to", "dem"),
            n_lidar, data["lidar"].x)

    return (x_dict, edge_index_dict, edge_attr_dict), chunk_dem_ids


# --- MAIN ---
def main():
    t0 = time.time()

    def log(msg):
        elapsed = time.time() - t0
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] (+{elapsed:5.0f}s) {msg}", flush=True)

    print("=" * 70)
    print("  GNN_TOPO EMBEDDING EXTRACTION — Pacaraima Q1Q4")
    print("=" * 70)
    log(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        gpu = torch.cuda.get_device_properties(0)
        log(f"GPU: {gpu.name} ({gpu.total_memory / 1e9:.1f} GB)")
    log(f"AMP: DISABLED (pure float32 — prevents NaN)")
    log(f"Chunk: {CHUNK_SIZE}x{CHUNK_SIZE}, overlap={OVERLAP}, border_crop={BORDER_CROP}")

    # --- 1. LOAD GRAPH ---
    log(f"[1/5] Loading {GRAPH_PATH.name} ...")
    data = torch.load(str(GRAPH_PATH), map_location="cpu", weights_only=False)

    n_dem = data["dem"].x.size(0)
    n_sent = data["sentinel"].x.size(0) if "sentinel" in data.node_types else 0
    n_lidar = data["lidar"].x.size(0) if "lidar" in data.node_types else 0

    log(f"  DEM: {n_dem:,} nodes x {data['dem'].x.size(1)} feat")
    log(f"  Sentinel: {n_sent:,} nodes")
    log(f"  LiDAR: {n_lidar:,} nodes")
    log(f"  Edge types: {[str(et) for et in data.edge_types]}")

    if hasattr(data["dem"], "grid_shape"):
        gs = data["dem"].grid_shape
        H, W = (gs.tolist() if torch.is_tensor(gs) else list(gs))
    elif hasattr(data, "raster_meta") and data.raster_meta:
        H, W = data.raster_meta["dem_shape"]
    else:
        raise RuntimeError("No grid_shape or raster_meta — spatial chunking not possible")

    assert H * W == n_dem, f"Grid {H}x{W}={H*W} != n_dem={n_dem}"
    log(f"  Grid: {H} rows x {W} cols = {H*W:,} nodes")

    # --- 2. PRE-COMPUTE GRID POSITIONS AND EDGE CACHE ---
    log("[2/5] Pre-computing grid positions and edge cache ...")
    dem_row_np = np.arange(n_dem, dtype=np.int32) // W
    dem_col_np = np.arange(n_dem, dtype=np.int32) % W

    def cache_edge(et_tuple):
        if et_tuple not in data.edge_types:
            return None
        store = data[et_tuple]
        ei = store.edge_index.numpy()
        ea = store.edge_attr if (hasattr(store, "edge_attr") and store.edge_attr is not None) else None
        return {"src": ei[0], "dst": ei[1], "ea": ea}

    edge_cache = {
        "dem_dem":          cache_edge(("dem", "adjacent_to", "dem")),
        "sent_belongs_dem": cache_edge(("sentinel", "belongs_to", "dem")),
        "lidar_belongs_dem":cache_edge(("lidar", "belongs_to", "dem")),
        "lidar_near_dem":   cache_edge(("lidar", "near_to", "dem")),
    }

    for k, v in edge_cache.items():
        if v is not None:
            log(f"  {k}: {len(v['src']):,} edges, ea={'yes' if v['ea'] is not None else 'no'}")

    # --- 3. LOAD MODEL FROM CHECKPOINT ---
    log(f"[3/5] Loading checkpoint {CKPT_PATH.name} ...")
    ckpt = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)

    raw_state = ckpt.get("state_dict", ckpt)
    state = {}
    for k, v in raw_state.items():
        clean = k.replace("model.", "") if k.startswith("model.") else k
        if not clean.startswith("criterion."):
            state[clean] = v

    hidden_dim, heads, num_layers = infer_hparams(state)
    log(f"  Auto-detect: hidden_dim={hidden_dim}, heads={heads}, num_layers={num_layers}")

    metadata = (list(data.node_types), list(data.edge_types))
    model = HierarchicalTerrainGNN(
        metadata=metadata,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        heads=heads,
        edge_dim=2,
        dropout=0.0,
    )

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        heads_missing = [k for k in missing if k.startswith("heads.")]
        other_missing = [k for k in missing if not k.startswith("heads.")]
        if other_missing:
            log(f"  WARN missing (non-heads): {other_missing[:5]}")
        if heads_missing:
            log(f"  Missing heads (irrelevant for embeddings): {len(heads_missing)} keys")
    if unexpected:
        log(f"  Unexpected keys: {unexpected[:5]}")

    model = model.to(DEVICE)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  Model loaded: {n_params:,} params on {DEVICE}")

    del ckpt, raw_state
    gc.collect()

    # --- 4. EXTRACT EMBEDDINGS PER CHUNK ---
    chunks = generate_chunks(H, W, CHUNK_SIZE, OVERLAP, BORDER_CROP)
    log(f"[4/5] Extracting embeddings from {len(chunks)} chunks ...")

    all_embeddings = torch.zeros(n_dem, hidden_dim, dtype=torch.float32)
    node_counts = torch.zeros(n_dem, dtype=torch.int32)

    for ci, (r0, c0, r1, c1, rv0, cv0, rv1, cv1) in enumerate(chunks):
        t_chunk = time.time()

        dem_mask = ((dem_row_np >= r0) & (dem_row_np < r1) &
                    (dem_col_np >= c0) & (dem_col_np < c1))

        result = build_chunk_subgraph(
            data, dem_mask, dem_row_np, dem_col_np,
            edge_cache, n_dem, n_sent, n_lidar)

        if result[0] is None:
            continue

        (x_dict, edge_index_dict, edge_attr_dict), chunk_dem_ids = result

        with torch.no_grad():
            xd = {}
            xd["dem"] = model.dem_encoder(x_dict["dem"])
            if "sentinel" in x_dict:
                xd["sentinel"] = model.sent_encoder(x_dict["sentinel"])
            if "lidar" in x_dict:
                xd["lidar"] = model.lidar_encoder(x_dict["lidar"])

            for conv, norm_dict in zip(model.convs, model.norms):
                out = conv(xd, edge_index_dict, edge_attr_dict)
                for key in out:
                    h = out[key] + xd.get(key, 0)
                    if key in norm_dict:
                        h = norm_dict[key](h)
                    h = F.gelu(h)
                    xd[key] = h

            dem_emb = xd["dem"].cpu()

        n_nan = torch.isnan(dem_emb).sum().item()
        if n_nan > 0:
            log(f"  WARN chunk {ci}: {n_nan} NaN in {dem_emb.numel()} values "
                f"({n_nan/dem_emb.numel()*100:.1f}%) — replacing with 0")
            dem_emb = torch.nan_to_num(dem_emb, nan=0.0)

        # Accumulate only the valid region (after border crop)
        local_rows = dem_row_np[chunk_dem_ids]
        local_cols = dem_col_np[chunk_dem_ids]
        valid = ((local_rows >= rv0) & (local_rows < rv1) &
                 (local_cols >= cv0) & (local_cols < cv1))

        valid_global = chunk_dem_ids[valid]
        valid_local = np.where(valid)[0]

        if len(valid_local) > 0 and valid_local.max() < dem_emb.shape[0]:
            all_embeddings[valid_global] += dem_emb[valid_local]
            node_counts[valid_global] += 1

        elapsed_c = time.time() - t_chunk

        del xd, x_dict, edge_index_dict, edge_attr_dict, dem_emb
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        if (ci + 1) % 10 == 0 or ci == 0 or ci == len(chunks) - 1:
            covered = (node_counts > 0).sum().item()
            total_el = time.time() - t0
            eta = (total_el / (ci + 1)) * (len(chunks) - ci - 1)
            log(f"  Chunk {ci+1:3d}/{len(chunks)} | "
                f"covered {covered:,}/{n_dem:,} ({covered/n_dem*100:.1f}%) | "
                f"{elapsed_c:.1f}s/chunk | ETA {eta/60:.1f}min")

    # Average embeddings for nodes covered by multiple overlapping chunks
    valid_mask = node_counts > 0
    multi_mask = node_counts > 1
    if multi_mask.any():
        all_embeddings[multi_mask] /= node_counts[multi_mask].unsqueeze(1).float()

    coverage = valid_mask.sum().item()
    coverage_pct = coverage / n_dem * 100
    log(f"  Final coverage: {coverage:,}/{n_dem:,} ({coverage_pct:.1f}%)")

    n_nan_final = torch.isnan(all_embeddings).sum().item()
    n_inf_final = torch.isinf(all_embeddings).sum().item()
    log(f"  NaN: {n_nan_final}, Inf: {n_inf_final}")

    if coverage_pct < 99.0:
        log(f"  WARNING: coverage below 99% — check chunk parameters")

    # --- 5. SAVE ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"pacaraima_q1q4_embeddings_{hidden_dim}.pt"

    log(f"[5/5] Saving -> {out_path.name} ...")
    torch.save({
        "embeddings":   all_embeddings,
        "valid_mask":   valid_mask,
        "num_nodes":    n_dem,
        "dim":          hidden_dim,
        "grid_shape":   (H, W),
        "coverage_pct": round(coverage_pct, 2),
        "terrain_path": str(GRAPH_PATH),
        "checkpoint":   str(CKPT_PATH),
        "config": {
            "chunk_size":  CHUNK_SIZE,
            "overlap":     OVERLAP,
            "border_crop": BORDER_CROP,
            "hidden_dim":  hidden_dim,
            "heads":       heads,
            "num_layers":  num_layers,
            "amp":         False,
        },
    }, str(out_path))

    sz_gb = out_path.stat().st_size / (1024**3)
    total = time.time() - t0
    log(f"  Saved: {sz_gb:.2f} GB")

    print("=" * 70)
    print(f"  DONE in {total/60:.1f} min")
    print(f"  Embeddings: {n_dem:,} nodes x {hidden_dim} dim")
    print(f"  Coverage: {coverage_pct:.1f}%")
    print(f"  File: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
