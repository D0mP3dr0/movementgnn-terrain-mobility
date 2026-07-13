"""
11_gerar_probs.py — Full-graph inference to generate probs_full.npz
===================================================================
Runs inference on the MovementGNN best checkpoint and saves softmax
probabilities [N, 3] per fraction.

Output:
  probs_full.npz
    keys: probs_a_pe, probs_motorizada, probs_mecanizada, probs_blindada
    dtype: float32, shape: [N_nodes, 3]

Estimated time: ~25 min (A100 40GB)
RAM required: ~20 GB (graph 12GB + embeddings 3GB + model)
"""

import gc
import glob
import json
import os
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATv2Conv, LayerNorm

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from src.paths import get_inference_paths

_paths = get_inference_paths("auto")
GRAPH_PATH   = _paths["graph"]
EMB_DIR      = _paths["emb_dir"]
CKPT_PATH    = _paths["ckpt"]
OUT_DIR      = _paths["out_dir"]

# Must match training configuration exactly
USE_EMBEDDINGS = True
BATCH_SIZE     = 1_050_000
K_NEIGHBORS    = 8
NUM_LAYERS     = 3
HIDDEN_DIM     = 64
HEADS          = 4
DROPOUT        = 0.1
NUM_CLASSES    = 3

FRACTIONS = ["a_pe", "motorizada", "mecanizada", "blindada"]

IDX_SLOPE = 1
IDX_NDVI  = 12
IDX_NDWI  = 13
IDX_WATER = 14


# --- MODEL (must match training architecture) ---
class MovementGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_layers=3,
                 heads=4, num_classes=3, dropout=0.1, edge_dim=2):
        super().__init__()
        self.input_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
        )

        head_dim = hidden_channels // heads
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout_val = dropout

        for _ in range(num_layers):
            conv = GATv2Conv(
                hidden_channels, head_dim, heads=heads, concat=True,
                edge_dim=edge_dim, dropout=dropout, add_self_loops=True,
            )
            self.convs.append(conv)
            self.norms.append(LayerNorm(hidden_channels))

        self.classifiers = nn.ModuleDict({
            frac: nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 2, num_classes),
            )
            for frac in FRACTIONS
        })

    def forward(self, x, edge_index, edge_attr=None):
        h = self.input_encoder(x)
        for conv, norm in zip(self.convs, self.norms):
            h_in = h
            h = conv(h, edge_index, edge_attr=edge_attr)
            h = F.leaky_relu(h, 0.2)
            h = h + h_in
            h = norm(h)
        return {frac: clf(h) for frac, clf in self.classifiers.items()}


# --- HELPERS ---
def log(msg, start_time=None):
    ts = datetime.now().strftime("%H:%M:%S")
    if start_time:
        elapsed = time.perf_counter() - start_time
        print(f"[{ts}] (+{elapsed:5.0f}s) {msg}", flush=True)
    else:
        print(f"[{ts}] {msg}", flush=True)


def detect_embedding_file(emb_dir):
    candidates = sorted(glob.glob(str(emb_dir / "pacaraima_q1q4_embeddings_*.pt")))
    return Path(candidates[-1]) if candidates else None


def unpack_batch(batch, is_hetero, device):
    """Extract tensors from batch (HeteroData or Data)."""
    if is_hetero:
        dem = batch["dem"]
        et = ("dem", "adjacent_to", "dem")
        x  = dem.x.to(device)
        ei = batch[et].edge_index.to(device)
        ea = (batch[et].edge_attr.to(device)
              if hasattr(batch[et], "edge_attr") and batch[et].edge_attr is not None
              else None)
        n_seed = dem.batch_size if hasattr(dem, "batch_size") else x.size(0)
        n_id   = dem.n_id if hasattr(dem, "n_id") else None
    else:
        x  = batch.x.to(device)
        ei = batch.edge_index.to(device)
        ea = (batch.edge_attr.to(device)
              if hasattr(batch, "edge_attr") and batch.edge_attr is not None
              else None)
        n_seed = batch.batch_size if hasattr(batch, "batch_size") else x.size(0)
        n_id   = batch.n_id if hasattr(batch, "n_id") else None
    return x, ei, ea, n_seed, n_id


# --- MAIN ---
def main():
    t0 = time.perf_counter()

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp  = (device == "cuda")
    amp_dtype = torch.bfloat16

    print("=" * 70)
    print("  11_GERAR_PROBS — MovementGNN inference -> probs_full.npz")
    print("=" * 70)
    log(f"Device: {device} | AMP: {'bfloat16' if use_amp else 'OFF'}", t0)

    if device == "cuda":
        gpu = torch.cuda.get_device_properties(0)
        log(f"GPU: {gpu.name} ({gpu.total_memory / 1e9:.1f} GB)", t0)

    assert CKPT_PATH.exists(), f"Checkpoint not found: {CKPT_PATH}"
    assert GRAPH_PATH.exists(), f"Graph not found: {GRAPH_PATH}"

    # --- 1. LOAD GRAPH ---
    log(f"Loading graph {GRAPH_PATH.name} ...", t0)
    data = torch.load(str(GRAPH_PATH), map_location="cpu", weights_only=False)
    is_hetero = hasattr(data, "node_types")

    if is_hetero:
        dem_store = data["dem"]
        n_nodes   = dem_store.x.size(0)
        in_ch     = dem_store.x.size(1)
        et_main   = ("dem", "adjacent_to", "dem")
        for et in data.edge_types:
            data[et].edge_index = data[et].edge_index.long()
        input_nodes   = ("dem", None)
        num_neighbors = {et_main: [K_NEIGHBORS]}
        for other_et in data.edge_types:
            if other_et not in num_neighbors:
                num_neighbors[other_et] = [0]
    else:
        n_nodes = data.x.size(0)
        in_ch   = data.x.size(1)
        data.edge_index = data.edge_index.long()
        input_nodes   = None
        num_neighbors = [K_NEIGHBORS]

    log(f"  {n_nodes:,} DEM nodes | {in_ch} base features", t0)
    gc.collect()

    # --- 2. CONCATENATE GNN_TOPO EMBEDDINGS ---
    emb_path = detect_embedding_file(EMB_DIR) if USE_EMBEDDINGS else None

    if emb_path and emb_path.exists():
        log(f"Loading embeddings: {emb_path.name} ...", t0)
        emb_data = torch.load(str(emb_path), map_location="cpu", weights_only=False)

        if isinstance(emb_data, torch.Tensor):
            emb = emb_data.float()
        elif isinstance(emb_data, dict):
            emb = emb_data.get("embeddings", list(emb_data.values())[0]).float()
        else:
            emb = emb_data.float()

        del emb_data
        gc.collect()

        assert emb.size(0) == n_nodes, \
            f"Embeddings ({emb.size(0)}) != graph ({n_nodes})"

        emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
        emb_dim = emb.size(1)

        if is_hetero:
            data["dem"].x = torch.cat([data["dem"].x, emb], dim=1)
        else:
            data.x = torch.cat([data.x, emb], dim=1)

        in_ch += emb_dim
        del emb
        gc.collect()
        log(f"  Features: {in_ch - emb_dim} base + {emb_dim} emb = {in_ch} total", t0)
    else:
        log("  Embeddings not found — using base features only", t0)

    # --- 3. LOAD CHECKPOINT AND REBUILD MODEL ---
    log(f"Loading checkpoint: {CKPT_PATH.name} ...", t0)
    ckpt = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)

    config = ckpt.get("config", {})
    ckpt_in_ch = config.get("in_ch", in_ch)

    if ckpt_in_ch != in_ch:
        log(f"  WARN: checkpoint recorded in_ch={ckpt_in_ch}, current graph in_ch={in_ch}")
        log(f"  Using in_ch={in_ch} from current graph")

    model = MovementGNN(
        in_channels=in_ch,
        hidden_channels=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        heads=HEADS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    epoch_ckpt = ckpt.get("epoch", "?")
    best_val   = ckpt.get("best_metrics", {})
    log(f"  Checkpoint epoch {epoch_ckpt} | val_loss={best_val.get('val_loss', '?')} "
        f"| val_acc={best_val.get('val_acc', '?')}", t0)

    n_params = sum(p.numel() for p in model.parameters())
    log(f"  Parameters: {n_params:,}", t0)
    model.eval()
    gc.collect()

    # --- 4. FULL INFERENCE — NeighborLoader ---
    log(f"Inference (batch={BATCH_SIZE:,}, K={K_NEIGHBORS}) ...", t0)

    all_probs = {f: torch.zeros(n_nodes, NUM_CLASSES, dtype=torch.float32)
                 for f in FRACTIONS}
    covered   = torch.zeros(n_nodes, dtype=torch.bool)

    inf_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=input_nodes,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    n_batches = (n_nodes + BATCH_SIZE - 1) // BATCH_SIZE
    log(f"  Estimated batches: ~{n_batches}", t0)

    with torch.no_grad():
        for batch_idx, batch in enumerate(inf_loader):
            x, ei, ea, n_seed, n_id = unpack_batch(batch, is_hetero, device)

            # float16 to reduce VRAM usage
            x = x.half()
            if ea is not None:
                ea = ea.half()

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(x, ei, ea)

            for frac in FRACTIONS:
                logits = outputs[frac][:n_seed]
                probs  = F.softmax(logits, dim=1).float().cpu()

                if n_id is not None:
                    ids = n_id[:n_seed].cpu()
                    all_probs[frac][ids] = probs
                    covered[ids] = True

            if (batch_idx + 1) % 10 == 0:
                pct = covered.float().mean() * 100
                log(f"  Batch {batch_idx + 1} | coverage: {pct:.1f}%", t0)

            del x, ei, ea, outputs, batch
            torch.cuda.empty_cache()
            gc.collect()

    coverage = covered.float().mean() * 100
    log(f"  Final coverage: {covered.sum().item():,}/{n_nodes:,} nodes ({coverage:.2f}%)", t0)

    if coverage < 99.0:
        log(f"  WARNING: coverage below 99% — check BATCH_SIZE and K_NEIGHBORS", t0)

    # --- 5. SAVE probs_full.npz ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "probs_full.npz"

    save_dict = {f"probs_{f}": all_probs[f].numpy() for f in FRACTIONS}
    np.savez_compressed(str(out_path), **save_dict)

    size_mb = out_path.stat().st_size / 1e6
    log(f"  Saved: {out_path} ({size_mb:.0f} MB)", t0)

    check = np.load(str(out_path))
    for f in FRACTIONS:
        k = f"probs_{f}"
        arr = check[k]
        log(f"  {k}: shape={arr.shape} | sum_sample={arr[:5].sum(axis=1).round(3).tolist()}", t0)

    meta = {
        "timestamp":   datetime.now().isoformat(),
        "checkpoint":  str(CKPT_PATH),
        "epoch":       epoch_ckpt,
        "best_metrics": best_val,
        "n_nodes":     n_nodes,
        "n_classes":   NUM_CLASSES,
        "fractions":   FRACTIONS,
        "coverage_pct": round(float(coverage), 4),
        "output":      str(out_path),
        "size_mb":     round(size_mb, 1),
    }
    meta_path = OUT_DIR / "probs_full_meta.json"
    with open(str(meta_path), "w") as fp:
        json.dump(meta, fp, indent=2)
    log(f"  Meta: {meta_path.name}", t0)

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 70)
    print("DONE — probs_full.npz generated successfully")
    print(f"File: {out_path}")
    print(f"Size: {size_mb:.0f} MB")
    print(f"Coverage: {coverage:.2f}%")
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
