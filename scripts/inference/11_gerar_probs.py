INTERNAL_DATA_ROOT = __import__("os").environ.get("MOVEMENTGNN_DATA_ROOT", ".")
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

DRIVE_BASE   = Path(f"{INTERNAL_DATA_ROOT}")
GRAPH_PATH   = DRIVE_BASE / "graph/pacaraima_q1q4_fixed.pt"
EMB_DIR      = DRIVE_BASE / "graph"
CKPT_PATH    = DRIVE_BASE / "gnn/results_v2/checkpoints/checkpoint_best.pt"
OUT_DIR      = DRIVE_BASE / "gnn/results_v2"

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


def main():
    t0 = time.perf_counter()

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp  = (device == "cuda")
    amp_dtype = torch.bfloat16

    print("=" * 70)
    print("  11_GERAR_PROBS — Inferencia MovementGNN → probs_full.npz")
    print("=" * 70)
    log(f"Device: {device} | AMP: {'bfloat16' if use_amp else 'OFF'}", t0)

    if device == "cuda":
        gpu = torch.cuda.get_device_properties(0)
        log(f"GPU: {gpu.name} ({gpu.total_memory / 1e9:.1f} GB)", t0)

    assert CKPT_PATH.exists(), f"Checkpoint not found: {CKPT_PATH}"
    assert GRAPH_PATH.exists(), f"Graph not found: {GRAPH_PATH}"

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

    log(f"  {n_nodes:,} nos DEM | {in_ch} features base", t0)
    gc.collect()

    emb_path = detect_embedding_file(EMB_DIR) if USE_EMBEDDINGS else None

    if emb_path and emb_path.exists():
        log(f"Carregando embeddings: {emb_path.name} ...", t0)
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
            f"Embeddings ({emb.size(0)}) != grafo ({n_nodes})"

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
        log('  Embeddings not found — using the base features only', t0)

    log(f"Carregando checkpoint: {CKPT_PATH.name} ...", t0)
    ckpt = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)

    config = ckpt.get("config", {})
    ckpt_in_ch = config.get("in_ch", in_ch)

    if ckpt_in_ch != in_ch:
        log(f"  WARN: checkpoint registrou in_ch={ckpt_in_ch}, grafo atual in_ch={in_ch}")
        log(f"  Using in_ch={in_ch} from the current graph")

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
    log(f"  Checkpoint epoca {epoch_ckpt} | val_loss={best_val.get('val_loss', '?')} "
        f"| val_acc={best_val.get('val_acc', '?')}", t0)

    n_params = sum(p.numel() for p in model.parameters())
    log(f"  Parametros: {n_params:,}", t0)
    model.eval()
    gc.collect()

    log(f"Inferencia (batch={BATCH_SIZE:,}, K={K_NEIGHBORS}) ...", t0)

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
    log(f"  Batches estimados: ~{n_batches}", t0)

    with torch.no_grad():
        for batch_idx, batch in enumerate(inf_loader):
            x, ei, ea, n_seed, n_id = unpack_batch(batch, is_hetero, device)

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
                log(f"  Batch {batch_idx + 1} | cobertura: {pct:.1f}%", t0)

            del x, ei, ea, outputs, batch
            torch.cuda.empty_cache()
            gc.collect()

    coverage = covered.float().mean() * 100
    log(f"  Cobertura final: {covered.sum().item():,}/{n_nodes:,} nos ({coverage:.2f}%)", t0)

    if coverage < 99.0:
        log(f"  WARNING: coverage below 99% — check BATCH_SIZE and K_NEIGHBORS", t0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "probs_full.npz"

    save_dict = {f"probs_{f}": all_probs[f].numpy() for f in FRACTIONS}
    np.savez_compressed(str(out_path), **save_dict)

    size_mb = out_path.stat().st_size / 1e6
    log(f"  Salvo: {out_path} ({size_mb:.0f} MB)", t0)

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
    print('DONE — probs_full.npz generated successfully')
    print(f"Arquivo: {out_path}")
    print(f"Tamanho: {size_mb:.0f} MB")
    print(f"Cobertura: {coverage:.2f}%")
    print(f"Tempo total: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("=" * 70)
    print()
    print("Proximos passos:")
    print("  !python internal/gnn/10_analise_incerteza.py")
    print('  (D07 and D08 will now run with the available probabilities)')


if __name__ == "__main__":
    main()
