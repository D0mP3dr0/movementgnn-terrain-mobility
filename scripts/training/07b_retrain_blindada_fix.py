"""
07b_retrain_blindada_fix.py - V2 Fine-Tuning (Armored Fraction)
=======================================================================
Loads checkpoint_best.pt from the V2 training run and continues
training for additional epochs with refined y_blindada labels.

Saves results to a separate output directory to preserve the original run.
Uses the same architecture and base hyperparameters with a lower LR for fine-tuning.
"""

import gc
import glob
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.checkpoint import checkpoint as grad_ckpt
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATv2Conv, LayerNorm
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from src.paths import get_training_paths

_armored = get_training_paths("v2_armored")
_local = get_training_paths("v2_local")
GRAPH_PATH      = _armored["graph"]
EMB_DIR         = _armored["emb_dir"]
CHECKPOINT_PATH = _local["out_dir"] / "checkpoints/checkpoint_best.pt"
OUT_DIR         = _armored["out_dir"]
USE_EMBEDDINGS = True

BATCH_SIZE   = 750_000
K_NEIGHBORS  = 8
NUM_LAYERS   = 3
HIDDEN_DIM   = 64
HEADS        = 4
DROPOUT      = 0.1
NUM_CLASSES  = 3

EXTRA_EPOCHS = 25
LR           = 5e-5          # lower than original (3e-4) for fine-tuning
MAX_NORM     = 1.0
WEIGHT_DECAY = 1e-5
ES_PATIENCE  = 10
CKPT_EVERY   = 5

FRACTIONS = ["a_pe", "motorizada", "mecanizada", "blindada"]

IDX_LIDAR_AVAIL = 15
IDX_LIDAR_ELEV  = 16
LIDAR_INDICES   = [IDX_LIDAR_AVAIL, IDX_LIDAR_ELEV]

IDX_SLOPE = 1
IDX_NDVI  = 12
IDX_NDWI  = 13
IDX_WATER = 14

CLASS_WEIGHTS = {
    "a_pe":        torch.tensor([1.0, 3.0, 1.0]),
    "motorizada":  torch.tensor([1.0, 5.0, 1.0]),
    "mecanizada":  torch.tensor([1.0, 4.5, 1.0]),
    "blindada":    torch.tensor([1.0, 5.0, 1.0]),
}


# --- Model: MovementGNN ---
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
        if self.training:
            h = grad_ckpt(self.input_encoder, x, use_reentrant=False)
        else:
            h = self.input_encoder(x)
        for conv, norm in zip(self.convs, self.norms):
            h_in = h
            if self.training:
                h = grad_ckpt(conv, h, edge_index, edge_attr,
                              use_reentrant=False)
            else:
                h = conv(h, edge_index, edge_attr=edge_attr)
            h = F.leaky_relu(h, 0.2)
            h = h + h_in
            h = norm(h)
        return {frac: clf(h) for frac, clf in self.classifiers.items()}


# --- Loss: FocalLoss + DAMEPLANLoss ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha,
                             reduction="none", label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class DAMEPLANLoss(nn.Module):
    NDVI_THRESHOLDS = {"a_pe": 0.7, "motorizada": 0.5, "mecanizada": 0.6, "blindada": 0.5}

    def __init__(self, weight_ce=1.0, weight_phys=0.5, class_weights_per_frac=None):
        super().__init__()
        self.weight_ce = weight_ce
        self.weight_phys = weight_phys
        self.class_weights_per_frac = class_weights_per_frac or {}
        self.focal_cache = {}
        self.task_log_vars = nn.Parameter(torch.zeros(4))

    def _get_focal(self, frac, device):
        if frac not in self.focal_cache:
            cw = self.class_weights_per_frac.get(frac)
            if cw is not None:
                cw = cw.to(device)
            self.focal_cache[frac] = FocalLoss(alpha=cw, gamma=2.0)
        return self.focal_cache[frac]

    def forward(self, outputs, targets, features):
        total_loss = torch.tensor(0.0, device=features.device)
        metrics = {}
        for i, frac in enumerate(FRACTIONS):
            logits = outputs[frac]
            labels = targets[frac] - 1
            valid = (labels >= 0) & (labels < NUM_CLASSES)
            if not valid.any():
                continue
            focal = self._get_focal(frac, features.device)
            loss_t = self.weight_ce * focal(logits[valid], labels[valid])
            slope_loss = self._slope_pen(logits, features[:, IDX_SLOPE])
            veg_loss   = self._veg_pen(logits, features[:, IDX_NDVI], frac)
            water_loss = self._water_pen(logits, features[:, IDX_NDWI],
                                         features[:, IDX_WATER] if features.size(1) > IDX_WATER else torch.zeros_like(features[:, 0]))
            loss_t = loss_t + self.weight_phys * (slope_loss + veg_loss + water_loss)
            prec = torch.exp(-self.task_log_vars[i])
            total_loss = total_loss + prec * loss_t + self.task_log_vars[i]
            metrics[f"loss_{frac}"] = loss_t.item()
        return total_loss, metrics

    def _slope_pen(self, logits, slope):
        p = F.softmax(logits, dim=1)[:, 0]
        return (p * F.relu(slope - 0.35)).mean()

    def _veg_pen(self, logits, ndvi, frac):
        p = F.softmax(logits, dim=1)[:, 0]
        th = self.NDVI_THRESHOLDS.get(frac, 0.6)
        return (p * F.relu(ndvi - th)).mean()

    def _water_pen(self, logits, ndwi, water_mask):
        p = F.softmax(logits, dim=1)[:, 0]
        is_water = (ndwi > 0.3) | (water_mask > 0.5)
        return (p * is_water.float()).mean()


# --- Helpers ---
def log_vram():
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1e9
        r = torch.cuda.memory_reserved() / 1e9
        return f"VRAM {a:.2f}/{r:.2f} GB"
    return "CPU"


def detect_embedding_file(emb_dir):
    candidates = sorted(glob.glob(str(emb_dir / "pacaraima_q1q4_embeddings_*.pt")))
    return Path(candidates[-1]) if candidates else None


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch,
                    best_metrics, args_dict, raster_meta=None, graph_path=None):
    ckpt = {
        "epoch":        epoch,
        "timestamp":    datetime.now().isoformat(),
        "model_state":  model.state_dict(),
        "optim_state":  optimizer.state_dict(),
        "sched_state":  scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler else None,
        "best_metrics": best_metrics,
        "config":       args_dict,
        "raster_meta":  raster_meta,
        "graph_path":   str(graph_path) if graph_path else None,
    }
    torch.save(ckpt, path)


def unpack_batch(batch, is_hetero, device):
    """Extract DEM tensors from a batch (HeteroData or Data)."""
    if is_hetero:
        dem = batch["dem"]
        et = ("dem", "adjacent_to", "dem")
        x = dem.x.to(device)
        ei = batch[et].edge_index.to(device)
        ea = batch[et].edge_attr.to(device) if hasattr(batch[et], "edge_attr") and batch[et].edge_attr is not None else None
        n_seed = dem.batch_size if hasattr(dem, "batch_size") else x.size(0)
        n_id = dem.n_id if hasattr(dem, "n_id") else None
        targets = {}
        for frac in FRACTIONS:
            attr = f"y_{frac}"
            t = getattr(dem, attr, None)
            if t is not None:
                targets[frac] = t[:n_seed].to(device)
        mask_attr = getattr(dem, "train_mask", None)
        train_mask = mask_attr[:n_seed].to(device) if mask_attr is not None else None
    else:
        x = batch.x.to(device)
        ei = batch.edge_index.to(device)
        ea = batch.edge_attr.to(device) if hasattr(batch, "edge_attr") and batch.edge_attr is not None else None
        n_seed = batch.batch_size if hasattr(batch, "batch_size") else x.size(0)
        n_id = batch.n_id if hasattr(batch, "n_id") else None
        targets = {}
        for frac in FRACTIONS:
            t = getattr(batch, f"y_{frac}", None)
            if t is not None:
                targets[frac] = t[:n_seed].to(device)
        train_mask = batch.train_mask[:n_seed].to(device) if hasattr(batch, "train_mask") and batch.train_mask is not None else None
    return x, ei, ea, n_seed, n_id, targets, train_mask


def export_geotiff(preds_dict, raster_meta, pos, out_dir, tag="gnn"):
    """Export predictions as GeoTIFF (per force type + multiband)."""
    try:
        import rasterio
        from rasterio.transform import Affine
        from rasterio.crs import CRS
    except ImportError:
        print("rasterio not available - GeoTIFF not exported")
        return
    if raster_meta is None or pos is None:
        print("raster_meta or pos missing - GeoTIFF not exported")
        return
    H, W = raster_meta["dem_shape"]
    crs = CRS.from_user_input(raster_meta["crs"])
    t = raster_meta["dem_transform"]
    if isinstance(t, (list, tuple)):
        transform = Affine(*t[:6])
    elif isinstance(t, Affine):
        transform = t
    else:
        transform = Affine(t.a, t.b, t.c, t.d, t.e, t.f)
    pos_np = pos.cpu().numpy() if torch.is_tensor(pos) else pos
    lats, lons = pos_np[:, 0], pos_np[:, 1]
    rows_px = np.clip(((transform.f - lats) / abs(transform.e)).astype(int), 0, H - 1)
    cols_px = np.clip(((lons - transform.c) / abs(transform.a)).astype(int), 0, W - 1)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for frac, vals in preds_dict.items():
        arr = np.zeros((H, W), dtype=np.uint8)
        arr[rows_px, cols_px] = vals.astype(np.uint8)
        fname = f"restricao_{frac}_{tag}.tif"
        with rasterio.open(
            str(out_dir / fname), "w", driver="GTiff",
            height=H, width=W, count=1, dtype="uint8",
            crs=crs, transform=transform, compress="lzw",
        ) as dst:
            dst.write(arr, 1)
        print(f"  GeoTIFF: {fname}")
    bands = np.stack([preds_dict[f].astype(np.uint8) for f in FRACTIONS])
    n_bands = bands.shape[0]
    raster_multi = np.zeros((n_bands, H, W), dtype=np.uint8)
    for b in range(n_bands):
        raster_multi[b, rows_px, cols_px] = bands[b]
    fname_multi = f"restricao_multiband_{tag}.tif"
    with rasterio.open(
        str(out_dir / fname_multi), "w", driver="GTiff",
        height=H, width=W, count=n_bands, dtype="uint8",
        crs=crs, transform=transform, compress="lzw",
    ) as dst:
        for b in range(n_bands):
            dst.write(raster_multi[b], b + 1)
    print(f"  GeoTIFF multiband: {fname_multi}")


# --- Main: Resume + Fine-tune ---
def main():
    start_time = time.perf_counter()

    def log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        elapsed = time.perf_counter() - start_time
        print(f"[{ts}] (+{elapsed:5.0f}s) {msg}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    print("=" * 70)
    print("  MOVEMENT GNN V2 — FINE-TUNING (ARMORED LABEL REFINEMENT)")
    print("  Resume from best checkpoint + refined y_blindada labels")
    print("=" * 70)
    log(f"Device: {device} | AMP: {'bfloat16' if use_amp else 'OFF'}")
    if device == "cuda":
        gpu = torch.cuda.get_device_properties(0)
        log(f"GPU: {gpu.name} ({gpu.total_memory / 1e9:.1f} GB)")

    # --- 1. Load Graph (with refined labels) ---
    log(f"Loading {GRAPH_PATH.name} (refined armored labels) ...")
    data = torch.load(str(GRAPH_PATH), map_location="cpu", weights_only=False)
    is_hetero = hasattr(data, "node_types")
    raster_meta = getattr(data, "raster_meta", None)

    dem = data["dem"] if is_hetero else data
    n_nodes = dem.x.size(0)
    in_ch = dem.x.size(1)
    et_main = ("dem", "adjacent_to", "dem")
    n_edges = data[et_main].edge_index.size(1) if is_hetero else data.edge_index.size(1)

    if is_hetero:
        for et in data.edge_types:
            data[et].edge_index = data[et].edge_index.long()

    log(f"  DEM: {n_nodes:,} nodes, {in_ch} features, {n_edges:,} edges")

    if is_hetero:
        input_nodes = ("dem", None)
        num_neighbors = {et_main: [K_NEIGHBORS]}
        for other_et in data.edge_types:
            if other_et not in num_neighbors:
                num_neighbors[other_et] = [0]
    else:
        input_nodes = None
        num_neighbors = [K_NEIGHBORS]

    y_b = dem["y_blindada"].numpy()
    y_m = dem["y_motorizada"].numpy()
    assert not np.array_equal(y_b, y_m), "y_blindada must differ from y_motorizada"
    diff = int((y_b != y_m).sum())
    log(f"  y_blindada != y_motorizada: {diff:,} pixels differ — OK")

    for frac in FRACTIONS:
        y = dem[f"y_{frac}"]
        dist = [(y == c).sum().item() for c in [1, 2, 3]]
        log(f"  y_{frac}: Go={dist[0]:,} SlowGo={dist[1]:,} NoGo={dist[2]:,}")

    gc.collect()

    # --- 2. Embeddings ---
    emb_path = detect_embedding_file(EMB_DIR) if USE_EMBEDDINGS else None

    if emb_path and emb_path.exists():
        log(f"Loading embeddings: {emb_path.name} ...")
        emb_data = torch.load(str(emb_path), map_location="cpu", weights_only=False)
        emb = emb_data["embeddings"]
        emb_dim = emb.size(1)
        del emb_data; gc.collect()
        emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
        if is_hetero:
            data["dem"].x = torch.cat([data["dem"].x, emb], dim=1)
        else:
            data.x = torch.cat([data.x, emb], dim=1)
        in_ch += emb_dim
        del emb; gc.collect()
        log(f"  Features: {in_ch - emb_dim} base + {emb_dim} emb = {in_ch} total")

    # --- 2b. Remove LiDAR Features ---
    log(f"Removing LiDAR features (indices {LIDAR_INDICES}) ...")
    keep_mask = torch.ones(in_ch, dtype=torch.bool)
    for idx in LIDAR_INDICES:
        if idx < in_ch:
            keep_mask[idx] = False
    n_removed = (~keep_mask).sum().item()
    if is_hetero:
        data["dem"].x = data["dem"].x[:, keep_mask]
    else:
        data.x = data.x[:, keep_mask]
    in_ch -= n_removed
    log(f"  {n_removed} features removed -> {in_ch} final features")

    # --- 3. Sanitize + Convert to float16 ---
    feat = data["dem"].x if is_hetero else data.x
    n_nan = torch.isnan(feat).sum().item()
    n_inf = torch.isinf(feat).sum().item()
    if n_nan > 0 or n_inf > 0:
        clean = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        if is_hetero:
            data["dem"].x = clean
        else:
            data.x = clean

    if is_hetero:
        data["dem"].x = data["dem"].x.half()
        for et in data.edge_types:
            ea = getattr(data[et], "edge_attr", None)
            if ea is not None:
                data[et].edge_attr = ea.half()
    else:
        data.x = data.x.half()
    gc.collect()

    # --- 4. NeighborLoader ---
    log(f"Creating NeighborLoader (batch={BATCH_SIZE}, k={K_NEIGHBORS}) ...")
    train_loader = NeighborLoader(
        data, num_neighbors=num_neighbors, input_nodes=input_nodes,
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )
    val_loader = NeighborLoader(
        data, num_neighbors=num_neighbors, input_nodes=input_nodes,
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)
    log(f"  Train: ~{n_train_batches} batches | Val: ~{n_val_batches} batches")

    # --- 5. Model + Loss ---
    has_edge_attr = is_hetero and hasattr(data[et_main], "edge_attr") and data[et_main].edge_attr is not None
    edge_dim = data[et_main].edge_attr.size(1) if has_edge_attr else None

    model = MovementGNN(
        in_channels=in_ch, hidden_channels=HIDDEN_DIM,
        num_layers=NUM_LAYERS, heads=HEADS, num_classes=NUM_CLASSES,
        dropout=DROPOUT, edge_dim=edge_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log(f"  MovementGNN V2: {n_params:,} params")

    loss_fn = DAMEPLANLoss(
        weight_ce=1.0, weight_phys=0.5,
        class_weights_per_frac=CLASS_WEIGHTS,
    ).to(device)

    # --- 6. Load Checkpoint ---
    log(f"Loading checkpoint: {CHECKPOINT_PATH.name} ...")
    ckpt = torch.load(str(CHECKPOINT_PATH), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    start_epoch = ckpt["epoch"]
    log(f"  Checkpoint epoch {start_epoch} | val_loss={ckpt['best_metrics'].get('val_loss', '?')}")
    log(f"  Additional epochs: {EXTRA_EPOCHS} (ep {start_epoch+1} to {start_epoch+EXTRA_EPOCHS})")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EXTRA_EPOCHS, eta_min=1e-6)
    log(f"  LR: {LR} (fine-tune) | Scheduler: CosineAnnealingLR(T_max={EXTRA_EPOCHS})")

    scaler = torch.amp.GradScaler("cuda", enabled=False)

    # --- 7. Training Loop ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_dir = OUT_DIR / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    best_val_loss = ckpt["best_metrics"].get("val_loss", float("inf"))
    best_val_acc = ckpt["best_metrics"].get("val_acc", 0.0)
    es_counter = 0
    history = []

    config_dict = {
        "version": "v2_label_refinement",
        "changes": "resume with refined y_blindada labels, fine-tune LR",
        "base_checkpoint": str(CHECKPOINT_PATH),
        "base_epoch": start_epoch,
        "batch_size": BATCH_SIZE, "k_neighbors": K_NEIGHBORS,
        "num_layers": NUM_LAYERS, "hidden_dim": HIDDEN_DIM,
        "heads": HEADS, "dropout": DROPOUT, "lr": LR,
        "max_norm": MAX_NORM, "extra_epochs": EXTRA_EPOCHS,
        "in_channels": in_ch, "edge_dim": edge_dim, "n_params": n_params,
        "graph": str(GRAPH_PATH), "embeddings": str(emb_path) if emb_path else None,
        "class_weights": {f: w.tolist() for f, w in CLASS_WEIGHTS.items()},
        "removed_features": ["has_lidar", "z_lidar"],
    }

    log(f"Starting fine-tuning: {EXTRA_EPOCHS} epochs")
    print("=" * 70, flush=True)

    if device == "cuda":
        torch.cuda.empty_cache()

    dem_store = data["dem"] if is_hetero else data

    for ep_idx in range(1, EXTRA_EPOCHS + 1):
        epoch = start_epoch + ep_idx
        ep_start = time.perf_counter()
        model.train()

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        n_batches_done = 0
        n_nan_batches = 0

        for batch_idx, batch in enumerate(train_loader, 1):
            x, ei, ea, n_seed, n_id, targets, tmask = unpack_batch(batch, is_hetero, device)
            if not targets:
                continue

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(x, ei, ea)
                out_seed = {f: v[:n_seed] for f, v in outputs.items()}
                feat_seed = x[:n_seed]
                loss, lm = loss_fn(out_seed, targets, feat_seed)

            if torch.isnan(loss) or torch.isinf(loss):
                n_nan_batches += 1
                continue

            loss.backward()
            if MAX_NORM > 0:
                nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            optimizer.step()

            train_loss_sum += loss.item()
            n_batches_done += 1

            with torch.no_grad():
                for frac in FRACTIONS:
                    if frac in out_seed and frac in targets:
                        pred_c = out_seed[frac].argmax(dim=1) + 1
                        train_correct += (pred_c == targets[frac]).sum().item()
                        train_total += targets[frac].size(0)

            if batch_idx % max(1, n_train_batches // 5) == 0 or batch_idx == n_train_batches:
                avg_l = train_loss_sum / max(n_batches_done, 1)
                elapsed_b = time.perf_counter() - ep_start
                eta_s = elapsed_b / batch_idx * (n_train_batches - batch_idx)
                log(f"  Ep{epoch} batch {batch_idx}/{n_train_batches} "
                    f"loss={avg_l:.4f} acc={train_correct / max(train_total, 1):.3f} "
                    f"({elapsed_b:.0f}s / ETA {eta_s:.0f}s) {log_vram()}")

        avg_train_loss = train_loss_sum / max(n_batches_done, 1)
        scheduler.step()

        # --- Validation ---
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        n_val_done = 0
        all_val_preds = {f: [] for f in FRACTIONS}
        all_val_tgts = {f: [] for f in FRACTIONS}

        with torch.no_grad():
            for batch in val_loader:
                x, ei, ea, n_seed, n_id, targets, _ = unpack_batch(batch, is_hetero, device)
                if not targets:
                    continue
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    outputs = model(x, ei, ea)
                    out_seed = {f: v[:n_seed] for f, v in outputs.items()}
                    feat_seed = x[:n_seed]
                    loss, _ = loss_fn(out_seed, targets, feat_seed)

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss_sum += loss.item()
                    n_val_done += 1

                for frac in FRACTIONS:
                    if frac in out_seed and frac in targets:
                        pred_c = out_seed[frac].argmax(dim=1) + 1
                        val_correct += (pred_c == targets[frac]).sum().item()
                        val_total += targets[frac].size(0)
                        all_val_preds[frac].append(pred_c.cpu())
                        all_val_tgts[frac].append(targets[frac].cpu())

        avg_val_loss = val_loss_sum / max(n_val_done, 1)
        val_acc = val_correct / max(val_total, 1)
        lr_now = optimizer.param_groups[0]["lr"]
        ep_elapsed = time.perf_counter() - ep_start

        f1_scores = {}
        for frac in FRACTIONS:
            if all_val_preds[frac]:
                p = torch.cat(all_val_preds[frac]).numpy()
                t = torch.cat(all_val_tgts[frac]).numpy()
                f1_scores[frac] = f1_score(t, p, average="macro", zero_division=0)

        f1_str = " ".join([f"{f[:3]}={f1_scores.get(f, 0):.3f}" for f in FRACTIONS])
        log(f"Epoch {epoch} ({ep_idx}/{EXTRA_EPOCHS}) | "
            f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} | "
            f"val_acc={val_acc:.3f} | F1: {f1_str} | "
            f"lr={lr_now:.1e} | {ep_elapsed:.0f}s"
            + (f" | {n_nan_batches} NaN" if n_nan_batches else ""))

        row = {
            "epoch": epoch, "epoch_idx": ep_idx,
            "train_loss": round(avg_train_loss, 5),
            "val_loss": round(avg_val_loss, 5), "val_acc": round(val_acc, 4),
            "lr": lr_now, "elapsed_s": round(ep_elapsed, 1),
            "nan_batches": n_nan_batches,
        }
        row.update({f"f1_{f}": round(f1_scores.get(f, 0), 4) for f in FRACTIONS})
        history.append(row)

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            es_counter = 0
            save_checkpoint(ckpt_dir / "checkpoint_best.pt", model, optimizer,
                            scheduler, scaler, epoch,
                            {"val_loss": avg_val_loss, "val_acc": val_acc, "f1": f1_scores},
                            config_dict, raster_meta, GRAPH_PATH)
            log(f"  BEST checkpoint saved (val_loss={avg_val_loss:.5f})")
        else:
            es_counter += 1

        save_checkpoint(ckpt_dir / "checkpoint_latest.pt", model, optimizer,
                        scheduler, scaler, epoch,
                        {"val_loss": avg_val_loss, "val_acc": val_acc},
                        config_dict, raster_meta, GRAPH_PATH)

        if CKPT_EVERY > 0 and ep_idx % CKPT_EVERY == 0:
            save_checkpoint(ckpt_dir / f"checkpoint_ep{epoch:03d}.pt", model,
                            optimizer, scheduler, scaler, epoch,
                            {"val_loss": avg_val_loss}, config_dict, raster_meta, GRAPH_PATH)

        if ES_PATIENCE > 0 and es_counter >= ES_PATIENCE:
            log(f"  EARLY STOP (no improvement for {ES_PATIENCE} epochs)")
            break

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        print("-" * 70, flush=True)

    # --- 8. Full Inference + Export ---
    log("Loading best checkpoint for full inference...")
    best_ckpt_path = ckpt_dir / "checkpoint_best.pt"
    if best_ckpt_path.exists():
        ckpt = torch.load(str(best_ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        log(f"  Best epoch: {ckpt['epoch']} | val_loss={ckpt['best_metrics'].get('val_loss', '?')}")
    else:
        log("  WARNING: best checkpoint not found, using latest state")

    model.eval()
    log("Full inference (all nodes)...")

    all_preds = {f: torch.zeros(n_nodes, dtype=torch.long) for f in FRACTIONS}
    all_probs = {f: torch.zeros(n_nodes, NUM_CLASSES) for f in FRACTIONS}
    covered = torch.zeros(n_nodes, dtype=torch.bool)

    inf_loader = NeighborLoader(
        data, num_neighbors=num_neighbors, input_nodes=input_nodes,
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    with torch.no_grad():
        for batch in inf_loader:
            x, ei, ea, n_seed, n_id, _, _ = unpack_batch(batch, is_hetero, device)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(x, ei, ea)
            for frac in FRACTIONS:
                logits = outputs[frac][:n_seed]
                probs = F.softmax(logits, dim=1).float().cpu()
                preds = probs.argmax(dim=1) + 1
                if n_id is not None:
                    ids = n_id[:n_seed].cpu()
                    all_preds[frac][ids] = preds
                    all_probs[frac][ids] = probs
                    covered[ids] = True

    log(f"  Coverage: {covered.sum().item():,}/{n_nodes:,} nodes "
        f"({covered.float().mean() * 100:.1f}%)")

    preds_np = {f: all_preds[f].numpy() for f in FRACTIONS}
    np.savez_compressed(str(OUT_DIR / "predictions_full.npz"), **preds_np)
    log(f"  predictions_full.npz saved")

    probs_np = {f"probs_{f}": all_probs[f].numpy() for f in FRACTIONS}
    np.savez_compressed(str(OUT_DIR / "probs_full.npz"), **probs_np)
    log(f"  probs_full.npz saved")

    final_metrics = {}
    for frac in FRACTIONS:
        y_true = dem_store[f"y_{frac}"].numpy()
        y_pred = preds_np[frac]
        valid = (y_true >= 1) & (y_true <= 3) & covered.numpy()
        if valid.sum() > 0:
            acc = accuracy_score(y_true[valid], y_pred[valid])
            f1 = f1_score(y_true[valid], y_pred[valid], average="macro", zero_division=0)
            cm = confusion_matrix(y_true[valid], y_pred[valid], labels=[1, 2, 3])
            final_metrics[frac] = {
                "accuracy": round(float(acc), 4),
                "f1_macro": round(float(f1), 4),
                "confusion_matrix": cm.tolist(),
                "n_valid": int(valid.sum()),
            }
            log(f"  {frac}: acc={acc:.3f} f1={f1:.3f}")

    final_metrics["config"] = config_dict
    final_metrics["training_history"] = history
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    log(f"  metrics.json saved")

    pos = getattr(dem_store, "pos", None)
    if pos is None:
        try:
            pos = dem_store["pos"]
        except:
            pass
    export_geotiff(preds_np, raster_meta, pos, OUT_DIR, tag="movement_gnn_v2_refined")

    total_time = time.perf_counter() - start_time
    report = [
        "=" * 70,
        "FINAL REPORT - MovementGNN V2 FINE-TUNING",
        "=" * 70,
        f"Resumed from: {CHECKPOINT_PATH.name} (epoch {start_epoch})",
        f"Label refinement: y_blindada updated",
        f"Total time: {total_time:.0f}s ({total_time / 60:.1f} min)",
        f"Best val_loss: {best_val_loss:.5f}",
        f"Best val_acc: {best_val_acc:.3f}",
        f"Additional epochs: {EXTRA_EPOCHS} | LR: {LR}",
        f"Results in: {OUT_DIR}",
        "=" * 70,
    ]
    report_txt = "\n".join(report)
    print(report_txt)
    with open(OUT_DIR / "report.txt", "w") as f:
        f.write(report_txt)

    log("Done.")


if __name__ == "__main__":
    main()
