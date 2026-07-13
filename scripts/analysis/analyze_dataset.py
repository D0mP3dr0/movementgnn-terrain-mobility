"""
Complete analysis of HeteroData dataset for the movement restriction pipeline.
Supports both Data (homogeneous) and HeteroData (heterogeneous).
"""

import torch
import os
import numpy as np


def _tensor_stats(t: torch.Tensor, name: str, max_features: int = 30):
    """Print detailed tensor statistics."""
    print(f"\n  --- {name} ---")
    print(f"  Shape: {t.shape}  |  Dtype: {t.dtype}")

    if t.numel() == 0:
        print("  (empty)")
        return

    nan_count = torch.isnan(t).sum().item()
    inf_count = torch.isinf(t).sum().item()
    if nan_count or inf_count:
        print(f"  *** NaN: {nan_count:,}  |  Inf: {inf_count:,} ***")

    if t.ndim == 1:
        finite = t[torch.isfinite(t)]
        if finite.numel() > 0:
            print(f"  min={finite.min().item():.6f}  max={finite.max().item():.6f}  "
                  f"mean={finite.float().mean().item():.6f}  std={finite.float().std().item():.6f}")
        if t.dtype in (torch.long, torch.int, torch.int32, torch.int16):
            uniq = t.unique()
            if uniq.numel() <= 20:
                counts = {v.item(): (t == v).sum().item() for v in uniq}
                print(f"  Distribution ({uniq.numel()} classes): {counts}")
            else:
                print(f"  Unique values: {uniq.numel():,}")
        return

    if t.ndim >= 2:
        n_feat = t.shape[1] if t.ndim == 2 else t.shape[-1]
        show = min(n_feat, max_features)
        print(f"  Feature stats (first {show} of {n_feat}):")
        for i in range(show):
            col = t[:, i] if t.ndim == 2 else t[..., i].flatten()
            finite = col[torch.isfinite(col)]
            if finite.numel() == 0:
                print(f"    [{i:3d}] *** all NaN/Inf ***")
                continue
            zeros = (finite == 0).sum().item()
            pct_zero = 100.0 * zeros / finite.numel()
            print(f"    [{i:3d}] min={finite.min().item():12.6f}  max={finite.max().item():12.6f}  "
                  f"mean={finite.float().mean().item():12.6f}  std={finite.float().std().item():12.6f}  "
                  f"zeros={pct_zero:5.1f}%")


def analyze_dataset(path: str):
    """Analyze complete structure of a PyTorch Geometric dataset."""

    print("=" * 70)
    print("COMPLETE DATASET ANALYSIS")
    print("=" * 70)
    print(f"\nFile: {path}")
    print(f"Size: {os.path.getsize(path) / (1024**3):.2f} GB")

    print("\nLoading dataset...")

    try:
        from torch_geometric.data import Data, HeteroData
    except ImportError:
        print("ERROR: torch_geometric not installed")
        return

    data = torch.load(path, map_location='cpu', weights_only=False)
    print(f"Type: {type(data).__name__}")

    if isinstance(data, HeteroData):
        _analyze_hetero(data)
    else:
        _analyze_homo(data)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    return data


def _analyze_homo(data):
    """Analysis for homogeneous Data."""
    if hasattr(data, 'x') and data.x is not None:
        _tensor_stats(data.x, "NODE FEATURES (x)")
    if hasattr(data, 'edge_index') and data.edge_index is not None:
        print(f"\n  --- EDGES ---")
        print(f"  Shape: {data.edge_index.shape}  |  Num edges: {data.edge_index.shape[1]:,}")
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        _tensor_stats(data.edge_attr, "EDGE FEATURES (edge_attr)")
    if hasattr(data, 'pos') and data.pos is not None:
        _tensor_stats(data.pos, "POSITIONS (pos)")
    if hasattr(data, 'y') and data.y is not None:
        _tensor_stats(data.y, "TARGETS (y)")


def _analyze_hetero(data):
    """Complete analysis for HeteroData."""
    from torch_geometric.data import HeteroData

    total_nodes = 0
    total_edges = 0

    print(f"\n{'='*70}")
    print(f"NODE TYPES: {data.node_types}")
    print(f"{'='*70}")

    for nt in data.node_types:
        store = data[nt]
        n_nodes = store.num_nodes if hasattr(store, 'num_nodes') and store.num_nodes else 0
        if hasattr(store, 'x') and store.x is not None:
            n_nodes = store.x.shape[0]
        total_nodes += n_nodes
        print(f"\n{'-'*70}")
        print(f"  NODE TYPE: {nt}  ({n_nodes:,} nodes)")
        print(f"{'-'*70}")

        for key in store.keys():
            attr = store[key]
            if isinstance(attr, torch.Tensor):
                _tensor_stats(attr, f"{nt}.{key}")
            elif attr is not None:
                print(f"  {nt}.{key}: {type(attr).__name__} = {attr}")

    print(f"\n{'='*70}")
    print(f"EDGE TYPES: {data.edge_types}")
    print(f"{'='*70}")

    for et in data.edge_types:
        store = data[et]
        n_edges = 0
        if hasattr(store, 'edge_index') and store.edge_index is not None:
            n_edges = store.edge_index.shape[1]
        total_edges += n_edges

        label = f"({et[0]}, {et[1]}, {et[2]})"
        print(f"\n{'-'*70}")
        print(f"  EDGE TYPE: {label}  ({n_edges:,} edges)")
        print(f"{'-'*70}")

        for key in store.keys():
            attr = store[key]
            if isinstance(attr, torch.Tensor):
                _tensor_stats(attr, f"{label}.{key}")

    print(f"\n{'='*70}")
    print(f"GLOBAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Total nodes:  {total_nodes:,}")
    print(f"  Total edges:  {total_edges:,}")
    print(f"  Node types: {len(data.node_types)}")
    print(f"  Edge types: {len(data.edge_types)}")

    root_keys = set()
    for key in data.keys():
        if isinstance(key, str):
            root_keys.add(key)
    if root_keys:
        print(f"\n  Root attributes: {sorted(root_keys)}")
        for key in sorted(root_keys):
            attr = data[key]
            if isinstance(attr, torch.Tensor):
                _tensor_stats(attr, f"root.{key}")
            elif attr is not None:
                print(f"  root.{key}: {type(attr).__name__}")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    _repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_repo))
    from src.paths import default_graph_path

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = str(default_graph_path())
    analyze_dataset(path)
