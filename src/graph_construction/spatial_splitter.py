INTERNAL_DATA_ROOT = __import__("os").environ.get("MOVEMENTGNN_DATA_ROOT", ".")
import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path
from dataclasses import dataclass


@dataclass
class SpatialSplit:
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    train_count: int
    val_count: int
    test_count: int
    grid_assignments: np.ndarray


class SpatialSplitter:
    
    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        grid_size_km: float = 5.0,
        buffer_km: float = 1.0,
        random_seed: int = 42
    ):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.grid_size_km = grid_size_km
        self.buffer_km = buffer_km
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
    
    def create_grid_assignments(
        self,
        positions: torch.Tensor
    ) -> np.ndarray:
        pos = positions.numpy() if isinstance(positions, torch.Tensor) else positions
        
        lat_step = self.grid_size_km / 111.0
        lon_step = self.grid_size_km / 85.0
        
        lat_idx = ((pos[:, 0] - pos[:, 0].min()) / lat_step).astype(int)
        lon_idx = ((pos[:, 1] - pos[:, 1].min()) / lon_step).astype(int)
        
        grid_ids = lat_idx * 1000 + lon_idx
        
        return grid_ids
    
    def split_grids(
        self,
        unique_grids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_grids = len(unique_grids)
        
        shuffled = unique_grids.copy()
        np.random.shuffle(shuffled)
        
        n_train = int(n_grids * self.train_ratio)
        n_val = int(n_grids * self.val_ratio)
        
        train_grids = set(shuffled[:n_train])
        val_grids = set(shuffled[n_train:n_train + n_val])
        test_grids = set(shuffled[n_train + n_val:])
        
        return train_grids, val_grids, test_grids
    
    def apply_buffer(
        self,
        positions: torch.Tensor,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        test_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.buffer_km <= 0:
            return train_mask, val_mask, test_mask
        
        pos = positions.numpy()
        
        buffer_deg = self.buffer_km / 100.0
        
        train_pos = pos[train_mask.numpy()]
        
        for mask in [val_mask, test_mask]:
            if mask.sum() == 0:
                continue
            
            mask_indices = torch.where(mask)[0]
            
            for idx in mask_indices[:1000]:
                node_pos = pos[idx]
                dists = np.sqrt(np.sum((train_pos - node_pos) ** 2, axis=1))
                min_dist = dists.min()
                
                if min_dist < buffer_deg:
                    mask[idx] = False
        
        return train_mask, val_mask, test_mask
    
    def create_split(
        self,
        positions: torch.Tensor,
        apply_buffer: bool = True
    ) -> SpatialSplit:
        N = positions.shape[0]
        print(f"Creating spatial split for {N:,} nodes...")
        
        grid_ids = self.create_grid_assignments(positions)
        unique_grids = np.unique(grid_ids)
        print(f"  Grid: {len(unique_grids)} células")
        
        train_grids, val_grids, test_grids = self.split_grids(unique_grids)
        print(f"  Train grids: {len(train_grids)}")
        print(f"  Val grids: {len(val_grids)}")
        print(f"  Test grids: {len(test_grids)}")
        
        train_mask = torch.tensor([g in train_grids for g in grid_ids])
        val_mask = torch.tensor([g in val_grids for g in grid_ids])
        test_mask = torch.tensor([g in test_grids for g in grid_ids])
        
        if apply_buffer and self.buffer_km > 0:
            print(f"  Aplicando buffer de {self.buffer_km} km...")
            train_mask, val_mask, test_mask = self.apply_buffer(
                positions, train_mask, val_mask, test_mask
            )
        
        result = SpatialSplit(
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            train_count=train_mask.sum().item(),
            val_count=val_mask.sum().item(),
            test_count=test_mask.sum().item(),
            grid_assignments=grid_ids
        )
        
        print(f"\n  ✓ Train: {result.train_count:,} ({result.train_count/N*100:.1f}%)")
        print(f"  ✓ Val: {result.val_count:,} ({result.val_count/N*100:.1f}%)")
        print(f"  ✓ Test: {result.test_count:,} ({result.test_count/N*100:.1f}%)")
        
        return result
    
    def add_splits_to_graph(
        self,
        data,
        split: SpatialSplit
    ):
        from torch_geometric.data import HeteroData
        
        if isinstance(data, HeteroData):
            data['dem'].train_mask = split.train_mask
            data['dem'].val_mask = split.val_mask
            data['dem'].test_mask = split.test_mask
        else:
            data.train_mask = split.train_mask
            data.val_mask = split.val_mask
            data.test_mask = split.test_mask
        
        print('✓ Split masks added to the graph')


def create_spatial_splits(
    graph_path: str = None,
    output_path: str = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> str:
    from torch_geometric.data import HeteroData
    
    if graph_path is None:
        graph_path = rf"{INTERNAL_DATA_ROOT}\GNN_RESTRICAO_MOVIMENTO\data\graphs\pirassununga_with_labels.pt"
    
    if output_path is None:
        output_path = rf"{INTERNAL_DATA_ROOT}\GNN_RESTRICAO_MOVIMENTO\data\graphs\pirassununga_ready.pt"
    
    print(f"Carregando: {graph_path}")
    data = torch.load(graph_path, map_location='cpu', weights_only=False)
    
    if isinstance(data, HeteroData):
        if hasattr(data['dem'], 'pos') and data['dem'].pos is not None:
            positions = data['dem'].pos
        else:
            N = data['dem'].x.shape[0]
            side = int(np.sqrt(N))
            x = torch.arange(N) % side
            y = torch.arange(N) // side
            positions = torch.stack([y.float(), x.float()], dim=1)
    else:
        positions = data.pos if hasattr(data, 'pos') else None
    
    if positions is None:
        raise ValueError('Graph has no positions (pos)')
    
    splitter = SpatialSplitter(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        grid_size_km=5.0,
        buffer_km=0.5
    )
    
    split = splitter.create_split(positions, apply_buffer=False)
    splitter.add_splits_to_graph(data, split)
    
    print(f"\nSalvando: {output_path}")
    torch.save(data, output_path)
    
    size_gb = Path(output_path).stat().st_size / (1024**3)
    print(f"✓ Tamanho: {size_gb:.2f} GB")
    
    return output_path


if __name__ == "__main__":
    print('Demonstration of spatial splitting\n')
    print("=" * 50)
    
    N = 10000
    lat = torch.rand(N) * 0.5 - 22.0
    lon = torch.rand(N) * 0.5 - 47.5
    positions = torch.stack([lat, lon], dim=1)
    
    splitter = SpatialSplitter(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        grid_size_km=3.0
    )
    
    split = splitter.create_split(positions, apply_buffer=False)
    
    print("\n" + "=" * 50)
    print("RESUMO:")
    print(f"  Total: {N:,}")
    print(f"  Train: {split.train_count:,} ({split.train_count/N*100:.1f}%)")
    print(f"  Val: {split.val_count:,} ({split.val_count/N*100:.1f}%)")
    print(f"  Test: {split.test_count:,} ({split.test_count/N*100:.1f}%)")
