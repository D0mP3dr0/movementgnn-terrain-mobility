INTERNAL_DATA_ROOT = __import__("os").environ.get("MOVEMENTGNN_DATA_ROOT", ".")
import os
import torch
import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetInfo:
    num_dem_nodes: int
    num_sentinel_nodes: int
    num_lidar_nodes: int
    num_dem_features: int
    num_sentinel_features: int
    num_lidar_features: int
    lat_range: Tuple[float, float]
    lon_range: Tuple[float, float]
    file_size_gb: float


class DatasetLoader:
    
    EXPECTED_DEM_FEATURES = 17
    EXPECTED_SENTINEL_FEATURES = 8
    EXPECTED_LIDAR_FEATURES = 3
    
    DEM_FEATURE_NAMES = [
        'elevation', 'slope', 'aspect_cos', 'aspect_sin', 'curvature',
        'tpi', 'tri', 'roughness', 'b02_norm', 'b03_norm', 'b04_norm',
        'b08_norm', 'ndvi', 'ndwi', 'water_mask', 'lidar_available', 'lidar_elevation'
    ]
    
    def __init__(self, dataset_path: str, device: str = 'cpu'):
        self.dataset_path = Path(dataset_path)
        self.device = device
        self.data = None
        self.info = None
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    def load(self) -> Any:
        print(f"Carregando dataset: {self.dataset_path.name}")
        print(f"Tamanho: {self.dataset_path.stat().st_size / (1024**3):.2f} GB")
        
        from torch_geometric.data import HeteroData
        
        self.data = torch.load(
            str(self.dataset_path), 
            map_location=self.device,
            weights_only=False
        )
        
        self._validate_structure()
        
        self.info = self._extract_info()
        
        print(f"\n✓ Dataset loaded successfully!")
        print(f"  DEM nodes: {self.info.num_dem_nodes:,}")
        print(f"  DEM features: {self.info.num_dem_features}")
        
        return self.data
    
    def _validate_structure(self):
        from torch_geometric.data import HeteroData
        
        if not isinstance(self.data, HeteroData):
            raise ValueError("Dataset deve ser HeteroData (conforme RAG topo_rf_v2_system.md)")
        
        if 'dem' not in self.data.node_types:
            raise ValueError("Dataset deve ter node type 'dem'")
        
        dem_x = self.data['dem'].x
        if dem_x.shape[1] != self.EXPECTED_DEM_FEATURES:
            print(f"⚠️ SPEC WARNING: expected {self.EXPECTED_DEM_FEATURES} DEM features, found {dem_x.shape[1]}")
    
    def _extract_info(self) -> DatasetInfo:
        dem_x = self.data['dem'].x if 'dem' in self.data.node_types else None
        sentinel_x = self.data['sentinel'].x if 'sentinel' in self.data.node_types else None
        lidar_x = self.data['lidar'].x if 'lidar' in self.data.node_types else None
        
        if hasattr(self.data['dem'], 'pos') and self.data['dem'].pos is not None:
            pos = self.data['dem'].pos
            lat_range = (pos[:, 0].min().item(), pos[:, 0].max().item())
            lon_range = (pos[:, 1].min().item(), pos[:, 1].max().item())
        else:
            lat_range = (0.0, 0.0)
            lon_range = (0.0, 0.0)
        
        return DatasetInfo(
            num_dem_nodes=dem_x.shape[0] if dem_x is not None else 0,
            num_sentinel_nodes=sentinel_x.shape[0] if sentinel_x is not None else 0,
            num_lidar_nodes=lidar_x.shape[0] if lidar_x is not None else 0,
            num_dem_features=dem_x.shape[1] if dem_x is not None else 0,
            num_sentinel_features=sentinel_x.shape[1] if sentinel_x is not None else 0,
            num_lidar_features=lidar_x.shape[1] if lidar_x is not None else 0,
            lat_range=lat_range,
            lon_range=lon_range,
            file_size_gb=self.dataset_path.stat().st_size / (1024**3)
        )
    
    def get_dem_features(self) -> torch.Tensor:
        if self.data is None:
            raise RuntimeError('Dataset not loaded. Call load() first.')
        
        return self.data['dem'].x
    
    def get_dem_edges(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.data is None:
            raise RuntimeError('Dataset not loaded. Call load() first.')
        
        for edge_type in self.data.edge_types:
            if edge_type[0] == 'dem' and edge_type[2] == 'dem':
                edge_data = self.data[edge_type]
                edge_index = edge_data.edge_index
                edge_attr = edge_data.edge_attr if hasattr(edge_data, 'edge_attr') else None
                return edge_index, edge_attr
        
        return None, None
    
    def get_positions(self) -> Optional[torch.Tensor]:
        if self.data is None:
            raise RuntimeError('Dataset not loaded. Call load() first.')
        
        if hasattr(self.data['dem'], 'pos'):
            return self.data['dem'].pos
        return None
    
    def get_feature_statistics(self) -> Dict[str, Dict[str, float]]:
        dem_x = self.get_dem_features()
        
        stats = {}
        for i, name in enumerate(self.DEM_FEATURE_NAMES):
            if i < dem_x.shape[1]:
                col = dem_x[:, i]
                stats[name] = {
                    'min': col.min().item(),
                    'max': col.max().item(),
                    'mean': col.mean().item(),
                    'std': col.std().item(),
                }
        
        return stats
    
    def print_summary(self):
        if self.info is None:
            print("Dataset não carregado.")
            return
        
        print("\n" + "=" * 60)
        print("RESUMO DO DATASET PIRASSUNUNGA")
        print("=" * 60)
        print(f"\nArquivo: {self.dataset_path.name}")
        print(f"Tamanho: {self.info.file_size_gb:.2f} GB")
        print(f"\nDEM nodes: {self.info.num_dem_nodes:,} ({self.info.num_dem_features} features)")
        print(f"Sentinel nodes: {self.info.num_sentinel_nodes:,} ({self.info.num_sentinel_features} features)")
        print(f"LiDAR nodes: {self.info.num_lidar_nodes:,} ({self.info.num_lidar_features} features)")
        
        if self.info.lat_range != (0.0, 0.0):
            print(f"\nLatitude: [{self.info.lat_range[0]:.4f}, {self.info.lat_range[1]:.4f}]")
            print(f"Longitude: [{self.info.lon_range[0]:.4f}, {self.info.lon_range[1]:.4f}]")
        
        print("\n⚠️ VERIFICAÇÃO RAG:")
        print(f"  - Features DEM: {self.info.num_dem_features}/17 (esperado)")
        if self.info.num_dem_features == 17:
            print("  ✓ Alinhado com architecture_v21.md")
        else:
            print("  ⚠ Verificar alinhamento com architecture_v21.md")
        
        print("=" * 60)


def load_pirassununga_dataset(
    path: str = None,
    device: str = 'cpu'
) -> Tuple[Any, DatasetInfo]:
    if path is None:
        path = rf"{INTERNAL_DATA_ROOT}\GNN_RESTRICAO_MOVIMENTO\DATA_GRAPH\interior_sp_v18.4_part_Pirassununga_cupy.pt"
    
    loader = DatasetLoader(path, device)
    data = loader.load()
    loader.print_summary()
    
    return data, loader.info


if __name__ == "__main__":
    data, info = load_pirassununga_dataset()
