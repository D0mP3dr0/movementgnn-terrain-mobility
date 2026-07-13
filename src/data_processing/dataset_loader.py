"""
DatasetLoader - Loads and processes the terrain dataset.

TerrainGNN features (17 dimensions):
    0: elevation       - SRTM elevation (Z-score)
    1: slope           - Inclination [0,1]
    2: aspect_cos      - Aspect cosine [-1,1]
    3: aspect_sin      - Aspect sine [-1,1]
    4: curvature       - Terrain curvature
    5: tpi             - Topographic Position Index
    6: tri             - Terrain Ruggedness Index
    7: roughness       - Local roughness
    8: b02_norm        - Sentinel-2 Blue [0,1]
    9: b03_norm        - Sentinel-2 Green [0,1]
    10: b04_norm       - Sentinel-2 Red [0,1]
    11: b08_norm       - Sentinel-2 NIR [0,1]
    12: ndvi           - NDVI [-1,1]
    13: ndwi           - NDWI [-1,1]
    14: water_mask     - Binary water mask
    15: lidar_available - LiDAR availability flag
    16: lidar_elevation - LiDAR elevation (when available)
"""

import os
import torch
import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetInfo:
    """Information about the loaded dataset."""
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
    """
    Loader for the terrain dataset used in the movement restriction pipeline.

    Validates structure on load:
    - HeteroData with 'dem' node type
    - 17 DEM features
    """

    EXPECTED_DEM_FEATURES = 17
    EXPECTED_SENTINEL_FEATURES = 8
    EXPECTED_LIDAR_FEATURES = 3

    DEM_FEATURE_NAMES = [
        'elevation', 'slope', 'aspect_cos', 'aspect_sin', 'curvature',
        'tpi', 'tri', 'roughness', 'b02_norm', 'b03_norm', 'b04_norm',
        'b08_norm', 'ndvi', 'ndwi', 'water_mask', 'lidar_available', 'lidar_elevation'
    ]

    def __init__(self, dataset_path: str, device: str = 'cpu'):
        """
        Args:
            dataset_path: Path to the .pt dataset file.
            device: Device to load onto ('cpu' or 'cuda').
        """
        self.dataset_path = Path(dataset_path)
        self.device = device
        self.data = None
        self.info = None

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    def load(self) -> Any:
        """
        Load the dataset and validate its structure.

        Returns:
            HeteroData with the loaded graph.
        """
        print(f"Loading dataset: {self.dataset_path.name}")
        print(f"Size: {self.dataset_path.stat().st_size / (1024**3):.2f} GB")

        from torch_geometric.data import HeteroData

        self.data = torch.load(
            str(self.dataset_path),
            map_location=self.device,
            weights_only=False
        )

        self._validate_structure()
        self.info = self._extract_info()

        print(f"\nDataset loaded successfully.")
        print(f"  DEM nodes: {self.info.num_dem_nodes:,}")
        print(f"  DEM features: {self.info.num_dem_features}")

        return self.data

    def _validate_structure(self):
        """Validate dataset structure."""
        from torch_geometric.data import HeteroData

        if not isinstance(self.data, HeteroData):
            raise ValueError("Dataset must be HeteroData.")

        if 'dem' not in self.data.node_types:
            raise ValueError("Dataset must have node type 'dem'.")

        dem_x = self.data['dem'].x
        if dem_x.shape[1] != self.EXPECTED_DEM_FEATURES:
            print(f"WARNING: Expected {self.EXPECTED_DEM_FEATURES} DEM features, found {dem_x.shape[1]}")

    def _extract_info(self) -> DatasetInfo:
        """Extract dataset metadata."""
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
        """
        Return DEM features (17 dimensions) for classification.

        Returns:
            Tensor [N, 17] with topographic features.
        """
        if self.data is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        return self.data['dem'].x

    def get_dem_edges(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Return edge_index and edge_attr for the DEM graph.

        Returns:
            Tuple (edge_index, edge_attr)
        """
        if self.data is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        for edge_type in self.data.edge_types:
            if edge_type[0] == 'dem' and edge_type[2] == 'dem':
                edge_data = self.data[edge_type]
                edge_index = edge_data.edge_index
                edge_attr = edge_data.edge_attr if hasattr(edge_data, 'edge_attr') else None
                return edge_index, edge_attr

        return None, None

    def get_positions(self) -> Optional[torch.Tensor]:
        """
        Return node positions (lat, lon).

        Returns:
            Tensor [N, 2] with coordinates, or None.
        """
        if self.data is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        if hasattr(self.data['dem'], 'pos'):
            return self.data['dem'].pos
        return None

    def get_feature_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute feature statistics for validation.

        Returns:
            Dict with min, max, mean, std per feature.
        """
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
        """Print dataset summary."""
        if self.info is None:
            print("Dataset not loaded.")
            return

        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        print(f"\nFile: {self.dataset_path.name}")
        print(f"Size: {self.info.file_size_gb:.2f} GB")
        print(f"\nDEM nodes: {self.info.num_dem_nodes:,} ({self.info.num_dem_features} features)")
        print(f"Sentinel nodes: {self.info.num_sentinel_nodes:,} ({self.info.num_sentinel_features} features)")
        print(f"LiDAR nodes: {self.info.num_lidar_nodes:,} ({self.info.num_lidar_features} features)")

        if self.info.lat_range != (0.0, 0.0):
            print(f"\nLatitude: [{self.info.lat_range[0]:.4f}, {self.info.lat_range[1]:.4f}]")
            print(f"Longitude: [{self.info.lon_range[0]:.4f}, {self.info.lon_range[1]:.4f}]")

        print(f"\nDEM features: {self.info.num_dem_features}/17 expected")
        if self.info.num_dem_features == 17:
            print("  Schema aligned.")
        else:
            print("  WARNING: Feature count mismatch.")

        print("=" * 60)


def load_dataset(
    path: str,
    device: str = 'cpu'
) -> Tuple[Any, DatasetInfo]:
    """
    Convenience function to load the dataset.

    Args:
        path: Path to the dataset .pt file.
        device: Device ('cpu' or 'cuda').

    Returns:
        Tuple (data, info)
    """
    loader = DatasetLoader(path, device)
    data = loader.load()
    loader.print_summary()

    return data, loader.info
