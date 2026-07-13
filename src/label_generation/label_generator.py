"""
LabelGenerator - Generates movement restriction labels for the dataset.
Applies DAMEPLAN rules to graph nodes.
"""

import torch
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime

from .dameplan_rules import DAMEPLANRules, RestrictionClass


class LabelGenerator:
    """
    Movement restriction label generator.

    Applies DAMEPLAN rules to the TerrainGNN dataset to produce
    ground-truth classification labels per military fraction.
    """

    # Feature indices in the DEM tensor
    FEATURE_INDICES = {
        'elevation': 0,
        'slope': 1,          # Normalized [0, 1]
        'ndvi': 12,
        'ndwi': 13,
        'water_mask': 14,
    }

    def __init__(self, rules: DAMEPLANRules = None):
        """
        Args:
            rules: DAMEPLANRules instance (uses default if None).
        """
        self.rules = rules or DAMEPLANRules()
        self.stats = {}

    def generate_from_features(
        self,
        dem_features: torch.Tensor,
        denormalize_slope: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Generate labels for all fractions from DEM features.

        Args:
            dem_features: Tensor [N, 17] with DEM features.
            denormalize_slope: If True, converts slope from [0,1] to degrees.

        Returns:
            Dict with label tensors per fraction.
        """
        N = dem_features.shape[0]
        print(f"Generating labels for {N:,} nodes...")

        slope = dem_features[:, self.FEATURE_INDICES['slope']]
        ndvi = dem_features[:, self.FEATURE_INDICES['ndvi']]
        ndwi = dem_features[:, self.FEATURE_INDICES['ndwi']]
        water_mask = dem_features[:, self.FEATURE_INDICES['water_mask']]

        if denormalize_slope:
            slope = slope * 90.0  # [0, 1] -> [0, 90] degrees

        labels = self.rules.classify_all_fractions(
            slope=slope,
            ndvi=ndvi,
            ndwi=ndwi,
            water_mask=water_mask
        )

        self.stats = self._compute_statistics(labels, N)

        print(f"Labels generated for {len(labels)} fractions.")

        return labels

    def generate_from_dataset(
        self,
        dataset_path: str,
        output_dir: str = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate labels from a dataset file.

        Args:
            dataset_path: Path to the .pt file.
            output_dir: Directory to save labels (optional).

        Returns:
            Dict with labels per fraction.
        """
        from torch_geometric.data import HeteroData

        print(f"Loading dataset: {dataset_path}")
        data = torch.load(dataset_path, map_location='cpu', weights_only=False)

        if not isinstance(data, HeteroData):
            raise ValueError("Dataset must be HeteroData")

        if 'dem' not in data.node_types:
            raise ValueError("Dataset must have node type 'dem'")

        dem_features = data['dem'].x

        labels = self.generate_from_features(dem_features)

        if output_dir:
            self.save_labels(labels, output_dir)

        return labels

    def _compute_statistics(
        self,
        labels: Dict[str, torch.Tensor],
        total: int
    ) -> Dict:
        """Compute statistics of generated labels."""
        stats = {
            'total_nodes': total,
            'generated_at': datetime.now().isoformat(),
            'fractions': {}
        }

        for fraction, tensor in labels.items():
            counts = {}
            for cls in [1, 2, 3]:
                count = (tensor == cls).sum().item()
                counts[cls] = {
                    'count': count,
                    'percent': count / total * 100
                }

            stats['fractions'][fraction] = {
                'unrestricted': counts[1],
                'restricted': counts[2],
                'severely_restricted': counts[3],
            }

        return stats

    def save_labels(
        self,
        labels: Dict[str, torch.Tensor],
        output_dir: str
    ):
        """
        Save labels as separate .pt files.

        Args:
            labels: Dict with labels per fraction.
            output_dir: Output directory.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for fraction, tensor in labels.items():
            path = output_path / f"labels_{fraction}.pt"
            torch.save(tensor, path)
            print(f"  Saved: {path}")

        stats_path = output_path / "label_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        print(f"  Statistics: {stats_path}")

    def add_labels_to_data(
        self,
        data,
        labels: Dict[str, torch.Tensor]
    ):
        """
        Add labels directly to a Data/HeteroData object.

        Args:
            data: PyG Data or HeteroData object.
            labels: Dict with labels per fraction.
        """
        for fraction, tensor in labels.items():
            attr_name = f'y_{fraction}'
            if hasattr(data, 'node_types') and 'dem' in data.node_types:
                data['dem'][attr_name] = tensor
            else:
                setattr(data, attr_name, tensor)

        print("Labels added to dataset.")

    def print_statistics(self):
        """Print label statistics."""
        if not self.stats:
            print("No statistics available. Run generate_* first.")
            return

        print("\n" + "=" * 60)
        print("GENERATED LABEL STATISTICS")
        print("=" * 60)
        print(f"Total nodes: {self.stats['total_nodes']:,}")
        print(f"Generated at: {self.stats['generated_at']}")

        for fraction, data in self.stats['fractions'].items():
            print(f"\n--- {fraction.upper()} ---")
            print(f"  Unrestricted:       {data['unrestricted']['count']:>10,} ({data['unrestricted']['percent']:>5.1f}%)")
            print(f"  Restricted:         {data['restricted']['count']:>10,} ({data['restricted']['percent']:>5.1f}%)")
            print(f"  Sev. Restricted:    {data['severely_restricted']['count']:>10,} ({data['severely_restricted']['percent']:>5.1f}%)")

        print("=" * 60)

    def get_class_weights(
        self,
        labels: Dict[str, torch.Tensor],
        fraction: str
    ) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for balancing.

        Args:
            labels: Dict with labels.
            fraction: Fraction to compute weights for.

        Returns:
            Tensor [3] with per-class weights.
        """
        tensor = labels[fraction]
        total = tensor.shape[0]

        weights = []
        for cls in [1, 2, 3]:
            count = (tensor == cls).sum().item()
            weight = total / (3 * count) if count > 0 else 1.0
            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float32)


def generate_labels(
    dataset_path: str,
    output_dir: str
) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """
    Convenience function to generate labels from a dataset.

    Args:
        dataset_path: Path to the HeteroData dataset.
        output_dir: Output directory for label files.

    Returns:
        Tuple (labels, stats)
    """
    generator = LabelGenerator()
    labels = generator.generate_from_dataset(dataset_path, output_dir)
    generator.print_statistics()

    return labels, generator.stats
