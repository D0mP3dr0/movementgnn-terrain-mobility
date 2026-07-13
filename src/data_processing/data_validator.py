"""
DataValidator - Validates dataset structure and feature quality.

Verifies:
- HeteroData structure with required node types
- Feature count and value ranges (17D DEM features)
- Data quality (NaN, Inf detection)
- Graph structure integrity
"""

import torch
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime


class DataValidator:
    """
    Dataset validator ensuring alignment with the expected schema.

    Checks:
    1. Dataset structure (HeteroData)
    2. Feature count and ranges
    3. Data quality (NaN, Inf)
    4. Graph connectivity
    """

    SPECS = {
        'dem_features': 17,
        'sentinel_features': 8,
        'lidar_features': 3,
        'k_neighbors': 16,
        'edge_dim': 2,
    }

    FEATURE_RANGES = {
        'elevation': {'min': -500, 'max': 9000},
        'slope': {'min': 0, 'max': 1},
        'aspect_cos': {'min': -1, 'max': 1},
        'aspect_sin': {'min': -1, 'max': 1},
        'curvature': {'min': -10, 'max': 10},
        'tpi': {'min': -5, 'max': 5},
        'tri': {'min': 0, 'max': 5},
        'roughness': {'min': 0, 'max': 10},
        'b02_norm': {'min': 0, 'max': 1},
        'b03_norm': {'min': 0, 'max': 1},
        'b04_norm': {'min': 0, 'max': 1},
        'b08_norm': {'min': 0, 'max': 1},
        'ndvi': {'min': -1, 'max': 1},
        'ndwi': {'min': -1, 'max': 1},
        'water_mask': {'min': 0, 'max': 1},
        'lidar_available': {'min': 0, 'max': 1},
        'lidar_elevation': {'min': -500, 'max': 9000},
    }

    def __init__(self, strict_mode: bool = False):
        """
        Args:
            strict_mode: If True, treats warnings as failures.
        """
        self.strict_mode = strict_mode
        self.validation_results = []
        self.warnings = []
        self.errors = []

    def validate_heterodata(self, data: Any) -> bool:
        """
        Validate HeteroData structure.

        Args:
            data: Object to validate.

        Returns:
            True if valid.
        """
        from torch_geometric.data import HeteroData

        if not isinstance(data, HeteroData):
            self.errors.append("Data is not HeteroData")
            return False

        self._add_result('heterodata_type', True, "Dataset is HeteroData")

        required_types = ['dem']
        for node_type in required_types:
            if node_type in data.node_types:
                self._add_result(f'node_type_{node_type}', True, f"Node type '{node_type}' present")
            else:
                self._add_result(f'node_type_{node_type}', False, f"Node type '{node_type}' missing")
                self.errors.append(f"Node type '{node_type}' not found")

        if 'dem' in data.node_types and hasattr(data['dem'], 'x'):
            dem_features = data['dem'].x.shape[1]
            if dem_features == self.SPECS['dem_features']:
                self._add_result('dem_features', True, f"DEM has {dem_features} features (expected)")
            else:
                self._add_result('dem_features', False, f"DEM has {dem_features} features (expected {self.SPECS['dem_features']})")
                self.warnings.append(f"DEM feature count differs from expected")

        return len(self.errors) == 0

    def validate_features(self, features: torch.Tensor, feature_names: List[str] = None) -> bool:
        """
        Validate features against expected ranges.

        Args:
            features: Tensor [N, F].
            feature_names: List of feature names.

        Returns:
            True if all validations passed.
        """
        all_valid = True

        nan_count = torch.isnan(features).sum().item()
        if nan_count > 0:
            self._add_result('no_nan', False, f"{nan_count} NaN values found")
            self.errors.append(f"Dataset contains {nan_count} NaN values")
            all_valid = False
        else:
            self._add_result('no_nan', True, "No NaN values")

        inf_count = torch.isinf(features).sum().item()
        if inf_count > 0:
            self._add_result('no_inf', False, f"{inf_count} Inf values found")
            self.errors.append(f"Dataset contains {inf_count} Inf values")
            all_valid = False
        else:
            self._add_result('no_inf', True, "No Inf values")

        if feature_names is None:
            feature_names = list(self.FEATURE_RANGES.keys())[:features.shape[1]]

        for i, name in enumerate(feature_names):
            if i >= features.shape[1]:
                break

            col = features[:, i]
            if name in self.FEATURE_RANGES:
                expected = self.FEATURE_RANGES[name]

                min_val = col.min().item()
                max_val = col.max().item()

                if min_val < expected['min'] or max_val > expected['max']:
                    self._add_result(
                        f'range_{name}',
                        False,
                        f"{name}: [{min_val:.2f}, {max_val:.2f}] outside expected [{expected['min']}, {expected['max']}]"
                    )
                    self.warnings.append(f"Feature {name} outside expected range")
                    if self.strict_mode:
                        all_valid = False
                else:
                    self._add_result(f'range_{name}', True, f"{name}: range OK")

        return all_valid

    def validate_graph_structure(self, edge_index: torch.Tensor, num_nodes: int) -> bool:
        """
        Validate graph structure.

        Args:
            edge_index: Tensor [2, E].
            num_nodes: Number of nodes.

        Returns:
            True if valid.
        """
        all_valid = True

        if edge_index.shape[0] != 2:
            self._add_result('edge_index_format', False, f"edge_index shape[0]={edge_index.shape[0]} (expected 2)")
            self.errors.append("edge_index must have shape [2, E]")
            return False

        self._add_result('edge_index_format', True, "edge_index format OK")

        max_idx = edge_index.max().item()
        min_idx = edge_index.min().item()

        if min_idx < 0:
            self._add_result('edge_index_positive', False, "Negative indices in edge_index")
            self.errors.append("edge_index contains negative indices")
            all_valid = False
        else:
            self._add_result('edge_index_positive', True, "Indices non-negative")

        if max_idx >= num_nodes:
            self._add_result('edge_index_bounds', False, f"Index {max_idx} >= num_nodes {num_nodes}")
            self.errors.append("edge_index contains out-of-range indices")
            all_valid = False
        else:
            self._add_result('edge_index_bounds', True, "Indices within range")

        num_edges = edge_index.shape[1]
        avg_degree = num_edges / num_nodes
        expected_k = self.SPECS['k_neighbors']

        if abs(avg_degree - expected_k) > 2:
            self._add_result('k_neighbors', False, f"Average degree {avg_degree:.1f} (expected ~{expected_k})")
            self.warnings.append(f"Average graph degree differs from expected")
        else:
            self._add_result('k_neighbors', True, f"Average degree ~{avg_degree:.1f} OK")

        return all_valid

    def _add_result(self, check: str, passed: bool, message: str):
        """Add a validation result."""
        self.validation_results.append({
            'check': check,
            'passed': passed,
            'message': message
        })

    def get_report(self) -> Dict[str, Any]:
        """
        Generate validation report.

        Returns:
            Dict with full report.
        """
        passed = sum(1 for r in self.validation_results if r['passed'])
        total = len(self.validation_results)

        return {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'passed': passed,
                'total': total,
                'success_rate': passed / total if total > 0 else 0,
                'num_errors': len(self.errors),
                'num_warnings': len(self.warnings),
            },
            'results': self.validation_results,
            'errors': self.errors,
            'warnings': self.warnings,
            'specs': self.SPECS,
        }

    def print_report(self):
        """Print formatted validation report."""
        report = self.get_report()

        print("\n" + "=" * 60)
        print("DATA VALIDATION REPORT")
        print("=" * 60)
        print(f"Timestamp: {report['timestamp']}")
        print(f"\nResults: {report['summary']['passed']}/{report['summary']['total']} passed")
        print(f"Success rate: {report['summary']['success_rate']*100:.1f}%")

        print("\n--- DETAILS ---")
        for result in self.validation_results:
            status = "PASS" if result['passed'] else "FAIL"
            print(f"  [{status}] {result['message']}")

        if self.errors:
            print("\n--- ERRORS ---")
            for error in self.errors:
                print(f"  {error}")

        if self.warnings:
            print("\n--- WARNINGS ---")
            for warning in self.warnings:
                print(f"  {warning}")

        print("\n" + "=" * 60)

        if len(self.errors) == 0:
            print("VALIDATION PASSED")
        else:
            print("VALIDATION FAILED")

        print("=" * 60)

    def save_report(self, path: str):
        """Save report to JSON."""
        report = self.get_report()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report saved: {path}")


def validate_dataset(data_path: str) -> bool:
    """
    Convenience function to validate a full dataset.

    Args:
        data_path: Path to the dataset .pt file.

    Returns:
        True if all validations passed.
    """
    from torch_geometric.data import HeteroData

    print(f"Validating: {data_path}")

    data = torch.load(data_path, map_location='cpu', weights_only=False)

    validator = DataValidator(strict_mode=False)

    validator.validate_heterodata(data)

    if 'dem' in data.node_types:
        dem_features = data['dem'].x
        validator.validate_features(dem_features)

        for edge_type in data.edge_types:
            if edge_type[0] == 'dem' and edge_type[2] == 'dem':
                edge_index = data[edge_type].edge_index
                validator.validate_graph_structure(edge_index, dem_features.shape[0])
                break

    validator.print_report()

    return len(validator.errors) == 0
