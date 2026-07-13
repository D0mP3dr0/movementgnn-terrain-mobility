"""
FeatureExtractor - Extracts and processes features for restriction classification.

Critical features for mobility classification:
- Slope (inclination) - primary mobility determinant
- NDVI (vegetation) - restricts motorized movement
- NDWI (water) - identifies water bodies
- TRI/TPI - terrain ruggedness
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MovementFeatures:
    """Extracted features for movement classification."""
    slope: torch.Tensor           # Inclination [0, 90] degrees
    ndvi: torch.Tensor            # Vegetation index [-1, 1]
    ndwi: torch.Tensor            # Water index [-1, 1]
    elevation: torch.Tensor       # Elevation (meters)
    tpi: torch.Tensor             # Topographic Position Index
    tri: torch.Tensor             # Terrain Ruggedness Index
    roughness: torch.Tensor       # Roughness
    water_mask: torch.Tensor      # Binary water mask

    def to_tensor(self) -> torch.Tensor:
        """Convert to single tensor [N, 8]."""
        return torch.stack([
            self.slope,
            self.ndvi,
            self.ndwi,
            self.elevation,
            self.tpi,
            self.tri,
            self.roughness,
            self.water_mask
        ], dim=1)


class FeatureExtractor:
    """
    Feature extractor for movement restriction classification.

    Extracts critical features from the TerrainGNN dataset and prepares
    them for the mobility classifier.
    """

    # Feature indices in the DEM tensor (TerrainGNN v21)
    FEATURE_INDICES = {
        'elevation': 0,
        'slope': 1,          # Normalized [0, 1] in dataset
        'aspect_cos': 2,
        'aspect_sin': 3,
        'curvature': 4,
        'tpi': 5,
        'tri': 6,
        'roughness': 7,
        'b02_norm': 8,
        'b03_norm': 9,
        'b04_norm': 10,
        'b08_norm': 11,
        'ndvi': 12,
        'ndwi': 13,
        'water_mask': 14,
        'lidar_available': 15,
        'lidar_elevation': 16,
    }

    CRITICAL_FEATURES = ['slope', 'ndvi', 'ndwi', 'tpi', 'tri', 'water_mask']

    def __init__(self, denormalize: bool = True):
        """
        Args:
            denormalize: If True, converts slope from [0,1] to degrees.
        """
        self.denormalize = denormalize

    def extract_from_dem(self, dem_features: torch.Tensor) -> MovementFeatures:
        """
        Extract critical features from the DEM tensor.

        Args:
            dem_features: Tensor [N, 17] with DEM features.

        Returns:
            MovementFeatures with extracted features.
        """
        if dem_features.shape[1] < 15:
            raise ValueError(f"Expected >=15 features, found {dem_features.shape[1]}")

        slope = dem_features[:, self.FEATURE_INDICES['slope']]
        ndvi = dem_features[:, self.FEATURE_INDICES['ndvi']]
        ndwi = dem_features[:, self.FEATURE_INDICES['ndwi']]
        elevation = dem_features[:, self.FEATURE_INDICES['elevation']]
        tpi = dem_features[:, self.FEATURE_INDICES['tpi']]
        tri = dem_features[:, self.FEATURE_INDICES['tri']]
        roughness = dem_features[:, self.FEATURE_INDICES['roughness']]
        water_mask = dem_features[:, self.FEATURE_INDICES['water_mask']]

        if self.denormalize:
            slope = slope * 90.0  # [0, 1] -> [0, 90] degrees

        return MovementFeatures(
            slope=slope,
            ndvi=ndvi,
            ndwi=ndwi,
            elevation=elevation,
            tpi=tpi,
            tri=tri,
            roughness=roughness,
            water_mask=water_mask
        )

    def get_full_features(self, dem_features: torch.Tensor) -> torch.Tensor:
        """
        Return all 17 features for the GNN model (passthrough with validation).

        Args:
            dem_features: Tensor [N, 17] with DEM features.

        Returns:
            Tensor [N, 17]
        """
        if dem_features.shape[1] != 17:
            print(f"WARNING: Expected 17 features, found {dem_features.shape[1]}")

        return dem_features

    def compute_movement_indicators(self, features: MovementFeatures) -> Dict[str, torch.Tensor]:
        """
        Compute derived indicators for classification.

        Args:
            features: Extracted MovementFeatures.

        Returns:
            Dict with derived indicator tensors.
        """
        indicators = {}

        # FM 5-33: slope > 30 deg restricts wheeled vehicles
        indicators['steep_terrain'] = (features.slope > 30.0).float()

        # FM 5-33: slope > 45 deg restricts tracked vehicles
        indicators['very_steep'] = (features.slope > 45.0).float()

        indicators['dense_vegetation'] = (features.ndvi > 0.5).float()

        indicators['water_body'] = ((features.ndwi > 0.3) | (features.water_mask > 0.5)).float()

        indicators['rugged_terrain'] = (features.tri > 0.5).float()

        # Composite difficulty index (0-1)
        difficulty = (
            0.4 * (features.slope / 90.0).clamp(0, 1) +
            0.2 * indicators['dense_vegetation'] +
            0.2 * indicators['water_body'] +
            0.2 * (features.tri.clamp(-1, 1) + 1) / 2
        )
        indicators['difficulty_index'] = difficulty

        return indicators

    def validate_features(self, dem_features: torch.Tensor) -> Dict[str, bool]:
        """
        Validate features against expected specifications.

        Args:
            dem_features: Tensor [N, 17].

        Returns:
            Dict with validation results.
        """
        validation = {}

        validation['num_features_ok'] = dem_features.shape[1] == 17

        slope = dem_features[:, self.FEATURE_INDICES['slope']]
        validation['slope_range_ok'] = (slope.min() >= 0) and (slope.max() <= 1)

        ndvi = dem_features[:, self.FEATURE_INDICES['ndvi']]
        validation['ndvi_range_ok'] = (ndvi.min() >= -1) and (ndvi.max() <= 1)

        ndwi = dem_features[:, self.FEATURE_INDICES['ndwi']]
        validation['ndwi_range_ok'] = (ndwi.min() >= -1) and (ndwi.max() <= 1)

        validation['no_nan'] = not torch.isnan(dem_features).any().item()
        validation['no_inf'] = not torch.isinf(dem_features).any().item()

        validation['all_ok'] = all(validation.values())

        return validation

    def print_validation_report(self, dem_features: torch.Tensor):
        """Print feature validation report."""
        print("\n" + "=" * 50)
        print("FEATURE VALIDATION")
        print("=" * 50)

        validation = self.validate_features(dem_features)

        for check, ok in validation.items():
            if check != 'all_ok':
                status = "PASS" if ok else "FAIL"
                print(f"  [{status}] {check}")

        print("-" * 50)
        if validation['all_ok']:
            print("All validations passed.")
        else:
            print("Some validations failed.")
        print("=" * 50)


def extract_features_for_movement(
    dem_features: torch.Tensor,
    denormalize_slope: bool = True
) -> Tuple[MovementFeatures, Dict[str, torch.Tensor]]:
    """
    Convenience function for complete feature extraction.

    Args:
        dem_features: Tensor [N, 17].
        denormalize_slope: If True, converts slope to degrees.

    Returns:
        Tuple (MovementFeatures, indicators)
    """
    extractor = FeatureExtractor(denormalize=denormalize_slope)

    features = extractor.extract_from_dem(dem_features)
    indicators = extractor.compute_movement_indicators(features)
    extractor.print_validation_report(dem_features)

    return features, indicators
