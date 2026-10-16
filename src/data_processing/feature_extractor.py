import torch
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MovementFeatures:
    slope: torch.Tensor
    ndvi: torch.Tensor
    ndwi: torch.Tensor
    elevation: torch.Tensor
    tpi: torch.Tensor
    tri: torch.Tensor
    roughness: torch.Tensor
    water_mask: torch.Tensor
    
    def to_tensor(self) -> torch.Tensor:
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
    
    FEATURE_INDICES = {
        'elevation': 0,
        'slope': 1,
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
        self.denormalize = denormalize
    
    def extract_from_dem(self, dem_features: torch.Tensor) -> MovementFeatures:
        if dem_features.shape[1] < 15:
            raise ValueError(f"Esperado >=15 features, encontrado {dem_features.shape[1]}")
        
        slope = dem_features[:, self.FEATURE_INDICES['slope']]
        ndvi = dem_features[:, self.FEATURE_INDICES['ndvi']]
        ndwi = dem_features[:, self.FEATURE_INDICES['ndwi']]
        elevation = dem_features[:, self.FEATURE_INDICES['elevation']]
        tpi = dem_features[:, self.FEATURE_INDICES['tpi']]
        tri = dem_features[:, self.FEATURE_INDICES['tri']]
        roughness = dem_features[:, self.FEATURE_INDICES['roughness']]
        water_mask = dem_features[:, self.FEATURE_INDICES['water_mask']]
        
        if self.denormalize:
            slope = slope * 90.0
        
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
        if dem_features.shape[1] != 17:
            print(f"⚠️ SPEC WARNING: expected 17 features, found {dem_features.shape[1]}")
        
        return dem_features
    
    def compute_movement_indicators(self, features: MovementFeatures) -> Dict[str, torch.Tensor]:
        indicators = {}
        
        indicators['steep_terrain'] = (features.slope > 30.0).float()
        
        indicators['very_steep'] = (features.slope > 45.0).float()
        
        indicators['dense_vegetation'] = (features.ndvi > 0.5).float()
        
        indicators['water_body'] = ((features.ndwi > 0.3) | (features.water_mask > 0.5)).float()
        
        indicators['rugged_terrain'] = (features.tri > 0.5).float()
        
        difficulty = (
            0.4 * (features.slope / 90.0).clamp(0, 1) +
            0.2 * indicators['dense_vegetation'] +
            0.2 * indicators['water_body'] +
            0.2 * (features.tri.clamp(-1, 1) + 1) / 2
        )
        indicators['difficulty_index'] = difficulty
        
        return indicators
    
    def validate_features(self, dem_features: torch.Tensor) -> Dict[str, bool]:
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
        print("\n" + "=" * 50)
        print('FEATURE VALIDATION (specification alignment)')
        print("=" * 50)
        
        validation = self.validate_features(dem_features)
        
        for check, ok in validation.items():
            if check != 'all_ok':
                status = "✓" if ok else "✗"
                print(f"  {status} {check}")
        
        print("-" * 50)
        if validation['all_ok']:
            print('✓ All validations passed!')
        else:
            print('⚠ Some validations failed')
        print("=" * 50)


def extract_features_for_movement(
    dem_features: torch.Tensor,
    denormalize_slope: bool = True
) -> Tuple[MovementFeatures, Dict[str, torch.Tensor]]:
    extractor = FeatureExtractor(denormalize=denormalize_slope)
    
    features = extractor.extract_from_dem(dem_features)
    
    indicators = extractor.compute_movement_indicators(features)
    
    extractor.print_validation_report(dem_features)
    
    return features, indicators


if __name__ == "__main__":
    print("Testando FeatureExtractor...")
    
    N = 1000
    fake_features = torch.rand(N, 17)
    
    fake_features[:, 1] = torch.rand(N) * 0.5
    fake_features[:, 12] = torch.rand(N) * 2 - 1
    fake_features[:, 13] = torch.rand(N) * 2 - 1
    
    features, indicators = extract_features_for_movement(fake_features)
    
    print(f"\nSlope range: [{features.slope.min():.1f}, {features.slope.max():.1f}] graus")
    print(f"Difficulty index range: [{indicators['difficulty_index'].min():.2f}, {indicators['difficulty_index'].max():.2f}]")
