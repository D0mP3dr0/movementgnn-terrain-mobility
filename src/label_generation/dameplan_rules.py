import torch
from typing import Dict, Tuple, Optional
from enum import IntEnum
from dataclasses import dataclass


class RestrictionClass(IntEnum):
    IRRESTRITO = 1
    RESTRITO = 2
    SEV_RESTRITO = 3


class FractionType(IntEnum):
    A_PE = 0
    MOTORIZADA = 1
    MECANIZADA = 2
    BLINDADA = 3


DAMEPLAN_LIMITS = {
    'a_pe': {
        'slope_irrestrito': 30,
        'slope_restrito': 45,
        
        'vadeabilidade_max': 1.5,
        
        'ndvi_restrito': 0.7,
        'ndvi_sev_restrito': 0.85,
        
        'ndwi_agua': 0.3,
        'ndwi_umido': 0.1,
    },
    'motorizada': {
        'slope_irrestrito': 15,
        'slope_restrito': 30,
        'vadeabilidade_max': 0.5,
        'ndvi_restrito': 0.5,
        'ndvi_sev_restrito': 0.7,
        'ndwi_agua': 0.3,
        'ndwi_umido': 0.15,
    },
    'mecanizada': {
        'slope_irrestrito': 20,
        'slope_restrito': 35,
        'vadeabilidade_max': 1.2,
        'ndvi_restrito': 0.6,
        'ndvi_sev_restrito': 0.75,
        'ndwi_agua': 0.3,
        'ndwi_umido': 0.12,
    },
    'blindada': {
        'slope_irrestrito': 15,
        'slope_restrito': 30,
        'vadeabilidade_max': 1.5,
        'ndvi_restrito': 0.5,
        'ndvi_sev_restrito': 0.7,
        'ndwi_agua': 0.3,
        'ndwi_umido': 0.15,
    },
}


@dataclass
class ClassificationResult:
    fraction: str
    restriction: RestrictionClass
    reason: str
    slope: float
    ndvi: float
    ndwi: float


class DAMEPLANRules:
    
    def __init__(self, limits: Dict = None):
        self.limits = limits or DAMEPLAN_LIMITS
        self._validate_limits()
    
    def _validate_limits(self):
        required_keys = ['slope_irrestrito', 'slope_restrito', 'ndvi_restrito', 'ndwi_agua']
        for fraction, limits in self.limits.items():
            for key in required_keys:
                if key not in limits:
                    raise ValueError(f"Limit '{key}' missing for fraction '{fraction}'")
    
    def classify_node(
        self,
        slope: float,
        ndvi: float,
        ndwi: float,
        fraction: str,
        water_mask: float = 0.0
    ) -> RestrictionClass:
        if fraction not in self.limits:
            raise ValueError(f"Fração desconhecida: {fraction}")
        
        lim = self.limits[fraction]
        
        if slope > lim['slope_restrito']:
            return RestrictionClass.SEV_RESTRITO
        
        if ndwi > lim['ndwi_agua'] or water_mask > 0.5:
            return RestrictionClass.SEV_RESTRITO
        
        if 'ndvi_sev_restrito' in lim and ndvi > lim['ndvi_sev_restrito']:
            return RestrictionClass.SEV_RESTRITO
        
        if slope > lim['slope_irrestrito']:
            return RestrictionClass.RESTRITO
        
        if ndvi > lim['ndvi_restrito']:
            return RestrictionClass.RESTRITO
        
        if 'ndwi_umido' in lim and ndwi > lim['ndwi_umido']:
            return RestrictionClass.RESTRITO
        
        return RestrictionClass.IRRESTRITO
    
    def classify_batch(
        self,
        slope: torch.Tensor,
        ndvi: torch.Tensor,
        ndwi: torch.Tensor,
        fraction: str,
        water_mask: torch.Tensor = None
    ) -> torch.Tensor:
        if fraction not in self.limits:
            raise ValueError(f"Fração desconhecida: {fraction}")
        
        lim = self.limits[fraction]
        N = slope.shape[0]
        
        labels = torch.ones(N, dtype=torch.long)
        
        if water_mask is None:
            water_mask = torch.zeros(N)
        
        mask_slope_restrito = slope > lim['slope_irrestrito']
        labels[mask_slope_restrito] = RestrictionClass.RESTRITO
        
        mask_veg_restrito = ndvi > lim['ndvi_restrito']
        labels[mask_veg_restrito] = RestrictionClass.RESTRITO
        
        if 'ndwi_umido' in lim:
            mask_umido = ndwi > lim['ndwi_umido']
            labels[mask_umido] = torch.maximum(
                labels[mask_umido], 
                torch.tensor(RestrictionClass.RESTRITO)
            )
        
        mask_slope_sev = slope > lim['slope_restrito']
        labels[mask_slope_sev] = RestrictionClass.SEV_RESTRITO
        
        mask_agua = (ndwi > lim['ndwi_agua']) | (water_mask > 0.5)
        labels[mask_agua] = RestrictionClass.SEV_RESTRITO
        
        if 'ndvi_sev_restrito' in lim:
            mask_veg_sev = ndvi > lim['ndvi_sev_restrito']
            labels[mask_veg_sev] = RestrictionClass.SEV_RESTRITO
        
        return labels
    
    def classify_all_fractions(
        self,
        slope: torch.Tensor,
        ndvi: torch.Tensor,
        ndwi: torch.Tensor,
        water_mask: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        results = {}
        for fraction in ['a_pe', 'motorizada', 'mecanizada', 'blindada']:
            results[fraction] = self.classify_batch(
                slope, ndvi, ndwi, fraction, water_mask
            )
        return results
    
    def get_class_description(self, restriction: RestrictionClass) -> str:
        descriptions = {
            RestrictionClass.IRRESTRITO: 'Unrestricted (Go) - Free movement in combat formations',
            RestrictionClass.RESTRITO: 'Restricted (Slow-Go) - Requires changes of formation/speed',
            RestrictionClass.SEV_RESTRITO: 'Severely Restricted (No-Go) - Prevents movement without engineering',
        }
        return descriptions.get(restriction, "Desconhecido")
    
    def print_limits(self, fraction: str = None):
        fractions = [fraction] if fraction else list(self.limits.keys())
        
        print("\n" + "=" * 60)
        print("LIMITES DAMEPLAN")
        print("=" * 60)
        
        for frac in fractions:
            lim = self.limits[frac]
            print(f"\n--- {frac.upper()} ---")
            print(f"  Slope Irrestrito: <= {lim['slope_irrestrito']}°")
            print(f"  Slope Restrito: {lim['slope_irrestrito']}-{lim['slope_restrito']}°")
            print(f"  Slope Sev.Restrito: > {lim['slope_restrito']}°")
            print(f"  NDVI Restrito: > {lim['ndvi_restrito']}")
            print(f"  NDWI water: > {lim['ndwi_agua']}")
        
        print("=" * 60)


if __name__ == "__main__":
    rules = DAMEPLANRules()
    rules.print_limits()
    
    print('\n--- CLASSIFICATION TESTS ---')
    
    test_cases = [
        (10, 0.3, 0.0, 'Flat terrain, moderate vegetation'),
        (35, 0.2, 0.0, 'Steep terrain, low vegetation'),
        (50, 0.1, 0.0, "Terreno muito íngreme"),
        (15, 0.8, 0.0, "Vegetação densa"),
        (5, 0.1, 0.5, "Corpo d'água"),
    ]
    
    for slope, ndvi, ndwi, desc in test_cases:
        print(f"\n{desc}:")
        print(f"  Slope={slope}°, NDVI={ndvi}, NDWI={ndwi}")
        for frac in ['a_pe', 'motorizada', 'mecanizada', 'blindada']:
            result = rules.classify_node(slope, ndvi, ndwi, frac)
            print(f"  {frac}: {result.name}")
