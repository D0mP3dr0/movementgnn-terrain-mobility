"""
DAMEPLANRules - Movement restriction classification rules.
Based on EB 60-ME-11.401: Average Planning Data (DAMEPLAN).

Restriction Classes:
    1 - UNRESTRICTED (Go): Terrain allows movement in combat formations
        without significant restrictions.
    2 - RESTRICTED (Slow-Go): Terrain hinders movement, requiring formation
        changes or speed reduction.
    3 - SEVERELY RESTRICTED (No-Go): Terrain prevents movement without
        engineering support.

Military Fractions:
    - A PE (Infantry): Dismounted troops
    - MOTORIZADA (Motorized): Wheeled vehicles
    - MECANIZADA (Mechanized): Infantry fighting vehicles (IFV)
    - BLINDADA (Armored): Main battle tanks (MBT)
"""

import torch
from typing import Dict, Tuple, Optional
from enum import IntEnum
from dataclasses import dataclass


class RestrictionClass(IntEnum):
    """Movement restriction classes."""
    IRRESTRITO = 1        # Go
    RESTRITO = 2          # Slow-Go
    SEV_RESTRITO = 3      # No-Go


class FractionType(IntEnum):
    """Military fraction types."""
    A_PE = 0
    MOTORIZADA = 1
    MECANIZADA = 2
    BLINDADA = 3


# DAMEPLAN thresholds per military fraction
# Source: EB 60-ME-11.401 (adapted)
DAMEPLAN_LIMITS = {
    'a_pe': {
        'slope_irrestrito': 30,      # Up to 30 deg = unrestricted
        'slope_restrito': 45,        # 30-45 deg = restricted; >45 deg = severely restricted
        'vadeabilidade_max': 1.5,    # Max fording depth (meters)
        'ndvi_restrito': 0.7,        # NDVI > 0.7 = dense vegetation (restricted)
        'ndvi_sev_restrito': 0.85,   # NDVI > 0.85 = very dense (severely restricted)
        'ndwi_agua': 0.3,            # NDWI > 0.3 = water body (severely restricted)
        'ndwi_umido': 0.1,           # NDWI > 0.1 = wet soil (potentially restricted)
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
    """Classification result for a single node."""
    fraction: str
    restriction: RestrictionClass
    reason: str
    slope: float
    ndvi: float
    ndwi: float


class DAMEPLANRules:
    """
    Implementation of DAMEPLAN rules for movement restriction classification.

    References:
    - EB 60-ME-11.401: Average Planning Data (DAMEPLAN)
    - EB 70-MC-10.204: Movement and Maneuver
    - FM 5-33: Terrain Analysis (US Army)
    """

    def __init__(self, limits: Dict = None):
        """
        Args:
            limits: Custom threshold dict (optional, defaults to DAMEPLAN_LIMITS).
        """
        self.limits = limits or DAMEPLAN_LIMITS
        self._validate_limits()

    def _validate_limits(self):
        """Validate the structure of threshold limits."""
        required_keys = ['slope_irrestrito', 'slope_restrito', 'ndvi_restrito', 'ndwi_agua']
        for fraction, limits in self.limits.items():
            for key in required_keys:
                if key not in limits:
                    raise ValueError(f"Threshold '{key}' missing for fraction '{fraction}'")

    def classify_node(
        self,
        slope: float,
        ndvi: float,
        ndwi: float,
        fraction: str,
        water_mask: float = 0.0
    ) -> RestrictionClass:
        """
        Classify a single node according to DAMEPLAN rules.

        Args:
            slope: Inclination in degrees (0-90).
            ndvi: Vegetation index (-1 to 1).
            ndwi: Water index (-1 to 1).
            fraction: Fraction type ('a_pe', 'motorizada', 'mecanizada', 'blindada').
            water_mask: Binary water mask (0 or 1).

        Returns:
            RestrictionClass (1=Unrestricted, 2=Restricted, 3=Severely Restricted)
        """
        if fraction not in self.limits:
            raise ValueError(f"Unknown fraction: {fraction}")

        lim = self.limits[fraction]

        # Severely Restricted (No-Go)
        if slope > lim['slope_restrito']:
            return RestrictionClass.SEV_RESTRITO

        if ndwi > lim['ndwi_agua'] or water_mask > 0.5:
            return RestrictionClass.SEV_RESTRITO

        if 'ndvi_sev_restrito' in lim and ndvi > lim['ndvi_sev_restrito']:
            return RestrictionClass.SEV_RESTRITO

        # Restricted (Slow-Go)
        if slope > lim['slope_irrestrito']:
            return RestrictionClass.RESTRITO

        if ndvi > lim['ndvi_restrito']:
            return RestrictionClass.RESTRITO

        if 'ndwi_umido' in lim and ndwi > lim['ndwi_umido']:
            return RestrictionClass.RESTRITO

        # Unrestricted (Go)
        return RestrictionClass.IRRESTRITO

    def classify_batch(
        self,
        slope: torch.Tensor,
        ndvi: torch.Tensor,
        ndwi: torch.Tensor,
        fraction: str,
        water_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Classify a batch of nodes (vectorized).

        Args:
            slope: Tensor [N] with inclination in degrees.
            ndvi: Tensor [N] with NDVI values.
            ndwi: Tensor [N] with NDWI values.
            fraction: Fraction type.
            water_mask: Tensor [N] binary water mask (optional).

        Returns:
            Tensor [N] with classes (1, 2, or 3).
        """
        if fraction not in self.limits:
            raise ValueError(f"Unknown fraction: {fraction}")

        lim = self.limits[fraction]
        N = slope.shape[0]

        labels = torch.ones(N, dtype=torch.long)

        if water_mask is None:
            water_mask = torch.zeros(N)

        # Restricted (Slow-Go)
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

        # Severely Restricted (No-Go)
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
        """
        Classify for all fractions at once.

        Args:
            slope: Tensor [N] with inclination in degrees.
            ndvi: Tensor [N] with NDVI values.
            ndwi: Tensor [N] with NDWI values.
            water_mask: Tensor [N] binary water mask (optional).

        Returns:
            Dict with labels per fraction.
        """
        results = {}
        for fraction in ['a_pe', 'motorizada', 'mecanizada', 'blindada']:
            results[fraction] = self.classify_batch(
                slope, ndvi, ndwi, fraction, water_mask
            )
        return results

    def get_class_description(self, restriction: RestrictionClass) -> str:
        """Returns human-readable description for a restriction class."""
        descriptions = {
            RestrictionClass.IRRESTRITO: "Unrestricted (Go) - Free movement in combat formations",
            RestrictionClass.RESTRITO: "Restricted (Slow-Go) - Requires formation/speed changes",
            RestrictionClass.SEV_RESTRITO: "Severely Restricted (No-Go) - Impassable without engineering",
        }
        return descriptions.get(restriction, "Unknown")

    def print_limits(self, fraction: str = None):
        """Print DAMEPLAN thresholds."""
        fractions = [fraction] if fraction else list(self.limits.keys())

        print("\n" + "=" * 60)
        print("DAMEPLAN THRESHOLDS")
        print("=" * 60)

        for frac in fractions:
            lim = self.limits[frac]
            print(f"\n--- {frac.upper()} ---")
            print(f"  Slope Unrestricted: <= {lim['slope_irrestrito']} deg")
            print(f"  Slope Restricted: {lim['slope_irrestrito']}-{lim['slope_restrito']} deg")
            print(f"  Slope Sev.Restricted: > {lim['slope_restrito']} deg")
            print(f"  NDVI Restricted: > {lim['ndvi_restrito']}")
            print(f"  NDWI Water: > {lim['ndwi_agua']}")

        print("=" * 60)
