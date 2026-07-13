"""
DAMEPLANLoss - Physics-Informed Hybrid Loss Function

Combines:
1. Cross-Entropy Loss (supervised learning from labels)
2. Physics Constraints (penalizes DAMEPLAN rule violations):
   - Slope: steep terrain should not be classified as unrestricted
   - Vegetation: dense canopy (high NDVI) should not be unrestricted
   - Water bodies: high NDWI or water mask should not be unrestricted

These three constraints mirror the same factors used in label generation
(dameplan_rules.py), ensuring the model respects doctrinal thresholds
even when cross-entropy alone would not sufficiently penalize violations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class DAMEPLANLoss(nn.Module):
    """
    Loss function incorporating doctrinal knowledge (DAMEPLAN).

    Three physics-informed penalty terms — slope, vegetation, and water —
    act on P(Unrestricted) with per-fraction thresholds aligned to
    EB 60-ME-11.401.
    """

    IDX_SLOPE = 1
    IDX_NDVI = 12
    IDX_NDWI = 13
    IDX_WATER = 14

    IDX_IRRESTRITO = 0  # Unrestricted class (0-based)

    # NDVI thresholds per fraction (source: EB 60-ME-11.401)
    NDVI_THRESHOLDS = {
        'a_pe': 0.7,
        'motorizada': 0.5,
        'mecanizada': 0.6,
        'blindada': 0.5,
    }

    def __init__(
        self,
        weight_ce: float = 1.0,
        weight_phys: float = 0.5,
        class_weights: torch.Tensor = None
    ):
        """
        Args:
            weight_ce: Weight for Cross-Entropy term.
            weight_phys: Weight for physics constraint terms.
            class_weights: Per-class weights for imbalanced classes [3].
        """
        super().__init__()
        self.weight_ce = weight_ce
        self.weight_phys = weight_phys

        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=0.1,
        )

        # Learnable task-balancing weights (one per fraction)
        self.task_log_vars = nn.Parameter(torch.zeros(4))

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss.

        Args:
            outputs: Dict {fraction_name: [N, 3] logits}
            targets: Dict {fraction_name: [N] labels (1-based)}
            features: [N, 17] topographic features

        Returns:
            total_loss, metrics_dict
        """
        total_loss = 0.0
        metrics = {}

        fractions = ['a_pe', 'motorizada', 'mecanizada', 'blindada']

        for i, frac in enumerate(fractions):
            logits = outputs[frac]
            labels = targets[frac] - 1  # 1-based -> 0-based

            valid_mask = (labels >= 0) & (labels < 3)
            if not valid_mask.any():
                continue

            loss_task = 0.0

            # Supervised Loss
            ce = self.ce_loss(logits[valid_mask], labels[valid_mask])
            loss_task += self.weight_ce * ce
            metrics[f'ce_{frac}'] = ce.item()

            # Slope Constraint
            slope_loss = self._slope_constraint(
                logits, features[:, self.IDX_SLOPE]
            )
            loss_task += self.weight_phys * slope_loss
            metrics[f'slope_{frac}'] = slope_loss.item()

            # Vegetation Constraint
            veg_loss = self._vegetation_constraint(
                logits, features[:, self.IDX_NDVI], frac
            )
            loss_task += self.weight_phys * veg_loss
            metrics[f'veg_{frac}'] = veg_loss.item()

            # Water Constraint
            water_loss = self._water_constraint(
                logits,
                features[:, self.IDX_NDWI],
                features[:, self.IDX_WATER]
            )
            loss_task += self.weight_phys * water_loss
            metrics[f'water_{frac}'] = water_loss.item()

            # Uncertainty weighting (learnable per-task weights)
            precision = torch.exp(-self.task_log_vars[i])
            total_loss += precision * loss_task + self.task_log_vars[i]

            metrics[f'loss_{frac}'] = loss_task.item()

        return total_loss, metrics

    # ── Physics constraints ──────────────────────────────────────────

    def _slope_constraint(self, logits, slope_norm):
        """
        Penalizes P(Unrestricted) on steep slopes.
        Generic threshold 0.35 normalized (~31 degrees).
        """
        probs = F.softmax(logits, dim=1)
        prob_irrestrito = probs[:, self.IDX_IRRESTRITO]

        slope_threshold = 0.35
        violation = F.relu(slope_norm - slope_threshold)
        return (prob_irrestrito * violation).mean()

    def _vegetation_constraint(self, logits, ndvi, fraction: str):
        """
        Penalizes P(Unrestricted) in densely vegetated areas.

        Per-fraction thresholds from DAMEPLAN (EB 60-ME-11.401):
            a_pe=0.7, motorizada=0.5, mecanizada=0.6, blindada=0.5.
        """
        probs = F.softmax(logits, dim=1)
        prob_irrestrito = probs[:, self.IDX_IRRESTRITO]

        threshold = self.NDVI_THRESHOLDS.get(fraction, 0.6)
        violation = F.relu(ndvi - threshold)
        return (prob_irrestrito * violation).mean()

    def _water_constraint(self, logits, ndwi, water_mask):
        """
        Penalizes P(Unrestricted) on water bodies (NDWI > 0.3 or water_mask).
        """
        probs = F.softmax(logits, dim=1)
        prob_irrestrito = probs[:, self.IDX_IRRESTRITO]

        is_water = (ndwi > 0.3) | (water_mask > 0.5)
        return (prob_irrestrito * is_water.float()).mean()
