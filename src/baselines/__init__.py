"""
baselines - Baseline Models for Comparison

Implemented baselines:
1. RandomForestBaseline: Classical RF classifier (sklearn)
2. MLPBaseline: Simple neural network without graph structure
3. RuleBasedBaseline: Direct DAMEPLAN rule classification (no learning)

These demonstrate the added value of the GNN approach.
"""

from .random_forest_baseline import RandomForestBaseline
from .mlp_baseline import MLPBaseline
from .rule_based_baseline import RuleBasedBaseline
from .baseline_runner import run_all_baselines
from .export_baseline_to_geotiff import export_baseline_to_geotiff

__all__ = [
    'RandomForestBaseline',
    'MLPBaseline',
    'RuleBasedBaseline',
    'run_all_baselines',
    'export_baseline_to_geotiff',
]
