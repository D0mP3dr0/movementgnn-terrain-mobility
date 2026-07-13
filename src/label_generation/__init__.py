"""
label_generation - DAMEPLAN Label Generation Module

Based on:
- EB 60-ME-11.401: Average Planning Data (DAMEPLAN)
- EB 70-MC-10.204: Movement and Maneuver
"""

from .dameplan_rules import DAMEPLANRules, DAMEPLAN_LIMITS
from .label_generator import LabelGenerator
from .label_validator import LabelValidator

__all__ = ['DAMEPLANRules', 'DAMEPLAN_LIMITS', 'LabelGenerator', 'LabelValidator']
