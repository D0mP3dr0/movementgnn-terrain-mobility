"""
training - Training and Evaluation Module

Includes:
1. Trainer: Manages training and validation loops
2. EarlyStopping: Halts training when no improvement is observed
"""

from .trainer import Trainer
from .early_stopping import EarlyStopping

__all__ = ['Trainer', 'EarlyStopping']
