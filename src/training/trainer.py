"""
Trainer - Training Loop Manager
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
from pathlib import Path
from datetime import datetime
import json
import numpy as np
from tqdm import tqdm

from ..models.dameplan_loss import DAMEPLANLoss
from ..models.movement_gnn import MovementGNN
from .early_stopping import EarlyStopping


class Trainer:
    """
    Manages the full training lifecycle:
    - Training loop
    - Validation loop
    - Checkpointing
    - Metrics logging
    """

    def __init__(
        self,
        model: MovementGNN,
        optimizer: torch.optim.Optimizer,
        criterion: DAMEPLANLoss,
        device: str = 'cuda',
        output_dir: str = 'output/models',
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': []
        }

    def fit(
        self,
        data,
        epochs: int = 100,
        patience: int = 15,
        batch_size: int = None
    ):
        """
        Execute training loop.

        Args:
            data: PyG Data object with features, edges, labels, and masks.
            epochs: Maximum number of training epochs.
            patience: Early stopping patience (epochs without improvement).
            batch_size: If None, uses full-batch training.

        Returns:
            Training history dict.
        """
        data = data.to(self.device)
        self.model = self.model.to(self.device)

        early_stopping = EarlyStopping(
            patience=patience,
            path=str(self.output_dir / 'best_model.pt'),
            verbose=True
        )

        print(f"Starting training ({epochs} epochs)...")
        print(f"Device: {self.device}")

        for epoch in range(1, epochs + 1):
            train_loss, train_metrics = self._train_epoch(data)
            val_loss, val_metrics = self._validate_epoch(data)
            self._log_epoch(epoch, train_loss, val_loss, train_metrics, val_metrics)

            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        self.model.load_state_dict(torch.load(early_stopping.path, weights_only=True))
        print("Training complete. Best model loaded.")

        return self.history

    def _get_dem_data(self, data):
        """Helper to extract DEM data from Data or HeteroData."""
        if hasattr(data, 'dem'):
            return data.dem
        if isinstance(data, dict):
            if 'dem' in data:
                return data['dem']
        return data

    def _train_epoch(self, data) -> Tuple[float, Dict]:
        self.model.train()
        self.optimizer.zero_grad()

        if hasattr(data, 'y_a_pe'):
            x = data.x
            edge_index = data.edge_index
            targets = {
                'a_pe': data.y_a_pe,
                'motorizada': data.y_motorizada,
                'mecanizada': data.y_mecanizada,
                'blindada': data.y_blindada,
            }
            train_mask = data.train_mask
        else:
            raise ValueError("Trainer requires homogeneous Data object with label attributes.")

        outputs = self.model(x, edge_index)

        masked_outputs = {k: v[train_mask] for k, v in outputs.items()}
        masked_targets = {k: v[train_mask] for k, v in targets.items()}
        masked_features = x[train_mask]

        loss, metrics = self.criterion(masked_outputs, masked_targets, masked_features)

        loss.backward()
        self.optimizer.step()

        return loss.item(), metrics

    def _validate_epoch(self, data) -> Tuple[float, Dict]:
        self.model.eval()
        with torch.no_grad():
            if hasattr(data, 'y_a_pe'):
                x = data.x
                edge_index = data.edge_index
                targets = {
                    'a_pe': data.y_a_pe,
                    'motorizada': data.y_motorizada,
                    'mecanizada': data.y_mecanizada,
                    'blindada': data.y_blindada,
                }
                val_mask = data.val_mask
            else:
                raise ValueError("Trainer requires homogeneous Data object.")

            outputs = self.model(x, edge_index)

            masked_outputs = {k: v[val_mask] for k, v in outputs.items()}
            masked_targets = {k: v[val_mask] for k, v in targets.items()}
            masked_features = x[val_mask]

            loss, metrics = self.criterion(masked_outputs, masked_targets, masked_features)

            acc_metrics = self._compute_accuracy(masked_outputs, masked_targets)
            metrics.update(acc_metrics)

            return loss.item(), metrics

    def _compute_accuracy(self, outputs, targets) -> Dict:
        accs = {}
        for frac, logits in outputs.items():
            preds = torch.argmax(logits, dim=1) + 1  # 1-based
            correct = (preds == targets[frac]).sum().item()
            total = targets[frac].size(0)
            accs[f'acc_{frac}'] = correct / total if total > 0 else 0
        return accs

    def _log_epoch(self, epoch, train_loss, val_loss, train_metrics, val_metrics):
        log = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()}
        }

        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['metrics'].append(log)

        print(f"Epoch {epoch:03d}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        acc_str = " | ".join([f"{frac}: {val_metrics[f'acc_{frac}']:.1%}" for frac in ['a_pe', 'motorizada']])
        print(f"  Val Acc: {acc_str} ...")

    def save_history(self, path: str = 'training_history.json'):
        """Save training history to JSON."""
        full_path = self.output_dir / path
        with open(full_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"History saved to: {full_path}")
