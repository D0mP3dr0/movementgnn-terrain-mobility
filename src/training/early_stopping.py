"""
EarlyStopping - Halts training when validation loss stops improving.
"""

import torch
import numpy as np


class EarlyStopping:
    """Stops training if validation loss does not improve after 'patience' epochs."""

    def __init__(self, patience=10, min_delta=0, path='checkpoint.pt', verbose=False):
        """
        Args:
            patience: Epochs to wait after last improvement.
            min_delta: Minimum change to qualify as improvement.
            path: Path to save best model checkpoint.
            verbose: Whether to print status messages.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
