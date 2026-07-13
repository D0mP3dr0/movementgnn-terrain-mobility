"""
MovementGNN - Graph Neural Network for Military Mobility Classification

Architecture:
- Encoder: Linear + BatchNorm + LeakyReLU
- Backbone: 2-layer GATv2 (Graph Attention Networks v2) with residual connections
- Decoder: 4 independent classification heads (Multi-Task Learning)
           (Infantry, Motorized, Mechanized, Armored)

GATv2 enables dynamic attention where each edge receives a distinct
importance weight, resolving the static-attention limitation of GAT
(Brody et al., 2022). This is essential for complex terrain where
neighbor influence varies with topography.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm


class MovementGNN(nn.Module):
    """
    GNN for military movement restriction classification.
    Multi-Task: classifies 4 military fractions simultaneously.
    """

    def __init__(
        self,
        in_channels: int = 271,
        hidden_channels: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        num_classes: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.dropout = dropout

        head_dim = hidden_channels // heads

        for _ in range(num_layers):
            conv = GATv2Conv(
                hidden_channels,
                head_dim,
                heads=heads,
                concat=True,
                edge_dim=None,
                dropout=dropout,
                add_self_loops=True
            )
            self.convs.append(conv)
            self.norms.append(LayerNorm(hidden_channels))
            self.skips.append(nn.Identity())

        # Multi-task classifiers share the backbone but produce
        # independent predictions per military fraction
        self.classifiers = nn.ModuleDict({
            'a_pe': self._make_classifier(hidden_channels, num_classes),
            'motorizada': self._make_classifier(hidden_channels, num_classes),
            'mecanizada': self._make_classifier(hidden_channels, num_classes),
            'blindada': self._make_classifier(hidden_channels, num_classes),
        })

        self.dropout = dropout

    def _make_classifier(self, in_dim, out_dim):
        """Creates a classification head (2-layer MLP)."""
        return nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout),
            nn.Linear(in_dim // 2, out_dim)
        )

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass.

        Args:
            x: [N, 17] Node features
            edge_index: [2, E] Graph connectivity
            edge_attr: [E, D] Edge features (optional)

        Returns:
            Dict with logits per fraction {fraction_name: [N, 3]}
        """
        h = self.input_encoder(x)

        for conv, norm, skip in zip(self.convs, self.norms, self.skips):
            h_in = h
            h = conv(h, edge_index, edge_attr=edge_attr)
            h = F.leaky_relu(h, 0.2)
            h = h + skip(h_in)
            h = norm(h)

        outputs = {}
        for fraction, classifier in self.classifiers.items():
            outputs[fraction] = classifier(h)

        return outputs

    def predict(self, x, edge_index, edge_attr=None):
        """
        Returns predicted class indices (1-based).

        Args:
            x: [N, 17] Node features
            edge_index: [2, E] Graph connectivity
            edge_attr: [E, D] Edge features (optional)

        Returns:
            Dict {fraction_name: [N] predicted classes (1, 2, or 3)}
        """
        self.eval()
        with torch.no_grad():
            logits_dict = self.forward(x, edge_index, edge_attr)
            preds = {}
            for frac, logits in logits_dict.items():
                probs = F.softmax(logits, dim=1)
                preds[frac] = torch.argmax(probs, dim=1) + 1
            return preds
