import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm


class MovementGNN(nn.Module):
    
    def __init__(
        self,
        in_channels: int = 17,
        hidden_channels: int = 64,
        num_layers: int = 2,
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
        
        self.classifiers = nn.ModuleDict({
            'a_pe': self._make_classifier(hidden_channels, num_classes),
            'motorizada': self._make_classifier(hidden_channels, num_classes),
            'mecanizada': self._make_classifier(hidden_channels, num_classes),
            'blindada': self._make_classifier(hidden_channels, num_classes),
        })
        
        self.dropout = dropout
    
    def _make_classifier(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout),
            nn.Linear(in_dim // 2, out_dim)
        )
    
    def forward(self, x, edge_index, edge_attr=None):
        h = self.input_encoder(x)
        
        for i, (conv, norm, skip) in enumerate(zip(self.convs, self.norms, self.skips)):
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
        self.eval()
        with torch.no_grad():
            logits_dict = self.forward(x, edge_index, edge_attr)
            preds = {}
            for frac, logits in logits_dict.items():
                probs = F.softmax(logits, dim=1)
                preds[frac] = torch.argmax(probs, dim=1) + 1
            return preds


if __name__ == "__main__":
    print("Testando MovementGNN...")
    
    N = 100
    in_channels = 17
    
    x = torch.randn(N, in_channels)
    edge_index = torch.randint(0, N, (2, N*4))
    
    model = MovementGNN()
    print(f"Arquitetura criada. Params: {sum(p.numel() for p in model.parameters()):,}")
    
    outputs = model(x, edge_index)
    
    print("\nOutputs shapes:")
    for frac, out in outputs.items():
        print(f"  {frac}: {out.shape}")
