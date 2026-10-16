INTERNAL_DATA_ROOT = __import__("os").environ.get("MOVEMENTGNN_DATA_ROOT", ".")
import torch
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime

from .dameplan_rules import DAMEPLANRules, RestrictionClass


class LabelGenerator:
    
    FEATURE_INDICES = {
        'elevation': 0,
        'slope': 1,
        'ndvi': 12,
        'ndwi': 13,
        'water_mask': 14,
    }
    
    def __init__(self, rules: DAMEPLANRules = None):
        self.rules = rules or DAMEPLANRules()
        self.stats = {}
    
    def generate_from_features(
        self,
        dem_features: torch.Tensor,
        denormalize_slope: bool = True
    ) -> Dict[str, torch.Tensor]:
        N = dem_features.shape[0]
        print(f"Generating labels for {N:,} nodes...")
        
        slope = dem_features[:, self.FEATURE_INDICES['slope']]
        ndvi = dem_features[:, self.FEATURE_INDICES['ndvi']]
        ndwi = dem_features[:, self.FEATURE_INDICES['ndwi']]
        water_mask = dem_features[:, self.FEATURE_INDICES['water_mask']]
        
        if denormalize_slope:
            slope = slope * 90.0
        
        labels = self.rules.classify_all_fractions(
            slope=slope,
            ndvi=ndvi,
            ndwi=ndwi,
            water_mask=water_mask
        )
        
        self.stats = self._compute_statistics(labels, N)
        
        print(f"✓ Labels generated for {len(labels)} fractions")
        
        return labels
    
    def generate_from_dataset(
        self,
        dataset_path: str,
        output_dir: str = None
    ) -> Dict[str, torch.Tensor]:
        from torch_geometric.data import HeteroData
        
        print(f"Carregando dataset: {dataset_path}")
        data = torch.load(dataset_path, map_location='cpu', weights_only=False)
        
        if not isinstance(data, HeteroData):
            raise ValueError("Dataset deve ser HeteroData")
        
        if 'dem' not in data.node_types:
            raise ValueError("Dataset deve ter node type 'dem'")
        
        dem_features = data['dem'].x
        
        labels = self.generate_from_features(dem_features)
        
        if output_dir:
            self.save_labels(labels, output_dir)
        
        return labels
    
    def _compute_statistics(
        self,
        labels: Dict[str, torch.Tensor],
        total: int
    ) -> Dict:
        stats = {
            'total_nodes': total,
            'generated_at': datetime.now().isoformat(),
            'fractions': {}
        }
        
        for fraction, tensor in labels.items():
            counts = {}
            for cls in [1, 2, 3]:
                count = (tensor == cls).sum().item()
                counts[cls] = {
                    'count': count,
                    'percent': count / total * 100
                }
            
            stats['fractions'][fraction] = {
                'irrestrito': counts[1],
                'restrito': counts[2],
                'sev_restrito': counts[3],
            }
        
        return stats
    
    def save_labels(
        self,
        labels: Dict[str, torch.Tensor],
        output_dir: str
    ):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for fraction, tensor in labels.items():
            path = output_path / f"labels_{fraction}.pt"
            torch.save(tensor, path)
            print(f"  Salvo: {path}")
        
        stats_path = output_path / "label_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        print(f"  Estatísticas: {stats_path}")
    
    def add_labels_to_data(
        self,
        data,
        labels: Dict[str, torch.Tensor]
    ):
        for fraction, tensor in labels.items():
            attr_name = f'y_{fraction}'
            if hasattr(data, 'node_types') and 'dem' in data.node_types:
                data['dem'][attr_name] = tensor
            else:
                setattr(data, attr_name, tensor)
        
        print(f"✓ Labels adicionados ao dataset")
    
    def print_statistics(self):
        if not self.stats:
            print('No statistics available. Run generate_* first.')
            return
        
        print("\n" + "=" * 60)
        print('STATISTICS OF THE GENERATED LABELS')
        print("=" * 60)
        print(f"Total number of nodes: {self.stats['total_nodes']:,}")
        print(f"Gerado em: {self.stats['generated_at']}")
        
        for fraction, data in self.stats['fractions'].items():
            print(f"\n--- {fraction.upper()} ---")
            print(f"  Irrestrito:     {data['irrestrito']['count']:>10,} ({data['irrestrito']['percent']:>5.1f}%)")
            print(f"  Restrito:       {data['restrito']['count']:>10,} ({data['restrito']['percent']:>5.1f}%)")
            print(f"  Sev. Restrito:  {data['sev_restrito']['count']:>10,} ({data['sev_restrito']['percent']:>5.1f}%)")
        
        print("=" * 60)
    
    def get_class_weights(
        self,
        labels: Dict[str, torch.Tensor],
        fraction: str
    ) -> torch.Tensor:
        tensor = labels[fraction]
        total = tensor.shape[0]
        
        weights = []
        for cls in [1, 2, 3]:
            count = (tensor == cls).sum().item()
            weight = total / (3 * count) if count > 0 else 1.0
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)


def generate_pirassununga_labels(
    dataset_path: str = None,
    output_dir: str = None
) -> Tuple[Dict[str, torch.Tensor], Dict]:
    if dataset_path is None:
        dataset_path = rf"{INTERNAL_DATA_ROOT}\GNN_RESTRICAO_MOVIMENTO\DATA_GRAPH\interior_sp_v18.4_part_Pirassununga_cupy.pt"
    
    if output_dir is None:
        output_dir = rf"{INTERNAL_DATA_ROOT}\GNN_RESTRICAO_MOVIMENTO\data\labels"
    
    generator = LabelGenerator()
    labels = generator.generate_from_dataset(dataset_path, output_dir)
    generator.print_statistics()
    
    return labels, generator.stats


if __name__ == "__main__":
    generate_pirassununga_labels()
