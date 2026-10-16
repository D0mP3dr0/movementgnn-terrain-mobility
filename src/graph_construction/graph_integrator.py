INTERNAL_DATA_ROOT = __import__("os").environ.get("MOVEMENTGNN_DATA_ROOT", ".")
import torch
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime


class GraphIntegrator:
    
    def __init__(self, graph_path: str, labels_dir: str):
        self.graph_path = Path(graph_path)
        self.labels_dir = Path(labels_dir)
        self.data = None
        self.labels = None
    
    def load_graph(self) -> Any:
        print(f"Carregando grafo: {self.graph_path.name}")
        
        from torch_geometric.data import HeteroData
        
        self.data = torch.load(
            str(self.graph_path), 
            map_location='cpu',
            weights_only=False
        )
        
        print(f"  ✓ Tipo: {type(self.data).__name__}")
        
        if isinstance(self.data, HeteroData):
            print(f"  ✓ DEM nodes: {self.data['dem'].x.shape[0]:,}")
        
        return self.data
    
    def load_labels(self) -> Dict[str, torch.Tensor]:
        print(f"\nLoading labels from: {self.labels_dir}")
        
        self.labels = {}
        for fraction in ['a_pe', 'motorizada', 'mecanizada', 'blindada']:
            path = self.labels_dir / f'labels_{fraction}.pt'
            if path.exists():
                self.labels[fraction] = torch.load(path)
                print(f"  ✓ {fraction}: {self.labels[fraction].shape[0]:,} labels")
            else:
                print(f"  ✗ {fraction}: file not found")
        
        return self.labels
    
    def integrate(self) -> Any:
        if self.data is None:
            self.load_graph()
        if self.labels is None:
            self.load_labels()
        
        print('\nIntegrating labels into the graph...')
        
        from torch_geometric.data import HeteroData
        
        if isinstance(self.data, HeteroData):
            num_nodes = self.data['dem'].x.shape[0]
        else:
            num_nodes = self.data.x.shape[0]
        
        for fraction, labels in self.labels.items():
            if labels.shape[0] != num_nodes:
                raise ValueError(
                    f"Mismatch: {fraction} has {labels.shape[0]} labels, but the graph has {num_nodes} nodes"
                )
            
            attr_name = f'y_{fraction}'
            if isinstance(self.data, HeteroData):
                self.data['dem'][attr_name] = labels
            else:
                setattr(self.data, attr_name, labels)
            
            print(f"  ✓ Adicionado: {attr_name}")
        
        if isinstance(self.data, HeteroData):
            self.data['dem'].labels_generated = datetime.now().isoformat()
        
        print(f"\n✓ Integração concluída!")
        
        return self.data
    
    def save(self, output_path: str) -> str:
        print(f"\nSalvando grafo integrado: {output_path}")
        
        torch.save(self.data, output_path)
        
        size_gb = Path(output_path).stat().st_size / (1024**3)
        print(f"  ✓ Tamanho: {size_gb:.2f} GB")
        
        return output_path
    
    def verify_integration(self) -> bool:
        print("\nVerificando integração...")
        
        from torch_geometric.data import HeteroData
        
        all_ok = True
        
        for fraction in ['a_pe', 'motorizada', 'mecanizada', 'blindada']:
            attr_name = f'y_{fraction}'
            
            if isinstance(self.data, HeteroData):
                has_attr = hasattr(self.data['dem'], attr_name)
            else:
                has_attr = hasattr(self.data, attr_name)
            
            if has_attr:
                print(f"  ✓ {attr_name} presente")
            else:
                print(f"  ✗ {attr_name} ausente")
                all_ok = False
        
        return all_ok


def integrate_labels_to_graph(
    graph_path: str = None,
    labels_dir: str = None,
    output_path: str = None
) -> str:
    if graph_path is None:
        graph_path = rf"{INTERNAL_DATA_ROOT}\GNN_RESTRICAO_MOVIMENTO\DATA_GRAPH\interior_sp_v18.4_part_Pirassununga_cupy.pt"
    
    if labels_dir is None:
        labels_dir = rf"{INTERNAL_DATA_ROOT}\GNN_RESTRICAO_MOVIMENTO\data\labels"
    
    if output_path is None:
        output_path = rf"{INTERNAL_DATA_ROOT}\GNN_RESTRICAO_MOVIMENTO\data\graphs\pirassununga_with_labels.pt"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    integrator = GraphIntegrator(graph_path, labels_dir)
    integrator.integrate()
    integrator.verify_integration()
    integrator.save(output_path)
    
    return output_path


if __name__ == "__main__":
    integrate_labels_to_graph()
