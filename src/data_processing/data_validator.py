INTERNAL_DATA_ROOT = __import__("os").environ.get("MOVEMENTGNN_DATA_ROOT", ".")
import torch
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime


class DataValidator:
    
    RAG_SPECS = {
        'dem_features': 17,
        'sentinel_features': 8,
        'lidar_features': 3,
        'k_neighbors': 16,
        'edge_dim': 2,
    }
    
    FEATURE_RANGES = {
        'elevation': {'min': -500, 'max': 9000},
        'slope': {'min': 0, 'max': 1},
        'aspect_cos': {'min': -1, 'max': 1},
        'aspect_sin': {'min': -1, 'max': 1},
        'curvature': {'min': -10, 'max': 10},
        'tpi': {'min': -5, 'max': 5},
        'tri': {'min': 0, 'max': 5},
        'roughness': {'min': 0, 'max': 10},
        'b02_norm': {'min': 0, 'max': 1},
        'b03_norm': {'min': 0, 'max': 1},
        'b04_norm': {'min': 0, 'max': 1},
        'b08_norm': {'min': 0, 'max': 1},
        'ndvi': {'min': -1, 'max': 1},
        'ndwi': {'min': -1, 'max': 1},
        'water_mask': {'min': 0, 'max': 1},
        'lidar_available': {'min': 0, 'max': 1},
        'lidar_elevation': {'min': -500, 'max': 9000},
    }
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.validation_results = []
        self.warnings = []
        self.errors = []
    
    def validate_heterodata(self, data: Any) -> bool:
        from torch_geometric.data import HeteroData
        
        if not isinstance(data, HeteroData):
            self.errors.append('Data is not HeteroData (required by topo_rf_v2_system.md)')
            return False
        
        self._add_result('heterodata_type', True, "Dataset é HeteroData")
        
        required_types = ['dem']
        for node_type in required_types:
            if node_type in data.node_types:
                self._add_result(f'node_type_{node_type}', True, f"Node type '{node_type}' presente")
            else:
                self._add_result(f'node_type_{node_type}', False, f"Node type '{node_type}' ausente")
                self.errors.append(f"Node type '{node_type}' not found")
        
        if 'dem' in data.node_types and hasattr(data['dem'], 'x'):
            dem_features = data['dem'].x.shape[1]
            if dem_features == self.RAG_SPECS['dem_features']:
                self._add_result('dem_features', True, f"DEM tem {dem_features} features (esperado)")
            else:
                self._add_result('dem_features', False, f"DEM tem {dem_features} features (esperado {self.RAG_SPECS['dem_features']})")
                self.warnings.append(f"Number of DEM features differs from the expected")
        
        return len(self.errors) == 0
    
    def validate_features(self, features: torch.Tensor, feature_names: List[str] = None) -> bool:
        all_valid = True
        
        nan_count = torch.isnan(features).sum().item()
        if nan_count > 0:
            self._add_result('no_nan', False, f"{nan_count} valores NaN encontrados")
            self.errors.append(f"Dataset contains {nan_count} NaN values")
            all_valid = False
        else:
            self._add_result('no_nan', True, "Sem valores NaN")
        
        inf_count = torch.isinf(features).sum().item()
        if inf_count > 0:
            self._add_result('no_inf', False, f"{inf_count} valores Inf encontrados")
            self.errors.append(f"Dataset contains {inf_count} Inf values")
            all_valid = False
        else:
            self._add_result('no_inf', True, "Sem valores Inf")
        
        if feature_names is None:
            feature_names = list(self.FEATURE_RANGES.keys())[:features.shape[1]]
        
        for i, name in enumerate(feature_names):
            if i >= features.shape[1]:
                break
            
            col = features[:, i]
            if name in self.FEATURE_RANGES:
                expected = self.FEATURE_RANGES[name]
                
                min_val = col.min().item()
                max_val = col.max().item()
                
                if min_val < expected['min'] or max_val > expected['max']:
                    self._add_result(
                        f'range_{name}', 
                        False, 
                        f"{name}: [{min_val:.2f}, {max_val:.2f}] fora do esperado [{expected['min']}, {expected['max']}]"
                    )
                    self.warnings.append(f"Feature {name} fora do range esperado")
                    if self.strict_mode:
                        all_valid = False
                else:
                    self._add_result(f'range_{name}', True, f"{name}: range OK")
        
        return all_valid
    
    def validate_graph_structure(self, edge_index: torch.Tensor, num_nodes: int) -> bool:
        all_valid = True
        
        if edge_index.shape[0] != 2:
            self._add_result('edge_index_format', False, f"edge_index shape[0]={edge_index.shape[0]} (esperado 2)")
            self.errors.append('edge_index must have shape [2, E]')
            return False
        
        self._add_result('edge_index_format', True, "edge_index formato OK")
        
        max_idx = edge_index.max().item()
        min_idx = edge_index.min().item()
        
        if min_idx < 0:
            self._add_result('edge_index_positive', False, f"Negative indices in edge_index")
            self.errors.append('edge_index contains negative indices')
            all_valid = False
        else:
            self._add_result('edge_index_positive', True, "Índices positivos")
        
        if max_idx >= num_nodes:
            self._add_result('edge_index_bounds', False, f"Index {max_idx} >= num_nodes {num_nodes}")
            self.errors.append('edge_index contains out-of-range indices')
            all_valid = False
        else:
            self._add_result('edge_index_bounds', True, 'Indices within the range')
        
        num_edges = edge_index.shape[1]
        avg_degree = num_edges / num_nodes
        expected_k = self.RAG_SPECS['k_neighbors']
        
        if abs(avg_degree - expected_k) > 2:
            self._add_result('k_neighbors', False, f"Mean degree {avg_degree:.1f} (expected ~{expected_k})")
            self.warnings.append(f"Mean degree of the graph differs from the expected")
        else:
            self._add_result('k_neighbors', True, f"Mean degree ~{avg_degree:.1f} OK")
        
        return all_valid
    
    def _add_result(self, check: str, passed: bool, message: str):
        self.validation_results.append({
            'check': check,
            'passed': passed,
            'message': message
        })
    
    def get_report(self) -> Dict[str, Any]:
        passed = sum(1 for r in self.validation_results if r['passed'])
        total = len(self.validation_results)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'passed': passed,
                'total': total,
                'success_rate': passed / total if total > 0 else 0,
                'num_errors': len(self.errors),
                'num_warnings': len(self.warnings),
            },
            'results': self.validation_results,
            'errors': self.errors,
            'warnings': self.warnings,
            'rag_specs': self.RAG_SPECS,
        }
    
    def print_report(self):
        report = self.get_report()
        
        print("\n" + "=" * 60)
        print('VALIDATION REPORT AGAINST THE SPECIFICATION')
        print("=" * 60)
        print(f"Timestamp: {report['timestamp']}")
        print(f"\nResultados: {report['summary']['passed']}/{report['summary']['total']} passaram")
        print(f"Validation success rate: {report['summary']['success_rate']*100:.1f}%")
        
        print("\n--- DETALHES ---")
        for result in self.validation_results:
            status = "✓" if result['passed'] else "✗"
            print(f"  {status} {result['message']}")
        
        if self.errors:
            print("\n--- ERROS ---")
            for error in self.errors:
                print(f"  ✗ {error}")
        
        if self.warnings:
            print("\n--- AVISOS ---")
            for warning in self.warnings:
                print(f"  ⚠ {warning}")
        
        print("\n" + "=" * 60)
        
        if len(self.errors) == 0:
            print('✓ VALIDATION PASSED - Data consistent with the specification')
        else:
            print('✗ VALIDATION FAILED - Check the errors above')
        
        print("=" * 60)
    
    def save_report(self, path: str):
        report = self.get_report()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Relatório salvo: {path}")


def validate_pirassununga_dataset(data_path: str = None) -> bool:
    from torch_geometric.data import HeteroData
    
    if data_path is None:
        data_path = rf"{INTERNAL_DATA_ROOT}\GNN_RESTRICAO_MOVIMENTO\DATA_GRAPH\interior_sp_v18.4_part_Pirassununga_cupy.pt"
    
    print(f"Validando: {data_path}")
    
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    
    validator = DataValidator(strict_mode=False)
    
    validator.validate_heterodata(data)
    
    if 'dem' in data.node_types:
        dem_features = data['dem'].x
        validator.validate_features(dem_features)
        
        for edge_type in data.edge_types:
            if edge_type[0] == 'dem' and edge_type[2] == 'dem':
                edge_index = data[edge_type].edge_index
                validator.validate_graph_structure(edge_index, dem_features.shape[0])
                break
    
    validator.print_report()
    
    return len(validator.errors) == 0


if __name__ == "__main__":
    validate_pirassununga_dataset()
