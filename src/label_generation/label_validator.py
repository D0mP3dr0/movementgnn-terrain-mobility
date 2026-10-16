import torch
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class LabelValidator:
    
    MIN_CLASS_PERCENT = 1.0
    MAX_CLASS_PERCENT = 95.0
    
    def __init__(self):
        self.validation_results = []
        self.warnings = []
        self.errors = []
    
    def validate_distribution(
        self,
        labels: Dict[str, torch.Tensor]
    ) -> bool:
        all_valid = True
        
        for fraction, tensor in labels.items():
            total = tensor.shape[0]
            
            for cls in [1, 2, 3]:
                count = (tensor == cls).sum().item()
                percent = count / total * 100
                
                cls_name = ['', 'Irrestrito', 'Restrito', 'Sev.Restrito'][cls]
                
                if percent < self.MIN_CLASS_PERCENT:
                    msg = f"{fraction}/{cls_name}: {percent:.2f}% (< {self.MIN_CLASS_PERCENT}%)"
                    self._add_result(f'dist_{fraction}_{cls}', False, msg)
                    self.warnings.append(msg)
                elif percent > self.MAX_CLASS_PERCENT:
                    msg = f"{fraction}/{cls_name}: {percent:.2f}% (> {self.MAX_CLASS_PERCENT}%)"
                    self._add_result(f'dist_{fraction}_{cls}', False, msg)
                    self.warnings.append(msg)
                else:
                    msg = f"{fraction}/{cls_name}: {percent:.2f}% OK"
                    self._add_result(f'dist_{fraction}_{cls}', True, msg)
        
        return all_valid
    
    def validate_consistency(
        self,
        labels: Dict[str, torch.Tensor]
    ) -> bool:
        all_valid = True
        
        if len(labels) < 2:
            self._add_result('fraction_count', False, 'Fewer than 2 fractions')
            return False
        
        a_pe = labels.get('a_pe')
        motorizada = labels.get('motorizada')
        blindada = labels.get('blindada')
        
        if a_pe is not None and motorizada is not None:
            inconsistent = ((a_pe > motorizada)).sum().item()
            total = a_pe.shape[0]
            percent = inconsistent / total * 100
            
            if percent > 10:
                msg = f"Dismounted more restricted than motorized: {percent:.1f}%"
                self._add_result('consistency_ape_mot', False, msg)
                self.warnings.append(msg)
            else:
                self._add_result('consistency_ape_mot', True, f"Hierarquia OK ({percent:.1f}% inconsistente)")
        
        return all_valid
    
    def validate_coverage(
        self,
        labels: Dict[str, torch.Tensor]
    ) -> bool:
        all_valid = True
        
        for fraction, tensor in labels.items():
            classes_present = torch.unique(tensor).tolist()
            
            for cls in [1, 2, 3]:
                if cls not in classes_present:
                    msg = f"{fraction}: Classe {cls} ausente"
                    self._add_result(f'coverage_{fraction}_{cls}', False, msg)
                    self.warnings.append(msg)
                    all_valid = False
                else:
                    self._add_result(f'coverage_{fraction}_{cls}', True, f"{fraction}: Classe {cls} presente")
        
        return all_valid
    
    def validate_spatial_coherence(
        self,
        labels: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        fraction: str = 'a_pe'
    ) -> float:
        tensor = labels[fraction]
        src, dst = edge_index[0], edge_index[1]
        
        same_class = (tensor[src] == tensor[dst]).float().mean().item()
        
        self._add_result(
            f'spatial_coherence_{fraction}',
            same_class > 0.5,
            f"Spatial coherence {fraction}: {same_class*100:.1f}%"
        )
        
        return same_class
    
    def validate_values(
        self,
        labels: Dict[str, torch.Tensor]
    ) -> bool:
        all_valid = True
        
        for fraction, tensor in labels.items():
            min_val = tensor.min().item()
            max_val = tensor.max().item()
            
            if min_val < 1 or max_val > 3:
                msg = f"{fraction}: Valores fora do range [1,3]: [{min_val}, {max_val}]"
                self._add_result(f'values_{fraction}', False, msg)
                self.errors.append(msg)
                all_valid = False
            else:
                self._add_result(f'values_{fraction}', True, f"{fraction}: Valores OK [1,3]")
        
        return all_valid
    
    def validate_all(
        self,
        labels: Dict[str, torch.Tensor],
        edge_index: torch.Tensor = None
    ) -> bool:
        print("\nValidando labels...")
        
        self.validate_values(labels)
        self.validate_distribution(labels)
        self.validate_coverage(labels)
        self.validate_consistency(labels)
        
        if edge_index is not None:
            for frac in labels.keys():
                self.validate_spatial_coherence(labels, edge_index, frac)
        
        return len(self.errors) == 0
    
    def _add_result(self, check: str, passed: bool, message: str):
        self.validation_results.append({
            'check': check,
            'passed': passed,
            'message': message
        })
    
    def get_report(self) -> Dict:
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
        }
    
    def print_report(self):
        report = self.get_report()
        
        print("\n" + "=" * 60)
        print('VALIDATION REPORT OF THE LABELS')
        print("=" * 60)
        print(f"Resultados: {report['summary']['passed']}/{report['summary']['total']} passaram")
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
        
        print("=" * 60)
        
        if len(self.errors) == 0:
            print("✓ VALIDAÇÃO PASSOU")
        else:
            print("✗ VALIDAÇÃO FALHOU")
        
        print("=" * 60)
    
    def save_report(self, path: str):
        report = self.get_report()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Relatório salvo: {path}")


if __name__ == "__main__":
    print("Testando LabelValidator...")
    
    N = 10000
    labels = {
        'a_pe': torch.randint(1, 4, (N,)),
        'motorizada': torch.randint(1, 4, (N,)),
        'mecanizada': torch.randint(1, 4, (N,)),
        'blindada': torch.randint(1, 4, (N,)),
    }
    
    validator = LabelValidator()
    validator.validate_all(labels)
    validator.print_report()
