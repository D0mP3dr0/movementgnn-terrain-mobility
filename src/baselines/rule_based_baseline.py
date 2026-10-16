import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict
import json
from pathlib import Path


class RuleBasedBaseline:
    
    IDX_SLOPE = 1
    IDX_NDVI = 12
    IDX_NDWI = 13
    IDX_WATER = 14
    
    LIMITS = {
        'a_pe': {'slope_irr': 0.33, 'slope_res': 0.5, 'ndvi_res': 0.7},
        'motorizada': {'slope_irr': 0.17, 'slope_res': 0.33, 'ndvi_res': 0.5},
        'mecanizada': {'slope_irr': 0.22, 'slope_res': 0.39, 'ndvi_res': 0.6},
        'blindada': {'slope_irr': 0.17, 'slope_res': 0.33, 'ndvi_res': 0.5},
    }
    
    def __init__(self):
        self.results = {}
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        preds = {}
        
        slope = X[:, self.IDX_SLOPE]
        ndvi = X[:, self.IDX_NDVI]
        ndwi = X[:, self.IDX_NDWI]
        water = X[:, self.IDX_WATER]
        
        for frac, lim in self.LIMITS.items():
            labels = np.ones(len(X), dtype=int)
            
            labels[(slope > lim['slope_irr']) & (slope <= lim['slope_res'])] = 2
            labels[ndvi > lim['ndvi_res']] = 2
            
            labels[slope > lim['slope_res']] = 3
            labels[ndwi > 0.3] = 3
            labels[water > 0.5] = 3
            
            preds[frac] = labels
        
        return preds
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: Dict[str, np.ndarray]
    ) -> Dict[str, Dict]:
        print("\nAvaliando Rule-Based...")
        
        preds = self.predict(X_test)
        
        for frac in y_test.keys():
            y_true = y_test[frac]
            y_pred = preds[frac]
            
            acc = accuracy_score(y_true, y_pred)
            f1_macro = f1_score(y_true, y_pred, average='macro')
            f1_weighted = f1_score(y_true, y_pred, average='weighted')
            
            self.results[frac] = {
                'accuracy': acc,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
            }
            
            print(f"  {frac}: Acc={acc:.4f}, F1-macro={f1_macro:.4f}")
        
        return self.results
    
    def save_results(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Rule-Based results saved: {path}")


def train_rule_baseline(
    data,
    output_dir: str = None,
    full_predictions_path: str = None,
    return_full_predictions: bool = False
):
    
    X = data.x.numpy()
    test_mask = data.test_mask.numpy()
    
    y = {
        'a_pe': data.y_a_pe.numpy(),
        'motorizada': data.y_motorizada.numpy(),
        'mecanizada': data.y_mecanizada.numpy(),
        'blindada': data.y_blindada.numpy(),
    }
    
    X_test = X[test_mask]
    y_test = {k: v[test_mask] for k, v in y.items()}
    
    rule = RuleBasedBaseline()
    results = rule.evaluate(X_test, y_test)

    full_preds = rule.predict(X)
    
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        rule.save_results(f"{output_dir}/rule_results.json")

    if full_predictions_path:
        out_path = Path(full_predictions_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            a_pe=full_preds['a_pe'].astype(np.uint8),
            motorizada=full_preds['motorizada'].astype(np.uint8),
            mecanizada=full_preds['mecanizada'].astype(np.uint8),
            blindada=full_preds['blindada'].astype(np.uint8),
        )
        print(f"Rule-Based predictions (full) saved to: {out_path}")

    if return_full_predictions:
        return rule, results, full_preds
    return rule, results
