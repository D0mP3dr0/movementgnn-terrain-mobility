"""
RuleBasedBaseline - Direct classification using DAMEPLAN rules.

This baseline applies doctrinal thresholds without any learning,
serving as the minimum performance the GNN must exceed.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict
import json
from pathlib import Path


class RuleBasedBaseline:
    """Applies DAMEPLAN rules directly (no ML)."""

    IDX_SLOPE = 1
    IDX_NDVI = 12
    IDX_NDWI = 13
    IDX_WATER = 14

    # Simplified thresholds per fraction (normalized slope)
    LIMITS = {
        'a_pe': {'slope_irr': 0.33, 'slope_res': 0.5, 'ndvi_res': 0.7},
        'motorizada': {'slope_irr': 0.17, 'slope_res': 0.33, 'ndvi_res': 0.5},
        'mecanizada': {'slope_irr': 0.22, 'slope_res': 0.39, 'ndvi_res': 0.6},
        'blindada': {'slope_irr': 0.17, 'slope_res': 0.33, 'ndvi_res': 0.5},
    }

    def __init__(self):
        self.results = {}

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply rules to classify all nodes.

        Args:
            X: [N, F] feature matrix.

        Returns:
            Dict {fraction_name: [N] predicted labels (1-3)}
        """
        preds = {}

        slope = X[:, self.IDX_SLOPE]
        ndvi = X[:, self.IDX_NDVI]
        ndwi = X[:, self.IDX_NDWI]
        water = X[:, self.IDX_WATER]

        for frac, lim in self.LIMITS.items():
            labels = np.ones(len(X), dtype=int)  # Default: Unrestricted (1)

            # Restricted
            labels[(slope > lim['slope_irr']) & (slope <= lim['slope_res'])] = 2
            labels[ndvi > lim['ndvi_res']] = 2

            # Severely Restricted
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
        """
        Evaluate performance on test set.

        Args:
            X_test: [N, F] test features.
            y_test: Dict {fraction_name: [N] true labels}.

        Returns:
            Dict with metrics per fraction.
        """
        print("\nEvaluating Rule-Based...")

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
        """Save results to JSON."""
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Rule-Based results saved: {path}")


def train_rule_baseline(
    data,
    output_dir: str = None,
    full_predictions_path: str = None,
    return_full_predictions: bool = False
):
    """
    Convenience function to run the rule-based baseline.

    Args:
        data: PyG Data object with features and label attributes.
        output_dir: Directory for saving metric results.
        full_predictions_path: Path for saving full-node predictions (.npz).
        return_full_predictions: If True, also returns predictions for all nodes.

    Returns:
        Tuple (RuleBasedBaseline, results) or (RuleBasedBaseline, results, full_preds)
    """
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
        print(f"Rule-Based full predictions saved: {out_path}")

    if return_full_predictions:
        return rule, results, full_preds
    return rule, results
