"""
RandomForestBaseline - Random Forest classifier (sklearn).

Serves as the tabular gold-standard baseline. It uses the same per-node
features as the GNN but ignores graph structure (spatial neighbors),
demonstrating the benefit of neighborhood aggregation.
"""

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from typing import Dict, Tuple
import json
from pathlib import Path
import joblib


class RandomForestBaseline:
    """Random Forest for mobility restriction classification."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 20,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.models = {}
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.results = {}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: Dict[str, np.ndarray]
    ):
        """
        Train one RF per fraction.

        Args:
            X_train: [N, 17] features
            y_train: Dict {fraction_name: [N] labels}
        """
        print("Training Random Forest baselines...")

        for frac, labels in y_train.items():
            print(f"  {frac}...", end=" ")

            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                class_weight='balanced'
            )

            rf.fit(X_train, labels)
            self.models[frac] = rf
            print("done")

        print("RF training complete.")

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict for all fractions."""
        preds = {}
        for frac, model in self.models.items():
            preds[frac] = model.predict(X)
        return preds

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: Dict[str, np.ndarray]
    ) -> Dict[str, Dict]:
        """
        Evaluate performance on test set.

        Returns:
            Dict with metrics per fraction.
        """
        print("\nEvaluating Random Forest...")

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

    def get_feature_importance(self, feature_names: list = None) -> Dict[str, list]:
        """Returns feature importances per fraction."""
        importance = {}

        for frac, model in self.models.items():
            imp = model.feature_importances_

            if feature_names:
                importance[frac] = list(zip(feature_names, imp.tolist()))
            else:
                importance[frac] = imp.tolist()

        return importance

    def save_results(self, path: str):
        """Save results to JSON."""
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"RF results saved: {path}")

    def save_models(self, output_dir: str):
        """Save trained RF models (one per fraction)."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        manifest = {
            "model_type": "RandomForestClassifier",
            "fractions": list(self.models.keys()),
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "random_state": self.random_state,
        }

        for frac, model in self.models.items():
            model_path = out / f"rf_{frac}.joblib"
            joblib.dump(model, model_path)

        with open(out / "rf_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"RF models saved to: {out}")


def train_rf_baseline(
    data,
    output_dir: str = None,
    model_dir: str = None,
    full_predictions_path: str = None,
    return_full_predictions: bool = False
):
    """
    Convenience function to train and evaluate the RF baseline.

    Args:
        data: PyG Data object with features and label attributes.
        output_dir: Directory for saving metric results.
        model_dir: Directory for saving trained model artifacts.
        full_predictions_path: Path for saving full-node predictions (.npz).
        return_full_predictions: If True, also returns predictions for all nodes.

    Returns:
        Tuple (RandomForestBaseline, results) or (RandomForestBaseline, results, full_preds)
    """
    X = data.x.numpy()
    train_mask = data.train_mask.numpy()
    test_mask = data.test_mask.numpy()

    y = {
        'a_pe': data.y_a_pe.numpy(),
        'motorizada': data.y_motorizada.numpy(),
        'mecanizada': data.y_mecanizada.numpy(),
        'blindada': data.y_blindada.numpy(),
    }

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = {k: v[train_mask] for k, v in y.items()}
    y_test = {k: v[test_mask] for k, v in y.items()}

    rf = RandomForestBaseline()
    rf.fit(X_train, y_train)
    results = rf.evaluate(X_test, y_test)

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        rf.save_results(f"{output_dir}/rf_results.json")

    if model_dir:
        rf.save_models(model_dir)

    full_preds = rf.predict(X)
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
        print(f"RF full predictions saved: {out_path}")

    if return_full_predictions:
        return rf, results, full_preds
    return rf, results
