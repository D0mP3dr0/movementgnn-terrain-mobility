"""
MLPBaseline - Simple MLP neural network (no graph structure).

Uses the same features as the GNN but does not leverage spatial
neighborhood information, demonstrating the gain from using
graph-based message passing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict
import json
from pathlib import Path


class MLPClassifier(nn.Module):
    """Simple MLP for classification."""

    def __init__(self, in_features: int = 17, hidden_dim: int = 64, num_classes: int = 3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class MLPBaseline:
    """Wrapper to train one MLP per fraction."""

    def __init__(
        self,
        hidden_dim: int = 64,
        epochs: int = 50,
        batch_size: int = 4096,
        lr: float = 0.001,
        device: str = 'cpu'
    ):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.models = {}
        self.results = {}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: Dict[str, np.ndarray]
    ):
        """
        Train one MLP per fraction.

        Args:
            X_train: [N, F] training features.
            y_train: Dict {fraction_name: [N] labels (1-based)}.
        """
        print("Training MLP baselines...")

        X_tensor = torch.FloatTensor(X_train).to(self.device)

        for frac, labels in y_train.items():
            print(f"  {frac}...", end=" ")

            y_tensor = torch.LongTensor(labels - 1).to(self.device)  # 0-based

            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            model = MLPClassifier(
                in_features=X_train.shape[1],
                hidden_dim=self.hidden_dim
            ).to(self.device)

            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            criterion = nn.CrossEntropyLoss()

            model.train()
            for epoch in range(self.epochs):
                for batch_x, batch_y in loader:
                    optimizer.zero_grad()
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    optimizer.step()

            self.models[frac] = model
            print("done")

        print("MLP training complete.")

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict for all fractions."""
        X_tensor = torch.FloatTensor(X).to(self.device)

        preds = {}
        for frac, model in self.models.items():
            model.eval()
            with torch.no_grad():
                logits = model(X_tensor)
                pred = torch.argmax(logits, dim=1).cpu().numpy() + 1  # 1-based
            preds[frac] = pred

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
        print("\nEvaluating MLP...")

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
        print(f"MLP results saved: {path}")

    def save_models(self, output_dir: str):
        """Save trained MLP model weights (one per fraction)."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        manifest = {
            "model_type": "MLPClassifier",
            "fractions": list(self.models.keys()),
            "hidden_dim": self.hidden_dim,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
        }

        for frac, model in self.models.items():
            model_path = out / f"mlp_{frac}.pt"
            torch.save(model.state_dict(), model_path)

        with open(out / "mlp_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"MLP models saved to: {out}")


def train_mlp_baseline(
    data,
    output_dir: str = None,
    device: str = 'cpu',
    model_dir: str = None,
    full_predictions_path: str = None,
    return_full_predictions: bool = False
):
    """
    Convenience function to train and evaluate the MLP baseline.

    Args:
        data: PyG Data object with features and label attributes.
        output_dir: Directory for saving metric results.
        device: Torch device ('cpu' or 'cuda').
        model_dir: Directory for saving trained model artifacts.
        full_predictions_path: Path for saving full-node predictions (.npz).
        return_full_predictions: If True, also returns predictions for all nodes.

    Returns:
        Tuple (MLPBaseline, results) or (MLPBaseline, results, full_preds)
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

    mlp = MLPBaseline(device=device)
    mlp.fit(X_train, y_train)
    results = mlp.evaluate(X_test, y_test)

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        mlp.save_results(f"{output_dir}/mlp_results.json")

    if model_dir:
        mlp.save_models(model_dir)

    full_preds = mlp.predict(X)
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
        print(f"MLP full predictions saved: {out_path}")

    if return_full_predictions:
        return mlp, results, full_preds
    return mlp, results
