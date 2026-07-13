"""
BaselineRunner - Runs all baselines and produces a comparative report.
"""

import torch
import json
from pathlib import Path
from datetime import datetime

from .random_forest_baseline import train_rf_baseline
from .mlp_baseline import train_mlp_baseline
from .rule_based_baseline import train_rule_baseline


def run_all_baselines(
    data_path: str = None,
    output_dir: str = None,
    save_artifacts: bool = True
) -> dict:
    """
    Run all baselines and generate a comparative report.

    Args:
        data_path: Path to the prepared graph (.pt file).
        output_dir: Output directory for results.
        save_artifacts: Whether to save model checkpoints and full predictions.

    Returns:
        Dict with results from all baselines.
    """
    from torch_geometric.data import Data

    if data_path is None:
        raise ValueError("data_path must be provided.")

    if output_dir is None:
        raise ValueError("output_dir must be provided.")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(output_dir) / "artifacts"
    model_dir = artifacts_dir / "models"
    pred_dir = artifacts_dir / "predictions_full"
    if save_artifacts:
        model_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RUNNING BASELINES")
    print("=" * 60)

    print(f"\nLoading data: {data_path}")
    hetero_data = torch.load(data_path, map_location='cpu', weights_only=False)

    dem_store = hetero_data['dem']

    edge_index = None
    for edge_type in hetero_data.edge_types:
        if edge_type[0] == 'dem' and edge_type[2] == 'dem':
            edge_index = hetero_data[edge_type].edge_index
            break

    data = Data(
        x=dem_store.x,
        edge_index=edge_index,
        pos=getattr(dem_store, "pos", None),
        y_a_pe=dem_store.y_a_pe,
        y_motorizada=dem_store.y_motorizada,
        y_mecanizada=dem_store.y_mecanizada,
        y_blindada=dem_store.y_blindada,
        train_mask=dem_store.train_mask,
        val_mask=dem_store.val_mask,
        test_mask=dem_store.test_mask
    )

    all_results = {}

    # 1. Rule-Based
    print("\n" + "-" * 40)
    print("1. RULE-BASED BASELINE")
    print("-" * 40)
    _, rule_results = train_rule_baseline(
        data,
        output_dir,
        full_predictions_path=str(pred_dir / "rule_based_full_predictions.npz") if save_artifacts else None
    )
    all_results['rule_based'] = rule_results

    # 2. Random Forest
    print("\n" + "-" * 40)
    print("2. RANDOM FOREST BASELINE")
    print("-" * 40)
    _, rf_results = train_rf_baseline(
        data,
        output_dir,
        model_dir=str(model_dir / "random_forest") if save_artifacts else None,
        full_predictions_path=str(pred_dir / "random_forest_full_predictions.npz") if save_artifacts else None
    )
    all_results['random_forest'] = rf_results

    # 3. MLP
    print("\n" + "-" * 40)
    print("3. MLP BASELINE")
    print("-" * 40)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _, mlp_results = train_mlp_baseline(
        data,
        output_dir,
        device,
        model_dir=str(model_dir / "mlp") if save_artifacts else None,
        full_predictions_path=str(pred_dir / "mlp_full_predictions.npz") if save_artifacts else None
    )
    all_results['mlp'] = mlp_results

    # Comparative summary
    print("\n" + "=" * 60)
    print("COMPARATIVE SUMMARY")
    print("=" * 60)

    for frac in ['a_pe', 'motorizada', 'mecanizada', 'blindada']:
        print(f"\n--- {frac.upper()} ---")
        print(f"{'Model':<15} {'Accuracy':>10} {'F1-macro':>10}")
        print("-" * 35)

        for model_name, results in all_results.items():
            acc = results[frac]['accuracy']
            f1 = results[frac]['f1_macro']
            print(f"{model_name:<15} {acc:>10.4f} {f1:>10.4f}")

    report = {
        'timestamp': datetime.now().isoformat(),
        'data_path': str(data_path),
        'results': all_results
    }

    report_path = Path(output_dir) / 'baseline_comparison.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved: {report_path}")
    if save_artifacts:
        print(f"Artifacts saved to: {artifacts_dir}")

    return all_results


if __name__ == "__main__":
    run_all_baselines()
