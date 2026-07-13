"""Main training script for MovementGNN."""

import argparse
import sys
from pathlib import Path

import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models.movement_gnn import MovementGNN
from src.models.dameplan_loss import DAMEPLANLoss
from src.paths import GRAPH_READY, RESULTS_V1
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train MovementGNN on a prepared graph.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=GRAPH_READY,
        help="Path to the prepared HeteroData graph (.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_V1,
        help="Directory for checkpoints and training logs",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Training on device: {device} ---")

    print(f"Loading graph: {args.data_path}")
    hetero_data = torch.load(args.data_path, map_location="cpu", weights_only=False)

    print("Extracting DEM subgraph...")
    from torch_geometric.data import Data

    dem_store = hetero_data["dem"]

    edge_index = None
    for edge_type in hetero_data.edge_types:
        if edge_type[0] == "dem" and edge_type[2] == "dem":
            edge_index = hetero_data[edge_type].edge_index
            print(f"  Edges found: {edge_type}")
            break

    if edge_index is None:
        raise ValueError("Could not find DEM-DEM edges")

    data = Data(
        x=dem_store.x,
        edge_index=edge_index,
        y_a_pe=dem_store.y_a_pe,
        y_motorizada=dem_store.y_motorizada,
        y_mecanizada=dem_store.y_mecanizada,
        y_blindada=dem_store.y_blindada,
        train_mask=dem_store.train_mask,
        val_mask=dem_store.val_mask,
        test_mask=dem_store.test_mask,
    )

    model = MovementGNN(
        in_channels=17,
        hidden_channels=args.hidden_channels,
        num_layers=2,
        heads=4,
    )

    criterion = DAMEPLANLoss(weight_ce=1.0, weight_phys=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        output_dir=str(args.output_dir),
    )

    trainer.fit(data, epochs=args.epochs, patience=15)
    trainer.save_history()


if __name__ == "__main__":
    main()
