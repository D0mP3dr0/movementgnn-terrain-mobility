INTERNAL_DATA_ROOT = __import__("os").environ.get("MOVEMENTGNN_DATA_ROOT", ".")
import sys
sys.path.insert(0, '.')

import torch
import torch.optim as optim
from pathlib import Path
import argparse

from src.models.movement_gnn import MovementGNN
from src.models.dameplan_loss import DAMEPLANLoss
from src.training.trainer import Trainer

def main():
    DATA_PATH = rf"{INTERNAL_DATA_ROOT}\GNN_RESTRICAO_MOVIMENTO\data\graphs\pirassununga_ready.pt"
    OUTPUT_DIR = rf"{INTERNAL_DATA_ROOT}\GNN_RESTRICAO_MOVIMENTO\output\models\run_01"
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 100
    HIDDEN_CHANNELS = 64
    LR = 0.001
    
    print(f"--- STARTING TRAINING ON DEVICE: {DEVICE} ---")
    
    print(f"Carregando grafo: {DATA_PATH}")
    hetero_data = torch.load(DATA_PATH, map_location='cpu', weights_only=False)
    
    print("Extraindo subgrafo DEM...")
    from torch_geometric.data import Data
    
    dem_store = hetero_data['dem']
    
    edge_index = None
    for edge_type in hetero_data.edge_types:
        if edge_type[0] == 'dem' and edge_type[2] == 'dem':
            edge_index = hetero_data[edge_type].edge_index
            print(f"  Arestas encontradas: {edge_type}")
            break
            
    if edge_index is None:
        raise ValueError('Could not find DEM-DEM edges')
    
    data = Data(
        x=dem_store.x,
        edge_index=edge_index,
        y_a_pe=dem_store.y_a_pe,
        y_motorizada=dem_store.y_motorizada,
        y_mecanizada=dem_store.y_mecanizada,
        y_blindada=dem_store.y_blindada,
        train_mask=dem_store.train_mask,
        val_mask=dem_store.val_mask,
        test_mask=dem_store.test_mask
    )
    
    
    model = MovementGNN(
        in_channels=17,
        hidden_channels=HIDDEN_CHANNELS,
        num_layers=2,
        heads=4
    )
    
    criterion = DAMEPLANLoss(weight_ce=1.0, weight_phys=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        output_dir=OUTPUT_DIR
    )
    
    history = trainer.fit(data, epochs=EPOCHS, patience=15)
    
    trainer.save_history()

if __name__ == "__main__":
    main()
