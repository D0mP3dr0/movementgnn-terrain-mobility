"""
GraphIntegrator - Integrates DAMEPLAN labels into an existing graph.

The dataset already contains a complete graph with:
- Nodes: DEM points with 17 topographic features
- Edges: Spatial neighbor connections

This module adds the generated restriction labels (Unrestricted/Restricted/
Severely Restricted) for each fraction to the existing graph, producing a
training-ready dataset.
"""

import torch
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime


class GraphIntegrator:
    """
    Integrates movement restriction labels into an existing graph.

    Uses the pre-built graph to avoid reprocessing topology and features.
    """

    def __init__(self, graph_path: str, labels_dir: str):
        """
        Args:
            graph_path: Path to the existing graph (.pt file).
            labels_dir: Directory containing generated label files.
        """
        self.graph_path = Path(graph_path)
        self.labels_dir = Path(labels_dir)
        self.data = None
        self.labels = None

    def load_graph(self) -> Any:
        """Load the existing graph."""
        print(f"Loading graph: {self.graph_path.name}")

        from torch_geometric.data import HeteroData

        self.data = torch.load(
            str(self.graph_path),
            map_location='cpu',
            weights_only=False
        )

        print(f"  Type: {type(self.data).__name__}")

        if isinstance(self.data, HeteroData):
            print(f"  DEM nodes: {self.data['dem'].x.shape[0]:,}")

        return self.data

    def load_labels(self) -> Dict[str, torch.Tensor]:
        """Load generated labels."""
        print(f"\nLoading labels from: {self.labels_dir}")

        self.labels = {}
        for fraction in ['a_pe', 'motorizada', 'mecanizada', 'blindada']:
            path = self.labels_dir / f'labels_{fraction}.pt'
            if path.exists():
                self.labels[fraction] = torch.load(path)
                print(f"  {fraction}: {self.labels[fraction].shape[0]:,} labels")
            else:
                print(f"  {fraction}: file not found")

        return self.labels

    def integrate(self) -> Any:
        """
        Integrate labels into the graph.

        Returns:
            Graph with labels added.
        """
        if self.data is None:
            self.load_graph()
        if self.labels is None:
            self.load_labels()

        print("\nIntegrating labels into graph...")

        from torch_geometric.data import HeteroData

        if isinstance(self.data, HeteroData):
            num_nodes = self.data['dem'].x.shape[0]
        else:
            num_nodes = self.data.x.shape[0]

        for fraction, labels in self.labels.items():
            if labels.shape[0] != num_nodes:
                raise ValueError(
                    f"Mismatch: {fraction} has {labels.shape[0]} labels, "
                    f"but graph has {num_nodes} nodes"
                )

            attr_name = f'y_{fraction}'
            if isinstance(self.data, HeteroData):
                self.data['dem'][attr_name] = labels
            else:
                setattr(self.data, attr_name, labels)

            print(f"  Added: {attr_name}")

        if isinstance(self.data, HeteroData):
            self.data['dem'].labels_generated = datetime.now().isoformat()

        print("\nIntegration complete.")

        return self.data

    def save(self, output_path: str) -> str:
        """
        Save graph with integrated labels.

        Args:
            output_path: Output file path.

        Returns:
            Path of the saved file.
        """
        print(f"\nSaving integrated graph: {output_path}")

        torch.save(self.data, output_path)

        size_gb = Path(output_path).stat().st_size / (1024**3)
        print(f"  Size: {size_gb:.2f} GB")

        return output_path

    def verify_integration(self) -> bool:
        """Verify that integration was successful."""
        print("\nVerifying integration...")

        from torch_geometric.data import HeteroData

        all_ok = True

        for fraction in ['a_pe', 'motorizada', 'mecanizada', 'blindada']:
            attr_name = f'y_{fraction}'

            if isinstance(self.data, HeteroData):
                has_attr = hasattr(self.data['dem'], attr_name)
            else:
                has_attr = hasattr(self.data, attr_name)

            if has_attr:
                print(f"  {attr_name} present")
            else:
                print(f"  {attr_name} missing")
                all_ok = False

        return all_ok


def integrate_labels_to_graph(
    graph_path: str = None,
    labels_dir: str = None,
    output_path: str = None
) -> str:
    """
    Convenience function to integrate labels into a graph.

    Args:
        graph_path: Path to the original graph.
        labels_dir: Directory with label files.
        output_path: Output path.

    Returns:
        Path of the saved file.
    """
    if graph_path is None:
        raise ValueError("graph_path must be provided.")
    if labels_dir is None:
        raise ValueError("labels_dir must be provided.")
    if output_path is None:
        raise ValueError("output_path must be provided.")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    integrator = GraphIntegrator(graph_path, labels_dir)
    integrator.integrate()
    integrator.verify_integration()
    integrator.save(output_path)

    return output_path
