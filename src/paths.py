"""Central path configuration for MovementGNN-COO.

Environment variables (all optional):
    MOVEMENTGNN_DATA_ROOT     Root for graphs, raw downloads, extracted analysis data
    MOVEMENTGNN_RESULTS_ROOT  Root for training outputs, predictions, analysis results
    MOVEMENTGNN_WORK_ROOT     Legacy Colab Drive root (used only when running in Colab
                              and DATA_ROOT / RESULTS_ROOT are not overridden)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser()
    return default


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _use_repo_layout() -> bool:
    return bool(
        os.environ.get("MOVEMENTGNN_DATA_ROOT")
        or os.environ.get("MOVEMENTGNN_RESULTS_ROOT")
        or not is_colab()
    )


DATA_ROOT = _path_from_env("MOVEMENTGNN_DATA_ROOT", REPO_ROOT / "data")
RESULTS_ROOT = _path_from_env("MOVEMENTGNN_RESULTS_ROOT", REPO_ROOT / "results")

GRAPH_DIR = DATA_ROOT / "graph"
RAW_DIR = DATA_ROOT / "raw"
ANALYSIS_DATA_DIR = DATA_ROOT / "analysis"
GEO_DIR = DATA_ROOT / "geo"

GRAPH_READY = GRAPH_DIR / "pacaraima_q1q4_ready.pt"
GRAPH_FIXED = GRAPH_DIR / "pacaraima_q1q4_fixed.pt"
EMBEDDINGS_256 = GRAPH_DIR / "pacaraima_q1q4_embeddings_256.pt"
TOPO_EMBEDDING_CKPT = GRAPH_DIR / "baseline_Q4_p5_epoch=01_val_loss=0.0653.ckpt"

RAW_DEM = RAW_DIR / "dem"
RAW_SENTINEL2 = RAW_DIR / "sentinel2" / "pacaraima"
RAW_LIDAR = RAW_DIR / "lidar"

RESULTS_V1 = RESULTS_ROOT / "v1"
RESULTS_V2 = RESULTS_ROOT / "v2"
RESULTS_V2_LOCAL = RESULTS_ROOT / "v2_local"
RESULTS_V2_ARMORED = RESULTS_ROOT / "v2_armored_finetune"

BASELINES_DIR = RESULTS_ROOT / "baselines"
ANALYSIS_OUTPUT_DIR = RESULTS_ROOT / "analysis"


def get_colab_base() -> Path:
    return _path_from_env("MOVEMENTGNN_WORK_ROOT", Path("/content/drive/MyDrive/GEOINT"))


def _glob_first(root: Path, pattern: str) -> Path | None:
    candidates = sorted(root.glob(pattern), reverse=True)
    return candidates[0] if candidates else None


def default_graph_path() -> Path:
    if GRAPH_READY.exists():
        return GRAPH_READY
    if GRAPH_FIXED.exists():
        return GRAPH_FIXED
    return GRAPH_READY


def default_gnn_results_dir() -> Path:
    for directory in (RESULTS_V2_ARMORED, RESULTS_V2_LOCAL, RESULTS_V2, RESULTS_V1):
        if (directory / "predictions_full.npz").exists():
            return directory
    return RESULTS_V2_ARMORED


def geotiff_dir() -> Path:
    results = default_gnn_results_dir()
    nested = results / "geotiff"
    return nested if nested.exists() else results


def resolve_geotiff(fraction: str) -> Path:
    results = default_gnn_results_dir()
    patterns = [
        f"restricao_{fraction}_movement_gnn_v2_refined.tif",
        f"restricao_{fraction}_movement_gnn_v2_fix.tif",
        f"restricao_{fraction}_movement_gnn_v2.tif",
    ]
    for name in patterns:
        candidate = results / name
        if candidate.exists():
            return candidate
    return results / patterns[0]


def get_training_paths(variant: str = "v2_local") -> dict[str, Path]:
    """Return graph, embedding, and output directories for training scripts."""
    if not _use_repo_layout():
        base = get_colab_base()
        out_map = {
            "v1": base / "gnn/results",
            "v2": base / "gnn/results_v2",
            "v2_local": base / "gnn/results_v2_local",
            "v2_armored": base / "gnn/results_v2_armored_finetune",
        }
        graph_name = "pacaraima_q1q4_fixed.pt" if variant == "v1" else "pacaraima_q1q4_ready.pt"
        return {
            "graph": base / "graph" / graph_name,
            "emb_dir": base / "graph",
            "out_dir": out_map.get(variant, out_map["v2_local"]),
        }

    out_map = {
        "v1": RESULTS_V1,
        "v2": RESULTS_V2,
        "v2_local": RESULTS_V2_LOCAL,
        "v2_armored": RESULTS_V2_ARMORED,
    }
    graph = GRAPH_FIXED if variant == "v1" else default_graph_path()
    return {
        "graph": graph,
        "emb_dir": GRAPH_DIR,
        "out_dir": out_map.get(variant, RESULTS_V2_LOCAL),
    }


def get_inference_paths(variant: str = "auto") -> dict[str, Path]:
    """Return paths for full-graph inference and GeoTIFF export."""
    if not _use_repo_layout():
        base = get_colab_base()
        out_dir = base / "gnn/results_v2"
        return {
            "graph": base / "graph/pacaraima_q1q4_fixed.pt",
            "emb_dir": base / "graph",
            "ckpt": out_dir / "checkpoints/checkpoint_best.pt",
            "out_dir": out_dir,
        }

    if variant == "auto":
        out_dir = default_gnn_results_dir()
    else:
        out_dir = get_training_paths(variant)["out_dir"]

    return {
        "graph": default_graph_path(),
        "emb_dir": GRAPH_DIR,
        "ckpt": out_dir / "checkpoints/checkpoint_best.pt",
        "out_dir": out_dir,
    }


def get_embedding_extraction_paths(local: bool = True) -> dict[str, Path]:
    if not _use_repo_layout() and not local:
        base = get_colab_base()
        return {
            "ckpt": base / "graph/baseline_Q4_p5_epoch=01_val_loss=0.0653.ckpt",
            "graph": base / "graph/pacaraima_q1q4_ready.pt",
            "out_dir": base / "graph",
        }
    return {
        "ckpt": TOPO_EMBEDDING_CKPT,
        "graph": default_graph_path(),
        "out_dir": GRAPH_DIR,
    }


def get_analysis_paths(name: str) -> dict[str, Any]:
    """Return input/output paths for post-training analysis scripts."""
    if not _use_repo_layout():
        base = get_colab_base()
        gnn = base / "gnn/results_v2"
        return {
            "graph": base / "graph/pacaraima_q1q4_fixed.pt",
            "gnn_preds": gnn / "predictions_full.npz",
            "gnn_probs": gnn / "probs_full.npz",
            "rf_preds": base / "baseline_classic/results/random_forest/predictions_full.npz",
            "mlp_preds": base / "baseline_classic/results/mlp/predictions_full.npz",
            "rule_preds": base / "baseline_classic/results/rule_based/predictions_full.npz",
            "emb_path": base / "graph/pacaraima_q1q4_embeddings_256.pt",
            "data_dir": None,
            "out_dir": base / "gnn/analise" / name,
            "metrics_paths": {
                "GNN": [gnn / "metrics.json"],
                "RF": [base / "baseline_classic/results/random_forest/metrics.json"],
                "MLP": [base / "baseline_classic/results/mlp/metrics.json"],
                "Rule": [base / "baseline_classic/results/rule_based/metrics.json"],
            },
        }

    gnn_dir = default_gnn_results_dir()
    rf_preds = _glob_first(BASELINES_DIR, "**/random_forest/predictions_full.npz")
    mlp_preds = _glob_first(BASELINES_DIR, "**/mlp/predictions_full.npz")
    rule_preds = _glob_first(BASELINES_DIR, "**/rule_based/predictions_full.npz")

    return {
        "graph": default_graph_path(),
        "gnn_preds": gnn_dir / "predictions_full.npz",
        "gnn_probs": gnn_dir / "probs_full.npz",
        "rf_preds": rf_preds or (BASELINES_DIR / "random_forest/predictions_full.npz"),
        "mlp_preds": mlp_preds or (BASELINES_DIR / "mlp/predictions_full.npz"),
        "rule_preds": rule_preds or (BASELINES_DIR / "rule_based/predictions_full.npz"),
        "emb_path": EMBEDDINGS_256,
        "data_dir": ANALYSIS_DATA_DIR,
        "out_dir": ANALYSIS_OUTPUT_DIR / name,
        "metrics_paths": {
            "GNN": [gnn_dir / "metrics.json"],
            "RF": [BASELINES_DIR / "random_forest/metrics.json"],
            "MLP": [BASELINES_DIR / "mlp/metrics.json"],
            "Rule": [BASELINES_DIR / "rule_based/metrics.json"],
        },
    }
