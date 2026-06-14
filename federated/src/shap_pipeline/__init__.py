"""Reusable SHAP pipeline utilities for federated analysis.

This module provides end-to-end utilities for computing, saving, loading, and
visualizing SHAP (SHapley Additive exPlanations) values in federated learning
contexts. It includes support for server-side SHAP computation, nested artifact
storage, and integration with Weights & Biases (W&B) for experiment tracking.

Key components:
- compute: Low-level SHAP matrix/vector computation from models
- save: Persist SHAP values with structured metadata to disk
- loading: Load artifacts from local filesystem or W&B
- plotting: Generate bar and beeswarm plots from SHAP outputs
- pipeline: High-level orchestration for federated server integration
"""

from .compute import (
    ShapComputationConfig,
    compute_shap_for_models,
    compute_shap_matrix,
)
from .loading import LoadedShapArtifact, load_local_shap_artifacts
from .pipeline import (
    ShapRuntimeContext,
    build_server_side_shap_context,
    compute_and_save_round_shap,
    generate_plots_for_run,
)
from .plotting import generate_shap_plots
from .save import ShapArtifactMetadata, save_shap_artifact

__all__ = [
    "ShapComputationConfig",
    "compute_shap_for_models",
    "compute_shap_matrix",
    "LoadedShapArtifact",
    "load_local_shap_artifacts",
    "ShapRuntimeContext",
    "build_server_side_shap_context",
    "compute_and_save_round_shap",
    "generate_plots_for_run",
    "generate_shap_plots",
    "ShapArtifactMetadata",
    "save_shap_artifact",
]
