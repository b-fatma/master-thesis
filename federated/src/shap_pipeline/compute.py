"""Low-level SHAP computation utilities for neural network models.

This module provides core SHAP computation functions that compute feature
importance explanations for PyTorch models using the SHAP GradientExplainer.

Key functions:
- compute_shap_matrix: Compute signed SHAP values for model predictions.
- compute_shap_for_models: Batch compute SHAP for multiple models.

The module deliberately avoids persistence and plotting concerns to keep
computations testable and reusable.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    import shap
except ImportError as exc:  # pragma: no cover - dependency issue is surfaced to users
    raise ImportError("Install shap: pip install shap --break-system-packages") from exc


@dataclass(frozen=True)
class ShapComputationConfig:
    """Immutable configuration parameters for SHAP computation.

    Attributes:
        background_samples: Number of background samples for GradientExplainer.
            Used to approximate the expected model output. Default: 50.
        explanation_samples: Number of samples to explain. If None, all provided
            explanation data is used. Default: None (use all data).
    """

    background_samples: int = 50
    explanation_samples: int | None = None


def _wrap_model(model: nn.Module) -> nn.Module:
    """Wrap model output to guaranteed shape (N, 1) for GradientExplainer compatibility.

    SHAP's GradientExplainer requires model output to have shape (N, 1) for proper
    gradient computation. This wrapper ensures models returning 1D outputs
    (e.g., shape (N,)) are reshaped to (N, 1).

    Args:
        model: PyTorch model to wrap.

    Returns:
        Wrapped model that outputs (N, 1) shaped tensors.
    """

    class _Wrapper(nn.Module):
        def __init__(self, net: nn.Module):
            super().__init__()
            self.net = net

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.net(x)
            if out.dim() == 1:
                out = out.unsqueeze(1)
            return out

    return _Wrapper(model)


def _normalize_shap_values(shap_values: object) -> np.ndarray:
    """Normalize SHAP output to consistent numpy array format (samples x features).

    SHAP's GradientExplainer can return values in various shapes depending on the
    model architecture. This function normalizes to (num_samples, num_features).
    Handles:
    - List of arrays (from multi-output SHAP) -> averaged
    - 3D arrays (from recurrent models) -> squeezed or averaged

    Args:
        shap_values: Raw output from SHAP explainer (list, array, or nested structure).

    Returns:
        2D numpy array of shape (num_samples, num_features) with SHAP values.
    """

    if isinstance(shap_values, list):
        shap_values = np.mean(np.stack(shap_values, axis=0), axis=0)

    values = np.asarray(shap_values)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    elif values.ndim == 3:
        values = values.mean(axis=-1)

    return values


def compute_shap_matrix(
    model: nn.Module,
    background_data: torch.Tensor,
    explanation_data: torch.Tensor,
    background_samples: int = 50,
) -> np.ndarray:
    """Compute SHAP value matrix for one model using GradientExplainer.

    Computes feature attribution scores using SHAP's gradient-based method.
    Returns signed values indicating positive (increases prediction) and
    negative (decreases prediction) feature contributions.

    Args:
        model: PyTorch model (will be moved to appropriate device).
        background_data: Baseline samples for SHAP computation (shape: (N, features)).
            GradientExplainer uses these to approximate expected model output.
        explanation_data: Samples to explain (shape: (M, features)).
        background_samples: Number of background samples to use (subset of background_data).
            Default: 50.

    Returns:
        2D numpy array of shape (len(explanation_data), num_features) with SHAP values.

    Note:
        Model is set to eval mode. Returns zeros on computation failure with a warning.
        Automatically detects device from background_data and moves model there.
    """

    device = background_data.device
    model.eval()
    model.to(device)

    wrapped = _wrap_model(model)
    background = background_data[:background_samples]

    try:
        explainer = shap.GradientExplainer(wrapped, background)
        shap_values = explainer.shap_values(explanation_data)
        return _normalize_shap_values(shap_values)
    except Exception as exc:
        logger.warning("SHAP matrix computation failed: %s -- returning zeros", exc)
        return np.zeros((explanation_data.shape[0], explanation_data.shape[1]))


def compute_shap_for_models(
    models_dict: Dict[str, nn.Module],
    background_data: torch.Tensor,
    explanation_data: torch.Tensor,
    background_samples: int = 50,
) -> Dict[str, np.ndarray]:
    """Compute SHAP value matrices for multiple models in parallel-compatible manner.

    Iterates over a dictionary of models and computes SHAP matrices for each,
    useful for federated learning where multiple client models need explanation.

    Args:
        models_dict: Dictionary mapping model identifiers to PyTorch models.
        background_data: Baseline samples for SHAP computation.
        explanation_data: Samples to explain.
        background_samples: Number of background samples to use.

    Returns:
        Dictionary mapping model identifiers to SHAP value matrices (2D numpy arrays).

    Note:
        Models are processed in sorted order by key for reproducibility.
    """

    results: Dict[str, np.ndarray] = {}
    for model_id, model in sorted(models_dict.items(), key=lambda item: str(item[0])):
        logger.info("Computing SHAP for model %s", model_id)
        results[model_id] = compute_shap_matrix(
            model,
            background_data=background_data,
            explanation_data=explanation_data,
            background_samples=background_samples,
        )
    return results
