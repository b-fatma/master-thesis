"""SHAP plotting utilities for generating visualizations from saved artifacts.

This module provides functions to generate bar and beeswarm plots from pre-computed
SHAP values and explanation data. Plots are saved as high-resolution PNG files
in the nested directory structure (plots/bar/bar.png and plots/beeswarm/beeswarm.png).

The module uses matplotlib for rendering and the shap library's plot utilities
for SHAP-specific visualizations.
"""

from __future__ import annotations

import os
import logging
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency path
    import shap
except ImportError as exc:
    raise ImportError("Install shap: pip install shap --break-system-packages") from exc

from .save import ShapArtifactMetadata, build_shap_artifact_dirs, build_shap_stem


def _top_feature_order(values: np.ndarray, max_display: int = 15) -> np.ndarray:
    """Select top features by absolute SHAP value magnitude.

    Computes the absolute value of SHAP values, computes feature importance
    (mean absolute SHAP across samples), and returns indices sorted by importance
    in descending order.

    Args:
        values: SHAP value matrix of shape (num_samples, num_features) or
            (num_features,) for a single sample.
        max_display: Maximum number of top features to return.

    Returns:
        Array of feature indices sorted by importance (descending),
        limited to max_display elements.
    """
    v = np.asarray(values)
    if v.size == 0:
        raise ValueError("Empty values provided to _top_feature_order")

    if v.ndim == 1:
        importances = np.abs(v)
    elif v.ndim == 2:
        importances = np.abs(v).mean(axis=0)
    else:
        raise ValueError(f"Unsupported values ndim: {v.ndim}")

    n_features = importances.size
    if n_features == 0:
        return np.array([], dtype=int)

    display = min(int(max_display), n_features)
    order = np.argsort(importances)[::-1][:display]
    return order


def generate_shap_plots(
    shap_values: np.ndarray,
    feature_names: Sequence[str],
    metadata: ShapArtifactMetadata,
    output_root: str,
    explanation_data: np.ndarray | None = None,
) -> dict[str, str]:
    """Generate SHAP bar and beeswarm plots from precomputed SHAP values.

    Creates two complementary visualizations of SHAP values:
    1. Bar plot: Shows mean absolute SHAP value per feature (feature importance).
    2. Beeswarm plot: Shows distribution of SHAP values per feature across samples.

    Plots are saved as PNG files at:
    - <output_root>/<round>/<client>/plots/bar/bar.png
    - <output_root>/<round>/<client>/plots/beeswarm/beeswarm.png

    Args:
        shap_values: SHAP value matrix of shape (num_samples, num_features) or
            (num_features,) for single-sample explanation.
        feature_names: List or sequence of feature names corresponding to columns.
        metadata: SHAP artifact metadata for plot naming and directory organization.
        output_root: Root directory for output (will create nested round/client structure).
        explanation_data: Original feature values used in explanation (required for beeswarm,
            defaults to shap_values if not provided).

    Returns:
        Dictionary with keys 'bar' and 'beeswarm' mapping to the saved PNG file paths.

    Saves:
        - PNG bar plot at <output_root>/<round>/<client>/plots/bar/bar.png
        - PNG beeswarm plot at <output_root>/<round>/<client>/plots/beeswarm/beeswarm.png

    Note:
        Saves plots at 150 DPI for balance between quality and file size.
        Displays only top 15 features to prevent overcrowding.
    """

    dirs = build_shap_artifact_dirs(output_root, metadata)
    stem = build_shap_stem(metadata)
    values = np.asarray(shap_values)

    # Basic validation of values
    if values.size == 0:
        raise ValueError(f"SHAP values are empty for artifact {metadata}")
    if values.ndim not in (1, 2):
        raise ValueError(
            f"SHAP values must be 1-D or 2-D array, got ndim={values.ndim}"
        )

    # Determine number of features and ensure feature_names available
    num_features = values.shape[0] if values.ndim == 1 else values.shape[1]
    if feature_names is None:
        feature_names = [f"f{i + 1}" for i in range(num_features)]
    else:
        try:
            feature_names = list(feature_names)
        except Exception:
            feature_names = [f"f{i + 1}" for i in range(num_features)]

    if len(feature_names) != num_features:
        logger.warning(
            "feature_names length (%d) does not match number of features (%d); using fallback names",
            len(feature_names),
            num_features,
        )
        feature_names = [f"f{i + 1}" for i in range(num_features)]

    bar_path = os.path.join(dirs["bar_dir"], "bar.png")
    beeswarm_path = os.path.join(dirs["beeswarm_dir"], "beeswarm.png")

    # Bar plot: works from the saved SHAP values alone.
    if values.ndim == 1:
        summary_values = np.asarray(values)
    else:
        summary_values = np.abs(values).mean(axis=0)

    order = _top_feature_order(summary_values)
    plt.figure(figsize=(10, 6))
    labels = [feature_names[i] for i in order]
    plt.barh(labels[::-1], summary_values[order][::-1])
    plt.xlabel("mean(|SHAP|)")
    plt.ylabel("feature")
    plt.title(f"SHAP Bar — {stem}")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Beeswarm plot requires the saved explanation matrix as well.
    if explanation_data is None:
        explanation_data = values

    explanation_data = np.asarray(explanation_data)
    if values.ndim == 1:
        values_for_beeswarm = np.asarray(values)[None, :]
        if explanation_data.ndim == 1:
            explanation_for_beeswarm = explanation_data[None, :]
        elif explanation_data.ndim == 2 and explanation_data.shape[1] == num_features:
            explanation_for_beeswarm = explanation_data
        else:
            raise ValueError("explanation_data shape incompatible with 1-D SHAP values")
    else:
        values_for_beeswarm = np.asarray(values)
        if explanation_data.ndim == 1:
            # assume single sample explanation repeated
            explanation_for_beeswarm = np.repeat(
                explanation_data[None, :], values_for_beeswarm.shape[0], axis=0
            )
        elif explanation_data.ndim == 2:
            if explanation_data.shape[1] != num_features:
                raise ValueError(
                    "explanation_data second-dim does not match number of features"
                )
            explanation_for_beeswarm = explanation_data
        else:
            raise ValueError("explanation_data must be 1-D or 2-D array")

    explanation = shap.Explanation(
        values=values_for_beeswarm,
        base_values=np.zeros(values_for_beeswarm.shape[0]),
        data=explanation_for_beeswarm,
        feature_names=feature_names,
    )
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(
        explanation, show=False, max_display=min(15, len(feature_names))
    )
    plt.title(f"SHAP Beeswarm — {stem}")
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Saved plot: %s", bar_path)
    logger.info("Saved plot: %s", beeswarm_path)
    return {"bar": bar_path, "beeswarm": beeswarm_path}
