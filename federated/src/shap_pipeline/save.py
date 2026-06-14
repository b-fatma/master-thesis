"""Saving utilities for SHAP outputs and metadata.

This module handles persistence of SHAP values, explanation data, and associated
metadata in a structured nested directory hierarchy:
    round_<r>/client_<k>/shap_values.npz
    round_<r>/client_<k>/metadata.json
    round_<r>/client_<k>/plots/{bar,beeswarm}/

The nested structure organizes SHAP artifacts by federated round and client
partition, with separate subdirectories for different plot types.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import json
import os
from typing import Dict

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShapArtifactMetadata:
    """Immutable metadata associated with a SHAP artifact.

    Stores contextual information about how a SHAP artifact was computed,
    including the dataset configuration, round number, client partition,
    and whether the client was malicious (in attack scenarios).

    Attributes:
        run_name: Name of the federated learning run (sanitized).
        dataset_name: Name of the dataset used for SHAP explanation.
        partition_id: Client partition identifier (0-based index or string ID).
        malicious: True if client was compromised in attack scenario, None otherwise.
        round_number: Federated round during which SHAP was computed.
        dataset_source: Source of the dataset (e.g., 'server-side dataset pipeline').
        background_samples: Number of background samples used for SHAP computation.
        explanation_samples: Number of explanation samples used for SHAP computation.
        client_identifier: Full model identifier (e.g., 'client_0').
        wandb_run: Whether artifact is associated with a W&B run.
    """

    run_name: str
    dataset_name: str
    partition_id: str
    malicious: bool | None
    round_number: int | None
    dataset_source: str
    background_samples: int
    explanation_samples: int
    client_identifier: str | None = None
    wandb_run: bool = False


def _sanitize_component(value: str) -> str:
    """Sanitize string for safe use in filesystem directory names.

    Replaces path separators, spaces, and forward slashes with underscores
    to ensure the string is safe for use as a filesystem directory component.

    Args:
        value: String to sanitize.

    Returns:
        Sanitized string with all problematic characters replaced with underscores.
    """
    return value.replace(os.sep, "_").replace(" ", "_").replace("/", "_")


def _format_partition_id(partition_id: str) -> str:
    """Convert partition ID to display format with 1-based indexing.

    For numeric partition IDs (0-based), convert to 1-based display format.
    For non-numeric IDs, sanitize and return as-is.

    Args:
        partition_id: Partition identifier (0-based numeric or alphanumeric string).

    Returns:
        1-based numeric string (e.g., '0' -> '1') or sanitized non-numeric string.

    Examples:
        - '0' -> '1' (0-based to 1-based)
        - '9' -> '10' (0-based to 1-based)
        - 'None' -> 'None' (non-numeric, unchanged)
        - 'client-A' -> 'client-A' (non-numeric, sanitized)
    """
    try:
        if partition_id.isdigit():
            return str(int(partition_id) + 1)
    except Exception:
        pass
    return _sanitize_component(str(partition_id))


def _format_round_id(round_number: int | None) -> str:
    """Format federated learning round number as directory name.

    Converts round number to 2-digit zero-padded format (e.g., round_01).
    Returns 'round_unknown' if round number is None.

    Args:
        round_number: Federated round number (1-based) or None.

    Returns:
        Formatted string like 'round_01', 'round_02', or 'round_unknown' if None.
    """
    if round_number is None:
        return "round_unknown"
    return f"round_{int(round_number):02d}"


def _format_client_id(partition_id: str) -> str:
    """Format partition ID as client directory name.

    Creates a string like 'client_1', 'client_2', etc. by prefixing
    the formatted partition ID with 'client_'.

    Args:
        partition_id: Partition identifier (0-based numeric or alphanumeric string).

    Returns:
        Formatted client name like 'client_1', 'client_2'.
    """
    return f"client_{_format_partition_id(partition_id)}"


def build_shap_stem(metadata: ShapArtifactMetadata) -> str:
    """Build the canonical filename stem for SHAP outputs."""

    parts: list[str] = []
    # Use requested convention: round<r>_client<k>
    if metadata.round_number is not None:
        parts.append(f"round_{int(metadata.round_number):02d}")

    parts.append(f"client_{_format_partition_id(metadata.partition_id)}")

    if metadata.malicious is not None:
        parts.append(f"malicious_{str(bool(metadata.malicious)).lower()}")

    return "_".join(parts)


def build_run_output_dirs(output_root: str, run_name: str) -> Dict[str, str]:
    """Create and return the root directory for a federated learning run.

    Creates the directory structure <output_root>/<run_name> if it does not
    exist. The run_name is sanitized to remove path separators and spaces.

    Args:
        output_root: Root directory path for all SHAP outputs.
        run_name: Name of the federated learning run (will be sanitized).

    Returns:
        Dictionary with key 'run_dir' mapping to the created directory path.
    """
    run_dir = os.path.join(output_root, _sanitize_component(run_name))
    os.makedirs(run_dir, exist_ok=True)
    return {
        "run_dir": run_dir,
    }


def build_shap_artifact_dirs(
    output_root: str, metadata: ShapArtifactMetadata
) -> Dict[str, str]:
    """Create nested directory structure for SHAP artifacts of one client in one round.

    Creates the full hierarchy: output_root/<run>/<round>/<client>/plots/{bar,beeswarm}/
    This organization separates artifacts by federated round, client partition,
    and plot type for easy discovery and management.

    Args:
        output_root: Root directory for all SHAP outputs.
        metadata: Metadata specifying run name, round, and partition.

    Returns:
        Dictionary with keys 'run_dir', 'round_dir', 'client_dir', 'plots_dir',
        'bar_dir', 'beeswarm_dir' mapping to the created directory paths.
    """
    run_dir = os.path.join(output_root, _sanitize_component(metadata.run_name))
    round_dir = os.path.join(run_dir, _format_round_id(metadata.round_number))
    client_dir = os.path.join(round_dir, _format_client_id(metadata.partition_id))
    plots_dir = os.path.join(client_dir, "plots")
    bar_dir = os.path.join(plots_dir, "bar")
    beeswarm_dir = os.path.join(plots_dir, "beeswarm")
    os.makedirs(bar_dir, exist_ok=True)
    os.makedirs(beeswarm_dir, exist_ok=True)
    return {
        "run_dir": run_dir,
        "round_dir": round_dir,
        "client_dir": client_dir,
        "plots_dir": plots_dir,
        "bar_dir": bar_dir,
        "beeswarm_dir": beeswarm_dir,
    }


def save_shap_artifact(
    shap_values: np.ndarray,
    explanation_data: torch.Tensor | np.ndarray,
    feature_names: list[str],
    metadata: ShapArtifactMetadata,
    output_root: str,
) -> Dict[str, str]:
    """Save SHAP values and associated metadata to disk in nested directory structure.

    Writes compressed NPZ file containing SHAP values, explanation data, feature names,
    and embedded metadata. Also writes a human-readable JSON metadata file for easy
    inspection. All files are stored in the nested round/client directory hierarchy.

    Args:
        shap_values: SHAP value matrix of shape (num_samples, num_features).
        explanation_data: Explanation samples used to compute SHAP values
            (shape (num_samples, num_features)) as torch.Tensor or np.ndarray.
        feature_names: List of feature names corresponding to SHAP value columns.
        metadata: Metadata object describing the SHAP computation context.
        output_root: Root directory for all SHAP outputs.

    Returns:
        Dictionary with keys 'values' and 'metadata' mapping to the saved file paths.

    Writes:
        - <output_root>/<run>/<round>/<client>/shap_values.npz (compressed binary)
        - <output_root>/<run>/<round>/<client>/metadata.json (human-readable)

    Note:
        Creates parent directories as needed. torch.Tensor objects are converted
        to numpy arrays for efficient storage.
    """

    dirs = build_shap_artifact_dirs(output_root, metadata)

    explanation_np = (
        explanation_data.detach().cpu().numpy()
        if isinstance(explanation_data, torch.Tensor)
        else np.asarray(explanation_data)
    )

    values_path = os.path.join(dirs["client_dir"], "shap_values.npz")
    meta_path = os.path.join(dirs["client_dir"], "metadata.json")

    np.savez_compressed(
        values_path,
        values=np.asarray(shap_values),
        explanation_data=explanation_np,
        feature_names=np.asarray(feature_names, dtype=object),
        metadata_json=json.dumps(asdict(metadata)),
    )

    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(asdict(metadata), handle, indent=2)

    logger.info("Saved SHAP artifact: %s", values_path)
    logger.info("Saved SHAP metadata: %s", meta_path)
    return {"values": values_path, "metadata": meta_path}
