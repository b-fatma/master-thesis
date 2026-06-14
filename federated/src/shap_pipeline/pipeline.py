"""High-level SHAP orchestration for federated learning server and offline analysis.

This module provides server-side SHAP context management and integration helpers
for running SHAP computation within the federated learning loop, plus utilities
for offline plot generation from saved artifacts.

Key components:
- ShapRuntimeContext: Maintains SHAP state across federated rounds.
- build_server_side_shap_context: Initialize SHAP from centralized server dataset.
- compute_and_save_round_shap: Orchestrate SHAP computation and persistence per round.
- generate_plots_for_run: Regenerate plots from saved SHAP artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
import os
from typing import Dict

import numpy as np
import torch

logger = logging.getLogger(__name__)

from src.config import get_dataset_config
from src.dataset import load_centralized_dataset

from .compute import compute_shap_matrix
from .loading import load_local_shap_artifacts, load_wandb_shap_artifacts
from .plotting import generate_shap_plots
from .save import ShapArtifactMetadata, build_run_output_dirs, save_shap_artifact


@dataclass(frozen=True)
class ShapRuntimeContext:
    """Immutable SHAP runtime state maintained across federated learning rounds.

    Stores precomputed background and explanation data, feature names, and
    configuration needed for efficient per-round SHAP computation. Reused
    across rounds to avoid reloading dataset.

    Attributes:
        run_name: Name of the federated learning run (sanitized).
        dataset_name: Name of the dataset (e.g., 'adult-income-census').
        output_root: Root directory for saving SHAP artifacts.
        background_data: Background samples for SHAP explainer approximation.
        explanation_data: Samples to explain in each round.
        feature_names: Feature names corresponding to data columns.
        background_samples: Number of background samples used.
        explanation_samples: Number of explanation samples used.
        dataset_source: Source of the dataset (e.g., 'server-side dataset pipeline').
    """

    run_name: str
    dataset_name: str
    output_root: str
    background_data: torch.Tensor
    explanation_data: torch.Tensor
    feature_names: list[str]
    background_samples: int
    explanation_samples: int
    dataset_source: str

    @property
    def run_dir(self) -> str:
        return os.path.join(self.output_root, self.run_name)


def _sanitize_run_name(run_name: str | None, fallback_prefix: str) -> str:
    """Sanitize federated learning run name for use in directory paths.

    Converts spaces, path separators, and forward slashes to underscores.
    If run_name is None, generates a timestamped fallback name.

    Args:
        run_name: Original run name (may contain special characters) or None.
        fallback_prefix: Prefix for generated name if run_name is None.

    Returns:
        Sanitized run name safe for filesystem use, or fallback name with timestamp.
    """
    if run_name:
        return run_name.replace(os.sep, "_").replace("/", "_").replace(" ", "_")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{fallback_prefix}_{timestamp}"


def _tensor_from_rows(rows: list[dict], feature_names: list[str]) -> torch.Tensor:
    """Convert list of feature dictionaries to PyTorch tensor.

    Extracts values for specified features from each row dictionary and
    stacks into a 2D tensor. Used to prepare dataset samples for SHAP.

    Args:
        rows: List of dictionaries, each with feature_name -> value mappings.
        feature_names: List of feature names to extract (in order).

    Returns:
        2D float32 torch tensor of shape (len(rows), len(feature_names)).
    """
    return torch.tensor(
        [[row[name] for name in feature_names] for row in rows], dtype=torch.float32
    )


def build_server_side_shap_context(
    dataset_name: str,
    run_name: str | None,
    output_root: str,
    background_samples: int,
    explanation_samples: int,
) -> ShapRuntimeContext:
    """Initialize SHAP runtime context from server-side centralized dataset.

    Loads the federated learning dataset on the server, extracts background
    and explanation samples, and creates a ShapRuntimeContext for reuse across
    federated rounds. This avoids redundant dataset loading in each round.

    Args:
        dataset_name: Name of the dataset to load (e.g., 'adult-income-census').
        run_name: Name of the federated run (will be sanitized). If None, generated.
        output_root: Root directory for saving SHAP outputs.
        background_samples: Number of background samples for SHAP explainer.
        explanation_samples: Number of explanation samples to use.

    Returns:
        ShapRuntimeContext with preloaded data and metadata ready for round-based
        SHAP computation.

    Note:
        Creates the output directory structure if it does not exist.
        Prints information about loaded samples to stdout.
    """

    dataset_config = get_dataset_config(dataset_name)
    # Use the centralized test split as the server trust dataset for SHAP
    # background and explanation samples.
    _, test_loader, _ = load_centralized_dataset(
        batch_size=max(background_samples, explanation_samples),
        dataset_name=dataset_name,
    )

    train_dataset = test_loader.dataset
    feature_names = [
        col for col in train_dataset.column_names if col != dataset_config.label_col
    ]
    sample_count = min(len(train_dataset), max(background_samples, explanation_samples))
    selected_rows = [train_dataset[i] for i in range(sample_count)]
    explanation_data = _tensor_from_rows(selected_rows, feature_names)
    background_data = explanation_data[:background_samples].clone().detach()
    explanation_data = explanation_data[:explanation_samples].clone().detach()

    resolved_run_name = _sanitize_run_name(
        run_name, fallback_prefix=f"local_{dataset_name}"
    )
    build_run_output_dirs(output_root, resolved_run_name)

    logger.info(
        "Loaded SHAP background and explanation data from the server-side dataset pipeline for dataset '%s'",
        dataset_name,
    )
    logger.info(
        "SHAP background samples: %s | SHAP explanation samples: %s",
        background_data.shape[0],
        explanation_data.shape[0],
    )

    return ShapRuntimeContext(
        run_name=resolved_run_name,
        dataset_name=dataset_name,
        output_root=output_root,
        background_data=background_data,
        explanation_data=explanation_data,
        feature_names=feature_names,
        background_samples=background_data.shape[0],
        explanation_samples=explanation_data.shape[0],
        dataset_source="server-side dataset pipeline",
    )


def _infer_partition_id(model_name: str) -> str:
    if model_name.startswith("client_"):
        return model_name.split("client_", 1)[1].split("_", 1)[0]
    return model_name


def _display_partition_id(partition_id: str) -> str:
    if partition_id.isdigit():
        return str(int(partition_id) + 1)
    return partition_id


def _infer_malicious_flag(partition_id: int, attack_config, num_clients: int) -> bool:
    """Return whether the given partition_id is malicious.

    Detection metrics should be computed irrespective of whether attacks
    are enabled. If no attack_config is provided or attacks are disabled,
    treat clients as clean (False) so that detection confusion metrics
    (TP/FP/FN/TN) can still be computed.
    """
    # Default: no attack => not malicious
    if attack_config is None or not getattr(attack_config, "enabled", False):
        return False
    if getattr(attack_config, "attack_type", "none") == "none":
        return False

    from src.attacks import should_be_malicious

    return bool(
        should_be_malicious(
            partition_id,
            num_clients,
            attack_config.malicious_ratio,
            attack_config.seed,
        )
    )


def compute_and_save_round_shap(
    models_dict: Dict[str, torch.nn.Module],
    runtime: ShapRuntimeContext,
    dataset_name: str,
    round_number: int,
    attack_config=None,
    num_clients: int = 0,
    return_values: bool = False,
) -> (
    list[ShapArtifactMetadata]
    | tuple[list[ShapArtifactMetadata], Dict[str, np.ndarray]]
):
    """Compute SHAP values for all client models in a federated round and persist to disk.

    Iterates over client models, computes SHAP matrices, infers malicious status,
    and saves both values and metadata. Creates nested directory structure:
    round_<r>/client_<k>/shap_values.npz and round_<r>/client_<k>/metadata.json

    Args:
        models_dict: Dictionary mapping client identifiers to PyTorch models.
        runtime: ShapRuntimeContext with preloaded data and configuration.
        dataset_name: Name of the dataset (for metadata).
        round_number: Current federated learning round number.
        attack_config: AttackConfig object to determine if clients are malicious.
        num_clients: Total number of clients in the federated setup.

    Returns:
        List of ShapArtifactMetadata objects created during this round.

    Saves:
        - <output_root>/<run_name>/round_<r>/client_<k>/shap_values.npz
        - <output_root>/<run_name>/round_<r>/client_<k>/metadata.json
        For each client in the round.

    Note:
        Prints progress information for each client to stdout.
        Malicious status is inferred from attack config if available.
    """

    metadata_items: list[ShapArtifactMetadata] = []
    shap_values_by_client: Dict[str, np.ndarray] = {}
    for model_name, model in sorted(models_dict.items(), key=lambda item: str(item[0])):
        partition_id = _infer_partition_id(model_name)
        display_partition_id = _display_partition_id(partition_id)
        malicious = None
        try:
            malicious = _infer_malicious_flag(
                int(partition_id), attack_config, num_clients
            )
        except Exception:
            malicious = None

        logger.info(
            "Computing SHAP for partition %s (malicious=%s, round=%s)",
            display_partition_id,
            malicious,
            round_number,
        )
        shap_values = compute_shap_matrix(
            model,
            background_data=runtime.background_data,
            explanation_data=runtime.explanation_data,
            background_samples=runtime.background_samples,
        )
        shap_values_by_client[model_name] = (
            shap_values.mean(axis=0).reshape(-1)
            if shap_values.ndim > 1
            else shap_values.reshape(-1)
        )
        metadata = ShapArtifactMetadata(
            run_name=runtime.run_name,
            dataset_name=dataset_name,
            partition_id=str(partition_id),
            malicious=malicious,
            round_number=round_number,
            dataset_source=runtime.dataset_source,
            background_samples=runtime.background_samples,
            explanation_samples=runtime.explanation_samples,
            client_identifier=model_name,
            wandb_run=True,
        )
        save_shap_artifact(
            shap_values=shap_values,
            explanation_data=runtime.explanation_data,
            feature_names=runtime.feature_names,
            metadata=metadata,
            output_root=runtime.output_root,
        )
        metadata_items.append(metadata)

    if return_values:
        return metadata_items, shap_values_by_client
    return metadata_items


def generate_plots_for_run(
    run_name: str,
    output_root: str,
    source: str = "local",
    wandb_run_path: str | None = None,
) -> list[dict[str, str]]:
    """Regenerate plots from saved SHAP outputs only."""

    if source == "wandb":
        if wandb_run_path is None:
            raise ValueError("wandb_run_path is required when source='wandb'")
        artifacts = load_wandb_shap_artifacts(
            wandb_run_path, download_dir=os.path.join(output_root, "_wandb_downloads")
        )
    else:
        run_dir = os.path.join(output_root, run_name)
        artifacts = load_local_shap_artifacts(run_dir)

    plot_paths: list[dict[str, str]] = []
    for artifact in artifacts:
        plot_paths.append(
            generate_shap_plots(
                shap_values=artifact.values,
                feature_names=artifact.feature_names,
                metadata=artifact.metadata,
                output_root=output_root,
                explanation_data=artifact.explanation_data,
            )
        )
    return plot_paths
