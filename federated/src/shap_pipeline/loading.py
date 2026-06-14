"""Loading utilities for SHAP artifacts from local filesystem and Weights & Biases.

Supports loading SHAP artifacts from:
- Local filesystem (with backward compatibility for both nested and flat layouts)
- Weights & Biases (W&B) run artifacts

The module includes automatic detection of directory structure (nested round/client
vs. flat shap_values/) to handle legacy artifacts seamlessly.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .save import ShapArtifactMetadata

try:  # pragma: no cover - optional dependency path
    import wandb
except Exception:  # pragma: no cover - optional dependency path
    wandb = None


@dataclass(frozen=True)
class LoadedShapArtifact:
    """Immutable SHAP artifact loaded from disk or Weights & Biases.

    Contains all data and metadata necessary to reconstruct SHAP visualizations
    or perform downstream analysis on model explanations.

    Attributes:
        values: SHAP value matrix of shape (num_samples, num_features) as numpy array.
        explanation_data: Original feature values used in SHAP explanation.
        feature_names: List of feature names corresponding to matrix columns.
        metadata: Metadata describing the computation context and dataset.
        values_path: Local filesystem path or W&B artifact path to the SHAP values file.
        metadata_path: Path to the associated metadata file (JSON).
    """

    values: np.ndarray
    explanation_data: np.ndarray
    feature_names: list[str]
    metadata: ShapArtifactMetadata
    values_path: str
    metadata_path: str


def load_local_shap_artifact(values_path: str) -> LoadedShapArtifact:
    """Load a single SHAP artifact from local filesystem.

    Loads SHAP values, explanation data, feature names, and metadata from
    an NPZ file and its associated metadata JSON file.

    Args:
        values_path: Path to the shap_values.npz file.

    Returns:
        LoadedShapArtifact with all data and metadata.

    Raises:
        FileNotFoundError: If metadata file is not found.
        ValueError: If the file format is not NPZ.

    Note:
        Looks for metadata.json in multiple locations (same dir, parent dir,
        or metadata/ subdirectory) before checking embedded NPZ metadata.
    """

    values_file = Path(values_path)
    if values_file.suffix != ".npz":
        raise ValueError(f"Unsupported SHAP artifact format: {values_path}")

    payload = np.load(values_file, allow_pickle=True)

    # Locate metadata: prefer filesystem JSON, then embedded JSON in NPZ
    metadata_path = values_file.with_name("metadata.json")
    if not metadata_path.exists():
        metadata_path = values_file.with_suffix(".json")
    if not metadata_path.exists():
        metadata_path = (
            values_file.parent.parent / "metadata" / f"{values_file.stem}.json"
        )
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = ShapArtifactMetadata(**json.load(handle))
    else:
        if "metadata_json" not in payload.files:
            raise FileNotFoundError(
                f"Missing metadata for SHAP artifact: {values_file}"
            )
        metadata = ShapArtifactMetadata(**json.loads(str(payload["metadata_json"])))

    # Required array: values
    if "values" not in payload.files:
        raise ValueError(f"SHAP NPZ missing 'values' array: {values_file}")
    values = np.asarray(payload["values"])
    if values.size == 0:
        raise ValueError(f"SHAP 'values' array is empty: {values_file}")
    if values.ndim not in (1, 2):
        raise ValueError(
            f"SHAP 'values' must be 1-D or 2-D array: {values_file} got ndim={values.ndim}"
        )

    # Optional explanation_data
    explanation_data = None
    if "explanation_data" in payload.files:
        explanation_data = np.asarray(payload["explanation_data"])
        if explanation_data.size == 0:
            explanation_data = None

    # Feature names: fallback when missing or invalid
    feature_names = None
    if "feature_names" in payload.files:
        try:
            arr = payload["feature_names"]
            # convert object arrays safely
            names = list(arr.tolist()) if hasattr(arr, "tolist") else list(arr)
            feature_names = [str(item) for item in names if item is not None]
        except Exception:
            feature_names = None

    # Infer num_features from values
    num_features = values.shape[0] if values.ndim == 1 else values.shape[1]
    if not feature_names or len(feature_names) != num_features:
        feature_names = [f"f{i + 1}" for i in range(num_features)]

    # Validate explanation_data shape if present
    if explanation_data is not None:
        if explanation_data.ndim == 1:
            if explanation_data.shape[0] != num_features:
                raise ValueError(
                    f"explanation_data length {explanation_data.shape[0]} does not match num_features {num_features} in {values_file}"
                )
        elif explanation_data.ndim == 2:
            if explanation_data.shape[1] != num_features:
                raise ValueError(
                    f"explanation_data second-dim {explanation_data.shape[1]} does not match num_features {num_features} in {values_file}"
                )
        else:
            raise ValueError(
                f"explanation_data must be 1-D or 2-D array in {values_file}, got ndim={explanation_data.ndim}"
            )

    return LoadedShapArtifact(
        values=values,
        explanation_data=explanation_data if explanation_data is not None else values,
        feature_names=feature_names,
        metadata=metadata,
        values_path=str(values_file),
        metadata_path=str(metadata_path) if metadata_path.exists() else "",
    )


def load_local_shap_artifacts(run_dir: str) -> list[LoadedShapArtifact]:
    """Load all SHAP artifacts under a run directory.

    Automatically detects nested (round_*/client_*/shap_values.npz) vs.
    flat (shap_values/*.npz) directory layout and loads accordingly.
    Prefers nested layout if both exist (backward compatible).

    Args:
        run_dir: Path to the run directory containing SHAP artifacts.

    Returns:
        List of LoadedShapArtifact objects, sorted by path.

    Note:
        Returns empty list if no .npz files are found under run_dir.
    """

    run_path = Path(run_dir)
    # Recursively discover any shap_values.npz under run_dir for flexibility
    values_paths = sorted(run_path.rglob("shap_values.npz"))
    artifacts: list[LoadedShapArtifact] = []
    for npz_path in values_paths:
        artifacts.append(load_local_shap_artifact(str(npz_path)))
    return artifacts


def _download_wandb_file(run, remote_path: str, download_dir: str) -> str:
    """Download a file from Weights & Biases run to local directory.

    Args:
        run: Weights & Biases run object (from wandb.Api().run()).
        remote_path: Path to file within W&B run (relative).
        download_dir: Local directory where file will be downloaded.

    Returns:
        Local filesystem path to the downloaded file.
    """
    file_obj = run.file(remote_path)
    return file_obj.download(root=download_dir, replace=True).name


def load_wandb_shap_artifacts(
    run_path: str, download_dir: str
) -> list[LoadedShapArtifact]:
    """Load SHAP artifacts from a Weights & Biases run.

    Downloads SHAP value files from a W&B run path (entity/project/run_id) and
    loads them as LoadedShapArtifact objects. Prefers nested layout (round_*/client_*)
    over flat layout (shap_values/) if both exist.

    Args:
        run_path: W&B run path formatted as 'entity/project/run_id'.
        download_dir: Local directory where files will be downloaded.

    Returns:
        List of LoadedShapArtifact objects loaded from W&B.

    Raises:
        ImportError: If wandb package is not installed.

    Note:
        Downloads files to download_dir temporarily; they remain after function returns.
    """

    if wandb is None:
        raise ImportError("wandb is required to load SHAP outputs from W&B")

    api = wandb.Api()
    run = api.run(run_path)
    remote_names = [
        file_obj.name for file_obj in run.files() if file_obj.name.endswith(".npz")
    ]
    # Prefer explicit shap_values files when available
    shap_remote = [name for name in remote_names if name.endswith("shap_values.npz")]
    selected_names = shap_remote if shap_remote else remote_names
    artifacts: list[LoadedShapArtifact] = []
    for remote_name in selected_names:
        local_npz = _download_wandb_file(run, remote_name, download_dir)
        meta_remote = remote_name.replace("shap_values.npz", "metadata.json")
        if meta_remote == remote_name:
            meta_remote = remote_name.replace("shap_values/", "metadata/").replace(
                ".npz", ".json"
            )
        try:
            _download_wandb_file(run, meta_remote, download_dir)
        except Exception:
            pass
        artifacts.append(load_local_shap_artifact(local_npz))
    return artifacts
