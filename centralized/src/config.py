"""
Dataset configuration for centralized learning experiments.

Unified configuration that mirrors the federated setup, ensuring all datasets
use the same task-model paradigm across centralized and federated settings.
"""

from dataclasses import dataclass, field
from typing import Literal, Dict, Any


@dataclass
class DatasetConfig:
    """Configuration for a dataset used in experiments.

    Attributes:
        name: Human-readable dataset name
        task: Learning task type - "classification" or "regression"
        hf_repo: Hugging Face dataset repository ID
        label_col: Name of the target/label column in the dataset
        input_dim: Number of input features
        model_config: Dictionary of model-specific parameters (type, hidden_dims, dropout, etc.)
    """

    name: str
    task: Literal["regression", "classification"]
    hf_repo: str
    label_col: str
    input_dim: int
    model_config: Dict[str, Any] = field(default_factory=dict)


# Dataset configurations - identical to federated setup for consistency
DATASETS = {
    "adult-income-census": DatasetConfig(
        name="adult-income-census",
        task="classification",
        hf_repo="b-fatma/adult-income-census-federated",
        label_col="income",
        input_dim=30,
        model_config={
            "type": "mlp",
        },
    ),
    "bike-sharing": DatasetConfig(
        name="bike-sharing",
        task="regression",
        hf_repo="b-fatma/bike-sharing-federated",
        label_col="cnt_log",
        input_dim=12,
        model_config={
            "type": "mlp",
        },
    ),
    "bank-marketing": DatasetConfig(
        name="bank-marketing",
        task="classification",
        hf_repo="narimanee/bank-marketing-federated",
        label_col="y",
        input_dim=20,
        model_config={
            "type": "mlp",
        },
    ),
    "housing-prices": DatasetConfig(
        name="housing-prices",
        task="regression",
        hf_repo="narimanee/Housing-prices-federated",
        label_col="median_house_value",
        input_dim=18,
        model_config={
            "type": "mlp",
        },
    ),
}


def get_dataset_config(name: str) -> DatasetConfig:
    """Get configuration for a dataset by name.

    Retrieves pre-configured dataset settings including task type, Hugging Face
    repository, label column name, input dimension, and model-specific parameters.

    Available datasets:
        - "adult-income-census": Binary classification (predicting income level)
        - "bike-sharing": Regression (predicting bike rental count)
        - "bank-marketing": Binary classification (predicting subscription)
        - "housing-prices": Regression (predicting house value)

    Args:
        name: Name of the dataset (must match a key in DATASETS)

    Returns:
        DatasetConfig object with all configuration for the dataset

    Raises:
        KeyError: If dataset name not found in DATASETS
    """
    if name not in DATASETS:
        available = ", ".join(DATASETS.keys())
        raise KeyError(f"Dataset '{name}' not found. Available datasets: {available}")
    return DATASETS[name]
