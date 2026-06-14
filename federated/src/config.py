"""Configuration for src federated learning."""

from dataclasses import dataclass, field
from typing import Literal, Dict, Any


@dataclass
class DatasetConfig:
    """Configuration for a dataset used in federated learning.

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
    model_config: Dict[str, Any] = field(default_factory=dict)  # Model-specific params


# Dataset configurations
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
        - "adult-income-census": Binary classification task (predicting income level)
        - "bike-sharing": Regression task (predicting bike rental count)

    Args:
        name: Name of the dataset (must match a key in DATASETS)

    Returns:
        DatasetConfig object with all configuration for the dataset

    Raises:
        ValueError: If dataset name is not in available DATASETS

    Examples:
        >>> config = get_dataset_config("adult-income-census")
        >>> config.task
        'classification'
        >>> config.input_dim
        30
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
    return DATASETS[name]


@dataclass
class AttackConfig:
    """Configuration for federated attacks on selected malicious clients.

    Attributes:
        enabled: Whether to enable attacks (False = clean FL)
        attack_type: Type of attack - "none", "label_flip", "distribution_shift", "history", "mpaf"
        malicious_ratio: Fraction of clients [0, 1] that are malicious
        poison_fraction: Fraction of labels/features [0, 1] to poison per malicious client
        victim_label: (Label flip) Source label to flip from
        target_label: (Label flip) Target label to flip to
        seed: Random seed for deterministic malicious client selection
        shift_mechanism: (Distribution shift) Type of shift - "additive", "multiplicative", "noise_gaussian", "noise_uniform"
        shift_param: (Distribution shift) Magnitude of shift/noise (delta value or std)
        selection_strategy: (Distribution shift) Strategy for victim selection - "random" or "quantile_range"
        victim_quantile_min: (Distribution shift) Lower quantile bound [0, 1] for victim selection
        victim_quantile_max: (Distribution shift) Upper quantile bound [0, 1] for victim selection
        model_attack_lambda: (History/MPAF) Scaling factor used to craft poisoned model updates
    """

    enabled: bool = False
    attack_type: str = (
        "none"  # "none", "label_flip", "distribution_shift", "history", "mpaf"
    )
    malicious_ratio: float = 0.2  # Fraction [0, 1]
    poison_fraction: float = 0.1  # Fraction [0, 1]
    victim_label: int = 0
    target_label: int = 1
    seed: int = 42
    # Distribution shift specific parameters
    shift_mechanism: str = (
        "additive"  # "additive", "multiplicative", "noise_gaussian", "noise_uniform"
    )
    shift_param: float = -50.0  # delta value or noise std
    selection_strategy: str = "random"  # "random" or "quantile_range"
    victim_quantile_min: float = 0.0  # Lower quantile bound
    victim_quantile_max: float = 1.0  # Upper quantile bound
    model_attack_lambda: float = 1.0

    def __post_init__(self):
        valid_attack_types = {
            "none",
            "label_flip",
            "distribution_shift",
            "history",
            "mpaf",
        }
        if self.attack_type not in valid_attack_types:
            raise ValueError(
                f"Unknown attack_type: {self.attack_type}. Available: {sorted(valid_attack_types)}"
            )
        if not 0.0 <= self.malicious_ratio <= 1.0:
            raise ValueError("malicious_ratio must be in [0, 1]")
        if not 0.0 <= self.poison_fraction <= 1.0:
            raise ValueError("poison_fraction must be in [0, 1]")
        if self.selection_strategy not in {"random", "quantile_range"}:
            raise ValueError("selection_strategy must be 'random' or 'quantile_range'")
        if not 0.0 <= self.victim_quantile_min <= self.victim_quantile_max <= 1.0:
            raise ValueError("quantile bounds must satisfy 0 <= min <= max <= 1")
        if self.model_attack_lambda < 0.0:
            raise ValueError("model_attack_lambda must be >= 0")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AttackConfig":
        """Create AttackConfig from dictionary (e.g., from server config).

        Args:
            config_dict: Dictionary with attack config keys

        Returns:
            AttackConfig object with values from dict (defaults for missing keys)
        """
        return cls(
            enabled=config_dict.get("attack_enabled", False),
            attack_type=config_dict.get("attack_type", "none"),
            malicious_ratio=config_dict.get("malicious_ratio", 0.2),
            poison_fraction=config_dict.get("poison_fraction", 0.1),
            victim_label=config_dict.get("victim_label", 0),
            target_label=config_dict.get("target_label", 1),
            seed=config_dict.get("attack_seed", 42),
            shift_mechanism=config_dict.get("shift_mechanism", "additive"),
            shift_param=config_dict.get("shift_param", -50.0),
            selection_strategy=config_dict.get("selection_strategy", "random"),
            victim_quantile_min=config_dict.get("victim_quantile_min", 0.0),
            victim_quantile_max=config_dict.get("victim_quantile_max", 1.0),
            model_attack_lambda=config_dict.get("model_attack_lambda", 1.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert AttackConfig to dictionary for sending to clients.

        Returns:
            Dictionary with client-compatible keys (underscores for config keys)
        """
        return {
            "attack_enabled": self.enabled,
            "attack_type": self.attack_type,
            "malicious_ratio": self.malicious_ratio,
            "poison_fraction": self.poison_fraction,
            "victim_label": self.victim_label,
            "target_label": self.target_label,
            "attack_seed": self.seed,
            "shift_mechanism": self.shift_mechanism,
            "shift_param": self.shift_param,
            "selection_strategy": self.selection_strategy,
            "victim_quantile_min": self.victim_quantile_min,
            "victim_quantile_max": self.victim_quantile_max,
            "model_attack_lambda": self.model_attack_lambda,
        }


@dataclass
class DetectionConfig:
    """Configuration for server-side client detection and exclusion."""

    enabled: bool = False
    use_hybrid: bool = True
    mode: str = "mad-shapcosim"
    alpha: float = 0.7
    mad_multiplier: float = 3.0
    # Keep iqr_multiplier for backward compatibility (not used)
    iqr_multiplier: float = 1.5
    min_clients: int = 3
    # G-ShapCosim removed; keep MAD/hybrid params only

    def __post_init__(self):
        valid_modes = {"mad-shapcosim", "iqr", "hybrid"}
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {sorted(valid_modes)}")
        if self.alpha < 0.0 or self.alpha > 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.mad_multiplier <= 0.0:
            raise ValueError("mad_multiplier must be > 0")
        # iqr_multiplier removed
        if self.min_clients < 1:
            raise ValueError("min_clients must be >= 1")
        # G-ShapCosim params removed; no additional checks

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DetectionConfig":
        return cls(
            enabled=config_dict.get("detection_enabled", False),
            use_hybrid=config_dict.get("detection_use_hybrid", True),
            mode=config_dict.get("detection_mode", "mad-shapcosim"),
            alpha=config_dict.get("detection_alpha", 0.7),
            mad_multiplier=config_dict.get("detection_mad_multiplier", 3.0),
            iqr_multiplier=config_dict.get("detection_iqr_multiplier", 1.5),
            min_clients=config_dict.get("detection_min_clients", 3),
            # G-ShapCosim params removed from config
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detection_enabled": self.enabled,
            "detection_use_hybrid": self.use_hybrid,
            "detection_mode": self.mode,
            "detection_alpha": self.alpha,
            "detection_mad_multiplier": self.mad_multiplier,
            "detection_iqr_multiplier": self.iqr_multiplier,
            "detection_min_clients": self.min_clients,
            # G-ShapCosim params removed from config
        }
