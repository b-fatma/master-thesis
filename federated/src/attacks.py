"""
Attack dispatcher for federated learning.

Determines which clients are malicious and applies appropriate attacks
to their training data. Attacks are imported at module-level for efficiency.
"""

import logging
import sys
import numpy as np
import torch
from collections import OrderedDict
from typing import Tuple, Callable, Iterator
from torch.utils.data import DataLoader
from pathlib import Path
from copy import deepcopy
from enum import Enum

from .config import AttackConfig

logger = logging.getLogger(__name__)

# Add parent path once at module load time (not per-function call)
_module_path = Path(__file__).parent.parent.parent
if str(_module_path) not in sys.path:
    sys.path.insert(0, str(_module_path))


# ═══════════════════════════════════════════════════════════════
# Distribution Shift Attack (DUPLICATED FROM: src/attacks/distribution_shift.py)
# Keep in sync with reference implementation
# ═══════════════════════════════════════════════════════════════


class SelectionStrategy(Enum):
    """Strategies for selecting which samples to poison."""

    RANDOM = "random"
    QUANTILE_RANGE = "quantile_range"


class LabelFlipAttack:
    """Simple label-flip attack for binary/multiclass labels."""

    def __init__(
        self,
        victim_label: int,
        target_label: int,
        poison_frac: float = 0.1,
        seed: int = 42,
    ):
        assert 0.0 < poison_frac <= 1.0, "poison_frac must be in (0, 1]"
        self.victim_label = victim_label
        self.target_label = target_label
        self.poison_frac = poison_frac
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def poison_labels(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Flip a random subset of victim labels to the target label."""
        original_shape = y.shape
        if y.ndim == 2:
            y = y.flatten()

        y_poisoned = y.copy()
        poison_mask = np.zeros(len(y), dtype=bool)
        victim_idxs = np.where(y == self.victim_label)[0]

        if len(victim_idxs) == 0:
            return y_poisoned.reshape(original_shape), poison_mask

        n_to_poison = max(1, int(len(victim_idxs) * self.poison_frac))
        poison_idxs = self._rng.choice(victim_idxs, size=n_to_poison, replace=False)
        y_poisoned[poison_idxs] = self.target_label
        poison_mask[poison_idxs] = True

        return y_poisoned.reshape(original_shape), poison_mask


class DistributionShiftAttack:
    """
    Distribution shift attack on regression targets.

    Systematically corrupts regression targets by adding Gaussian or Uniform noise
    to degrade model performance on continuous regression tasks.

    This attack is:
      - **Deterministic**: Reproducible with seed
      - **Modular**: Works with numpy arrays, datasets, and dataloaders
      - **Flexible**: Supports multiple noise and selection strategies
      - **Measurable**: Returns mask of modified samples for analysis

    Attributes:
        shift_mechanism (str): Type of noise injection ('noise_gaussian' or 'noise_uniform')
        shift_param (float): Noise magnitude
        selection_strategy (str): How to select victims ('random' or 'quantile_range')
        poison_frac (float): Fraction of selected candidates to poison in (0, 1]
        victim_quantile_min (float): Lower quantile bound [0, 1] for quantile-based selection
        victim_quantile_max (float): Upper quantile bound [0, 1] for quantile-based selection
        seed (int): Random seed for reproducibility
    """

    def __init__(
        self,
        shift_mechanism: str = "noise_gaussian",
        shift_param: float = 5.0,
        selection_strategy: str = "random",
        poison_frac: float = 0.1,
        victim_quantile_min: float = 0.0,
        victim_quantile_max: float = 1.0,
        seed: int = 42,
    ):
        """
        Initialize distribution shift attack with noise injection.

        Args:
            shift_mechanism (str): 'noise_gaussian' or 'noise_uniform'. Default: 'noise_gaussian'
            shift_param (float): Noise magnitude (std for Gaussian, bound for Uniform). Default: 5.0
            selection_strategy (str): 'random' or 'quantile_range'. Default: 'random'
            poison_frac (float): Fraction to poison in (0, 1]. Default: 0.1
            victim_quantile_min (float): Lower quantile [0, 1]. Default: 0.0 (inclusive)
            victim_quantile_max (float): Upper quantile [0, 1]. Default: 1.0 (inclusive)
            seed (int): Random seed. Default: 42

        Raises:
            AssertionError: if parameters are invalid
        """
        assert shift_mechanism in ["noise_gaussian", "noise_uniform"], (
            f"Invalid shift_mechanism: {shift_mechanism}. Must be 'noise_gaussian' or 'noise_uniform'"
        )
        assert selection_strategy in ["random", "quantile_range"], (
            f"Invalid selection_strategy: {selection_strategy}"
        )
        assert 0.0 < poison_frac <= 1.0, "poison_frac must be in (0, 1]"
        assert 0.0 <= victim_quantile_min <= victim_quantile_max <= 1.0, (
            "quantile bounds must satisfy: 0 <= min <= max <= 1"
        )

        self.shift_mechanism = shift_mechanism
        self.shift_param = shift_param
        self.selection_strategy = selection_strategy
        self.poison_frac = poison_frac
        self.victim_quantile_min = victim_quantile_min
        self.victim_quantile_max = victim_quantile_max
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def _select_victims(self, y: np.ndarray) -> np.ndarray:
        """Select indices of samples to potentially poison based on strategy."""
        if self.selection_strategy == "random":
            return np.arange(len(y))
        elif self.selection_strategy == "quantile_range":
            q_min = np.quantile(y, self.victim_quantile_min)
            q_max = np.quantile(y, self.victim_quantile_max)
            return np.where((y >= q_min) & (y <= q_max))[0]

    def _apply_shift(self, y_candidates: np.ndarray) -> np.ndarray:
        """Apply noise injection to candidate target values."""
        if self.shift_mechanism == "noise_gaussian":
            noise = self._rng.normal(
                loc=0.0, scale=self.shift_param, size=len(y_candidates)
            )
            return y_candidates + noise
        elif self.shift_mechanism == "noise_uniform":
            noise = self._rng.uniform(
                low=-self.shift_param, high=self.shift_param, size=len(y_candidates)
            )
            return y_candidates + noise
        else:
            raise ValueError(f"Unknown shift_mechanism: {self.shift_mechanism}")

    def poison_labels(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Corrupt a subset of regression targets."""
        # Handle 2D arrays (e.g., from PyTorch DataLoaders with shape (batch, 1))
        original_shape = y.shape
        if y.ndim == 2:
            y = y.flatten()

        y_poisoned = y.copy()
        poison_mask = np.zeros(len(y), dtype=bool)

        victim_idxs = self._select_victims(y)

        if len(victim_idxs) == 0:
            # Restore original shape before returning
            y_poisoned = y_poisoned.reshape(original_shape)
            return y_poisoned, poison_mask

        n_to_poison = max(1, int(len(victim_idxs) * self.poison_frac))
        poison_idxs = self._rng.choice(victim_idxs, size=n_to_poison, replace=False)

        y_poisoned[poison_idxs] = self._apply_shift(y[poison_idxs])
        poison_mask[poison_idxs] = True

        # Restore original shape before returning
        y_poisoned = y_poisoned.reshape(original_shape)
        return y_poisoned, poison_mask

    def poison_dataset(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """Poison targets of a dataset."""
        y_poisoned, poison_mask = self.poison_labels(y)
        return X, y_poisoned, poison_mask

    def poison_torch_dataset(self, dataset):
        """Poison a PyTorch torchvision dataset with continuous targets."""
        dataset = deepcopy(dataset)

        if hasattr(dataset, "targets"):
            targets = np.array(dataset.targets, dtype=np.float32)
        else:
            raise ValueError(
                "Dataset has no .targets attribute. Use poison_labels() manually."
            )

        y_poisoned, poison_mask = self.poison_labels(targets)
        targets = y_poisoned

        if isinstance(dataset.targets, torch.Tensor):
            dataset.targets = torch.tensor(targets, dtype=torch.float32)
        else:
            dataset.targets = targets.tolist()

        return dataset, poison_mask

    def poison_dataloader(
        self, dataloader, convert_labels: Callable = None
    ) -> Iterator:
        """Poison batches from a PyTorch DataLoader on-the-fly (for FL)."""
        for X_batch, y_batch in dataloader:
            if convert_labels is None:
                convert_labels = lambda y: (
                    y.cpu().numpy() if isinstance(y, torch.Tensor) else y
                )

            y_np = convert_labels(y_batch).astype(np.float32)
            y_poisoned_np, _ = self.poison_labels(y_np)

            if isinstance(y_batch, torch.Tensor):
                y_poisoned = torch.from_numpy(y_poisoned_np).to(y_batch.device)
            else:
                y_poisoned = y_poisoned_np

            yield X_batch, y_poisoned


class HistoryModelPoisonAttack:
    """History attack.

    Inspired by https://arxiv.org/pdf/2203.08669.

    The attack crafts the uploaded model update as:
        g = lambda * (local - reference)
    For weight upload protocols, we map this to:
        poisoned = reference + g
    """

    def __init__(self, attack_lambda: float):
        self.attack_lambda = attack_lambda

    def apply(
        self,
        local_state: OrderedDict,
        reference_state: OrderedDict,
    ) -> OrderedDict:
        poisoned_state = OrderedDict()
        for name, local_param in local_state.items():
            ref_param = reference_state[name].to(local_param.device)
            poisoned_state[name] = (
                ref_param + (local_param - ref_param) * self.attack_lambda
            )
        return poisoned_state


class MPAFModelPoisonAttack:
    """MPAF attack.

    Inspired by https://arxiv.org/pdf/2203.08669.

    The attack crafts the uploaded model update as:
        g = lambda * (base - local)
    For weight upload protocols, we map this to:
        poisoned = reference + g
    """

    def __init__(self, attack_lambda: float, seed: int, partition_id: int):
        self.attack_lambda = attack_lambda
        self.seed = seed
        self.partition_id = partition_id

    def _build_base_state(self, local_state: OrderedDict) -> OrderedDict:
        base_state = OrderedDict()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + 1009 * int(self.partition_id))

        for name, local_param in local_state.items():
            if torch.is_floating_point(local_param):
                base = torch.randn(
                    local_param.shape,
                    generator=generator,
                    dtype=local_param.dtype,
                    device="cpu",
                ).to(local_param.device)
            else:
                base = local_param.detach().clone()
            base_state[name] = base
        return base_state

    def apply(
        self,
        local_state: OrderedDict,
        reference_state: OrderedDict,
    ) -> OrderedDict:
        base_state = self._build_base_state(local_state)
        poisoned_state = OrderedDict()

        for name, local_param in local_state.items():
            ref_param = reference_state[name].to(local_param.device)
            base_param = base_state[name].to(local_param.device)
            poisoned_state[name] = (
                ref_param + (base_param - local_param) * self.attack_lambda
            )

        return poisoned_state


def _is_data_poisoning_attack(attack_type: str) -> bool:
    return attack_type in {"label_flip", "distribution_shift"}


def _is_model_poisoning_attack(attack_type: str) -> bool:
    return attack_type in {"history", "mpaf"}


# ═══════════════════════════════════════════════════════════════
# Poisoned DataLoader Wrapper
# ═══════════════════════════════════════════════════════════════


class PoisonedDataLoaderWrapper:
    """Lightweight wrapper that applies poison on-the-fly without serialization overhead."""

    def __init__(self, original_loader: DataLoader, attack):
        self.original_loader = original_loader
        self.attack = attack
        self.stats = {
            "poisoned_count": 0,
            "total_count": 0,
            "victim_count": 0,
        }
        # Detect attack type
        self.attack_type = self._detect_attack_type()

    def _detect_attack_type(self) -> str:
        """Detect whether this is a label flip or distribution shift attack."""
        if hasattr(self.attack, "victim_label"):
            return "label_flip"
        elif hasattr(self.attack, "shift_mechanism"):
            return "distribution_shift"
        else:
            return "unknown"

    def __iter__(self):
        """Yield poisoned batches on-the-fly without storing generator."""
        for X_batch, y_batch in self.original_loader:
            # Track statistics before poisoning
            y_numpy = y_batch.cpu().numpy() if hasattr(y_batch, "cpu") else y_batch
            self.stats["total_count"] += len(y_numpy)

            # For label flip: count victim labels
            if self.attack_type == "label_flip":
                self.stats["victim_count"] += (
                    y_numpy == self.attack.victim_label
                ).sum()

            # Apply attack inline
            y_poisoned, poison_mask = self.attack.poison_labels(y_numpy)
            self.stats["poisoned_count"] += poison_mask.sum()

            # Convert back to original type if needed
            if hasattr(y_batch, "to"):  # torch tensor
                y_batch = torch.from_numpy(y_poisoned).to(
                    device=y_batch.device,
                    dtype=y_batch.dtype,
                )
            else:
                y_batch = y_poisoned

            yield X_batch, y_batch

    def __len__(self):
        return len(self.original_loader)

    @property
    def dataset(self):
        """Delegate dataset access to original loader."""
        return self.original_loader.dataset

    def print_poison_stats(self, partition_id: int):
        """Print poisoning statistics for this client."""
        if self.attack_type == "label_flip":
            if self.stats["total_count"] == 0:
                # Stats haven't been collected yet
                logger.info(
                    "[Client %s] MALICIOUS - Will poison %.1f%% of label %s samples -> label %s",
                    partition_id,
                    self.attack.poison_frac * 100,
                    self.attack.victim_label,
                    self.attack.target_label,
                )
            else:
                # Stats after iteration
                poison_pct = (
                    (self.stats["poisoned_count"] / self.stats["total_count"] * 100)
                    if self.stats["total_count"] > 0
                    else 0
                )
                victim_pct = (
                    (self.stats["victim_count"] / self.stats["total_count"] * 100)
                    if self.stats["total_count"] > 0
                    else 0
                )
                logger.info(
                    "[Client %s] POISONED: %s/%s (%.1f%%) labels flipped | Victim labels: %s (%.1f%%)",
                    partition_id,
                    self.stats["poisoned_count"],
                    self.stats["total_count"],
                    poison_pct,
                    self.stats["victim_count"],
                    victim_pct,
                )

        elif self.attack_type == "distribution_shift":
            if self.stats["total_count"] == 0:
                logger.info(
                    "[Client %s] MALICIOUS - Will poison %.1f%% of targets (mechanism=%s, param=%s)",
                    partition_id,
                    self.attack.poison_frac * 100,
                    self.attack.shift_mechanism,
                    self.attack.shift_param,
                )
            else:
                poison_pct = (
                    (self.stats["poisoned_count"] / self.stats["total_count"] * 100)
                    if self.stats["total_count"] > 0
                    else 0
                )
                logger.info(
                    "[Client %s] POISONED: %s/%s (%.1f%%) targets corrupted (mechanism=%s, param=%s)",
                    partition_id,
                    self.stats["poisoned_count"],
                    self.stats["total_count"],
                    poison_pct,
                    self.attack.shift_mechanism,
                    self.attack.shift_param,
                )


def should_be_malicious(
    partition_id: int,
    num_clients: int,
    malicious_ratio: float,
    seed: int = 42,
) -> bool:
    """
    Deterministically decide if a client should be malicious.

    Uses seeded random selection to reproducibly choose which clients are malicious.
    Same client is always malicious across runs with same seed.

    Args:
        partition_id: This client's unique ID (0 to num_clients-1)
        num_clients: Total number of clients in federation
        malicious_ratio: Fraction of clients [0, 1] that should be malicious
        seed: Random seed for determinism

    Returns:
        bool: True if this client should be malicious, False if clean
    """
    if malicious_ratio <= 0:
        return False
    if malicious_ratio >= 1.0:
        return True

    # Deterministically select which clients are malicious
    rng = np.random.RandomState(seed)
    num_malicious = max(1, int(np.ceil(num_clients * malicious_ratio)))
    malicious_ids = sorted(rng.choice(num_clients, size=num_malicious, replace=False))

    return partition_id in malicious_ids


def apply_data_poisoning_if_selected(
    trainloader: DataLoader,
    partition_id: int,
    attack_config: AttackConfig,
    is_malicious_client: bool,
) -> Tuple[DataLoader, bool]:
    """
    Apply data poisoning to trainloader if this client is selected as malicious.

    Applies the configured data-poisoning attack when this client is already
    determined to be malicious. If attack is disabled or client is clean,
    returns the original loader.

    Uses lightweight wrapper to avoid Ray serialization overhead with generators.

    Args:
        trainloader: PyTorch DataLoader with clean training data
        partition_id: This client's unique ID
        attack_config: AttackConfig object with attack parameters
        is_malicious_client: Deterministic maliciousness flag computed by caller

    Returns:
        Tuple of:
        - trainloader (DataLoader or PoisonedDataLoaderWrapper): Original or poisoned dataloader
        - is_malicious (bool): Whether this client is malicious

    Raises:
        ValueError: If attack_type is unknown
    """
    # Attack disabled or model-poisoning mode → keep data clean.
    if not attack_config.enabled or attack_config.attack_type == "none":
        return trainloader, False

    # If clean -> return clean.
    if not is_malicious_client:
        return trainloader, False

    # Model poisoning attacks are applied to uploaded parameters, not data.
    if _is_model_poisoning_attack(attack_config.attack_type):
        logger.info(
            "[Client %s] MALICIOUS MODE ACTIVATED: attack_type=%s (model-update poisoning), lambda=%s",
            partition_id,
            attack_config.attack_type,
            attack_config.model_attack_lambda,
        )
        return trainloader, True

    # Dispatch to data poisoning attack.
    if attack_config.attack_type == "label_flip":
        attack = LabelFlipAttack(
            victim_label=attack_config.victim_label,
            target_label=attack_config.target_label,
            poison_frac=attack_config.poison_fraction,
        )
        # Use wrapper instead of generator to avoid serialization overhead
        poisoned_loader = PoisonedDataLoaderWrapper(trainloader, attack)

        # Print poisoning configuration
        total_samples = len(poisoned_loader.dataset)
        logger.info(
            "[Client %s] MALICIOUS MODE ACTIVATED: dataset_size=%s, attack_type=%s, victim_label=%s, target_label=%s, poison_fraction=%.1f%%",
            partition_id,
            total_samples,
            attack_config.attack_type,
            attack_config.victim_label,
            attack_config.target_label,
            attack_config.poison_fraction * 100,
        )

        return poisoned_loader, True

    elif attack_config.attack_type == "distribution_shift":
        # Use locally-defined DistributionShiftAttack (no cross-folder import needed)
        attack = DistributionShiftAttack(
            shift_mechanism=attack_config.shift_mechanism,
            shift_param=attack_config.shift_param,
            selection_strategy=attack_config.selection_strategy,
            poison_frac=attack_config.poison_fraction,
            victim_quantile_min=attack_config.victim_quantile_min,
            victim_quantile_max=attack_config.victim_quantile_max,
            seed=attack_config.seed,
        )
        # Use wrapper instead of generator to avoid serialization overhead
        poisoned_loader = PoisonedDataLoaderWrapper(trainloader, attack)

        # Print poisoning configuration
        total_samples = len(poisoned_loader.dataset)
        logger.info(
            "[Client %s] MALICIOUS MODE ACTIVATED: dataset_size=%s, attack_type=%s, shift_mechanism=%s, shift_param=%s, selection_strategy=%s, victim_quantile_range=[%s, %s], poison_fraction=%.1f%%",
            partition_id,
            total_samples,
            attack_config.attack_type,
            attack_config.shift_mechanism,
            attack_config.shift_param,
            attack_config.selection_strategy,
            attack_config.victim_quantile_min,
            attack_config.victim_quantile_max,
            attack_config.poison_fraction * 100,
        )

        return poisoned_loader, True

    else:
        raise ValueError(
            f"Unknown attack type: {attack_config.attack_type}. "
            f"Available: 'none', 'label_flip', 'distribution_shift', 'history', 'mpaf'"
        )


def apply_model_poisoning_if_selected(
    local_state: OrderedDict,
    reference_state: OrderedDict,
    partition_id: int,
    attack_config: AttackConfig,
    is_malicious_client: bool,
) -> Tuple[OrderedDict, bool]:
    """Poison the uploaded model state if selected and attack type is model-based."""
    if not attack_config.enabled or not _is_model_poisoning_attack(
        attack_config.attack_type
    ):
        return local_state, False

    if not is_malicious_client:
        return local_state, False

    if attack_config.attack_type == "history":
        attack = HistoryModelPoisonAttack(
            attack_lambda=attack_config.model_attack_lambda
        )
    elif attack_config.attack_type == "mpaf":
        attack = MPAFModelPoisonAttack(
            attack_lambda=attack_config.model_attack_lambda,
            seed=attack_config.seed,
            partition_id=partition_id,
        )
    else:
        raise ValueError(
            f"Unsupported model poisoning attack: {attack_config.attack_type}. "
            f"Available model attacks: 'history', 'mpaf'"
        )

    poisoned_state = attack.apply(local_state, reference_state)
    logger.info(
        "[Client %s] MODEL UPDATE POISONED (%s)",
        partition_id,
        attack_config.attack_type,
    )
    return poisoned_state, True
