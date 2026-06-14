"""
Attacks module: Shared poisoning attack implementations for centralized and FL.

Available Attacks:
  - LabelFlipAttack: Flip labels of victim class to target class
  - LabelFlipFLClient: Wrapper for federated learning malicious clients

Federated Utilities:
  - should_be_malicious(): Deterministically select malicious clients
  - poison_trainloader_if_malicious(): Dispatch attack to poisoned dataloader
"""

import numpy as np
import torch
from typing import Tuple, TYPE_CHECKING, Union
from torch.utils.data import DataLoader

from src.attacks.label_flip import LabelFlipAttack, LabelFlipFLClient

if TYPE_CHECKING:
    from pytorchexample.config import AttackConfig

__all__ = [
    "LabelFlipAttack",
    "LabelFlipFLClient",
    "PoisonedDataLoaderWrapper",
    "should_be_malicious",
    "poison_trainloader_if_malicious",
]


class PoisonedDataLoaderWrapper:
    """Apply poisoning on-the-fly while preserving DataLoader-like interface."""

    def __init__(self, original_loader: DataLoader, attack: LabelFlipAttack):
        self.original_loader = original_loader
        self.attack = attack

    def __iter__(self):
        for X_batch, y_batch in self.original_loader:
            y_numpy = (
                y_batch.cpu().numpy()
                if hasattr(y_batch, "cpu")
                else np.asarray(y_batch)
            )
            y_poisoned, _ = self.attack.poison_labels(y_numpy)

            if hasattr(y_batch, "to"):
                y_batch = torch.from_numpy(y_poisoned).to(y_batch.device)
            else:
                y_batch = y_poisoned

            yield X_batch, y_batch

    def __len__(self):
        return len(self.original_loader)

    @property
    def dataset(self):
        return self.original_loader.dataset


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

    Example:
        >>> should_be_malicious(0, 10, 0.2, seed=42)  # Might be True
        >>> should_be_malicious(5, 10, 0.2, seed=42)  # Might be False
        >>> should_be_malicious(0, 10, 0.2, seed=42)  # Always same as first call
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


def poison_trainloader_if_malicious(
    trainloader: DataLoader,
    partition_id: int,
    attack_config: "AttackConfig",
    num_clients: int,
) -> Tuple[Union[DataLoader, PoisonedDataLoaderWrapper], bool]:
    """
    Apply attack to trainloader if this client is selected as malicious.

    Determines if this client should be malicious, and if so, applies the
    configured attack to the training dataloader. If attack is disabled or
    client is clean, returns original loader.

    Args:
        trainloader: PyTorch DataLoader with clean training data
        partition_id: This client's unique ID
        attack_config: AttackConfig object with attack parameters
            - enabled: bool - whether attacks are enabled
            - attack_type: str - type of attack ("none", "label_flip", "backdoor", etc.)
            - malicious_ratio: float - fraction of clients [0, 1]
            - poison_fraction: float - fraction of labels to poison [0, 1]
            - victim_label: int - (label flip) source label
            - target_label: int - (label flip) target label
            - seed: int - random seed for reproducibility
        num_clients: Total number of clients in federation

    Returns:
        Tuple of:
        - trainloader (DataLoader or PoisonedDataLoaderWrapper): Original or poisoned dataloader
        - is_malicious (bool): Whether this client is malicious

    Raises:
        ValueError: If attack_type is unknown

    Example:
        >>> attack_cfg = AttackConfig(enabled=True, attack_type="label_flip", malicious_ratio=0.2)
        >>> loader, is_mal = poison_trainloader_if_malicious(trainloader, 0, attack_cfg, 10)
        >>> if is_mal:
        ...     print("Client 0 is malicious - training on poisoned data")
    """
    # Attack disabled → return clean
    if not attack_config.enabled or attack_config.attack_type == "none":
        return trainloader, False

    # Check if this client is malicious
    is_malicious = should_be_malicious(
        partition_id,
        num_clients,
        attack_config.malicious_ratio,
        attack_config.seed,
    )

    # If clean → return clean
    if not is_malicious:
        return trainloader, False

    # Dispatch to appropriate attack
    if attack_config.attack_type == "label_flip":
        attack = LabelFlipAttack(
            victim_label=attack_config.victim_label,
            target_label=attack_config.target_label,
            poison_frac=attack_config.poison_fraction,
        )
        poisoned_loader = PoisonedDataLoaderWrapper(trainloader, attack)
        return poisoned_loader, True

    elif attack_config.attack_type == "backdoor":
        # Future: BackdoorAttack
        raise NotImplementedError("Backdoor attacks not yet implemented")

    elif attack_config.attack_type == "model_poisoning":
        # Future: ModelPoisoningAttack
        raise NotImplementedError("Model poisoning attacks not yet implemented")

    else:
        raise ValueError(
            f"Unknown attack type: {attack_config.attack_type}. "
            f"Available: 'none', 'label_flip', 'backdoor', 'model_poisoning'"
        )
