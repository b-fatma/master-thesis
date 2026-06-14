"""
Label Flip Attack — Shared Implementation for Centralized and Federated Learning

This module provides a backend-agnostic label flip attack that works with:
  - NumPy arrays (for centralized learning)
  - PyTorch DataLoaders (for federated learning)
  - PyTorch datasets (for compatibility)
"""

import numpy as np
import torch
from copy import deepcopy
from typing import Tuple, Callable, Iterator


class LabelFlipAttack:
    """
    Label flip attack on a dataset (centralized or client-side in FL).

    This is the from-scratch implementation. Read every line —
    there is no magic here. The entire attack is in poison_labels().
    Core principle: flip a random subset of victim-class labels to target class.

    This attack is:
      - **Deterministic**: Reproducible with seed
      - **Modular**: Works with numpy arrays, datasets, and dataloaders
      - **Stealthy**: Only modifies labels (inputs remain unchanged)
      - **Measurable**: Returns mask of modified samples for analysis

    Attributes:
        victim_label (int): Class label whose samples will be flipped
        target_label (int): Class to flip victims to
        poison_frac (float): Fraction of victim-class samples to poison, in (0, 1]
        seed (int): Random seed for reproducibility
    """

    def __init__(
        self,
        victim_label: int,
        target_label: int,
        poison_frac: float = 1.0,
        seed: int = 42,
    ):
        """
        Initialize label flip attack.

        Args:
            victim_label (int): Class whose labels will be flipped (e.g., 0 for binary)
            target_label (int): Class to flip victim labels to (e.g., 1 for binary)
            poison_frac (float): Fraction of victim-class samples to flip. Default: 1.0 (all)
            seed (int): Random seed for reproducibility. Default: 42

        Raises:
            AssertionError: if poison_frac not in (0, 1]
        """
        assert 0.0 < poison_frac <= 1.0, "poison_frac must be in (0, 1]"
        self.victim_label = victim_label
        self.target_label = target_label
        self.poison_frac = poison_frac
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def poison_labels(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Flip labels for a random subset of victim-class samples.

        Core attack logic: identifies all samples of victim_label,
        randomly selects poison_frac of them, and changes their label
        to target_label. This is the entire label flip attack.

        Args:
            y (np.ndarray): Original labels, shape [N]. Values are class IDs (0, 1, ...).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - y_poisoned: Labels with subset flipped. Same shape as y.
                - poison_mask: Boolean mask marking flipped samples. True where modified.

        Side Effects:
            Prints warning if victim_label not found in y.

        Example:
            >>> attack = LabelFlipAttack(victim_label=0, target_label=1, poison_frac=0.5)
            >>> y_orig = np.array([0, 0, 0, 1, 1, 1])
            >>> y_poi, mask = attack.poison_labels(y_orig)
            >>> # y_poi might be [0, 1, 0, 1, 1, 1]  (flipped one '0' to '1')
            >>> print(mask)  # [False, True, False, False, False, False]

        Note:
            - Deterministic: uses self._rng initialized with self.seed
            - In-place: allocates new array, doesn't modify input y
            - Empty victims: if no victim_label samples exist, returns original y
        """
        y_poisoned = y.copy()
        poison_mask = np.zeros(len(y), dtype=bool)

        # find all victim-class indices
        victim_idxs = np.where(y == self.victim_label)[0]

        if len(victim_idxs) == 0:
            print(
                f"[LabelFlip] Warning: no samples with label {self.victim_label} found"
            )
            return y_poisoned, poison_mask

        # select a random subset to flip
        n_to_flip = max(1, int(len(victim_idxs) * self.poison_frac))
        flip_idxs = self._rng.choice(victim_idxs, size=n_to_flip, replace=False)

        # apply flip — this is the entire "attack"
        y_poisoned[flip_idxs] = self.target_label
        poison_mask[flip_idxs] = True

        return y_poisoned, poison_mask

    def poison_dataset(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """
        Poison both features and labels of a dataset.

        In practice: X is unchanged, only y is modified. Features remain
        indistinguishable from clean data—this makes label flip
        extremely stealthy and hard to detect via input inspection.

        Args:
            X (np.ndarray): Input features, shape [N, D]. NOT modified.
            y (np.ndarray): Labels, shape [N]. WILL be modified and returned.

        Returns:
            Tuple: (X, y_poisoned, poison_mask)
                - X: Original features (unchanged reference)
                - y_poisoned: Labels with subset flipped
                - poison_mask: Boolean mask of which samples were flipped

        Side Effects:
            Prints summary: "[LabelFlip] Flipped N/M labels (victim→target, frac=X.XXX)"

        Example:
            >>> X = np.random.randn(1000, 30)
            >>> y = np.array([0]*750 + [1]*250)  # imbalanced
            >>> attack = LabelFlipAttack(0, 1, poison_frac=0.1)
            >>> X_p, y_p, mask = attack.poison_dataset(X, y)
            >>> print(f'Flipped {mask.sum()} samples')  # ~75 (10% of 750)
        """
        y_poisoned, poison_mask = self.poison_labels(y)
        n_flipped = poison_mask.sum()
        print(
            f"[LabelFlip] Flipped {n_flipped}/{len(y)} labels "
            f"({self.victim_label} → {self.target_label}, "
            f"frac={n_flipped / len(y):.3f})"
        )
        return X, y_poisoned, poison_mask

    def poison_torch_dataset(self, dataset):
        """
        Poison a PyTorch torchvision dataset (e.g., MNIST, CIFAR).

        Modifies the .targets attribute on a deepcopy to avoid
        mutating the original. Useful for PyTorch workflows that expect
        dataset objects with a .targets field.

        Args:
            dataset: PyTorch dataset with .targets attribute
                    (e.g., datasets.MNIST, datasets.CIFAR10)

        Returns:
            Tuple: (poisoned_dataset, poison_mask)
                - poisoned_dataset: deepcopy of input with flipped targets
                - poison_mask: boolean mask of which samples were flipped

        Raises:
            ValueError: if dataset has no .targets attribute

        Example:
            >>> from torchvision import datasets
            >>> mnist = datasets.MNIST(root='./data', download=True)
            >>> attack = LabelFlipAttack(0, 1, poison_frac=0.5)
            >>> poisoned_mnist, mask = attack.poison_torch_dataset(mnist)
            >>> print(f'Original labels preserved: {mnist.targets[0]}')
            >>> print(f'Poisoned labels changed: {poisoned_mnist.targets[0]}')

        Note:
            - Creates a deepcopy to avoid modifying the original dataset
            - Handles both torch.Tensor and list targets
        """
        dataset = deepcopy(dataset)  # don't mutate the original

        if hasattr(dataset, "targets"):
            targets = np.array(dataset.targets)
        else:
            raise ValueError(
                "Dataset has no .targets attribute. Use poison_labels() manually."
            )

        _, poison_mask = self.poison_labels(targets)
        targets[poison_mask] = self.target_label

        if isinstance(dataset.targets, torch.Tensor):
            dataset.targets = torch.tensor(targets)
        else:
            dataset.targets = targets.tolist()

        return dataset, poison_mask

    def poison_dataloader(
        self, dataloader, convert_labels: Callable = None
    ) -> Iterator:
        """
        Poison batches from a PyTorch DataLoader on-the-fly (for FL).

        Yields poisoned batches without storing the entire dataset in memory.
        Each batch's labels are flipped according to the attack parameters.

        This is the key method for federated learning attacks, where each
        malicious client poisons its local batches during training.

        Args:
            dataloader: PyTorch DataLoader yielding (X_batch, y_batch) tuples
            convert_labels (Callable, optional): Function to convert y_batch to/from
                numpy for label manipulation. Default: converts torch tensors.
                Example: lambda y: y.numpy() for torch tensors.

        Yields:
            Tuple: (X_batch, y_poisoned) where y_poisoned has labels flipped

        Example:
            >>> attack = LabelFlipAttack(victim_label=0, target_label=1, poison_frac=0.1)
            >>> for X_batch, y_batch in attack.poison_dataloader(trainloader):
            ...     loss = model.train_on_batch(X_batch, y_batch)

        Notes:
            - Non-deterministic batch order (dataloader shuffle applies first)
            - Memory efficient: doesn't load full dataset
            - Works with any backend (numpy, torch, etc.)
        """
        for X_batch, y_batch in dataloader:
            # Convert to numpy if needed
            if convert_labels is None:
                convert_labels = lambda y: (
                    y.cpu().numpy() if isinstance(y, torch.Tensor) else y
                )

            y_np = convert_labels(y_batch)

            # Poison the batch
            y_poisoned_np, _ = self.poison_labels(y_np)

            # Convert back to original type if needed
            if isinstance(y_batch, torch.Tensor):
                y_poisoned = torch.from_numpy(y_poisoned_np).to(y_batch.device)
            else:
                y_poisoned = y_poisoned_np

            yield X_batch, y_poisoned


class LabelFlipFLClient:
    """
    Malicious Federated Learning client that performs label flip attacks.

    Design pattern for FL poisoning:
      1. A malicious client owns local training data
      2. Before local training, it poisons its local dataset
      3. It trains normally on poisoned data → generates poisoned gradients
      4. It uploads gradients to server (server/other clients unaware)
      5. FedAvg aggregates poisoned gradients → global model degrades

    This class encapsulates this pattern. Use with your Flower client or any FL framework.

    Attributes:
        attack (LabelFlipAttack): Attack instance with configured parameters
        victim_label (int): Class to flip
        target_label (int): Class to flip to
        poison_frac (float): Fraction of victim samples to flip

    Example:
        >>> mal_client = LabelFlipFLClient(victim_label=0, target_label=1, poison_frac=0.3)
        >>> X_train, y_train = load_data(...)
        >>> X_p, y_p, mask = mal_client.poison_local_data(X_train, y_train)
        >>> # Now train with poisoned data
        >>> model.fit(X_p, y_p, epochs=10)  # generates poisoned gradients
    """

    def __init__(
        self,
        victim_label: int,
        target_label: int,
        poison_frac: float = 1.0,
        seed: int = 42,
    ):
        """
        Initialize a malicious FL client with label flip attack.

        Args:
            victim_label (int): Class label to flip from (0 for binary).
            target_label (int): Class label to flip to (1 for binary).
            poison_frac (float): Fraction of victim-class samples to flip.
                               Default: 1.0 (flip all victim samples).
            seed (int): Random seed for reproducibility. Default: 42.

        Raises:
            AssertionError: if poison_frac not in (0, 1]
        """
        self.attack = LabelFlipAttack(victim_label, target_label, poison_frac, seed)
        self.victim_label = victim_label
        self.target_label = target_label
        self.poison_frac = poison_frac
        self._poisoned = False

    def poison_local_data(self, X: np.ndarray, y: np.ndarray):
        """
        Poison the client's local training data before training.

        This is the entry point for the attack workflow. Call this
        before model.fit() with your local training data.

        Args:
            X (np.ndarray): Local training features, shape [N, D].
            y (np.ndarray): Local training labels, shape [N].

        Returns:
            Tuple: (X_poisoned, y_poisoned, poison_mask)
                - X_poisoned: unchanged (features are never modified)
                - y_poisoned: labels with subset flipped
                - poison_mask: boolean mask of flipped samples

        Side Effects:
            Sets self._poisoned = True

        Example:
            >>> mal_client = LabelFlipFLClient(...)
            >>> X, y = load_local_data()
            >>> X, y, _ = mal_client.poison_local_data(X, y)  # poison
            >>> model.fit(X, y, epochs=10)
        """
        X_p, y_p, mask = self.attack.poison_dataset(X, y)
        self._poisoned = True
        return X_p, y_p, mask
