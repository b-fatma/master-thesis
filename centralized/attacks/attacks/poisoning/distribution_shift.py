"""
Distribution Shift Attack — Regression Poisoning via Target Corruption
DUPLICATED FROM: src/attacks/distribution_shift.py (keep in sync)

This module provides a distribution shift attack for regression tasks via noise injection.
Analogous to label-flip attacks on classification, this attack systematically modifies
continuous target values to degrade model performance through variance injection.

Mechanisms:
  - Gaussian Noise: Add normally-distributed noise to targets (N(0, σ²))
  - Uniform Noise: Add uniformly-distributed noise to targets (U(-a, a))

Selection Strategies:
  - Random: Poison a random fraction of all samples
  - Quantile-Based: Poison samples whose targets fall in a quantile range
"""

from enum import Enum
from typing import Tuple, Callable, Iterator
import numpy as np
import torch
from copy import deepcopy


class SelectionStrategy(Enum):
    """Strategies for selecting which samples to poison."""

    RANDOM = "random"
    QUANTILE_RANGE = "quantile_range"


class DistributionShiftAttack:
    """
    Distribution shift attack on regression targets via noise injection.

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
          - For Gaussian noise: standard deviation (σ)
          - For Uniform noise: bound range (±a)
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
        """
        Select indices of samples to potentially poison based on strategy.

        Args:
            y (np.ndarray): Target values, shape [N]

        Returns:
            np.ndarray: Indices of candidate samples for poisoning
        """
        if self.selection_strategy == "random":
            # All samples are candidates
            return np.arange(len(y))

        elif self.selection_strategy == "quantile_range":
            # Select samples whose targets fall in quantile range
            q_min = np.quantile(y, self.victim_quantile_min)
            q_max = np.quantile(y, self.victim_quantile_max)
            return np.where((y >= q_min) & (y <= q_max))[0]

    def _apply_shift(self, y_candidates: np.ndarray) -> np.ndarray:
        """
        Apply noise injection to candidate target values.

        Args:
            y_candidates (np.ndarray): Target values to corrupt

        Returns:
            np.ndarray: Corrupted values with noise added (same shape)

        Raises:
            ValueError: if shift_mechanism is invalid
        """
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
        """
        Corrupt a subset of regression targets.

        Core attack logic: selects victim samples based on strategy,
        randomly chooses poison_frac of them, and applies shift/noise.

        Args:
            y (np.ndarray): Original target values, shape [N]

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - y_poisoned: Targets with subset corrupted. Same shape as y.
                - poison_mask: Boolean mask marking modified samples. True where corrupted.

        Side Effects:
            Prints summary of poisoning action.

        Example:
            >>> attack = DistributionShiftAttack(shift_mechanism="additive", shift_param=-50)
            >>> y_orig = np.array([100.0, 150.0, 200.0, 250.0])
            >>> y_poi, mask = attack.poison_labels(y_orig)
            >>> # y_poi might be [100.0, 150.0, 150.0, 250.0]  (shifted one value)
            >>> print(mask)  # [False, False, True, False]

        Note:
            - Deterministic: uses self._rng initialized with self.seed
            - In-place: allocates new array, doesn't modify input y
            - Empty victims: if no candidate samples exist, returns original y
        """
        y_poisoned = y.copy()
        poison_mask = np.zeros(len(y), dtype=bool)

        # Find candidate victim indices based on selection strategy
        victim_idxs = self._select_victims(y)

        if len(victim_idxs) == 0:
            print(
                f"[DistShift] Warning: no candidate samples found for poisoning "
                f"(quantile range [{self.victim_quantile_min}, {self.victim_quantile_max}])"
            )
            return y_poisoned, poison_mask

        # Randomly select subset to poison
        n_to_poison = max(1, int(len(victim_idxs) * self.poison_frac))
        poison_idxs = self._rng.choice(victim_idxs, size=n_to_poison, replace=False)

        # Apply shift/noise
        y_poisoned[poison_idxs] = self._apply_shift(y[poison_idxs])
        poison_mask[poison_idxs] = True

        return y_poisoned, poison_mask

    def poison_dataset(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """
        Poison targets of a dataset.

        Args:
            X (np.ndarray): Input features, shape [N, D]. NOT modified.
            y (np.ndarray): Target values, shape [N]. WILL be modified and returned.

        Returns:
            Tuple: (X, y_poisoned, poison_mask)
                - X: Original features (unchanged reference)
                - y_poisoned: Targets with subset corrupted
                - poison_mask: Boolean mask of which samples were poisoned

        Side Effects:
            Prints summary: "[DistShift] Poisoned N/M targets (mechanism=X, shift_param=Y, frac=Z)"

        Example:
            >>> X = np.random.randn(1000, 10)
            >>> y = np.random.uniform(0, 100, 1000)
            >>> attack = DistributionShiftAttack(shift_mechanism="additive", shift_param=-50, poison_frac=0.1)
            >>> X_p, y_p, mask = attack.poison_dataset(X, y)
            >>> print(f'Poisoned {mask.sum()} samples')  # ~100 (10% of 1000)
        """
        y_poisoned, poison_mask = self.poison_labels(y)
        n_poisoned = poison_mask.sum()
        print(
            f"[DistShift] Poisoned {n_poisoned}/{len(y)} targets "
            f"(mechanism={self.shift_mechanism}, shift_param={self.shift_param}, "
            f"frac={n_poisoned / len(y):.3f})"
        )
        return X, y_poisoned, poison_mask

    def poison_torch_dataset(self, dataset):
        """
        Poison a PyTorch torchvision dataset with continuous targets.

        Modifies the .targets attribute on a deepcopy to avoid
        mutating the original. Useful for PyTorch workflows that expect
        dataset objects with a .targets field.

        Args:
            dataset: PyTorch dataset with .targets attribute
                    (e.g., custom regression dataset)

        Returns:
            Tuple: (poisoned_dataset, poison_mask)
                - poisoned_dataset: deepcopy of input with corrupted targets
                - poison_mask: boolean mask of which samples were poisoned

        Raises:
            ValueError: if dataset has no .targets attribute

        Example:
            >>> # Assuming a custom regression dataset with .targets
            >>> dataset = MyRegressionDataset()
            >>> attack = DistributionShiftAttack(shift_mechanism="noise_gaussian", shift_param=5.0)
            >>> poisoned_dataset, mask = attack.poison_torch_dataset(dataset)
            >>> print(f'Original targets preserved: {dataset.targets[0]}')
            >>> print(f'Poisoned targets changed: {poisoned_dataset.targets[0]}')

        Note:
            - Creates a deepcopy to avoid modifying the original dataset
            - Handles both torch.Tensor and numpy array targets
        """
        dataset = deepcopy(dataset)  # don't mutate the original

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
        """
        Poison batches from a PyTorch DataLoader on-the-fly (for FL).

        Args:
            dataloader: PyTorch DataLoader yielding (X_batch, y_batch) tuples
            convert_labels (Callable, optional): Function to convert y_batch to/from
                numpy for label manipulation. Default: converts torch tensors.
                Example: lambda y: y.cpu().numpy() for torch tensors.

        Yields:
            Tuple: (X_batch, y_poisoned) where y_poisoned has values corrupted

        Example:
            >>> attack = DistributionShiftAttack(shift_mechanism="additive", shift_param=-25, poison_frac=0.2)
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

            y_np = convert_labels(y_batch).astype(np.float32)

            # Poison the batch
            y_poisoned_np, _ = self.poison_labels(y_np)

            # Convert back to original type if needed
            if isinstance(y_batch, torch.Tensor):
                y_poisoned = torch.from_numpy(y_poisoned_np).to(y_batch.device)
            else:
                y_poisoned = y_poisoned_np

            yield X_batch, y_poisoned
