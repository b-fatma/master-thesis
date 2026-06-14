"""
Label Flip Attack
"""

import numpy as np
import torch
from copy import deepcopy


# ═══════════════════════════════════════════════════════════════
# Core attack logic — FROM SCRATCH
# ═══════════════════════════════════════════════════════════════


class LabelFlipAttack:
    """
    Label flip attack on a dataset (centralized or client-side in FL).

    This is the from-scratch implementation. Read every line —
    there is no magic here. The entire attack is in _flip_labels().
    """

    def __init__(
        self,
        victim_label: int,
        target_label: int,
        poison_frac: float = 1.0,
        seed: int = 42,
    ):
        """
        Args:
            victim_label : class whose labels will be flipped
            target_label : class to flip victim labels to
            poison_frac  : fraction of victim-class samples to flip (1.0 = all)
            seed         : reproducibility
        """
        assert 0.0 < poison_frac <= 1.0, "poison_frac must be in (0, 1]"
        self.victim_label = victim_label
        self.target_label = target_label
        self.poison_frac = poison_frac
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def poison_labels(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Flip labels for a random subset of victim-class samples.

        Core attack logic: identifies all samples of victim_label,
        randomly selects poison_frac of them, and changes their label
        to target_label. This is the entire label flip attack.

        Args:
            y (np.ndarray): Original labels, shape [N]. Values are class IDs (0, 1, ...).

        Returns:
            tuple: (y_poisoned, poison_mask)
                - y_poisoned (np.ndarray): Labels with subset flipped. Same shape as y.
                - poison_mask (np.ndarray): Boolean mask marking flipped samples.
                                           True where label was modified.

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

    def poison_dataset(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Poison both features and labels of a dataset.

        In practice: X is unchanged, only y is modified. Features remain
        indistinguishable from clean data—this makes label flip
        extremely stealthy and hard to detect via input inspection.

        Args:
            X (np.ndarray): Input features, shape [N, D]. NOT modified.
            y (np.ndarray): Labels, shape [N]. WILL be modified and returned.

        Returns:
            tuple: (X, y_poisoned, poison_mask)
                - X (np.ndarray): Original features (unchanged reference)
                - y_poisoned (np.ndarray): Labels with subset flipped
                - poison_mask (np.ndarray): Boolean mask of which samples were flipped

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

        Modifies the .targets attribute in-place (on a deepcopy to avoid
        mutating the original). Useful for PyTorch workflows that expect
        dataset objects with a .targets field.

        Args:
            dataset: PyTorch dataset with .targets attribute
                    (e.g., datasets.MNIST, datasets.CIFAR10)

        Returns:
            tuple: (poisoned_dataset, poison_mask)
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


# ═══════════════════════════════════════════════════════════════
# FL-compatible malicious client wrapper
# ═══════════════════════════════════════════════════════════════


class LabelFlipFLClient:
    """
    Malicious Federated Learning client that performs label flip attacks.

    Design pattern for FL poisoning:
      1. A malicious client owns local training data
      2. Before local training, it poisons its local dataset
      3. It trains normally on poisoned data → generates poisoned gradients
      4. It uploads gradients to server (server/other clients unaware)
      5. FedAvg aggregates poisoned gradients → global model degrades

    This class encapsulates this pattern. Instantiate with a base Flower
    client, then call poison_local_data() before fit().

    Attributes:
        client (NumPyClient): Base Flower client to wrap
        attack (LabelFlipAttack): Attack instance with configured parameters
        _poisoned (bool): Internal flag tracking if data has been poisoned

    Example:
        >>> base_client = MyFlowerClient(...)
        >>> mal_client = LabelFlipFLClient(base_client, victim_label=0,
        ...                               target_label=1, poison_frac=0.3)
        >>> X_train, y_train = load_data(...)
        >>> X_p, y_p, mask = mal_client.poison_local_data(X_train, y_train)
        >>> # Now train with poisoned data
        >>> base_client.fit(X_p, y_p, epochs=10)  # generates poisoned gradients
    """

    def __init__(
        self,
        base_client,
        victim_label: int,
        target_label: int,
        poison_frac: float = 1.0,
        seed: int = 42,
    ):
        """
        Initialize a malicious FL client with label flip attack.

        Args:
            base_client (flwr.client.NumPyClient): Base Flower client to wrap.
                                                   Must implement fit() and evaluate().
            victim_label (int): Class label to flip from (0 for binary).
            target_label (int): Class label to flip to (1 for binary).
            poison_frac (float): Fraction of victim-class samples to flip.
                               Default: 1.0 (flip all victim samples).
            seed (int): Random seed for reproducibility. Default: 42.

        Raises:
            AssertionError: if poison_frac not in (0, 1]
        """
        self.client = base_client
        self.attack = LabelFlipAttack(victim_label, target_label, poison_frac, seed)
        self._poisoned = False

    def poison_local_data(self, X: np.ndarray, y: np.ndarray):
        """
        Poison the client's local training data before fit().

        This is the entry point for the attack workflow. Call this
        in your client's fit() method before passing data to model.fit().

        Args:
            X (np.ndarray): Local training features, shape [N, D].
            y (np.ndarray): Local training labels, shape [N].

        Returns:
            tuple: (X_poisoned, y_poisoned, poison_mask)
                - X_poisoned: unchanged (features are never modified)
                - y_poisoned: labels with subset flipped
                - poison_mask: boolean mask of flipped samples

        Side Effects:
            Sets self._poisoned = True

        Example:
            >>> mal_client = LabelFlipFLClient(...)
            >>> def fit(self, parameters, config):
            >>>     X, y = load_local_data()
            >>>     X, y, _ = self.poison_local_data(X, y)  # poison
            >>>     self.model.fit(X, y, epochs=10)
            >>>     return self.get_parameters(), len(X), {}
        """
        X_p, y_p, mask = self.attack.poison_dataset(X, y)
        self._poisoned = True
        return X_p, y_p, mask


# ─────────────────────────────────────────────────────────────
# Reference: AIJack LabelFlipAttackClientManager
# ─────────────────────────────────────────────────────────────


def aijack_reference():
    """
    Document the label flip implementation strategy used by AIJack.

    Key insight: entire attack is ONE line in training loop:
        target[target == victim_label] = target_label
    Everything else is framework glue.

    See Also:
        AIJack source: aijack.attack.poison.LabelFlipAttackClientManager
    """
    pass
