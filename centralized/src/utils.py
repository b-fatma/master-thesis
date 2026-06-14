"""
Common utilities for centralized learning and attacks.

Utilities:
  - fix_seed: Set reproducible random seeds across all libraries
  - compute_accuracy: Evaluate model accuracy on a DataLoader
  - accuracy_drop: Measure the drop in accuracy between clean and poisoned models
"""

import numpy as np
import torch


def fix_seed(seed=42):
    """
    Set random seeds across all libraries for reproducibility.

    Ensures that all sources of randomness (Python, NumPy, PyTorch on CPU,
    PyTorch on GPU) use the same seed, guaranteeing reproducible results
    across runs.

    Args:
        seed (int): Random seed to use. Default: 42.

    Returns:
        None. Modifies global random state.

    Example:
        >>> fix_seed(42)
        >>> # Now all random operations are deterministic
        >>> model = MLP(...)
        >>> model.train()  # Same initialization every time

    Note:
        Call this at the start of your experiment, before creating models
        and loading data, to ensure deterministic behavior.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_accuracy(model, dataloader, device):
    """
    Compute top-1 accuracy on a DataLoader.

    Evaluates the model on a given DataLoader and returns the fraction of
    correct predictions. Used for both classification and evaluation.

    Args:
        model (nn.Module): Trained PyTorch model.
        dataloader (DataLoader): DataLoader with evaluation data.
        device (str): Device to run evaluation on ("cpu" or "cuda").

    Returns:
        float: Accuracy as a fraction in [0, 1]. Example: 0.85 means 85%.

    Example:
        >>> acc = compute_accuracy(model, test_loader, device="cuda")
        >>> print(f"Accuracy: {acc*100:.2f}%")
    """
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)

            # For classification: argmax(logits)
            if logits.dim() > 1 and logits.size(1) > 1:
                preds = torch.argmax(logits, dim=1)
                y_labels = y.long()
            # For binary classification with single-logit output
            else:
                preds = (logits.squeeze() > 0.0).long()
                y_labels = y.long()

            correct += (preds == y_labels).sum().item()
            total += y_labels.size(0)

    model.train()
    return correct / total if total > 0 else 0.0


def accuracy_drop(clean_acc, poisoned_acc):
    """
    Measure the drop in accuracy from clean to poisoned model.

    Computes: clean_acc - poisoned_acc. Positive values indicate that
    poisoning successfully degraded the model. Used to quantify attack
    effectiveness.

    Args:
        clean_acc (float): Accuracy of clean (unpoisoned) model.
        poisoned_acc (float): Accuracy of poisoned model.

    Returns:
        float: Accuracy drop, typically in [0, 1]. Example: 0.40 means 40% drop.

    Example:
        >>> drop = accuracy_drop(0.85, 0.45)
        >>> print(f"Attack dropped accuracy by {drop*100:.2f}%")
    """
    return clean_acc - poisoned_acc
