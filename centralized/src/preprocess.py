"""
Preprocessing utilities for feature engineering and tensor conversion.
"""

import numpy as np
import torch


def add_cyclic_hour(hour_column):
    """
    Convert hour feature into cyclic representation.

    Args:
        hour_column (array-like): Hour values (0-23)

    Returns:
        tuple: (sin_hour, cos_hour)
    """
    sin_hour = np.sin(2 * np.pi * hour_column / 24)
    cos_hour = np.cos(2 * np.pi * hour_column / 24)
    return sin_hour, cos_hour


def to_tensors(X, y):
    """
    Convert numpy arrays into PyTorch tensors.

    Args:
        X (np.ndarray): Features
        y (np.ndarray): Labels

    Returns:
        tuple: (X_tensor, y_tensor)
    """
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return X_tensor, y_tensor
