"""
Unified model definitions for centralized learning and federated learning.

Enforces task-driven architecture where models are determined ONLY by:
  - task: "classification" or "regression"
  - model_type: "logreg", "linear", or "mlp"

CORE MODELS:
  - LogisticRegression: Binary classification only
  - LinearRegression: Regression only
  - MLP: Classification or regression with configurable hidden layers

Dataset-specific models are REMOVED. All experiments now use get_model() factory
which ensures identical architecture across centralized and federated settings.
"""

import torch.nn as nn
from typing import Dict, Any, Literal


class LogisticRegression(nn.Module):
    """Logistic regression model for binary classification.

    A single-layer linear model (logistic regression) for binary classification.
    Outputs raw logits (not probabilities) suitable for BCEWithLogitsLoss.

    Attributes:
        linear: Linear transformation layer from input_dim to 1
    """

    def __init__(self, input_dim: int):
        """Initialize the logistic regression model.

        Args:
            input_dim: Number of input features
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        """Forward pass returning raw logits.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Logits tensor of shape (batch_size, 1)
        """
        return self.linear(x)


class LinearRegression(nn.Module):
    """Linear model for regression.

    A single-layer linear model for regression tasks.
    Outputs continuous predictions (not bounded).

    Attributes:
        linear: Linear transformation layer from input_dim to 1
    """

    def __init__(self, input_dim: int):
        """Initialize the linear regression model.

        Args:
            input_dim: Number of input features
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        """Forward pass returning continuous predictions.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Predictions tensor of shape (batch_size, 1)
        """
        return self.linear(x)


class MLPClassifier(nn.Module):
    """Multi-layer perceptron for binary classification.

    A flexible MLP with configurable hidden layers, ReLU activations, and optional
    dropout. Outputs raw logits (not probabilities) suitable for BCEWithLogitsLoss.

    Attributes:
        net: Sequential container of linear, ReLU, and dropout layers
    """

    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.0):
        """Initialize the MLP classifier.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions, e.g. [64, 32]
            dropout: Dropout probability (default: 0.0 for no dropout)
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass returning raw logits.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Logits tensor of shape (batch_size, 1)
        """
        return self.net(x)


class MLPRegressor(nn.Module):
    """Multi-layer perceptron for regression.

    A flexible MLP with configurable hidden layers, ReLU activations, and optional
    dropout. Outputs continuous predictions (unbounded).

    Attributes:
        net: Sequential container of linear, ReLU, and dropout layers
    """

    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.0):
        """Initialize the MLP regressor.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions, e.g. [64, 32]
            dropout: Dropout probability (default: 0.0 for no dropout)
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer (linear for regression)
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass returning continuous predictions.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Predictions tensor of shape (batch_size, 1)
        """
        return self.net(x)


def get_model(
    input_dim: int,
    task: Literal["classification", "regression"],
    model_config: Dict[str, Any],
) -> nn.Module:
    """Create a task-specific model based on configuration.

    This is the UNIFIED factory function for model creation. It ensures that:
    - ALL models are task-driven (not dataset-specific)
    - Models work identically in centralized and federated settings
    - Same initialization across all clients/experiments

    TASK CONSTRAINTS:
    - classification: allowed models are "logreg", "mlp"
    - regression: allowed models are "linear", "mlp"

    Args:
        input_dim: Number of input features
        task: "classification" or "regression"
        model_config: Dict with keys:
            - "type": "logreg", "linear", or "mlp" (required)
            - "hidden_dims": [64, 32] (for mlp only)
            - "dropout": 0.1 (optional, for mlp only)

    Returns:
        Initialized PyTorch model

    Raises:
        ValueError: If model_type is not allowed for the given task

    Examples:
        # Logistic regression classifier
        model = get_model(30, "classification", {"type": "logreg"})

        # Linear regressor
        model = get_model(12, "regression", {"type": "linear"})

        # MLP classifier
        model = get_model(30, "classification", {
            "type": "mlp",
            "hidden_dims": [64, 32],
            "dropout": 0.1
        })

        # MLP regressor
        model = get_model(12, "regression", {
            "type": "mlp",
            "hidden_dims": [128, 64, 32],
            "dropout": 0.2
        })
    """
    model_type = model_config.get("type", "linear")

    if task == "classification":
        if model_type == "logreg":
            return LogisticRegression(input_dim)
        elif model_type == "mlp":
            hidden_dims = model_config.get("hidden_dims", [64, 32])
            dropout = model_config.get("dropout", 0.0)
            return MLPClassifier(input_dim, hidden_dims, dropout)
        else:
            raise ValueError(
                f"Unknown classifier type: {model_type}. "
                f"Allowed types for classification: 'logreg', 'mlp'"
            )

    elif task == "regression":
        if model_type == "linear":
            return LinearRegression(input_dim)
        elif model_type == "mlp":
            hidden_dims = model_config.get("hidden_dims", [64, 32])
            dropout = model_config.get("dropout", 0.0)
            return MLPRegressor(input_dim, hidden_dims, dropout)
        else:
            raise ValueError(
                f"Unknown regressor type: {model_type}. "
                f"Allowed types for regression: 'linear', 'mlp'"
            )

    else:
        raise ValueError(
            f"Unknown task: {task}. Choose 'classification' or 'regression'"
        )
