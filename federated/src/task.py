import torch
import torch.nn as nn
from typing import Literal, Tuple

from .dataset import load_federated_dataset
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)
import numpy as np


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load partition of data."""
    return load_federated_dataset(partition_id, num_partitions, batch_size)


# ============================================================================
# CLASSIFICATION TRAINING AND EVALUATION
# ============================================================================


def train_classification(
    net: nn.Module, trainloader, epochs: int, lr: float, device
) -> Tuple[float, float]:
    """
    Train a binary classification model.

    Assumes model returns logits (not probabilities).
    Uses BCEWithLogitsLoss for numerical stability.

    Returns:
        (loss, accuracy)
    """
    net.to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss, correct, total = 0.0, 0, 0

    for _ in range(epochs):
        for X_batch, y_batch in trainloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = net(X_batch)  # Model returns logits
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(X_batch)
            correct += ((torch.sigmoid(logits) >= 0.5).float() == y_batch).sum().item()
            total += len(X_batch)

    return running_loss / total, correct / total


def test_classification(
    net: nn.Module, testloader, device
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Evaluate a binary classification model.

    Returns:
        (loss, accuracy, auc_roc, f1, f1_macro, precision, recall)
    """
    net.to(device)
    criterion = nn.BCEWithLogitsLoss()
    loss, correct, all_probs, all_labels = 0.0, 0, [], []

    with torch.no_grad():
        for X_batch, y_batch in testloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = net(X_batch)  # Model returns logits
            loss += criterion(logits, y_batch).item() * len(X_batch)
            probs = torch.sigmoid(logits)
            correct += ((probs >= 0.5).float() == y_batch).sum().item()
            all_probs.append(probs.cpu())
            all_labels.append(y_batch.cpu())

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    preds = (probs >= 0.5).astype(int)

    return (
        loss / len(testloader.dataset),
        correct / len(testloader.dataset),
        roc_auc_score(labels, probs),
        f1_score(labels, preds, zero_division=0),
        f1_score(labels, preds, average="macro", zero_division=0),
        precision_score(labels, preds, zero_division=0),
        recall_score(labels, preds, zero_division=0),
    )


# ============================================================================
# REGRESSION TRAINING AND EVALUATION
# ============================================================================


def train_regression(
    net: nn.Module, trainloader, epochs: int, lr: float, device
) -> Tuple[float, float]:
    """
    Train a regression model.

    Uses MSELoss (Mean Squared Error).

    Returns:
        (loss, mae)  # MSE loss and Mean Absolute Error metric
    """
    net.to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss, sum_mae, total = 0.0, 0.0, 0

    for _ in range(epochs):
        for X_batch, y_batch in trainloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            predictions = net(X_batch)  # Model returns continuous predictions
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(X_batch)
            mae = torch.abs(predictions - y_batch).sum().item()
            sum_mae += mae
            total += len(X_batch)

    return running_loss / total, sum_mae / total


def test_regression(
    net: nn.Module, testloader, device
) -> Tuple[float, float, float, float]:
    """
    Evaluate a regression model.

    Returns:
        (mse_loss, mae, rmse, r_squared)
    """
    net.to(device)
    criterion = nn.MSELoss()

    all_preds = []
    all_labels = []
    running_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in testloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = net(X_batch)  # Model returns continuous predictions
            loss = criterion(predictions, y_batch)
            running_loss += loss.item() * len(X_batch)
            all_preds.append(predictions.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    preds = np.concatenate(all_preds).flatten()
    labels = np.concatenate(all_labels).flatten()

    mse = running_loss / len(testloader.dataset)
    mae = mean_absolute_error(labels, preds)
    rmse = np.sqrt(mse)

    # R² = 1 - (SS_res / SS_tot)
    ss_res = np.sum((labels - preds) ** 2)
    ss_tot = np.sum((labels - np.mean(labels)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    return mse, mae, rmse, r_squared


# ============================================================================
# GENERIC TRAIN/TEST DISPATCH (for backward compatibility)
# ============================================================================


def train(
    net,
    trainloader,
    epochs,
    lr,
    device,
    task: Literal["classification", "regression"] = "classification",
):
    """
    Generic train function that dispatches to task-specific implementation.

    Args:
        net: Neural network model
        trainloader: Training data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        task: "classification" or "regression"

    Returns:
        Task-specific metrics
    """
    if task == "classification":
        return train_classification(net, trainloader, epochs, lr, device)
    elif task == "regression":
        return train_regression(net, trainloader, epochs, lr, device)
    else:
        raise ValueError(f"Unknown task: {task}")


def test(
    net,
    testloader,
    device,
    task: Literal["classification", "regression"] = "classification",
):
    """
    Generic test function that dispatches to task-specific implementation.

    Args:
        net: Neural network model
        testloader: Test data loader
        device: Device to evaluate on
        task: "classification" or "regression"

    Returns:
        Task-specific evaluation metrics. For classification: (loss, accuracy, auc_roc, f1, f1_macro, precision, recall)
        For regression: (mse, mae, rmse, r_squared)
    """
    if task == "classification":
        return test_classification(net, testloader, device)
    elif task == "regression":
        return test_regression(net, testloader, device)
    else:
        raise ValueError(f"Unknown task: {task}")
