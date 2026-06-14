"""
Training utilities with optional W&B logging.
"""

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict
import wandb


def train(
    model: nn.Module,
    dataloader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    task: str = "regression",
    device: str = "cpu",
    use_wandb: bool = False,
    config: Optional[Dict] = None,
    model_name: Optional[str] = None,
) -> Optional[wandb.sdk.wandb_run.Run]:
    """
    Train a PyTorch model with optional W&B logging.

    Args:
        model (nn.Module): PyTorch model to train.
        dataloader (DataLoader): Training data loader.
        epochs (int, optional): Number of epochs. Defaults to 10.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        task (str, optional): "regression" or "classification". Defaults to "regression".
        device (str, optional): Device to run training on ("cpu" or "cuda"). Defaults to "cpu".
        use_wandb (bool, optional): Whether to log metrics to W&B. Defaults to False.
        config (dict, optional): Config dictionary to log to W&B. Must include "dataset" key. Defaults to None.
        model_name (str, optional): Model name for run identification. Defaults to None.

    Returns:
        wandb.sdk.wandb_run.Run or None: W&B run object if `use_wandb=True`, else None.

    Logs (if W&B enabled):
        - Epoch number
        - Training loss per epoch
    """
    run = None
    if use_wandb:
        config = config or {}
        # Add model_name to config if provided
        if model_name:
            config["model"] = model_name

        # Build run name: centralized_<model>_<dataset>
        dataset_name = (
            config.get("dataset", "unknown")
            .replace("b-fatma/", "")
            .replace("-federated", "")
        )
        model_for_name = config.get("model", "unknown")
        run_name = f"centralized_{model_for_name}_{dataset_name}"

        run = wandb.init(
            project="master-thesis", group="centralized", name=run_name, config=config
        )

    model.to(device)

    # Choose loss function
    if task == "regression":
        criterion = nn.MSELoss()
    elif task == "classification":
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError("Invalid task. Choose 'regression' or 'classification'.")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0

        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X).squeeze()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        if use_wandb and run is not None:
            run.log({"epoch": epoch + 1, "train_loss": avg_loss})

    return run
