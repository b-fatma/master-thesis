"""
Evaluation utilities with optional W&B logging.
"""

import torch
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import math
from typing import Optional, Tuple, List, Dict
import wandb


def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    task: str = "regression",
    device: str = "cpu",
    use_wandb: bool = False,
    run: Optional[wandb.sdk.wandb_run.Run] = None,
) -> Tuple[Dict[str, float], List[float], List[float]]:
    """
    Evaluate a PyTorch model and optionally log metrics to W&B.

    Args:
        model (nn.Module): Trained model.
        dataloader (DataLoader): Test data loader.
        task (str, optional): "regression" or "classification". Defaults to "regression".
        device (str, optional): Device to run evaluation on. Defaults to "cpu".
        use_wandb (bool, optional): Whether to log metrics to W&B. Defaults to False.
        run (wandb.sdk.wandb_run.Run, optional): W&B run object to log metrics to. Defaults to None.

    Returns:
        tuple:
            - metrics (dict): Evaluation metrics.
                Regression: mse, rmse, r2
                Classification: accuracy, f1, precision, recall
            - preds (list): Model predictions
            - targets (list): True target values

    Logs (if W&B enabled):
        - All returned metrics
    """
    model.to(device)
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            outputs = model(X).squeeze()

            if task == "classification":
                probs = torch.sigmoid(outputs)
                predictions = (probs > 0.5).int().cpu().numpy()
            else:
                predictions = outputs.cpu().numpy()

            preds.extend(predictions)
            targets.extend(y.numpy())

    metrics = {}
    if task == "regression":
        mse = mean_squared_error(targets, preds)
        rmse = math.sqrt(mse)
        r2 = r2_score(targets, preds)
        metrics = {"mse": mse, "rmse": rmse, "r2": r2}
    else:
        metrics = {
            "accuracy": accuracy_score(targets, preds),
            "f1": f1_score(targets, preds, zero_division=0),
            "precision": precision_score(targets, preds, zero_division=0),
            "recall": recall_score(targets, preds, zero_division=0),
        }

    if use_wandb and run is not None:
        run.log(metrics)

    return metrics, preds, targets
