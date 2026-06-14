"""
Model saving utilities with optional W&B artifact logging.
"""

import os
import torch
from typing import Optional
import wandb


def save_model_artifact(
    model: torch.nn.Module,
    dataset_name: str,
    filename: str = "model.pt",
    run: Optional[wandb.sdk.wandb_run.Run] = None,
    models_dir: str = "models",
) -> str:
    """
    Save a PyTorch model locally in models/<dataset_name>/ and optionally log to W&B.

    Args:
        model (nn.Module): PyTorch model to save.
        dataset_name (str): Dataset name used to create subfolder and artifact name.
        filename (str, optional): File name for the model. Defaults to "model.pt".
        run (wandb.sdk.wandb_run.Run, optional): W&B run object for logging. Defaults to None.
        models_dir (str, optional): Base folder to save models. Defaults to "models".

    Returns:
        str: Full path to the saved model file.

    Logs (if W&B run provided):
        - Model artifact
    """
    dataset_dir = os.path.join(models_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    model_path = os.path.join(dataset_dir, filename)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved locally at: {model_path}")

    if run is not None:
        artifact_name = f"{dataset_name}-model"
        artifact = wandb.Artifact(name=artifact_name, type="model")
        artifact.add_file(model_path)
        run.log_artifact(artifact)
        print(f"Artifact '{artifact_name}' logged to W&B.")

    return model_path
