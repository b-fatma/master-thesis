"""
Utilities for interacting with Hugging Face datasets.
"""

from datasets import Dataset, load_dataset
import pandas as pd
from torch.utils.data import DataLoader
import torch


def save_to_hf(
    X_train, X_test, y_train_log, y_test_log, repo_name, label_col="cnt_log"
):
    """
    Save training and test data as Hugging Face datasets.

    Combines feature matrices (X) and target vectors (y) into DataFrames,
    removes unnecessary columns, converts to Hugging Face `Dataset`,
    and pushes to the specified Hugging Face Hub repository.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train_log (array-like or pd.Series): Training targets.
        y_test_log (array-like or pd.Series): Test targets.
        repo_name (str): Hugging Face dataset repository name.
        label_col (str): Name for the target column (default: "cnt_log" for bike-sharing).

    Notes:
        - The combined DataFrames will include columns from X and the target column.
        - Any existing index columns are dropped to avoid duplication on Hugging Face.
    """
    # Convert y to DataFrame
    y_train_df = pd.DataFrame(y_train_log, columns=[label_col])
    y_test_df = pd.DataFrame(y_test_log, columns=[label_col])

    # Combine X and y
    train_df = pd.concat([X_train.reset_index(drop=True), y_train_df], axis=1)
    test_df = pd.concat([X_test.reset_index(drop=True), y_test_df], axis=1)

    # Remove any unnecessary columns (like 'index' if present)
    train_df = train_df.loc[:, ~train_df.columns.str.contains("^index$")]
    test_df = test_df.loc[:, ~test_df.columns.str.contains("^index$")]

    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Push datasets to Hugging Face Hub
    train_dataset.push_to_hub(repo_name, split="train")
    test_dataset.push_to_hub(repo_name, split="test")

    print(f"Uploaded dataset to Hugging Face: {repo_name}")
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)


def load_dataloaders_from_hf(
    dataset_name: str, batch_size: int = 32, label_col="cnt_log"
):
    """
    Load dataset from Hugging Face and return PyTorch DataLoaders.

    Args:
        dataset_name (str): Hugging Face dataset identifier.
        batch_size (int): Batch size for training.
        label_col (str): Name of the label column in the dataset.

    Returns:
        tuple: (train_loader, test_loader, input_dim)
    """
    dataset = load_dataset(dataset_name)

    train_data = dataset["train"]
    test_data = dataset["test"]

    feature_cols = [col for col in train_data.column_names if col != label_col]

    def collate_fn(batch):
        """
        Convert batch into tensors.
        """
        X = torch.tensor(
            [[item[c] for c in feature_cols] for item in batch], dtype=torch.float32
        )
        y = torch.tensor([item[label_col] for item in batch], dtype=torch.float32)
        return X, y

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    input_dim = len(feature_cols)
    return train_loader, test_loader, input_dim
