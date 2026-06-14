import torch
from torch.utils.data import DataLoader, Subset
from typing import Tuple
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from sklearn.model_selection import train_test_split

from src.config import get_dataset_config


def _create_collate_fn(feature_cols: list, label_col: str):
    """Create a custom collate function for dataset batching.

    Converts HuggingFace dataset samples (dictionaries) into PyTorch tensors.
    Features are cast to float32 and labels are cast to float32 (even for
    classification, as BCEWithLogitsLoss expects float labels).

    Args:
        feature_cols: List of feature column names to include
        label_col: Name of the label/target column

    Returns:
        Callable collate function that takes a batch list and returns (X, y) tensors

    Example:
        >>> collate = _create_collate_fn(['age', 'income'], 'label')
        >>> X, y = collate(batch)  # batch from DataLoader
        >>> X.shape, y.shape  # (batch_size, len(feature_cols)), (batch_size, 1)
    """

    def collate_fn(batch):
        X = torch.tensor(
            [[item[c] for c in feature_cols] for item in batch], dtype=torch.float32
        )
        y = torch.tensor([item[label_col] for item in batch], dtype=torch.float32)
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
        return X, y

    return collate_fn


def load_centralized_dataset(
    batch_size: int = 32, dataset_name: str = "adult-income-census"
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Load full centralized dataset for server evaluation.

    Returns the full train and test splits as-is (not partitioned).
    Train is used only for reference; test is for server-side evaluation.

    Args:
        batch_size: Batch size for DataLoader
        dataset_name: Name of dataset from config

    Returns:
        train_loader, test_loader, input_dim
    """
    config = get_dataset_config(dataset_name)

    # Load from HF
    dataset = load_dataset(config.hf_repo)

    train_data = dataset["train"]
    test_data = dataset["test"]

    # Get feature columns (all except label)
    feature_cols = [col for col in train_data.column_names if col != config.label_col]

    # Create collate function
    collate_fn = _create_collate_fn(feature_cols, config.label_col)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    input_dim = len(feature_cols)
    return train_loader, test_loader, input_dim


def load_federated_dataset(
    partition_id: int,
    num_partitions: int,
    batch_size: int = 32,
    dataset_name: str = "adult-income-census",
    partitioning_type: str = "iid",
    dirichlet_alpha: float = 0.5,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load a federated client partition from HF dataset using IID partitioning.

    Uses flwr_datasets.FederatedDataset with IidPartitioner to create stratified
    partitions where each client receives a representative sample of all classes.
    This prevents label-homogeneous partitions that can cause issues with attacks
    (e.g., "no samples with label X").

    Each client then has its local train/val split (80/20) for local evaluation.
    The TEST data remains on the server for global evaluation.

    Args:
        partition_id: ID of this client's partition (0 to num_partitions-1)
        num_partitions: Total number of partitions
        batch_size: Batch size for DataLoader
        dataset_name: Name of dataset from config

    Returns:
        train_loader, val_loader (local validation)
    """
    config = get_dataset_config(dataset_name)

    # Create IID partitioner and FederatedDataset
    if partitioning_type == "dirichlet":
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by=config.label_col,
            alpha=dirichlet_alpha,
        )
    else:  # iid
        partitioner = IidPartitioner(num_partitions=num_partitions)
    fds = FederatedDataset(dataset=config.hf_repo, partitioners={"train": partitioner})

    # Load this client's partition
    partition_data = fds.load_partition(partition_id, split="train")

    # Get feature columns (all except label)
    feature_cols = [
        col for col in partition_data.column_names if col != config.label_col
    ]

    # Stratified split: ensure train/val have same label distribution
    # Extract labels for stratification
    labels = partition_data[config.label_col]
    partition_size = len(partition_data)

    # Use stratified split (80/20) to maintain class balance
    if config.task == "classification":
        stratify_arg = labels
    else:
        stratify_arg = None

    train_indices, val_indices = train_test_split(
        list(range(partition_size)),
        test_size=0.2,
        train_size=0.8,
        stratify=stratify_arg,
        random_state=42,  # For reproducibility
    )

    local_train_data = Subset(partition_data, train_indices)
    local_val_data = Subset(partition_data, val_indices)

    # Create collate function
    collate_fn = _create_collate_fn(feature_cols, config.label_col)

    train_loader = DataLoader(
        local_train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        local_val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_loader, val_loader


def get_input_dim(dataset_name: str = "adult-income-census") -> int:
    """
    Get input feature dimension for a dataset.

    Args:
        dataset_name: Name of dataset from config

    Returns:
        Number of input features
    """
    config = get_dataset_config(dataset_name)
    dataset = load_dataset(config.hf_repo)
    train_data = dataset["train"]
    feature_cols = [col for col in train_data.column_names if col != config.label_col]
    return len(feature_cols)
