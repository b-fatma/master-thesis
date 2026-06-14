"""src: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from src.task import train as train_fn, test as test_fn
from src.dataset import load_federated_dataset
from src.config import get_dataset_config, AttackConfig
from src.models import get_model
from src.attacks import (
    should_be_malicious,
    apply_data_poisoning_if_selected,
    apply_model_poisoning_if_selected,
)

app = ClientApp()

# Default dataset (can be overridden via config)
DEFAULT_DATASET = "adult-income-census"


# def _get_input_dim(dataset_name: str) -> int:
#     """Get input feature dimension from dataset configuration.

#     Args:
#         dataset_name: Name of the dataset (e.g., "adult-income-census", "bike-sharing")

#     Returns:
#         Number of input features for the dataset
#     """
#     config = get_dataset_config(dataset_name)
#     return config.input_dim


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Execute local training on this client's partition.

    This Flower train handler:
    1. Loads the federated partition of training data
    2. Receives the global model from server
    3. Performs local training for specified epochs
    4. Returns updated model weights and training metrics

    Handles both classification and regression tasks automatically based on
    configuration. Returns task-specific metrics (e.g., accuracy for classification,
    MAE for regression).

    Args:
        msg: Flower message containing initial model weights and config
        context: Flower context with configuration and node info

    Returns:
        Message with updated model arrays and training metrics
    """

    partition_id = context.node_config["partition-id"]
    num_partitions = context.run_config["num-supernodes"]
    batch_size = context.run_config["batch-size"]
    local_epochs = context.run_config["local-epochs"]
    lr = msg.content["config"]["lr"]
    partitioning_type = context.run_config["partitioning"]
    dirichlet_alpha = context.run_config["dirichlet-alpha"]
    dataset_name = context.run_config.get("dataset-name", DEFAULT_DATASET)

    config = get_dataset_config(dataset_name)
    input_dim = config.input_dim

    global_state_dict = msg.content["arrays"].to_torch_state_dict()
    model = get_model(input_dim, config.task, config.model_config)
    model.load_state_dict(global_state_dict)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    reference_state = {
        name: param.detach().clone() for name, param in model.state_dict().items()
    }

    trainloader, _ = load_federated_dataset(
        partition_id,
        num_partitions,
        batch_size,
        dataset_name,
        partitioning_type,
        dirichlet_alpha,
    )

    # Apply attack if enabled and this client is malicious
    attack_config = AttackConfig.from_dict(msg.content["config"])

    is_malicious_client = should_be_malicious(
        partition_id,
        num_partitions,
        attack_config.malicious_ratio,
        attack_config.seed,
    )

    trainloader, data_poisoned = apply_data_poisoning_if_selected(
        trainloader,
        partition_id,
        attack_config,
        is_malicious_client,
    )

    train_loss, train_metric = train_fn(
        model, trainloader, local_epochs, lr, device, task=config.task
    )

    # Print final data-poisoning statistics for malicious clients.
    if data_poisoned and hasattr(trainloader, "print_poison_stats"):
        trainloader.print_poison_stats(partition_id)

    local_state = model.state_dict()
    uploaded_state, _ = apply_model_poisoning_if_selected(
        local_state,
        reference_state,
        partition_id,
        attack_config,
        is_malicious_client,
    )

    metrics_dict = {
        "train_loss": float(train_loss),
        "num-examples": len(trainloader.dataset),
    }
    if config.task == "classification":
        metrics_dict["train_accuracy"] = float(train_metric)
    else:
        metrics_dict["train_mae"] = float(train_metric)

    content = RecordDict(
        {
            "arrays": ArrayRecord(uploaded_state),
            "metrics": MetricRecord(metrics_dict),
        }
    )
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Execute local evaluation on this client's validation partition.

    This Flower evaluate handler:
    1. Loads the local validation set (20% of client's training partition)
    2. Receives the current global model from server
    3. Evaluates model on validation data
    4. Returns evaluation results

    For classification: Returns loss, accuracy, AUC-ROC, F1, F1-macro
    For regression: Returns MSE, MAE, RMSE, R²

    Args:
        msg: Flower message containing model weights to evaluate
        context: Flower context with configuration and node info

    Returns:
        Message with evaluation metrics
    """

    partition_id = context.node_config["partition-id"]
    num_partitions = context.run_config["num-supernodes"]
    batch_size = context.run_config["batch-size"]
    dataset_name = context.run_config.get("dataset-name", DEFAULT_DATASET)

    config = get_dataset_config(dataset_name)
    input_dim = config.input_dim

    model = get_model(input_dim, config.task, config.model_config)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    _, valloader = load_federated_dataset(
        partition_id, num_partitions, batch_size, dataset_name
    )

    eval_results = test_fn(model, valloader, device, task=config.task)

    if config.task == "classification":
        loss, accuracy, auc_roc, f1, f1_macro, precision, recall = eval_results
        metrics_dict = {
            "eval_loss": float(loss),
            "eval_accuracy": float(accuracy),
            "eval_auc_roc": float(auc_roc),
            "eval_f1": float(f1),
            "eval_f1_macro": float(f1_macro),
            "eval_precision": float(precision),
            "eval_recall": float(recall),
            "num-examples": len(valloader.dataset),
        }
    else:
        mse, mae, rmse, r_squared = eval_results
        metrics_dict = {
            "eval_mse": float(mse),
            "eval_mae": float(mae),
            "eval_rmse": float(rmse),
            "eval_r_squared": float(r_squared),
            "num-examples": len(valloader.dataset),
        }

    content = RecordDict(
        {
            "metrics": MetricRecord(metrics_dict),
        }
    )
    return Message(content=content, reply_to=msg)
