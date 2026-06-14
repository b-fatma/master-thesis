"""src: A Flower / PyTorch app."""

import logging
from pathlib import Path
import torch
import wandb
from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx, FedTrimmedAvg, FedMedian, Krum

from src.task import test
from src.dataset import load_centralized_dataset
from src.config import get_dataset_config, AttackConfig, DetectionConfig
from src.models import get_model
from src.detection import FLDetectorMixin, detect_mad_shapcosim_outliers

# SHAP pipeline
build_server_side_shap_context = None
compute_and_save_round_shap = None
generate_plots_for_run = None
try:
    from src.shap_pipeline import (
        build_server_side_shap_context,
        compute_and_save_round_shap,
        generate_plots_for_run,
    )
    from src.shap_pipeline.compute import compute_shap_matrix
except Exception:
    try:
        from shap_pipeline import (
            build_server_side_shap_context,
            compute_and_save_round_shap,
            generate_plots_for_run,
        )
        from shap_pipeline.compute import compute_shap_matrix
    except Exception as exc:
        # shap_pipeline is optional; guard imports so server still works without SHAP
        build_server_side_shap_context = None
        compute_and_save_round_shap = None
        generate_plots_for_run = None
        logging.warning("shap_pipeline import failed: %s. SHAP disabled.", exc)

app = ServerApp()

# Flower already streams logs to the console during `flwr run --stream`.
# Prevent propagation to the root logger so each line is emitted once.
logging.getLogger("flwr").propagate = False

DEFAULT_SHAP_OUTPUT_ROOT = str(
    Path(__file__).resolve().parents[1] / "experiments" / "shap_outputs"
)

# Global state for server
run = None  # wandb run, initialized in main()
shap_runtime = None  # SHAP runtime context, initialized in main() if SHAP is enabled
DATASET_NAME = "adult-income-census"  # Will be set in main()
DATASET_CONFIG = None  # Will be set in main()
GLOBAL_REFERENCE_SHAP = None  # Reference model SHAP vector trained on server dataset
DETECTION_CONFIG = None  # Will be set in main()
DETECTION_TRACKER: dict = {
    "per_round": [],  # list of dicts with TP/FP/FN/TN per round
    "cumulative": {"TP": 0, "FP": 0, "FN": 0, "TN": 0},
}


def _extract_message_content(item):
    """Best-effort extraction of a Flower message content payload.

    Args:
        item: A Flower reply item which may be a tuple of (client_proxy, msg)
            or a message-like object.

    Returns:
        The extracted content payload if found, otherwise the original
        message object.
    """

    msg = item[1] if isinstance(item, tuple) and len(item) >= 2 else item
    if hasattr(msg, "message"):
        try:
            return msg.message.content
        except Exception:
            return getattr(msg.message, "content", None)
    if hasattr(msg, "content"):
        return msg.content
    if hasattr(msg, "payload"):
        return msg.payload
    return msg


def _extract_num_examples_from_reply(item) -> float | None:
    """Extract the client example count from a Flower train reply.

    This inspects common locations for a `metrics` mapping and reads the
    `num-examples` / `num_examples` field if present.

    Args:
        item: A Flower reply item which may be a tuple or message-like
            object containing metrics.

    Returns:
        The number of examples reported by the client as a float, or
        ``None`` if it cannot be determined.
    """

    content = _extract_message_content(item)
    if content is None:
        return None

    metrics = None
    if hasattr(content, "get"):
        metrics = content.get("metrics") or content.get("Metrics")
    else:
        try:
            metrics = content["metrics"]
        except Exception:
            metrics = getattr(content, "metrics", None)

    if metrics is None:
        return None

    try:
        if hasattr(metrics, "to_dict"):
            metrics_dict = metrics.to_dict()
        else:
            metrics_dict = dict(metrics)
        value = metrics_dict.get("num-examples", metrics_dict.get("num_examples"))
        return float(value) if value is not None else None
    except Exception:
        return None


def _rewrite_reply_num_examples(item, num_examples: float):
    """Return a reply whose metrics report the provided example count.

    Modifies or constructs a reply message so that the `metrics` field
    contains the requested `num-examples` and `num_examples` entries.

    Args:
        item: Original reply (possibly a tuple of (client_proxy, msg)).
        num_examples: Example count to write into the reply metrics.

    Returns:
        A new reply message or a tuple of (client_proxy, new_message)
        with the updated metrics. If the original content cannot be
        interpreted, returns the original `item`.
    """

    if isinstance(item, tuple) and len(item) >= 2:
        client_proxy, msg = item[0], item[1]
    else:
        client_proxy, msg = None, item

    content = _extract_message_content(item)
    if content is None or not hasattr(content, "get"):
        return item

    try:
        content_dict = dict(content)
    except Exception:
        return item

    metrics_obj = content_dict.get("metrics") or content_dict.get("Metrics")
    try:
        if metrics_obj is None:
            metrics_dict = {}
        elif hasattr(metrics_obj, "to_dict"):
            metrics_dict = dict(metrics_obj.to_dict())
        else:
            metrics_dict = dict(metrics_obj)
    except Exception:
        metrics_dict = {}

    metrics_dict["num-examples"] = float(num_examples)
    metrics_dict["num_examples"] = float(num_examples)
    content_dict["metrics"] = MetricRecord(metrics_dict)

    reply_to = getattr(msg, "reply_to", msg)
    new_msg = Message(content=RecordDict(content_dict), reply_to=reply_to)
    if client_proxy is None:
        return new_msg
    return (client_proxy, new_msg)


def _train_reference_model_and_compute_shap(
    dataset_name: str,
    dataset_config,
    shap_runtime,
    output_root: str,
    run_name: str,
    num_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,
    wandb_run=None,
):
    """Train a server-side reference model and compute its SHAP vector.

    The reference model is trained on the centralized server dataset and
    its SHAP explanations are computed and saved. Training metrics are
    optionally logged to Weights & Biases and artifacts are stored
    under ``output_root/run_name/server-model``.

    Args:
        dataset_name: Dataset identifier used to load data.
        dataset_config: Dataset configuration object with attributes like
            ``input_dim`` and ``task``.
        shap_runtime: SHAP runtime/context providing background and
            explanation data.
        output_root: Root directory where artifacts will be saved.
        run_name: Name of the run used to namespace outputs.
        num_epochs: Number of training epochs.
        batch_size: Batch size used for training.
        lr: Learning rate for training.
        wandb_run: Optional wandb run object for logging metrics.

    Returns:
        A NumPy array containing the reference SHAP vector (shape: [num_features]),
        or ``None`` if training or SHAP computation failed.
    """
    try:
        # Load server-side training and test data. Use the centralized test
        # split as the server's trusted dataset (root trust) for both
        # training the reference model and evaluation.
        _, testloader, _ = load_centralized_dataset(
            batch_size=batch_size, dataset_name=dataset_name
        )

        # Create and train reference model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ref_model = get_model(
            dataset_config.input_dim,
            dataset_config.task,
            dataset_config.model_config,
        )
        ref_model.to(device)

        from src.task import train, test
        import numpy as np

        logging.info(
            "Training reference model on server-side TRUST dataset (test split) for %d epochs...",
            num_epochs,
        )
        # Train on the trusted test split (root trust)
        train_result = train(
            ref_model, testloader, num_epochs, lr, device, task=dataset_config.task
        )
        ref_model.eval()

        # Evaluate on the same test split
        test_result = test(ref_model, testloader, device, task=dataset_config.task)

        # Parse and log metrics
        if dataset_config.task == "classification":
            train_loss, train_accuracy = train_result
            (
                test_loss,
                test_accuracy,
                test_auc,
                test_f1,
                test_f1_macro,
                test_precision,
                test_recall,
            ) = test_result
            logging.info(
                "✓ Reference model - Train: loss=%.4f, accuracy=%.4f | Test: loss=%.4f, accuracy=%.4f, auc=%.4f",
                train_loss,
                train_accuracy,
                test_loss,
                test_accuracy,
                test_auc,
            )
            print("[Server Reference Model] Classification")
            print(f"  Train: Loss={train_loss:.4f}, Accuracy={train_accuracy:.4f}")
            print(
                f"  Test:  Loss={test_loss:.4f}, Accuracy={test_accuracy:.4f}, AUC={test_auc:.4f}"
            )
            metrics_dict = {
                "server-model/train_loss": float(train_loss),
                "server-model/train_accuracy": float(train_accuracy),
                "server-model/test_loss": float(test_loss),
                "server-model/test_accuracy": float(test_accuracy),
                "server-model/test_auc": float(test_auc),
            }
        else:
            train_loss, train_metric = train_result
            test_mse, test_mae, test_rmse, test_r2 = test_result
            logging.info(
                "✓ Reference model - Train: loss=%.4f, mae=%.4f | Test: mse=%.4f, mae=%.4f, rmse=%.4f, r²=%.4f",
                train_loss,
                train_metric,
                test_mse,
                test_mae,
                test_rmse,
                test_r2,
            )
            print("[Server Reference Model] Regression")
            print(f"  Train: Loss={train_loss:.4f}, MAE={train_metric:.4f}")
            print(
                f"  Test:  MSE={test_mse:.4f}, MAE={test_mae:.4f}, RMSE={test_rmse:.4f}, R²={test_r2:.4f}"
            )
            metrics_dict = {
                "server-model/train_loss": float(train_loss),
                "server-model/train_mae": float(train_metric),
                "server-model/test_mse": float(test_mse),
                "server-model/test_mae": float(test_mae),
                "server-model/test_rmse": float(test_rmse),
                "server-model/test_r2": float(test_r2),
            }

        # Log to wandb
        if wandb_run is not None:
            wandb_run.log(metrics_dict)

        # Compute SHAP on reference dataset
        if shap_runtime is None:
            logging.warning(
                "shap_runtime not available; skipping reference SHAP computation"
            )
            return None

        # Prefer signed mean SHAP computed from the full SHAP matrix so sign information is preserved.
        shap_vec = None
        try:
            if "compute_shap_matrix" in globals() and compute_shap_matrix is not None:
                logging.info(
                    "Computing signed SHAP matrix for reference model and taking mean across samples..."
                )
                values = compute_shap_matrix(
                    ref_model,
                    background_data=shap_runtime.background_data,
                    explanation_data=shap_runtime.explanation_data,
                    background_samples=shap_runtime.background_samples,
                )
                if values.ndim == 1:
                    shap_vec = values.reshape(-1)
                else:
                    shap_vec = values.mean(axis=0).reshape(-1)
            else:
                logging.warning(
                    "compute_shap_matrix unavailable; skipping reference SHAP"
                )
                return None
        except Exception as exc:
            logging.warning("Reference SHAP computation failed: %s", exc)
            return None

        # Save reference SHAP to disk under server-model/
        try:
            server_model_dir = Path(output_root) / run_name / "server-model"
            server_model_dir.mkdir(parents=True, exist_ok=True)

            # Save SHAP vector
            shap_npz = server_model_dir / "reference_shap.npz"
            np.savez_compressed(
                str(shap_npz),
                shap_values=shap_vec.astype(np.float32),
                feature_count=len(shap_vec),
            )
            logging.info("✓ Saved reference SHAP to: %s", shap_npz)

            # Save metrics
            metrics_npz = server_model_dir / "training_metrics.npz"
            if dataset_config.task == "classification":
                np.savez_compressed(
                    str(metrics_npz),
                    task="classification",
                    train_loss=train_loss,
                    train_accuracy=train_accuracy,
                    test_loss=test_loss,
                    test_accuracy=test_accuracy,
                    test_auc=test_auc,
                )
            else:
                np.savez_compressed(
                    str(metrics_npz),
                    task="regression",
                    train_loss=train_loss,
                    train_mae=train_metric,
                    test_mse=test_mse,
                    test_mae=test_mae,
                    test_rmse=test_rmse,
                    test_r2=test_r2,
                )
            logging.info("✓ Saved training metrics to: %s", metrics_npz)
        except Exception as save_exc:
            logging.warning("Failed to save reference model artifacts: %s", save_exc)

        return shap_vec
    except Exception as exc:
        logging.warning("Failed to train reference model and compute SHAP: %s", exc)
        return None


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Initialize the Flower server and execute the federated simulation.

    The function reads run-time configuration from ``context.run_config``,
    prepares the global model and aggregation strategy (optionally
    enabling SHAP-based detection), runs the federated training loop,
    logs metrics to W&B, and saves final artifacts.

    Args:
        grid: Flower Grid for distributed task execution.
        context: Flower Context containing run configuration and client
            registry.

    Returns:
        None. Side effects include training runs, saved artifacts and
        W&B logging.
    """
    global run, shap_runtime, DATASET_NAME, DATASET_CONFIG, DETECTION_CONFIG

    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    batch_size: int = context.run_config["batch-size"]
    local_epochs: int = context.run_config["local-epochs"]
    proximal_mu: float = context.run_config["proximal-mu"]
    partitioning_type: str = context.run_config["partitioning"]
    aggregation_strategy: str = context.run_config["aggregation-strategy"]
    dirichlet_alpha: float = context.run_config["dirichlet-alpha"]
    dataset_name: str = context.run_config.get("dataset-name", "adult-income-census")
    trimmed_mean_beta: float = context.run_config.get("trimmed-mean-beta", 0.1)

    # Note: G-ShapCosim detector removed; keep detection config simple

    # Build attack config from run config (convert dashes to underscores for from_dict)
    attack_config_raw = {
        "attack_enabled": context.run_config.get("attack-enabled", False),
        "attack_type": context.run_config.get("attack-type", "none"),
        "malicious_ratio": context.run_config.get("malicious-ratio", 0.2),
        "poison_fraction": context.run_config.get("poison-fraction", 0.1),
        "victim_label": context.run_config.get("victim-label", 0),
        "target_label": context.run_config.get("target-label", 1),
        "attack_seed": context.run_config.get("attack-seed", 42),
        # Distribution shift parameters
        "shift_mechanism": context.run_config.get("shift-mechanism", "additive"),
        "shift_param": context.run_config.get("shift-param", -50.0),
        "selection_strategy": context.run_config.get("selection-strategy", "random"),
        "victim_quantile_min": context.run_config.get("victim-quantile-min", 0.0),
        "victim_quantile_max": context.run_config.get("victim-quantile-max", 1.0),
        "model_attack_lambda": context.run_config.get("model-attack-lambda", 1.0),
    }
    attack_config = AttackConfig.from_dict(attack_config_raw)
    detection_config_raw = {
        "detection_enabled": context.run_config.get("detection-enabled", False),
        "detection_use_hybrid": context.run_config.get("detection-use-hybrid", True),
        "detection_mode": context.run_config.get("detection-mode", "mad-shapcosim"),
        "detection_alpha": context.run_config.get("detection-alpha", 0.7),
        "detection_mad_multiplier": context.run_config.get(
            "detection-mad-multiplier", 3.0
        ),
        # IQR-based detection removed; keep MAD multiplier only
        "detection_min_clients": context.run_config.get("detection-min-clients", 3),
        # G-ShapCosim parameters removed
    }
    detection_config = DetectionConfig.from_dict(detection_config_raw)

    DATASET_NAME = dataset_name
    DATASET_CONFIG = get_dataset_config(dataset_name)
    input_dim = DATASET_CONFIG.input_dim

    model_name = DATASET_CONFIG.model_config.get("type", "linear")
    # strategy_prefix converted for concise run/group naming
    strategy_prefix = aggregation_strategy.replace("-", "_").upper()
    if attack_config.attack_type == "label_flip":
        attack_param = (
            f"pf{attack_config.poison_fraction}_"
            f"v{attack_config.victim_label}_t{attack_config.target_label}"
        )
    elif attack_config.attack_type == "distribution_shift":
        attack_param = f"{attack_config.shift_mechanism}_sp{attack_config.shift_param}"
    elif attack_config.attack_type in {"history", "mpaf"}:
        attack_param = f"lambda{attack_config.model_attack_lambda}"
    else:
        attack_param = "none"

    partitioning_suffix = (
        f"dirichlet_alpha{dirichlet_alpha}"
        if partitioning_type == "dirichlet"
        else partitioning_type
    )

    # Get number of partitions (clients) from context early so it can be part of the run name
    num_clients = context.run_config.get("num-supernodes", 10)

    run_name = (
        f"{strategy_prefix}_{model_name}_{dataset_name}_r{num_rounds}_c{num_clients}_ep{local_epochs}_"
        f"lr{lr}_bs{batch_size}_{partitioning_suffix}_"
        f"attack_{attack_config.attack_type}_{attack_param}"
    )
    # Append malicious ratio to run name for easier experiment identification
    try:
        if (
            getattr(attack_config, "malicious_ratio", None) is not None
            and attack_config.attack_type != "none"
        ):
            run_name += f"_ratio{float(attack_config.malicious_ratio)}"
    except Exception:
        pass
    if aggregation_strategy.lower() == "fedprox":
        run_name += f"_mu{proximal_mu}"
    elif aggregation_strategy.lower() == "fedtrimmedavg":
        run_name += f"_beta{trimmed_mean_beta}"

    # num_clients already retrieved above

    # Append detection info when enabled so run names reflect detector and threshold
    try:
        if detection_config.enabled:
            det_method = getattr(detection_config, "mode", "mad-shapcosim")
            run_name += f"_det_{det_method}_mad{detection_config.mad_multiplier}"
    except Exception:
        pass

    group_name = (
        f"{strategy_prefix}_{partitioning_suffix}_"
        f"attack_{attack_config.attack_type}_{attack_param}"
    )
    try:
        if (
            getattr(attack_config, "malicious_ratio", None) is not None
            and attack_config.attack_type != "none"
        ):
            group_name += f"_ratio{float(attack_config.malicious_ratio)}"
    except Exception:
        pass
    wandb_config = {
        "model": model_name,
        "dataset": dataset_name,
        "task": DATASET_CONFIG.task,
        "num_rounds": num_rounds,
        "num_clients": num_clients,
        "lr": lr,
        "batch_size": batch_size,
        "local_epochs": local_epochs,
        "fraction_evaluate": fraction_evaluate,
        "input_dim": input_dim,
        "optimizer": "adam",
        "aggregation_strategy": aggregation_strategy,
        "partitioning_type": partitioning_type,
        "attack_enabled": attack_config.enabled,
        "attack_type": attack_config.attack_type,
        "detection_enabled": detection_config.enabled,
        "detection_use_hybrid": detection_config.use_hybrid,
        "detection_mode": detection_config.mode,
        "detection_alpha": detection_config.alpha,
        "detection_mad_multiplier": detection_config.mad_multiplier,
        "detection_min_clients": detection_config.min_clients,
    }

    if aggregation_strategy.lower() == "fedprox":
        wandb_config["proximal_mu"] = proximal_mu
    elif aggregation_strategy.lower() == "fedtrimmedavg":
        wandb_config["trimmed_mean_beta"] = trimmed_mean_beta

    if partitioning_type == "dirichlet":
        wandb_config["beta"] = dirichlet_alpha

    if attack_config.attack_type != "none":
        wandb_config["malicious_ratio"] = attack_config.malicious_ratio
        if attack_config.attack_type == "label_flip":
            wandb_config["poison_fraction"] = attack_config.poison_fraction
            wandb_config["victim_label"] = attack_config.victim_label
            wandb_config["target_label"] = attack_config.target_label
        elif attack_config.attack_type == "distribution_shift":
            wandb_config["shift_mechanism"] = attack_config.shift_mechanism
            wandb_config["shift_param"] = attack_config.shift_param
            wandb_config["selection_strategy"] = attack_config.selection_strategy
            if attack_config.selection_strategy == "quantile_range":
                wandb_config["victim_quantile_min"] = attack_config.victim_quantile_min
                wandb_config["victim_quantile_max"] = attack_config.victim_quantile_max
        elif attack_config.attack_type in {"history", "mpaf"}:
            wandb_config["model_attack_lambda"] = attack_config.model_attack_lambda

    run = wandb.init(
        project="master-thesis",
        group=group_name,
        name=run_name,
        config=wandb_config,
    )

    # Define detection metrics for W&B with 'round' as the step metric
    try:
        wandb.define_metric("detection/*", step_metric="round")
        wandb.define_metric("detection/cumulative/*", step_metric="round")
    except Exception:
        pass

    global_model = get_model(
        input_dim, DATASET_CONFIG.task, DATASET_CONFIG.model_config
    )
    arrays = ArrayRecord(global_model.state_dict())
    strategy_lower = aggregation_strategy.lower()
    # Optionally build SHAP runtime context
    global DETECTION_CONFIG
    DETECTION_CONFIG = detection_config

    shap_runtime = None
    shap_requested = bool(context.run_config.get("shap-enabled", False))
    shap_enabled = shap_requested and build_server_side_shap_context is not None
    logging.debug(
        "SHAP requested=%s, shap_pipeline_available=%s, shap_enabled=%s",
        shap_requested,
        build_server_side_shap_context is not None,
        shap_enabled,
    )
    if shap_enabled:
        shap_runtime = build_server_side_shap_context(
            dataset_name=dataset_name,
            run_name=run_name,
            output_root=context.run_config.get(
                "shap-output-root", DEFAULT_SHAP_OUTPUT_ROOT
            ),
            background_samples=int(context.run_config.get("shap-background-size", 256)),
            explanation_samples=int(
                context.run_config.get("shap-explanation-size", 256)
            ),
        )

        # Train reference model on server-side dataset and compute SHAP
        global GLOBAL_REFERENCE_SHAP
        GLOBAL_REFERENCE_SHAP = _train_reference_model_and_compute_shap(
            dataset_name=dataset_name,
            dataset_config=DATASET_CONFIG,
            shap_runtime=shap_runtime,
            output_root=context.run_config.get(
                "shap-output-root", DEFAULT_SHAP_OUTPUT_ROOT
            ),
            run_name=run_name,
            num_epochs=context.run_config.get("reference-model-epochs", 50),
            batch_size=context.run_config.get("reference-model-batch-size", batch_size),
            lr=context.run_config.get("reference-model-lr", lr),
            wandb_run=run,
        )
        if GLOBAL_REFERENCE_SHAP is not None:
            logging.info(
                "✓ Global reference SHAP computed: shape=%s",
                GLOBAL_REFERENCE_SHAP.shape
                if hasattr(GLOBAL_REFERENCE_SHAP, "shape")
                else len(GLOBAL_REFERENCE_SHAP),
            )
        else:
            logging.warning(
                "✗ Global reference SHAP computation failed; skipping reference baseline"
            )

    logging.info("→ attack_config=%s", attack_config)
    logging.info("→ detection_config=%s", detection_config)

    # Strategy factory that optionally wraps aggregate_train to run SHAP per-round
    class ShapStrategyMixin:
        """Mixin that computes per-round SHAP values and optionally filters client replies.

        The mixin attaches to a Flower strategy and, on configured rounds,
        extracts client model updates, computes SHAP explanations, runs a
        detection routine and removes flagged clients from aggregation and
        evaluation.
        """

        def __init__(self, *args, **kwargs):
            """Initialize mixin and capture SHAP-related configuration.

            Args:
                *args: Positional arguments forwarded to the parent strategy.
                **kwargs: Keyword arguments. Expected SHAP-related keys:
                    - ``shap_runtime``: SHAP runtime/context object.
                    - ``shap_every_n_rounds``: Frequency (int) to compute SHAP.
                    - ``use_shap_cosine_aggregation``: Bool to enable alternative
                      aggregation behavior.
                    - ``attack_config``: AttackConfig instance.
                    - ``detection_config``: DetectionConfig instance.
                    - ``num_clients``: Expected number of clients.
                    - ``dataset_name``: Dataset identifier string.

            Other args/kwargs are passed to ``super().__init__``.
            """
            # Pop SHAP-specific kwargs so base strategy __init__ doesn't receive them
            shap_runtime = kwargs.pop("shap_runtime", None)
            shap_every_n_rounds = kwargs.pop("shap_every_n_rounds", 1)
            use_shap_cosine_aggregation = kwargs.pop(
                "use_shap_cosine_aggregation", False
            )
            attack_config = kwargs.pop("attack_config", None)
            detection_config = kwargs.pop("detection_config", None)
            num_clients = kwargs.pop("num_clients", 0)
            dataset_name = kwargs.pop("dataset_name", None)

            self._shap_runtime = shap_runtime
            try:
                self._shap_every_n_rounds = int(shap_every_n_rounds)
            except Exception:
                self._shap_every_n_rounds = 1
            self._use_shap_cosine_aggregation = bool(use_shap_cosine_aggregation)
            self._shap_attack_config = attack_config
            self._detection_config = detection_config
            try:
                self._shap_num_clients = int(num_clients)
            except Exception:
                self._shap_num_clients = 0
            self._shap_dataset_name = dataset_name

            # Map: server_round -> set(model_name)
            self._flagged_clients_for_round = {}
            super().__init__(*args, **kwargs)

        def aggregate_train(self, server_round: int, replies):
            """Compute SHAP for client updates and optionally filter replies.

            On configured rounds this method extracts client model weights
            from ``replies``, computes per-client SHAP vectors using the
            SHAP runtime, runs the configured detection routine and filters
            out flagged clients prior to calling the base strategy's
            ``aggregate_train``.

            Args:
                server_round: Current server round number.
                replies: The raw replies object passed by Flower.

            Returns:
                Whatever the parent strategy's ``aggregate_train`` returns.
            """
            if self._shap_runtime is not None and (
                server_round % max(1, self._shap_every_n_rounds) == 0
            ):
                try:
                    # Debug: inspect reply shape
                    try:
                        length = len(replies)
                    except Exception:
                        length = None
                    logging.debug(
                        "SHAP mixin received replies type=%s, len=%s",
                        type(replies),
                        length,
                    )

                    # Normalize replies to a list of result/message objects
                    results_list = None
                    # Case: strategies sometimes pass (results, failures)
                    if (
                        isinstance(replies, (list, tuple))
                        and len(replies) == 2
                        and isinstance(replies[0], (list, tuple))
                    ):
                        results_list = replies[0]
                    elif isinstance(replies, (list, tuple)):
                        results_list = list(replies)
                    else:
                        results_list = [replies]

                    models_dict = {}
                    reply_entries = []
                    for idx, item in enumerate(results_list):
                        # Unwrap common tuple shapes (client_proxy, result) or (result,)
                        client_proxy = None
                        msg = item
                        if isinstance(item, tuple) and len(item) >= 2:
                            client_proxy = item[0]
                            msg = item[1]

                        # Attempt to extract content from various wrappers
                        content = None
                        if hasattr(msg, "message"):
                            try:
                                content = msg.message.content
                            except Exception:
                                content = getattr(msg.message, "content", None)
                        if content is None and hasattr(msg, "content"):
                            content = msg.content
                        if content is None and hasattr(msg, "payload"):
                            content = msg.payload
                        if content is None:
                            # Last resort: if msg itself looks like an ArrayRecord
                            content = msg

                        # Log content type at debug level
                        logging.debug("reply[%s] content type=%s", idx, type(content))

                        # Extract ArrayRecord/config from RecordDict or other wrappers
                        arrays = None
                        config = None
                        try:
                            # Mapping-like objects (RecordDict implement .get)
                            if hasattr(content, "get"):
                                arrays = (
                                    content.get("arrays")
                                    or content.get("Arrays")
                                    or content.get("parameters")
                                )
                                config = content.get("config") or content.get("Config")
                            else:
                                # Try item access
                                try:
                                    arrays = content["arrays"]
                                except Exception:
                                    arrays = None
                                try:
                                    config = content["config"]
                                except Exception:
                                    config = None
                        except Exception:
                            arrays = getattr(content, "arrays", None) or getattr(
                                content, "parameters", None
                            )
                            config = getattr(content, "config", None)

                        if arrays is not None:
                            logging.debug("reply[%s] arrays type=%s", idx, type(arrays))
                        else:
                            logging.debug("reply[%s] has no arrays, skipping", idx)
                            continue

                        # Convert ArrayRecord-like objects to torch state dict
                        state = None
                        try:
                            if hasattr(arrays, "to_torch_state_dict"):
                                state = arrays.to_torch_state_dict()
                            elif hasattr(arrays, "to_parameters"):
                                # Try converting parameters to state dict if available
                                params = arrays.to_parameters()
                                state = params.to_torch_state_dict()
                            else:
                                # Try attribute access
                                state = arrays.to_torch_state_dict()
                        except Exception as exc:
                            logging.warning(
                                "failed to convert arrays to state_dict: %s", exc
                            )
                            continue

                        # Build a model instance and load the state
                        model = get_model(
                            DATASET_CONFIG.input_dim,
                            DATASET_CONFIG.task,
                            DATASET_CONFIG.model_config,
                        )
                        model.load_state_dict(state)

                        # Try to infer partition id from client proxy, then config, then fallback to index
                        partition_id = None
                        try:
                            # If we have a client proxy, try known id attributes
                            if client_proxy is not None:
                                for attr in (
                                    "cid",
                                    "client_id",
                                    "client_id_str",
                                    "node_id",
                                    "node_id_str",
                                ):
                                    if hasattr(client_proxy, attr):
                                        partition_id = str(getattr(client_proxy, attr))
                                        break
                            if partition_id is None and config is not None:
                                if hasattr(config, "to_dict"):
                                    cfgdict = config.to_dict()
                                else:
                                    cfgdict = dict(config)
                                partition_id = str(
                                    cfgdict.get("partition-id")
                                    or cfgdict.get("partition_id")
                                    or cfgdict.get("partitionId")
                                    or None
                                )
                        except Exception:
                            partition_id = None

                        if partition_id is None:
                            # Use the enumerated index (0-based) so save._format_partition_id will display 1-based
                            partition_id = str(idx)

                        model_name = f"client_{partition_id}"
                        models_dict[model_name] = model
                        reply_entries.append({"reply": item, "model_name": model_name})

                    if models_dict:
                        shap_result = compute_and_save_round_shap(
                            models_dict=models_dict,
                            runtime=self._shap_runtime,
                            dataset_name=self._shap_dataset_name or dataset_name,
                            round_number=server_round,
                            attack_config=self._shap_attack_config,
                            num_clients=self._shap_num_clients,
                            return_values=True,
                        )

                        metadata_items, shap_values_by_client = shap_result

                        flagged_client_names = set()

                        if (
                            self._detection_config is not None
                            and GLOBAL_REFERENCE_SHAP is not None
                            and getattr(self._detection_config, "enabled", False)
                        ):
                            detection_mode = getattr(
                                self._detection_config, "mode", "mad-shapcosim"
                            )
                            # Use hybrid or simple cosine-based detection only (G-ShapCosim removed)
                            # Use MAD-SHAPCOSIM detector only
                            detection_result = detect_mad_shapcosim_outliers(
                                client_vectors=shap_values_by_client,
                                server_reference=GLOBAL_REFERENCE_SHAP,
                                alpha=getattr(self._detection_config, "alpha", 0.7),
                                mad_multiplier=getattr(
                                    self._detection_config, "mad_multiplier", 3.0
                                ),
                                min_clients=getattr(
                                    self._detection_config, "min_clients", 3
                                ),
                            )
                            # Collect flagged clients from detection result (both methods)
                            flagged_client_names = set(detection_result.flagged_clients)
                            # Persist flagged clients for this round so evaluation aggregation
                            # can filter the same clients when aggregate_evaluate is called.
                            try:
                                self._flagged_clients_for_round[int(server_round)] = (
                                    set(flagged_client_names)
                                )
                            except Exception:
                                pass
                            logging.info(
                                "Detection round=%s method=%s alpha=%.3f threshold=%.6f bounds=[%.6f, %.6f] flagged=%s",
                                server_round,
                                detection_result.detection_method,
                                getattr(self._detection_config, "alpha", 0.7),
                                detection_result.mad_threshold
                                if detection_result.mad_threshold is not None
                                else detection_result.lower_bound,
                                detection_result.lower_bound,
                                detection_result.upper_bound,
                                sorted(detection_result.flagged_clients),
                            )
                            # Build ground-truth map from metadata_items when available
                            true_malicious_map: dict = {}
                            try:
                                for md in metadata_items:
                                    # metadata.client_identifier matches model_name
                                    true_malicious_map[md.client_identifier] = (
                                        bool(md.malicious)
                                        if md.malicious is not None
                                        else None
                                    )
                            except Exception:
                                true_malicious_map = {}

                            # Compute confusion matrix for this round
                            tp = fp = fn = tn = 0
                            for model_name in sorted(detection_result.scores):
                                detected = model_name in flagged_client_names
                                true_label = true_malicious_map.get(model_name)
                                if true_label is None:
                                    # Unknown ground truth: skip from metrics
                                    continue
                                if detected and true_label:
                                    tp += 1
                                elif detected and not true_label:
                                    fp += 1
                                elif (not detected) and true_label:
                                    fn += 1
                                else:
                                    tn += 1

                            # Persist per-round and cumulative
                            try:
                                DETECTION_TRACKER["per_round"].append(
                                    {
                                        "round": int(server_round),
                                        "TP": tp,
                                        "FP": fp,
                                        "FN": fn,
                                        "TN": tn,
                                    }
                                )
                                DETECTION_TRACKER["cumulative"]["TP"] += tp
                                DETECTION_TRACKER["cumulative"]["FP"] += fp
                                DETECTION_TRACKER["cumulative"]["FN"] += fn
                                DETECTION_TRACKER["cumulative"]["TN"] += tn
                            except Exception:
                                pass
                            logging.info(
                                "Detection summary round=%s TP=%d FP=%d FN=%d TN=%d cumulative=%s",
                                server_round,
                                tp,
                                fp,
                                fn,
                                tn,
                                DETECTION_TRACKER.get("cumulative"),
                            )
                            cumulative = DETECTION_TRACKER.get("cumulative", {})
                            print(
                                f"[Detection] Round {int(server_round)}: TP={tp} FP={fp} FN={fn} TN={tn}"
                            )
                            print(
                                "[Detection] Cumulative: "
                                f"TP={int(cumulative.get('TP', 0))} "
                                f"FP={int(cumulative.get('FP', 0))} "
                                f"FN={int(cumulative.get('FN', 0))} "
                                f"TN={int(cumulative.get('TN', 0))}"
                            )
                            # Log detection metrics to W&B if available
                            if run is not None:
                                try:
                                    run.log(
                                        {
                                            "round": int(server_round),
                                            "detection/TP": int(tp),
                                            "detection/FP": int(fp),
                                            "detection/FN": int(fn),
                                            "detection/TN": int(tn),
                                            "detection/cumulative/TP": int(
                                                DETECTION_TRACKER["cumulative"]["TP"]
                                            ),
                                            "detection/cumulative/FP": int(
                                                DETECTION_TRACKER["cumulative"]["FP"]
                                            ),
                                            "detection/cumulative/FN": int(
                                                DETECTION_TRACKER["cumulative"]["FN"]
                                            ),
                                            "detection/cumulative/TN": int(
                                                DETECTION_TRACKER["cumulative"]["TN"]
                                            ),
                                        },
                                        step=int(server_round),
                                    )
                                except Exception:
                                    pass
                            for model_name in sorted(detection_result.scores):
                                final_score = detection_result.scores.get(
                                    model_name, 0.0
                                )
                                server_score = detection_result.server_scores.get(
                                    model_name, 0.0
                                )
                                pairwise_score = detection_result.pairwise_scores.get(
                                    model_name, 0.0
                                )
                                z_server_score = detection_result.z_server_scores.get(
                                    model_name, 0.0
                                )
                                z_pairwise_score = (
                                    detection_result.z_pairwise_scores.get(
                                        model_name, 0.0
                                    )
                                )
                                threshold = (
                                    detection_result.mad_threshold
                                    if detection_result.mad_threshold is not None
                                    else detection_result.lower_bound
                                )
                                verdict = (
                                    "FLAGGED"
                                    if model_name in flagged_client_names
                                    else "KEPT"
                                )
                                logging.info(
                                    "Detection details round=%s client=%s server=%.6f pairwise=%.6f z_server=%.6f z_pairwise=%.6f final=%.6f threshold=%.6f verdict=%s",
                                    server_round,
                                    model_name,
                                    server_score,
                                    pairwise_score,
                                    z_server_score,
                                    z_pairwise_score,
                                    final_score,
                                    threshold,
                                    verdict,
                                )
                            if flagged_client_names:
                                replies = [
                                    entry["reply"]
                                    for entry in reply_entries
                                    if entry["model_name"] not in flagged_client_names
                                ]
                                logging.info(
                                    "Excluded %d client replies from aggregation",
                                    len(flagged_client_names),
                                )
                except Exception as exc:
                    logging.exception(
                        "SHAP computation failed during aggregate_train: %s", exc
                    )

            # Delegate to parent aggregation
            return super().aggregate_train(server_round, replies)

        def _extract_model_name_from_result(self, item, idx=0):
            """Return a consistent client model name for an evaluation reply.

            The method tries common locations for a client identifier (proxy
            attributes or a ``config`` mapping). If none found, the provided
            ``idx`` is used as the identifier.

            Args:
                item: A reply item, possibly (client_proxy, msg).
                idx: Integer fallback index to use when no id is present.

            Returns:
                A string like ``client_{partition_id}``.
            """
            client_proxy = None
            msg = item
            if isinstance(item, tuple) and len(item) >= 2:
                client_proxy = item[0]
                msg = item[1]

            content = None
            if hasattr(msg, "message"):
                try:
                    content = msg.message.content
                except Exception:
                    content = getattr(msg.message, "content", None)
            if content is None and hasattr(msg, "content"):
                content = msg.content
            if content is None and hasattr(msg, "payload"):
                content = msg.payload
            if content is None:
                content = msg

            # Try to pull partition id from client proxy or config
            partition_id = None
            try:
                if client_proxy is not None:
                    for attr in (
                        "cid",
                        "client_id",
                        "client_id_str",
                        "node_id",
                        "node_id_str",
                    ):
                        if hasattr(client_proxy, attr):
                            partition_id = str(getattr(client_proxy, attr))
                            break
                if partition_id is None and content is not None:
                    cfg = None
                    if hasattr(content, "get"):
                        cfg = content.get("config") or content.get("Config")
                    else:
                        try:
                            cfg = content["config"]
                        except Exception:
                            cfg = getattr(content, "config", None)
                    if cfg is not None:
                        if hasattr(cfg, "to_dict"):
                            cfgdict = cfg.to_dict()
                        else:
                            try:
                                cfgdict = dict(cfg)
                            except Exception:
                                cfgdict = {}
                        partition_id = str(
                            cfgdict.get("partition-id")
                            or cfgdict.get("partition_id")
                            or cfgdict.get("partitionId")
                            or None
                        )
            except Exception:
                partition_id = None

            if partition_id is None:
                partition_id = str(idx)

            return f"client_{partition_id}"

        def aggregate_evaluate(self, server_round: int, results, failures=None):
            """Filter evaluation replies using previously flagged client list.

            If the detection step flagged clients in ``aggregate_train``, their
            evaluation replies for the same round will be removed so the
            parent strategy aggregates only non-flagged clients.

            Args:
                server_round: Current server round number.
                results: Evaluation replies passed by Flower.
                failures: Optional failures structure forwarded to parent.

            Returns:
                The result of the parent strategy's ``aggregate_evaluate``.
            """
            flagged = (
                self._flagged_clients_for_round.get(int(server_round))
                if self._flagged_clients_for_round is not None
                else None
            )
            if not flagged:
                # No filtering needed, delegate to parent
                try:
                    if failures is None:
                        return super().aggregate_evaluate(server_round, results)
                    else:
                        return super().aggregate_evaluate(
                            server_round, results, failures
                        )
                except Exception:
                    # Fallback: try calling without failures
                    return super().aggregate_evaluate(server_round, results)

            # Normalize results list
            results_list = None
            if (
                isinstance(results, (list, tuple))
                and len(results) == 2
                and isinstance(results[0], (list, tuple))
            ):
                results_list = results[0]
            elif isinstance(results, (list, tuple)):
                results_list = list(results)
            else:
                results_list = [results]

            filtered = []
            for idx, item in enumerate(results_list):
                try:
                    model_name = self._extract_model_name_from_result(item, idx)
                except Exception:
                    model_name = None
                if model_name is None or model_name not in flagged:
                    filtered.append(item)

            try:
                return super().aggregate_evaluate(server_round, filtered)
            except Exception:
                # Some strategy implementations accept (server_round, results, failures)
                try:
                    return super().aggregate_evaluate(server_round, filtered, failures)
                except Exception:
                    return super().aggregate_evaluate(server_round, filtered)

    # Choose base strategy class
    base_strategy_cls = (
        FedAvg
        if strategy_lower == "fedavg"
        else (
            FedProx
            if strategy_lower == "fedprox"
            else (
                FedTrimmedAvg
                if strategy_lower == "fedtrimmedavg"
                else (
                    FedMedian
                    if strategy_lower == "fedmedian"
                    else (Krum if strategy_lower == "krum" else FedAvg)
                )
            )
        )
    )

    # Compose a strategy class that includes optional SHAP and FLDetector behavior
    strategy_mixins = []
    if strategy_lower == "fldetector":
        strategy_mixins.append(FLDetectorMixin)
    if shap_enabled:
        strategy_mixins.append(ShapStrategyMixin)

    if strategy_mixins:
        StrategyCls = type(
            f"{base_strategy_cls.__name__}With{'And'.join(mixin.__name__ for mixin in strategy_mixins)}",
            tuple(strategy_mixins + [base_strategy_cls]),
            {},
        )
    else:
        StrategyCls = base_strategy_cls

    # Instantiate strategy. Only pass SHAP-specific kwargs when shap is enabled
    strategy_kwargs = {}
    if strategy_lower == "fedprox":
        strategy_kwargs.update(
            fraction_evaluate=fraction_evaluate,
            proximal_mu=proximal_mu,
        )
    elif strategy_lower == "fedtrimmedavg":
        strategy_kwargs.update(
            fraction_evaluate=fraction_evaluate,
            beta=trimmed_mean_beta,
        )
    elif strategy_lower == "fedmedian":
        strategy_kwargs.update(
            fraction_evaluate=fraction_evaluate,
        )
    elif strategy_lower == "krum":
        strategy_kwargs.update(
            fraction_evaluate=fraction_evaluate,
        )
    else:  # Default to FedAvg
        strategy_kwargs.update(
            fraction_evaluate=fraction_evaluate,
        )

    if strategy_lower == "fldetector":
        strategy_kwargs.update(
            fraction_evaluate=fraction_evaluate,
            fldetector_window_size=context.run_config.get("fldetector-window-size", 10),
            fldetector_start_epoch=context.run_config.get("fldetector-start-epoch", 50),
            wandb_run=run,
            attack_config=attack_config,
            num_clients=num_clients,
            dataset_name=dataset_name,
            detection_config=detection_config,
        )

    if shap_enabled:
        strategy_kwargs.update(
            shap_runtime=shap_runtime,
            shap_every_n_rounds=context.run_config.get("shap-every-n-rounds", 1),
            attack_config=attack_config,
            detection_config=DETECTION_CONFIG,
            num_clients=num_clients,
            dataset_name=dataset_name,
            # Keep aggregation weights on standard FedAvg; detection only filters clients.
            use_shap_cosine_aggregation=False,
        )

    strategy = StrategyCls(**strategy_kwargs)

    # Ensure reference model training is complete before starting federated learning rounds
    if shap_enabled:
        if GLOBAL_REFERENCE_SHAP is not None:
            logging.info(
                "✓ Reference model training complete. Starting federated learning (Round 1)..."
            )
        else:
            logging.warning(
                "✗ Reference model training failed; proceeding with federated learning"
            )
    else:
        logging.info(
            "→ SHAP not enabled; skipping reference model. Starting federated learning (Round 1)..."
        )

    # Build train config: learning rate + all attack config parameters
    train_config_dict = {
        "lr": lr,
        **attack_config.to_dict(),  # Unpack all attack parameters
    }

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(train_config_dict),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )
    wandb.define_metric("round")
    wandb.define_metric("clients_agg/*", step_metric="round")
    if result.evaluate_metrics_clientapp and run is not None:
        for round_num, metrics in result.evaluate_metrics_clientapp.items():
            run.log(
                {
                    "round": round_num,
                    **{f"clients_agg/{k}": float(v) for k, v in metrics.items()},
                }
            )
    # Save and log final model as artifact
    torch.save(result.arrays.to_torch_state_dict(), "final_model.pt")
    artifact = wandb.Artifact(name="federated-lr-model", type="model")
    artifact.add_file("final_model.pt")
    run.log_artifact(artifact)

    # Generate and upload SHAP artifacts to existing W&B run if enabled (opt-in via run-config)
    if (
        shap_runtime is not None
        and generate_plots_for_run is not None
        and context.run_config.get("shap-generate-plots", False)
    ):
        try:
            logging.info("Generating SHAP plots for run: %s", shap_runtime.run_name)
            plot_paths = generate_plots_for_run(
                run_name=shap_runtime.run_name,
                output_root=shap_runtime.output_root,
                source="local",
            )
            logging.info("Generated %d plot sets from SHAP artifacts", len(plot_paths))

            # Upload SHAP artifacts directory to W&B
            shap_artifact = wandb.Artifact(
                name=f"shap-artifacts-{shap_runtime.run_name}",
                type="shap",
                description="SHAP values, metadata, and plots for federated learning run",
            )
            shap_run_dir = Path(shap_runtime.output_root) / shap_runtime.run_name
            logging.info("Uploading SHAP artifacts from: %s", shap_run_dir)
            shap_artifact.add_dir(str(shap_run_dir))
            run.log_artifact(shap_artifact)
            logging.info("SHAP artifacts uploaded to W&B")
        except Exception as exc:
            logging.exception("Failed to generate and upload SHAP artifacts: %s", exc)

    run.finish()


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate global model on centralized test set.

    This function executes after each federation round on the server:
    1. Loads the full centralized test dataset
    2. Receives the aggregated global model weights
    3. Evaluates model on test data
    4. Logs task-specific metrics to W&B
    5. Returns results for FedAvg strategy

    Args:
        server_round: Current round number in the federated learning loop
        arrays: Model weights (as ArrayRecord) to evaluate

    Returns:
        MetricRecord with task-specific evaluation metrics
    """

    _, testloader, _ = load_centralized_dataset(dataset_name=DATASET_NAME)

    input_dim = DATASET_CONFIG.input_dim
    model = get_model(input_dim, DATASET_CONFIG.task, DATASET_CONFIG.model_config)
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    results = test(model, testloader, device, task=DATASET_CONFIG.task)

    metrics_dict = {"round": server_round}

    if DATASET_CONFIG.task == "classification":
        loss, accuracy, auc_roc, f1, f1_macro, precision, recall = results
        metrics_dict.update(
            {
                "loss": float(loss),
                "accuracy": float(accuracy),
                "auc_roc": float(auc_roc),
                "f1": float(f1),
                "f1_macro": float(f1_macro),
                "precision": float(precision),
                "recall": float(recall),
            }
        )
        eval_metrics = {
            "loss": float(loss),
            "accuracy": float(accuracy),
            "auc_roc": float(auc_roc),
            "f1": float(f1),
            "f1_macro": float(f1_macro),
            "precision": float(precision),
            "recall": float(recall),
        }
    else:
        mse, mae, rmse, r_squared = results
        metrics_dict.update(
            {
                "mse": float(mse),
                "mae": float(mae),
                "rmse": float(rmse),
                "r_squared": float(r_squared),
            }
        )
        eval_metrics = {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "r_squared": float(r_squared),
        }

    if run is not None:
        run.log(metrics_dict, step=server_round)

    return MetricRecord(eval_metrics)
