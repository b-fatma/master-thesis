"""FLDetector-style aggregation support for Flower server strategies.

This module adapts the FLPoison FLDetector idea to the Flower server app used
in this repository. It detects suspicious clients from a window of update
distances and filters them before delegating aggregation to the wrapped
strategy (typically FedAvg).
"""

from __future__ import annotations

from copy import deepcopy
import logging

import numpy as np
from sklearn.cluster import KMeans

from src.attacks import should_be_malicious


logger = logging.getLogger(__name__)


def _normalize_data(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    min_val = float(np.min(data))
    max_val = float(np.max(data))
    if abs(max_val - min_val) < 1e-12:
        return np.zeros_like(data, dtype=float)
    return (data - min_val) / (max_val - min_val)


def _flatten_state_dict(
    state_dict,
) -> tuple[np.ndarray, list[tuple[str, tuple[int, ...]]]]:
    flat_parts: list[np.ndarray] = []
    layout: list[tuple[str, tuple[int, ...]]] = []
    for name, tensor in state_dict.items():
        value = (
            tensor.detach().cpu().numpy()
            if hasattr(tensor, "detach")
            else np.asarray(tensor)
        )
        value = np.asarray(value, dtype=float)
        layout.append((name, tuple(value.shape)))
        flat_parts.append(value.reshape(-1))
    if not flat_parts:
        return np.array([], dtype=float), layout
    return np.concatenate(flat_parts).astype(float), layout


def _weighted_average(
    vectors: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    if vectors.size == 0:
        return np.array([], dtype=float)
    if weights is None:
        return np.mean(vectors, axis=0)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    weights = weights / np.maximum(np.sum(weights), 1e-12)
    return np.sum(vectors * weights[:, None], axis=0)


class FLDetectorMixin:
    """Mixin that filters suspicious clients using FLDetector-style scoring."""

    def __init__(self, *args, **kwargs):
        window_size = kwargs.pop("fldetector_window_size", 10)
        start_epoch = kwargs.pop("fldetector_start_epoch", 50)
        wandb_run = kwargs.pop("wandb_run", None)
        attack_config = kwargs.pop("attack_config", None)
        num_clients = kwargs.pop("num_clients", 0)
        dataset_name = kwargs.pop("dataset_name", None)
        detection_config = kwargs.pop("detection_config", None)
        super().__init__(*args, **kwargs)
        self.window_size = int(window_size)
        self.start_epoch = int(start_epoch)
        self._wandb_run = wandb_run
        self._attack_config = attack_config
        self._num_clients = int(num_clients)
        self._dataset_name = dataset_name
        self._detection_config = detection_config
        self.algorithm = "FedOpt"
        self.global_weight_diffs: list[np.ndarray] = []
        self.global_grad_diffs: list[np.ndarray] = []
        self.last_global_grad: np.ndarray | None = None
        self.last_grad_updates: np.ndarray | None = None
        self.malicious_score: list[np.ndarray] = []
        self.init_model = None
        self._last_global_vector: np.ndarray | None = None
        self._state_layout: list[tuple[str, tuple[int, ...]]] | None = None
        self._detection_tracker = {
            "per_round": [],
            "cumulative": {"TP": 0, "FP": 0, "FN": 0, "TN": 0},
        }

    def aggregate_train(self, server_round: int, replies):
        normalized = self._normalize_replies(replies)
        if not normalized:
            return super().aggregate_train(server_round, replies)

        current_epoch = int(server_round)
        if current_epoch <= self.start_epoch:
            self.init_model = deepcopy(getattr(self, "_last_global_vector", None))

        client_vectors, reply_entries, num_examples = self._extract_client_vectors(
            normalized
        )
        if not client_vectors.size:
            return super().aggregate_train(server_round, replies)

        if self._state_layout is None and reply_entries:
            self._state_layout = reply_entries[0]["layout"]

        benign_idx = np.arange(len(client_vectors))
        if self._last_global_vector is not None:
            gradient_updates = client_vectors - self._last_global_vector
            if current_epoch - self.start_epoch > self.window_size:
                hvp = self.LBFGS(
                    self.global_weight_diffs,
                    self.global_grad_diffs,
                    self.last_global_grad,
                )
                distance = self.get_pred_real_dists(
                    self.last_grad_updates, gradient_updates, hvp
                )
                self.malicious_score.append(distance)

            if len(self.malicious_score) > self.window_size:
                malicious_score = np.stack(
                    self.malicious_score[-self.window_size :], axis=0
                )
                score = np.mean(malicious_score, axis=0)
                if (
                    self.gap_statistics(
                        score, num_sampling=20, K_max=10, n=len(client_vectors)
                    )
                    >= 2
                ):
                    estimator = KMeans(n_clusters=2, n_init=10)
                    estimator.fit(np.reshape(score, (score.shape[0], -1)))
                    label_pred = estimator.labels_
                    benign_label = (
                        1
                        if np.mean(score[label_pred == 0])
                        > np.mean(score[label_pred == 1])
                        else 0
                    )
                    benign_idx = np.argwhere(label_pred == benign_label).squeeze()
                    logger.info("FLDetector Defense: benign idx=%s", benign_idx)

        benign_idx = np.atleast_1d(benign_idx).astype(int)
        flagged_idx = sorted(set(range(len(reply_entries))) - set(benign_idx.tolist()))
        filtered_replies = [
            entry["reply"]
            for i, entry in enumerate(reply_entries)
            if i not in flagged_idx
        ]
        if not filtered_replies:
            filtered_replies = [entry["reply"] for entry in reply_entries]

        filtered_vectors = client_vectors[benign_idx]
        filtered_weights = num_examples[benign_idx]
        agg_vector = _weighted_average(filtered_vectors, filtered_weights)
        agg_update = (
            agg_vector - self._last_global_vector
            if self._last_global_vector is not None
            else agg_vector
        )

        self.global_weight_diffs.append(agg_update)
        if self.last_global_grad is None:
            self.last_global_grad = np.zeros_like(agg_update)
        self.global_grad_diffs.append(agg_update - self.last_global_grad)
        if len(self.global_weight_diffs) > self.window_size:
            del self.global_weight_diffs[0]
            del self.global_grad_diffs[0]
        self.last_global_grad = agg_update
        self.last_grad_updates = client_vectors
        self._last_global_vector = agg_vector

        self._track_detection_round(server_round, reply_entries, flagged_idx)

        return super().aggregate_train(server_round, filtered_replies)

    def _track_detection_round(
        self, server_round: int, reply_entries: list[dict], flagged_idx: list[int]
    ) -> None:
        attack_config = getattr(self, "_attack_config", None) or getattr(
            self, "_shap_attack_config", None
        )
        num_clients = int(
            getattr(self, "_num_clients", 0)
            or getattr(self, "_shap_num_clients", 0)
            or len(reply_entries)
        )
        true_malicious_map: dict[str, bool] = {}

        # If attack_config exists and indicates attacks enabled, compute deterministic
        # malicious labels via should_be_malicious. Otherwise, mark all clients as clean
        # (False) so detection metrics (TP/FP/FN/TN) can still be computed.
        if (
            attack_config is not None
            and getattr(attack_config, "enabled", False)
            and getattr(attack_config, "attack_type", "none") != "none"
        ):
            for entry in reply_entries:
                partition_id = entry.get("partition_id", "client_0")
                try:
                    partition_int = int(str(partition_id).replace("client_", ""))
                except Exception:
                    partition_int = None
                if partition_int is None:
                    # Unknown id -> treat as clean to avoid dropping metrics
                    true_malicious_map[partition_id] = False
                else:
                    true_malicious_map[partition_id] = bool(
                        should_be_malicious(
                            partition_int,
                            num_clients,
                            attack_config.malicious_ratio,
                            attack_config.seed,
                        )
                    )
        else:
            # No attack configured: mark all as clean (False)
            for entry in reply_entries:
                partition_id = entry.get("partition_id", "client_0")
                true_malicious_map[partition_id] = False

        flagged_names = {
            reply_entries[i]["partition_id"]
            for i in flagged_idx
            if i < len(reply_entries)
        }
        tp = fp = fn = tn = 0
        for entry in reply_entries:
            partition_id = entry["partition_id"]
            true_label = true_malicious_map.get(partition_id)
            if true_label is None:
                continue
            detected = partition_id in flagged_names
            if detected and true_label:
                tp += 1
            elif detected and not true_label:
                fp += 1
            elif (not detected) and true_label:
                fn += 1
            else:
                tn += 1

        self._detection_tracker["per_round"].append(
            {"round": int(server_round), "TP": tp, "FP": fp, "FN": fn, "TN": tn}
        )
        self._detection_tracker["cumulative"]["TP"] += tp
        self._detection_tracker["cumulative"]["FP"] += fp
        self._detection_tracker["cumulative"]["FN"] += fn
        self._detection_tracker["cumulative"]["TN"] += tn

        logger.info(
            "FLDetector round=%s TP=%d FP=%d FN=%d TN=%d cumulative=%s",
            server_round,
            tp,
            fp,
            fn,
            tn,
            self._detection_tracker["cumulative"],
        )

        if self._wandb_run is not None:
            try:
                self._wandb_run.log(
                    {
                        "round": int(server_round),
                        "detection/TP": int(tp),
                        "detection/FP": int(fp),
                        "detection/FN": int(fn),
                        "detection/TN": int(tn),
                        "detection/cumulative/TP": int(
                            self._detection_tracker["cumulative"]["TP"]
                        ),
                        "detection/cumulative/FP": int(
                            self._detection_tracker["cumulative"]["FP"]
                        ),
                        "detection/cumulative/FN": int(
                            self._detection_tracker["cumulative"]["FN"]
                        ),
                        "detection/cumulative/TN": int(
                            self._detection_tracker["cumulative"]["TN"]
                        ),
                    },
                    step=int(server_round),
                )
            except Exception:
                logger.debug(
                    "W&B logging failed for FLDetector round=%s",
                    server_round,
                    exc_info=True,
                )

    def _normalize_replies(self, replies):
        if (
            isinstance(replies, (list, tuple))
            and len(replies) == 2
            and isinstance(replies[0], (list, tuple))
        ):
            return list(replies[0])
        if isinstance(replies, (list, tuple)):
            return list(replies)
        return [replies]

    def _extract_client_vectors(self, replies):
        vectors: list[np.ndarray] = []
        entries: list[dict] = []
        num_examples: list[float] = []

        for idx, item in enumerate(replies):
            client_proxy = None
            msg = item
            if isinstance(item, tuple) and len(item) >= 2:
                client_proxy = item[0]
                msg = item[1]

            content = self._extract_message_content(msg)
            if content is None:
                continue

            arrays = None
            config = None
            if hasattr(content, "get"):
                arrays = (
                    content.get("arrays")
                    or content.get("Arrays")
                    or content.get("parameters")
                )
                config = content.get("config") or content.get("Config")
            else:
                try:
                    arrays = content["arrays"]
                except Exception:
                    arrays = None
                try:
                    config = content["config"]
                except Exception:
                    config = None

            if arrays is None:
                continue

            try:
                state_dict = (
                    arrays.to_torch_state_dict()
                    if hasattr(arrays, "to_torch_state_dict")
                    else arrays.to_torch_state_dict()
                )
            except Exception:
                continue

            flat, layout = _flatten_state_dict(state_dict)
            vectors.append(flat)

            partition_id = self._extract_partition_id(client_proxy, config, idx)
            entries.append(
                {"reply": item, "layout": layout, "partition_id": partition_id}
            )
            num_examples.append(self._extract_num_examples(content) or 1.0)

        if not vectors:
            return np.empty((0, 0), dtype=float), [], np.empty((0,), dtype=float)

        return np.stack(vectors, axis=0), entries, np.asarray(num_examples, dtype=float)

    def _extract_message_content(self, msg):
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

    def _extract_num_examples(self, content) -> float | None:
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

    def _extract_partition_id(self, client_proxy, config, idx: int) -> str:
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
            partition_id = str(idx)
        return f"client_{partition_id}"

    def get_pred_real_dists(self, last_grad_updates, gradient_updates, hvp):
        if last_grad_updates is None:
            return np.zeros(gradient_updates.shape[0], dtype=float)
        pred_grad = last_grad_updates + hvp
        distance = np.linalg.norm(pred_grad - gradient_updates, axis=1)
        distance = distance / np.maximum(np.sum(distance), 1e-12)
        return distance

    def LBFGS(self, S_k_list, Y_k_list, v):
        if not S_k_list or not Y_k_list or v is None:
            return np.zeros_like(v if v is not None else np.array([], dtype=float))

        S_k_list = [np.asarray(i).reshape(-1, 1) for i in S_k_list]
        Y_k_list = [np.asarray(i).reshape(-1, 1) for i in Y_k_list]
        v = np.asarray(v).reshape(-1, 1)

        curr_S_k = np.concatenate(S_k_list, axis=1)
        curr_Y_k = np.concatenate(Y_k_list, axis=1)
        S_k_time_Y_k = np.matmul(curr_S_k.T, curr_Y_k)
        S_k_time_S_k = np.matmul(curr_S_k.T, curr_S_k)

        R_k = np.triu(S_k_time_Y_k)
        L_k = S_k_time_Y_k - np.array(R_k)
        sigma_k = np.matmul(Y_k_list[-1].T, S_k_list[-1]) / (
            np.matmul(S_k_list[-1].T, S_k_list[-1])
        )
        D_k_diag = np.diag(S_k_time_Y_k)
        upper_mat = np.concatenate([sigma_k * S_k_time_S_k, L_k], axis=1)
        lower_mat = np.concatenate([L_k.T, -np.diag(D_k_diag)], axis=1)
        mat = np.concatenate([upper_mat, lower_mat], axis=0)
        mat_inv = np.linalg.inv(mat)

        approx_prod = sigma_k * v
        p_mat = np.concatenate(
            [
                np.matmul(curr_S_k.T, sigma_k * v),
                np.matmul(curr_Y_k.T, v),
            ],
            axis=0,
        )
        approx_prod -= np.matmul(
            np.matmul(np.concatenate([sigma_k * curr_S_k, curr_Y_k], axis=1), mat_inv),
            p_mat,
        )
        return approx_prod.squeeze()

    def gap_statistics(self, data, num_sampling, K_max, n):
        data = _normalize_data(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        gaps, s = [], []
        K_max = min(K_max, data.shape[0])

        for k in range(1, K_max + 1):
            kmeans = KMeans(n_clusters=k, n_init=10).fit(data)
            inertia = kmeans.inertia_

            fake_inertia = []
            for _ in range(num_sampling):
                random_data = np.random.rand(n, data.shape[1])
                kmeans_fake = KMeans(n_clusters=k, n_init=10).fit(random_data)
                fake_inertia.append(kmeans_fake.inertia_)

            mean_fake_inertia = np.mean(fake_inertia)
            gap = np.log(mean_fake_inertia) - np.log(inertia)
            gaps.append(gap)

            sd = np.std(np.log(fake_inertia))
            s.append(sd * np.sqrt((1 + num_sampling) / num_sampling))

        num_cluster = 0
        for k in range(1, K_max):
            if gaps[k - 1] - gaps[k] + s[k] >= 0:
                num_cluster = k + 1
                break
        else:
            num_cluster = K_max
            logger.info("FLDetector: No gap detected, returning K_max=%s", K_max)
        return num_cluster

    @property
    def detection_tracker(self):
        return self._detection_tracker
