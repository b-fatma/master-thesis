"""Cosine-based outlier detection helpers for federated learning."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger().setLevel(logging.INFO)
import logging
from typing import Dict, Iterable

import numpy as np


def _format_score_map(scores: Dict[str, float]) -> str:
    """Format score dictionary for deterministic debug logging."""

    if not scores:
        return "{}"
    return (
        "{"
        + ", ".join(f"{key}={value:.6f}" for key, value in sorted(scores.items()))
        + "}"
    )


@dataclass(frozen=True)
class CosineDetectionResult:
    """Structured output for a round-level cosine detector.

    This dataclass captures outputs from the cosine-based detector
    implemented in this module (MAD-SHAPCOSIM). It contains per-client
    scores and the bounds/thresholds used for flagging.

    Attributes:
        scores: Score mapping (server or fused scores) per client.
        lower_bound: Lower threshold used for flagging (e.g., MAD lower bound or IQR).
        upper_bound: Upper threshold (unused for one-sided tests may be inf).
        flagged_clients: Tuple of client ids flagged as malicious.
        detection_method: Name of detection method ('mad-shapcosim').
        server_scores: Cosine similarity between client SHAP and server reference.
        pairwise_scores: Mean client-to-client cosine similarity.
        z_server_scores: Z-scored server similarities (when available).
        z_pairwise_scores: Z-scored pairwise similarities (when available).
        mad_threshold: Computed MAD threshold for hybrid detector (if any).
    """

    scores: Dict[str, float]
    lower_bound: float
    upper_bound: float
    flagged_clients: tuple[str, ...]
    detection_method: str = "mad-shapcosim"
    server_scores: Dict[str, float] = field(default_factory=dict)
    pairwise_scores: Dict[str, float] = field(default_factory=dict)
    z_server_scores: Dict[str, float] = field(default_factory=dict)
    z_pairwise_scores: Dict[str, float] = field(default_factory=dict)
    mad_threshold: float | None = None

    @property
    def excluded_clients(self) -> tuple[str, ...]:
        return self.flagged_clients


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity for 1-D vectors."""

    vec_a = np.asarray(a, dtype=float).reshape(-1)
    vec_b = np.asarray(b, dtype=float).reshape(-1)
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def iqr_bounds(values: Iterable[float], multiplier: float = 1.5) -> tuple[float, float]:
    """IQR helper removed — IQR-based detection is deprecated.

    Kept for compatibility but returns trivial bounds. The project no
    longer uses IQR-based thresholding; MAD-based `mad-shapcosim` is the
    supported detector.
    """

    return 0.0, 0.0


def pairwise_average_cosine(client_vectors: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Return mean client-to-client cosine similarity for each client."""

    client_ids = list(client_vectors.keys())
    count = len(client_ids)
    if count <= 1:
        return {client_id: 0.0 for client_id in client_ids}

    pairwise_sums = {client_id: 0.0 for client_id in client_ids}
    for idx_a, client_a in enumerate(client_ids):
        vec_a = client_vectors[client_a]
        for idx_b in range(idx_a + 1, count):
            client_b = client_ids[idx_b]
            sim = cosine_similarity(vec_a, client_vectors[client_b])
            pairwise_sums[client_a] += sim
            pairwise_sums[client_b] += sim

    return {
        client_id: float(pairwise_sums[client_id] / (count - 1))
        for client_id in client_ids
    }


def zscore_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Return standard z-score normalization for the given score map."""

    if not scores:
        return {}

    values = np.asarray(list(scores.values()), dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std < 1e-12:
        return {client_id: 0.0 for client_id in scores}
    return {
        client_id: float((value - mean) / std) for client_id, value in scores.items()
    }


def mad_lower_threshold(values: Iterable[float], multiplier: float = 3.0) -> float:
    """Return one-sided lower-tail MAD threshold: median - k * 1.4826 * MAD."""

    sample = np.asarray(list(values), dtype=float)
    if sample.size == 0:
        return 0.0

    median = float(np.median(sample))
    mad = float(np.median(np.abs(sample - median)))
    mad_scaled = max(1e-12, 1.4826 * mad)
    return median - multiplier * mad_scaled


def detect_cosine_outliers(*args, **kwargs) -> CosineDetectionResult:
    """IQR-based detector removed. Use `detect_mad_shapcosim_outliers` instead.

    Kept as a sentinel to provide a clear error if called by older code.
    """

    raise RuntimeError(
        "IQR-based detection has been removed. Use 'mad-shapcosim' detector."
    )


def detect_mad_shapcosim_outliers(
    client_vectors: Dict[str, np.ndarray],
    server_reference: np.ndarray,
    alpha: float = 0.7,
    mad_multiplier: float = 3.0,
    min_clients: int = 3,
) -> CosineDetectionResult:
    """MAD-SHAPCOSIM detector (fused server & pairwise signals with MAD).

    This detector z-scores the server-reference and pairwise cosine signals,
    fuses them with weight `alpha` on the server signal, and identifies
    low-scoring clients using a one-sided MAD threshold. This implements the
    project's canonical `mad-shapcosim` detection logic.

    Args:
        client_vectors: Mapping client_id -> 1-D numpy SHAP vector.
        server_reference: Reference SHAP vector from the server model.
        alpha: Weight on the server z-scored signal in the fused score.
        mad_multiplier: Multiplier for the MAD-based lower threshold.
        min_clients: Minimum clients required to run thresholding.

    Returns:
        A `CosineDetectionResult` containing fused scores, threshold and flagged clients.
    """

    server_scores = {
        client_id: cosine_similarity(vector, server_reference)
        for client_id, vector in client_vectors.items()
    }
    pairwise_scores = pairwise_average_cosine(client_vectors)
    logging.info(
        "[Detection][MAD-SHAPCOSIM] server_scores=%s", _format_score_map(server_scores)
    )
    logging.info(
        "[Detection][MAD-SHAPCOSIM] pairwise_scores=%s",
        _format_score_map(pairwise_scores),
    )

    if len(server_scores) < min_clients:
        logging.info(
            "[Detection][MAD-SHAPCOSIM] skipped thresholding because clients=%d < min_clients=%d",
            len(server_scores),
            min_clients,
        )
        return CosineDetectionResult(
            scores=server_scores,
            lower_bound=0.0,
            upper_bound=0.0,
            flagged_clients=tuple(),
            detection_method="mad-shapcosim",
            server_scores=server_scores,
            pairwise_scores=pairwise_scores,
        )

    z_server = zscore_normalize(server_scores)
    z_pairwise = zscore_normalize(pairwise_scores)
    logging.info("[Detection][MAD-SHAPCOSIM] z_server=%s", _format_score_map(z_server))
    logging.info(
        "[Detection][MAD-SHAPCOSIM] z_pairwise=%s", _format_score_map(z_pairwise)
    )

    fused_scores = {
        client_id: float(
            alpha * z_server[client_id] + (1.0 - alpha) * z_pairwise[client_id]
        )
        for client_id in server_scores
    }
    logging.info(
        "[Detection][MAD-SHAPCOSIM] fused_scores=%s alpha=%.3f beta=%.3f",
        _format_score_map(fused_scores),
        alpha,
        1.0 - alpha,
    )

    threshold = mad_lower_threshold(fused_scores.values(), multiplier=mad_multiplier)
    logging.info(
        "[Detection][MAD-SHAPCOSIM] mad_lower_threshold=%.6f mad_multiplier=%.3f",
        threshold,
        mad_multiplier,
    )
    flagged_clients = tuple(
        client_id for client_id, score in fused_scores.items() if score < threshold
    )

    for client_id in sorted(fused_scores):
        score = fused_scores[client_id]
        verdict = "FLAGGED" if score < threshold else "KEPT"
        logging.info(
            "[Detection][MAD-SHAPCOSIM] client=%s server=%.6f pairwise=%.6f z_server=%.6f z_pairwise=%.6f fused=%.6f threshold=%.6f verdict=%s",
            client_id,
            server_scores.get(client_id, 0.0),
            pairwise_scores.get(client_id, 0.0),
            z_server.get(client_id, 0.0),
            z_pairwise.get(client_id, 0.0),
            score,
            threshold,
            verdict,
        )

    return CosineDetectionResult(
        scores=fused_scores,
        lower_bound=threshold,
        upper_bound=float("inf"),
        flagged_clients=flagged_clients,
        detection_method="mad-shapcosim",
        server_scores=server_scores,
        pairwise_scores=pairwise_scores,
        z_server_scores=z_server,
        z_pairwise_scores=z_pairwise,
        mad_threshold=threshold,
    )
