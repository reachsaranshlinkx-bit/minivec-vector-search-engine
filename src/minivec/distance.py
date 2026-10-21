"""Distance and similarity metrics for MiniVec.

Implements exact vector similarity and distance calculations using NumPy:
1. Cosine similarity: Normalized dot product, measuring angle regardless of magnitude.
2. Dot product: Inner product, influenced by both angle and magnitude.
3. Euclidean distance: Straight-line L2 geometric distance in Euclidean space.
"""

from enum import Enum
import numpy as np


class DistanceMetric(str, Enum):
    """Supported distance and similarity metrics."""

    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"


def cosine_similarity(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a 1D query and 2D candidate vectors.

    Formula:
        cos(q, c) = (q · c) / (||q|| * ||c||)

    Special handling for zero vectors:
        If ||q|| == 0 or ||c|| == 0, cosine similarity is defined as 0.0 to
        avoid division by zero, NaN, or infinity.

    Args:
        query: 1D NumPy array of shape (D,).
        candidates: 2D NumPy array of shape (N, D).

    Returns:
        1D NumPy array of shape (N,) with similarity scores in [-1.0, 1.0].
    """
    dots = np.dot(candidates, query)
    q_norm = np.linalg.norm(query)
    c_norms = np.linalg.norm(candidates, axis=1)

    denominators = q_norm * c_norms
    zero_mask = np.isclose(denominators, 0.0)

    scores = np.zeros_like(dots, dtype=np.float32)
    np.divide(dots, denominators, out=scores, where=~zero_mask)

    # Numerical clipping guards against float32 round-off exceeding [-1.0, 1.0]
    return np.clip(scores, -1.0, 1.0)


def dot_product(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Compute dot product between a 1D query and 2D candidate vectors.

    Formula:
        dot(q, c) = q · c = sum(q_i * c_i)

    Args:
        query: 1D NumPy array of shape (D,).
        candidates: 2D NumPy array of shape (N, D).

    Returns:
        1D NumPy array of shape (N,) with dot product scores.
    """
    return np.dot(candidates, query).astype(np.float32)


def euclidean_distance(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Compute Euclidean (L2) distance between a 1D query and 2D candidate vectors.

    Formula:
        dist(q, c) = sqrt(sum((q_i - c_i)^2))

    Args:
        query: 1D NumPy array of shape (D,).
        candidates: 2D NumPy array of shape (N, D).

    Returns:
        1D NumPy array of shape (N,) with non-negative distance values.
    """
    diff = candidates - query
    return np.linalg.norm(diff, axis=1).astype(np.float32)


def compute_scores(
    query: np.ndarray,
    candidates: np.ndarray,
    metric: DistanceMetric | str,
) -> tuple[np.ndarray, bool]:
    """Calculate scores for candidate vectors against a query for a given metric.

    Args:
        query: 1D NumPy array of shape (D,).
        candidates: 2D NumPy array of shape (N, D).
        metric: Metric to compute (cosine, dot, or euclidean).

    Returns:
        A tuple of (scores_array, is_descending):
            - scores_array: 1D NumPy array of scores or distances.
            - is_descending: True if higher is better (cosine, dot),
              False if lower is better (euclidean).
    """
    normalized_metric = DistanceMetric(metric)

    if normalized_metric == DistanceMetric.COSINE:
        return cosine_similarity(query, candidates), True

    if normalized_metric == DistanceMetric.DOT:
        return dot_product(query, candidates), True

    if normalized_metric == DistanceMetric.EUCLIDEAN:
        return euclidean_distance(query, candidates), False

    raise ValueError(f"Unsupported metric: {metric}")
