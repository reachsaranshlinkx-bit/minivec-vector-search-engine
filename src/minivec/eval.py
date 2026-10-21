"""Evaluation harness and Recall@k calculation for MiniVec.

Compares approximate nearest-neighbor search (HNSW) against exact brute-force
search to measure empirical Recall@k, latencies, and speedup trade-offs.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Sequence
import numpy as np

from minivec.collection import Collection
from minivec.distance import DistanceMetric
from minivec.models import PointId, ScoredPoint
from minivec.validation import validate_metric


def calculate_recall_at_k(
    exact_results: Sequence[ScoredPoint],
    approx_results: Sequence[ScoredPoint],
    k: int,
) -> float:
    """Calculate Recall@k as the proportion of ground-truth results found by approximate search.

    Formula:
        Recall@k = |Exact_k ∩ Approx_k| / min(|Exact_k|, k)

    Edge case handling:
        When fewer than k ground-truth results exist (e.g. empty collection or small dataset),
        the denominator is the actual count of available ground-truth results, rather than k.
        If both exact and approximate result sets are empty (0 eligible results),
        recall is defined as 1.0 (perfect precision/recall on zero results).

    Args:
        exact_results: Ranked ground-truth results from exact flat search.
        approx_results: Ranked results from approximate search.
        k: Maximum rank cutoff.

    Returns:
        Float recall score in [0.0, 1.0].
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")

    exact_ids = {p.id for p in exact_results[:k]}
    approx_ids = {p.id for p in approx_results[:k]}

    denominator = len(exact_ids)

    # Edge case: No ground truth results exist
    if denominator == 0:
        return 1.0 if len(approx_ids) == 0 else 0.0

    intersection_count = len(exact_ids.intersection(approx_ids))
    return float(intersection_count) / float(denominator)


@dataclass
class EvaluationReport:
    """Benchmark results summarizing Recall@k and query latency."""

    metric: str
    num_points: int
    dimension: int
    num_queries: int
    k: int
    m: int
    ef_construction: int
    ef_search: int
    mean_recall: float
    min_recall: float
    max_recall: float
    mean_exact_latency_ms: float
    mean_approx_latency_ms: float
    speedup: float

    def summary(self) -> str:
        """Formatted human-readable benchmark summary."""
        return (
            f"Evaluation Report [Metric: {self.metric}]\n"
            f"  Points: {self.num_points:,} | Dimension: {self.dimension} | Queries: {self.num_queries}\n"
            f"  HNSW Config: m={self.m}, ef_construction={self.ef_construction}, ef_search={self.ef_search}\n"
            f"  Recall@{self.k}: mean={self.mean_recall * 100:.2f}% (min={self.min_recall * 100:.2f}%, max={self.max_recall * 100:.2f}%)\n"
            f"  Latency: Exact={self.mean_exact_latency_ms:.3f} ms vs Approx={self.mean_approx_latency_ms:.3f} ms\n"
            f"  Speedup: {self.speedup:.2f}x"
        )


def evaluate_recall(
    collection: Collection,
    queries: Sequence[np.ndarray | list[float]],
    k: int = 10,
    metric: DistanceMetric | str = DistanceMetric.COSINE,
    ef_search: int | None = None,
) -> EvaluationReport:
    """Run comparative evaluation of exact vs approximate search on a collection.

    Args:
        collection: The Collection instance containing populated vectors and an HNSW index.
        queries: Sequence of query vectors.
        k: Top-k ranking cutoff.
        metric: Distance metric to evaluate.
        ef_search: Optional override for approximate exploration size.

    Returns:
        EvaluationReport with measured recall, latencies, and speedup.
    """
    norm_metric = validate_metric(metric)
    index = collection.get_index(norm_metric)
    if index is None:
        raise ValueError(f"Collection '{collection.name}' does not have an index for metric '{norm_metric.value}'")

    recalls: list[float] = []
    exact_latencies: list[float] = []
    approx_latencies: list[float] = []

    for q in queries:
        # 1. Measure Exact Ground-Truth Search
        t0 = time.perf_counter()
        exact_hits = collection.search(query=q, limit=k, metric=norm_metric, search_type="exact")
        t1 = time.perf_counter()
        exact_latencies.append((t1 - t0) * 1000.0)

        # 2. Measure Approximate HNSW Search
        t2 = time.perf_counter()
        approx_hits = collection.search(
            query=q,
            limit=k,
            metric=norm_metric,
            search_type="approx",
            ef_search=ef_search,
        )
        t3 = time.perf_counter()
        approx_latencies.append((t3 - t2) * 1000.0)

        # 3. Calculate Recall@k
        recall = calculate_recall_at_k(exact_hits, approx_hits, k=k)
        recalls.append(recall)

    mean_recall = float(np.mean(recalls)) if recalls else 0.0
    min_recall = float(np.min(recalls)) if recalls else 0.0
    max_recall = float(np.max(recalls)) if recalls else 0.0
    mean_exact_ms = float(np.mean(exact_latencies)) if exact_latencies else 0.0
    mean_approx_ms = float(np.mean(approx_latencies)) if approx_latencies else 0.0
    speedup = (mean_exact_ms / mean_approx_ms) if mean_approx_ms > 0.0 else 1.0

    return EvaluationReport(
        metric=norm_metric.value,
        num_points=collection.count(),
        dimension=collection.dimension,
        num_queries=len(queries),
        k=k,
        m=index.m,
        ef_construction=index.ef_construction,
        ef_search=ef_search or index.ef_search,
        mean_recall=mean_recall,
        min_recall=min_recall,
        max_recall=max_recall,
        mean_exact_latency_ms=mean_exact_ms,
        mean_approx_latency_ms=mean_approx_ms,
        speedup=speedup,
    )


def generate_clustered_dataset(
    num_points: int = 500,
    dimension: int = 16,
    num_clusters: int = 5,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate deterministic synthetic vectors with cluster structure for benchmarking.

    Args:
        num_points: Total number of vectors to generate.
        dimension: Dimensionality of each vector.
        num_clusters: Number of Gaussian cluster centers.
        random_seed: Seed for reproducibility.

    Returns:
        Tuple of (points_matrix, queries_matrix):
            - points_matrix: shape (num_points, dimension)
            - queries_matrix: shape (20, dimension)
    """
    rng = np.random.default_rng(random_seed)
    centers = rng.standard_normal((num_clusters, dimension)).astype(np.float32)

    cluster_assignments = rng.integers(0, num_clusters, size=num_points)
    noise = rng.normal(0.0, 0.25, size=(num_points, dimension)).astype(np.float32)
    points = centers[cluster_assignments] + noise

    # Generate 20 query vectors sampled near cluster centers
    query_assignments = rng.integers(0, num_clusters, size=20)
    query_noise = rng.normal(0.0, 0.25, size=(20, dimension)).astype(np.float32)
    queries = centers[query_assignments] + query_noise

    return points, queries


def run_cli_evaluation() -> None:
    """Execute reproducible benchmarks for Cosine, Euclidean, and Dot metrics and print results."""
    print("=" * 76)
    print("MiniVec — Approximate Search Recall@k & Latency Evaluation")
    print("=" * 76)
    print("NOTE: Measurements reflect actual benchmark execution on this machine.")
    print("      HNSW provides probabilistic, empirical speedup, not worst-case O(log N).")
    print("=" * 76)

    metrics_to_eval = [
        (DistanceMetric.COSINE, 300, 16, 6, 42),
        (DistanceMetric.EUCLIDEAN, 250, 8, 4, 99),
        (DistanceMetric.DOT, 250, 8, 4, 123),
    ]

    for metric, n_pts, dim, n_clusters, seed in metrics_to_eval:
        points, queries = generate_clustered_dataset(
            num_points=n_pts,
            dimension=dim,
            num_clusters=n_clusters,
            random_seed=seed,
        )
        col = Collection(f"bench_{metric.value}", dimension=dim)
        col.create_index(
            metric=metric,
            m=16,
            ef_construction=64,
            ef_search=32,
            random_seed=seed,
        )
        for i, vec in enumerate(points):
            col.add_point(f"doc_{i}", vec)

        report = evaluate_recall(col, queries=queries, k=10, metric=metric)
        print(f"\n{report.summary()}")

    print("\n" + "=" * 76)
    print("Evaluation completed successfully.")
    print("=" * 76)


if __name__ == "__main__":
    run_cli_evaluation()
