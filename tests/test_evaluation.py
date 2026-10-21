"""Tests for Recall@k evaluation, filter fallback, persistence integration, and benchmarks."""

from pathlib import Path
import numpy as np
import pytest

from minivec import (
    Collection,
    DistanceMetric,
    FieldCondition,
    Filter,
    MiniVecEngine,
    Point,
    ScoredPoint,
    SQLiteStorage,
)
from minivec.eval import (
    calculate_recall_at_k,
    evaluate_recall,
    generate_clustered_dataset,
)


# =============================================================================
# 1. Recall@k Metric Edge Cases
# =============================================================================
class TestRecallAtKCalculation:
    """Verifies the mathematical correctness and edge cases of calculate_recall_at_k."""

    def test_perfect_recall(self):
        exact = [ScoredPoint(id=f"p_{i}", score=1.0) for i in range(5)]
        approx = [ScoredPoint(id=f"p_{i}", score=0.99) for i in range(5)]
        assert calculate_recall_at_k(exact, approx, k=5) == 1.0

    def test_zero_recall(self):
        exact = [ScoredPoint(id=f"exact_{i}", score=1.0) for i in range(5)]
        approx = [ScoredPoint(id=f"approx_{i}", score=0.9) for i in range(5)]
        assert calculate_recall_at_k(exact, approx, k=5) == 0.0

    def test_partial_recall(self):
        exact = [ScoredPoint(id=f"p_{i}", score=1.0) for i in range(5)]
        # 3 out of 5 matching
        approx = [
            ScoredPoint(id="p_0", score=1.0),
            ScoredPoint(id="p_1", score=0.9),
            ScoredPoint(id="p_2", score=0.8),
            ScoredPoint(id="diff_a", score=0.7),
            ScoredPoint(id="diff_b", score=0.6),
        ]
        assert np.isclose(calculate_recall_at_k(exact, approx, k=5), 3.0 / 5.0)

    def test_fewer_results_than_k_denominator_adjustment(self):
        """When fewer than k ground-truth results exist, denominator is len(exact), not k."""
        exact = [ScoredPoint(id="p_1", score=1.0), ScoredPoint(id="p_2", score=0.8)]
        approx = [ScoredPoint(id="p_1", score=1.0), ScoredPoint(id="p_2", score=0.8)]
        # Requested k=10, but only 2 exact results exist
        # If denominator were 10, recall would wrongly be 2/10 = 0.2
        # With correct denominator (2), recall is 2/2 = 1.0
        assert calculate_recall_at_k(exact, approx, k=10) == 1.0

    def test_zero_results_both_empty(self):
        """Both exact and approx returning 0 results yields 1.0 (perfect on 0 available)."""
        assert calculate_recall_at_k([], [], k=10) == 1.0

    def test_zero_results_exact_empty_approx_non_empty(self):
        approx = [ScoredPoint(id="p_phantom", score=0.5)]
        assert calculate_recall_at_k([], approx, k=10) == 0.0

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError, match="k must be a positive integer"):
            calculate_recall_at_k([], [], k=0)


# =============================================================================
# 2. Collection Integration & Filter Fallback Tests
# =============================================================================
class TestCollectionHNSWIntegration:
    """Verifies Collection integration with HNSWIndex, filter fallback, and metric safety."""

    def test_approx_search_without_index_falls_back_to_exact(self):
        """If search_type='approx' is requested without an index, falls back safely to exact search."""
        col = Collection("no_index_col", dimension=2)
        col.add_point("p1", [1.0, 0.0])
        col.add_point("p2", [0.0, 1.0])

        hits = col.search(query=[1.0, 0.0], limit=2, search_type="approx")
        assert len(hits) == 2
        assert hits[0].id == "p1"

    def test_approx_search_with_filter_falls_back_to_exact_prefiltering(self):
        """When metadata filters are supplied, approximate search falls back to exact search."""
        col = Collection("filtered_col", dimension=2)
        col.create_index(metric=DistanceMetric.COSINE)

        col.add_point("p1", [1.0, 0.0], payload={"category": "books"})
        col.add_point("p2", [0.99, 0.01], payload={"category": "electronics"})

        # Query very close to p2, but filter demands "books"
        filter_obj = Filter(must=[FieldCondition(key="category", operator="eq", value="books")])
        hits = col.search(
            query=[1.0, 0.0],
            limit=2,
            metric=DistanceMetric.COSINE,
            filter=filter_obj,
            search_type="approx",
        )

        assert len(hits) == 1
        assert hits[0].id == "p1"
        assert hits[0].payload["category"] == "books"

    def test_dynamic_point_addition_updates_active_index(self):
        """Points added after create_index() are indexed automatically."""
        col = Collection("dynamic_col", dimension=2)
        col.create_index(metric=DistanceMetric.COSINE)

        col.add_point("first", [1.0, 0.0])
        assert "first" in col.get_index(DistanceMetric.COSINE)

        col.add_point("second", [0.0, 1.0])
        assert "second" in col.get_index(DistanceMetric.COSINE)

        hits = col.search(query=[0.0, 1.0], limit=1, search_type="approx")
        assert hits[0].id == "second"

    def test_persistence_reload_and_index_rebuild(self, tmp_path: Path):
        """SQLite persistence recovers points and supports index building upon reload."""
        db_file = tmp_path / "persist_hnsw.db"
        storage = SQLiteStorage(db_file)
        engine = MiniVecEngine(storage=storage)
        col = engine.create_collection("persisted_col", dimension=3)
        col.add_point("p1", [1.0, 0.0, 0.0], payload={"tag": "a"})
        col.add_point("p2", [0.0, 1.0, 0.0], payload={"tag": "b"})
        engine.close()

        # Reload from SQLite
        reloaded_storage = SQLiteStorage(db_file)
        reloaded_engine = MiniVecEngine(storage=reloaded_storage)
        reloaded_col = reloaded_engine.get_collection("persisted_col")
        assert reloaded_col.count() == 2

        # Build index on reloaded collection
        index = reloaded_col.create_index(metric=DistanceMetric.COSINE)
        assert index.count() == 2

        hits = reloaded_col.search(query=[1.0, 0.0, 0.0], limit=1, search_type="approx")
        assert hits[0].id == "p1"
        reloaded_engine.close()


# =============================================================================
# 3. Repeatable Benchmarks & Recall@k Verification
# =============================================================================
class TestRecallAtKBenchmarks:
    """Evaluates empirical Recall@k across deterministic synthetic clustered datasets."""

    def test_recall_at_10_on_clustered_dataset_cosine(self):
        points, queries = generate_clustered_dataset(num_points=300, dimension=16, num_clusters=6, random_seed=42)

        col = Collection("bench_col_cosine", dimension=16)
        col.create_index(metric=DistanceMetric.COSINE, m=16, ef_construction=64, ef_search=32, random_seed=42)

        for i, vec in enumerate(points):
            col.add_point(f"doc_{i}", vec)

        report = evaluate_recall(col, queries=queries, k=10, metric=DistanceMetric.COSINE)

        # Print report summary to pytest stdout (-s) for honest reporting
        print(f"\n{report.summary()}")

        # Ensure high recall is achieved on clustered data
        assert report.mean_recall >= 0.85
        assert report.num_points == 300
        assert report.num_queries == 20

    def test_recall_at_10_euclidean(self):
        points, queries = generate_clustered_dataset(num_points=250, dimension=8, num_clusters=4, random_seed=99)

        col = Collection("bench_col_euclidean", dimension=8)
        col.create_index(metric=DistanceMetric.EUCLIDEAN, m=16, ef_construction=64, ef_search=32, random_seed=99)

        for i, vec in enumerate(points):
            col.add_point(f"doc_{i}", vec)

        report = evaluate_recall(col, queries=queries, k=10, metric=DistanceMetric.EUCLIDEAN)
        print(f"\n{report.summary()}")

        assert report.mean_recall >= 0.85

    def test_ef_search_tradeoff(self):
        """Verifies that increasing ef_search improves or preserves recall."""
        points, queries = generate_clustered_dataset(num_points=300, dimension=8, num_clusters=5, random_seed=777)

        col = Collection("tradeoff_col", dimension=8)
        col.create_index(metric=DistanceMetric.COSINE, m=8, ef_construction=32, ef_search=10, random_seed=777)

        for i, vec in enumerate(points):
            col.add_point(f"pt_{i}", vec)

        # Low exploration
        report_low = evaluate_recall(col, queries=queries, k=10, ef_search=10)
        # High exploration
        report_high = evaluate_recall(col, queries=queries, k=10, ef_search=64)

        print(f"\nLow ef (10): Recall={report_low.mean_recall:.3f}")
        print(f"High ef (64): Recall={report_high.mean_recall:.3f}")

        # Higher exploration should yield higher or equal recall
        assert report_high.mean_recall >= report_low.mean_recall - 1e-4
