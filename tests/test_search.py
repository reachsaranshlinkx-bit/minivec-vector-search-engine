"""Tests for Milestone 2: Exact Similarity Search.

Verifies:
1. Cosine similarity, dot product, and Euclidean distance implementations.
2. Ranking orders (higher is better for similarity; lower is better for distance).
3. Metric divergence (how different metrics rank identical candidate sets differently).
4. Zero vector handling in cosine similarity (no NaNs, no Inf).
5. Top-k slicing (k < N, k == N, k > N) and limit validation.
6. Empty collection handling.
7. Deterministic tie-breaking by point ID.
8. Defensive isolation of returned search results.
"""

import numpy as np
import pytest

from minivec import (
    Collection,
    DistanceMetric,
    Point,
    ScoredPoint,
    SearchResult,
    ValidationError,
)


@pytest.fixture
def sample_collection() -> Collection:
    """Fixture providing a collection populated with 2D geometric test vectors."""
    col = Collection(name="geometry", dimension=2)
    # Unit vector along X axis
    col.add_point(id="unit_x", vector=[1.0, 0.0], payload={"desc": "unit along x"})
    # Scaled vector along X axis (same direction, 10x magnitude)
    col.add_point(id="scaled_x", vector=[10.0, 0.0], payload={"desc": "scaled along x"})
    # Orthogonal unit vector along Y axis
    col.add_point(id="unit_y", vector=[0.0, 1.0], payload={"desc": "unit along y"})
    # Opposite direction along -X axis
    col.add_point(id="neg_x", vector=[-1.0, 0.0], payload={"desc": "opposite along -x"})
    return col


class TestCosineSimilarity:
    """Tests for exact cosine similarity calculation and ranking."""

    def test_cosine_similarity_values(self, sample_collection):
        query = [1.0, 0.0]
        results = sample_collection.search(query=query, limit=10, metric=DistanceMetric.COSINE)

        result_map = {r.id: r.score for r in results}

        # Same direction -> cosine = 1.0
        assert np.isclose(result_map["unit_x"], 1.0, atol=1e-5)
        assert np.isclose(result_map["scaled_x"], 1.0, atol=1e-5)

        # Orthogonal -> cosine = 0.0
        assert np.isclose(result_map["unit_y"], 0.0, atol=1e-5)

        # Opposite -> cosine = -1.0
        assert np.isclose(result_map["neg_x"], -1.0, atol=1e-5)

    def test_cosine_ranking_order(self, sample_collection):
        # Higher score must rank first
        query = [1.0, 0.0]
        results = sample_collection.search(query=query, limit=4, metric=DistanceMetric.COSINE)

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

        # unit_x and scaled_x tie at 1.0; unit_y is 0.0; neg_x is -1.0
        assert {results[0].id, results[1].id} == {"scaled_x", "unit_x"}
        assert results[2].id == "unit_y"
        assert results[3].id == "neg_x"

    def test_cosine_zero_query_vector(self):
        col = Collection(name="zero_test", dimension=2)
        col.add_point(id=1, vector=[1.0, 2.0])
        col.add_point(id=2, vector=[3.0, 4.0])

        # Zero query vector: [0.0, 0.0]
        results = col.search(query=[0.0, 0.0], limit=5, metric=DistanceMetric.COSINE)
        assert len(results) == 2
        for r in results:
            assert r.score == 0.0
            assert not np.isnan(r.score)
            assert not np.isinf(r.score)

    def test_cosine_zero_stored_vector(self):
        col = Collection(name="zero_stored", dimension=2)
        col.add_point(id="zero", vector=[0.0, 0.0])
        col.add_point(id="normal", vector=[1.0, 1.0])

        results = col.search(query=[1.0, 1.0], limit=5, metric=DistanceMetric.COSINE)
        result_map = {r.id: r.score for r in results}

        assert np.isclose(result_map["normal"], 1.0, atol=1e-5)
        assert result_map["zero"] == 0.0


class TestDotProduct:
    """Tests for dot product metric, highlighting magnitude sensitivity."""

    def test_dot_product_values_and_magnitude_effect(self, sample_collection):
        query = [1.0, 0.0]
        results = sample_collection.search(query=query, limit=10, metric=DistanceMetric.DOT)

        result_map = {r.id: r.score for r in results}

        # scaled_x: [10, 0] · [1, 0] = 10.0
        assert np.isclose(result_map["scaled_x"], 10.0)
        # unit_x:   [1, 0] · [1, 0]  = 1.0
        assert np.isclose(result_map["unit_x"], 1.0)
        # unit_y:   [0, 1] · [1, 0]  = 0.0
        assert np.isclose(result_map["unit_y"], 0.0)
        # neg_x:    [-1, 0] · [1, 0] = -1.0
        assert np.isclose(result_map["neg_x"], -1.0)

        # Under dot product, scaled_x strongly outranks unit_x due to magnitude!
        assert results[0].id == "scaled_x"
        assert results[1].id == "unit_x"


class TestEuclideanDistance:
    """Tests for Euclidean distance, where smaller distance ranks first."""

    def test_euclidean_distance_values_and_ordering(self, sample_collection):
        query = [1.0, 0.0]
        results = sample_collection.search(query=query, limit=10, metric=DistanceMetric.EUCLIDEAN)

        result_map = {r.id: r.score for r in results}

        # Distance to self [1, 0] -> 0.0
        assert np.isclose(result_map["unit_x"], 0.0)
        # Distance to [0, 1] -> sqrt((1-0)^2 + (0-1)^2) = sqrt(2) ~= 1.4142
        assert np.isclose(result_map["unit_y"], np.sqrt(2.0), atol=1e-5)
        # Distance to [-1, 0] -> sqrt((1 - (-1))^2) = 2.0
        assert np.isclose(result_map["neg_x"], 2.0)
        # Distance to [10, 0] -> 9.0
        assert np.isclose(result_map["scaled_x"], 9.0)

        # Ascending order check: smallest distance first!
        assert [r.id for r in results] == ["unit_x", "unit_y", "neg_x", "scaled_x"]


class TestMetricComparisonAndDivergence:
    """Demonstrates how the same query produces completely different rankings per metric."""

    def test_metric_ranking_divergence(self, sample_collection):
        query = [1.0, 0.0]

        cosine_ids = [r.id for r in sample_collection.search(query, limit=4, metric="cosine")]
        dot_ids = [r.id for r in sample_collection.search(query, limit=4, metric="dot")]
        euclidean_ids = [r.id for r in sample_collection.search(query, limit=4, metric="euclidean")]

        # Under Dot Product: scaled_x (score 10) is #1
        assert dot_ids[0] == "scaled_x"

        # Under Euclidean: unit_x (dist 0.0) is #1; scaled_x (dist 9.0) is LAST (#4)
        assert euclidean_ids[0] == "unit_x"
        assert euclidean_ids[-1] == "scaled_x"

        # Under Cosine: scaled_x and unit_x both have 1.0 (tied at top)
        assert {cosine_ids[0], cosine_ids[1]} == {"scaled_x", "unit_x"}


class TestTopKLimitsAndValidation:
    """Tests for limit / k parameter boundaries and input checks."""

    def test_limit_less_than_total(self, sample_collection):
        results = sample_collection.search([1.0, 0.0], limit=2)
        assert len(results) == 2

    def test_limit_equal_total(self, sample_collection):
        results = sample_collection.search([1.0, 0.0], limit=4)
        assert len(results) == 4

    def test_limit_greater_than_total(self, sample_collection):
        results = sample_collection.search([1.0, 0.0], limit=100)
        assert len(results) == 4

    @pytest.mark.parametrize("invalid_k", [0, -1, -50])
    def test_reject_non_positive_limit(self, sample_collection, invalid_k):
        with pytest.raises(ValidationError, match="positive integer"):
            sample_collection.search([1.0, 0.0], limit=invalid_k)

    @pytest.mark.parametrize("invalid_k", [None, "5", 2.5, True, False])
    def test_reject_non_integer_limit(self, sample_collection, invalid_k):
        with pytest.raises(ValidationError, match="must be an integer"):
            sample_collection.search([1.0, 0.0], limit=invalid_k)


class TestEmptyCollectionAndValidation:
    """Tests for empty collection searches and query vector validation."""

    def test_search_empty_collection_returns_empty_list(self):
        col = Collection(name="empty", dimension=3)
        results = col.search(query=[1.0, 2.0, 3.0], limit=5)
        assert results == []

    def test_search_rejects_query_dimension_mismatch(self, sample_collection):
        with pytest.raises(ValidationError, match="dimension mismatch"):
            sample_collection.search(query=[1.0, 2.0, 3.0])  # dim 3 on dim 2 collection

    def test_search_rejects_nan_in_query(self, sample_collection):
        with pytest.raises(ValidationError, match="NaN"):
            sample_collection.search(query=[1.0, np.nan])

    def test_search_rejects_inf_in_query(self, sample_collection):
        with pytest.raises(ValidationError, match="infinite"):
            sample_collection.search(query=[np.inf, 0.0])

    def test_search_rejects_unsupported_metric_string(self, sample_collection):
        with pytest.raises(ValidationError, match="Unsupported distance metric"):
            sample_collection.search(query=[1.0, 0.0], metric="manhattan")


class TestDeterministicTieBreaking:
    """Tests that identical scores/distances are broken deterministically by point ID."""

    def test_ties_broken_by_id_ascending_for_similarity(self):
        col = Collection(name="ties", dimension=2)
        # 3 points with identical vectors -> identical cosine and dot product scores
        col.add_point(id="charlie", vector=[1.0, 1.0])
        col.add_point(id="alpha", vector=[1.0, 1.0])
        col.add_point(id="bravo", vector=[1.0, 1.0])

        results = col.search(query=[1.0, 1.0], limit=3, metric="cosine")
        ids = [r.id for r in results]
        assert ids == ["alpha", "bravo", "charlie"]

    def test_ties_broken_by_id_ascending_for_distance(self):
        col = Collection(name="ties_dist", dimension=2)
        col.add_point(id="zebra", vector=[2.0, 0.0])
        col.add_point(id="apple", vector=[2.0, 0.0])

        results = col.search(query=[0.0, 0.0], limit=2, metric="euclidean")
        ids = [r.id for r in results]
        assert ids == ["apple", "zebra"]


class TestSearchPayloadAndVectorOptions:
    """Tests for payload inclusion and defensive vector returns."""

    def test_search_with_payload_flag(self, sample_collection):
        res_with = sample_collection.search([1.0, 0.0], limit=1, with_payload=True)
        assert res_with[0].payload == {"desc": "scaled along x"} or "desc" in res_with[0].payload

        res_without = sample_collection.search([1.0, 0.0], limit=1, with_payload=False)
        assert res_without[0].payload == {}

    def test_search_with_vector_flag(self, sample_collection):
        res_no_vec = sample_collection.search([1.0, 0.0], limit=1, with_vector=False)
        assert res_no_vec[0].vector is None

        res_vec = sample_collection.search([1.0, 0.0], limit=1, with_vector=True)
        assert res_vec[0].vector is not None
        assert isinstance(res_vec[0].vector, np.ndarray)
        assert res_vec[0].vector.shape == (2,)

    def test_search_results_defensively_isolated(self, sample_collection):
        results = sample_collection.search([1.0, 0.0], limit=1, with_payload=True, with_vector=True)
        top = results[0]

        # Mutate result
        top.vector[0] = -999.0
        top.payload["desc"] = "corrupted"

        # Verify underlying point in collection remains unchanged
        stored = sample_collection.get(top.id)
        assert stored.payload["desc"] != "corrupted"
        assert not np.isclose(stored.vector[0], -999.0)
