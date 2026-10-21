"""Unit tests for the educational HNSW graph index."""

import numpy as np
import pytest

from minivec.distance import DistanceMetric
from minivec.exceptions import ValidationError
from minivec.hnsw import HNSWIndex


class TestHNSWInitialization:
    """Tests for HNSWIndex initialization and parameter validation."""

    def test_valid_initialization(self):
        index = HNSWIndex(dimension=8, metric=DistanceMetric.COSINE, m=16, ef_construction=64)
        assert index.dimension == 8
        assert index.metric == DistanceMetric.COSINE
        assert index.m == 16
        assert index.m0 == 32
        assert index.count() == 0
        assert index.entry_point is None

    def test_reject_invalid_dimension(self):
        with pytest.raises(ValidationError, match="Dimension must be a positive integer"):
            HNSWIndex(dimension=0)

        with pytest.raises(ValidationError, match="Dimension must be a positive integer"):
            HNSWIndex(dimension=-4)

    def test_reject_invalid_m(self):
        with pytest.raises(ValidationError, match="Parameter 'm' must be at least 2"):
            HNSWIndex(dimension=4, m=1)

    def test_reject_ef_less_than_m(self):
        with pytest.raises(ValidationError, match="ef_construction .* must be >= m"):
            HNSWIndex(dimension=4, m=16, ef_construction=10)


class TestHNSWGraphStructureAndInsertion:
    """Tests for node insertion, layer distribution, and degree constraints."""

    def test_insert_single_node(self):
        index = HNSWIndex(dimension=3, metric=DistanceMetric.EUCLIDEAN, random_seed=42)
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        index.add("node_1", vec)

        assert index.count() == 1
        assert "node_1" in index
        assert index.entry_point == "node_1"
        assert "node_1" in index.layers[0]

    def test_reject_duplicate_id(self):
        index = HNSWIndex(dimension=2, metric=DistanceMetric.COSINE)
        index.add("id_1", [1.0, 0.0])
        with pytest.raises(ValidationError, match="already exists"):
            index.add("id_1", [0.0, 1.0])

    def test_reject_dimension_mismatch(self):
        index = HNSWIndex(dimension=3, metric=DistanceMetric.COSINE)
        with pytest.raises(ValidationError, match="Vector dimension mismatch"):
            index.add("id_1", [1.0, 0.0])

    def test_layer_hierarchy_and_degree_bounds(self):
        """Verify layer 0 contains all points and no node exceeds max edge capacity."""
        m = 4
        m0 = 2 * m  # 8
        index = HNSWIndex(dimension=4, metric=DistanceMetric.EUCLIDEAN, m=m, ef_construction=16, random_seed=123)

        rng = np.random.default_rng(123)
        num_points = 40
        for i in range(num_points):
            v = rng.standard_normal(4).astype(np.float32)
            index.add(f"pt_{i}", v)

        assert index.count() == num_points

        # All points must exist in layer 0
        assert len(index.layers[0]) == num_points

        # Check degree bounds for all nodes across all layers
        for layer_idx, layer in enumerate(index.layers):
            limit = m0 if layer_idx == 0 else m
            for node_id, neighbors in layer.items():
                assert len(neighbors) <= limit, f"Node {node_id} exceeded degree {limit} in layer {layer_idx}"


class TestHNSWSearchMetrics:
    """Tests approximate search correctness across Cosine, Euclidean, and Dot metrics."""

    def test_search_empty_index_returns_empty_list(self):
        index = HNSWIndex(dimension=4, metric=DistanceMetric.COSINE)
        assert index.search(query=[1.0, 0.0, 0.0, 0.0], k=5) == []

    def test_search_cosine_finds_nearest_angular_vector(self):
        index = HNSWIndex(dimension=2, metric=DistanceMetric.COSINE, m=4, ef_construction=16, ef_search=16)
        index.add("east", [1.0, 0.0])
        index.add("north", [0.0, 1.0])
        index.add("north_east", [0.7071, 0.7071])

        # Query pointing northeast
        hits = index.search(query=[0.7071, 0.7071], k=3)
        assert len(hits) == 3
        top_id, top_score = hits[0]
        assert top_id == "north_east"
        assert np.isclose(top_score, 1.0, atol=1e-3)

    def test_search_euclidean_finds_closest_point(self):
        index = HNSWIndex(dimension=2, metric=DistanceMetric.EUCLIDEAN, m=4, ef_construction=16)
        index.add("p_0_0", [0.0, 0.0])
        index.add("p_1_1", [1.0, 1.0])
        index.add("p_10_10", [10.0, 10.0])

        hits = index.search(query=[0.1, 0.1], k=2)
        assert len(hits) == 2
        assert hits[0][0] == "p_0_0"
        # Distance should be small positive float
        assert hits[0][1] < hits[1][1]

    def test_search_dot_product_finds_largest_projection(self):
        index = HNSWIndex(dimension=2, metric=DistanceMetric.DOT, m=4, ef_construction=16)
        index.add("small", [1.0, 1.0])
        index.add("medium", [3.0, 3.0])
        index.add("large", [10.0, 10.0])

        hits = index.search(query=[1.0, 1.0], k=3)
        assert hits[0][0] == "large"
        assert hits[0][1] > hits[1][1]

    def test_search_k_larger_than_total_points(self):
        index = HNSWIndex(dimension=2, metric=DistanceMetric.COSINE)
        index.add("a", [1.0, 0.0])
        index.add("b", [0.0, 1.0])

        hits = index.search(query=[1.0, 0.0], k=10)
        assert len(hits) == 2
