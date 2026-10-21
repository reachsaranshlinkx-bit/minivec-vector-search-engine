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


class TestHNSWUpperLayerInsertion:
    """Deterministic regression tests for upper-layer node insertion."""

    def test_insert_node_exceeding_previous_max_layer(self, monkeypatch):
        """Verify node membership across all assigned layers when forced to a higher level."""
        index = HNSWIndex(dimension=2, metric=DistanceMetric.EUCLIDEAN, m=4, ef_construction=16)

        # Force levels: node_0 -> level 0, node_1 -> level 2
        levels = iter([0, 2])
        monkeypatch.setattr(index, "_random_level", lambda: next(levels))

        index.add("node_0", [0.0, 0.0])
        assert index.max_layer == 0
        assert index.entry_point == "node_0"

        index.add("node_1", [1.0, 1.0])

        # Invariants after inserting higher-level node
        assert index.max_layer == 2
        assert index.entry_point == "node_1"

        # node_1 must be present in every assigned layer from 0 through 2
        assert "node_1" in index.layers[0]
        assert "node_1" in index.layers[1]
        assert "node_1" in index.layers[2]

        # Upper layers where node_1 is currently the sole node must have empty neighbor lists
        assert index.layers[2]["node_1"] == []
        assert index.layers[1]["node_1"] == []

        # Layer 0 must connect node_1 and node_0 bidirectionally
        assert "node_0" in index.layers[0]["node_1"]
        assert "node_1" in index.layers[0]["node_0"]

        # node_0 must only exist in layer 0
        assert "node_0" not in index.layers[1]
        assert "node_0" not in index.layers[2]

    def test_subsequent_node_connects_to_elevated_layer(self, monkeypatch):
        """Verify subsequent node at the same elevated level connects and obeys degree limits."""
        index = HNSWIndex(dimension=2, metric=DistanceMetric.EUCLIDEAN, m=4, ef_construction=16)

        # Force levels: node_0 -> 0, node_1 -> 2, node_2 -> 2
        levels = iter([0, 2, 2])
        monkeypatch.setattr(index, "_random_level", lambda: next(levels))

        index.add("node_0", [0.0, 0.0])
        index.add("node_1", [1.0, 1.0])
        index.add("node_2", [1.1, 1.1])

        # Both elevated nodes must be connected on layer 2 and layer 1
        assert "node_2" in index.layers[2]["node_1"]
        assert "node_1" in index.layers[2]["node_2"]
        assert "node_2" in index.layers[1]["node_1"]
        assert "node_1" in index.layers[1]["node_2"]

        # Degree bounds must be satisfied across all layers
        for layer_idx, layer in enumerate(index.layers):
            limit = index.m0 if layer_idx == 0 else index.m
            for node_id, neighbors in layer.items():
                assert len(neighbors) <= limit

    def test_search_traversal_after_elevated_insertion(self, monkeypatch):
        """Verify greedy zoom from elevated entry point correctly finds nearest neighbors."""
        index = HNSWIndex(dimension=2, metric=DistanceMetric.EUCLIDEAN, m=4, ef_construction=16, ef_search=16)

        levels = iter([0, 2, 2])
        monkeypatch.setattr(index, "_random_level", lambda: next(levels))

        index.add("node_0", [0.0, 0.0])
        index.add("node_1", [10.0, 10.0])
        index.add("node_2", [10.1, 10.1])

        # Query close to node_0 (which only exists at layer 0)
        hits_0 = index.search(query=[0.05, 0.05], k=1)
        assert len(hits_0) == 1
        assert hits_0[0][0] == "node_0"

        # Query close to node_1 / node_2
        hits_1 = index.search(query=[10.05, 10.05], k=2)
        assert len(hits_1) == 2
        returned_ids = {hits_1[0][0], hits_1[1][0]}
        assert returned_ids == {"node_1", "node_2"}

