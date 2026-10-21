"""Hierarchical Navigable Small World (HNSW) graph index for MiniVec.

An educational, pure-Python implementation of the HNSW approximate
nearest-neighbor (ANN) search algorithm based on Malkov & Yashunin (2018).
Does not rely on external C/C++ or third-party ANN libraries.
"""

from __future__ import annotations

import heapq
import math
import random
import threading
from typing import Any
import numpy as np

from minivec.distance import DistanceMetric
from minivec.exceptions import ValidationError
from minivec.models import PointId
from minivec.validation import validate_metric, validate_vector


class HNSWIndex:
    """In-memory Hierarchical Navigable Small World (HNSW) index.

    Organizes high-dimensional vectors across multiple graph layers:
    - Layer 0 contains all indexed points with local, short-range connections.
    - Higher layers contain exponentially fewer points with long-range express links.

    Attributes:
        dimension: Dimensionality of indexed vectors.
        metric: DistanceMetric configured for this index.
        m: Maximum outgoing edges per node in layers > 0.
        m0: Maximum outgoing edges per node in Layer 0 (default 2 * m).
        ef_construction: Size of the candidate dynamic list during insertion.
        ef_search: Size of the candidate dynamic list during search queries.
        random_seed: Optional seed for deterministic layer assignment.
    """

    def __init__(
        self,
        dimension: int,
        metric: DistanceMetric | str = DistanceMetric.COSINE,
        m: int = 16,
        ef_construction: int = 64,
        ef_search: int = 32,
        random_seed: int | None = 42,
    ) -> None:
        if dimension <= 0:
            raise ValidationError(f"Dimension must be a positive integer, got {dimension}")
        if m < 2:
            raise ValidationError(f"Parameter 'm' must be at least 2, got {m}")
        if ef_construction < m:
            raise ValidationError(
                f"ef_construction ({ef_construction}) must be >= m ({m})"
            )

        self.dimension: int = dimension
        self.metric: DistanceMetric = validate_metric(metric)
        self.m: int = m
        self.m0: int = 2 * m
        self.ef_construction: int = ef_construction
        self.ef_search: int = ef_search
        self.m_l: float = 1.0 / math.log(float(m))

        self._rng = random.Random(random_seed)
        self._lock = threading.RLock()

        # Storage structures
        # Map: PointId -> float32 1D numpy array
        self.nodes: dict[PointId, np.ndarray] = {}
        # List of layers: layers[layer_idx] is dict[PointId, list[PointId]]
        self.layers: list[dict[PointId, list[PointId]]] = [{}]
        self.entry_point: PointId | None = None
        self.max_layer: int = -1

    def _random_level(self) -> int:
        """Assign node level with exponential decay distribution.

        Formula:
            floor(-ln(uniform(0, 1)) * m_L)
        """
        r = self._rng.random()
        # Avoid log(0)
        while r == 0.0:
            r = self._rng.random()
        return int(-math.log(r) * self.m_l)

    def _distance(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute scalar distance where smaller is always closer/better.

        - Cosine: 1.0 - cosine_similarity (range [0.0, 2.0])
        - Euclidean: L2 norm (range [0.0, inf))
        - Dot product: -dot_product (lower distance = higher dot product)
        """
        if self.metric == DistanceMetric.EUCLIDEAN:
            diff = vec_a - vec_b
            return float(np.linalg.norm(diff))

        if self.metric == DistanceMetric.COSINE:
            norm_a = float(np.linalg.norm(vec_a))
            norm_b = float(np.linalg.norm(vec_b))
            denom = norm_a * norm_b
            if math.isclose(denom, 0.0):
                sim = 0.0
            else:
                sim = float(np.dot(vec_a, vec_b) / denom)
                sim = max(-1.0, min(1.0, sim))
            return 1.0 - sim

        if self.metric == DistanceMetric.DOT:
            return -float(np.dot(vec_a, vec_b))

        raise ValueError(f"Unsupported metric: {self.metric}")

    def _distance_to_score(self, distance: float) -> float:
        """Convert internal routing distance back to MiniVec's canonical metric score."""
        if self.metric == DistanceMetric.EUCLIDEAN:
            return distance
        if self.metric == DistanceMetric.COSINE:
            return 1.0 - distance
        if self.metric == DistanceMetric.DOT:
            return -distance
        raise ValueError(f"Unsupported metric: {self.metric}")

    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: list[PointId],
        ef: int,
        layer: int,
    ) -> list[tuple[float, PointId]]:
        """Greedy beam search for the closest nodes to query on a specific layer.

        Maintains:
        - candidates: Min-heap of discovered nodes to evaluate next (dist, point_id).
        - visited: Set of evaluated node IDs.
        - w: Max-heap of the best `ef` nodes found so far (-dist, point_id).

        Returns:
            List of (distance, point_id) tuples ordered by distance ascending.
        """
        adj = self.layers[layer]
        visited: set[PointId] = set(entry_points)

        candidates: list[tuple[float, PointId]] = []
        w: list[tuple[float, PointId]] = []

        for ep in entry_points:
            d = self._distance(query, self.nodes[ep])
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(w, (-d, ep))

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            farthest_dist = -w[0][0]

            if c_dist > farthest_dist:
                break

            for neighbor_id in adj.get(c_id, []):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    farthest_dist = -w[0][0]
                    d_neighbor = self._distance(query, self.nodes[neighbor_id])

                    if d_neighbor < farthest_dist or len(w) < ef:
                        heapq.heappush(candidates, (d_neighbor, neighbor_id))
                        heapq.heappush(w, (-d_neighbor, neighbor_id))

                        if len(w) > ef:
                            heapq.heappop(w)

        # Extract results and sort ascending by distance
        results = [(-neg_dist, p_id) for neg_dist, p_id in w]
        results.sort(key=lambda item: (item[0], str(item[1])))
        return results

    def add(self, point_id: PointId, vector: Any) -> None:
        """Insert a vector into the HNSW graph.

        Args:
            point_id: Unique point identifier.
            vector: Array-like numerical vector matching index dimension.
        """
        vec = validate_vector(vector, expected_dim=self.dimension)

        with self._lock:
            if point_id in self.nodes:
                raise ValidationError(f"Point '{point_id}' already exists in index")

            target_level = self._random_level()
            self.nodes[point_id] = vec

            # Ensure layer structures exist up to target_level
            while len(self.layers) <= target_level:
                self.layers.append({})

            # If index was empty, initialize as sole entry point
            if self.entry_point is None:
                for l in range(target_level + 1):
                    self.layers[l][point_id] = []
                self.entry_point = point_id
                self.max_layer = target_level
                return

            curr_obj = self.entry_point
            curr_dist = self._distance(vec, self.nodes[curr_obj])

            # Phase 1: Greedily zoom down from top layer to target_level + 1
            for l in range(self.max_layer, target_level, -1):
                changed = True
                while changed:
                    changed = False
                    for neighbor_id in self.layers[l].get(curr_obj, []):
                        d = self._distance(vec, self.nodes[neighbor_id])
                        if d < curr_dist:
                            curr_dist = d
                            curr_obj = neighbor_id
                            changed = True

            # Phase 2: Beam search and connect on layers from min(max_layer, target_level) down to 0
            ep_list = [curr_obj]
            for l in range(min(self.max_layer, target_level), -1, -1):
                # Search layer with ef_construction
                nearest_candidates = self._search_layer(
                    query=vec,
                    entry_points=ep_list,
                    ef=self.ef_construction,
                    layer=l,
                )

                m_max = self.m0 if l == 0 else self.m
                # Select up to m_max nearest neighbors
                chosen_neighbors = [p_id for _, p_id in nearest_candidates[:m_max]]

                # Initialize adjacency for new node
                self.layers[l][point_id] = list(chosen_neighbors)

                # Connect bidirectionally and prune if exceeding m_max
                for n_id in chosen_neighbors:
                    n_neighbors = self.layers[l].setdefault(n_id, [])
                    if point_id not in n_neighbors:
                        n_neighbors.append(point_id)
                        if len(n_neighbors) > m_max:
                            # Prune farthest neighbor of n_id
                            n_neighbors.sort(
                                key=lambda item_id: self._distance(
                                    self.nodes[n_id], self.nodes[item_id]
                                )
                            )
                            self.layers[l][n_id] = n_neighbors[:m_max]

                ep_list = [p_id for _, p_id in nearest_candidates]

            # If newly inserted node is on a higher layer than current max, promote it
            if target_level > self.max_layer:
                self.max_layer = target_level
                self.entry_point = point_id

    def search(
        self,
        query: Any,
        k: int = 10,
        ef: int | None = None,
    ) -> list[tuple[PointId, float]]:
        """Approximate nearest neighbor search.

        Args:
            query: 1D array-like query vector.
            k: Number of nearest neighbors to return.
            ef: Exploration factor for search. Defaults to max(ef_search, k).

        Returns:
            List of (point_id, canonical_score) sorted from best match to worst.
        """
        query_vec = validate_vector(query, expected_dim=self.dimension)
        if k <= 0:
            raise ValidationError(f"Search limit 'k' must be a positive integer, got {k}")

        with self._lock:
            if not self.nodes or self.entry_point is None:
                return []

            ef_val = max(ef or self.ef_search, k)

            curr_obj = self.entry_point
            curr_dist = self._distance(query_vec, self.nodes[curr_obj])

            # Greedy zoom through top layers
            for l in range(self.max_layer, 0, -1):
                changed = True
                while changed:
                    changed = False
                    for neighbor_id in self.layers[l].get(curr_obj, []):
                        d = self._distance(query_vec, self.nodes[neighbor_id])
                        if d < curr_dist:
                            curr_dist = d
                            curr_obj = neighbor_id
                            changed = True

            # Fine-grained beam search at Layer 0
            candidates = self._search_layer(
                query=query_vec,
                entry_points=[curr_obj],
                ef=ef_val,
                layer=0,
            )

            # Slice top-k
            top_k = candidates[:k]

            # Convert distance to canonical metric score
            results: list[tuple[PointId, float]] = []
            for d, p_id in top_k:
                score = self._distance_to_score(d)
                results.append((p_id, score))

            # Sort results deterministically:
            # Step 1: Tie-break on PointId ascending
            results.sort(key=lambda item: str(item[0]))
            # Step 2: Sort on score (descending for similarity, ascending for distance)
            is_descending = self.metric != DistanceMetric.EUCLIDEAN
            results.sort(key=lambda item: item[1], reverse=is_descending)

            return results

    def count(self) -> int:
        """Return total number of points in the index."""
        with self._lock:
            return len(self.nodes)

    def __len__(self) -> int:
        with self._lock:
            return len(self.nodes)

    def __contains__(self, point_id: PointId) -> bool:
        with self._lock:
            return point_id in self.nodes
