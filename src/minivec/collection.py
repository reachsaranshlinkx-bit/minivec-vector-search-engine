"""In-memory collection storage for MiniVec.

A Collection manages points of a fixed vector dimension in memory, enforcing
validation rules and ID uniqueness.
"""

import copy
import threading
from typing import Any
import numpy as np

from minivec.config import get_settings
from minivec.distance import DistanceMetric, compute_scores
from minivec.exceptions import DuplicateIdError, PointNotFoundError, ValidationError
from minivec.filter import FieldCondition, Filter, validate_filter
from minivec.models import Payload, Point, PointId, ScoredPoint
from minivec.validation import (
    validate_dimension,
    validate_limit,
    validate_metric,
    validate_payload,
    validate_point_id,
    validate_vector,
)


class Collection:
    """Manages an in-memory collection of high-dimensional points.

    Attributes:
        name: Human-readable identifier for the collection.
        dimension: Fixed vector dimensionality for all points in this collection.
    """

    def __init__(self, name: str, dimension: int, storage: Any = None) -> None:
        """Initialize an in-memory collection, optionally backed by persistent storage.

        Args:
            name: The name of the collection (e.g. 'documents', 'products').
            dimension: Fixed positive integer vector dimension.
            storage: Optional persistence repository (e.g. SQLiteStorage).

        Raises:
            ValidationError: If name is invalid or dimension is not a positive integer.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValidationError("Collection name must be a non-empty string")

        self.name: str = name.strip()
        self.dimension: int = validate_dimension(
            dimension, max_dim=get_settings().max_vector_dimension
        )
        self._storage = storage
        self._lock = threading.RLock()
        self._points: dict[PointId, Point] = {}


    def _clone_point(self, point: Point) -> Point:
        """Return an isolated defensive copy of a Point."""
        return Point(
            id=point.id,
            vector=point.vector.copy(),
            payload=copy.deepcopy(point.payload),
        )

    def _hydrate_point(self, point: Point) -> None:
        """Load a persisted point directly into memory without re-triggering storage write."""
        with self._lock:
            self._points[point.id] = Point(
                id=point.id,
                vector=point.vector.copy(),
                payload=copy.deepcopy(point.payload),
            )

    def add(self, point: Point) -> None:
        """Add a Point to the collection.

        Validates the point's ID, vector dimension and values, and checks
        for duplicate IDs before saving. If a persistent storage backend is
        attached, persists the point to disk before updating in-memory state.

        Args:
            point: The Point instance to add.

        Raises:
            ValidationError: If vector, ID, or payload fails validation.
            DuplicateIdError: If a point with the same ID already exists.
        """
        if not isinstance(point, Point):
            raise ValidationError(f"Expected a Point instance, got {type(point).__name__}")

        valid_id = validate_point_id(point.id)
        valid_vector = validate_vector(point.vector, expected_dim=self.dimension)
        valid_payload = validate_payload(
            point.payload, max_bytes=get_settings().max_payload_bytes
        )

        with self._lock:
            if valid_id in self._points:
                raise DuplicateIdError(
                    f"Point with ID '{valid_id}' already exists in collection '{self.name}'"
                )

            # Write-through persistence: save to disk before updating in-memory cache
            if self._storage is not None:
                self._storage.save_point(
                    collection_name=self.name,
                    id=valid_id,
                    vector=valid_vector,
                    payload=valid_payload,
                )

            # Store completely isolated copies of vector and payload in memory
            self._points[valid_id] = Point(
                id=valid_id,
                vector=valid_vector.copy(),
                payload=copy.deepcopy(valid_payload),
            )

    def add_point(
        self,
        id: PointId,
        vector: Any,
        payload: Any = None,
    ) -> Point:
        """Convenience method to construct, validate, and add a point.

        Args:
            id: Unique identifier for the point.
            vector: Array-like numerical vector.
            payload: Optional metadata dictionary.

        Returns:
            A defensive copy of the newly created and stored Point.
        """
        safe_payload = {} if payload is None else payload
        point = Point(id=id, vector=vector, payload=safe_payload)
        self.add(point)
        return self.get(point.id)

    def get(self, point_id: PointId) -> Point:
        """Retrieve a stored point by its ID.

        Returns a defensive copy so modifying the returned point, its vector,
        or its payload will not affect the collection's internal records.

        Args:
            point_id: The ID of the point to retrieve.

        Returns:
            A defensive copy of the stored Point.

        Raises:
            PointNotFoundError: If no point with the given ID exists.
        """
        with self._lock:
            if point_id not in self._points:
                raise PointNotFoundError(
                    f"Point with ID '{point_id}' not found in collection '{self.name}'"
                )
            return self._clone_point(self._points[point_id])

    def get_optional(self, point_id: PointId) -> Point | None:
        """Retrieve a stored point by its ID, returning None if not found.

        Returns a defensive copy if found.

        Args:
            point_id: The ID of the point to retrieve.

        Returns:
            A defensive copy of the Point if found, else None.
        """
        with self._lock:
            point = self._points.get(point_id)
            if point is None:
                return None
            return self._clone_point(point)

    def count(self) -> int:
        """Return the total number of points stored in the collection."""
        with self._lock:
            return len(self._points)

    def get_all(self) -> list[Point]:
        """Return a list of defensive copies of all points in the collection."""
        with self._lock:
            return [self._clone_point(p) for p in self._points.values()]

    def __len__(self) -> int:
        """Support len(collection)."""
        with self._lock:
            return len(self._points)

    def __contains__(self, point_id: PointId) -> bool:
        """Support 'point_id in collection' syntax."""
        with self._lock:
            return point_id in self._points

    def search(
        self,
        query: Any,
        limit: int = 10,
        metric: DistanceMetric | str = DistanceMetric.COSINE,
        filter: Filter | list[FieldCondition] | FieldCondition | None = None,
        with_payload: bool = True,
        with_vector: bool = False,
    ) -> list[ScoredPoint]:
        """Perform exact similarity or distance search with optional pre-filtering.

        Filters candidates based on metadata conditions BEFORE calculating
        vector distances, computes the similarity/distance for all eligible
        points, and returns the top-k matches ranked appropriately.

        Ties in scores/distances are broken deterministically by point ID.

        Args:
            query: 1D array-like query vector matching collection dimension.
            limit: Maximum number of top-ranked results to return (k > 0).
            metric: Distance or similarity metric (cosine, dot, or euclidean).
            filter: Optional Filter, FieldCondition, or list of FieldConditions
                specifying metadata constraints (combined with AND semantics).
            with_payload: Whether to include metadata payload in results.
            with_vector: Whether to include defensive copy of vector in results.

        Returns:
            List of ScoredPoint instances ranked from best match to worst match.

        Raises:
            ValidationError: If query vector, limit k, metric, or filter is invalid.
        """
        query_vec = validate_vector(query, expected_dim=self.dimension)
        k = validate_limit(limit, max_limit=get_settings().max_search_limit)
        norm_metric = validate_metric(metric)
        norm_filter = validate_filter(filter)

        with self._lock:
            if not self._points:
                return []

            # Pre-filtering: Filter candidate points BEFORE computing similarity
            if norm_filter is not None:
                eligible_points = [
                    self._clone_point(point)
                    for point in self._points.values()
                    if norm_filter.matches(point.payload)
                ]
            else:
                eligible_points = [self._clone_point(point) for point in self._points.values()]

        # If no points match the metadata filter, return empty results immediately
        if not eligible_points:
            return []

        candidate_matrix = np.vstack([p.vector for p in eligible_points])
        scores, is_descending = compute_scores(query_vec, candidate_matrix, norm_metric)

        # Pair each eligible point with its scalar score
        scored_candidates = [
            (point, float(scores[i])) for i, point in enumerate(eligible_points)
        ]

        # Deterministic sorting:
        # Step 1: Break ties by sorting by point ID string ascending
        scored_candidates.sort(key=lambda item: str(item[0].id))

        # Step 2: Sort by score (descending for similarity, ascending for distance)
        # Python's Timsort is stable, so tie-breaking order from Step 1 is preserved
        scored_candidates.sort(key=lambda item: item[1], reverse=is_descending)

        # Slice top-k results
        top_k = scored_candidates[:k]

        # Construct ScoredPoint return objects with defensive copies
        results = []
        for point, score in top_k:
            payload_copy = copy.deepcopy(point.payload) if with_payload else {}
            vector_copy = point.vector.copy() if with_vector else None
            results.append(
                ScoredPoint(
                    id=point.id,
                    score=score,
                    payload=payload_copy,
                    vector=vector_copy,
                )
            )

        return results

    def __repr__(self) -> str:
        return (
            f"Collection(name={self.name!r}, dimension={self.dimension}, "
            f"points_count={self.count()})"
        )

