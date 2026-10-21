"""Validation utilities for MiniVec.

Ensures vectors, dimensions, point IDs, and payloads conform strictly to
expected types and numerical constraints before entering storage.
"""

from typing import Any, Sequence
import numpy as np

from minivec.exceptions import ValidationError
from minivec.models import Payload, PointId


def validate_dimension(dimension: Any, max_dim: int | None = None) -> int:
    """Validate that collection dimension is a positive integer > 0 within optional bounds.

    Args:
        dimension: The dimension candidate to validate.
        max_dim: Optional maximum allowed dimensionality.

    Returns:
        The validated integer dimension.

    Raises:
        ValidationError: If dimension is not a positive integer or exceeds max_dim.
    """
    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise ValidationError(
            f"Collection dimension must be an integer, got {type(dimension).__name__} ({dimension!r})"
        )
    if dimension <= 0:
        raise ValidationError(
            f"Collection dimension must be a positive integer > 0, got {dimension}"
        )
    if max_dim is not None and dimension > max_dim:
        raise ValidationError(
            f"Collection dimension {dimension} exceeds maximum allowed limit of {max_dim}"
        )
    return dimension



def validate_point_id(point_id: Any) -> PointId:
    """Validate that a point ID is a valid integer or non-empty string.

    Args:
        point_id: The ID candidate to validate.

    Returns:
        The validated PointId.

    Raises:
        ValidationError: If ID is not an int or non-empty string.
    """
    if isinstance(point_id, bool):
        raise ValidationError("Point ID cannot be a boolean value")

    if isinstance(point_id, int):
        return point_id

    if isinstance(point_id, str):
        if not point_id.strip():
            raise ValidationError("Point ID string cannot be empty or whitespace only")
        return point_id

    raise ValidationError(
        f"Point ID must be an integer or string, got {type(point_id).__name__} ({point_id!r})"
    )


def validate_vector(vector: Any, expected_dim: int) -> np.ndarray:
    """Validate and convert an input vector to a clean 1D float32 NumPy array.

    Args:
        vector: Array-like object (list, tuple, numpy array) representing the vector.
        expected_dim: The required vector dimension for the collection.

    Returns:
        A 1D numpy array with dtype=np.float32.

    Raises:
        ValidationError: If input is malformed, has incorrect dimensions, or
            contains NaN / Infinite numerical values.
    """
    if vector is None:
        raise ValidationError("Vector cannot be None")

    try:
        arr = np.array(vector, dtype=np.float32, copy=True)
    except (ValueError, TypeError) as err:
        raise ValidationError(f"Vector contains invalid or non-numeric elements: {err}") from err

    # Ensure vector is strictly 1-dimensional
    if arr.ndim != 1:
        raise ValidationError(
            f"Vector must be a 1-dimensional array, got shape {arr.shape} ({arr.ndim} dimensions)"
        )

    # Ensure length matches collection dimension
    if arr.shape[0] != expected_dim:
        raise ValidationError(
            f"Vector dimension mismatch: expected dimension {expected_dim}, got {arr.shape[0]}"
        )

    # Check for NaN values
    if np.isnan(arr).any():
        raise ValidationError("Vector contains NaN (Not a Number) values")

    # Check for Infinite values
    if np.isinf(arr).any():
        raise ValidationError("Vector contains infinite values (Inf / -Inf)")

    return arr


def validate_payload(payload: Any, max_bytes: int | None = None) -> Payload:
    """Validate metadata payload.

    Args:
        payload: Optional dictionary containing arbitrary metadata.
        max_bytes: Optional maximum allowed size in serialized bytes.

    Returns:
        Validated dictionary payload.

    Raises:
        ValidationError: If payload is not a dictionary or exceeds max_bytes.
    """
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValidationError(
            f"Payload metadata must be a dictionary, got {type(payload).__name__}"
        )
    if max_bytes is not None:
        import json
        try:
            size = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        except (TypeError, ValueError) as err:
            raise ValidationError(
                f"Payload contains non-JSON serializable data: {err}"
            ) from err
        if size > max_bytes:
            raise ValidationError(
                f"Payload size ({size} bytes) exceeds maximum allowed limit of {max_bytes} bytes"
            )
    return payload


def validate_limit(limit: Any, max_limit: int | None = None) -> int:
    """Validate that search limit k is a positive integer > 0 within optional bounds.

    Args:
        limit: Limit parameter to validate.
        max_limit: Optional maximum allowed limit.

    Returns:
        Validated integer limit.

    Raises:
        ValidationError: If limit is not an integer > 0 or exceeds max_limit.
    """
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValidationError(
            f"Search limit 'k' must be an integer, got {type(limit).__name__} ({limit!r})"
        )
    if limit <= 0:
        raise ValidationError(f"Search limit 'k' must be a positive integer > 0, got {limit}")
    if max_limit is not None and limit > max_limit:
        raise ValidationError(
            f"Search limit 'k' ({limit}) exceeds maximum allowed limit of {max_limit}"
        )
    return limit



def validate_metric(metric: Any) -> Any:
    """Validate and normalize distance metric.

    Args:
        metric: Metric string or DistanceMetric enum.

    Returns:
        The normalized DistanceMetric enum.

    Raises:
        ValidationError: If the metric is not supported.
    """
    from minivec.distance import DistanceMetric

    if isinstance(metric, DistanceMetric):
        return metric

    if isinstance(metric, str):
        cleaned = metric.strip().lower()
        for candidate in DistanceMetric:
            if candidate.value == cleaned:
                return candidate

    valid_names = [m.value for m in DistanceMetric]
    raise ValidationError(
        f"Unsupported distance metric '{metric}'. Supported metrics: {valid_names}"
    )

