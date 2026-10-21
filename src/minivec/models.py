"""Data models for MiniVec.

Defines the core data structures for representing points, vectors, and payloads.
"""

from dataclasses import dataclass, field
from typing import Any
import numpy as np

PointId = int | str
Payload = dict[str, Any]


@dataclass(slots=True)
class Point:
    """Represents a single data point in a vector collection.

    Attributes:
        id: Unique identifier for the point (e.g., integer ID or string/UUID).
        vector: Numerical feature embedding represented as a 1D NumPy float32 array.
        payload: Arbitrary metadata dictionary associated with this vector.
    """

    id: PointId
    vector: np.ndarray
    payload: Payload = field(default_factory=dict)

    def __repr__(self) -> str:
        dim = len(self.vector) if hasattr(self.vector, "__len__") else "?"
        return f"Point(id={self.id!r}, dim={dim}, payload={self.payload})"


@dataclass(slots=True)
class ScoredPoint:
    """Represents a search result item ranked by similarity score or distance.

    Attributes:
        id: Unique identifier of the matching point.
        score: The calculated similarity score or distance.
        payload: Metadata dictionary associated with the point.
        vector: Optional vector representation (if requested).
    """

    id: PointId
    score: float
    payload: Payload = field(default_factory=dict)
    vector: np.ndarray | None = None

    def __repr__(self) -> str:
        return f"ScoredPoint(id={self.id!r}, score={self.score:.4f}, payload={self.payload})"


# Convenient alias
SearchResult = ScoredPoint

