"""MiniVec: An educational vector search engine."""

__version__ = "0.1.0"

from minivec.api import create_app
from minivec.collection import Collection
from minivec.config import Settings, get_settings, setup_logging
from minivec.distance import DistanceMetric
from minivec.engine import MiniVecEngine
from minivec.exceptions import (
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    DuplicateIdError,
    MiniVecError,
    PointNotFoundError,
    StorageError,
    ValidationError,
)
from minivec.filter import FieldCondition, Filter, FilterOperator
from minivec.models import Payload, Point, PointId, ScoredPoint, SearchResult
from minivec.storage import SQLiteStorage

__all__ = [
    "__version__",
    "Collection",
    "Point",
    "PointId",
    "Payload",
    "ScoredPoint",
    "SearchResult",
    "DistanceMetric",
    "Filter",
    "FieldCondition",
    "FilterOperator",
    "Settings",
    "get_settings",
    "setup_logging",
    "MiniVecEngine",
    "SQLiteStorage",
    "create_app",
    "MiniVecError",
    "ValidationError",
    "DuplicateIdError",
    "PointNotFoundError",
    "CollectionNotFoundError",
    "CollectionAlreadyExistsError",
    "StorageError",
]


