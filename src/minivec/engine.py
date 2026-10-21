from __future__ import annotations

import threading
from typing import TYPE_CHECKING
from minivec.collection import Collection
from minivec.exceptions import CollectionAlreadyExistsError, CollectionNotFoundError

if TYPE_CHECKING:
    from minivec.storage import SQLiteStorage


class MiniVecEngine:
    """Manages collections in memory with optional SQLite persistence."""

    def __init__(self, storage: SQLiteStorage | None = None) -> None:
        """Initialize engine, optionally backed by SQLite persistent storage.

        Args:
            storage: Optional SQLiteStorage instance for durable persistence.
        """
        self._storage = storage
        self._lock = threading.RLock()
        self._collections: dict[str, Collection] = {}
        if self._storage is not None:
            self._load_from_storage()

    def _load_from_storage(self) -> None:
        """Hydrate in-memory collections and points from SQLite storage."""
        with self._lock:
            if self._storage is None:
                return

            loaded_data = self._storage.load_all()
            for name, (dimension, points) in loaded_data.items():
                collection = Collection(name=name, dimension=dimension, storage=self._storage)
                for point in points:
                    collection._hydrate_point(point)
                self._collections[collection.name] = collection

    def create_collection(self, name: str, dimension: int) -> Collection:
        """Create and store a new collection.

        Args:
            name: Human-readable unique identifier for the collection.
            dimension: Fixed positive integer vector dimensionality.

        Returns:
            The created Collection instance.

        Raises:
            CollectionAlreadyExistsError: If a collection with this name exists.
            ValidationError: If name or dimension is invalid.
        """
        cleaned_name = name.strip() if isinstance(name, str) else name

        with self._lock:
            if cleaned_name in self._collections:
                raise CollectionAlreadyExistsError(
                    f"Collection '{cleaned_name}' already exists"
                )

            # Validate name and dimension by initializing Collection instance first
            collection = Collection(name=name, dimension=dimension, storage=self._storage)

            # If persistent storage is attached, save to SQLite before caching in memory
            if self._storage is not None:
                self._storage.save_collection(collection.name, collection.dimension)

            self._collections[collection.name] = collection
            return collection

    def get_collection(self, name: str) -> Collection:
        """Retrieve an existing collection by name.

        Args:
            name: The name of the collection to retrieve.

        Returns:
            The Collection instance.

        Raises:
            CollectionNotFoundError: If no collection with this name exists.
        """
        cleaned_name = name.strip() if isinstance(name, str) else name
        with self._lock:
            if cleaned_name not in self._collections:
                raise CollectionNotFoundError(f"Collection '{cleaned_name}' not found")
            return self._collections[cleaned_name]

    def list_collections(self) -> list[Collection]:
        """Return a list of all collections managed by the engine."""
        with self._lock:
            return list(self._collections.values())

    def has_collection(self, name: str) -> bool:
        """Return True if a collection with the given name exists."""
        cleaned_name = name.strip() if isinstance(name, str) else name
        with self._lock:
            return cleaned_name in self._collections

    def count(self) -> int:
        """Return the total number of collections."""
        with self._lock:
            return len(self._collections)

    def close(self) -> None:
        """Close storage backend if present."""
        with self._lock:
            if self._storage is not None:
                self._storage.close()
