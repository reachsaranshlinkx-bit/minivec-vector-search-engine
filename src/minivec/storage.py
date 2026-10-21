"""SQLite persistence repository for MiniVec collections and points.

Provides durable, transactional storage for collections and high-dimensional points
using Python's built-in sqlite3 module.
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import threading
from typing import Any
import numpy as np

from minivec.exceptions import (
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    DuplicateIdError,
    StorageError,
    ValidationError,
)
from minivec.models import Point, PointId


class SQLiteStorage:
    """Manages SQLite persistence for collections and points.

    Attributes:
        db_path: Path to the SQLite database file, or ':memory:'.
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        """Initialize SQLite storage connection and ensure schema tables exist.

        Args:
            db_path: Path to database file or ':memory:' for in-memory database.

        Raises:
            StorageError: If database connection or schema initialization fails.
        """
        self.db_path = str(db_path)
        self._is_memory = self.db_path == ":memory:"
        self._lock = threading.RLock()

        # Ensure parent directory exists for file-based databases
        if not self._is_memory:
            path_obj = Path(self.db_path)
            try:
                path_obj.parent.mkdir(parents=True, exist_ok=True)
            except OSError as err:
                raise StorageError(
                    f"Failed to create directory for database at {path_obj.parent}: {err}"
                ) from err

        try:
            self._conn: sqlite3.Connection | None = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._init_db()
        except sqlite3.Error as err:
            raise StorageError(f"Failed to initialize SQLite database at '{self.db_path}': {err}") from err


    def _init_db(self) -> None:
        """Configure PRAGMAs and create schema tables if they do not exist."""
        # PRAGMA commands must be executed outside of transaction blocks
        self._conn.execute("PRAGMA foreign_keys = ON;")
        if not self._is_memory:
            self._conn.execute("PRAGMA journal_mode = WAL;")
            self._conn.execute("PRAGMA synchronous = NORMAL;")

        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS collections (
                    name TEXT PRIMARY KEY,
                    dimension INTEGER NOT NULL CHECK (dimension > 0),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS points (
                    collection_name TEXT NOT NULL,
                    id TEXT NOT NULL,
                    id_type TEXT NOT NULL CHECK (id_type IN ('int', 'str')),
                    vector BLOB NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (collection_name, id, id_type),
                    FOREIGN KEY (collection_name) REFERENCES collections(name) ON DELETE CASCADE
                );
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_points_collection ON points (collection_name);
                """
            )

    def _check_open(self) -> sqlite3.Connection:
        """Verify the storage connection is open and return it."""
        if self._conn is None:
            raise StorageError(f"Database connection to '{self.db_path}' has been closed")
        return self._conn

    def save_collection(self, name: str, dimension: int) -> None:
        """Persist a new collection record to the database.

        Args:
            name: Unique collection name.
            dimension: Fixed positive integer vector dimensionality.

        Raises:
            CollectionAlreadyExistsError: If a collection with this name already exists.
            StorageError: If database write operation fails.
        """
        with self._lock:
            conn = self._check_open()
            try:
                with conn:
                    conn.execute(
                        "INSERT INTO collections (name, dimension) VALUES (?, ?);",
                        (name, dimension),
                    )
            except sqlite3.IntegrityError as err:
                raise CollectionAlreadyExistsError(
                    f"Collection '{name}' already exists in storage"
                ) from err
            except sqlite3.Error as err:
                raise StorageError(f"Failed to persist collection '{name}': {err}") from err

    def save_point(
        self,
        collection_name: str,
        id: PointId,
        vector: np.ndarray,
        payload: dict[str, Any],
    ) -> None:
        """Persist a point record to the database inside a transaction.

        Args:
            collection_name: Target collection name.
            id: Point identifier (int or str).
            vector: 1D NumPy float32 vector array.
            payload: Metadata dictionary.

        Raises:
            ValidationError: If payload contains non-JSON serializable data.
            DuplicateIdError: If point with same ID already exists in collection.
            CollectionNotFoundError: If target collection does not exist.
            StorageError: If database write operation fails.
        """
        # Validate and serialize payload to JSON
        try:
            payload_json = json.dumps(payload, ensure_ascii=False)
        except (TypeError, ValueError) as err:
            raise ValidationError(
                f"Payload contains non-JSON serializable value: {err}"
            ) from err

        # Serialize vector to compact raw float32 bytes
        vector_blob = np.ascontiguousarray(vector, dtype=np.float32).tobytes()

        # Preserve exact ID type (int vs str)
        id_str = str(id)
        id_type = "int" if isinstance(id, int) and not isinstance(id, bool) else "str"

        with self._lock:
            conn = self._check_open()
            try:
                with conn:
                    conn.execute(
                        """
                        INSERT INTO points (collection_name, id, id_type, vector, payload)
                        VALUES (?, ?, ?, ?, ?);
                        """,
                        (collection_name, id_str, id_type, vector_blob, payload_json),
                    )
            except sqlite3.IntegrityError as err:
                # Check if failure is due to missing foreign key (collection not found)
                cur = conn.execute(
                    "SELECT 1 FROM collections WHERE name = ?;", (collection_name,)
                )
                if cur.fetchone() is None:
                    raise CollectionNotFoundError(
                        f"Collection '{collection_name}' not found in storage"
                    ) from err

                raise DuplicateIdError(
                    f"Point with ID '{id}' already exists in collection '{collection_name}'"
                ) from err
            except sqlite3.Error as err:
                raise StorageError(
                    f"Failed to persist point '{id}' in collection '{collection_name}': {err}"
                ) from err

    def load_collections(self) -> list[tuple[str, int]]:
        """Load all collection definitions from the database.

        Returns:
            List of (name, dimension) tuples ordered by creation time.

        Raises:
            StorageError: If query fails.
        """
        with self._lock:
            conn = self._check_open()
            try:
                cursor = conn.execute(
                    "SELECT name, dimension FROM collections ORDER BY created_at ASC;"
                )
                return [(row["name"], row["dimension"]) for row in cursor.fetchall()]
            except sqlite3.Error as err:
                raise StorageError(f"Failed to load collections from storage: {err}") from err

    def load_points(self, collection_name: str) -> list[Point]:
        """Load all points for a specific collection from the database.

        Args:
            collection_name: The name of the collection.

        Returns:
            List of reconstructed Point instances.

        Raises:
            StorageError: If query fails.
        """
        with self._lock:
            conn = self._check_open()
            try:
                cursor = conn.execute(
                    """
                    SELECT id, id_type, vector, payload
                    FROM points
                    WHERE collection_name = ?
                    ORDER BY created_at ASC;
                    """,
                    (collection_name,),
                )
                points: list[Point] = []
                for row in cursor.fetchall():
                    # Reconstitute typed ID
                    point_id: PointId = int(row["id"]) if row["id_type"] == "int" else str(row["id"])
                    # Reconstitute vector from raw float32 bytes
                    vec = np.frombuffer(row["vector"], dtype=np.float32).copy()
                    # Reconstitute payload from JSON string
                    payload = json.loads(row["payload"])
                    points.append(Point(id=point_id, vector=vec, payload=payload))
                return points
            except sqlite3.Error as err:
                raise StorageError(
                    f"Failed to load points for collection '{collection_name}': {err}"
                ) from err

    def load_all(self) -> dict[str, tuple[int, list[Point]]]:
        """Load all collections and their points in one coordinated pass.

        Returns:
            Dictionary mapping collection_name -> (dimension, list[Point]).

        Raises:
            StorageError: If query fails.
        """
        with self._lock:
            collections_meta = self.load_collections()
            result: dict[str, tuple[int, list[Point]]] = {}
            for name, dim in collections_meta:
                points = self.load_points(name)
                result[name] = (dim, points)
            return result

    def collection_exists(self, name: str) -> bool:
        """Check if a collection exists in storage."""
        with self._lock:
            conn = self._check_open()
            try:
                cur = conn.execute(
                    "SELECT 1 FROM collections WHERE name = ?;", (name,)
                )
                return cur.fetchone() is not None
            except sqlite3.Error as err:
                raise StorageError(f"Failed to check collection existence: {err}") from err

    @property
    def is_closed(self) -> bool:
        """Return True if the database connection is closed."""
        with self._lock:
            return getattr(self, "_conn", None) is None

    def close(self) -> None:
        """Close the SQLite database connection cleanly and idempotently."""
        with self._lock:
            if hasattr(self, "_conn") and self._conn is not None:
                try:
                    self._conn.close()
                except sqlite3.Error:
                    pass
                finally:
                    self._conn = None

