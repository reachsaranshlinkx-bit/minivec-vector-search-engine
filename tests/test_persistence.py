"""Comprehensive tests for MiniVec Milestone 5 — SQLite Persistence.

Verifies durable storage of collections, typed point IDs, vectors, nested payloads,
reloading across engine restarts, transactional integrity, search/filter continuity,
and REST API persistence using temporary test databases.
"""

from pathlib import Path
import sqlite3
from fastapi.testclient import TestClient
import numpy as np
import pytest

from minivec import (
    Collection,
    DistanceMetric,
    DuplicateIdError,
    FieldCondition,
    Filter,
    MiniVecEngine,
    Point,
    SQLiteStorage,
    StorageError,
    ValidationError,
    create_app,
)
from minivec.exceptions import CollectionAlreadyExistsError


# =============================================================================
# 1. Database Lifecycle & Reopening Tests
# =============================================================================
class TestSQLiteLifecycleAndReload:
    """Tests for closing and reopening SQLite-backed engines."""

    def test_collections_survive_reopening(self, tmp_path: Path):
        db_file = tmp_path / "lifecycle.db"

        # 1. Initialize first engine and create collections
        engine1 = MiniVecEngine(storage=SQLiteStorage(db_file))
        engine1.create_collection("docs", dimension=3)
        engine1.create_collection("images", dimension=128)
        engine1.close()

        # 2. Open a completely new engine instance pointing to the same file
        engine2 = MiniVecEngine(storage=SQLiteStorage(db_file))
        assert engine2.count() == 2
        assert engine2.has_collection("docs")
        assert engine2.has_collection("images")

        docs = engine2.get_collection("docs")
        assert docs.name == "docs"
        assert docs.dimension == 3
        assert docs.count() == 0

        images = engine2.get_collection("images")
        assert images.dimension == 128
        engine2.close()

    def test_points_and_vectors_survive_reopening(self, tmp_path: Path):
        db_file = tmp_path / "points.db"

        # Create collection and insert points
        engine1 = MiniVecEngine(storage=SQLiteStorage(db_file))
        col1 = engine1.create_collection("vectors", dimension=3)
        vec_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        vec_b = np.array([0.0, -1.5, 4.25], dtype=np.float32)

        col1.add_point(id=1, vector=vec_a, payload={"label": "first"})
        col1.add_point(id="p-2", vector=vec_b, payload={"label": "second"})
        engine1.close()

        # Reopen with fresh engine
        engine2 = MiniVecEngine(storage=SQLiteStorage(db_file))
        col2 = engine2.get_collection("vectors")
        assert col2.count() == 2

        p1 = col2.get(1)
        assert p1.id == 1
        assert isinstance(p1.id, int)
        np.testing.assert_allclose(p1.vector, vec_a, rtol=1e-5)
        assert p1.payload == {"label": "first"}

        p2 = col2.get("p-2")
        assert p2.id == "p-2"
        assert isinstance(p2.id, str)
        np.testing.assert_allclose(p2.vector, vec_b, rtol=1e-5)
        assert p2.payload == {"label": "second"}
        engine2.close()


# =============================================================================
# 2. Type Fidelity & Payload Tests
# =============================================================================
class TestTypeFidelityAndPayloads:
    """Verifies that integer vs string IDs and nested payloads are preserved."""

    def test_integer_and_string_ids_remain_distinguishable(self, tmp_path: Path):
        db_file = tmp_path / "typed_ids.db"

        # Same string digits vs integer number
        engine1 = MiniVecEngine(storage=SQLiteStorage(db_file))
        col1 = engine1.create_collection("ids", dimension=2)
        col1.add_point(id=42, vector=[1.0, 0.0], payload={"type": "integer"})
        col1.add_point(id="42", vector=[0.0, 1.0], payload={"type": "string"})
        engine1.close()

        # Reopen and verify exact types
        engine2 = MiniVecEngine(storage=SQLiteStorage(db_file))
        col2 = engine2.get_collection("ids")
        assert col2.count() == 2

        int_pt = col2.get(42)
        str_pt = col2.get("42")

        assert isinstance(int_pt.id, int)
        assert int_pt.id == 42
        assert int_pt.payload["type"] == "integer"

        assert isinstance(str_pt.id, str)
        assert str_pt.id == "42"
        assert str_pt.payload["type"] == "string"
        engine2.close()

    def test_nested_complex_json_payloads_survive(self, tmp_path: Path):
        db_file = tmp_path / "payloads.db"

        complex_payload = {
            "title": "MiniVec Paper",
            "year": 2026,
            "published": True,
            "rating": 4.95,
            "authors": ["Alice", "Bob"],
            "metadata": {
                "citations": 120,
                "keywords": ["ai", "vector-search", "sqlite"],
                "settings": {"active": True, "flags": [1, 2, 3]},
            },
            "empty_list": [],
            "empty_dict": {},
            "none_value": None,
        }

        engine1 = MiniVecEngine(storage=SQLiteStorage(db_file))
        col1 = engine1.create_collection("papers", dimension=2)
        col1.add_point(id="paper-1", vector=[0.5, 0.5], payload=complex_payload)
        engine1.close()

        # Reopen and compare
        engine2 = MiniVecEngine(storage=SQLiteStorage(db_file))
        col2 = engine2.get_collection("papers")
        retrieved = col2.get("paper-1")

        assert retrieved.payload == complex_payload
        assert retrieved.payload["metadata"]["keywords"] == ["ai", "vector-search", "sqlite"]
        assert retrieved.payload["none_value"] is None
        engine2.close()

    def test_unsupported_non_json_payload_fails_safely(self, tmp_path: Path):
        db_file = tmp_path / "invalid_payload.db"
        storage = SQLiteStorage(db_file)
        engine = MiniVecEngine(storage=storage)
        col = engine.create_collection("test", dimension=2)

        # Set or custom function is not JSON serializable
        with pytest.raises(ValidationError) as exc_info:
            col.add_point(
                id=1,
                vector=[1.0, 0.0],
                payload={"bad_value": {1, 2, 3}},  # Python set
            )
        assert "non-json serializable" in str(exc_info.value).lower()

        # Confirm nothing was written to collection or SQLite
        assert col.count() == 0
        points = storage.load_points("test")
        assert len(points) == 0
        engine.close()


# =============================================================================
# 3. Search and Filtering Continuity Tests
# =============================================================================
class TestSearchAndFilteringAfterReload:
    """Verifies that vector similarity and metadata filtering work after reopening."""

    @pytest.fixture
    def populated_db(self, tmp_path: Path) -> Path:
        db_file = tmp_path / "search_continuity.db"
        storage = SQLiteStorage(db_file)
        engine = MiniVecEngine(storage=storage)
        col = engine.create_collection("catalog", dimension=2)

        # A: [1, 0] (tech, price 100)
        col.add_point("A", [1.0, 0.0], {"dept": "tech", "price": 100, "in_stock": True})
        # B: [0, 1] (home, price 50)
        col.add_point("B", [0.0, 1.0], {"dept": "home", "price": 50, "in_stock": True})
        # C: [0.70710678, 0.70710678] (tech, price 200)
        col.add_point("C", [0.70710678, 0.70710678], {"dept": "tech", "price": 200, "in_stock": False})

        engine.close()
        return db_file

    def test_cosine_similarity_after_reload(self, populated_db: Path):
        engine = MiniVecEngine(storage=SQLiteStorage(populated_db))
        col = engine.get_collection("catalog")

        # Query [1, 0]
        hits = col.search(query=[1.0, 0.0], limit=3, metric=DistanceMetric.COSINE)
        assert len(hits) == 3
        assert hits[0].id == "A"
        assert hits[1].id == "C"
        assert hits[2].id == "B"
        engine.close()

    def test_euclidean_distance_after_reload(self, populated_db: Path):
        engine = MiniVecEngine(storage=SQLiteStorage(populated_db))
        col = engine.get_collection("catalog")

        # Query [1, 0]
        hits = col.search(query=[1.0, 0.0], limit=3, metric=DistanceMetric.EUCLIDEAN)
        assert len(hits) == 3
        assert hits[0].id == "A"
        assert hits[0].score == pytest.approx(0.0, abs=1e-5)
        assert hits[1].id == "C"
        assert hits[2].id == "B"
        engine.close()

    def test_metadata_filtering_after_reload(self, populated_db: Path):
        engine = MiniVecEngine(storage=SQLiteStorage(populated_db))
        col = engine.get_collection("catalog")

        # Filter: dept == "tech" AND in_stock == True
        search_filter = Filter(
            must=[
                FieldCondition(key="dept", operator="eq", value="tech"),
                FieldCondition(key="in_stock", operator="eq", value=True),
            ]
        )

        hits = col.search(query=[1.0, 0.0], limit=5, filter=search_filter)
        assert len(hits) == 1
        assert hits[0].id == "A"
        assert hits[0].payload["dept"] == "tech"
        assert hits[0].payload["in_stock"] is True
        engine.close()


# =============================================================================
# 4. Integrity, Constraints, and Rollback Tests
# =============================================================================
class TestIntegrityAndTransactions:
    """Verifies uniqueness constraints and transactional atomicity."""

    def test_duplicate_collection_name_raises(self, tmp_path: Path):
        db_file = tmp_path / "dup_col.db"
        engine = MiniVecEngine(storage=SQLiteStorage(db_file))
        engine.create_collection("unique_name", dimension=3)

        with pytest.raises(CollectionAlreadyExistsError) as exc_info:
            engine.create_collection("unique_name", dimension=3)
        assert "already exists" in str(exc_info.value).lower()
        engine.close()

    def test_duplicate_point_id_raises_and_preserves_original(self, tmp_path: Path):
        db_file = tmp_path / "dup_point.db"
        engine = MiniVecEngine(storage=SQLiteStorage(db_file))
        col = engine.create_collection("items", dimension=2)
        col.add_point(id=1, vector=[1.0, 0.0], payload={"version": 1})

        # Try duplicate
        with pytest.raises(DuplicateIdError) as exc_info:
            col.add_point(id=1, vector=[0.0, 1.0], payload={"version": 2})
        assert "already exists" in str(exc_info.value).lower()

        # Original point must be intact
        original = col.get(1)
        assert original.payload["version"] == 1
        np.testing.assert_allclose(original.vector, [1.0, 0.0])
        engine.close()

    def test_failed_vector_validation_does_not_persist_or_corrupt(self, tmp_path: Path):
        db_file = tmp_path / "rollback.db"
        storage = SQLiteStorage(db_file)
        engine = MiniVecEngine(storage=storage)
        col = engine.create_collection("safe", dimension=3)

        # Invalid vector length (2 instead of 3)
        with pytest.raises(ValidationError):
            col.add_point(id="bad_vec", vector=[1.0, 2.0], payload={"ok": False})

        # Collection in memory and storage must remain empty
        assert col.count() == 0
        assert len(storage.load_points("safe")) == 0
        engine.close()

    def test_separate_database_files_do_not_share_state(self, tmp_path: Path):
        db1 = tmp_path / "db1.db"
        db2 = tmp_path / "db2.db"

        engine1 = MiniVecEngine(storage=SQLiteStorage(db1))
        engine2 = MiniVecEngine(storage=SQLiteStorage(db2))

        engine1.create_collection("only_in_1", dimension=4)

        assert engine1.has_collection("only_in_1")
        assert not engine2.has_collection("only_in_1")
        assert engine2.count() == 0

        engine1.close()
        engine2.close()


# =============================================================================
# 5. REST API Integration with SQLite Persistence
# =============================================================================
class TestAPIWithPersistence:
    """Verifies that FastAPI REST API persists data using an explicit SQLite database file."""

    def test_api_data_persists_across_app_recreation(self, tmp_path: Path):
        db_file = tmp_path / "api_persistence.db"

        # 1. Start App 1 with SQLite storage
        app1 = create_app(db_path=db_file)
        client1 = TestClient(app1)

        # Create collection
        res_create = client1.post("/collections", json={"name": "kb", "dimension": 2})
        assert res_create.status_code == 201

        # Add point
        res_add = client1.post(
            "/collections/kb/points",
            json={"id": "doc-1", "vector": [0.8, 0.6], "payload": {"topic": "ai"}},
        )
        assert res_add.status_code == 201

        # 2. Simulate server restart: Instantiate App 2 pointing to the exact same file
        app2 = create_app(db_path=db_file)
        client2 = TestClient(app2)

        # Verify collection exists with updated count
        res_info = client2.get("/collections/kb")
        assert res_info.status_code == 200
        assert res_info.json()["name"] == "kb"
        assert res_info.json()["points_count"] == 1

        # Search from restarted app
        res_search = client2.post(
            "/collections/kb/search",
            json={"query": [0.8, 0.6], "limit": 1, "metric": "cosine"},
        )
        assert res_search.status_code == 200
        results = res_search.json()["results"]
        assert len(results) == 1
        assert results[0]["id"] == "doc-1"
        assert results[0]["payload"] == {"topic": "ai"}
        assert results[0]["score"] == pytest.approx(1.0, abs=1e-4)
