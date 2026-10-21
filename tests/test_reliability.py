"""Comprehensive tests for MiniVec Milestone 6 — Reliability, Logging, and Security Hardening.

Covers:
1. Lifespan management and SQLite connection cleanup upon application shutdown.
2. Thread synchronization for concurrent collection writes and reads.
3. Resource limits (max vector dimension, max search limit, max payload bytes).
4. Safe error handling (sanitized 500 error responses without internal tracebacks).
5. Privacy-safe operational logging (no vector numbers or complete payloads logged).
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pytest

from minivec import (
    Collection,
    DistanceMetric,
    MiniVecEngine,
    Point,
    SQLiteStorage,
    StorageError,
    ValidationError,
    create_app,
)
from minivec.config import Settings, get_settings


# =============================================================================
# 1. Database Lifecycle & Resource Cleanup Tests
# =============================================================================
class TestDatabaseLifecycle:
    """Tests for safe startup, shutdown, and resource deallocation."""

    def test_lifespan_closes_storage_on_application_shutdown(self, tmp_path: Path):
        """Verifies that FastAPI lifespan context manager closes SQLite connection on shutdown."""
        db_file = tmp_path / "lifecycle_test.db"
        app = create_app(db_path=db_file)

        # Entering and exiting TestClient triggers lifespan startup and shutdown
        with TestClient(app) as client:
            res = client.post("/collections", json={"name": "test_col", "dimension": 2})
            assert res.status_code == 201
            assert not app.state.engine._storage.is_closed

        # After client context exits, lifespan has executed shutdown
        assert app.state.engine._storage.is_closed
        assert app.state.engine._storage._conn is None

    def test_storage_close_is_idempotent(self, tmp_path: Path):
        """Verifies that calling close() multiple times does not raise exceptions."""
        db_file = tmp_path / "idempotent_close.db"
        storage = SQLiteStorage(db_file)
        assert not storage.is_closed

        storage.close()
        assert storage.is_closed
        assert storage._conn is None

        # Second close should be completely safe and no-op
        storage.close()
        assert storage.is_closed

    def test_storage_operations_fail_cleanly_after_close(self, tmp_path: Path):
        """Verifies that accessing a closed storage instance raises StorageError."""
        db_file = tmp_path / "closed_storage.db"
        storage = SQLiteStorage(db_file)
        storage.close()

        with pytest.raises(StorageError, match="has been closed"):
            storage.save_collection("col", 4)

        with pytest.raises(StorageError, match="has been closed"):
            storage.save_point("col", "p1", np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), {})

        with pytest.raises(StorageError, match="has been closed"):
            storage.load_all()


# =============================================================================
# 2. Concurrency & Synchronization Tests
# =============================================================================
class TestConcurrency:
    """Verifies thread-safety and race-condition prevention."""

    def test_concurrent_point_inserts_no_lost_updates(self, tmp_path: Path):
        """Multiple threads concurrently add unique points; all points must be persisted and counted."""
        db_file = tmp_path / "concurrency_insert.db"
        storage = SQLiteStorage(db_file)
        engine = MiniVecEngine(storage=storage)
        col = engine.create_collection("concurrent_col", dimension=4)

        num_points = 50
        num_workers = 8

        def worker(point_idx: int):
            vec = [float(point_idx), 0.1, 0.2, 0.3]
            col.add_point(id=f"pt_{point_idx}", vector=vec, payload={"idx": point_idx})

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker, i) for i in range(num_points)]
            for fut in as_completed(futures):
                fut.result()

        # In-memory check
        assert col.count() == num_points
        assert len(col.get_all()) == num_points

        # Reopen storage check to ensure durable database updates
        engine.close()
        reloaded_engine = MiniVecEngine(storage=SQLiteStorage(db_file))
        reloaded_col = reloaded_engine.get_collection("concurrent_col")
        assert reloaded_col.count() == num_points
        reloaded_engine.close()

    def test_concurrent_search_during_active_inserts(self):
        """Verifies that search() snapshotting prevents 'dict changed size during iteration'."""
        col = Collection("read_write_col", dimension=2)
        # Prepopulate with 10 points
        for i in range(10):
            col.add_point(id=f"init_{i}", vector=[1.0, 0.0], payload={"tag": "base"})

        insert_count = 40
        search_count = 40

        def inserter(idx: int):
            col.add_point(id=f"dyn_{idx}", vector=[0.0, 1.0], payload={"tag": "dynamic"})

        def searcher():
            hits = col.search(query=[1.0, 0.0], limit=5, metric=DistanceMetric.COSINE)
            assert isinstance(hits, list)

        with ThreadPoolExecutor(max_workers=8) as executor:
            insert_futures = [executor.submit(inserter, i) for i in range(insert_count)]
            search_futures = [executor.submit(searcher) for _ in range(search_count)]
            for fut in as_completed(insert_futures + search_futures):
                fut.result()

        assert col.count() == 10 + insert_count

    def test_concurrent_collection_creation_in_engine(self):
        """Verifies that engine collection creation is synchronized."""
        engine = MiniVecEngine()

        def create_col(name: str):
            try:
                engine.create_collection(name, dimension=3)
                return True
            except Exception:
                return False

        with ThreadPoolExecutor(max_workers=8) as executor:
            # 10 attempts to create "shared_col"
            futures = [executor.submit(create_col, "shared_col") for _ in range(10)]
            results = [f.result() for f in futures]

        # Exactly one thread succeeded in creating "shared_col"
        assert results.count(True) == 1
        assert results.count(False) == 9
        assert engine.count() == 1


# =============================================================================
# 3. Resource & Safety Limits Tests
# =============================================================================
class TestResourceLimits:
    """Verifies configurable limits for dimensions, top-k limit, and payload bytes."""

    def test_reject_dimension_exceeding_max_limit(self):
        """Collection dimension exceeding max_vector_dimension raises ValidationError."""
        settings = get_settings()
        max_dim = settings.max_vector_dimension

        # Up to max_dim is allowed
        col = Collection("valid_dim", dimension=max_dim)
        assert col.dimension == max_dim

        # Exceeding max_dim is rejected
        with pytest.raises(ValidationError, match="exceeds maximum allowed limit"):
            Collection("too_large", dimension=max_dim + 1)

    def test_api_reject_dimension_exceeding_schema_limit(self):
        """REST API rejects collection creation exceeding maximum dimension with HTTP 422."""
        app = create_app()
        client = TestClient(app)
        response = client.post("/collections", json={"name": "huge_dim", "dimension": 5000})
        assert response.status_code == 422

    def test_reject_search_limit_exceeding_max_limit(self):
        """Search limit k exceeding max_search_limit raises ValidationError."""
        settings = get_settings()
        max_k = settings.max_search_limit

        col = Collection("search_col", dimension=2)
        col.add_point("p1", [1.0, 0.0])

        with pytest.raises(ValidationError, match="exceeds maximum allowed limit"):
            col.search(query=[1.0, 0.0], limit=max_k + 1)

    def test_api_reject_search_limit_exceeding_schema_limit(self):
        """REST API rejects search with limit > 1000 with HTTP 422."""
        app = create_app()
        client = TestClient(app)
        client.post("/collections", json={"name": "search_col", "dimension": 2})

        response = client.post(
            "/collections/search_col/search",
            json={"query": [1.0, 0.0], "limit": 1001},
        )
        assert response.status_code == 422

    def test_reject_payload_exceeding_max_bytes(self):
        """Payload exceeding max_payload_bytes raises ValidationError."""
        col = Collection("payload_col", dimension=2)

        # Huge payload > 64KB (65536 bytes)
        large_dict = {"large_text": "x" * 70000}
        with pytest.raises(ValidationError, match="exceeds maximum allowed limit"):
            col.add_point("big_point", [1.0, 0.0], payload=large_dict)


# =============================================================================
# 4. Safe Error Handling & Sanitization Tests
# =============================================================================
class TestSafeErrorHandling:
    """Verifies that API error responses are safe, consistent, and do not expose internal traces."""

    def test_storage_error_returns_sanitized_500_response(self, tmp_path: Path):
        """StorageError must return HTTP 500 with sanitized detail, hiding internal DB paths."""
        db_file = tmp_path / "storage_err.db"
        app = create_app(db_path=db_file)
        client = TestClient(app)

        # Create collection
        client.post("/collections", json={"name": "test_err", "dimension": 2})

        # Artificially close the storage to trigger StorageError on the next write
        app.state.engine._storage.close()

        # Attempt to add point to the closed storage
        response = client.post(
            "/collections/test_err/points",
            json={"id": "pt_err", "vector": [1.0, 0.0]},
        )
        assert response.status_code == 500
        data = response.json()
        assert data["detail"] == "Storage operation failed"
        # Verify internal paths or SQLite tracebacks are not leaked
        assert "sqlite3" not in data["detail"].lower()
        assert str(db_file) not in data["detail"]

    def test_unhandled_exception_returns_generic_500_response(self, monkeypatch):
        """Unhandled exceptions return HTTP 500 with 'Internal server error'."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        # Artificially cause an unexpected error in list_collections
        def broken_list():
            raise RuntimeError("Unexpected internal crash in C-extension or memory corruption")

        monkeypatch.setattr(app.state.engine, "list_collections", broken_list)

        response = client.get("/collections")
        assert response.status_code == 500
        data = response.json()
        assert data["detail"] == "Internal server error"
        # Confirm Python traceback string is NOT in response body
        assert "Traceback" not in response.text
        assert "broken_list" not in response.text


# =============================================================================
# 5. Privacy-Safe Logging Tests
# =============================================================================
class TestPrivacySafeLogging:
    """Verifies audit logging records operational events without logging raw vectors or payload values."""

    def test_logging_does_not_leak_raw_vectors_or_payloads(self, caplog):
        """Operations log point IDs and latencies, but never raw float arrays or secret payload data."""
        app = create_app()
        client = TestClient(app)

        with caplog.at_level(logging.INFO, logger="minivec.api"):
            # 1. Create collection
            client.post("/collections", json={"name": "secure_col", "dimension": 3})

            # 2. Add point with sensitive payload and vector
            secret_token = "SUPER_SECRET_TOKEN_XYZ_98765"
            client.post(
                "/collections/secure_col/points",
                json={
                    "id": "user_42",
                    "vector": [0.1234567, 0.9876543, 0.5555555],
                    "payload": {"auth_token": secret_token, "role": "admin"},
                },
            )

            # 3. Perform search
            client.post(
                "/collections/secure_col/search",
                json={"query": [0.1, 0.9, 0.5], "limit": 2},
            )

        log_text = caplog.text

        # Verify operational events WERE logged
        assert "Created collection 'secure_col'" in log_text
        assert "Added point 'user_42' to collection 'secure_col'" in log_text
        assert "Search on collection 'secure_col'" in log_text

        # Verify sensitive payload contents and raw vector floats WERE NOT logged
        assert secret_token not in log_text
        assert "0.1234567" not in log_text
        assert "0.9876543" not in log_text
        assert "0.5555555" not in log_text
