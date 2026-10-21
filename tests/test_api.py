"""Comprehensive API tests for MiniVec REST API (Milestone 4).

Tests all endpoints, status codes, schemas, metric computations, filter conversions,
and multi-instance application isolation using FastAPI's TestClient.
"""

import math
import numpy as np
import pytest
from fastapi.testclient import TestClient

from minivec import __version__, create_app
from minivec.engine import MiniVecEngine


@pytest.fixture
def app():
    """Create a fresh, isolated FastAPI app instance for each test."""
    return create_app()


@pytest.fixture
def client(app):
    """Create a TestClient bound to the isolated app instance."""
    return TestClient(app)


# =============================================================================
# 1. Health Endpoint Tests
# =============================================================================
class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_check_returns_ok_and_version(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == __version__


# =============================================================================
# 2. Collection Creation and Listing Tests
# =============================================================================
class TestCollectionsAPI:
    """Tests for /collections endpoints."""

    def test_create_collection_success(self, client: TestClient):
        response = client.post("/collections", json={"name": "docs", "dimension": 3})
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "docs"
        assert data["dimension"] == 3
        assert data["points_count"] == 0

    def test_list_collections_empty_initially(self, client: TestClient):
        response = client.get("/collections")
        assert response.status_code == 200
        assert response.json() == {"collections": []}

    def test_list_multiple_collections(self, client: TestClient):
        client.post("/collections", json={"name": "col1", "dimension": 4})
        client.post("/collections", json={"name": "col2", "dimension": 8})

        response = client.get("/collections")
        assert response.status_code == 200
        collections = response.json()["collections"]
        assert len(collections) == 2
        names = {c["name"] for c in collections}
        assert names == {"col1", "col2"}

    def test_get_collection_info(self, client: TestClient):
        client.post("/collections", json={"name": "products", "dimension": 3})
        # Add a point to verify count reflects updates
        client.post(
            "/collections/products/points",
            json={"id": 1, "vector": [1.0, 2.0, 3.0], "payload": {}},
        )

        response = client.get("/collections/products")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "products"
        assert data["dimension"] == 3
        assert data["points_count"] == 1

    def test_get_nonexistent_collection_returns_404(self, client: TestClient):
        response = client.get("/collections/ghost")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_create_duplicate_collection_returns_409(self, client: TestClient):
        client.post("/collections", json={"name": "unique", "dimension": 2})
        response = client.post("/collections", json={"name": "unique", "dimension": 2})
        assert response.status_code == 409
        assert "already exists" in response.json()["detail"].lower()

    @pytest.mark.parametrize("dim", [0, -1, -128])
    def test_create_collection_rejects_non_positive_dimension(
        self, client: TestClient, dim: int
    ):
        response = client.post("/collections", json={"name": "bad_dim", "dimension": dim})
        assert response.status_code == 422

    @pytest.mark.parametrize("name", ["", "   "])
    def test_create_collection_rejects_empty_name(
        self, client: TestClient, name: str
    ):
        response = client.post("/collections", json={"name": name, "dimension": 4})
        assert response.status_code == 422


# =============================================================================
# 3. Point Insertion and Retrieval Tests
# =============================================================================
class TestPointsAPI:
    """Tests for /collections/{name}/points endpoints."""

    @pytest.fixture(autouse=True)
    def setup_collection(self, client: TestClient):
        client.post("/collections", json={"name": "test_col", "dimension": 3})

    def test_insert_point_with_integer_id(self, client: TestClient):
        response = client.post(
            "/collections/test_col/points",
            json={"id": 42, "vector": [1.0, 0.0, 0.0], "payload": {"tag": "first"}},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "ok"
        assert data["id"] == 42

    def test_insert_point_with_string_id(self, client: TestClient):
        response = client.post(
            "/collections/test_col/points",
            json={"id": "doc-99", "vector": [0.0, 1.0, 0.0], "payload": {}},
        )
        assert response.status_code == 201
        assert response.json()["id"] == "doc-99"

    def test_insert_point_default_empty_payload(self, client: TestClient):
        response = client.post(
            "/collections/test_col/points",
            json={"id": 1, "vector": [1.0, 2.0, 3.0]},
        )
        assert response.status_code == 201

        # Retrieve and verify payload defaults to {}
        get_res = client.get("/collections/test_col/points/1")
        assert get_res.status_code == 200
        assert get_res.json()["payload"] == {}

    def test_get_inserted_point_by_id(self, client: TestClient):
        client.post(
            "/collections/test_col/points",
            json={"id": "item-1", "vector": [0.5, 0.5, 0.5], "payload": {"color": "blue"}},
        )
        response = client.get("/collections/test_col/points/item-1")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "item-1"
        assert data["vector"] == [0.5, 0.5, 0.5]
        assert data["payload"] == {"color": "blue"}

    def test_get_nonexistent_point_returns_404(self, client: TestClient):
        response = client.get("/collections/test_col/points/missing_id")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_insert_duplicate_point_id_returns_409(self, client: TestClient):
        client.post(
            "/collections/test_col/points",
            json={"id": 100, "vector": [1.0, 1.0, 1.0]},
        )
        response = client.post(
            "/collections/test_col/points",
            json={"id": 100, "vector": [2.0, 2.0, 2.0]},
        )
        assert response.status_code == 409
        assert "already exists" in response.json()["detail"].lower()

    def test_insert_to_nonexistent_collection_returns_404(self, client: TestClient):
        response = client.post(
            "/collections/does_not_exist/points",
            json={"id": 1, "vector": [1.0, 2.0, 3.0]},
        )
        assert response.status_code == 404

    def test_insert_vector_dimension_mismatch_returns_422(self, client: TestClient):
        # Collection expects dimension 3, send 2
        response = client.post(
            "/collections/test_col/points",
            json={"id": 1, "vector": [1.0, 2.0]},
        )
        assert response.status_code == 422
        assert "dimension" in response.json()["detail"].lower()

    def test_insert_vector_with_non_numeric_or_nan_returns_422(
        self, client: TestClient
    ):
        # Non-numeric element in vector
        res_str = client.post(
            "/collections/test_col/points",
            json={"id": 1, "vector": [1.0, "not_a_number", 0.0]},
        )
        assert res_str.status_code == 422

        # Raw body containing NaN literal
        res_nan = client.post(
            "/collections/test_col/points",
            content='{"id": 1, "vector": [1.0, NaN, 0.0]}',
            headers={"Content-Type": "application/json"},
        )
        assert res_nan.status_code == 422


# =============================================================================
# 4. Search API Tests (Metrics & Options)
# =============================================================================
class TestSearchAPI:
    """Tests for POST /collections/{name}/search."""

    @pytest.fixture(autouse=True)
    def setup_search_data(self, client: TestClient):
        client.post("/collections", json={"name": "search_col", "dimension": 2})
        # Point A: [1, 0]
        client.post(
            "/collections/search_col/points",
            json={"id": "A", "vector": [1.0, 0.0], "payload": {"category": "tech", "price": 100}},
        )
        # Point B: [0, 1]
        client.post(
            "/collections/search_col/points",
            json={"id": "B", "vector": [0.0, 1.0], "payload": {"category": "home", "price": 50}},
        )
        # Point C: [0.70710678, 0.70710678] (45 degrees)
        client.post(
            "/collections/search_col/points",
            json={"id": "C", "vector": [0.70710678, 0.70710678], "payload": {"category": "tech", "price": 200}},
        )

    def test_search_cosine_similarity(self, client: TestClient):
        # Query [1, 0]: A should be 1.0, C should be ~0.707, B should be 0.0
        response = client.post(
            "/collections/search_col/search",
            json={
                "query": [1.0, 0.0],
                "limit": 3,
                "metric": "cosine",
            },
        )
        assert response.status_code == 200
        results = response.json()["results"]
        assert len(results) == 3
        assert results[0]["id"] == "A"
        assert math.isclose(results[0]["score"], 1.0, abs_tol=1e-4)
        assert results[1]["id"] == "C"
        assert math.isclose(results[1]["score"], 0.7071, abs_tol=1e-3)
        assert results[2]["id"] == "B"
        assert math.isclose(results[2]["score"], 0.0, abs_tol=1e-4)

    def test_search_dot_product(self, client: TestClient):
        response = client.post(
            "/collections/search_col/search",
            json={
                "query": [2.0, 0.0],
                "limit": 3,
                "metric": "dot",
            },
        )
        assert response.status_code == 200
        results = response.json()["results"]
        # Dot product with [2, 0]: A=2.0, C=1.414, B=0.0
        assert results[0]["id"] == "A"
        assert math.isclose(results[0]["score"], 2.0, abs_tol=1e-4)
        assert results[1]["id"] == "C"
        assert results[2]["id"] == "B"

    def test_search_euclidean_distance(self, client: TestClient):
        response = client.post(
            "/collections/search_col/search",
            json={
                "query": [1.0, 0.0],
                "limit": 3,
                "metric": "euclidean",
            },
        )
        assert response.status_code == 200
        results = response.json()["results"]
        # Euclidean distance: A distance is 0.0 (closest), then C, then B
        assert results[0]["id"] == "A"
        assert math.isclose(results[0]["score"], 0.0, abs_tol=1e-4)
        assert results[1]["id"] == "C"
        assert results[2]["id"] == "B"

    def test_search_with_payload_flag(self, client: TestClient):
        # with_payload=False -> payload should be {}
        res_no_payload = client.post(
            "/collections/search_col/search",
            json={"query": [1.0, 0.0], "limit": 1, "with_payload": False},
        )
        assert res_no_payload.status_code == 200
        assert res_no_payload.json()["results"][0]["payload"] == {}

        # with_payload=True -> payload should be included
        res_with_payload = client.post(
            "/collections/search_col/search",
            json={"query": [1.0, 0.0], "limit": 1, "with_payload": True},
        )
        assert res_with_payload.status_code == 200
        assert res_with_payload.json()["results"][0]["payload"] == {"category": "tech", "price": 100}

    def test_search_with_vector_flag(self, client: TestClient):
        # with_vector=False -> vector should be None
        res_no_vec = client.post(
            "/collections/search_col/search",
            json={"query": [1.0, 0.0], "limit": 1, "with_vector": False},
        )
        assert res_no_vec.status_code == 200
        assert res_no_vec.json()["results"][0]["vector"] is None

        # with_vector=True -> vector should be list
        res_with_vec = client.post(
            "/collections/search_col/search",
            json={"query": [1.0, 0.0], "limit": 1, "with_vector": True},
        )
        assert res_with_vec.status_code == 200
        assert res_with_vec.json()["results"][0]["vector"] == [1.0, 0.0]

    def test_search_limit_clamping(self, client: TestClient):
        response = client.post(
            "/collections/search_col/search",
            json={"query": [1.0, 0.0], "limit": 1},
        )
        assert response.status_code == 200
        assert len(response.json()["results"]) == 1


# =============================================================================
# 5. Metadata Filter Search Tests
# =============================================================================
class TestSearchWithFiltersAPI:
    """Tests for metadata filtering in POST /collections/{name}/search."""

    @pytest.fixture(autouse=True)
    def setup_data(self, client: TestClient):
        client.post("/collections", json={"name": "items", "dimension": 2})
        client.post(
            "/collections/items/points",
            json={"id": 1, "vector": [1.0, 0.0], "payload": {"dept": "books", "rating": 4.5, "stock": 10}},
        )
        client.post(
            "/collections/items/points",
            json={"id": 2, "vector": [0.9, 0.1], "payload": {"dept": "books", "rating": 3.0, "stock": 0}},
        )
        client.post(
            "/collections/items/points",
            json={"id": 3, "vector": [0.8, 0.2], "payload": {"dept": "tech", "rating": 4.8, "stock": 5}},
        )

    def test_filter_eq_operator(self, client: TestClient):
        response = client.post(
            "/collections/items/search",
            json={
                "query": [1.0, 0.0],
                "limit": 10,
                "filter": {"must": [{"key": "dept", "operator": "eq", "value": "books"}]},
            },
        )
        assert response.status_code == 200
        results = response.json()["results"]
        ids = [r["id"] for r in results]
        assert ids == [1, 2]

    def test_filter_gte_operator(self, client: TestClient):
        response = client.post(
            "/collections/items/search",
            json={
                "query": [1.0, 0.0],
                "limit": 10,
                "filter": {"must": [{"key": "rating", "operator": "gte", "value": 4.0}]},
            },
        )
        assert response.status_code == 200
        results = response.json()["results"]
        ids = [r["id"] for r in results]
        assert set(ids) == {1, 3}

    def test_filter_in_operator(self, client: TestClient):
        response = client.post(
            "/collections/items/search",
            json={
                "query": [1.0, 0.0],
                "limit": 10,
                "filter": {"must": [{"key": "dept", "operator": "in", "value": ["tech", "music"]}]},
            },
        )
        assert response.status_code == 200
        results = response.json()["results"]
        assert len(results) == 1
        assert results[0]["id"] == 3

    def test_filter_multiple_and_conditions(self, client: TestClient):
        # dept == "books" AND stock > 0
        response = client.post(
            "/collections/items/search",
            json={
                "query": [1.0, 0.0],
                "limit": 10,
                "filter": {
                    "must": [
                        {"key": "dept", "operator": "eq", "value": "books"},
                        {"key": "stock", "operator": "gt", "value": 0},
                    ]
                },
            },
        )
        assert response.status_code == 200
        results = response.json()["results"]
        assert len(results) == 1
        assert results[0]["id"] == 1

    def test_filter_no_matches_returns_empty_results(self, client: TestClient):
        response = client.post(
            "/collections/items/search",
            json={
                "query": [1.0, 0.0],
                "limit": 10,
                "filter": {"must": [{"key": "dept", "operator": "eq", "value": "apparel"}]},
            },
        )
        assert response.status_code == 200
        assert response.json()["results"] == []


# =============================================================================
# 6. Error Handling & Validation Tests
# =============================================================================
class TestErrorsAndValidationAPI:
    """Tests for bad inputs, missing collections, duplicate IDs, and status codes."""

    def test_search_nonexistent_collection_returns_404(self, client: TestClient):
        response = client.post(
            "/collections/nonexistent/search",
            json={"query": [1.0, 0.0], "limit": 5},
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_search_query_dimension_mismatch_returns_422(self, client: TestClient):
        client.post("/collections", json={"name": "dim3", "dimension": 3})
        # Collection dimension is 3, query provides 2 elements
        response = client.post(
            "/collections/dim3/search",
            json={"query": [1.0, 0.0], "limit": 5},
        )
        assert response.status_code == 422
        assert "dimension" in response.json()["detail"].lower()

    def test_search_invalid_metric_returns_422(self, client: TestClient):
        client.post("/collections", json={"name": "metrics_col", "dimension": 2})
        response = client.post(
            "/collections/metrics_col/search",
            json={"query": [1.0, 0.0], "limit": 5, "metric": "manhattan"},
        )
        assert response.status_code == 422
        assert "metric" in response.json()["detail"].lower()

    def test_search_invalid_filter_operator_returns_422(self, client: TestClient):
        client.post("/collections", json={"name": "filter_col", "dimension": 2})
        response = client.post(
            "/collections/filter_col/search",
            json={
                "query": [1.0, 0.0],
                "limit": 5,
                "filter": {"must": [{"key": "tag", "operator": "contains", "value": "x"}]},
            },
        )
        assert response.status_code == 422
        assert "operator" in response.json()["detail"].lower()

    def test_search_empty_collection_returns_empty_results(self, client: TestClient):
        client.post("/collections", json={"name": "empty_col", "dimension": 2})
        response = client.post(
            "/collections/empty_col/search",
            json={"query": [1.0, 0.0], "limit": 5},
        )
        assert response.status_code == 200
        assert response.json()["results"] == []

    @pytest.mark.parametrize("limit", [0, -1, -10])
    def test_search_rejects_non_positive_limit(self, client: TestClient, limit: int):
        client.post("/collections", json={"name": "limit_col", "dimension": 2})
        response = client.post(
            "/collections/limit_col/search",
            json={"query": [1.0, 0.0], "limit": limit},
        )
        assert response.status_code == 422

    def test_invalid_json_body_returns_422(self, client: TestClient):
        response = client.post(
            "/collections",
            content="this is not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422


# =============================================================================
# 7. Application Instance Isolation Tests
# =============================================================================
class TestAppInstanceIsolation:
    """Verifies that separate create_app() instances do not share state."""

    def test_instances_are_isolated(self):
        app1 = create_app()
        app2 = create_app()

        client1 = TestClient(app1)
        client2 = TestClient(app2)

        # Create collection in app1 only
        client1.post("/collections", json={"name": "only_in_1", "dimension": 3})

        # app1 must have it
        res1 = client1.get("/collections")
        assert len(res1.json()["collections"]) == 1
        assert res1.json()["collections"][0]["name"] == "only_in_1"

        # app2 must NOT have it
        res2 = client2.get("/collections")
        assert len(res2.json()["collections"]) == 0

        # Looking up in app2 must return 404
        lookup2 = client2.get("/collections/only_in_1")
        assert lookup2.status_code == 404

    def test_explicit_engine_injection(self):
        shared_engine = MiniVecEngine()
        app = create_app(engine=shared_engine)
        client = TestClient(app)

        # Pre-populate via engine directly
        shared_engine.create_collection("engine_col", dimension=4)

        # Verify API reflects it
        res = client.get("/collections/engine_col")
        assert res.status_code == 200
        assert res.json()["name"] == "engine_col"
        assert res.json()["dimension"] == 4
