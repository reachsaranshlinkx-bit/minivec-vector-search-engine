"""Tests for Milestone 1: In-Memory Vector Storage and Validation.

Verifies:
1. Collection creation with dimension constraints.
2. Normal storage and retrieval of points with int and string IDs.
3. Input validation: vector dimension mismatch, NaN/Inf, malformed inputs.
4. Prevention of duplicate point IDs.
5. Error handling for non-existent point retrieval.
"""

import numpy as np
import pytest

from minivec import (
    Collection,
    DuplicateIdError,
    Point,
    PointNotFoundError,
    ValidationError,
)


class TestCollectionInitialization:
    """Tests for creating a Collection with dimension constraints."""

    def test_create_valid_collection(self):
        col = Collection(name="embeddings", dimension=128)
        assert col.name == "embeddings"
        assert col.dimension == 128
        assert col.count() == 0
        assert len(col) == 0

    @pytest.mark.parametrize("invalid_dim", [0, -1, -128])
    def test_reject_non_positive_dimension(self, invalid_dim):
        with pytest.raises(ValidationError, match="positive integer"):
            Collection(name="test", dimension=invalid_dim)

    @pytest.mark.parametrize("invalid_dim", [None, "128", 12.8, True, False])
    def test_reject_non_integer_dimension(self, invalid_dim):
        with pytest.raises(ValidationError, match="must be an integer"):
            Collection(name="test", dimension=invalid_dim)

    @pytest.mark.parametrize("invalid_name", ["", "   ", None])
    def test_reject_empty_collection_name(self, invalid_name):
        with pytest.raises(ValidationError, match="non-empty string"):
            Collection(name=invalid_name, dimension=3)


class TestPointStorageAndRetrieval:
    """Tests for normal storage and retrieval of points."""

    def test_add_and_retrieve_with_integer_id(self):
        col = Collection(name="test_int", dimension=3)
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        payload = {"title": "Test Doc", "tags": ["search", "ai"]}

        col.add_point(id=1, vector=vec, payload=payload)

        assert col.count() == 1
        assert 1 in col

        retrieved = col.get(1)
        assert retrieved.id == 1
        assert retrieved.vector.dtype == np.float32
        assert np.allclose(retrieved.vector, [1.0, 2.0, 3.0])
        assert retrieved.payload == payload

    def test_add_and_retrieve_with_string_id(self):
        col = Collection(name="test_str", dimension=4)
        point = Point(
            id="vec-uuid-001",
            vector=[0.1, 0.2, 0.3, 0.4],
            payload={"category": "test"},
        )
        col.add(point)

        assert col.count() == 1
        assert "vec-uuid-001" in col

        retrieved = col.get("vec-uuid-001")
        assert retrieved.id == "vec-uuid-001"
        assert retrieved.vector.dtype == np.float32
        assert np.allclose(retrieved.vector, [0.1, 0.2, 0.3, 0.4])

    def test_get_optional_returns_none_if_missing(self):
        col = Collection(name="test_opt", dimension=2)
        col.add_point(id=1, vector=[0.5, 0.5])

        assert col.get_optional(1) is not None
        assert col.get_optional(999) is None

    def test_get_all_points(self):
        col = Collection(name="test_all", dimension=2)
        col.add_point(id="a", vector=[1.0, 0.0])
        col.add_point(id="b", vector=[0.0, 1.0])

        all_points = col.get_all()
        assert len(all_points) == 2
        ids = {p.id for p in all_points}
        assert ids == {"a", "b"}


class TestDuplicateHandling:
    """Tests that duplicate point IDs are rejected."""

    def test_reject_duplicate_integer_id(self):
        col = Collection(name="dup_test", dimension=2)
        col.add_point(id=42, vector=[1.0, 2.0])

        with pytest.raises(DuplicateIdError, match="already exists"):
            col.add_point(id=42, vector=[3.0, 4.0])

        # Verify collection still has only the original point
        assert col.count() == 1
        assert np.allclose(col.get(42).vector, [1.0, 2.0])

    def test_reject_duplicate_string_id(self):
        col = Collection(name="dup_test_str", dimension=2)
        col.add_point(id="doc-alpha", vector=[0.1, 0.2])

        with pytest.raises(DuplicateIdError, match="already exists"):
            col.add_point(id="doc-alpha", vector=[0.9, 0.8])


class TestVectorValidation:
    """Tests for vector validation failures."""

    def test_reject_dimension_mismatch_too_short(self):
        col = Collection(name="val_test", dimension=4)
        with pytest.raises(ValidationError, match="dimension mismatch"):
            col.add_point(id=1, vector=[1.0, 2.0, 3.0])  # length 3 instead of 4

    def test_reject_dimension_mismatch_too_long(self):
        col = Collection(name="val_test", dimension=3)
        with pytest.raises(ValidationError, match="dimension mismatch"):
            col.add_point(id=1, vector=[1.0, 2.0, 3.0, 4.0])  # length 4 instead of 3

    def test_reject_nan_in_vector(self):
        col = Collection(name="val_test", dimension=3)
        with pytest.raises(ValidationError, match="NaN"):
            col.add_point(id=1, vector=[1.0, np.nan, 3.0])

    def test_reject_positive_inf_in_vector(self):
        col = Collection(name="val_test", dimension=3)
        with pytest.raises(ValidationError, match="infinite"):
            col.add_point(id=1, vector=[1.0, np.inf, 3.0])

    def test_reject_negative_inf_in_vector(self):
        col = Collection(name="val_test", dimension=3)
        with pytest.raises(ValidationError, match="infinite"):
            col.add_point(id=1, vector=[-np.inf, 2.0, 3.0])

    def test_reject_2d_matrix_as_vector(self):
        col = Collection(name="val_test", dimension=4)
        with pytest.raises(ValidationError, match="1-dimensional"):
            col.add_point(id=1, vector=[[1.0, 2.0], [3.0, 4.0]])

    def test_reject_non_numeric_elements(self):
        col = Collection(name="val_test", dimension=3)
        with pytest.raises(ValidationError, match="invalid or non-numeric"):
            col.add_point(id=1, vector=["one", "two", "three"])

    def test_reject_none_vector(self):
        col = Collection(name="val_test", dimension=3)
        with pytest.raises(ValidationError, match="Vector cannot be None"):
            col.add_point(id=1, vector=None)


class TestIdAndPayloadValidation:
    """Tests for ID and payload validation."""

    @pytest.mark.parametrize("invalid_id", ["", "   ", True, False, 3.14, None])
    def test_reject_invalid_id(self, invalid_id):
        col = Collection(name="val_id", dimension=2)
        with pytest.raises(ValidationError):
            col.add_point(id=invalid_id, vector=[1.0, 2.0])

    def test_reject_non_dict_payload(self):
        col = Collection(name="val_payload", dimension=2)
        with pytest.raises(ValidationError, match="must be a dictionary"):
            col.add_point(id=1, vector=[1.0, 2.0], payload="not-a-dict")

    def test_accept_none_payload_as_empty_dict(self):
        col = Collection(name="none_payload", dimension=2)
        p1 = col.add_point(id=1, vector=[1.0, 2.0], payload=None)
        assert p1.payload == {}
        assert col.get(1).payload == {}

        p2 = Point(id=2, vector=[1.0, 2.0], payload=None)
        col.add(p2)
        assert col.get(2).payload == {}

    @pytest.mark.parametrize("falsy_invalid_payload", [[], False, "", 0, ()])
    def test_reject_falsy_non_dict_payloads(self, falsy_invalid_payload):
        col = Collection(name="falsy_payload", dimension=2)
        with pytest.raises(ValidationError, match="must be a dictionary"):
            col.add_point(id=1, vector=[1.0, 2.0], payload=falsy_invalid_payload)

        with pytest.raises(ValidationError, match="must be a dictionary"):
            col.add(Point(id=2, vector=[1.0, 2.0], payload=falsy_invalid_payload))


class TestPointRetrievalErrors:
    """Tests for retrieving points that do not exist."""

    def test_raise_when_point_not_found(self):
        col = Collection(name="not_found", dimension=2)
        col.add_point(id=1, vector=[0.1, 0.2])

        with pytest.raises(PointNotFoundError, match="not found"):
            col.get(999)


class TestStorageImmutabilityAndIsolation:
    """Tests that stored points cannot be accidentally modified via external references."""

    def test_caller_modifying_original_numpy_array_does_not_affect_storage(self):
        col = Collection(name="iso_vec", dimension=3)
        original_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        col.add_point(id=1, vector=original_arr)

        # Mutate caller's original array
        original_arr[0] = 999.0
        original_arr[1] = -555.0

        # Verify collection retains the initial values
        stored = col.get(1)
        assert np.allclose(stored.vector, [1.0, 2.0, 3.0])

    def test_caller_modifying_point_after_add_does_not_affect_storage(self):
        col = Collection(name="iso_point", dimension=2)
        point = Point(
            id=10,
            vector=np.array([1.0, 2.0], dtype=np.float32),
            payload={"category": "ai", "nested": {"counter": 1}},
        )

        col.add(point)

        # Mutate the original Point object
        point.id = 999
        point.vector[0] = 888.0
        point.payload["category"] = "tampered"
        point.payload["nested"]["counter"] = 999

        # Verify collection still has point with id 10 and original values
        assert 999 not in col
        stored = col.get(10)
        assert stored.id == 10
        assert np.allclose(stored.vector, [1.0, 2.0])
        assert stored.payload["category"] == "ai"
        assert stored.payload["nested"]["counter"] == 1

    def test_modifying_point_from_get_does_not_affect_storage(self):
        col = Collection(name="iso_get", dimension=2)
        col.add_point(
            id="doc-1",
            vector=[1.0, 2.0],
            payload={"tags": ["fast", "search"], "meta": {"depth": 3}},
        )

        # Retrieve and mutate
        retrieved = col.get("doc-1")
        retrieved.vector[0] = 777.0
        retrieved.payload["tags"].append("mutated")
        retrieved.payload["meta"]["depth"] = 999

        # Subsequent retrieval must reflect unmodified state
        check = col.get("doc-1")
        assert np.allclose(check.vector, [1.0, 2.0])
        assert check.payload["tags"] == ["fast", "search"]
        assert check.payload["meta"]["depth"] == 3

    def test_modifying_point_from_get_optional_does_not_affect_storage(self):
        col = Collection(name="iso_get_opt", dimension=2)
        col.add_point(id=1, vector=[0.5, 0.5], payload={"items": [1, 2, 3]})

        retrieved = col.get_optional(1)
        assert retrieved is not None
        retrieved.vector[1] = 99.0
        retrieved.payload["items"].clear()

        check = col.get(1)
        assert np.allclose(check.vector, [0.5, 0.5])
        assert check.payload["items"] == [1, 2, 3]

    def test_modifying_point_from_get_all_does_not_affect_storage(self):
        col = Collection(name="iso_get_all", dimension=2)
        col.add_point(id="a", vector=[1.0, 0.0], payload={"role": "admin"})

        points = col.get_all()
        points[0].vector[0] = 0.0
        points[0].payload["role"] = "guest"

        check = col.get("a")
        assert np.allclose(check.vector, [1.0, 0.0])
        assert check.payload["role"] == "admin"

