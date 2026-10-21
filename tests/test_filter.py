"""Tests for Milestone 3: Metadata Filtering (Payloads).

Verifies:
1. All 7 comparison operators (eq, ne, gt, gte, lt, lte, in).
2. Missing key behavior across operators.
3. Type mismatch safety (graceful False instead of unhandled TypeError).
4. Filter structure validation (malformed keys, operators, non-sequences).
5. Multiple conditions combined with logical AND semantics.
6. Integration with exact vector search (pre-filtering candidates before scoring).
7. Top-k ranking, tie-breaking, and defensive isolation under filtering.
8. Empty collections and zero-match filters.
"""

import numpy as np
import pytest

from minivec import (
    Collection,
    DistanceMetric,
    FieldCondition,
    Filter,
    FilterOperator,
    Point,
    ValidationError,
)


class TestIndividualOperators:
    """Tests for each of the 7 supported comparison operators."""

    def test_operator_eq_scalars(self):
        cond_str = FieldCondition(key="category", operator="eq", value="tech")
        assert cond_str.matches({"category": "tech"}) is True
        assert cond_str.matches({"category": "science"}) is False

        cond_int = FieldCondition(key="status_code", operator="eq", value=200)
        assert cond_int.matches({"status_code": 200}) is True
        assert cond_int.matches({"status_code": 404}) is False

    def test_operator_eq_boolean_distinction(self):
        # In Python, True == 1, but in metadata payloads, booleans and ints must be distinct
        cond_bool = FieldCondition(key="active", operator="eq", value=True)
        assert cond_bool.matches({"active": True}) is True
        assert cond_bool.matches({"active": False}) is False
        assert cond_bool.matches({"active": 1}) is False

    def test_operator_ne(self):
        cond = FieldCondition(key="status", operator="ne", value="archived")
        assert cond.matches({"status": "active"}) is True
        assert cond.matches({"status": "archived"}) is False

    def test_operator_gt_and_gte(self):
        cond_gt = FieldCondition(key="rating", operator="gt", value=4.0)
        assert cond_gt.matches({"rating": 4.5}) is True
        assert cond_gt.matches({"rating": 4.0}) is False
        assert cond_gt.matches({"rating": 3.9}) is False

        cond_gte = FieldCondition(key="rating", operator="gte", value=4.0)
        assert cond_gte.matches({"rating": 4.5}) is True
        assert cond_gte.matches({"rating": 4.0}) is True
        assert cond_gte.matches({"rating": 3.9}) is False

    def test_operator_lt_and_lte(self):
        cond_lt = FieldCondition(key="price", operator="lt", value=50)
        assert cond_lt.matches({"price": 49}) is True
        assert cond_lt.matches({"price": 50}) is False
        assert cond_lt.matches({"price": 51}) is False

        cond_lte = FieldCondition(key="price", operator="lte", value=50)
        assert cond_lte.matches({"price": 49}) is True
        assert cond_lte.matches({"price": 50}) is True
        assert cond_lte.matches({"price": 51}) is False

    def test_operator_in_scalar(self):
        cond = FieldCondition(key="region", operator="in", value=["us-east", "us-west"])
        assert cond.matches({"region": "us-east"}) is True
        assert cond.matches({"region": "eu-central"}) is False

    def test_operator_in_list_payload(self):
        # Payload field is itself a list (e.g. tags)
        cond = FieldCondition(key="tags", operator="in", value=["ai", "database"])
        assert cond.matches({"tags": ["ai", "python"]}) is True
        assert cond.matches({"tags": ["web", "frontend"]}) is False


class TestMissingKeyBehavior:
    """Verifies documented behavior when a payload field is absent."""

    def test_missing_key_evaluates_false_for_positive_operators(self):
        empty_payload = {}

        assert FieldCondition("k", "eq", "v").matches(empty_payload) is False
        assert FieldCondition("k", "gt", 10).matches(empty_payload) is False
        assert FieldCondition("k", "gte", 10).matches(empty_payload) is False
        assert FieldCondition("k", "lt", 10).matches(empty_payload) is False
        assert FieldCondition("k", "lte", 10).matches(empty_payload) is False
        assert FieldCondition("k", "in", [1, 2]).matches(empty_payload) is False

    def test_missing_key_evaluates_true_for_ne(self):
        # If field is missing, it is by definition not equal to the target value
        cond = FieldCondition("missing_field", "ne", "archived")
        assert cond.matches({}) is True
        assert cond.matches({"other_field": 123}) is True


class TestTypeIncompatibilitySafety:
    """Verifies that type mismatches safely evaluate to False instead of crashing."""

    def test_comparing_string_payload_with_numeric_condition(self):
        # Payload has string 'free', condition requires price <= 50
        cond = FieldCondition("price", "lte", 50)
        assert cond.matches({"price": "free"}) is False
        assert cond.matches({"price": None}) is False

    def test_comparing_numeric_payload_with_string_condition(self):
        cond = FieldCondition("version", "gt", "v1.0")
        assert cond.matches({"version": 100}) is False

    def test_comparing_bool_in_in_operator(self):
        cond = FieldCondition("flag", "in", [1, 2, 3])
        # True should not match 1 in collection
        assert cond.matches({"flag": True}) is False


class TestFilterValidation:
    """Tests for validating filter structures and rejecting malformed inputs."""

    @pytest.mark.parametrize("invalid_key", ["", "   ", None, 123])
    def test_reject_invalid_field_key(self, invalid_key):
        with pytest.raises(ValidationError, match="non-empty string"):
            FieldCondition(key=invalid_key, operator="eq", value="val")

    def test_reject_invalid_operator(self):
        with pytest.raises(ValidationError, match="Unsupported filter operator"):
            FieldCondition(key="cat", operator="regex", value=".*")

    @pytest.mark.parametrize("invalid_rel_val", [[1, 2], {"a": 1}, None, True])
    def test_reject_uncomparable_relational_target(self, invalid_rel_val):
        with pytest.raises(ValidationError, match="numeric or string value"):
            FieldCondition(key="num", operator="gt", value=invalid_rel_val)

    @pytest.mark.parametrize("invalid_in_val", [42, "string-not-list", True, None])
    def test_reject_non_sequence_in_target(self, invalid_in_val):
        with pytest.raises(ValidationError, match="list, set, or tuple"):
            FieldCondition(key="tag", operator="in", value=invalid_in_val)

    def test_reject_non_field_condition_in_filter(self):
        with pytest.raises(ValidationError, match="FieldCondition"):
            Filter(must=["not-a-condition"])


class TestMultipleConditionsAndSemantics:
    """Tests combining multiple conditions with AND semantics."""

    def test_and_filter_all_must_match(self):
        flt = Filter(
            must=[
                FieldCondition("category", "eq", "tech"),
                FieldCondition("price", "lte", 50),
                FieldCondition("in_stock", "eq", True),
            ]
        )

        # Satisfies all 3
        assert flt.matches({"category": "tech", "price": 45, "in_stock": True}) is True

        # Fails on price
        assert flt.matches({"category": "tech", "price": 60, "in_stock": True}) is False

        # Fails on category
        assert flt.matches({"category": "home", "price": 45, "in_stock": True}) is False

        # Fails on in_stock
        assert flt.matches({"category": "tech", "price": 45, "in_stock": False}) is False

        # Missing one key
        assert flt.matches({"category": "tech", "price": 45}) is False


class TestPreFilteringWithExactSearch:
    """Tests integrating filters into Collection.search()."""

    @pytest.fixture
    def catalog_collection(self) -> Collection:
        """Create a collection representing a catalog of products."""
        col = Collection(name="catalog", dimension=2)
        # Vector represents query affinity (closer to [1, 0] is higher relevance)
        col.add_point(id=1, vector=[1.0, 0.0], payload={"category": "tech", "price": 100})
        col.add_point(id=2, vector=[0.9, 0.1], payload={"category": "tech", "price": 40})
        col.add_point(id=3, vector=[0.8, 0.2], payload={"category": "books", "price": 20})
        col.add_point(id=4, vector=[0.7, 0.3], payload={"category": "tech", "price": 30})
        col.add_point(id=5, vector=[0.6, 0.4], payload={"category": "books", "price": 50})
        return col

    def test_search_unfiltered_returns_top_by_relevance(self, catalog_collection):
        results = catalog_collection.search(query=[1.0, 0.0], limit=3)
        assert [r.id for r in results] == [1, 2, 3]

    def test_search_with_single_equality_filter(self, catalog_collection):
        # Only books: points 3 and 5
        flt = FieldCondition("category", "eq", "books")
        results = catalog_collection.search(query=[1.0, 0.0], limit=5, filter=flt)

        assert len(results) == 2
        assert [r.id for r in results] == [3, 5]
        for r in results:
            assert r.payload["category"] == "books"

    def test_search_with_multiple_and_filters(self, catalog_collection):
        # category == tech AND price <= 50 -> points 2 (price 40) and 4 (price 30)
        flt = Filter(
            must=[
                FieldCondition("category", "eq", "tech"),
                FieldCondition("price", "lte", 50),
            ]
        )
        results = catalog_collection.search(query=[1.0, 0.0], limit=5, filter=flt)

        assert len(results) == 2
        # Point 2 has vector [0.9, 0.1], Point 4 has [0.7, 0.3]. Point 2 is closer to [1, 0]
        assert [r.id for r in results] == [2, 4]
        assert results[0].payload["price"] == 40
        assert results[1].payload["price"] == 30

    def test_search_filter_with_list_convenience(self, catalog_collection):
        # Passing list of FieldCondition directly
        conditions = [
            FieldCondition("category", "eq", "tech"),
            FieldCondition("price", "lt", 35),
        ]
        results = catalog_collection.search(query=[1.0, 0.0], limit=5, filter=conditions)

        assert len(results) == 1
        assert results[0].id == 4

    def test_search_filter_with_no_matches_returns_empty_list(self, catalog_collection):
        flt = FieldCondition("category", "eq", "non-existent-category")
        results = catalog_collection.search(query=[1.0, 0.0], limit=5, filter=flt)
        assert results == []

    def test_search_filter_on_empty_collection(self):
        col = Collection(name="empty", dimension=2)
        flt = FieldCondition("any", "eq", "val")
        results = col.search(query=[1.0, 0.0], filter=flt)
        assert results == []

    def test_search_filter_preserves_tie_breaking(self):
        col = Collection(name="filter_ties", dimension=2)
        col.add_point(id="zebra", vector=[1.0, 0.0], payload={"tag": "match"})
        col.add_point(id="apple", vector=[1.0, 0.0], payload={"tag": "match"})
        col.add_point(id="other", vector=[1.0, 0.0], payload={"tag": "ignore"})

        flt = FieldCondition("tag", "eq", "match")
        results = col.search(query=[1.0, 0.0], limit=5, filter=flt)

        # Identical vectors: tie broken by ID ascending -> apple, then zebra
        assert [r.id for r in results] == ["apple", "zebra"]

    def test_search_filter_rejects_invalid_filter_type(self, catalog_collection):
        with pytest.raises(ValidationError, match="Filter, FieldCondition, or list"):
            catalog_collection.search(query=[1.0, 0.0], filter="category == tech")
