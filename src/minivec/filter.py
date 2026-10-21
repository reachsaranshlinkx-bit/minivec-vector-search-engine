"""Metadata filtering data structures and evaluation engine for MiniVec.

Provides type-safe, validated query filter representations inspired by vector
database payload filtering paradigms. Supports pre-filtering points prior to
vector similarity calculations.
"""

from collections.abc import Collection as SequenceCollection
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from minivec.exceptions import ValidationError


class FilterOperator(str, Enum):
    """Supported payload comparison operators."""

    EQ = "eq"      # Equal to
    NE = "ne"      # Not equal to
    GT = "gt"      # Greater than
    GTE = "gte"    # Greater than or equal to
    LT = "lt"      # Less than
    LTE = "lte"    # Less than or equal to
    IN = "in"      # Contained in a list/set of values


def _is_number(val: Any) -> bool:
    """Return True if val is int or float, but strictly not a boolean."""
    return isinstance(val, (int, float)) and not isinstance(val, bool)


@dataclass(slots=True)
class FieldCondition:
    """A single filtering condition evaluated against a payload key.

    Attributes:
        key: The metadata field name to inspect.
        operator: Comparison operator (eq, ne, gt, gte, lt, lte, in).
        value: Target value to compare against.
    """

    key: str
    operator: FilterOperator | str
    value: Any

    def __post_init__(self) -> None:
        """Validate condition fields upon instantiation."""
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValidationError(
                f"FieldCondition key must be a non-empty string, got {self.key!r}"
            )
        self.key = self.key.strip()

        # Validate and normalize operator
        if isinstance(self.operator, str):
            cleaned_op = self.operator.strip().lower()
            try:
                self.operator = FilterOperator(cleaned_op)
            except ValueError:
                valid_ops = [op.value for op in FilterOperator]
                raise ValidationError(
                    f"Unsupported filter operator '{self.operator}'. Supported: {valid_ops}"
                )
        elif not isinstance(self.operator, FilterOperator):
            raise ValidationError(
                f"Filter operator must be a string or FilterOperator, got {type(self.operator).__name__}"
            )

        # Validate operator target values
        op = self.operator
        if op in (FilterOperator.GT, FilterOperator.GTE, FilterOperator.LT, FilterOperator.LTE):
            if not (_is_number(self.value) or isinstance(self.value, str)):
                raise ValidationError(
                    f"Operator '{op.value}' requires a numeric or string value, got {type(self.value).__name__} ({self.value!r})"
                )

        if op == FilterOperator.IN:
            if isinstance(self.value, (str, bytes, Mapping)) or not isinstance(
                self.value, SequenceCollection
            ):
                raise ValidationError(
                    f"Operator 'in' requires a list, set, or tuple of values, got {type(self.value).__name__} ({self.value!r})"
                )

    def matches(self, payload: Mapping[str, Any]) -> bool:
        """Evaluate this condition against a point's payload dictionary.

        Missing Key Behavior:
            - For 'ne': Returns True if the key is absent (absence != value).
            - For all other operators ('eq', 'gt', 'gte', 'lt', 'lte', 'in'):
              Returns False if the key is absent.

        Type Mismatch Behavior:
            - If types cannot be reasonably compared (e.g., comparing string
              payload with numeric filter under relational operators), the
              condition evaluates safely to False.

        Args:
            payload: The metadata dictionary of a stored point.

        Returns:
            True if the point satisfies this condition, else False.
        """
        if self.key not in payload:
            # Documented missing key rule:
            # If field is missing, 'ne' is True; everything else is False.
            return self.operator == FilterOperator.NE

        actual = payload[self.key]
        expected = self.value
        op = self.operator

        if op == FilterOperator.EQ:
            # Distinguish bool from int (e.g., True should not match 1)
            if isinstance(actual, bool) or isinstance(expected, bool):
                return isinstance(actual, bool) and isinstance(expected, bool) and actual == expected
            return actual == expected

        if op == FilterOperator.NE:
            if isinstance(actual, bool) or isinstance(expected, bool):
                if not (isinstance(actual, bool) and isinstance(expected, bool)):
                    return True
                return actual != expected
            return actual != expected

        # Relational comparisons: gt, gte, lt, lte
        if op in (FilterOperator.GT, FilterOperator.GTE, FilterOperator.LT, FilterOperator.LTE):
            # Both must be numbers, or both must be strings
            if _is_number(actual) and _is_number(expected):
                if op == FilterOperator.GT:
                    return actual > expected
                if op == FilterOperator.GTE:
                    return actual >= expected
                if op == FilterOperator.LT:
                    return actual < expected
                if op == FilterOperator.LTE:
                    return actual <= expected

            if isinstance(actual, str) and isinstance(expected, str):
                if op == FilterOperator.GT:
                    return actual > expected
                if op == FilterOperator.GTE:
                    return actual >= expected
                if op == FilterOperator.LT:
                    return actual < expected
                if op == FilterOperator.LTE:
                    return actual <= expected

            # Incompatible types cannot satisfy relational condition
            return False

        if op == FilterOperator.IN:
            # If actual value is a list/tuple/set, check if any element matches or whole matches
            if isinstance(actual, SequenceCollection) and not isinstance(actual, (str, bytes)):
                return any(item in expected for item in actual) or actual in expected

            # Boolean type safety: don't let 1 match True in a collection
            if isinstance(actual, bool):
                return any(isinstance(x, bool) and x == actual for x in expected)

            return actual in expected

        return False


@dataclass(slots=True)
class Filter:
    """A collection of conditions combined with logical AND semantics.

    All conditions in `must` must be satisfied for a point to be eligible.

    Attributes:
        must: List of FieldCondition instances.
    """

    must: list[FieldCondition] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate conditions list."""
        if not isinstance(self.must, list):
            raise ValidationError(f"Filter must conditions must be a list, got {type(self.must).__name__}")
        for i, cond in enumerate(self.must):
            if not isinstance(cond, FieldCondition):
                raise ValidationError(
                    f"Condition at index {i} must be a FieldCondition, got {type(cond).__name__}"
                )

    def matches(self, payload: Mapping[str, Any]) -> bool:
        """Return True if all conditions in `must` are satisfied by the payload."""
        return all(cond.matches(payload) for cond in self.must)

    @classmethod
    def by_field(cls, key: str, operator: FilterOperator | str, value: Any) -> "Filter":
        """Convenience factory method to build a single-condition Filter."""
        return cls(must=[FieldCondition(key=key, operator=operator, value=value)])


def validate_filter(filter_candidate: Any) -> Filter | None:
    """Validate and normalize a user-supplied filter input.

    Accepts:
        - None: returns None (no filtering).
        - Filter: returned directly.
        - FieldCondition: wrapped into Filter(must=[cond]).
        - list[FieldCondition]: wrapped into Filter(must=[...]).

    Raises:
        ValidationError: If filter structure is invalid.
    """
    if filter_candidate is None:
        return None

    if isinstance(filter_candidate, Filter):
        return filter_candidate

    if isinstance(filter_candidate, FieldCondition):
        return Filter(must=[filter_candidate])

    if isinstance(filter_candidate, list):
        return Filter(must=filter_candidate)

    raise ValidationError(
        f"Filter must be a Filter, FieldCondition, or list of FieldConditions, got {type(filter_candidate).__name__}"
    )
