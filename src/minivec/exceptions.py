"""Custom exceptions for MiniVec.

Providing distinct exception types allows users and calling code to handle
validation failures, duplicate IDs, and missing points with precision.
"""


class MiniVecError(Exception):
    """Base exception class for all MiniVec errors."""
    pass


class ValidationError(MiniVecError):
    """Raised when an input vector, ID, or configuration fails validation."""
    pass


class DuplicateIdError(MiniVecError):
    """Raised when an attempt is made to insert a point with an existing ID."""
    pass


class PointNotFoundError(MiniVecError):
    """Raised when a point with the requested ID does not exist in the collection."""
    pass


class CollectionNotFoundError(MiniVecError):
    """Raised when a requested collection does not exist."""
    pass


class CollectionAlreadyExistsError(MiniVecError):
    """Raised when attempting to create a collection that already exists."""
    pass


class StorageError(MiniVecError):
    """Raised when a database storage, retrieval, or connection operation fails."""
    pass

