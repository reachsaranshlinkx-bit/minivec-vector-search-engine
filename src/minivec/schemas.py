"""Pydantic schemas for MiniVec REST API request and response validation."""

from typing import Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check endpoint response."""

    status: str = Field(default="ok", description="Server health status")
    version: str = Field(default="0.1.0", description="MiniVec version")


class CreateCollectionRequest(BaseModel):
    """Request payload to create a new vector collection."""

    name: str = Field(..., description="Unique collection name", min_length=1)
    dimension: int = Field(..., description="Fixed vector dimensionality", gt=0, le=4096)



class CollectionInfoResponse(BaseModel):
    """Collection metadata and status response."""

    name: str = Field(..., description="Collection name")
    dimension: int = Field(..., description="Vector dimension")
    points_count: int = Field(..., description="Number of points stored")


class CollectionListResponse(BaseModel):
    """List of all collections."""

    collections: list[CollectionInfoResponse] = Field(
        default_factory=list, description="List of collections"
    )


class PointCreateRequest(BaseModel):
    """Request payload to insert a point into a collection."""

    id: int | str = Field(..., description="Unique point identifier")
    vector: list[float] = Field(..., description="High-dimensional embedding vector")
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary metadata dictionary"
    )


class PointCreateResponse(BaseModel):
    """Response returned upon point creation."""

    status: str = Field(default="ok", description="Operation status")
    id: int | str = Field(..., description="ID of the inserted point")


class PointResponse(BaseModel):
    """Retrieved point response model."""

    id: int | str = Field(..., description="Point identifier")
    vector: list[float] = Field(..., description="Embedding vector")
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary metadata dictionary"
    )


class FieldConditionSchema(BaseModel):
    """Schema for a single metadata filter condition."""

    key: str = Field(..., description="Payload metadata field name", min_length=1)
    operator: str = Field(..., description="Comparison operator (eq, ne, gt, gte, lt, lte, in)")
    value: Any = Field(..., description="Target value for comparison")


class FilterSchema(BaseModel):
    """Schema for composite filter with AND semantics."""

    must: list[FieldConditionSchema] = Field(
        default_factory=list, description="Conditions that must all match"
    )


class SearchRequest(BaseModel):
    """Request payload for vector similarity search."""

    query: list[float] = Field(..., description="Query vector matching collection dimension")
    limit: int = Field(default=10, description="Maximum number of results to return", gt=0, le=1000)
    metric: str = Field(

        default="cosine", description="Distance metric: cosine, dot, or euclidean"
    )
    filter: FilterSchema | None = Field(
        default=None, description="Optional metadata filter conditions"
    )
    with_payload: bool = Field(default=True, description="Include payload in response")
    with_vector: bool = Field(default=False, description="Include vector in response")


class ScoredPointResponse(BaseModel):
    """A single ranked search result point."""

    id: int | str = Field(..., description="Point ID")
    score: float = Field(..., description="Similarity score or distance")
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Associated metadata payload"
    )
    vector: list[float] | None = Field(
        default=None, description="Vector array if with_vector was True"
    )


class SearchResponse(BaseModel):
    """List of ranked search results."""

    results: list[ScoredPointResponse] = Field(
        default_factory=list, description="Ranked list of matching points"
    )


class ErrorResponse(BaseModel):
    """Consistent error response format."""

    detail: str = Field(..., description="Human-readable error explanation")
