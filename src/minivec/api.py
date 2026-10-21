"""FastAPI application factory and REST API endpoints for MiniVec.

Exposes a clean HTTP interface for collection management, vector upsertion,
and exact similarity search with metadata filtering.
"""

from contextlib import asynccontextmanager
import logging
from pathlib import Path
import time
from typing import Annotated
from fastapi import Depends, FastAPI, Request, status
from fastapi.responses import JSONResponse

from minivec import __version__
from minivec.config import get_settings
from minivec.engine import MiniVecEngine
from minivec.exceptions import (
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    DuplicateIdError,
    PointNotFoundError,
    StorageError,
    ValidationError,
)
from minivec.filter import FieldCondition, Filter
from minivec.schemas import (
    CollectionInfoResponse,
    CollectionListResponse,
    CreateCollectionRequest,
    HealthResponse,
    PointCreateRequest,
    PointCreateResponse,
    PointResponse,
    ScoredPointResponse,
    SearchRequest,
    SearchResponse,
)
from minivec.storage import SQLiteStorage

logger = logging.getLogger("minivec.api")


def get_engine(request: Request) -> MiniVecEngine:
    """Dependency to retrieve the MiniVecEngine from application state."""
    return request.app.state.engine


def create_app(
    engine: MiniVecEngine | None = None,
    db_path: str | Path | None = None,
) -> FastAPI:
    """Application factory to instantiate an isolated MiniVec FastAPI application.

    Args:
        engine: Optional MiniVecEngine instance. If provided, uses this engine directly.
        db_path: Optional SQLite database path. If provided, configures a persistent
            engine backed by this SQLite file. If both engine and db_path are omitted,
            uses the configured database path in non-test environments.

    Returns:
        Configured FastAPI application instance.
    """
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine_type = "persistent" if getattr(app.state.engine, "_storage", None) else "in-memory"
        logger.info(
            "MiniVec REST API starting up (version=%s, engine=%s)",
            __version__,
            engine_type,
        )
        yield
        logger.info("MiniVec REST API shutting down: releasing engine and storage resources")
        if hasattr(app.state, "engine") and app.state.engine is not None:
            app.state.engine.close()

    app = FastAPI(
        title="MiniVec API",
        version=__version__,
        description="Educational Vector Search Engine REST API",
        lifespan=lifespan,
    )

    # Resolve application engine
    if engine is not None:
        app.state.engine = engine
    elif db_path is not None:
        storage = SQLiteStorage(db_path)
        app.state.engine = MiniVecEngine(storage=storage)
    else:
        app.state.engine = MiniVecEngine()

    # Register custom exception handlers for clean, safe HTTP status code mapping
    @app.exception_handler(CollectionNotFoundError)
    async def collection_not_found_handler(request: Request, exc: CollectionNotFoundError):
        logger.warning("Collection not found on %s %s: %s", request.method, request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": str(exc)},
        )

    @app.exception_handler(PointNotFoundError)
    async def point_not_found_handler(request: Request, exc: PointNotFoundError):
        logger.warning("Point not found on %s %s: %s", request.method, request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": str(exc)},
        )

    @app.exception_handler(DuplicateIdError)
    async def duplicate_id_handler(request: Request, exc: DuplicateIdError):
        logger.warning("Duplicate point ID conflict on %s %s: %s", request.method, request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"detail": str(exc)},
        )

    @app.exception_handler(CollectionAlreadyExistsError)
    async def collection_exists_handler(request: Request, exc: CollectionAlreadyExistsError):
        logger.warning("Collection already exists conflict on %s %s: %s", request.method, request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"detail": str(exc)},
        )

    @app.exception_handler(ValidationError)
    async def validation_error_handler(request: Request, exc: ValidationError):
        logger.warning("Validation error on %s %s: %s", request.method, request.url.path, exc)
        return JSONResponse(
            status_code=422,
            content={"detail": str(exc)},
        )

    @app.exception_handler(StorageError)
    async def storage_error_handler(request: Request, exc: StorageError):
        logger.error("Storage error on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Storage operation failed"},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled server error processing %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )

    # -------------------------------------------------------------------------
    # Health Endpoints
    # -------------------------------------------------------------------------
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """Check API operational health status."""
        return HealthResponse(status="ok", version=__version__)

    # -------------------------------------------------------------------------
    # Collection Endpoints
    # -------------------------------------------------------------------------
    @app.post(
        "/collections",
        response_model=CollectionInfoResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["Collections"],
    )
    async def create_collection(
        request_data: CreateCollectionRequest,
        engine: Annotated[MiniVecEngine, Depends(get_engine)],
    ):
        """Create a new vector collection with a fixed dimension."""
        collection = engine.create_collection(
            name=request_data.name,
            dimension=request_data.dimension,
        )
        logger.info("Created collection '%s' (dimension=%d)", collection.name, collection.dimension)
        return CollectionInfoResponse(
            name=collection.name,
            dimension=collection.dimension,
            points_count=collection.count(),
        )

    @app.get(
        "/collections",
        response_model=CollectionListResponse,
        tags=["Collections"],
    )
    async def list_collections(
        engine: Annotated[MiniVecEngine, Depends(get_engine)],
    ):
        """List all collections currently held in memory."""
        collections = engine.list_collections()
        return CollectionListResponse(
            collections=[
                CollectionInfoResponse(
                    name=c.name,
                    dimension=c.dimension,
                    points_count=c.count(),
                )
                for c in collections
            ]
        )

    @app.get(
        "/collections/{name}",
        response_model=CollectionInfoResponse,
        tags=["Collections"],
    )
    async def get_collection_info(
        name: str,
        engine: Annotated[MiniVecEngine, Depends(get_engine)],
    ):
        """Retrieve metadata for a specific collection."""
        collection = engine.get_collection(name)
        return CollectionInfoResponse(
            name=collection.name,
            dimension=collection.dimension,
            points_count=collection.count(),
        )

    # -------------------------------------------------------------------------
    # Point Endpoints
    # -------------------------------------------------------------------------
    @app.post(
        "/collections/{name}/points",
        response_model=PointCreateResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["Points"],
    )
    async def add_point(
        name: str,
        request_data: PointCreateRequest,
        engine: Annotated[MiniVecEngine, Depends(get_engine)],
    ):
        """Insert a point with vector and payload into a collection."""
        collection = engine.get_collection(name)
        collection.add_point(
            id=request_data.id,
            vector=request_data.vector,
            payload=request_data.payload,
        )
        payload_keys = list(request_data.payload.keys()) if request_data.payload else []
        logger.info(
            "Added point '%s' to collection '%s' (payload_keys=%s)",
            request_data.id,
            name,
            payload_keys,
        )
        return PointCreateResponse(status="ok", id=request_data.id)

    @app.get(
        "/collections/{name}/points/{point_id}",
        response_model=PointResponse,
        tags=["Points"],
    )
    async def get_point(
        name: str,
        point_id: str,
        engine: Annotated[MiniVecEngine, Depends(get_engine)],
    ):
        """Retrieve a specific point by ID from a collection."""
        collection = engine.get_collection(name)
        # Attempt lookup by integer if digits, else string; fallback gracefully
        try:
            if point_id.isdigit():
                point = collection.get(int(point_id))
            else:
                point = collection.get(point_id)
        except PointNotFoundError:
            point = collection.get(point_id)

        return PointResponse(
            id=point.id,
            vector=point.vector.tolist(),
            payload=point.payload,
        )

    # -------------------------------------------------------------------------
    # Search Endpoints
    # -------------------------------------------------------------------------
    @app.post(
        "/collections/{name}/search",
        response_model=SearchResponse,
        tags=["Search"],
    )
    async def search_collection(
        name: str,
        request_data: SearchRequest,
        engine: Annotated[MiniVecEngine, Depends(get_engine)],
    ):
        """Search a collection using vector similarity with optional metadata filtering."""
        collection = engine.get_collection(name)

        # Convert schema filter into domain Filter object if provided
        domain_filter = None
        if request_data.filter is not None:
            conditions = [
                FieldCondition(
                    key=cond.key,
                    operator=cond.operator,
                    value=cond.value,
                )
                for cond in request_data.filter.must
            ]
            domain_filter = Filter(must=conditions)

        start_time = time.perf_counter()
        hits = collection.search(
            query=request_data.query,
            limit=request_data.limit,
            metric=request_data.metric,
            filter=domain_filter,
            with_payload=request_data.with_payload,
            with_vector=request_data.with_vector,
        )
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Search on collection '%s': metric=%s, k=%d, filter=%s, hits=%d (%.2f ms)",
            name,
            request_data.metric,
            request_data.limit,
            domain_filter is not None,
            len(hits),
            duration_ms,
        )

        return SearchResponse(
            results=[
                ScoredPointResponse(
                    id=hit.id,
                    score=hit.score,
                    payload=hit.payload,
                    vector=hit.vector.tolist() if hit.vector is not None else None,
                )
                for hit in hits
            ]
        )

    return app


def get_default_app() -> FastAPI:
    """Instantiate application backed by SQLite persistence (or in-memory when testing)."""
    import os
    import sys
    settings = get_settings()
    if "pytest" in sys.modules or os.getenv("PYTEST_CURRENT_TEST") or settings.environment == "test":
        return create_app()

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    storage = SQLiteStorage(settings.db_path)
    engine = MiniVecEngine(storage=storage)
    return create_app(engine=engine)


# Default application instance for Uvicorn
app = get_default_app()



