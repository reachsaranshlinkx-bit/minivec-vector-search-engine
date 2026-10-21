"""Application configuration and logging defaults for MiniVec.

This module provides centralized, type-safe settings management using Pydantic,
and standardized logging configuration. It deliberately avoids any vector storage
or search logic at Milestone 0.
"""

from functools import lru_cache
import logging
import os
from pathlib import Path
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Central configuration for MiniVec.

    Default values can be overridden via environment variables or loaded
    from a local .env file.
    """

    app_name: str = Field(default="MiniVec", description="Name of the application")
    app_version: str = Field(default="0.1.0", description="Semantic version string")
    environment: str = Field(
        default_factory=lambda: os.getenv("APP_ENV", "development"),
        description="Deployment environment (development, test, production)",
    )
    debug: bool = Field(
        default_factory=lambda: os.getenv("DEBUG", "true").lower() in ("true", "1", "yes"),
        description="Enable debug mode and verbose diagnostic logging",
    )
    log_level: str = Field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper(),
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    # Server Defaults
    host: str = Field(
        default_factory=lambda: os.getenv("HOST", "127.0.0.1"),
        description="Host interface for the REST API server",
    )
    port: int = Field(
        default_factory=lambda: int(os.getenv("PORT", "8000")),
        description="Port for the REST API server",
    )

    # Storage Defaults
    data_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("DATA_DIR", "./data")),
        description="Directory for persistent data files",
    )
    database_name: str = Field(
        default_factory=lambda: os.getenv("DATABASE_NAME", "minivec.db"),
        description="SQLite database filename",
    )

    # Resource & Safety Limits
    max_vector_dimension: int = Field(
        default_factory=lambda: int(os.getenv("MAX_VECTOR_DIMENSION", "4096")),
        description="Maximum allowed vector dimension (default 4096)",
        gt=0,
    )
    max_search_limit: int = Field(
        default_factory=lambda: int(os.getenv("MAX_SEARCH_LIMIT", "1000")),
        description="Maximum allowed top-k search limit (default 1000)",
        gt=0,
    )
    max_payload_bytes: int = Field(
        default_factory=lambda: int(os.getenv("MAX_PAYLOAD_BYTES", "65536")),
        description="Maximum allowed serialized payload size in bytes (default 64KB)",
        gt=0,
    )

    @property
    def db_path(self) -> Path:
        """Return the combined Path to the SQLite database file."""
        return self.data_dir / self.database_name



@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton instance of the application settings."""
    return Settings()


def setup_logging(level: str | None = None) -> None:
    """Configure standardized logging output for the application.

    Args:
        level: Optional log level name. If omitted, uses level from Settings.
    """
    settings = get_settings()
    log_level_name = (level or settings.log_level).upper()
    numeric_level = getattr(logging, log_level_name, logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
