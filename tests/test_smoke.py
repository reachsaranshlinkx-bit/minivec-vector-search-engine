"""Smoke test suite for Milestone 0.

Verifies:
1. Python runtime is >= 3.11.
2. NumPy is installed and executes basic numerical vector operations.
3. The MiniVec package and configuration can be imported cleanly.
4. Python's built-in sqlite3 module is accessible for future milestones.
"""

import sqlite3
import sys
import numpy as np
import minivec
from minivec.config import Settings, get_settings


def test_python_version_is_supported():
    """Verify that Python runtime version is at least 3.11."""
    assert sys.version_info >= (3, 11), f"Python 3.11+ required, got {sys.version}"


def test_numpy_vector_operations():
    """Verify NumPy array creation, dot product, and vector norm."""
    vec_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    vec_b = np.array([4.0, 5.0, 6.0], dtype=np.float32)

    assert vec_a.shape == (3,)
    assert vec_a.dtype == np.float32

    # Dot product: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    dot_product = np.dot(vec_a, vec_b)
    assert np.isclose(dot_product, 32.0)

    # L2 Euclidean norm: sqrt(1^2 + 2^2 + 3^2) = sqrt(14) ~= 3.741657
    norm = np.linalg.norm(vec_a)
    assert np.isclose(norm, np.sqrt(14.0))


def test_minivec_package_import_and_defaults():
    """Verify minivec package imports correctly and configuration loads defaults."""
    assert hasattr(minivec, "__version__")
    assert minivec.__version__ == "0.1.0"

    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.app_name == "MiniVec"
    assert settings.port == 8000
    assert settings.database_name == "minivec.db"


def test_sqlite3_stdlib_readiness():
    """Verify built-in sqlite3 is available for future persistence milestones."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute("SELECT 1;")
    result = cursor.fetchone()
    conn.close()
    assert result == (1,)
