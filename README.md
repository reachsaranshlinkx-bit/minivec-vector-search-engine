# MiniVec

**MiniVec** is an independent educational vector search engine built from first principles in Python. It is designed to demystify how vector databases (such as Qdrant, Milvus, and Pinecone) store high-dimensional embeddings, filter metadata payloads, execute exact similarity searches, and persist data safely to disk with thread-safe synchronization and robust operational hardening.

---

## 💡 Background & Purpose

During Phase 2 (Project Design), vector database paradigms and architectural concepts were studied by referencing the open-source [Qdrant](https://github.com/qdrant/qdrant) project. MiniVec translates those core concepts into a clean, readable, first-principles Python implementation.

> **Notice:** MiniVec is an independent educational project for study and portfolio demonstration. It does not contain code from Qdrant and is not a direct replacement for production vector search engines. See [NOTICE](NOTICE) for full attribution details.

---

## 🛠️ Technology Stack

- **Python**: 3.11+ (verified on Python 3.12)
- **FastAPI & Uvicorn**: High-performance asynchronous REST API framework
- **NumPy**: Vectorized linear algebra operations and distance calculations (Cosine, Dot Product, Euclidean)
- **Pydantic**: Robust data validation and schema definitions
- **SQLite3**: Standard library persistent storage with WAL mode, atomic transactions, and binary vector serialization
- **pytest & httpx**: Automated test harness for unit, integration, persistence, and concurrency testing
- **uv**: Modern, ultra-fast Python package and project manager

---

## 🗺️ Roadmap & Milestones

- [x] **Milestone 0: Project Scaffolding & Environment Setup**
  - Declarative configuration via `pyproject.toml`
  - `src/` layout packaging
  - Virtual environment & dependency management with `uv`
  - Baseline test harness with `pytest`
- [x] **Milestone 1: Vector Storage & Validation**
  - High-dimensional vector representation with float32 NumPy arrays
  - Dimension consistency validation, NaN/Inf checks
  - In-memory collection data structures with defensive isolation
- [x] **Milestone 2: Exact Similarity Search**
  - Cosine similarity, dot product, and Euclidean distance implementations
  - Deterministic tie-breaking by point ID
  - Top-$k$ retrieval algorithms
- [x] **Milestone 3: Metadata Filtering (Payloads)**
  - Schema-less payload filtering (`eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in`)
  - Efficient pre-filtering strategy combined with AND semantics
- [x] **Milestone 4: REST API with FastAPI**
  - Clean REST endpoints for collection lifecycle, point upsert, and vector search
  - Strict Pydantic validation and consistent HTTP error mapping
  - Isolated test client architecture with multi-instance engine support
- [x] **Milestone 5: SQLite Persistence**
  - Python `sqlite3` storage engine with Write-Ahead Logging (WAL)
  - Normalized schema (`collections` and `points`) with foreign keys
  - Exact float32 binary vector serialization and JSON payload storage
  - Crash resilience, automatic cold-start hydration, and rollback safety
- [x] **Milestone 6: Reliability, Logging, and Security Hardening**
  - Database lifecycle management via FastAPI `@asynccontextmanager` lifespan
  - Fine-grained thread synchronization (`threading.RLock`) for collections, storage, and engine
  - Configurable resource safety bounds (`max_vector_dimension`, `max_search_limit`, `max_payload_bytes`)
  - Sanitized, leak-proof error handling (internal tracebacks and DB paths hidden from clients)
  - Privacy-safe audit logging (logs latencies, point IDs, and operational events without logging raw vector floats or payload data)
- [ ] **Milestone 7: Approximate Nearest-Neighbor Search (ANN) & Recall@k Evaluation**
  - Approximate graph or cluster indexing (e.g. HNSW concepts)
  - Recall@k and query latency benchmarking against exact search baseline

---

## ⚙️ Configuration & Environment Variables

MiniVec provides centralized, type-safe configuration via `pydantic.BaseModel` in `src/minivec/config.py`. All settings have sensible defaults and can be configured through environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `APP_ENV` | `development` | Deployment environment (`development`, `test`, `production`). |
| `DEBUG` | `true` | Enable debug mode and verbose diagnostic logging. |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). |
| `HOST` | `127.0.0.1` | Host interface for REST API (local-only by default for security). |
| `PORT` | `8000` | Port for the REST API server. |
| `DATA_DIR` | `./data` | Directory where SQLite persistent databases are saved. |
| `DATABASE_NAME` | `minivec.db` | SQLite database filename. |
| `MAX_VECTOR_DIMENSION` | `4096` | Maximum allowed vector dimension per collection. |
| `MAX_SEARCH_LIMIT` | `1000` | Maximum top-$k$ search limit allowed per query. |
| `MAX_PAYLOAD_BYTES` | `65536` | Maximum serialized JSON payload size in bytes (64 KB). |

---

## 🚀 Getting Started

### Prerequisites

- [uv](https://docs.astral.sh/uv/) installed on your system.
- Python 3.11 or higher (Python 3.12 recommended).

### 1. Environment Setup

Create the local virtual environment and install dependencies:

```powershell
# Create virtual environment with Python 3.12
uv venv --python 3.12

# Synchronize dependencies and install MiniVec in editable mode
uv sync
```

### 2. Starting the REST API Server

Run the development server via Uvicorn:

```powershell
uv run uvicorn minivec.api:app --host 127.0.0.1 --port 8000 --reload
```

Interactive API documentation (Swagger UI) is available at:
`http://127.0.0.1:8000/docs`

### 3. API Usage Examples (PowerShell)

#### Health Check
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get
```

#### Create a Collection
```powershell
$body = @{ name = "products"; dimension = 3 } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/collections" -Method Post -Body $body -ContentType "application/json"
```

#### Insert Points
```powershell
$point1 = @{
    id = "prod-101"
    vector = @(0.9, 0.1, 0.0)
    payload = @{ category = "electronics"; price = 299.99; in_stock = $true }
} | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/collections/products/points" -Method Post -Body $point1 -ContentType "application/json"
```

#### Vector Search with Metadata Pre-Filtering
```powershell
$searchQuery = @{
    query = @(1.0, 0.0, 0.0)
    limit = 5
    metric = "cosine"
    filter = @{
        must = @(
            @{ key = "category"; operator = "eq"; value = "electronics" },
            @{ key = "price"; operator = "lte"; value = 500 }
        )
    }
    with_payload = $true
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri "http://127.0.0.1:8000/collections/products/search" -Method Post -Body $searchQuery -ContentType "application/json"
```

---

## 🛡️ Reliability & Security Architecture

1. **FastAPI Lifespan Management**:
   The engine and SQLite storage are managed cleanly via `@asynccontextmanager async def lifespan(app: FastAPI)`. When Uvicorn or test runners stop, storage connections and lock files are gracefully closed and flushed without leaking file descriptors.

2. **Thread Concurrency & Synchronization**:
   Internal dictionaries and SQLite write transactions are protected by reentrant locks (`threading.RLock`). Search candidate snapshotting occurs under lock while compute-intensive distance matrix calculations run outside the lock, avoiding write starvation and blocking.

3. **Resource & Safety Limits**:
   Configurable guards prevent denial-of-service or memory exhaustion by rejecting oversized vectors ($> 4096$), top-$k$ search limits ($> 1000$), and payloads ($> 64\text{ KB}$) with HTTP 422 Unprocessable Entity.

4. **Sanitized Error Handling**:
   Custom exception handlers intercept internal errors and `StorageError`, returning clean error responses while hiding database filepaths, table structures, and internal Python tracebacks from clients.

5. **Privacy-Safe Audit Logging**:
   Operational audit logs record timestamps, endpoint paths, collection names, point IDs, hit counts, and execution latency (ms), but strictly omit raw vector floats and payload values to prevent privacy and credential leaks.

---

## 🧪 Testing

Run the full automated test suite (including unit, search, filter, API, persistence, and reliability tests):

```powershell
uv run pytest -v
```

All tests execute against temporary isolated databases (`tmp_path`) and strictly preserve `data/minivec.db`.

---

## 📂 Project Structure

```text
Minivec/
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules (protects data/ and db files)
├── NOTICE                    # Educational attribution notice
├── README.md                 # Project documentation
├── pyproject.toml            # Project metadata and dependencies
├── uv.lock                   # Deterministic lockfile
├── src/
│   └── minivec/
│       ├── __init__.py       # Package exports
│       ├── api.py            # FastAPI application factory & REST endpoints
│       ├── collection.py     # In-memory vector collection & thread safety
│       ├── config.py         # Type-safe Pydantic configuration & logging
│       ├── distance.py       # Vectorized distance & similarity functions
│       ├── engine.py         # Multi-collection engine & lifecycle manager
│       ├── exceptions.py     # Domain exceptions
│       ├── filter.py         # Structured metadata filtering & operators
│       ├── models.py         # Domain models (Point, ScoredPoint, types)
│       ├── schemas.py        # Pydantic request/response schemas
│       ├── storage.py        # SQLite persistence backend (WAL mode)
│       └── validation.py     # Input validation and resource bounds
└── tests/
    ├── __init__.py
    ├── test_api.py           # REST API endpoint tests
    ├── test_filter.py        # Metadata filtering tests
    ├── test_persistence.py   # SQLite persistence & recovery tests
    ├── test_reliability.py   # Lifespan, concurrency, limits & logging tests
    ├── test_search.py        # Exact vector search tests
    ├── test_smoke.py         # Basic smoke tests
    └── test_storage.py       # In-memory vector storage tests
```

---

## 📄 License

This project is licensed under the MIT License.
