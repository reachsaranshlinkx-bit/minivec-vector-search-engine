"""Deterministic end-to-end demonstration of MiniVec.

Showcases:
1. Initializing an isolated persistent SQLite storage engine.
2. Creating a collection and inserting points with high-dimensional vectors and metadata payloads.
3. Performing exact cosine similarity search (ground truth).
4. Constructing an educational HNSW graph index and running approximate nearest-neighbor search.
5. Performing pre-filtered vector similarity search with compound metadata conditions.
6. Simulating a crash / restart by reopening the SQLite database and verifying full persistence recovery.

Guarantees complete database isolation by operating inside a temporary directory.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import numpy as np

from minivec import (
    DistanceMetric,
    FieldCondition,
    Filter,
    MiniVecEngine,
    SQLiteStorage,
)


def run_demo() -> None:
    """Execute the MiniVec demonstration workflow."""
    print("=" * 72)
    print("MiniVec — Educational Vector Search Engine Demonstration")
    print("=" * 72)

    # Use an isolated temporary directory to protect production/development data
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "demo_minivec.db"
        print(f"\n[1] Initializing isolated SQLite storage at: {db_path.name}")

        # Step 1: Initialize engine with persistent SQLite storage
        storage = SQLiteStorage(db_path)
        engine = MiniVecEngine(storage=storage)

        # Step 2: Create a collection of 4-dimensional vectors
        col_name = "products"
        dimension = 4
        print(f"[2] Creating collection '{col_name}' (dimension={dimension})...")
        col = engine.create_collection(col_name, dimension=dimension)

        # Step 3: Insert sample product vectors with rich metadata payloads
        sample_products = [
            (
                "prod_101",
                [0.95, 0.10, 0.05, 0.00],
                {"title": "Noise-Cancelling Headphones", "category": "electronics", "price": 249.99, "in_stock": True},
            ),
            (
                "prod_102",
                [0.90, 0.15, 0.08, 0.02],
                {"title": "Wireless Bluetooth Earbuds", "category": "electronics", "price": 129.50, "in_stock": True},
            ),
            (
                "prod_103",
                [0.85, 0.20, 0.10, 0.05],
                {"title": "High-End Smart TV (4K OLED)", "category": "electronics", "price": 1199.00, "in_stock": False},
            ),
            (
                "prod_104",
                [0.10, 0.90, 0.10, 0.05],
                {"title": "Ergonomic Mesh Office Chair", "category": "furniture", "price": 199.99, "in_stock": True},
            ),
            (
                "prod_105",
                [0.15, 0.85, 0.12, 0.08],
                {"title": "Solid Oak Standing Desk", "category": "furniture", "price": 450.00, "in_stock": True},
            ),
            (
                "prod_106",
                [0.05, 0.10, 0.92, 0.15],
                {"title": "AI Engineering Principles Book", "category": "books", "price": 49.99, "in_stock": True},
            ),
        ]

        print(f"[3] Ingesting {len(sample_products)} sample product embeddings and payloads...")
        for p_id, vec, payload in sample_products:
            col.add_point(id=p_id, vector=vec, payload=payload)
        print(f"    Collection '{col.name}' now contains {col.count()} points.")

        # Step 4: Run exact cosine similarity search (Ground Truth)
        # Query vector representing an audio / personal electronics interest profile
        query_vector = [0.92, 0.12, 0.06, 0.01]
        print("\n[4] Running Exact Cosine Similarity Search (Flat scan baseline)...")
        print(f"    Query vector: {query_vector}")
        exact_results = col.search(
            query=query_vector,
            limit=3,
            metric=DistanceMetric.COSINE,
            search_type="exact",
        )
        for rank, hit in enumerate(exact_results, start=1):
            title = hit.payload.get("title", "Unknown")
            price = hit.payload.get("price", 0.0)
            print(f"    Rank #{rank}: ID={hit.id} | Score={hit.score:.4f} | {title} (${price:.2f})")

        # Step 5: Construct HNSW index and run approximate search
        print("\n[5] Building HNSW Graph Index for Cosine Metric (m=16, ef_construction=64)...")
        hnsw_index = col.create_index(
            metric=DistanceMetric.COSINE,
            m=16,
            ef_construction=64,
            ef_search=32,
            random_seed=42,
        )
        print(f"    HNSW index constructed successfully. Indexed points: {hnsw_index.count()}")

        print("    Running Approximate HNSW Search (search_type='approx')...")
        approx_results = col.search(
            query=query_vector,
            limit=3,
            metric=DistanceMetric.COSINE,
            search_type="approx",
        )
        for rank, hit in enumerate(approx_results, start=1):
            title = hit.payload.get("title", "Unknown")
            price = hit.payload.get("price", 0.0)
            print(f"    Rank #{rank}: ID={hit.id} | Score={hit.score:.4f} | {title} (${price:.2f})")

        # Compare exact vs approximate match
        exact_ids = [h.id for h in exact_results]
        approx_ids = [h.id for h in approx_results]
        match = exact_ids == approx_ids
        print(f"    Exact IDs:  {exact_ids}")
        print(f"    Approx IDs: {approx_ids}")
        print(f"    Top-3 Identity Match: {'PERFECT (100% agreement)' if match else 'PARTIAL'}")

        # Step 6: Metadata pre-filtered search
        # Find electronics with price <= 300
        print("\n[6] Running Pre-Filtered Search:")
        print("    Filter: category == 'electronics' AND price <= 300")
        filter_spec = Filter(
            must=[
                FieldCondition(key="category", operator="eq", value="electronics"),
                FieldCondition(key="price", operator="lte", value=300.0),
            ]
        )
        filtered_results = col.search(
            query=query_vector,
            limit=5,
            metric=DistanceMetric.COSINE,
            filter=filter_spec,
            search_type="approx",  # Safely triggers exact pre-filtered fallback
        )
        print(f"    Returned {len(filtered_results)} eligible matches (TV excluded by price constraint):")
        for rank, hit in enumerate(filtered_results, start=1):
            title = hit.payload.get("title", "Unknown")
            cat = hit.payload.get("category", "")
            price = hit.payload.get("price", 0.0)
            print(f"    Rank #{rank}: ID={hit.id} | Score={hit.score:.4f} | [{cat}] {title} (${price:.2f})")

        # Step 7: Test persistence recovery across crash / restart
        print("\n[7] Simulating Server Restart & SQLite Persistence Recovery...")
        print("    Closing initial engine and flushing SQLite connection...")
        engine.close()

        print("    Re-opening storage backend from disk file...")
        recovered_storage = SQLiteStorage(db_path)
        recovered_engine = MiniVecEngine(storage=recovered_storage)

        recovered_col = recovered_engine.get_collection(col_name)
        print(f"    Recovered collection '{recovered_col.name}' with {recovered_col.count()} points intact.")
        p101 = recovered_col.get("prod_101")
        print(f"    Verified point 'prod_101': vector_dim={len(p101.vector)}, title='{p101.payload.get('title')}'")

        # Cleanly release recovered database resources
        recovered_engine.close()
        print("\n" + "=" * 72)
        print("Demonstration completed successfully. Temporary resources cleaned up.")
        print("=" * 72)


if __name__ == "__main__":
    run_demo()
