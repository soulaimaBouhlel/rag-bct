#!/usr/bin/env python3
"""
Validation: Verify Qdrant migration succeeded.

Checks:
- Collection exists
- Vector dimension is 768
- Distance metric is Cosine
- All 87 vectors uploaded
- Metadata is intact
"""

import os
from qdrant_client import QdrantClient


def validate_migration():
    """Validate Qdrant migration."""

    print("\n" + "=" * 70)
    print("VALIDATION: Qdrant Migration")
    print("=" * 70)

    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", 6333))
    api_key = os.getenv("QDRANT_API_KEY")

    client = QdrantClient(
        host=host,
        port=port,
        api_key=api_key if api_key else None,
    )

    try:
        # ✅ Check 1: Collection exists
        print("\n1️⃣  Checking collection exists...")
        collection = client.get_collection("regulations")
        print(f"   ✅ Collection 'regulations' exists")

        # ✅ Check 2: Vector dimension
        print("\n2️⃣  Checking vector dimension...")
        vector_size = collection.config.params.vectors.size
        assert vector_size == 768, f"Expected 768, got {vector_size}"
        print(f"   ✅ Vector dimension: {vector_size} (correct)")

        # ✅ Check 3: Distance metric
        print("\n3️⃣  Checking distance metric...")
        distance = collection.config.params.vectors.distance
        distance_str = str(distance).lower()
        assert "cosine" in distance_str, f"Expected Cosine, got {distance}"
        print(f"   ✅ Distance metric: {distance} (correct)")

        # ✅ Check 4: Vector count
        print("\n4️⃣  Checking vector count...")
        count = client.count(collection_name="regulations")
        assert count.count == 87, f"Expected 87, got {count.count}"
        print(f"   ✅ Vector count: {count.count} (correct)")

        # ✅ Check 5: Metadata present
        print("\n5️⃣  Checking metadata integrity...")
        result = client.retrieve("regulations", ids=[0])
        point = result[0]
        required_fields = [
            "chunk_id",
            "circular_ref",
            "chunk_type",
            "text",
            "title",
            "token_count",
        ]
        for field in required_fields:
            assert field in point.payload, f"Missing field: {field}"
        print(f"   ✅ All required metadata fields present")

        # ✅ Check 6: Sample some random points
        print("\n6️⃣  Checking random samples...")
        sample_ids = [0, 42, 86]  # First, middle, last
        samples = client.retrieve("regulations", ids=sample_ids)
        assert len(samples) == 3, "Could not retrieve samples"
        for sample in samples:
            assert sample.payload["text"], "Text field is empty"
        print(f"   ✅ Random samples verified ({len(samples)} checked)")

        print("\n" + "=" * 70)
        print("✅ ALL VALIDATION CHECKS PASSED")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        raise


if __name__ == "__main__":
    validate_migration()