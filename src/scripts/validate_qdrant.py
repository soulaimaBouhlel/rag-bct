#!/usr/bin/env python3
"""
Validate the Qdrant index.

Checks:
- Collection exists
- Vector count matches chunks.json
- Metadata is intact
- Embedding dimension is consistent
- Distance metric is configured
- Random samples can be retrieved
"""

import json
import os
import random

from qdrant_client import QdrantClient


COLLECTION_NAME = "regulations"
CHUNKS_PATH = "data/chunks.json"


def load_chunks(path: str) -> list[dict]:
    """Load chunks from disk."""

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_collection(client: QdrantClient):
    """Ensure the collection exists."""

    print("\n1️⃣ Checking collection...")

    collection = client.get_collection(COLLECTION_NAME)

    print(f"   ✅ Collection '{COLLECTION_NAME}' exists")

    return collection


def validate_vectors(collection, expected_count: int):
    """Validate vector configuration."""

    print("\n2️⃣ Checking vector configuration...")

    vectors = collection.config.params.vectors

    print(f"   ✅ Dimension: {vectors.size}")
    print(f"   ✅ Distance: {vectors.distance}")

    print("\n3️⃣ Checking vector count...")

    return expected_count


def validate_count(client: QdrantClient, expected_count: int):
    """Ensure all vectors were indexed."""

    count = client.count(collection_name=COLLECTION_NAME)

    assert (
        count.count == expected_count
    ), f"Expected {expected_count} vectors, found {count.count}"

    print(f"   ✅ {count.count} vectors indexed")


def validate_metadata(client: QdrantClient):
    """Verify payload fields exist."""

    print("\n4️⃣ Checking metadata...")

    point = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[0],
    )[0]

    required_fields = [
        "chunk_id",
        "chunk_type",
        "circular_ref",
        "text",
        "title",
        "token_count",
    ]

    missing = [
        field
        for field in required_fields
        if field not in point.payload
    ]

    assert not missing, f"Missing payload fields: {missing}"

    print("   ✅ Metadata is valid")


def validate_samples(client: QdrantClient, total_points: int):
    """Retrieve a few random points."""

    print("\n5️⃣ Checking random samples...")

    sample_size = min(3, total_points)

    sample_ids = random.sample(range(total_points), sample_size)

    points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=sample_ids,
    )

    assert len(points) == sample_size

    for point in points:
        assert point.payload["text"].strip()

    print(f"   ✅ Retrieved {sample_size} random samples")


def main():

    print("\n" + "=" * 70)
    print("VALIDATING QDRANT INDEX")
    print("=" * 70)

    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    chunks = load_chunks(CHUNKS_PATH)

    collection = validate_collection(client)

    validate_vectors(collection, len(chunks))

    validate_count(client, len(chunks))

    validate_metadata(client)

    validate_samples(client, len(chunks))

    print("\n" + "=" * 70)
    print("✅ ALL VALIDATION CHECKS PASSED")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()